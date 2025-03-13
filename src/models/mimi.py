import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np

from contextlib import nullcontext
from einops import rearrange, repeat
from dataclasses import dataclass

from ..modules.base import State, StreamingModule
from ..modules.encoder import StreamingConv1d
from ..modules.decoder import StreamingConvTranspose1d
from ..modules.quantizer import BaseQuantizer, QuantizedResult, ResidualVectorQuantizer, SplitResidualVectorQuantizer
from ..utils import CUDAGraphed
from ..utils.contextlibs import no_compile

# ************ 重采样 ************
class ConvDownsample1d(nn.Module):
    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        self.conv = StreamingConv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
            pad_mode="replicate",
        )
        if not learnt:
            actual_conv = self.conv.conv.conv
            actual_conv.weight.requires_grad_(False)
            actual_conv.weight.data.fill_(1.0 / (2 * stride))

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.conv(x)
        if not self.learnt:
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y


class ConvTrUpsample1d(nn.Module):
    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        self.convtr = StreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
        )
        if not learnt:
            actual_convtr = self.convtr.convtr.convtr
            actual_convtr.weight.requires_grad_(False)
            actual_convtr.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.convtr(x)
        if not self.learnt:
            x_for_normalization = torch.ones_like(x[:1])
            normalization = self.convtr(x_for_normalization)
            y = y / normalization
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y


# ************ Mimi ************
class CompressionModel(StreamingModule[State]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: ...

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    @abc.abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
        ...

    @abc.abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

@dataclass
class _MimiState:
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None

    def reset(self):
        pass


class MimiModel(CompressionModel[_MimiState]):
    def __init__(
        self, 
        encoder: nn.Module,                                         # SEANet Encoder
        decoder: nn.Module,                                         # SEANet Decoder
        quantizer: BaseQuantizer,                                   # 量化器
        frame_rate: float,                                          # 量化帧率
        encoder_frame_rate: float,                                  # Encoder 输出帧率
        sample_rate: int,                                           # 采样率
        channels: int,                                              # 输入通道
        causal: bool = False,                                       # 因果卷积
        encoder_transformer: tp.Optional[nn.Module] = None,         # Encoder Transformer
        decoder_transformer: tp.Optional[nn.Module] = None,         # Decoder Transformer
        resample_method: str = "conv",                              # 重采样
        upsample_channel_wise_bug: bool = True,                 
        freeze_encoder: bool = False,                               
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        torch_compile_encoder_decoder: bool = False,                # 在 Encoder/Decoder 阶段禁用Pytorch编译
    ):
        super().__init__()
        self.encoder = encoder  
        self.decoder = decoder  
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

        # 冻结编码器权重, 包括 SEANet Encoder | Encoder Transformer | 量化器输入映射
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False

        # 冻结量化器
        if freeze_quantizer:
            self.quantizer.ema_frozen_(True)
        self.freeze_quantizer = freeze_quantizer
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )
        
        # 获取编码器维度用于重采样
        dimension = encoder.dimension
        assert isinstance(dimension, int)
        self.dimension = dimension

        # 如果编码器帧率与目标帧率不同,设置重采样
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (causal and resample_method == "interpolate"), "因果模型不能使用插值重采样"
            
            if resample_method in ["conv", "avg_pool"]:
                assert encoder_frame_rate > frame_rate, "只支持下采样"
                
                # 计算重采样步长
                downsample_stride = encoder_frame_rate / frame_rate
                assert downsample_stride == int(downsample_stride), f"{self.encoder_frame_rate} / {self._frame_rate}"
                
                # 参数化的 conv 下采样 or 平均池化
                learnt = resample_method == "conv"
                
                # 创建下采样和上采样模块
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension, 
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )
    
    def _init_streaming_state(self, batch_size: int) -> _MimiState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_dec = None
        graphed_tr_enc = None
        # 如果有transformer模块,创建其CUDA图
        if self.encoder_transformer is not None:
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        if self.decoder_transformer is not None:
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        return _MimiState(graphed_tr_enc, graphed_tr_dec)

    def _to_quantizer_framerate(self, x: torch.Tensor):
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self._frame_rate
        if frame_rate == new_frame_rate:
            return x

        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: torch.Tensor):
        """ 将目标帧率转换回编码器帧率 """
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self._frame_rate
        if frame_rate == new_frame_rate:
            return x
        
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)

    @property
    def _context_for_encoder_decoder(self):
        """返回编解码器的上下文管理器
        用于控制是否使用torch.compile加速
        """
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()

    def set_num_codebooks(self, n: int):
        """设置要使用的码本数量
        
        Args:
            n: 码本数量
        """
        self.quantizer.set_num_codebooks(n)

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        """
        x: [B, 1, T]
        QuantizedResult:
            x: torch.Tensor                                     # 量化后的张量
            codes: torch.Tensor                                 # 量化编码
            bandwidth: torch.Tensor                             # 带宽 (kb/s), 每个batch项
            penalty: tp.Optional[torch.Tensor] = None           # 每一层量化前后的损失
            metrics: dict = field(default_factory=dict)         # 存储指标的字典
        """
        assert x.dim() == 3
        length = x.shape[-1]
        
        # 冻结量化器
        if self.freeze_quantizer:
            if isinstance(self.quantizer, SplitResidualVectorQuantizer):
                self.quantizer.rvq_first.eval()
                for i in range(self.freeze_quantizer_level - self.quantizer.n_q_semantic):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()

        # SEANet Encoder
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)

        # Transformer
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)

        # Downsample
        emb = self._to_quantizer_framerate(emb)
        
        # Quantizer (Encoder => codes => Decoder)
        q_res, s_res = self.quantizer(emb, self._frame_rate)
        emb = q_res.x
        
        # Upsample
        emb = self._to_encoder_framerate(emb)

        # Transformer
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        # SEANet Decoder
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)
        
        # Clip
        out = out[..., :length]
        q_res.x = out
        return q_res, s_res

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        codes: [B, K, T]
        """
        assert x.dim() == 3
        state = self._streaming_state
        
        # SEANet Encoder
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        
        # Transformer
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb)
        
        # Downsample
        emb = self._to_quantizer_framerate(emb)

        # Quantizer-Encoder
        codes = self.quantizer.encode(emb)  
        return codes

    def decode(self, codes: torch.Tensor):
        """
        codes: [B, K, T]
        out: [B, 1, T]
        """
        state = self._streaming_state                   
        
        # Quantizer-Decoder
        emb = self.quantizer.decode(codes)

        # Upsample
        emb = self._to_encoder_framerate(emb)  # 调整帧率
        
        # Transformer
        if self.decoder_transformer is not None:
            if state is None:
                (emb,) = self.decoder_transformer(emb)
            else:
                assert state.graphed_tr_dec is not None
                (emb,) = state.graphed_tr_dec(emb)
                
        # SEANet Decoder
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)

        return out


if __name__ == "__main__":
    _seanet_kwargs = {
        "channels": 1,
        "dimension": 512,
        "causal": True,
        "n_filters": 64,
        "n_residual_layers": 1,
        "activation": "ELU",
        "compress": 2,
        "dilation_base": 2,
        "disable_norm_outer_blocks": 0,
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        # We train using weight_norm but then the weights are pre-processed for inference so
        # that we can use a normal convolution.
        "norm": "none",
        "pad_mode": "constant",
        "ratios": [8, 5, 4, 2],
        "true_skip": True,
    }
    _quantizer_kwargs = {
        "dimension": 256,
        "n_q": 32,
        "bins": 2048,
        "input_dimension": _seanet_kwargs["dimension"],
        "output_dimension": _seanet_kwargs["dimension"],
    }
    _transformer_kwargs = {
        "d_model": _seanet_kwargs["dimension"],
        "num_heads": 8,
        "num_layers": 8,
        "causal": True,
        "layer_scale": 0.01,
        "context": 250,
        "max_period": 10000,
        "gating": "none",
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "dim_feedforward": 2048,
        "input_dimension": _seanet_kwargs["dimension"],
        "output_dimensions": [_seanet_kwargs["dimension"]],
    }
    SAMPLE_RATE = 16000
    FRAME_RATE = 12.5
    device = "cpu"

    from ..modules import SEANetEncoder, SEANetDecoder, ProjectedTransformer, SplitResidualVectorQuantizer
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device)

    model.streaming_forever(2)

    inputs = torch.randn((2, 1, SAMPLE_RATE * 1)).to(device)
    codes = model.encode(inputs)
    print(codes)
    print(codes.shape)
    print(codes.sum())


    # out = model.decode(codes)
    # print(out.shape)
    # print(model(inputs)[0].x.shape)
    # print(model(inputs)[0].codes.shape)
    # print(model(inputs)[0].penalty)

    # print(model(inputs)[1].x.shape)
    # print(model)
