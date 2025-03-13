import torch
import torch.nn as nn
import typing as tp
import numpy as np

from .base import StreamingContainer
from .conv import StreamingConv1d
from .blocks import SEANetResnetBlock
from ..utils import torch_compile_lazy


class SEANetEncoder(StreamingContainer):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 512,
        n_filters: int = 32,                        # 卷积核数目
        kernel_size: int = 7,                       # 卷积核大小
        ratios: tp.List[int] = [8, 6, 5, 4],        # 每一层的步长
        causal: bool = True,
        n_residual_layers: int = 1,                 # 残差层的数目
        activation: str = "ELU",
        norm: str = "none",
        disable_norm_outer_blocks: int = 0,         # 不采取正则化的残差层
        compress: int = 2,
        true_skip: bool = True,
        residual_kernel_size: int = 3,              # 残差层的卷积核大小
        dilation_base: int = 2,                     
        pad_mode: str = "constant",
        norm_params: tp.Dict[str, tp.Any] = {},
        activation_params: dict = {"alpha": 1.0},
        last_kernel_size: int = 3,
        mask_fn: tp.Optional[nn.Module] = None,
        mask_position: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.ratios = ratios

        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        
        act = getattr(nn, activation)

        # 第一层用 1D 因果卷积, stride=1
        mult = 1
        model: tp.List[nn.Module] = []
        model += [StreamingConv1d(
            in_channels=channels,
            out_channels=mult*n_filters,
            kernel_size=kernel_size,
            stride=1,
            causal=causal
        )]
        if mask_fn is not None and mask_position == 0:
            model += [mask_fn]

        # 残差卷积
        for i, ratio in enumerate(self.ratios):
            block_norm = "none" if disable_norm_outer_blocks >= i + 2 else norm

            # 残差层
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            
            # 下采样
            model += [
                act(**activation_params),
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2
            if mask_fn is not None and mask_position == i + 1:
                model += [mask_fn]

        model += [
            act(**activation_params),
            StreamingConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=(
                    "none" if disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)
    
    @torch_compile_lazy         # 延迟编译
    def forward(self, x) -> torch.Tensor:
        return self.model(x)

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        "norm": "weight_norm",
        "pad_mode": "constant",
        "ratios": [8, 6, 5, 4],
        "true_skip": True,
    }

    encoder = SEANetEncoder(**_seanet_kwargs).to(device)
    
    batch_size = 32
    time_steps = 16000 * 8  # 约1秒的音频
    x = torch.randn(batch_size, 1, time_steps).to(device)
    
    try:
        output = encoder(x)
        print(f"Test passed! Output shape: {output.shape}")
        
        # 压缩率
        compression_ratio = time_steps / output.shape[-1]
        print(f"Compression ratio: {compression_ratio:.1f}x")
        assert output.shape[1] == 512, "Output dimension doesn't match specified dimension!"
        
    except Exception as e:
        print(f"Test failed! Error: {str(e)}")

