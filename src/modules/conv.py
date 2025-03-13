import math
import torch
import typing as tp
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from .base import StreamingModule

CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])

# ********************** 卷积 [管理状态缓存] **********************
@dataclass
class _StreamingConvState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None

class RawStreamingConv1d(nn.Conv1d, StreamingModule[_StreamingConvState]):
    """ 基本的一维流式卷积，这里的输入是已经 padding 之后的值 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.stride[0] <= self.kernel_size[0], "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
        return _StreamingConvState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: [B, C, T]
        """
        # 非流式卷积
        if self._streaming_state == None:
            return super().forward(input)

        # 流式卷积
        else:
            stride = self.stride[0]
            kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1

            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            
            B, C, T = input.shape
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))      
            offset = num_frames * stride                                    # [(N - 1) * S + K] - (K - S)                  
            self._streaming_state.previous = input[..., offset:]

            if num_frames > 0:
                input_length = offset + (kernel - stride)
                out = super().forward(input[..., :input_length])
            else:
                out = torch.empty(B, self.out_channels, 0, device=input.device, dtype=input.dtype)

            return out


class NormConv1d(nn.Module):
    def __init__(
        self, 
        *args,
        norm: str = "none",
        **kwargs
    ) -> None:
        super().__init__()
        assert norm in ["none", "weight_norm"]

        self.conv = RawStreamingConv1d(*args, **kwargs)
        if norm == "weight_norm":
            nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ********************** 卷积 [管理pad] **********************
def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))           
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)      # 用0填充

@dataclass
class _StreamingConv1dState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self):
        self.padding_to_add = self.original_padding_to_add


class StreamingConv1d(StreamingModule[_StreamingConv1dState]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,            
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        norm: str = 'none',
        pad_mode: str = "reflect",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
    ) -> None:
        super().__init__()

        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm=norm,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    @property
    def _stride(self) -> int:
        """ 滑动步长 """
        return self.conv.conv.stride[0]
    
    @property
    def _effective_kernel_size(self) -> int:
        """ 实际卷积核大小 """
        dilation = self.conv.conv.dilation[0]
        kernel = self.conv.conv.kernel_size[0]
        kernel = dilation * (kernel - 1) + 1
        return kernel

    @property
    def _padding_total(self) -> int:
        """ 一共要填充的pad数目
        注意这里填充的 pad 数目是 K-S 而不是 K-1
        if P=K-1, 当压缩到 int(L/S) 个 token 的时候, 序列尾部还剩下一些信息未参与卷积
        P = K-S 的时候, 可以通过右侧填充一些, 来使得所有的序列都能参与完整卷积计算
        """
        return self._effective_kernel_size - self._stride 

    def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
        assert self.causal, "只有因果卷积支持流式处理"        
        return _StreamingConv1dState(self._padding_total, self._padding_total)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state

        # 非流式
        if state is None:
            B, C, T = x.shape
            padding_total = self._padding_total
            num_frames = math.ceil((T + padding_total - self._effective_kernel_size) / self._stride + 1)
            extra_padding = (num_frames - 1) * self._stride + self._effective_kernel_size - T - padding_total
            
            if self.causal:
                # 确保所有的序列都能参与卷积计算
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )

        # 流式: 只有第一个 chunk 的数据会在左侧进行填充，后续数据都不做填充
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        
        return self.conv(x)


# ********************** 逆卷积 [管理缓存] **********************
@dataclass
class _StreamingConvTrState:
    partial: torch.Tensor | None = None

    def reset(self):
        self.partial = None


class RawStreamingConvTranspose1d(nn.ConvTranspose1d, StreamingModule[_StreamingConvTrState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert self.dilation[0] == 1, "No dilation for now"
        assert self.stride[0] <= self.kernel_size[0], "stride must be less than kernel_size."
        assert self.output_padding[0] == 0, "Output padding not supported."

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTrState:
        return _StreamingConvTrState()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        # 非流式
        if self._streaming_state is None:
            return super().forward(x)

        # 流式
        else:
            B, C, T = x.shape
            stride = self.stride[0]
            kernel = self.kernel_size[0]
            if T == 0:
                return torch.empty(B, self.out_channels, 0, device=x.device, dtype=x.dtype)     # [B, C, 0]
            
            # 对当前 Input 执行逆向卷积
            out = super().forward(x)
            OT = out.shape[-1]

            # 添加逆向卷积的重叠部分
            partial = self._streaming_state.partial
            if partial is not None:
                PT = partial.shape[-1]
                if self.bias is not None:
                    out[..., :PT] += partial - self.bias[:, None]
                else:
                    out[..., :PT] += partial
            
            # 保存 K-S 的数据到下一帧
            invalid_steps = kernel - stride
            partial = out[..., OT - invalid_steps :]
            out = out[..., : OT - invalid_steps]
            self._streaming_state.partial = partial
            return out


class NormConvTranspose1d(nn.Module):
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        assert norm in CONV_NORMALIZATIONS

        self.convtr = RawStreamingConvTranspose1d(*args, **kwargs)
        if norm == "weight_norm":
            nn.utils.weight_norm(self.convtr)

    def forward(self, x):
        x = self.convtr(x)
        return x

# ********************** 逆卷积 [管理pad] **********************
def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """ 移出左右两侧的 pad """
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


@dataclass
class _StreamingConvTr1dState:
    pass

    def reset(self):
        pass

class StreamingConvTranspose1d(StreamingModule[_StreamingConvTr1dState]):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (self.causal or self.trim_right_ratio == 1.0), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTr1dState:
        assert self.causal, "streaming is only supported for causal convtrs"
        return _StreamingConvTr1dState()

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)
        
        # 只有在非流式状态下对逆卷积输出进行裁剪
        # 流式状态因为输入没有 pad, 所以不做处理
        if not self.is_streaming:
            if self.causal:
                padding_right = math.ceil(padding_total * self.trim_right_ratio)
                padding_left = padding_total - padding_right
                y = unpad1d(y, (padding_left, padding_right))
            else:
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                y = unpad1d(y, (padding_left, padding_right))
        return y


if __name__ == "__main__":
    batch_size = 2
    in_channels = 1
    out_channels = 32
    kernel_size = 5
    stride = 2
    total_length = 20  
    chunk_size = 4     

    x = torch.randn(batch_size, in_channels, total_length)
    print(f"\n输入数据形状: {x.shape}")

    print("\n========== 测试一维卷积 ==========")
    conv = NormConv1d(
        in_channels, 
        out_channels, 
        kernel_size=kernel_size,
        stride=stride,
        norm="weight_norm"
    )

    # 1. 非流式处理
    with torch.no_grad():
        out_offline = conv(x)
    print(f"非流式输出形状: {out_offline.shape}")

    # 2. 流式处理
    with conv.conv.streaming(batch_size):
        outputs = []
        with torch.no_grad():
            for i in range(0, total_length, chunk_size):
                chunk = x[..., i:i+chunk_size]
                out_chunk = conv(chunk)
                if out_chunk.shape[-1] > 0:  # 只收集非空输出
                    outputs.append(out_chunk)
                print(out_chunk.shape)
        
        out_online = torch.cat(outputs, dim=-1)
        print(f"流式输出形状: {out_online.shape}")
    
    max_diff = (out_offline - out_online).abs().max().item()
    print(f"最大误差: {max_diff:.6f}")

    print("\n========== 测试一维转置卷积 ==========")
    convtr = NormConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        norm="weight_norm"
    )
    
    # 1. 非流式处理
    with torch.no_grad():
        out_offline = convtr(x)
    print(f"非流式输出形状: {out_offline.shape}")

    # 2. 流式处理
    with convtr.convtr.streaming(batch_size):
        outputs = []
        with torch.no_grad():
            for i in range(0, total_length, chunk_size):
                chunk = x[..., i:i+chunk_size]
                out_chunk = convtr(chunk)
                if out_chunk.shape[-1] > 0:  # 只收集非空输出
                    outputs.append(out_chunk)
                print(out_chunk.shape)

        out_online = torch.cat(outputs, dim=-1)
        print(f"流式输出形状: {out_online.shape}")
    
    # 验证结果
    max_diff = (out_offline[..., :-(kernel_size-stride)] - out_online).abs().max().item()
    print(f"最大误差: {max_diff:.6f}")