import torch
import torch.nn as nn

import typing as tp
from dataclasses import dataclass

from .base import StreamingModule, StreamingContainer
from .conv import StreamingConv1d


@dataclass
class _StreamingAddState:
    # 存储两个输入的残余数据
    previous_x: torch.Tensor | None = None          
    previous_y: torch.Tensor | None = None

    def reset(self):
        self.previous_x = None
        self.previous_y = None

class StreamingAdd(StreamingModule[_StreamingAddState]):
    """ 流式加法模块
    用于处理两个可能长度不同的张量的流式加法运算。
    在流式处理中，确保正确处理和对齐输入张量，并维护未处理的数据状态。
    """
    def _init_streaming_state(self, batch_size: int) -> _StreamingAddState:
        return _StreamingAddState()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        输入: x[t2], y[t2]
        状态: prev_x[t1], prev_y[t1]
        ↓
        拼接: x[t1+t2], y[t1+t2]
        ↓
        找到共同长度: m_l
        ↓
        处理: x[:m_l] + y[:m_l]
        ↓
        保存剩余: x[m_l:], y[m_l:]
        """
        if self._streaming_state is None:
            return x + y
        else:
            prev_x = self._streaming_state.previous_x
            prev_y = self._streaming_state.previous_y
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=-1)
            if prev_y is not None:
                y = torch.cat([prev_y, y], dim=-1)
            m_l = min(x.shape[-1], y.shape[-1])
            self._streaming_state.previous_x = x[..., m_l:]
            self._streaming_state.previous_y = y[..., m_l:]
            return x[..., :m_l] + y[..., :m_l]

class SEANetResnetBlock(StreamingContainer):
    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ) -> None:
        super().__init__()

        # 多层卷积
        hidden = dim // compress
        act = getattr(nn, activation)
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamingConv1d(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        
        self.block = nn.Sequential(*block)

        # 计算残差
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamingConv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

        # 输出隐变量和残差的形状可能不一致
        self.add = StreamingAdd()

    def forward(self, x):
        u, v = self.shortcut(x), self.block(x)
        return self.add(u, v)
