"""
@File   : decoder.py
@Time   : 2024/11/11 15:06:30
@Author : zharui 
@Email  : zharui@baidu.com
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np
from dataclasses import dataclass

from .base import StreamingContainer
from .blocks import SEANetResnetBlock
from .conv import StreamingConv1d, StreamingConvTranspose1d

from ..utils import torch_compile_lazy


class SEANetDecoder(StreamingContainer):
    def __init__(
        self,
        channels: int = 1,                                      # 音频维度
        dimension: int = 128,                                   # 隐变量维度
        n_filters: int = 32,                                    # 模型带宽
        n_residual_layers: int = 3,                             # 残差层数目
        ratios: tp.List[int] = [8, 5, 4, 2],                    # 降采样率
        activation: str = "ELU",                                # 激活函数
        activation_params: dict = {"alpha": 1.0},               # 激活函数参数
        final_activation: tp.Optional[str] = None,              
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,                                   # 第一层卷积核
        last_kernel_size: int = 3,                              # 最后一层卷积核
        residual_kernel_size: int = 3,                          # 残差卷积核
        dilation_base: int = 2,
        causal: bool = True,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamingConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = (
                "none"
                if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                else norm
            )
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamingConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    @torch_compile_lazy
    def forward(self, z):
        y = self.model(z)
        return y

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
        "norm": "weight_norm",
        "pad_mode": "constant",
        "ratios": [8, 6, 5, 4],
        "true_skip": True,
    }

    decoder = SEANetDecoder(**_seanet_kwargs)
    
    batch_size = 2
    num_frames = 25  # 约1秒的音频
    x = torch.randn(batch_size, 512, 25)
    
    try:
        output = decoder(x)
        print(f"Test passed! Output shape: {output.shape}")
        
        # 压缩率
        compression_ratio = output.shape[-1] / 25
        print(f"Compression ratio: {compression_ratio:.1f}x")
        
    except Exception as e:
        print(f"Test failed! Error: {str(e)}")