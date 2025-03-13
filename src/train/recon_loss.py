import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from einops import rearrange

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.functional.functional")

class MelSpecWrapper(nn.Module):
    """梅尔频谱图包装器,用于生成不同尺度的频谱图表示"""
    
    def __init__(self, 
                 n_fft: int,
                 hop_length: int,
                 n_mels: int = 80,
                 win_length: Optional[int] = None,
                 sample_rate: float = 22050,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 log_scale: bool = True,
                 normalized: bool = False,
                 floor_level: float = 1e-5):
        """
        Args:
            n_fft: FFT窗口大小
            hop_length: 帧移大小
            n_mels: 梅尔滤波器数量
            win_length: 窗口长度,默认等于n_fft
            sample_rate: 采样率
            f_min: 最小频率
            f_max: 最大频率
            log_scale: 是否使用对数刻度
            normalized: 是否归一化
            floor_level: 最小值阈值
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor_level = floor_level
        self.log_scale = log_scale
        
        # 创建梅尔频谱变换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            f_min=f_min,
            f_max=f_max,
            normalized=normalized,
            window_fn=torch.hann_window,
            center=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 生成梅尔频谱图
        x: [B, C, T]
        mel_spec: [B, C*n_mels, frames]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # 计算填充
        p = (self.n_fft - self.hop_length) // 2
        x = F.pad(x, (p, p), "reflect")
        
        # 确保最后一帧完整
        length = x.shape[-1]
        n_frames = (length - self.n_fft) / self.hop_length + 1
        ideal_length = (int(np.ceil(n_frames)) - 1) * self.hop_length + self.n_fft
        extra_padding = ideal_length - length
        if extra_padding > 0:
            x = F.pad(x, (0, extra_padding))
        
        # 生成频谱图
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        if self.log_scale:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        
        # 重塑输出
        mel_spec = rearrange(mel_spec, 'b c f t -> b (c f) t')
        return mel_spec

# note: 这里的参数设置和 Encodec 论文不太一样
class MultiScaleMelLoss(nn.Module):
    """ 多尺度梅尔频谱图重构损失 """
    def __init__(self,
                 sample_rate: int,                              # 音频采样率
                 n_mels: int = 64,                              # 梅尔滤波器数量
                 range_start: int = 6,                          # 2^6 = 64              
                 range_end: int = 11,                           # 2^11 = 2048
                 normalized: bool = False,
                 use_alpha_weights: bool = True,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 floor_level: float = 1e-5):
        super().__init__()
        
        self.specs = nn.ModuleList()
        self.alphas = []
        self.total_weight = 0
        self.normalized = normalized
        
        # 为每个尺度创建两个频谱图生成器(线性和对数尺度)
        for scale in range(range_start, range_end):
            n_fft = 2 ** scale
            hop_length = n_fft // 4
            
            # 线性尺度频谱图
            self.specs.append(
                MelSpecWrapper(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    f_min=f_min,
                    f_max=f_max,
                    log_scale=False,
                    floor_level=floor_level
                )
            )
            
            # 对数尺度频谱图
            self.specs.append(
                MelSpecWrapper(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    log_scale=True,
                    floor_level=floor_level
                )
            )
            
            # 计算该尺度的权重
            alpha = np.sqrt(n_fft - 1) if use_alpha_weights else 1
            self.alphas.append(alpha)
            self.total_weight += alpha + 1
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算多尺度频谱图损失
        x, y: [B, 1, T]
            
        Returns:
            loss: 多尺度频谱图损失值
        """
        loss = 0.0
        self.specs.to(x.device)
        
        # 在每个尺度上计算损失
        for i, alpha in enumerate(self.alphas):
            idx = i * 2  # 每个尺度有两个频谱图(线性和对数)
            
            # 线性频谱图损失 (L1损失)
            spec_x_linear = self.specs[idx](x)
            spec_y_linear = self.specs[idx](y)
            linear_loss = F.l1_loss(spec_x_linear, spec_y_linear)
            
            # 对数频谱图损失 (MSE损失)
            spec_x_log = self.specs[idx + 1](x)
            spec_y_log = self.specs[idx + 1](y)
            log_loss = F.mse_loss(spec_x_log, spec_y_log)
            
            # 组合损失
            loss += linear_loss + alpha * log_loss
        
        # 归一化损失(如果需要)
        if self.normalized:
            loss = loss / self.total_weight
            
        return loss

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/default.yaml")
    msspec_loss = MultiScaleMelLoss(**cfg.msspec)

    x = torch.randn(2, 1, 16000)
    y = torch.randn(2, 1, 16000)
    loss = msspec_loss(x, y)
    print(loss)