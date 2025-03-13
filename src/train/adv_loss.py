import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
import torch.distributed as dist
from torch.nn.utils import spectral_norm, weight_norm
from typing import List, Tuple, Dict, Any, Optional, Union
from contextlib import contextmanager

NORMS = {
    'none': lambda x: x,
    'weight_norm': weight_norm,
    'spectral_norm': spectral_norm
}

class Conv2dWithNorm(nn.Module):
    """ 参数规范化的2D卷积层 """
    def __init__(self, *args, norm: str = 'none', **kwargs):
        super().__init__()
        self.conv = NORMS[norm](nn.Conv2d(*args, **kwargs))
        
    def forward(self, x):
        return self.conv(x)

class STFTDiscriminator(nn.Module):
    """STFT判别器"""
    def __init__(self, 
                 filters: int = 32,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 max_filters: int = 1024,                       # 最大滤波器数量
                 filters_scale: int = 1,                        # 每层滤波器增长倍数
                 kernel_size: Tuple[int, int] = (3, 9),         # 在时间和频率维度使用不同的 kernel_size
                 dilations: List[int] = [1, 2, 4],
                 stride: Tuple[int, int] = (1, 2),
                 norm: str = 'weight_norm',
                 activation_slope: float = 0.2):
        super().__init__()
        
        # STFT变换 | wav -> image
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,                                # FFT 窗口         
            hop_length=hop_length,                      # 滑移距离
            win_length=win_length,                      # 窗口大小，通常等于 n_fft
            window_fn=torch.hann_window,                # 窗口函数
            normalized=True,                            # 是否归一化
            center=False,                               # 是否对信号填充
            pad_mode=None,                              # 填充模式
            power=None,
        )
        
        # 卷积层列表
        self.convs = nn.ModuleList()
        
        # 第一个卷积层，输出形状不变
        self.convs.append(
            Conv2dWithNorm(in_channels=2 * in_channels, 
                           out_channels=filters,
                           kernel_size=kernel_size,
                           padding=tuple(k//2 for k in kernel_size),
                           dilation=1,
                           stride=1)
        )
        
        # 中间的dilated卷积层
        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                Conv2dWithNorm(in_chs,
                               out_chs, 
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=(dilation, 1),
                               padding=((kernel_size[0]-1)*dilation//2, (kernel_size[1]-1)//2),
                               norm=norm)
            )
            in_chs = out_chs
            
        # 最后两个卷积层，不改变 feature map 形状
        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        k = (kernel_size[0], kernel_size[0])
        pad = tuple(x//2 for x in k)
        self.convs.append(Conv2dWithNorm(in_chs, out_chs, kernel_size=k, padding=pad, norm=norm))
        self.conv_post = Conv2dWithNorm(out_chs, out_channels, kernel_size=k, padding=pad, norm=norm)
        
        # 激活函数
        self.activation = nn.LeakyReLU(activation_slope)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B, C, T]
        ---
        out: [B, 1, time_frames, freq_bin]
        fmaps: 每一层的特征图 List [[B, H, time_frames, freq_bins']] 
        """
        # STFT变换并重排维度
        z = self.spec_transform(x)                          # [B, 1, freq_bins, time_frames]
        z = torch.cat([z.real, z.imag], dim=1) 
        z = rearrange(z, 'b c w t -> b c t w')              # [B, 2, time_frames, freq_bins]

        # 存储特征图
        # *NOTE: 2D 卷积只会改变频域长度，不会影响到时域 time_frames
        fmaps = []
        for conv in self.convs:
            z = self.activation(conv(z))
            fmaps.append(z)

        # 下采样
        out = self.conv_post(z)
            
        return out, fmaps

class MultiScaleSTFTDiscriminator(nn.Module):
    """ 多尺度STFT判别器 """
    def __init__(self,
                 filters: int = 32,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_ffts: List[int] = [1024, 2048, 512],
                 hop_lengths: List[int] = [256, 512, 128],
                 win_lengths: List[int] = [1024, 2048, 512],
                 **kwargs):
        super().__init__()

        self.discriminators = nn.ModuleList([
            STFTDiscriminator(
                filters=filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                **kwargs
            )
            for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        x: [B, 1, T]
        logits: List, 长度为 L 的列表, 每个元素形状为 [B, 1, T, F]
        fmaps: List, 长度为 L 的列表, 每个元素为长度 M 的列表,对应 M 层卷积, 每个元素形状为 [B, H, T, F]
        """
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)

        return logits, fmaps

class AdversarialLoss(nn.Module):
    """ 对抗损失计算 
    train_discriminator 函数计算判别损失，更新判别器参数
    forward 计算 1) 生成损失 2) 判别器的特征匹配损失
    """
    def __init__(self, cfg, device=None, is_distributed=False, local_rank=0):
        super().__init__()
        self.device = device or torch.device("cpu")

        # MS-STFT 判别器
        discriminator = MultiScaleSTFTDiscriminator(**cfg).to(self.device)

        if is_distributed:
            self.discriminator = nn.parallel.DistributedDataParallel(
                discriminator,
                device_ids=[local_rank],
                output_device=local_rank
            )
        else:
            self.discriminator = discriminator

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0003,                              
            betas=(0.5, 0.9),                       
            weight_decay=0.0                        
        )

        # 特征匹配损失
        self.feat_match_loss_fn = nn.L1Loss()

        # 生成器损失：最大化判别器在生成样本上的输出
        self.gen_loss_fn = lambda x: torch.tensor([0.], device=x.device) if x.numel() == 0 else -x.mean()

        # 判别器在真实样本上的损失：输出要大于1的margin
        self.disc_real_loss_fn = lambda x: -torch.mean(
            torch.min(x - 1, torch.tensor([0.], device=x.device).expand_as(x))
        )
        
        # 判别器在生成样本上的损失：输出要小于-1的margin
        self.disc_fake_loss_fn = lambda x: -torch.mean(
            torch.min(-x - 1, torch.tensor([0.], device=x.device).expand_as(x))
        )

    def broadcast_model(self, model: nn.Module, src: int = 0):
        """异步广播模型参数和缓冲区到所有worker"""
        if not dist.is_initialized():
            return
            
        handles = []
        
        # 异步广播参数
        for param in model.parameters():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                handle = dist.broadcast(param.data, src=src, async_op=True)
                handles.append(handle)
        
        # 异步广播缓冲区        
        for buf in model.buffers():
            if buf.dtype.is_floating_point or buf.dtype.is_complex:
                handle = dist.broadcast(buf.data, src=src, async_op=True)
                handles.append(handle)
            
        for handle in handles:
            handle.wait()

    @contextmanager
    def freeze_discriminator(self):
        """临时冻结判别器参数"""
        states = [p.requires_grad for p in self.discriminator.parameters()]
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, s in zip(self.discriminator.parameters(), states):
                p.requires_grad_(s)

    def train_discriminator(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """训练判别器
        fake, real: [B, 1, T] 
        """
        fake_logits, _ = self.discriminator(fake.detach())
        real_logits, _ = self.discriminator(real.detach())
        
        loss = sum(self.disc_fake_loss_fn(fake_logit) + self.disc_real_loss_fn(real_logit) 
                  for fake_logit, real_logit in zip(fake_logits, real_logits))
        loss = loss / len(fake_logits)  
        
        self.optimizer.zero_grad()
        loss.backward()
        # self._sync_gradients_and_buffers()
        self.optimizer.step()
        
        return loss.item()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算生成器损失
        fake, real: [B, 1, T] 
        """
        with self.freeze_discriminator():
            fake_logits, fake_features = self.discriminator(fake)
            real_logits, real_features = self.discriminator(real)
            
            # 生成器对抗损失
            gen_loss = sum(self.gen_loss_fn(logit) for logit in fake_logits)
            gen_loss = gen_loss / len(fake_logits)
            
            # 特征匹配损失
            feat_loss = torch.tensor(0., device=fake.device)
            if self.feat_match_loss_fn:
                # 逐个判别器计算
                for fmap_fake, fmap_real in zip(fake_features, real_features):
                    # 逐层计算
                    n_fmaps = 0
                    feat_loss_tmp = torch.tensor(0., device=fake.device)
                    for feat_fake, feat_real in zip(fmap_fake, fmap_real):
                        assert feat_fake.shape == feat_real.shape
                        feat_loss_tmp += self.feat_match_loss_fn(feat_fake, feat_real)
                        n_fmaps += 1
                    feat_loss += feat_loss_tmp / n_fmaps
                feat_loss = feat_loss / len(fake_features)
                
            return gen_loss, feat_loss
    
    def _sync_gradients_and_buffers(self):
        """同步梯度(分布式训练)"""
        if not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()

        # 同步梯度
        handles = []
        for param in self.discriminator.parameters():
            if param.grad is not None:
                handle = dist.all_reduce(
                    param.grad.data, 
                    op=dist.ReduceOp.SUM, 
                    async_op=True
                )
                handles.append((param, handle))
        
        # 等待梯度同步完成并缩放
        for param, handle in handles:
            handle.wait()
            param.grad.data /= world_size
            
        # 同步 buffers
        handles = []
        for buffer in self.discriminator.buffers():
            handle = dist.all_reduce(
                buffer.data,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            handles.append((buffer, handle))
            
        # 等待 buffers 同步完成并缩放    
        for buffer, handle in handles:
            handle.wait()
            buffer.data /= world_size

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/default.yaml")
    adv_loss = AdversarialLoss(cfg.msstftd)
    print(cfg.optim.ema.get('use', False))

    fake = torch.randn(2,1,16000)
    real = torch.randn(2,1,16000)

    gen_loss, feat_loss = adv_loss(fake, real)
    print(gen_loss, feat_loss)

    dis_loss = adv_loss.train_discriminator(fake, real)
    print(dis_loss)
