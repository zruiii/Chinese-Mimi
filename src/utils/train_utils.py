import os
import sys
import torch
import logging
import typing as tp
from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn

# === 状态管理 ==============================================================
@dataclass
class TrainingState:
    """ 训练状态管理 """
    is_distributed: bool = False
    local_rank: int = 0
    global_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device('cpu')

    # 训练进度
    epoch: int = 0
    current_stage = "train"

    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0

    def update_epoch(self):
        self.epoch += 1
    
    @classmethod
    def from_environ(cls) -> 'TrainingState':
        is_distributed = "WORLD_SIZE" in os.environ
        if is_distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            global_rank = int(os.environ['RANK'])
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_rank = global_rank = 0
            world_size = 1
        
        return cls(
                is_distributed=is_distributed,
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                device=device
            )

# === 日志记录 ==============================================================
class DistributedLogger:
    def __init__(
        self,
        cfg: DictConfig,
        state: TrainingState,
        log_dir: tp.Union[str, Path]
    ):
        self.cfg = cfg
        self.state = state
        self.log_dir = Path(log_dir)
        self.writers = {}
        self.logger = None
        
        # 创建日志目录
        # if self.state.is_distributed and dist.is_initialized():
        #     if self.state.is_main_process:
        #         self.log_dir.mkdir(parents=True, exist_ok=True)
        #     dist.barrier()                          
        # else:
        #     self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.state.is_main_process:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        # 设置基础日志系统
        self._setup_logging()
        
        # 只在主进程创建tensorboard writer
        if self.state.is_main_process:
            self.writers['train'] = SummaryWriter(
                log_dir=self.log_dir / 'tensorboard' / 'train'
            )
            self.writers['valid'] = SummaryWriter(
                log_dir=self.log_dir / 'tensorboard' / 'valid'
            )
            
    def _setup_logging(self):
        # 创建logger实例
        self.logger = logging.getLogger("distributed_training")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False           # 确保不会传播到 root logger 
        self.logger.handlers.clear()            # 清除已有的handlers

        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if self.state.is_main_process:
            # 控制台处理器（只在主进程添加）
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 主进程的文件处理器
            log_file = self.log_dir / self.cfg.logging.filename
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
        # else:
        #     # 其他进程只记录错误日志到单独的文件
        #     log_file = self.log_dir / f"rank{self.state.global_rank}.log"
        #     file_handler = logging.FileHandler(log_file)
        #     file_handler.setFormatter(formatter)
        #     file_handler.setLevel(logging.ERROR)  # 只记录错误
        #     self.logger.addHandler(file_handler)
    
    def info(self, msg: str, *args, **kwargs):
        if self.state.is_main_process:
            self.logger.info(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        if self.state.is_main_process:
            self.logger.error(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        if self.state.is_main_process:
            self.logger.warning(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        if self.state.is_main_process:
            self.logger.debug(msg, *args, **kwargs)
    
    def log_metrics(self, metrics: tp.Dict[str, float], step: int, stage: str = 'train'):
        """ 写入 tensorboard """
        if not self.state.is_main_process:
            return
        
        for name, value in metrics.items():
            self.writers[stage].add_scalar(name, value, step)           # tensorboard
        
    
    def close(self):
        """关闭所有handlers和writers"""
        if self.state.is_main_process:
            for writer in self.writers.values():
                writer.close()
        
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

# todo: 流式编码还是存在 BUG
def mimi_streaming_encode(
    mimi_model: nn.Module,
    waveform: torch.Tensor,
    chunk_samples: int = 16000 * 2,
) -> torch.Tensor:
    """
    使用 Mimi 流式处理对长音频进行编码
    
    Args:
        mimi_model: Neural Audio Codec model (like Encodec)
        waveform: 输入波形，形状为 [1, 1, T] 或 [T,]
        chunk_samples: 每次处理的样本数，建议是frame_size的整数倍
        
    Returns:
        torch.Tensor: 编码后的token序列，形状为 [1, H, F]
    """
    batch_size = waveform.shape[0]

    # 重置模型流式状态
    mimi_model.streaming_forever(batch_size)
    mimi_model.reset_streaming()
    
    # 分块处理
    all_tokens = []
    with torch.no_grad():
        for start_idx in range(0, waveform.shape[-1], chunk_samples):
            # 提取当前块
            end_idx = min(start_idx + chunk_samples, waveform.shape[-1])
            chunk = waveform[:, :, start_idx:end_idx]
                
            # 编码当前块
            try:
                tokens = mimi_model.encode(chunk)
            except:
                raise ValueError(f"chunk of shape {chunk.shape} fail.")
            all_tokens.append(tokens)
    
    # 合并所有token
    audio_tokens = torch.cat(all_tokens, dim=-1)
    
    return audio_tokens