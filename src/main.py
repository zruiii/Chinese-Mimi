import os
import sys
import random
import logging
import multiprocessing
from pathlib import Path

import hydra
import datetime
import traceback
from omegaconf import DictConfig, OmegaConf
import numpy as np
import typing as tp
import torch
import torch.nn as nn
import torch.backends
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.train import MimiTrainer
from src.utils import TrainingState, DistributedLogger
from src.data import AudioDataset, prepare_audiodata
from src.models import MimiModel
from src.modules import (
    SEANetEncoder,
    SEANetDecoder,
    ProjectedTransformer,
    SplitResidualVectorQuantizer
)

logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

class TrainingSystem:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = None
        self.state = TrainingState.from_environ()

        self.output_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def initialize(self) -> None:
        self.setup_logger()
        self.setup_system()
        self.setup_ddp()

    def setup_logger(self) -> None:
        # 创建日志目录
        log_dir = Path(self.cfg.logging.dir) / self.output_dir

        # 日志记录器
        self.logger = DistributedLogger(
            cfg=self.cfg,
            state=self.state,
            log_dir=log_dir
        )

        if self.state.is_main_process:
            OmegaConf.save(self.cfg, log_dir / "config.yaml")

    def setup_system(self) -> None:
        # 设置多进程启动方式
        multiprocessing.set_start_method(self.cfg.system.mp_start_method)

        # CUDA 设置
        torch.backends.cudnn.benchmark = True           # 对于固定输入尺寸，可以提速
        torch.backends.cudnn.deterministic = False      # 关闭确定性模式以提升性能
        torch.set_float32_matmul_precision('high')
        
        # 设置随机种子
        seed = self.cfg.system.seed + self.state.global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 控制 CPU 计算线程数
        num_threads = self.cfg.system.num_threads
        torch.set_num_threads(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        
        if self.state.is_main_process:
            self.logger.info(
                f"System initialized with:\n"
                f"  - PyTorch version: {torch.__version__}\n"
                f"  - CUDA available: {torch.cuda.is_available()}\n"
                f"  - Seed: {seed}\n"
                f"  - Threads: {num_threads}\n"
                f"  - MP start method: {self.cfg.system.mp_start_method}"
            )
    
    def setup_ddp(self) -> bool:
        if dist.is_initialized():
            return True
            
        if not self.state.is_distributed:
            return False
            
        try:
            # 初始化进程组
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.state.world_size,
                rank=self.state.global_rank,
                timeout=datetime.timedelta(minutes=30)
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(self.state.local_rank)

            # 确保所有进程同步
            dist.barrier()  
            
            if self.state.is_main_process:
                self.logger.info(
                    f"Distributed training initialized:\n"
                    f"  - Backend: nccl\n"
                    f"  - World size: {self.state.world_size}\n"
                    f"  - Global rank: {self.state.global_rank}\n"
                    f"  - Local rank: {self.state.local_rank}\n"
                    f"  - Master addr: {os.environ.get('MASTER_ADDR', 'N/A')}\n"
                    f"  - Master port: {os.environ.get('MASTER_PORT', 'N/A')}"
                )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {str(e)}")
            raise e

    def build_model(self) -> nn.Module:
        # 基础参数设置
        sample_rate = self.cfg.sample_rate
        frame_rate = self.cfg.frame_rate
        
        # 构建各个组件
        encoder = SEANetEncoder(**self.cfg.seanet)
        decoder = SEANetDecoder(**self.cfg.seanet)
        
        encoder_transformer = ProjectedTransformer(
            device=self.state.device,
            **self.cfg.transformer
        )
        decoder_transformer = ProjectedTransformer(
            device=self.state.device,
            **self.cfg.transformer
        )
        
        quantizer = SplitResidualVectorQuantizer(**self.cfg.quantizer)
        
        # 构建完整模型
        model = MimiModel(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            channels=1,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            encoder_frame_rate=sample_rate / encoder.hop_length,
            causal=1,
            resample_method="conv",
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer
        ).to(self.state.device)

        # 打印参数
        if self.state.is_main_process:
            model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            mem_usage = model_size * 4 * 4 / 1000
            self.logger.info("Model size: %.2f M params", model_size)
            self.logger.info("Base memory usage, with model, grad and optim: %.2f GB", mem_usage)
        
        return model

    def build_dataloaders(self) -> tp.Tuple[DataLoader, DataLoader]:
        # 1. 检查并准备元数据文件
        data_cfg = OmegaConf.to_container(self.cfg.dataset, resolve=True)
        root_dir = data_cfg.pop('root_dir')
        train_file = data_cfg.pop('train_file')
        valid_file = data_cfg.pop('valid_file')
        valid_ratio = data_cfg.pop('valid_ratio')

        batch_size = data_cfg.pop('batch_size')
        num_workers = data_cfg.pop('num_workers')
        train_cfg = data_cfg.pop('train')
        valid_cfg = data_cfg.pop('valid')

        train_cfg = {**data_cfg, **train_cfg}
        valid_cfg = {**data_cfg, **valid_cfg}
        train_cfg = {k: v for k, v in train_cfg.items()}
        valid_cfg = {k: v for k, v in valid_cfg.items()}

        if not os.path.exists(train_file):
            if self.state.is_main_process:
                self.logger.info(f"Meta file not found at {train_file}, preparing metadata...")
            prepare_audiodata(root_dir, train_file, valid_file, valid_ratio)
            
        # 2. 构建数据集
        train_dataset = AudioDataset.from_meta(
            root=self.cfg.dataset.train_file,
            **train_cfg
        )
        valid_dataset = AudioDataset.from_meta(
            root=self.cfg.dataset.valid_file,
            **valid_cfg
        )

        if self.state.is_main_process:
            self.logger.info(f"Built datasets - Train: {len(train_dataset)} samples, Valid: {len(valid_dataset)} samples")

        # 3. 创建数据加载器
        loader_cfg = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'collate_fn': train_dataset.collactor
        }

        if self.state.is_distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=train_cfg['shuffle'])
            valid_sampler = DistributedSampler(valid_dataset, shuffle=valid_cfg['shuffle'])
            
            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                **loader_cfg
            )
            valid_loader = DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                **loader_cfg
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                shuffle=train_cfg['shuffle'],
                **loader_cfg
            )
            valid_loader = DataLoader(
                valid_dataset,
                shuffle=valid_cfg['shuffle'],
                **loader_cfg
            )
        
        # 4. 记录数据集配置
        def log_dataset_info(prefix: str, dataset: AudioDataset, cfg: DictConfig):
            """记录数据集的配置信息"""
            # 基础信息
            self.logger.info(f"{prefix} Dataset Config:")
            self.logger.info(f"├── Number of samples: {len(dataset)}")
            self.logger.info(f"├── Number of audio files: {len(dataset.meta)}")
            self.logger.info(f"├── Target sample rate: {dataset.sample_rate}")
            self.logger.info(f"├── Target channels: {dataset.channels}")
            
            # 音频切片设置
            if dataset.segment_duration is not None:
                self.logger.info(f"├── Segment duration: {dataset.segment_duration:.2f}s")
                self.logger.info(f"├── Min segment ratio: {dataset.min_segment_ratio:.2f}")
            else:
                self.logger.info("├── Segment duration: None (using full audio)")
            
            # 采样设置
            self.logger.info(f"├── Shuffle enabled: {dataset.shuffle}")
            if dataset.shuffle:
                self.logger.info(f"├── Shuffle seed: {dataset.shuffle_seed}")
            self.logger.info(f"├── Sample on duration: {dataset.sample_on_duration}")
            self.logger.info(f"└── Max retry read: {dataset.max_retry_read}")
            
            # 音频时长统计
            durations = [m.duration for m in dataset.meta]
            total_hours = sum(durations) / 3600
            self.logger.info(f"\n\n{prefix} Audio Statistics:")
            self.logger.info(f"├── Total duration: {total_hours:.2f} hours")
            self.logger.info(f"├── Average duration: {np.mean(durations):.2f}s")
            self.logger.info(f"├── Min duration: {min(durations):.2f}s")
            self.logger.info(f"└── Max duration: {max(durations):.2f}s")
        
        # 记录训练集和验证集信息
        if self.state.is_main_process:
            self.logger.info("\n" + "="*50 + "\nDataset Information\n" + "="*50)
            log_dataset_info("Training", train_dataset, train_cfg)
            log_dataset_info("Validation", valid_dataset, valid_cfg)
            
            # 记录数据加载器配置
            self.logger.info("\n\nDataLoader Config:")
            self.logger.info(f"├── Batch size: {loader_cfg['batch_size']}")
            self.logger.info(f"├── Num workers: {loader_cfg['num_workers']}")
            self.logger.info(f"├── Pin memory: {loader_cfg['pin_memory']}")
            self.logger.info(f"└── Distributed: {self.state.is_distributed}")
            self.logger.info("\n" + "="*50 + "\n")  

        return train_loader, valid_loader 

    def prepare_training(self) -> MimiTrainer:
        train_loader, valid_loader = self.build_dataloaders()
        model = self.build_model()

        trainer = MimiTrainer(
            cfg=self.cfg,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            state=self.state,
            logger=self.logger,
            output_dir=self.output_dir
        )

        return trainer

    def cleanup(self) -> None:
        """清理分布式环境"""
        if self.logger is not None:
            self.logger.close()

        if self.state.is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@hydra.main(config_path="../config", config_name="mimi")
def main(cfg: DictConfig):
    system = TrainingSystem(cfg)

    try:
        system.initialize()

        trainer = system.prepare_training()

        if cfg.checkpoint.resume_from_checkpoint:
            trainer.load_checkpoint(cfg.checkpoint.resume_from_checkpoint)

        trainer.train()

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # 获取详细的错误信息
        error_details = {
            'error_type': exc_type.__name__,
            'error_message': str(e),
            'rank': system.state.local_rank,
            'stack_trace': ''.join(traceback.format_tb(exc_traceback)),
            'local_variables': {
                name: repr(value)
                for name, value in locals().items()
                if not name.startswith('__') and name != 'e'
            }
        }

        # 记录详细错误信息
        system.logger.error("Training failed with the following details:")
        system.logger.error(f"Error Type: {error_details['error_type']}")
        system.logger.error(f"Error Message: {error_details['error_message']}")
        system.logger.error(f"Process Rank: {error_details['rank']}")
        system.logger.error(f"Stack Trace:\n{error_details['stack_trace']}")
        system.logger.error("Local Variables at Time of Error:")
        for var_name, var_value in error_details['local_variables'].items():
            system.logger.error(f"  {var_name}: {var_value}")

        # 如果是分布式训练，确保所有进程都停止
        if dist.is_initialized():
            system.logger.info("Attempting to terminate all distributed processes...")
            try:
                dist.destroy_process_group()
            except Exception as e:
                system.logger.error(f"Error while destroying process group: {str(e)}")

        raise

    finally:
        system.cleanup()

if __name__ == "__main__":
    main()