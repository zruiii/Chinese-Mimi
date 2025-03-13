import torch
import torch.nn as nn
import omegaconf
from collections import defaultdict
from omegaconf import DictConfig
from pathlib import Path
import torch.distributed as dist

from torch.utils.data import DataLoader
from contextlib import contextmanager

from ..modules import quantizer
from .adv_loss import AdversarialLoss
from .recon_loss import MultiScaleMelLoss
from .distill_loss import DistillLoss
from .balancer import LossBalancer
from .ema import ModuleDictEMA
from src.utils import TrainingState, DistributedLogger

class MimiTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        state: TrainingState,
        logger: DistributedLogger,
        output_dir: str
    ) -> None:
        # 基础配置 
        self.cfg = cfg
        self.logger = logger
        self.state = state
        self.device = state.device
        self.output_dir = output_dir

        model_save_path = Path(cfg.checkpoint.dir) / output_dir
        if state.is_main_process:
            model_save_path.mkdir(parents=True, exist_ok=True)
        self.model_save_path = model_save_path

        # 训练进度
        self._current_stage = None
        
        # 模型(先用小批量数据强制触发编译)
        with torch.no_grad():
            if hasattr(model, "quantizer"):
                training_state = model.quantizer.training
                model.quantizer.eval()      # 切换到评估模式，避免初始化码本
                
            dummy_input = torch.randn(1, 1, 16000*2).to(self.state.device)
            _ = model(dummy_input)
            
            if hasattr(model, "quantizer"):
                # 恢复原始状态
                model.quantizer.train(training_state)

        self.model = model
        if self.state.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.state.local_rank],
                output_device=self.state.local_rank,
                find_unused_parameters=False
            )
            self.model._set_static_graph()   # 启用静态图模式

        # 数据集
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.updates_per_epoch = cfg.optim.updates_per_epoch or len(train_loader)

        # 初始化组件
        self.only_adv = (cfg.losses.l1 == 0 and cfg.losses.msspec == 0)

        self._init_criterion()  # 损失函数
        self._init_optimizer()  # 优化器
        self._init_ema()        # EMA

        # 随机数生成器
        self.rng = torch.Generator(device="cpu")

    def _init_criterion(self):
        """ 初始化损失函数 """
        device = self.state.device
        loss_weights = {}
        self.adv_loss = AdversarialLoss(self.cfg.msstftd, device=device)
        if self.only_adv:
            self.recon_losses = nn.ModuleDict({})
        else:
            self.recon_losses = nn.ModuleDict({
                'l1': torch.nn.L1Loss(),
                'msspec': MultiScaleMelLoss(**self.cfg.msspec).to(device)
            })
        distillor = DistillLoss(self.cfg.seanet.dimension, 1024).to(device)

        # 配置损失权重
        loss_cfg = self.cfg.losses
        loss_weights['adv_msstftd'] = loss_cfg.get('adv')
        loss_weights['feat_msstftd'] = loss_cfg.get('feat')
        loss_weights['l1'] = loss_cfg.get('l1')
        loss_weights['msspec'] = loss_cfg.get('adv')

        # 同步模型参数
        if self.state.is_distributed:
            if not self.only_adv:
                self.broadcast_model(self.recon_losses['msspec'])
            self.distill_loss = nn.parallel.DistributedDataParallel(
                distillor,
                device_ids=[self.state.local_rank],
                output_device=self.state.local_rank
            )
        else:
            self.distill_loss = distillor

        # 初始化损失平衡器
        balancer_kwargs = omegaconf.OmegaConf.to_container(self.cfg.balancer, resolve=True)
        self.balancer = LossBalancer(weights=loss_weights, **balancer_kwargs)
    
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

    def _init_optimizer(self):
        """初始化优化器"""
        if self.cfg.optim.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optim.lr, **self.cfg.optim.adam)
        elif self.cfg.optim.optimizer == "adamw":
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if 'transformer' in name.lower():
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
                    
            param_groups = [
                {'params': decay_params, 'weight_decay': self.cfg.optim.adamw.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.cfg.optim.lr, betas=self.cfg.optim.adamw.betas)
        else:
            raise ValueError(f"Unsupported Optimizer: {self.cfg.optimizer}")

    def _init_ema(self):
        """初始化 EMA"""
        ema_cfg = self.cfg.optim.ema
        if ema_cfg.get('use', False):
            self.ema = ModuleDictEMA(
                module_dict=nn.ModuleDict({"model": self.model}),
                decay=ema_cfg.get('decay', 0.999),
                unbias=ema_cfg.get('unbias', True),
                device=self.device
            )

            self.logger.info(
                f'Initializing EMA on the model with decay = {ema_cfg.decay}'
                f' every {ema_cfg.updates} updates'
            )
        else:
            self.ema = None

    def _sync_gradients_and_buffers(self):
        """同步梯度(分布式训练)"""
        if not self.state.is_distributed:
            return
        
        # 同步梯度
        handles = []
        for param in self.model.parameters():
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
            param.grad.data /= self.state.world_size
            
        # 同步 buffers
        handles = []
        for buffer in self.model.buffers():
            handle = dist.all_reduce(
                buffer.data,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            handles.append((buffer, handle))
            
        # 等待 buffers 同步完成并缩放    
        for buffer, handle in handles:
            handle.wait()
            buffer.data /= self.state.world_size

    def _average_metrics(self, metrics: dict, count: float = 1.):
        """平均指标(分布式训练)"""
        if not self.state.is_distributed:
            return metrics

        # 转换为tensor并同步
        device = next(self.model.parameters()).device
        keys, values = zip(*metrics.items())
        tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
        tensor *= count
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # 计算平均值
        averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
        return dict(zip(keys, averaged))

    @contextmanager
    def _swap_ema_state(self):
        """EMA 模型状态切换的上下文管理器"""
        if self.ema is None:
            yield
        else:
            # 保存原始状态
            orig_states = {}
            for name, module in self.model.named_children():
                orig_states[name] = {
                    k: v.detach().clone() 
                    for k, v in module.state_dict().items()
                }

            # 加载 EMA 状态
            for module_name, module in self.model.named_children():
                if module_name in self.ema.state:
                    module_state = {
                        name: param.detach().clone()
                        for name, param in self.ema.state[module_name].items()
                    }
                    module.load_state_dict(module_state, strict=False)

            try:
                yield
            finally:
                # 恢复原始状态 
                for module_name, module in self.model.named_children():
                    if module_name in orig_states:
                        module.load_state_dict(orig_states[module_name], strict=False)

    def save_checkpoint(self):
        """保存检查点"""
        if not self.state.is_main_process:
            return
        
        try:
            # NOTE: RVQ 的 dropout 和 mask 操作中的随机状态无法保留
            state = {
                # 1. 基础训练状态
                'epoch': self.state.epoch,
                'distributed_state': {
                    'world_size': self.state.world_size,
                    'rank': self.state.global_rank,
                    'local_rank': self.state.local_rank
                } if self.state.is_distributed else None,
                
                # 2. 模型相关
                'model': self.model.state_dict(),
                'ema': self.ema.state_dict if self.ema else None,
                
                # 3. 优化器相关
                'optimizer': self.optimizer.state_dict(),
                'discriminator_optimizer': self.adv_loss.optimizer.state_dict(),
                
                # 4. 损失相关
                'balancer': self.balancer.state_dict(),
                
                # 5. 判别器和其他损失函数
                'discriminator': self.adv_loss.discriminator.state_dict(),
                'distill_loss': self.distill_loss.state_dict()
            }
        
            # 保存最新检查点
            if self.cfg.checkpoint.save_last:
                latest_path = self.model_save_path / 'checkpoint_latest.pt'
                torch.save(state, latest_path)
            
            # 定期保存
            if self.cfg.checkpoint.save_every and (self.state.epoch + 1) % self.cfg.checkpoint.save_every == 0:
                epoch_path = self.model_save_path / f'checkpoint_epoch{self.state.epoch + 1}.pt'
                torch.save(state, epoch_path)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
        
    def load_checkpoint(self, path: str):
        """加载检查点恢复训练"""
        state = torch.load(path, map_location=self.device)
        
        # 1. 恢复基础训练状态
        self.state.epoch = state['epoch'] + 1
        if self.cfg.checkpoint.reset_epoch:
            self.state.epoch = 0
            self.logger.info("reset epoch as 0")
        
        # 验证分布式训练状态
        if self.state.is_distributed:
            dist_state = state['distributed_state']
            assert dist_state is not None, "Checkpoint not from distributed training!"
            assert dist_state['world_size'] == self.state.world_size, "World size mismatch!"
        
        # 2. 恢复模型状态
        self.model.load_state_dict(state['model'])
        if self.ema and state['ema']:
            self.ema.load_state_dict(state['ema'])
            
        # 3. 恢复优化器状态
        self.optimizer.load_state_dict(state['optimizer'])
        self.adv_loss.optimizer.load_state_dict(state['discriminator_optimizer'])
        
        # 4. 恢复损失相关状态
        self.balancer.load_state_dict(state['balancer'])
        
        # 5. 恢复其他模型组件
        self.adv_loss.discriminator.load_state_dict(state['discriminator'])
        self.distill_loss.load_state_dict(state['distill_loss'])
        
        self.logger.info(f"Successfully restored training state from epoch {self.state.epoch}")


    def train(self):
        for epoch in range(self.state.epoch, self.cfg.optim.epochs):
            self.logger.info(f"Starting epoch {epoch}")
            
            # 设置随机种子
            self.rng.manual_seed(1234 + epoch)
            
            # 训练阶段
            self._current_stage = 'train'
            self.model.train()
            train_metrics = self.run_epoch()
            
            # 验证阶段
            self._current_stage = 'valid' 
            self.model.eval()
            with torch.no_grad():
                with self._swap_ema_state():
                    valid_metrics = self.run_epoch()
                    
            # 更新状态
            self._log_metrics('train', train_metrics)
            self._log_metrics('valid', valid_metrics)
            
            # 保存检查点
            self.save_checkpoint()       

            self.state.update_epoch()
                
    def run_epoch(self):
        """运行一个 epoch"""
        # 获取对应的数据加载器
        loader = self.train_loader if self._current_stage == 'train' else self.valid_loader
        
        # 分布式训练设置
        if self.state.is_distributed:
            if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
                loader.sampler.set_epoch(self.state.epoch)

        # 指标统计
        metrics_sum = defaultdict(float)
        metrics_count = defaultdict(int)
        # 新增: 用于每 k 步的指标统计
        step_metrics_sum = defaultdict(float)  
        step_metrics_count = defaultdict(int)

        updates = self.updates_per_epoch if self._current_stage == 'train' else len(loader)

        for idx, batch in enumerate(loader):
            wav = batch['wav']              # [B, 1, Time]
            embed = batch['embed']          # [B, Frame, D]
            if idx >= updates:
                break
            
            # 执行一步训练/验证
            metrics = self.run_step(wav, embed)
            
            # EMA 更新
            if self._current_stage == 'train' and self.ema is not None:
                if (idx + 1) % self.cfg.optim.ema.updates == 0:
                    self.logger.debug("EMA model setup")
                    self.ema.step()
                    
            # 更新指标统计
            for k, v in metrics.items():
                v = v.item() if torch.is_tensor(v) else v
                metrics_sum[k] += v
                metrics_count[k] += 1
                # 新增: 更新每k步的指标统计
                step_metrics_sum[k] += v
                step_metrics_count[k] += 1
            
            # 新增: 每k步打印一次指标
            if self._current_stage == 'train' and (idx + 1) % self.cfg.optim.print_freq == 0:
                # 计算平均指标
                step_metrics = {
                    k: step_metrics_sum[k] / step_metrics_count[k] 
                    for k in step_metrics_sum
                }
                # 多 GPU 同步指标
                step_metrics = self._average_metrics(step_metrics, self.cfg.optim.print_freq)
                
                # 打印日志
                metric_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in step_metrics.items() 
                    if not k.startswith('_')  # 跳过内部指标
                )
                self.logger.info(
                    f"[Epoch {self.state.epoch}][{self._current_stage}][{idx+1}/{updates}] {metric_str}"
                )

                # 清空每k步的指标统计
                step_metrics_sum.clear()
                step_metrics_count.clear()
                
        # 计算平均指标
        metrics = {k: metrics_sum[k] / metrics_count[k] for k in metrics_sum}
        
        # 多 GPU 同步指标
        return self._average_metrics(metrics, updates)

    def run_step(self, wav, embed):
        """执行一步训练/验证"""
        # == 1. 前向传播 ==
        x = wav.to(self.device)
        embed = embed.to(torch.float32).to(self.device)
        y = x.clone()

        qres, sres = self.model(x)
        assert isinstance(qres, quantizer.QuantizedResult)
        y_pred = qres.x                                         
        metrics = {'bandwidth': qres.bandwidth.mean().item()}

        # 训练判别器
        if self._current_stage == 'train':
            if torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                disc_loss = self.adv_loss.train_discriminator(y_pred, y)
                metrics['d_msstftd'] = disc_loss
                metrics['d_loss'] = disc_loss

        # == 2. 损失计算 ==
        balanced_losses = {}
        other_losses = {}

        # 码本中的量化损失
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty
            metrics['penalty'] = qres.penalty.item()

        # 生成器损失
        adv_gen_loss, feat_loss = self.adv_loss(y_pred, y)
        balanced_losses['adv_msstftd'] = adv_gen_loss
        balanced_losses['feat_msstftd'] = feat_loss
        metrics['adv_msstftd'] = adv_gen_loss.item()
        metrics['feat_msstftd'] = feat_loss.item()

        # 重构损失
        for loss_name, criterion in self.recon_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss
            metrics[loss_name] = loss.item()

        # 语义表征的蒸馏损失
        distill_loss = self.distill_loss(sres.x, embed)
        other_losses['distill'] = distill_loss
        metrics['distill'] = distill_loss.item()
        # if self.state.is_main_process: pdb.set_trace()

        if self._current_stage == 'train':
            # == 3. 反向传播 ==
            # 量化器惩罚项
            # other_loss = torch.tensor(0., device=self.device)
            # if "penalty" in other_losses:
            #     other_loss += other_losses['penalty']

            other_loss = sum(loss for loss in other_losses.values())

            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(p.grad.data.norm(p=2).pow(2) 
                           for p in self.model.parameters() if p.grad is not None)
                metrics['ratio1'] = ratio1.sqrt()

            # 平衡损失反向传播
            metrics['g_loss'] = self.balancer.backward(balanced_losses, y_pred)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2) 
                       for p in self.model.parameters() if p.grad is not None)
            metrics['ratio2'] = ratio2.sqrt()
            metrics.update(self.balancer.metrics)

            # == 4. 梯度裁剪 ==
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.optim.max_norm
                )
            
            # == 5. 同步梯度 ==
            self._sync_gradients_and_buffers()
            
            # == 6. 优化器更新参数 ==            
            self.optimizer.step()

            # == 7. 清零梯度 ==
            self.optimizer.zero_grad()

        return metrics

    def _log_metrics(self, stage: str, metrics: dict):
        """记录训练指标"""
        # 构建日志字符串
        metric_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items() 
            if not k.startswith('_')  # 跳过内部指标
        )
        
        self.logger.info(f"[Epoch {self.state.epoch}][{stage}] {metric_str}")
        self.logger.log_metrics(metrics, step=self.state.epoch, stage=stage)
