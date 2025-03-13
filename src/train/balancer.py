import typing as tp
from collections import defaultdict
import torch
from torch import autograd, distributed

class LossBalancer:
    """ 平衡多个损失函数的梯度 """
    def __init__(
        self,
        weights: tp.Dict[str, float],           # 每个损失的权重字典
        balance_grads: bool = True,
        total_norm: float = 1.0,                # 重缩放梯度时的参考范数
        ema_decay: float = 0.999,               # EMA 衰减率
        per_batch_item: bool = True,            # 是否按批次项计算平均范数
        epsilon: float = 1e-12,
        monitor: bool = False
    ):
        self.weights = weights
        self.balance_grads = balance_grads
        self.total_norm = total_norm or 1.0
        self.epsilon = epsilon
        self.monitor = monitor
        self.per_batch_item = per_batch_item
        self._metrics = {}
        
        self._fix = defaultdict(float)
        self._total = defaultdict(float)
        self.ema_decay = ema_decay or 1.0

        self._averager = self._create_averager(self.ema_decay)
        
    def _create_averager(self, beta: float):
        """ 重构移动平均计算器 """
        def update(metrics: dict, weight: float = 1) -> dict:
            for key, value in metrics.items():
                self._total[key] = self._total[key] * beta + weight * float(value)
                self._fix[key] = self._fix[key] * beta + weight
            return {key: tot / self._fix[key] for key, tot in self._total.items()}
        return update

    def _average_metrics(self, metrics: tp.Dict[str, float], count: float = 1.0):
        """
        在所有工作进程间平均指标
        
        参数：
        - metrics: 需要平均的指标字典
        - count: 非标准化权重
        """
        # 检查是否在分布式环境中
        if not self._is_distributed():
            return metrics
            
        # 准备数据进行分布式计算
        keys, values = zip(*metrics.items())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
        tensor *= count
        
        # 执行分布式归约
        if self._is_distributed():
            distributed.all_reduce(tensor, distributed.ReduceOp.SUM)
            
        # 计算平均值
        averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
        return dict(zip(keys, averaged))

    def _is_distributed(self):
        """检查是否在分布式环境中运行"""
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size() > 1
        return False

    # !WARNING: `backward` 实际上进行了两次反向求导，这在 DDP 环境下可能会导致同一参数标记多次
    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor) -> torch.Tensor:
        """计算损失的反向传播并返回有效训练损失
        - losses: 损失字典，键需要与weights中的键匹配
        - input: 模型输出的生成样本

        这里实际上存在两个约束:
        约束1: 各个损失的梯度范数比例要符合权重比例
        约束2: 所有梯度范数之和保持为指定的 total_norm=1.0
        """
        # 计算每个损失的梯度和范数
        norms = {}
        grads = {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [input], retain_graph=True)     # 只是计算梯度，但不会累积到 .grad 属性

            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
                
            norms[name] = norm
            grads[name] = grad

        # 计算批次大小和平均范数
        batch_size = 1 if not self.per_batch_item else len(input)
        avg_norms = self._average_metrics(self._averager(norms), batch_size)
        total_norm = sum(avg_norms.values())

        # 更新监控指标
        if self.monitor:
            self._metrics = {
                f'ratio_{k}': v / total_norm 
                for k, v in avg_norms.items()
            }

        # 计算期望的梯度比例
        total_weights = sum(self.weights[k] for k in avg_norms)
        desired_ratios = {
            k: w / total_weights 
            for k, w in self.weights.items()
        }

        # 组合梯度并计算有效损失
        out_grad = torch.zeros_like(input)
        effective_loss = torch.tensor(0., device=input.device, dtype=input.dtype)
        
        for name, avg_norm in avg_norms.items():
            if self.balance_grads:
                # 根据期望比例缩放梯度
                scale = desired_ratios[name] * self.total_norm / (self.epsilon + avg_norm)
            else:
                # 直接使用权重
                scale = self.weights[name]
                
            out_grad.add_(grads[name], alpha=scale)
            effective_loss = effective_loss + scale * losses[name].detach()

        # 执行反向传播
        input.backward(out_grad)

        return effective_loss

    @property
    def metrics(self):
        """获取监控指标"""
        return self._metrics
    
    def state_dict(self):
        """安全地保存状态"""
        return {
            'fix': dict(self._fix),
            'total': dict(self._total),
            'ema_decay': self.ema_decay
        }
    
    def load_state_dict(self, state):
        """安全地加载状态"""
        self._fix.clear()
        self._total.clear()
        self._fix.update(state['fix'])
        self._total.update(state['total'])
        self.ema_decay = state['ema_decay']