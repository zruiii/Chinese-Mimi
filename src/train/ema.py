import typing as tp
from collections import defaultdict
import torch
import torch.nn as nn

class ModuleDictEMA:
    """对 nn.ModuleDict 实现指数移动平均(EMA)
    1. 在初始化 ema 的时候，会将模型的浮点类型参数深度拷贝到 ema.state 中
    2. 在训练的时候, optimizer.step() 执行就会更新 self.model 的参数, 
    这个时候 ema.module_dict 也会同步更新；
    3. 模型参数更新后执行 ema.step 相当于基于更新后的 self.model 参数来更新 ema.state
    4. 整个 EMA 的作用相当于模型的一个副本 (即 ema.state)
    只不过它的参数会相对于 self.model 而言更平滑一些。
    """
    def __init__(
        self,
        module_dict: nn.ModuleDict,                     # 引用模型, 会随着模型更新
        decay: float = 0.999,                           # EMA 衰减率 
        unbias: bool = True,                            # 是否使用偏置修正
        device: tp.Union[torch.device, str] = 'cpu'     # 运行设备
    ):
        self.decay = decay                              
        self.module_dict = module_dict                  
        self.device = device                            
        self.unbias = unbias                         
        
        # EMA 状态字典,key为模块名,value为参数字典
        self.state = defaultdict(dict)
        # 用于偏置修正的计数器
        self.count = 0
        
        # 初始化 EMA 状态
        for module_name, module in self.module_dict.items():
            # 遍历模型参数和缓冲区
            for name, param in list(module.named_parameters()) + list(module.named_buffers()):
                # 只对浮点类型参数做 EMA
                if not param.is_floating_point():
                    continue
                    
                # 确定设备并初始化 EMA 状态
                target_device = self.device or param.device 
                if name not in self.state[module_name]:
                    # 深拷贝参数到指定设备
                    self.state[module_name][name] = param.detach().to(target_device, copy=True)

    def step(self):
        """更新 EMA,执行一步指数移动平均"""
        # 计算当前步的权重
        if self.unbias:
            # 使用偏置修正
            self.count = self.count * self.decay + 1
            weight = 1.0 / self.count  
        else:
            # 不使用偏置修正
            weight = 1.0 - self.decay
            
        # 遍历所有模块更新 EMA
        for module_name, module in self.module_dict.items():
            # 遍历参数和缓冲区
            for name, param in list(module.named_parameters()) + list(module.named_buffers()):
                if not param.is_floating_point():
                    continue
                
                # q_dropout=True 的时候 _embedding 是非持久化缓存, 需要跳过
                if name not in self.state[module_name]:
                    continue
                    
                # 目标设备
                target_device = self.device or param.device
                
                # 更新 EMA: 
                # new_ema = old_ema * (1-w) + param * w
                self.state[module_name][name].mul_(1 - weight)
                self.state[module_name][name].add_(param.detach().to(target_device), alpha=weight)

    @property
    def state_dict(self) -> dict:
        """返回 EMA 的状态字典,用于保存检查点"""
        return {
            'state': self.state,      # EMA参数状态
            'count': self.count       # 偏置修正计数器
        }

    def load_state_dict(self, state_dict: dict):
        """从状态字典加载 EMA 状态
        
        Args:
            state_dict: 包含 state 和 count 的状态字典
        """
        self.count = state_dict['count']
        
        # 加载参数状态
        for module_name, module in state_dict['state'].items():
            for param_name, param in module.items():
                self.state[module_name][param_name].copy_(param)