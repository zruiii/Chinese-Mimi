from functools import wraps
import inspect
import os
import typing as tp

import torch
from torch import cuda
from contextlib import contextmanager

# 全局变量，用于追踪是否在CUDA图中执行
_in_cuda_graph = False

# 全局变量，用于禁用CUDA图功能
_disable_cuda_graph = False

# 在调用 torch.compile 的时候，Pytorch 会检查全局变量 _compile_disabled
_compile_disabled: bool = True

def torch_compile_lazy(fun):
    """
    torch.compile 在被调用时就会立即创建进程池，即使这个被编译的函数还没有被实际使用
    这可能会带来不必要的资源开销，并且在使用 Ctrl+C 终止程序时会产生大量stderr输出
    """
    if os.environ.get("NO_TORCH_COMPILE"):
        return fun
    fun_compiled = None

    @wraps(fun)
    def _wrapped(*args, **kwargs):
        nonlocal fun_compiled
        if _compile_disabled:
            return fun(*args, **kwargs)
        if fun_compiled is None:
            fun_compiled = torch.compile(fun)
        return fun_compiled(*args, **kwargs)

    return _wrapped
    
@contextmanager
def _set_in_cuda_graph():
    """上下文管理器，用于标记代码块是否在CUDA图中执行"""
    global _in_cuda_graph
    # 确保不会嵌套使用CUDA图
    assert not _in_cuda_graph
    _in_cuda_graph = True
    try:
        yield
    finally:
        # 退出时恢复状态
        _in_cuda_graph = False

def _is_cuda_graph_enabled() -> bool:
    """检查环境是否支持CUDA图功能"""
    # 如果全局禁用，直接返回False
    if _disable_cuda_graph:
        return False
    # 检查环境变量是否禁用CUDA图
    no_cuda_graph = os.environ.get("NO_CUDA_GRAPH", "")
    if no_cuda_graph.lower() not in {"0", "no", "n", ""}:
        return False
    return True

def in_cuda_graph() -> bool:
    """判断当前是否在CUDA图中执行"""
    return _in_cuda_graph

class CUDAGraphed:
    """
    CUDA图优化装饰器类
    
    参数:
        func: 要优化的函数，参数中的张量应该是顶层参数，不应嵌套在数据结构中
        warmup_steps: 在启用CUDA图之前的预热步骤数
        disable: 是否禁用CUDA图优化
    """

    def __init__(self, func: tp.Callable, warmup_steps: int = 1, disable: bool = False):
        self.func = func
        self.warmup_steps = warmup_steps
        self.disable = disable
        # CUDA图对象
        self._graph: cuda.CUDAGraph | None = None
        # 缓存的输出结果
        self._output: tuple | None = None
        # 缓存的输入参数
        self._args: tuple | None = None

    def reset(self, warmup_steps: int = 0) -> None:
        """
        重置CUDA图状态
        在输入形状改变或外部状态改变时使用
        """
        self.warmup_steps = warmup_steps
        self._graph = None
        self._output = None
        self._args = None

    def __call__(self, *args, **kwargs) -> tp.Any:
        """函数调用入口"""
        # 不支持关键字参数
        if kwargs:
            raise RuntimeError("不支持命名参数")
        
        # 如果禁用或无法使用CUDA图，或者当前在CUDA图中，直接执行原函数
        if self.disable or not _is_cuda_graph_enabled() or in_cuda_graph():
            return self.func(*args, **kwargs)

        def _clone_tensors(args: tuple) -> tuple:
            """克隆所有张量参数"""
            out: list = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = arg.clone()
                out.append(arg)
            return tuple(out)

        def _match_values_copy_tensors(args: tuple, target_args: tuple) -> None:
            """
            检查参数匹配性并更新张量数据
            确保新参数与缓存的参数形状相同，非张量参数值相同
            """
            if len(args) != len(target_args):
                raise ValueError(
                    f"期望参数数量 {len(target_args)}，但获得了 {args}"
                )
            for idx, (source, target) in enumerate(zip(args, target_args)):
                if isinstance(target, torch.Tensor):
                    if not isinstance(source, torch.Tensor):
                        raise ValueError(
                            f"参数 #{idx} 应该是张量，但得到了 {source}"
                        )
                    if source.shape != target.shape:
                        raise ValueError(
                            f"参数 #{idx} 应该形状为 {target.shape}，但得到了形状 {source.shape}"
                        )
                    # 将新数据复制到缓存的张量中
                    target.copy_(source)
                else:
                    if isinstance(source, torch.Tensor):
                        raise ValueError(
                            f"参数 #{idx} 不应该是张量，但得到了张量"
                        )
                    if source is not target and source != target:
                        raise ValueError(
                            f"参数 #{idx} 值从 {target} 变为 {source}"
                        )

        # 设置CUDA图执行环境
        with _set_in_cuda_graph():
            # 如果图还未创建
            if self._graph is None:
                if self.warmup_steps <= 0:
                    # 创建新的CUDA图
                    self._graph = cuda.CUDAGraph()
                    # 克隆参数以确保独立性
                    self._args = _clone_tensors(args)
                    """
                    下面这个上下文管理只是记录 self.func() 图，但并不执行
                    # - 记录内存位置
                    # - 记录操作类型
                    # - 记录数据流向
                    """
                    with cuda.graph(self._graph):
                        self._output = self.func(*self._args)
                    # 首次执行图
                    self._graph.replay()
                    return self._output
                else:
                    # 预热阶段：直接执行函数
                    self.warmup_steps -= 1
                    return self.func(*args)
            else:
                # 图已存在：更新数据并重放
                assert self._args is not None
                assert self._output is not None
                # 检查并更新参数
                _match_values_copy_tensors(args, self._args)
                # 重放图
                self._graph.replay()
                return self._output
                