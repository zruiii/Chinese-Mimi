import abc
import typing as tp
import torch.nn as nn

from dataclasses import dataclass
from contextlib import contextmanager

class Resetble(tp.Protocol):
    def reset(self) -> None:
        pass

# 定义带约束的泛型，比如实现 reset 函数
State = tp.TypeVar('State', bound=Resetble)

class StreamingModule(abc.ABC, nn.Module, tp.Generic[State]):
    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State | None = None          # 流式处理状态
        self._streaming_propagate: bool = True              # 是否传播到子类
    
    @property
    def is_streaming(self) -> bool:
        return self._streaming_state is not None
    
    def set_streaming_propagate(self, streaming_propagate: bool) -> None:
        self._streaming_propagate = streaming_propagate

    @abc.abstractmethod
    def _init_streaming_state(self, batch_size: int) -> State:
        """ 初始化流式处理状态 """
        pass
    
    def _apply_named_streaming(self, fn: tp.Any) -> None:
        """ 将函数递归应用到所有流式模块 """
        def _handle_module(prefix: str, module: nn.Module, recurse: bool = True):
            propagate = True

            # 当遇到流式模块，并且对应的传播标签为False时，终止传播
            if isinstance(module, StreamingModule):
                if module._streaming_propagate:
                    fn(prefix, module)
                else:
                    propagate = False

            if not recurse:
                return

            if propagate:
                for name, child in module.named_children():
                    _handle_module(prefix + "." + name, child)
        
        _handle_module("", self, recurse=False)
        for name, child in self.named_children():
            _handle_module(name, child)
    
    def _start_streaming(self, batch_size: int) -> None:
        """ 启动流式处理模式 """
        def start_streaming(name: str, module: StreamingModule):
            module._streaming_state = module._init_streaming_state(batch_size)
    
        self._apply_named_streaming(start_streaming)
    
    def _stop_streaming(self) -> None:
        """ 停用流式处理模式 """
        def stop_streaming(name: str, module: StreamingModule):
            module._streaming_state = None
        
        self._apply_named_streaming(stop_streaming)

    def reset_streaming(self) -> None:
        """ 重置流式处理状态 """
        def _reset(name: str, module: StreamingModule):
            state = module._streaming_state
            if state is None:
                raise ValueError(f"Trying to reset streaming, but {name} wasn't streaming.")
            state.reset()
        self._apply_named_streaming(_reset)

    @contextmanager
    def streaming(self, batch_size: int) -> tp.Iterator[None]:
        """ 临时流式处理 """
        self._start_streaming(batch_size)
        try:
            yield     # 这里什么都不返回，因此是 Iterator[None]
        finally:
            self._stop_streaming()

    def streaming_forever(self, batch_size: int) -> None:
        """ 设置所有模块永久流式处理 """
        self._start_streaming(batch_size)
    
    def get_streaming_state(self) -> dict[str, tp.Any]:
        state: dict[str, tp.Any] = {}

        def _add(name: str, module: StreamingModule):
            state[name] = module._streaming_state
        
        self._apply_named_streaming(_add)
        return state
    
    def set_streaming_state(self, state: dict[str, tp.Any]):
        """ 自定义所有模块的流式处理状态 """

        def _set(name: str, module: StreamingModule):
            if name in state:
                module._streaming_state = state.pop(name)
            else:
                # * RuntimeError 通常用于表示程序运行时出现的错误，而不是编程错误（那种会用 ValueError 或 TypeError）。
                raise RuntimeError(f"Expected to find a streaming state for {name}.")
        
        self._apply_named_streaming(_set)
        if state:
            raise RuntimeError(f"Some states were not consumed: {list(state.keys())}")


@dataclass
class _NullState:
    def reset(self) -> None:
        pass

class StreamingContainer(StreamingModule[_NullState]):
    def _init_streaming_state(self, batch_size: int) -> _NullState:
        return _NullState()
