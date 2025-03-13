import abc
import sys
from collections import deque
import _collections_abc
from types import GenericAlias, MethodType

from contextlib import contextmanager

_compile_disabled: bool = False

@contextmanager
def no_compile():
    """Disable torch.compile locally. Now Pytorch 2.4 provides a function to do that."""
    global _compile_disabled

    prev_disabled = _compile_disabled
    _compile_disabled = True
    try:
        yield
    finally:
        _compile_disabled = prev_disabled

class AbstractContextManager(abc.ABC):

    """An abstract base class for context managers."""

    __class_getitem__ = classmethod(GenericAlias)

    def __enter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractContextManager:
            return _collections_abc._check_methods(C, "__enter__", "__exit__")
        return NotImplemented


class _BaseExitStack:
    """
    ExitStack 和 AsyncExitStack 的基类。
    这个类用于管理上下文管理器的进入和退出操作。
    主要用于同时处理多个上下文管理器的情况。
    即: 多个状态管理器的管理器

    # 不使用 ExitStack 的情况
    with open('file1.txt') as f1:
        with open('file2.txt') as f2:
            with open('file3.txt') as f3:
                # 处理文件

    # 使用 ExitStack
    with ExitStack() as stack:
        files = [stack.enter_context(open(f)) for f in filenames]
        # 处理文件
    """

    @staticmethod
    def _create_exit_wrapper(cm, cm_exit):
        """
        创建一个退出包装器
        将上下文管理器的 exit 方法绑定到实例上
        
        参数:
            cm: 上下文管理器对象
            cm_exit: 上下文管理器的退出方法
        """
        return MethodType(cm_exit, cm)  # MethodType用于将方法绑定到实例上

    @staticmethod
    def _create_cb_wrapper(callback, /, *args, **kwds):
        """
        创建一个回调函数的包装器
        
        参数:
            callback: 回调函数
            *args: 位置参数
            **kwds: 关键字参数
        """
        def _exit_wrapper(exc_type, exc, tb):
            # 创建一个符合 __exit__ 方法签名的包装器
            callback(*args, **kwds)
        return _exit_wrapper

    def __init__(self):
        """
        初始化方法
        创建一个双端队列来存储退出回调函数
        """
        self._exit_callbacks = deque()  # 使用双端队列存储退出回调

    def pop_all(self):
        """
        保存当前的上下文栈，并将其转移到一个新的实例中
        
        返回:
            new_stack: 包含当前所有回调的新栈实例
        """
        new_stack = type(self)()                            # 创建一个新的实例
        new_stack._exit_callbacks = self._exit_callbacks    # 转移回调队列
        self._exit_callbacks = deque()                      # 重置当前实例的回调队列
        return new_stack

    def push(self, exit):
        """
        注册一个具有标准 __exit__ 方法签名的回调函数
        
        参数:
            exit: 可以是上下文管理器对象或者普通的可调用对象
            
        返回:
            exit: 允许作为装饰器使用
        """
        _cb_type = type(exit)

        try:
            exit_method = _cb_type.__exit__  # 尝试获取 __exit__ 方法
        except AttributeError:
            # 如果不是上下文管理器，就假设它是一个可调用对象
            self._push_exit_callback(exit)
        else:
            # 如果是上下文管理器，则注册其 __exit__ 方法
            self._push_cm_exit(exit, exit_method)
        return exit  # 返回原对象，允许作为装饰器使用

    def enter_context(self, cm):
        """
        注册上下文管理器
        
        参数:
            cm: 上下文管理器
            
        返回:
            result: __enter__ 方法的返回值
        """
        _cm_type = type(cm)                 # 获取类型，以便查找特殊方法
        _exit = _cm_type.__exit__           # 获取退出方法
        result = _cm_type.__enter__(cm)     # 调用进入方法
        self._push_cm_exit(cm, _exit)       # 注册退出回调
        return result

    def callback(self, callback, /, *args, **kwds):
        """
        注册一个任意的回调函数及其参数
        注意：这种方式注册的回调不能抑制异常
        
        参数:
            callback: 回调函数
            *args: 位置参数
            **kwds: 关键字参数
            
        返回:
            callback: 允许作为装饰器使用
        """
        _exit_wrapper = self._create_cb_wrapper(callback, *args, **kwds)
        _exit_wrapper.__wrapped__ = callback  # 保存原始回调函数，便于后续检查
        self._push_exit_callback(_exit_wrapper)
        return callback

    def _push_cm_exit(self, cm, cm_exit):
        """
        辅助方法：正确注册 __exit__ 方法的回调
        
        参数:
            cm: 上下文管理器对象
            cm_exit: 退出方法
        """
        _exit_wrapper = self._create_exit_wrapper(cm, cm_exit)
        self._push_exit_callback(_exit_wrapper, True)

    def _push_exit_callback(self, callback, is_sync=True):
        """
        将退出回调添加到回调队列中
        
        参数:
            callback: 回调函数
            is_sync: 是否为同步回调（用于区分同步和异步操作）
        """
        self._exit_callbacks.append((is_sync, callback))


class ExitStack(_BaseExitStack, AbstractContextManager):
    """Context manager for dynamic management of a stack of exit callbacks.

    For example:
        with ExitStack() as stack:
            files = [stack.enter_context(open(fname)) for fname in filenames]
            # All opened files will automatically be closed at the end of
            # the with statement, even if attempts to open files later
            # in the list raise an exception.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        received_exc = exc_details[0] is not None

        # We manipulate the exception state so it behaves as though
        # we were actually nesting multiple with statements
        frame_exc = sys.exc_info()[1]
        def _fix_exception_context(new_exc, old_exc):
            # Context may not be correct, so find the end of the chain
            while 1:
                exc_context = new_exc.__context__
                if exc_context is None or exc_context is old_exc:
                    # Context is already set correctly (see issue 20317)
                    return
                if exc_context is frame_exc:
                    break
                new_exc = exc_context
            # Change the end of the chain to point to the exception
            # we expect it to reference
            new_exc.__context__ = old_exc

        # Callbacks are invoked in LIFO order to match the behaviour of
        # nested context managers
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            is_sync, cb = self._exit_callbacks.pop()
            assert is_sync
            try:
                if cb(*exc_details):
                    suppressed_exc = True
                    pending_raise = False
                    exc_details = (None, None, None)
            except:
                new_exc_details = sys.exc_info()
                # simulate the stack of exceptions by setting the context
                _fix_exception_context(new_exc_details[1], exc_details[1])
                pending_raise = True
                exc_details = new_exc_details
        if pending_raise:
            try:
                # bare "raise exc_details[1]" replaces our carefully
                # set-up context
                fixed_ctx = exc_details[1].__context__
                raise exc_details[1]
            except BaseException:
                exc_details[1].__context__ = fixed_ctx
                raise
        return received_exc and suppressed_exc

    def close(self):
        """Immediately unwind the context stack."""
        self.__exit__(None, None, None)
