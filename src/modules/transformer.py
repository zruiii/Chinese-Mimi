import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np

from einops import rearrange
from dataclasses import dataclass

from .rope import RotaryEmbedding
from .base import StreamingModule, StreamingContainer
from ..utils.contextlibs import no_compile, ExitStack
from ..utils.helper import get_activation, create_norm_fn, create_sin_embedding
from ..utils import torch_compile_lazy

from transformers.modeling_flash_attention_utils import _flash_attention_forward

# ********** KV Cache **********
class KVCacheResult(tp.NamedTuple):
    keys: torch.Tensor                      
    values: torch.Tensor                    
    positions: torch.Tensor                 

    @staticmethod
    def from_kv(keys: torch.Tensor, values:torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=keys.dtype)
        return KVCacheResult(keys, values, positions)

class RingKVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,                                  # 缓存容量
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )

        self.capacity = capacity
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)
    
    def reset(self):
        """重置缓存状态"""
        self.end_offset.zero_()
    
    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        """ 更新缓存 """
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        
        B, H, T, D = k.shape
        device = self.end_offset.device

        # 更新 KV Cache
        indexes = torch.arange(T, device=device, dtype=torch.long) + self.end_offset
        indexes = indexes % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)

        keys = self.cache[0]
        values = self.cache[1]

        # 更新 positions
        # 长度等于 capacity, 未填充的位置用-1标注
        # 记录了最近 capacity 个时刻在缓存中的位置
        indexes = torch.arange(self.capacity, device=device, dtype=torch.long)
        last_offset = self.end_offset + T - 1
        end_index = last_offset % self.capacity
        delta = indexes - end_index

        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        self.end_offset.add_(T)
        invalid = indexes >= self.end_offset
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


# ********** MultiHead Attention **********
@dataclass
class _MHAState:
    """多头注意力的状态管理类"""
    kv_cache: RingKVCache          # 键值对缓存
    offset: torch.Tensor           # 当前处理的位置（GPU）
    offset_cpu: int                # 当前处理的位置（CPU，用于权重索引）

    def reset(self):
        """重置状态"""
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class StreamingMultiheadAttention(StreamingModule[_MHAState]):
    """
    当模型采取因果计算(因果训练 or 正常推理)时，会采用掩码attention
    流式处理在这里主要体现在 KV Cache 的管理
    """
    def __init__(
        self,
        embed_dim: int,                         # 输入维度
        num_heads: int,                         # 注意力头数
        causal: bool = True,                    # 是否使用因果掩码
        context: tp.Optional[int] = None,       # 上下文窗口大小
        rope = None,                            # 旋转位置编码
        weights_per_step: int = 0,              # 每个时间步是否使用不同权重
        device = None,
        dtype = None,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.context = context
        self.rope = rope
        self.weights_per_step = weights_per_step
        self.use_flash_attn = use_flash_attn
        
        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 初始化投影矩阵
        out_dim = 3 * embed_dim  
        mult = weights_per_step if weights_per_step else 1
        
        # 输入投影（支持每步不同权重）
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        self.in_proj_weight = self.in_proj.weight
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, mult * embed_dim, bias=False, **factory_kwargs)

    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        """初始化流式处理状态"""
        if self.context is None:
            if self.weights_per_step:
                capacity = self.weights_per_step
            else:
                raise RuntimeError(
                    "Cannot create streaming KVCache without context or weights_per_step."
                )
        else:
            capacity = self.context
            
        device = self.in_proj_weight.device
        dtype = self.in_proj_weight.dtype
        
        kv_cache = RingKVCache(
            batch_size, 
            self.num_heads,
            self.head_dim,
            capacity,
            device,
            dtype
        )
        
        return _MHAState(
            kv_cache=kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0
        )

    def _complete_kv(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        """完成键值对的缓存更新"""
        state = self._streaming_state
        if state is None:
            # 非流式模式[训练阶段]：直接返回当前的键值对, positions 就是 [0,...,T]
            return KVCacheResult.from_kv(k, v)
        else:
            # 流式模式：更新并返回缓存的键值对
            return state.kv_cache.complete(k, v)

    def _multi_linear(
        self,
        num_linear: int,
        weight: torch.Tensor,
        x: torch.Tensor,
        offset: int
    ) -> torch.Tensor:
        """对每个时间步使用不同的线性变换
        
        Args:
            num_linear: 线性变换的数量
            weight: 权重矩阵
            x: 输入张量
            offset: 当前时间步的偏移量
        """
        B, T, C = x.shape
        ys = []
        chout, chin = weight.shape
        
        weight = weight.view(num_linear, -1, chin)      # [T, C_out, C_in]
        
        # 对每个时间步应用对应的权重
        for t in range(T):
            y = F.linear(x[:, t], weight[t + offset])
            ys.append(y)
        
        # 拼接结果
        out = torch.stack(ys, 1)
        return out

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        query, key, value: [B, T, C]
        """
        state = self._streaming_state
        T = query.shape[1]
        
        # 获取偏移量
        if state is None:
            offset = torch.zeros(1, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            assert self.causal, "Streaming only available for causal attention"
            offset = state.offset
            offset_cpu = state.offset_cpu
        
        # QKV 投影
        if self.weights_per_step:
            projected = self._multi_linear(
                self.weights_per_step,
                self.in_proj_weight,
                query,
                offset_cpu
            )
        else:
            projected = F.linear(query, self.in_proj_weight)            # [B, T, C']
        
        # 分离 Q, K, V
        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)   # [B, H, T, D]

        # 应用位置编码
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)
        
        # 获取完整的键值对
        k, v, pos_k = self._complete_kv(k, v)
        
        # 计算注意力掩码
        if self.causal:
            pos_k = pos_k.view(1, -1)
            pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
            delta = pos_q - pos_k                               # [T, Context]
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
        else:
            attn_bias = None
        
        # 计算注意力分数
        if self.use_flash_attn:
            x = _flash_attention_forward(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=0.0,
            )
        else:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=0.0
            )
        # 重新排列维度
        x = rearrange(x, "b h t d -> b t (h d)")
        
        # 输出投影
        if self.weights_per_step:
            x = self._multi_linear(
                self.weights_per_step,
                self.out_proj.weight,
                x,
                offset_cpu
            )
        else:
            x = self.out_proj(x)
        
        # 更新状态
        if state is not None:
            state.offset.add_(T)
            state.offset_cpu += T
        
        return x


# ********** TransformerLayer **********
class LayerScale(nn.Module):
    """层缩放模块
    对每个通道学习一个缩放因子，用于控制残差连接的强度。
    """
    def __init__(
        self,
        channels: int,
        init: float = 1e-4,             # 初始缩放值
        device = None,
        dtype = None
    ):
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((channels,), init, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class ActivationGating(nn.Module):
    """ 门控激活模块 """
    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        activation,
        **factory_kwargs
    ):
        super().__init__()
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        => h: [B, T, 2D]
        => z: activation(h₁) ⊙ (h₂)
        """
        # 输入投影
        x = self.linear_in(x)
        B, T, _ = x.shape
        
        # 分离门控信号和主要特征
        x = x.view(B, T, 2, -1)
        
        # 应用门控：activation(gate) ⊙ feature
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        
        # 输出投影
        return self.linear_out(x)

@dataclass
class _LayerState:
    """Transformer层的状态"""
    offset_cpu: int

    def reset(self):
        self.offset_cpu = 0

class StreamingTransformerLayer(StreamingModule[_LayerState]):
    """
    相比于传统的 Transformer, 这里在 FFN 中引入参数化的门控机制
    并且引入了层缩放机制管理残差网络
    流式处理主要体现在对 FFN 计算中的时间步管理 (only if 每个时间步采用不同的门控机制)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,                                 # 注意力头数
        dim_feedforward: int | list[int] = 2048,        # 前馈网络维度
        causal: bool = False,                           # 是否因果
        context: tp.Optional[int] = None,               # causal mask 感受野
        rope: tp.Optional[RotaryEmbedding] = None,      # 是否采用 RoPE 编码
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,         # 层缩放初始值
        gating: str = "none",                           # 新增参数
        weights_per_step: int = 0,
        activation=F.gelu,
        skip_self_attn: bool = False,                   # 跳过Self-atten层
        device = None,
        dtype = None,
        use_flash_attn: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.skip_self_attn = skip_self_attn
        if not skip_self_attn:
            self.self_attn = StreamingMultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                causal=causal,
                context=context,
                weights_per_step=weights_per_step,
                rope=rope,
                use_flash_attn=use_flash_attn,
                **factory_kwargs
            )
            
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        
        # 层缩放
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)
        
        # 前馈网络
        self.weights_per_step = weights_per_step
        self.gating = None
        self.linear1 = None
        self.linear2 = None
        self.activation = activation
        
        # 设置前馈网络或门控机制
        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == weights_per_step, \
                f"Length of dim_feedforward {len(dim_feedforward)} != weights_per_step {weights_per_step}"
        
        if gating == "none":
            # 使用标准前馈网络
            assert not weights_per_step, "weights_per_step without gating not supported"
            assert not isinstance(dim_feedforward, list), \
                "List dim_feedforward without gating not supported"
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)
        else:
            # 使用门控机制
            if weights_per_step:
                # 每个时间步使用不同的门控
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * weights_per_step
                assert isinstance(dim_feedforward, list)
                self.gating = nn.ModuleList([
                    self._make_gating(gating, dim, **factory_kwargs)
                    for dim in dim_feedforward
                ])
            else:
                # 使用单个门控
                assert isinstance(dim_feedforward, int)
                self.gating = self._make_gating(gating, dim_feedforward, **factory_kwargs)
    
    def _make_gating(self, gating: str, dim_feedforward: int, **factory_kwargs) -> nn.Module:
        """创建门控模块
        
        Args:
            name: 门控类型
            dim: 输入维度
            dim_feedforward: 前馈网络维度
        """
        # 创建门控模块
        gating = ActivationGating(
            dim=self.d_model,
            dim_feedforward=dim_feedforward,
            activation=gating,
            **factory_kwargs
        )
        
        # 验证参数数量不超过标准前馈网络
        max_params = 2 * self.d_model * dim_feedforward
        params = sum(p.numel() for p in gating.parameters())
        assert params <= max_params, f"{gating} gating has {params} params, max is {max_params}"
        
        return gating

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        return _LayerState(offset_cpu=0)

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        """带层缩放的自注意力块"""
        if self.skip_self_attn:
            return x

        # 1. 归一化
        x_norm = self.norm1(x)
        # 2. 自注意力
        attn_output = self.self_attn(x_norm, x_norm, x_norm)
        # 3. 应用层缩放并做残差连接
        return x + self.layer_scale_1(attn_output)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """前馈网络或门控块"""
        state = self._streaming_state
        offset = 0 if state is None else state.offset_cpu
        
        x_orig = x
        x = self.norm2(x)
        
        if self.gating is None:
            assert self.linear1 is not None and self.linear2 is not None
            x = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                B, T, D = x.shape
                ys = []
                for t in range(T):
                    y = self.gating[offset + t](x[:, t:t+1])
                    ys.append(y)
                x = torch.cat(ys, dim=1)
            else:
                x = self.gating(x)
        
        return x_orig + self.layer_scale_2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        with ExitStack() as stack:
            if x.device.type != 'cuda':
                stack.enter_context(no_compile())
            x = self._sa_block(x)
            x = self._ff_block(x)
        
            if self._streaming_state is not None:
                self._streaming_state.offset_cpu += x.shape[1]
                
            return x

# ********** Transformer **********
@dataclass
class _TransformerState:
    offset: torch.Tensor

    def reset(self):
        self.offset.zero_()

class StreamingTransformer(StreamingModule[_TransformerState]):
    """
    输入 Transformer 之前添加正弦绝对位置编码
    计算 attention 的时候才使用 rope 来编码相对位置信息
    流式状态管理绝对位置信息
    """
    def __init__(
        self,
        d_model: int,                                           
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        device=None,
        dtype=None,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    use_flash_attn=use_flash_attn,
                    **kwargs,
                )
            )

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(offset=torch.zeros(1, device=device, dtype=torch.long))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        state = self._streaming_state
        if state is None:
            offset = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            offset = state.offset

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offset.view(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)         # [1, T, C]
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if state is not None:
            state.offset.add_(T)
        return x

class ProjectedTransformer(StreamingContainer):
    """ 
    在 Transformer 之前添加一个映射层转换输入信号维度
    利用多个输出线性层来支持不同维度的输出
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )
    
    def forward(self, x, *args, **kwargs):
        """
        x: [B, D, T]
        """
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = y.transpose(1, 2)
            ys.append(y)
        return ys

if __name__ == "__main__":
    # """ 测试 Transformer
    _transformer_kwargs = {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 8,
        "causal": True,
        "layer_scale": 0.01,
        "context": 250,
        "max_period": 10000,
        "gating": "none",
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "dim_feedforward": 2048,
        "input_dimension": 512,
        "output_dimensions": [512],
    }
    encoder_transformer = ProjectedTransformer(
        device="cpu", **_transformer_kwargs
    )

    # 非流式推理
    # z = torch.randn(1, 512, 25)
    # print(f"输入矩阵大小: {z.shape}")

    # encoder_transformer._stop_streaming()
    # (embed,) = encoder_transformer(z)
    # print(f"非流式推理表征元素和: {embed.sum()}")

    # # 开启流式推理状态
    # encoder_transformer._start_streaming(1)
    # (embed,) = encoder_transformer(z)
    # print(f"流式推理表征元素和: {embed.sum()}")

    # 非流式推理
    z = torch.randn(1, 512, 250)
    print(f"输入矩阵大小: {z.shape}")

    # encoder_transformer._stop_streaming()
    # (embed,) = encoder_transformer(z)
    # print(f"非流式推理表征元素和: {embed.sum()}")

    # 开启流式推理状态
    encoder_transformer._start_streaming(1)

    for _ in range(3):
        (embed,) = encoder_transformer(z)
        print(f"流式推理表征元素和: {embed.sum()}")