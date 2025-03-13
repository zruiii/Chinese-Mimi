import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np

from torch import distributed
from einops import rearrange, repeat
from dataclasses import dataclass, field

from ..utils.helper import is_distributed, zero_scalar

# ************ 编码本(欧式空间) ************
class _CodebookForwardResult(tp.NamedTuple):
    codes: torch.Tensor                     # 量化编码
    quantized: torch.Tensor                 # 量化表征
    metrics: tp.Dict[str, torch.Tensor]     # 存储相关指标


class EuclideanCodebook(nn.Module):
    """ 欧式空间下的码本 """
    def __init__(
        self,
        dim: int,                                   # 表征维度
        codebook_size: int,                         # 码本大小
        decay: float = 0.99,                        # EMA 衰减率
        threshold_usage_ratio: float = 0.1,         # 码本最低利用率 | 至少达到平均利用率的10%
        replaced_usage_ratio: float = 1.0,          # 替代编码的使用率默认为平均利用率
        check_unused_every: int = 5,                # 检查码本利用率的间隔
        epsilon: float = 1e-5,                      
    ):
        super().__init__()
        self.decay = decay
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False

        # 注册缓冲区
        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))  # 码本是否已经初始化的标识符
        self.register_buffer("cluster_usage", torch.ones(codebook_size))                # 每个编码的使用率  [Q]
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, dim))          # 每个编码覆盖的表征之和 [Q, D]
        self.register_buffer("_embedding", None, persistent=False)                      # 临时缓冲区

    @property
    def embedding(self) -> torch.Tensor:
        """ 码本表征: [Q, D] """
        if self._embedding is None:
            embedding = (
                self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    @property
    def initialized(self) -> bool:
        """ 标记码本是否已经换成初始化 """
        if not self._cached_initialized:
            self._cached_initialized = self._initialized.item()
        return self._cached_initialized

    def _sample_vectors(self, samples, num):
        """从样本中随机采样向量"""
        num_samples = samples.shape[0]
        device = samples.device
        
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)
        return samples[indices]

    def _run_kmeans(self, samples, num_clusters, num_iters=50):
        """ 使用K-means算法初始化码本
        samples: [N, D]
        num_clusters: K
        """
        dim = samples.shape[-1]
        means = self._sample_vectors(samples, num_clusters)                 # [K, D]
        
        for _ in range(num_iters):
            dists = torch.cdist(samples[None], means[None], p=2)[0]         # [N, K]
            buckets = dists.argmin(dim=-1)                                  # [N]

            # 统计每个聚类的样本数
            bins = torch.bincount(buckets, minlength=num_clusters)          # [K]
            bins.clamp_(min=1)
            
            # 对每一个簇求平均，更新聚类中心
            new_means = torch.zeros_like(means)
            new_means.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), samples)
            new_means /= bins.unsqueeze(-1)
            
            # 对于空聚类重新采样补充聚类中心
            zero_mask = bins == 0
            resampled = self._sample_vectors(samples, num_clusters)
            means = torch.where(zero_mask.unsqueeze(-1), resampled, new_means)
        
        return means, bins

    def _init_embedding(self, data: torch.Tensor) -> None:
        """ 码本初始化
        在分布式训练时, 只在rank 0上执行初始化, 然后广播给其他进程
        主要给 embedding_sum 和 cluster_usage 赋值
        """
        if self.initialized:
            return

        rank = 0
        if is_distributed():
            rank = distributed.get_rank()
            # 在分布式环境中收集所有GPU上的数据
            if rank == 0:
                other_shapes: tp.List[torch.Size] = [None] * distributed.get_world_size()
                distributed.gather_object(data.shape, other_shapes)
                other_data: tp.List[torch.Tensor] = [
                    torch.empty(shape, device=data.device, dtype=data.dtype) for shape in other_shapes]
                distributed.gather(data, other_data)
                data = torch.cat(other_data, dim=0)
            else:
                distributed.gather_object(data.shape)
                distributed.gather(data)
                
        if rank == 0:
            # 在主进程上执行K-means初始化
            embedding, cluster_usage = self._run_kmeans(data, self.codebook_size)
            self.embedding_sum.data.copy_(embedding * cluster_usage[:, None])       # 每个簇表征累积 = 中心表征 * 覆盖样本数
            self.cluster_usage.data.copy_(cluster_usage)                            # 每个簇覆盖样本数
            self._initialized.data.fill_(1)
            
        # 确保所有进程的缓冲区同步
        self._broadcast_buffers()

    def _check_expired_codes(self, batch_samples: torch.Tensor) -> torch.Tensor:
        """
        检查并替换使用率低的编码本向量
        每隔check_unused_every次迭代检查一次
        返回不合格编码的占比
        """
        assert batch_samples.dim() == 2
        if not self.initialized:
            return zero_scalar(batch_samples.device)

        self._next_unused_check -= 1
        if self._next_unused_check > 0:
            return zero_scalar(batch_samples.device)
        else:
            self._next_unused_check = self.check_unused_every
    
        # 找到低于使用率阈值的编码
        threshold_cluster_usage = self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        expired_codes = self.cluster_usage < threshold_cluster_usage

        # 从当前batch样本中随机采样向量作为新的编码
        new_vectors = self._sample_vectors(batch_samples, self.codebook_size)               # [K, D]
        replace_cluster_usage = self.replaced_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        self.embedding_sum[:] = torch.where(expired_codes[:, None], replace_cluster_usage * new_vectors, self.embedding_sum)
        self.cluster_usage[:] = torch.where(expired_codes, replace_cluster_usage, self.cluster_usage)

        # 同步所有进程的缓冲区
        self._broadcast_buffers()

        return expired_codes.float().mean()

    def _broadcast_buffers(self) -> None:
        """
        在分布式训练环境中，将rank 0的缓冲区广播到所有其他进程
        """
        if is_distributed():
            for buffer in self.buffers():
                distributed.broadcast(buffer, 0)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """ 检索最近聚类中心作为索引编码 """
        assert x.dim() == 2  
        dists = torch.cdist(x[None], self.embedding[None], p=2)[0]          # [N, codebook_size]
        codes = dists.argmin(dim=-1)                                        # [N]
        return codes
    
    @staticmethod
    def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入向量编码为最近的聚类中心的索引
        输入 x: [B, T, D] -> 输出 codes: [B, T]
        """
        shape = x.shape
        x = rearrange(x, "... d -> (...) d")                # [(B*T), D]
        codes = self._quantize(x)                           # [(B*T)]
        codes = codes.view(*shape[:-1])                     # [B, T]
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = F.embedding(codes, self.embedding)
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _CodebookForwardResult:
        """
        x: [B, T, D]
        训练过程主要通过 EMA 的方式更新下面两个参数:
        - self.cluster_usage: 记录每个编码的使用频率 [Q]
        - self.embedding_sum: 记录每个编码覆盖的所有样本表征之和 [Q, D]
        """
        shape = x.shape
        x = rearrange(x, "... d -> (...) d")                # [N, D]

        # 训练模式下初始化代码本
        if self.training and initialize:
            self._init_embedding(x.detach())

        # 向量量化: 离散聚类中心替代连续表征
        flat_codes = self._quantize(x)                      # [N]
        codes = flat_codes.view(*shape[:-1])                # [batch, ...]
        quantized = self.decode(codes)                      # [batch, ..., dim]

        # 训练时更新编码本统计信息
        metrics = {}
        if self.training:
            # 每间隔几步就检查使用率低的编码，并进行替换
            expired = self._check_expired_codes(x)
            metrics['rvq_expired'] = expired
            
            # 更新编码覆盖样本数
            cluster_usage = torch.zeros_like(self.cluster_usage)
            cluster_usage.scatter_add_(0, flat_codes, torch.ones_like(flat_codes, dtype=cluster_usage.dtype))
            self._ema_inplace(self.cluster_usage, cluster_usage, self.decay)
            # self.cluster_usage.data.mul_(self.decay).add_(cluster_usage, alpha=(1 - self.decay))

            # 更新编码本向量和
            embedding_sum = torch.zeros_like(self.embedding_sum)
            embedding_sum.scatter_add_(0, repeat(flat_codes, "n -> n d", d=self.dim), x)
            self._ema_inplace(self.embedding_sum, embedding_sum, self.decay)
            # self.embedding_sum.data.mul_(self.decay).add_(embedding_sum, alpha=(1 - self.decay))

            # 计算码本覆盖率的香农熵，取值 [0,1]
            if self.initialized:
                proba = self.cluster_usage / self.cluster_usage.sum()
                p_log_p = torch.where(proba == 0, zero_scalar(self.cluster_usage.device), proba * torch.log(proba))
                metrics['rvq_entropy'] = -p_log_p.sum() / math.log(self.codebook_size)

            # 清空 _embedding 缓存
            self.register_buffer("_embedding", None)

        return _CodebookForwardResult(quantized, codes, metrics)
    

# ************ 量化器组件 ************
class _VQForwardResult(tp.NamedTuple):
    quantized: torch.Tensor                 # 离散表征
    codes: torch.Tensor                     # 量化编码
    loss: torch.Tensor                      # 损失函数
    metrics: tp.Dict[str, torch.Tensor]     

class VectorQuantization(nn.Module):
    """
    向量量化，支持码本输入输出的表征维度对齐
    """
    def __init__(
        self,
        dim: int,                                           # 表征维度
        codebook_size: int,                                 # 码本大小
        codebook_dim: tp.Optional[int] = None,              # 码本表征维度
        decay: float = 0.99,                                # 码本 EMA 更新衰减率
        epsilon: float = 1e-5,                              
        threshold_usage_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim
        
        # 如果输入维度和码本维度不同，需要投影层
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()

        # 码本
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_usage_ratio=threshold_usage_ratio,
            **kwargs,
        )

    @property
    def embedding(self):
        return self._codebook.embedding

    @property
    def initialized(self):
        return self._codebook.initialized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T]
        """
        x = rearrange(x, "b d t -> b t d")
        x = self.project_in(x)
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, T]
        quantized: [B, D, T]
        """
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = rearrange(quantized, "b t d -> b d t")
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _VQForwardResult:
        """
        x: [B, D, T]
        """
        x = rearrange(x, "b d t -> b t d")
        x = self.project_in(x)
        quantized, codes, metrics = self._codebook(x, initialize=initialize)

        # 利用直通估计来计算 VQ 的损失
        if self.training:
            quantized = x + (quantized - x).detach()
            loss = F.mse_loss(x, quantized.detach())
        else:
            loss = zero_scalar(x.device)

        quantized = self.project_out(quantized)
        quantized = rearrange(quantized, "b t d -> b d t")

        return _VQForwardResult(quantized, codes, loss, metrics)


class ResidualVectorQuantization(nn.Module):
    """残差向量量化实现。
    通过多个VQ层级逐步量化输入信号，每一层处理前一层的残差。 residule = x - quantized
    物理含义: 每一层处理原始信号中未被量化器捕获的部分。
    """

    def __init__(self, *, num_quantizers: int, codebook_offset: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.codebook_offset = codebook_offset          # 码本偏移，用于区分语义和声学码本

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        """
        x: [B, D, T]
        out_indices: [K, B, T]
        """
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [K, B, T]
        quantized: [B, D, T]
        """
        quantized = zero_scalar(codes.device)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized    
    
    def forward(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> _VQForwardResult:
        """
        x: [B, D, T]
        n_q: 使用量化器数目
        """
        quantized_out = zero_scalar(x.device)
        residual = x  # 初始残差就是输入信号

        all_losses = []
        all_codes = []
        all_metrics: tp.Dict[str, torch.Tensor] = {}

        n_q = n_q or len(self.layers)
        previous_layer_is_initialized = True

        # !NOTE: 这里采用逐层初始化, 每次forward最多只初始化一个新的码本
        for i, layer in enumerate(self.layers[:n_q]):
            if self.training:
                this_layer_is_initialized = layer.initialized
            
            # 量化当前残差
            quantized, codes, loss, metrics = layer(
                residual, initialize=previous_layer_is_initialized
            )
            
            if self.training:
                previous_layer_is_initialized = this_layer_is_initialized

            # 计算新的残差并累积量化结果
            quantized = quantized.detach()    
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            # 收集每层的结果
            all_codes.append(codes)
            all_losses.append(loss)

            # 更新指标
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        if self.training:
            quantized_out = x + (quantized_out - x).detach()

        out_losses, out_codes = map(torch.stack, (all_losses, all_codes))       # [N_q], [N_q, B, T]
        return _VQForwardResult(quantized_out, out_codes, out_losses, all_metrics)


# ************ 完整量化器 ************
@dataclass
class QuantizedResult:
    """存储量化结果的数据类"""
    x: torch.Tensor                                     # 量化后的张量
    codes: torch.Tensor                                 # 量化编码
    bandwidth: torch.Tensor                             # 带宽 (kb/s), 每个batch项
    penalty: tp.Optional[torch.Tensor] = None           # 每一层量化前后的损失
    metrics: dict = field(default_factory=dict)         # 存储指标的字典

class BaseQuantizer(nn.Module):
    """量化器的基类"""
    def __init__(self):
        super().__init__()
        self._ema_frozen = False
    
    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        """前向传播函数
        Args:
            x: 输入张量
            frame_rate: 帧率,用于正确计算带宽
        Returns:
            QuantizedResult: 量化结果
        """
        raise NotImplementedError()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """将输入张量编码为离散的整数编码"""
        raise NotImplementedError()
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """将整数编码解码为量化向量"""
        raise NotImplementedError()
        
    @property
    def cardinality(self) -> int:
        """每个码本的基数"""
        raise NotImplementedError()
        
    @property
    def total_codebooks(self) -> int:
        """码本总数"""
        raise NotImplementedError()
        
    @property
    def num_codebooks(self) -> int:
        """活跃码本数量"""
        raise NotImplementedError()
        
    @property 
    def semantic_quantizer(self) -> 'BaseQuantizer':            # 因为是指向自身，所以需要引号
        """ 语义量化器 """
        return self
        
    @property
    def acoustic_quantizer(self) -> 'BaseQuantizer':
        """ 声学量化器 """
        return self
        
    def set_num_codebooks(self, n: int) -> None:
        """ 设置活跃码本数量 """
        raise NotImplementedError()
        
    @property
    def ema_frozen(self) -> bool:
        """ 是否对码本应用EMA """
        return self._ema_frozen
        
    def ema_frozen_(self, ema_frozen: bool) -> None:
        """ 设置是否对码本应用EMA """
        self._ema_frozen = ema_frozen

class ResidualVectorQuantizer(BaseQuantizer):
    """
    ResidualVectorQuantization 作为核心模块
    + 输入/输出映射层
    + 量化器随机 dropout
    """
    def __init__(
        self,
        dimension: int = 128,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
        n_q: int = 8,                                           # 量化器数量
        q_dropout: bool = False,                                # 随机丢弃部分量化器
        no_quantization_rate: float = 0.0,                      # 不进行量化概率
        bins: int = 1024,                                       # 码本大小
        decay: float = 0.99,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        codebook_offset: int = 0,
        force_projection: bool = False,                         # 量化前后强制使用MLP
    ):
        super().__init__()
        self.max_n_q = n_q                                     
        self.n_q = n_q                                          
        self.q_dropout = q_dropout                              
        self.no_quantization_rate = no_quantization_rate        
        self.dimension = dimension                             
        self.input_dimension = input_dimension or dimension    
        self.output_dimension = output_dimension or dimension  
        self.bins = bins                                       
        self.decay = decay                                     
        self.rng_dropout = random.Random(1234)                  

        # 根据需要创建输入投影层
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(self.input_dimension, self.dimension, 1, bias=False)

        # 根据需要创建输出投影层
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

        # 创建残差向量量化模块
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            threshold_usage_ratio=threshold_usage_ratio,
            replaced_usage_ratio=replaced_usage_ratio,
            codebook_offset=codebook_offset,
        )

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        x: [B, D, T]
        frame_rate: SEANetEncoder 输出帧率
        """
        n_q = self.n_q
        x = self.input_proj(x) 
        
        # 训练时的量化器随机丢弃
        if self.training and self.q_dropout:
            n_q = self.rng_dropout.randint(1, self.n_q)
            
        # 带宽
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        bw = torch.tensor(n_q * bw_per_q).to(x)

        # 进行向量量化
        quantized, codes, commit_loss, metrics = self.vq(x, n_q=n_q)
        
        # 训练时随机指定部分样本的输出和输入保持一致
        # 但是每一层的量化损失都会计算
        B, _, _ = quantized.shape
        if self.training and self.no_quantization_rate > 0:
            mask = (torch.rand(B, 1, 1, device=x.device) <= self.no_quantization_rate).float()
            quantized = x * mask + (1 - mask) * quantized
            
        # 输出投影
        quantized = self.output_proj(quantized)  
        codes = codes.transpose(0, 1)            # [B, N_q, T]
        
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss), metrics=metrics)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T]
        """
        n_q = self.n_q
        # 处理空输入的特殊情况
        if x.shape[-1] == 0:
            return torch.empty((x.shape[0], n_q, 0), device=x.device, dtype=torch.int64)

        # 输入投影
        x = self.input_proj(x)  # [B, dimension, T]
        codes = self.vq.encode(x, n_q=n_q)          # [K, B, T]
        codes = codes.transpose(0, 1)               # [B, K, T]
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, K, T]
        """
        codes = codes.transpose(0, 1)       # [K, B, T]
        quantized = self.vq.decode(codes)   # [B, dimension, T]
        quantized = self.output_proj(quantized)  # [B, output_dimension, T]
        return quantized

    @property
    def total_codebooks(self):
        """返回最大可用的码本数量"""
        return self.max_n_q

    @property
    def num_codebooks(self):
        """返回当前使用的码本数量"""
        return self.n_q

    def set_num_codebooks(self, n: int):
        """设置使用的码本数量
        
        参数:
            n (int): 要使用的码本数量，必须在[0, max_n_q]范围内
        """
        assert n >= 0 and n <= self.max_n_q
        self.n_q = n

    @property
    def cardinality(self) -> int:
        """返回每个码本的大小（包含的码字数量）"""
        return self.bins

class SplitResidualVectorQuantizer(BaseQuantizer):
    """
    分别建模 semantic 和 acoustic 信息
    """
    def __init__(
        self,
        *args,
        n_q: int = 8,                                  # 总的量化器数目 
        n_q_semantic: int = 1,                         # 语音量化器数目
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, "总量化器数量必须大于语义量化器数量"
        
        self.max_n_q = n_q                              # 最大量化器数量
        self.n_q_semantic = n_q_semantic                # 语义量化器数量
        self.n_q_acoustic = n_q - n_q_semantic          # 声学量化器数量
        
        q_dropout = kwargs.pop("q_dropout", False)
        
        # 创建语义量化器（第一阶段）
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        
        # 创建声学量化器（第二阶段）
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            codebook_offset=1,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def _renorm_and_add(
        self,
        first_val: torch.Tensor,
        rest_val: torch.Tensor,
        n_q_semantic: int,
        n_q_acoustic: int,
    ):
        """ 语义信息和声学信息的加权平均 """
        n_q = n_q_semantic + n_q_acoustic
        renorm_first_val = first_val * n_q_semantic / n_q
        renorm_rest_val = rest_val * n_q_acoustic / n_q
        return renorm_first_val + renorm_rest_val

    def forward(self, x: torch.Tensor, frame_rate: int):
        """前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, T]
                B: batch size
                C: 通道数
                T: 序列长度
            frame_rate (int): 输入的帧率
            
        返回:
            QuantizedResult: 量化结果，包含量化后的张量、码本索引等信息
        """
        # 第一阶段：语义量化
        semantic_result = self.rvq_first(x, frame_rate)
        if self.n_q == self.n_q_semantic:
            return semantic_result
            
        # 第二阶段：声学量化
        acoustic_result = self.rvq_rest(x, frame_rate)
        
        # 合并两个阶段的量化结果
        full_quantized_emb = semantic_result.x + acoustic_result.x  # [B, D, T]
        full_quantized_codes = torch.cat([semantic_result.codes, acoustic_result.codes], dim=1)  # [B, K, T]
        
        # 获取实际使用的量化器数量
        n_q_semantic = semantic_result.codes.shape[1]
        n_q_acoustic = acoustic_result.codes.shape[1]
        
        # 计算总带宽
        full_quantized_bandwidth = semantic_result.bandwidth + acoustic_result.bandwidth
        
        # 计算总的损失
        full_quantized_penalty = self._renorm_and_add(
            semantic_result.penalty, acoustic_result.penalty, n_q_semantic, n_q_acoustic
        )
        
        # 合并两个阶段的指标
        full_quantized_metrics = semantic_result.metrics
        for key, value in acoustic_result.metrics.items():
            if key in full_quantized_metrics:
                full_quantized_metrics[key] = self._renorm_and_add(
                    full_quantized_metrics[key], value, n_q_semantic, n_q_acoustic
                )
            else:
                full_quantized_metrics[key] = value
                
        result = QuantizedResult(
            full_quantized_emb,
            full_quantized_codes,
            full_quantized_bandwidth,
            penalty=full_quantized_penalty,
            metrics=full_quantized_metrics,
        )

        return result, semantic_result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, T]
        codes: [B, K, T]
        """
        # 第一阶段：语义编码
        codes = self.rvq_first.encode(x)  # [B, n_q_semantic, T]
        
        # 第二阶段：声学编码（如果需要）
        if self.n_q > self.n_q_semantic:
            acoustic_codes = self.rvq_rest.encode(x)           # [B, n_q_acoustic, T]
            codes = torch.cat([codes, acoustic_codes], dim=1)  # [B, K, T]
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, K, T]
        quantized: [B, D, T]
        """
        # 解码语义部分
        quantized = self.rvq_first.decode(codes[:, :self.n_q_semantic])
        
        # 解码声学部分（如果存在）
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic:])
        return quantized

    @property
    def total_codebooks(self):
        """返回两个阶段最大可用的码本总数"""
        return self.rvq_first.max_n_q + self.rvq_rest.max_n_q

    @property
    def num_codebooks(self):
        """返回当前使用的码本总数"""
        return self.rvq_first.num_codebooks + self.rvq_rest.num_codebooks

    @property
    def n_q(self):
        """返回当前使用的量化器总数"""
        return self.rvq_first.n_q + self.rvq_rest.n_q

    @property
    def dimension(self):
        """返回量化器的维度"""
        return self.rvq_first.dimension

    @property
    def semantic_quantizer(self) -> ResidualVectorQuantizer:
        """返回语义量化器（第一阶段）"""
        return self.rvq_first

    @property
    def acoustic_quantizer(self) -> ResidualVectorQuantizer:
        """返回声学量化器（第二阶段）"""
        return self.rvq_rest

    def set_num_codebooks(self, n: int):
        """设置要使用的码本总数
        
        参数:
            n (int): 要使用的码本总数，必须大于等于语义量化器数量且不超过最大码本数
        """
        assert n >= self.n_q_semantic and n <= self.total_codebooks
        self.rvq_rest.set_num_codebooks(n - self.n_q_semantic)

    @property
    def cardinality(self) -> int:
        """返回每个码本的大小
        
        注意：两个阶段的码本大小必须相同
        """
        assert self.rvq_rest.cardinality == self.rvq_first.cardinality
        return self.rvq_first.cardinality


if __name__ == "__main__":
    _quantizer_kwargs = {
        "dimension": 256,
        "n_q": 32,
        "bins": 2048,
        "input_dimension": 512,
        "output_dimension": 512
    }
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    print(quantizer.rvq_first.vq.layers[0]._codebook.initialized)
    
    inputs = torch.randn(2, 512, 25)

    # codes = quantizer.encode(inputs)
    # quantized = quantizer.decode(codes)
    # print(codes.shape, quantized.shape)
    # print(quantizer.rvq_first.vq.layers[0]._codebook.initialized)
    # print(quantizer.rvq_first.vq.layers[0]._codebook.training)

    quant_result, semantic_result = quantizer(inputs, frame_rate=12.5)
    # print(quantizer.rvq_first.vq.layers[0]._codebook.initialized)
    # print(quant_result.metrics.keys())
    # quant_result = quantizer(inputs, frame_rate=12.5)
    # quant_result = quantizer(inputs, frame_rate=12.5)
    # quant_result = quantizer(inputs, frame_rate=12.5)
    # print(quant_result.metrics.keys())
    print(semantic_result.x.shape)