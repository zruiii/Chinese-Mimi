import av
import time
import contextlib
import json
import torch
import torch.nn as nn
import typing as tp
import julius

from torch import distributed
from transformers import AutoTokenizer

# ************************ 计时 ************************
class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.st_time = None

    def __enter__(self):
        self.st_time = time.perf_counter_ns()

    def __exit__(self, *etc):
        self.et = time.perf_counter_ns() - self.st_time
        print(f"{self.prefix} {self.et*1e-6:.2f} ms")


# ************************ 数据加载 ************************
def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def audio_read(path, seek_time: float = 0, duration: float = -1.):
    """ 读取音频数据
    原始音频文件
    ↓
    解码 (PyAV)
    ↓ 
    PCM数据 (可能是不同的数值范围)
    ↓
    f32_pcm归一化 (统一到 [-1.0, 1.0])
    """
    # TODO: 添加日志
    with av.open(path) as audio_file:
        # 获取音频流
        stream = audio_file.streams.audio[0]
        sample_rate = stream.sample_rate

        # 计算帧参数
        target_frames = int(sample_rate * duration) if duration >= 0 else -1
        start_frame = int(sample_rate * seek_time)
        
        # 定位到指定时间点 (略微提前避免解码伪影) | stream.time_base = 1 / sample_rate
        seek_point = max(0, seek_time - 0.1)
        audio_file.seek(int(seek_point / stream.time_base), stream=stream)
        
        # 读取音频
        frames = []
        total_samples = 0
        for frame in audio_file.decode(streams=stream.index):
            current_pos = int(frame.rate * frame.pts * frame.time_base)         # 当前帧的位置
            strip_samples = max(0, start_frame - current_pos)                   # 0.1s对应的偏移量
            # import pdb; pdb.set_trace()

            # 处理单个帧
            audio_chunk = torch.from_numpy(frame.to_ndarray())      # [channels, samples]
            if audio_chunk.shape[0] != stream.channels:
                audio_chunk = audio_chunk.view(-1, stream.channels).t()     # [1, T]
            audio_chunk = audio_chunk[:, strip_samples:]
            frames.append(audio_chunk)
            total_samples += audio_chunk.shape[1]

            if target_frames > 0 and total_samples >= target_frames:
                break
        
        audio_data = torch.cat(frames, dim=1)
        if target_frames > 0:
            audio_data = audio_data[:, :target_frames]
        
        return f32_pcm(audio_data), sample_rate


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    # 重采样至目标频率
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))    

    # 转换到目标通道数
    C, T = wav.shape
    if C == to_channels:
        return wav
    elif C >= to_channels:
        if to_channels == 1:
            return wav.mean(dim=-2, keepdim=True)
        return wav[:to_channels, :]
    else:
        if C == 1:
            return wav.expand(to_channels, T)
        else:
            raise ValueError('The audio file has less channels than requested but is not mono.')

def align_tokens(trans_file: str, seek_time: float, 
                 segment_duration: float, frame_rate: float, text_tokenizer: AutoTokenizer):
    
    with open(trans_file, "r") as f:
        transcription = json.loads(f.read())

    audio_tokens_length = int(segment_duration * frame_rate)
    text_tokens = torch.full((audio_tokens_length, ), fill_value=text_tokenizer.tpad_token_id)

    for chunk in transcription["chunks"]:
        start_frame = -1
        start_time, end_time = chunk["timestamp"]

        if start_time < seek_time and end_time > seek_time:
            start_frame = 0
        elif start_time >= seek_time and start_time < seek_time + segment_duration:
            start_frame = int((start_time - seek_time) * frame_rate)

        if start_frame == -1:
            continue

        if start_frame > 0 and text_tokens[start_frame-1] == text_tokenizer.tpad_token_id:
            text_tokens[start_frame-1] = text_tokenizer.epad_token_id

        # NOTE: Text Tokenizer 可能和 Whisper Tokenizer 的词表不一致
        word_tokens = text_tokenizer.encode(chunk['text'], add_special_tokens=False)
        for j, token in enumerate(word_tokens):
            if start_frame + j < audio_tokens_length:
                text_tokens[start_frame + j] = token
    
    return text_tokens

# ************************ 模型加载 ************************
def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]

def get_activation(name: str):
    """获取激活函数
    
    Args:
        name: 激活函数名称
    Returns:
        对应的激活函数
    """
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish"]:
        return getattr(torch.nn.functional, name)
    elif name == "identity":
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation {name}")

class LayerNormF32(nn.LayerNorm):
    """Float32精度的LayerNorm
    可以在保持输入类型的同时，使用float32进行中间计算，提高数值稳定性
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 转换为float32进行计算
        x_f32 = input.float()
        # 使用父类的forward
        out_f32 = super().forward(x_f32)
        # 转换回输入类型
        return out_f32.to(input.dtype)

class RMSNorm(nn.Module):
    """RMSNorm归一化层
    使用均方根进行归一化，计算更快，且在某些情况下效果更好
    
    Args:
        dim: 特征维度
        eps: 数值稳定性参数
        dtype: 计算精度
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: tp.Optional[torch.dtype] = None,
        device = None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        # 可学习的缩放参数
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果指定了dtype，先转换类型
        x_dtype = x.dtype
        if self.dtype is not None:
            x = x.to(self.dtype)
        
        # 计算均方根
        var = torch.mean(x**2, dim=2, keepdim=True)
        # 归一化并应用缩放
        y = x * (self.alpha.to(var) * torch.rsqrt(var + self.eps))
        # 转换回原始类型
        return y.to(x_dtype)

def create_norm_fn(
    norm_type: str,
    dim: int,
    **kwargs
) -> nn.Module:
    """创建归一化层的工厂函数
    
    Args:
        norm_type: 归一化类型
            - layer_norm: 标准LayerNorm
            - layer_norm_f32: Float32精度的LayerNorm
            - rms_norm: RMSNorm
            - rms_norm_f32: Float32精度的RMSNorm
        dim: 特征维度
        **kwargs: 额外参数
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)  # f32版本不需要dtype参数
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "rms_norm_f32":
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

# ************************ 训练 ************************
def is_distributed() -> bool:
    """ 检查是否处于分布式计算 """
    return distributed.is_initialized() and distributed.get_world_size() > 1

# ************************ 采样 ************************
# todo: 简化
def multinomial(
    input: torch.Tensor, num_samples: int, replacement=False, *, generator=None
):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """
    input_ = input.reshape(-1, input.shape[-1])
    # We should probably be able to remove this once the following PR has landed:
    # https://github.com/pytorch/pytorch/pull/134818/files
    # In the meantime, we specialize the case no-replacement, nsamples=1 so as not
    # to have a synchronization point.
    if replacement or num_samples != 1:
        output_ = torch.multinomial(
            input_,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
    else:
        q = torch.empty_like(input_).exponential_(1, generator=generator)
        q = input_ / q
        output_ = q.argmax(dim=-1, keepdim=True)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output

def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs, indices = torch.topk(probs, k, dim=-1)
    next_token = multinomial(probs, num_samples=1)
    next_token = indices.gather(-1, next_token)
    return next_token

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample_token(
    logits: torch.Tensor,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Given logits of shape [*, Card], returns a LongTensor of shape [*]."""
    # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
    if use_sampling and temp > 0.0:
        probs = torch.softmax(logits / temp, dim=-1)
        if top_p > 0.0:
            next_token = sample_top_p(probs, p=top_p)
        elif top_k > 0:
            next_token = sample_top_k(probs, k=top_k)
        else:
            next_token = multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    assert next_token.shape[-1] == 1
    return next_token[..., 0]


if __name__ == "__main__":
    path = "../../data/WenetSpeech4TTS/Premium/WenetSpeech4TTS_Premium_0/wavs/X0000000021_240514196_S00041.wav"
    audio_read(path, seek_time=0.5, duration=1.0)