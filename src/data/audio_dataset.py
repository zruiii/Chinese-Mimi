import json
import random
import typing as tp

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass, fields
from pathlib import Path

from src.utils.helper import audio_read, convert_audio

import logging
logger = logging.getLogger(__name__)

@dataclass
class MimiAudioMeta:
    path: str
    duration: float
    sample_rate: int

    @classmethod
    def from_dict(cls, dictionary: dict):
        """ 读取样本，使用 field 避免字段不对齐 """
        base = {
            field.name: dictionary[field.name]
            for field in fields(cls) if field.name in dictionary
        }
        return cls(**base)

    def to_dict(self):
        return {
            field.name: self.__getattribute__(field.name)
            for field in fields(self)
        }


class AudioDataset(Dataset):
    def __init__(
        self,
        meta: tp.List[MimiAudioMeta],
        embed_dir: str,                                     # HuBERT 抽取特征路径
        num_samples: tp.Optional[int] = None,
        segment_duration: float = None,
        min_segment_ratio: float = 0.5,                     # 最低截断音频时长占目标时长的百分比
        min_audio_duration: tp.Optional[float] = None,      # 最小音频时长
        max_audio_duration: tp.Optional[float] = None,      # 最大音频时长
        sample_rate: int = 16000,                           # 目标音频采样率
        channels: int = 1,                                  # 目标音频通道数
        shuffle: bool = True,                               # 数据集在每个epoch是否打散
        shuffle_seed: int = 0,                              # 随机化     
        max_retry_read: int = 10,                           # 截断音频最大尝试次数
        sample_on_duration: bool = True,                    # 根据音频时长采样
        downsample_rate: int = 320,                         # HuBERT 下采样倍率
    ) -> None:
        super().__init__()
        self.num_samples = num_samples or len(meta)
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.max_retry_read = max_retry_read
        self.sample_on_duration = sample_on_duration

        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        if min_audio_duration is None and segment_duration is not None:
            self.min_audio_duration = segment_duration * min_segment_ratio              # 确保能采样到最低目标长度的音频

        assert len(meta) > 0, "音频数据加载失败"
        self.meta = self._filter_on_duration(meta)
        self.current_epoch = None
        self.sampling_probabilities = self.cal_probs()

        self.embed_dir = Path(embed_dir)
        self.downsaple_rate = downsample_rate
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index: int) -> torch.Tensor:
        # 每个样本一次性处理
        if self.segment_duration is None:
            file_meta = self.meta[index]
            wav, sr = audio_read(file_meta.path)
            wav = convert_audio(wav, from_rate=sr, to_rate=self.sample_rate, to_channels=self.channels)

        # 从单个样本中采样音频子片段
        else:
            rng = torch.Generator()
            if self.shuffle:
                if self.current_epoch is None:
                    rng.manual_seed(index + self.num_samples * random.randint(0, 2**24))                    # 完全随机
                else:
                    rng.manual_seed(index + self.num_samples * (self.current_epoch + self.shuffle_seed))    # 可重复 | seed(epoch, index) 
            else:
                rng.manual_seed(index)
            
            for retry in range(self.max_retry_read):
                file_meta = self._sample_file(index, rng)
                max_seek = max(0, file_meta.duration - self.segment_duration * self.min_segment_ratio)
                seek_time = torch.rand(1, generator=rng).item() * max_seek

                try:
                    # 采样音频波形
                    wav, sr = audio_read(file_meta.path, seek_time=seek_time, duration=self.segment_duration)
                    wav = convert_audio(wav, from_rate=sr, to_rate=self.sample_rate, to_channels=self.channels)

                    # 读取 hubert 表征并进行对应的裁剪
                    # embed_path = self.embed_dir / f"{Path(file_meta.path).stem}.hubert.npy"
                    # embed = torch.from_numpy(np.load(embed_path))

                    embed_path = self.embed_dir / f"{Path(file_meta.path).stem}.hubert.pt"
                    embed = torch.load(embed_path)

                    embed_sr = self.sample_rate / self.downsaple_rate
                    start_embed = int(seek_time * embed_sr)
                    duration_embed = int(self.segment_duration * embed_sr)
                    embed = embed[start_embed: start_embed + duration_embed, :]

                except Exception as e:
                    logger.error(f"文件裁剪失败 {file_meta.path}: {e}")
                    if retry == self.max_retry_read - 1:
                        raise
                else:
                    break
        
        return wav, embed

    
    def collactor(self, samples: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """ 对齐一个batch中的样本长度 """
        wavs, embeds = zip(*samples)

        # 计算目标长度
        target_wav_len = int(self.segment_duration * self.sample_rate)                          # segment_duration 对应波形长度
        target_embed_len = int(self.segment_duration * self.sample_rate / self.downsaple_rate)  # segment_duration 对应 Hubert 帧数

        # 对齐波形到固定长度 [B, 1, T]
        wavs = [F.pad(wav, (0, target_wav_len - wav.shape[-1])) for wav in wavs]
        wav_batch = torch.stack(wavs)

        # 对齐特征到固定长度 [B, T, D]
        embeds = [F.pad(embed, (0, 0, 0, target_embed_len - embed.shape[0])) for embed in embeds]
        embed_batch = torch.stack(embeds)

        return {"wav": wav_batch, "embed": embed_batch}


    def _sample_file(self, index: int, rng: torch.Generator) -> MimiAudioMeta:
        if not self.sample_on_duration:
            file_index = int(torch.randint(len(self.meta), (1,), generator=rng).item())                     # 均匀采样
        else:
            file_index = int(torch.multinomial(self.sampling_probabilities, 1, generator=rng).item())       # 按权采样

        return self.meta[file_index]

    def _filter_on_duration(self, meta):
        if self.min_audio_duration is not None:
            meta = [m for m in meta if m.duration > self.min_audio_duration]
        if self.max_audio_duration is not None:
            meta = [m for m in meta if m.duration < self.max_audio_duration]
        
        return meta

    def cal_probs(self):
        scores = []
        for file_meta in self.meta:
            score = 1.
            if self.sample_on_duration:
                score *= file_meta.duration
            scores.append(score)

        probs = torch.tensor(scores)
        probs /= probs.sum()
        return probs
    
    @classmethod
    def from_meta(cls, root: str, **kwargs):
        meta = []
        for line in open(root, "rb"):
            d = json.loads(line.strip())
            m = MimiAudioMeta.from_dict(d)
            meta.append(m)
        return cls(meta=meta, **kwargs)
    
