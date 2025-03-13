import soundfile as sf
import torch
from pydub import AudioSegment
import os
import numpy as np

def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
):
    """
    高效地加载音频文件并进行重采样
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率
        mono: 是否转换为单通道
        
    Returns:
        audio_data: 音频数据, 范围为 [-1, 1]
        sr: 采样率
    """
    # 获取文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.wav':
            # 使用 soundfile 加载 wav 文件
            audio_data, sr = sf.read(file_path)
            
            # 转换为单通道
            if mono and len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 重采样
            if sr != target_sr:
                # 使用 librosa 进行重采样，kaiser_best 提供最好的音质
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sr,
                    target_sr=target_sr,
                    res_type='kaiser_best'
                )
                sr = target_sr
                
        elif ext == '.m4a':
            # 使用 pydub 加载 m4a
            audio = AudioSegment.from_file(file_path, format="m4a")
            
            # 转换采样率
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            
            # 转换为单通道
            if mono and audio.channels > 1:
                audio = audio.set_channels(1)
            
            # 转换为 numpy array，并归一化到 [-1, 1]
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / np.iinfo(np.int16).max
            sr = target_sr
            
        else:
            # 对于其他格式，使用 librosa 加载
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_data, sr = librosa.load(
                    file_path,
                    sr=target_sr if target_sr else None,
                    mono=mono
                )
        
        # 确保数据类型是 float32
        audio_data = audio_data.astype(np.float32)
        
        # 确保音频幅度在 [-1, 1] 范围内
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        return audio_data
        
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {file_path}: {str(e)}")
