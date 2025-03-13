import random
import json
from pathlib import Path
import av
from tqdm import tqdm
import numpy as np

def get_audio_duration(audio_path):
    """获取音频时长"""
    try:
        with av.open(str(audio_path)) as container:
            stream = container.streams.audio[0]
            duration = float(stream.duration * stream.time_base)
            sample_rate = stream.sample_rate
            return duration, sample_rate
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def prepare_metadata(root_dir: str, 
                     train_file: str, 
                     valid_file: str, 
                     valid_ratio: int = 0.02, 
                     split_seed: int = 42,
                     category: str = "Premium"):
    """准备数据集的metadata文件"""
    root_path = Path(root_dir)
    all_metas = []
    
    # 遍历所有切片文件夹
    for _dir in sorted(root_path.glob(f"WenetSpeech4TTS_{category}_*")):
        wav_dir = _dir / "wavs"
        if not wav_dir.exists():
            continue
            
        # 处理当前切片下的所有音频
        print(f"Processing {_dir.name}...")
        for wav_path in tqdm(list(wav_dir.glob("*.wav"))):
            duration, sample_rate = get_audio_duration(wav_path)
            if duration is None:
                continue
                
            meta = {
                "path": str(wav_path.resolve()),
                "duration": duration,
                "sample_rate": sample_rate,
            }
            all_metas.append(meta)
    
    # 划分数据集
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(len(all_metas))
    all_metas = [all_metas[i] for i in indices]

    split_idx = int(len(all_metas) * (1 - valid_ratio))
    train_meta = all_metas[:split_idx]
    valid_meta = all_metas[split_idx:]

    # 保存metadata
    print(f"Total files: {len(all_metas)}; Training Set: {len(train_meta)}; Validation Set: {len(valid_meta)}")
    with open(train_file, "w", encoding="utf-8") as f:
        for meta in train_meta:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    with open(valid_file, "w", encoding="utf-8") as f:
        for meta in valid_meta:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    category = "Premium"
    root_dir = f"data/WenetSpeech4TTS/{category}"
    train_file = f"data/wenetspeech4tts_{category}_train.jsonl"
    valid_file = f"data/wenetspeech4tts_{category}_valid.jsonl"
    prepare_metadata(root_dir, train_file, valid_file, valid_ratio=0.01, category=category)