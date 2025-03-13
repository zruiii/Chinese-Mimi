import torch
from tqdm import tqdm
import json
from pathlib import Path
import traceback
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from .audio_dataset import MimiAudioMeta
import torch.multiprocessing as mp
from typing import List

def process_batch(
    rank: int,
    world_size: int,
    meta_list: List[MimiAudioMeta],
    hubert_path: str,
    save_dir: str,
    last_hidden: bool = False
):
    """单个GPU上的处理函数"""
    # 设置设备
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    
    # 初始化模型
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
    model = HubertModel.from_pretrained(hubert_path)
    model = model.to(device).eval()

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 计算当前进程要处理的数据范围
    chunk_size = len(meta_list) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else len(meta_list)
    process_meta_list = meta_list[start_idx:end_idx]

    # 处理分配到的音频文件
    for meta in tqdm(process_meta_list, desc=f"GPU {rank}"):
        feat_path = save_dir / f"{Path(meta.path).stem}.hubert.pt"
        if feat_path.exists():
            print(f"[GPU {rank}] {feat_path} exists, skipping...")
            continue
            
        try:
            # 读取音频
            wav, sr = sf.read(meta.path)
            
            # 提取特征
            with torch.no_grad():
                input_values = feature_extractor(wav, sampling_rate=sr, return_tensors="pt").input_values
                input_values = input_values.to(device)
                outputs = model(input_values, output_hidden_states=True)
                if last_hidden:
                    features = outputs.last_hidden_state
                else:
                    hidden_states = outputs.hidden_states
                    features = torch.mean(torch.stack(hidden_states), dim=0)
                
                features = features.to(torch.bfloat16)  
                features = features.cpu()
                features = features.squeeze(0)
            
            # 保存特征
            torch.save(features, feat_path)
            
        except Exception as e:
            error_msg = f"""
            [GPU {rank}] Error processing file: {meta.path}
            Error type: {type(e).__name__}
            Error message: {str(e)}
            Stack trace:
            {traceback.format_exc()}
            -----------------
            """
            print(error_msg)

def extract_hubert_features_multi_gpu(
    meta_path: str,
    hubert_path: str,
    save_dir: str,
    num_gpus: int = None,
    last_hidden: bool = False,
):
    """多GPU并行提取HuBERT特征"""
    # 确定可用的GPU数量
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    print(f"Using {num_gpus} GPUs")

    # 加载meta信息
    meta_list = []
    for line in open(meta_path, "rb"):
        meta = json.loads(line.strip())
        meta_list.append(MimiAudioMeta.from_dict(meta))
    
    print(f"Total files to process: {len(meta_list)}")

    # 启动多进程
    mp.spawn(
        process_batch,
        args=(num_gpus, meta_list, hubert_path, save_dir, last_hidden),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract HuBERT features from audio files using multiple GPUs')
    parser.add_argument('--meta_path', type=str, default="data/wenetspeech4tts_premium_valid.jsonl", 
                      help='Path to the metadata JSONL file')
    parser.add_argument('--hubert_path', type=str, default="models/chinese-hubert-large", 
                      help='Path to the HuBERT model directory')
    parser.add_argument('--save_dir', type=str, default="processed_data/WenetSpeech4TTS/Premium", 
                      help='Directory to save extracted features')
    parser.add_argument('--num_gpus', type=int, default=4,
                      help='Number of GPUs to use (default: all available GPUs)')
    args = parser.parse_args()

    extract_hubert_features_multi_gpu(
        args.meta_path,
        args.hubert_path,
        args.save_dir,
        args.num_gpus
    )