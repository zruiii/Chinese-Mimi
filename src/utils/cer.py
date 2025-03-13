import torch
from omegaconf import OmegaConf
from src.models import MimiModel
from src.modules import (
    SEANetEncoder, 
    SEANetDecoder, 
    ProjectedTransformer, 
    SplitResidualVectorQuantizer
)

from collections import OrderedDict

import whisperx
import numpy as np

def load_mimi(epoch, model_id, use_ema=True, device='cuda'):
    checkpoint_path = f'outputs/save/{model_id}/checkpoint_epoch{epoch}.pt'

    cfg = OmegaConf.load(f"outputs/logs/{model_id}/config.yaml")
    sample_rate = cfg.sample_rate
    frame_rate = cfg.frame_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建各个组件
    encoder = SEANetEncoder(**cfg.seanet)
    decoder = SEANetDecoder(**cfg.seanet)
    
    encoder_transformer = ProjectedTransformer(
        device=device,
        **cfg.transformer
    )
    decoder_transformer = ProjectedTransformer(
        device=device,
        **cfg.transformer
    )
    
    quantizer = SplitResidualVectorQuantizer(**cfg.quantizer)
    
    # 构建完整模型
    model = MimiModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        channels=1,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        encoder_frame_rate=sample_rate / encoder.hop_length,
        causal=1,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer
    ).to(device)

    """加载模型和EMA状态"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果有EMA状态，优先使用EMA状态
    if use_ema:
        print("Using EMA state")
        state_dict = OrderedDict()
        for name, param in checkpoint['model'].items():
            name = name.replace('module.', '')              # 移除DDP的'module.'前缀
            if name in checkpoint['ema']['state']['model']:
                param = checkpoint['ema']['state']['model'][name].detach().clone()
            else:
                param = param.detach().clone()
            state_dict[name] = param
    else:
        print("Using original model state")
        state_dict = OrderedDict(
            (k.replace('module.', ''), v.detach().clone())
            for k, v in checkpoint['model'].items()
        )
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_wer(ref: str, hyp: str) -> float:
    """计算词错误率 (Word Error Rate)
    
    Args:
        ref: 参考文本
        hyp: 待评估文本
        
    Returns:
        float: 词错误率,范围[0,1]
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # 计算编辑距离矩阵
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    d[:, 0] = np.arange(len(ref_words) + 1)
    d[0, :] = np.arange(len(hyp_words) + 1)
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
    
    return float(d[-1, -1]) / len(ref_words)

if __name__ == "__main__":
    import json
    device = 'cuda'

    # Basic configuration
    config = {
        'mimi_epoch': "45",
        'mimi_id': "20241230_214059",
        'save_dir': "data/WenetSpeech4TTS-test/",
        'test_file': "data/wenetspeech4tts_Standard_valid_part_25.jsonl"
    }

    """
    model = load_mimi(config['mimi_epoch'], config['mimi_id'], device=device)

    for line in open(config['test_file']):
        sample = json.loads(line.strip())
        file_path = sample['path']

        wav, sr = sf.read(file_path)
        wav_tensor = torch.from_numpy(wav).float()
        wav_tensor = f32_pcm(wav_tensor)                # 归一化
        wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)
        wav_tensor = wav_tensor.to(device)

        codes = model.encode(wav_tensor)
        recon = model.decode(codes)

        recon_np = recon.squeeze().cpu().numpy()
        import pdb; pdb.set_trace()
        sf.write(f"{config['save_dir']}/{file_path.split('/')[-1].split('.')[0]}.wav", recon_np, sr)
    """

    # 计算 WER
    model = whisperx.load_model(
        "models/faster-whisper-large-v3",
        "cuda",
        language="zh",
        compute_type="bf16"
    )
    
    for line in open(config['test_file']):
        sample = json.loads(line.strip())
        ref_file = sample['path']
        gen_file = f"{config['save_dir']}/{ref_file.split('/')[-1].split('.')[0]}.wav"

        # 加载音频
        ref_audio = whisperx.load_audio(ref_file)
        gen_audio = whisperx.load_audio(gen_file)
        
        # 转录
        ref_text = model.transcribe(ref_audio, batch_size=32)
        gen_text = model.transcribe(gen_audio, batch_size=32)

        import pdb; pdb.set_trace()



    