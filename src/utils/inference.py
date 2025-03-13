import os
import torch
from omegaconf import OmegaConf
from src.models import MimiModel
from src.modules import (
    SEANetEncoder, 
    SEANetDecoder, 
    ProjectedTransformer, 
    SplitResidualVectorQuantizer
)

import soundfile as sf
from collections import OrderedDict
from src.utils.helper import f32_pcm

def load_model(model, checkpoint_path, use_ema=True, device='cuda'):
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
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='Model Save Epoch.')
    parser.add_argument('--model-id', type=str, help='Model Version.')
    parser.add_argument('--test-dir', type=str, default="data/WenetSpeech4TTS/test", help='File Path for Test Files.')
    parser.add_argument('--save-dir', type=str, default="tmp", help='Save Path for Test Files.')
    args = parser.parse_args()


    # ********** 加载 Mimi **********
    epoch = args.epoch
    model_id = args.model_id
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

    model = load_model(model, checkpoint_path, use_ema=True, device=device)
    model.eval()

    # ********** 测试重构效果 **********
    save_dir = f"{args.save_dir}/{model_id}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        root_path = args.test_dir
        for file in os.listdir(root_path):
            file_path = os.path.join(root_path, file)
            wav, sr = sf.read(file_path)
            wav_tensor = torch.from_numpy(wav).float()
            wav_tensor = f32_pcm(wav_tensor)                # 归一化
            wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)
            wav_tensor = wav_tensor.to(device)

            codes = model.encode(wav_tensor)
            recon = model.decode(codes)

            recon_np = recon.squeeze().cpu().numpy()
            sf.write(f'{save_dir}/{file.split(".")[0]}_ep{epoch}.wav', recon_np, sr)
