import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from collections import OrderedDict

from src.models import MimiModel
from src.modules import (
    SEANetEncoder,
    SEANetDecoder,
    ProjectedTransformer,
    SplitResidualVectorQuantizer
)
from src.utils.helper import f32_pcm

import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mimi():
    _seanet_kwargs = {
        "channels": 1,
        "dimension": 512,
        "causal": True,
        "n_filters": 64,
        "n_residual_layers": 1,
        "activation": "ELU",
        "compress": 2,
        "dilation_base": 2,
        "disable_norm_outer_blocks": 0,
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        "norm": "none",
        "pad_mode": "constant",
        "ratios": [8, 5, 4, 2],
        "true_skip": True,
    }
    _quantizer_kwargs = {
        "dimension": 256,
        "n_q": 32,
        "bins": 2048,
        "input_dimension": _seanet_kwargs["dimension"],
        "output_dimension": _seanet_kwargs["dimension"],
    }
    _transformer_kwargs = {
        "d_model": _seanet_kwargs["dimension"],
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
        "input_dimension": _seanet_kwargs["dimension"],
        "output_dimensions": [_seanet_kwargs["dimension"]],
    }
    SAMPLE_RATE = 16000
    FRAME_RATE = 12.5

    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device)

    return model

def create_heatmap(codes):
    """Create a heatmap visualization of the codes"""
    plt.figure(figsize=(10, 4))
    plt.imshow(codes, aspect='auto', cmap='viridis')
    plt.colorbar(label='Code Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Code Dimensions')
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

class AudioCodecDemo:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.current_codes = None
        self.SAMPLE_RATE = 16000

    def normalize_audio(self, audio_data):
        """Convert PCM integers to float32 [-1, 1] range"""
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 2**15
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2**31
        
        return audio_data
        
    def process_audio(self, audio):
        """Convert audio to tensor with proper preprocessing"""
        if audio is None:
            return None
            
        sample_rate, data = audio
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        data = self.normalize_audio(data)
        
        # Resample if necessary
        if sample_rate != self.SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=self.SAMPLE_RATE)

        data = torch.from_numpy(data).float()
        
        # Convert to tensor
        audio_tensor = data.unsqueeze(0).unsqueeze(0)
        
        # Move to correct device
        audio_tensor = audio_tensor.to(self.device)
        
        return audio_tensor
        
    def encode_audio(self, audio):
        """Encode audio to discrete codes and visualize"""
        if audio is None:
            return None, "Please record audio first"
            
        audio_tensor = self.process_audio(audio)
        
        # Encode
        with torch.no_grad():
            codes = self.model.encode(audio_tensor)
        
        # Store codes on CPU
        self.current_codes = codes.cpu()
        
        # Remove batch dimension and convert to numpy for visualization
        codes_np = codes.squeeze(0).cpu().numpy()
        print(codes_np.shape)
        print(codes_np)

        # Create visualization
        heatmap = create_heatmap(codes_np)
        
        return heatmap, "Encoding successful"
        
    def decode_audio(self):
        """Decode current codes back to audio"""
        if self.current_codes is None:
            return None, "Please encode audio first"
        
        # Move codes to correct device
        codes = self.current_codes.to(self.device)
        
        # Decode
        with torch.no_grad():
            audio = self.model.decode(codes)
            
        # Convert to numpy and ensure correct format for gradio
        audio = audio.squeeze().cpu().numpy()
        
        # Ensure audio is normalized
        # audio = np.clip(audio, -1.0, 1.0)
        
        return (self.SAMPLE_RATE, audio), "Decoding successful"

def create_interface(model):
    demo = AudioCodecDemo(model)
    
    with gr.Blocks() as interface:
        gr.Markdown("# Neural Audio Codec Demo")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label=f"Input Audio (Target: {demo.SAMPLE_RATE}Hz)",
                    type="numpy"
                )
                
                encode_btn = gr.Button("Encode Audio")
                decode_btn = gr.Button("Decode Audio")
                
            with gr.Column():
                code_viz = gr.Image(
                    label="Code Visualization"
                )
                
                status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                audio_output = gr.Audio(
                    label="Reconstructed Audio",
                    type="numpy"
                )
        
        encode_btn.click(
            fn=demo.encode_audio,
            inputs=[audio_input],
            outputs=[code_viz, status]
        )
        
        decode_btn.click(
            fn=demo.decode_audio,
            inputs=[],
            outputs=[audio_output, status]
        )
        
        gr.Markdown("""
        ## Instructions
        1. Click the microphone icon to record audio
        2. Click 'Encode Audio' to convert to discrete codes
        3. The visualization shows the encoded representation
        4. Click 'Decode Audio' to reconstruct the original audio
        5. Click the play button to hear the reconstructed audio
        """)

    return interface

def load_model(model, checkpoint_path, use_ema=True, device='cuda'):
    """加载模型和EMA状态"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果有EMA状态，优先使用EMA状态
    if use_ema:
        print("Using EMA state")
        state_dict = OrderedDict()
        for name, param in checkpoint['model'].items():
            name = name.replace('module.', '')  # 移除DDP的'module.'前缀
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
    epoch = 31
    checkpoint_path = f'outputs/save/20241204_133147/checkpoint_epoch{epoch}.pt'
    model = get_mimi()
    model = load_model(model, checkpoint_path, use_ema=True, device=device)
    model.eval()

    interface = create_interface(model)  # Pass your model here
    interface.launch(share=True)