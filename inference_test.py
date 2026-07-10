import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from pathlib import Path
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.utils.file import audio_to_bytes

# Configuration
device = "cpu" # Force CPU due to MPS limitations
print(f"Using device: {device}")

llama_checkpoint_path = "checkpoints/fish-speech-1.5"
decoder_checkpoint_path = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
decoder_config_name = "firefly_gan_vq"

reference_audio_path = "/Users/taku.me/Library/CloudStorage/Dropbox/_inbox/AppData/Remotion/ai-news-podcast/remotion-project/public/audio/opening.mp3"
reference_text = "こんにちは、今日もいい天気ですね。" # Dummy text since we don't know the exact content
target_text = "こんにちは、これはフィッシュスピーチのテスト生成です。うまく聞こえていますか？"

def main():
    print("Loading models...")
    
    # Load Llama
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=llama_checkpoint_path,
        device=device,
        precision=torch.half,
        compile=False,
    )
    
    # Load VQ-GAN
    decoder_model = load_decoder_model(
        config_name=decoder_config_name,
        checkpoint_path=decoder_checkpoint_path,
        device=device,
    )
    
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=False,
        precision=torch.half,
    )
    
    print("Models loaded. Preparing request...")
    
    # Read reference audio
    with open(reference_audio_path, "rb") as f:
        audio_content = f.read()

    request = ServeTTSRequest(
        text=target_text,
        references=[
            ServeReferenceAudio(
                audio=audio_content,
                text=reference_text
            )
        ],
        reference_id=None,
        max_new_tokens=1024,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        format="wav",
    )

    print("Running inference...")
    
    # Run inference
    generator = inference_engine.inference(request)
    
    output_audio = b""
    for chunk in generator:
        if isinstance(chunk, bytes): # Wave header or audio data
             output_audio += chunk
        elif hasattr(chunk, "code") and chunk.code == "header":
             pass # Headers handling

    # Save output
    output_path = "output.wav"
    with open(output_path, "wb") as f:
        f.write(output_audio)
        
    print(f"Inference complete! Saved to {output_path}")

if __name__ == "__main__":
    main()
