import argparse
import os
import torch
from pathlib import Path

# LLM inference (text2semantic) runs on MPS for speed.
# VQ-GAN decoder stays on CPU: MPS has a Conv1d >65536 channel limit that breaks it.
if torch.backends.mps.is_available():
    llm_device = "mps"
elif torch.cuda.is_available():
    llm_device = "cuda"
else:
    llm_device = "cpu"

decoder_device = "cpu"
device = llm_device  # for display

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

def preprocess_japanese_text(text):
    import re
    text = re.sub(r'」([^。！？、，\s」])', r'」。\1', text)
    if text and text[-1] not in '。！？…」':
        text += '。'
    return text

# Default paths
DEFAULT_REF_AUDIO = None # We will use the specific reference ID if provided
FISH_SPEECH_VOICE_ID = "5161d41404314212af1254556477c17d"
# Dummy reference text - for better quality, this should match the content of the audio
DEFAULT_REF_TEXT = "2026年2月12日のAIニュースポッドキャストをお届けします。今日は、あなたのために厳選した6件のニュースをご紹介します。それでは、始めましょう。" 

def main():
    parser = argparse.ArgumentParser(description="Generate speech using Fish Speech 1.5")
    parser.add_argument("text", type=str, help="Text to generate speech for")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output wav filename")
    parser.add_argument("--ref_audio", type=str, default=DEFAULT_REF_AUDIO, help="Path to reference audio file")
    parser.add_argument("--ref_text", type=str, default=DEFAULT_REF_TEXT, help="Transcript of the reference audio")
    parser.add_argument("--ref_id", type=str, default=FISH_SPEECH_VOICE_ID, help="Fish Audio Voice ID")
    parser.add_argument("--chunk_length", type=int, default=200, help="Chunk length in UTF-8 bytes (default: 200)")
    parser.add_argument("--max_new_tokens", type=int, default=0, help="Max new tokens per chunk (0=auto, default: 0)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p sampling (default: 0.7)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"Input Text: {args.text}")
    print(f"Reference Audio: {args.ref_audio}")
    print(f"Output File: {args.output}")

    # Load Models
    print("Loading models... (This may take a moment)")
    
    llama_checkpoint_path = "checkpoints/fish-speech-1.5"
    decoder_checkpoint_path = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    decoder_config_name = "firefly_gan_vq"

    # Load Llama (MPS for fast inference)
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=llama_checkpoint_path,
        device=llm_device,
        precision=torch.bfloat16,
        compile=False,
    )

    # Load VQ-GAN decoder (CPU: MPS has a Conv1d >65536ch limitation)
    decoder_model = load_decoder_model(
        config_name=decoder_config_name,
        checkpoint_path=decoder_checkpoint_path,
        device=decoder_device,
    )
    
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=False,
        precision=torch.bfloat16,
        llm_device=llm_device,
    )
    
    
    # Check if we should use reference ID or local audio file
    references = []
    reference_id = None
    
    if args.ref_audio and os.path.exists(args.ref_audio):
        with open(args.ref_audio, "rb") as f:
            audio_content = f.read()
        references = [
            ServeReferenceAudio(
                audio=audio_content,
                text=args.ref_text
            )
        ]
    elif args.ref_id:
         # Use the provided Reference ID (from Fish Audio API, but mapped locally if model is available)
         # Note: ServeTTSRequest in local inference typically expects 'references' list with audio content.
         # If 'reference_id' is intended for a stored latent/embedding that the local model knows, 
         # we need to be sure the local model supports looking it up. 
         # However, based on the user request, they want to use the "model" used in the API.
         # If this ID corresponds to a specific speaker profile, we might need to load that specific checkpoint or Speaker ID.
         # For standard Fish Speech local inference, we usually need the audio file.
         # If the user implies using the SAME VOICE as the API, we need the original audio that created that voice ID.
         # Since we don't have the internal database of Fish Audio, we ideally need the source audio file.
         # BUT, if the user points to `opening.mp3` earlier, THAT IS the source of the voice.
         
         # Let's fallback to the opening.mp3 if no ref_audio is given, but printed a message about the ID.
         print(f"Using default Voice ID: {args.ref_id}")
         # We will use the known opening.mp3 as the reference for this ID since we have it locally.
         default_audio_path = "/Users/taku.me/Library/CloudStorage/Dropbox/_inbox/AppData/Remotion/ai-news-podcast/remotion-project/public/audio/opening.mp3"
         if os.path.exists(default_audio_path):
             with open(default_audio_path, "rb") as f:
                audio_content = f.read()
             references = [
                ServeReferenceAudio(
                    audio=audio_content,
                    text=DEFAULT_REF_TEXT
                )
             ]
         else:
             print("Warning: Default reference audio file not found. Generation might be random.")

    # Pre-process Japanese text for better sentence splitting
    processed_text = preprocess_japanese_text(args.text)
    if processed_text != args.text:
        print(f"Preprocessed Text: {processed_text}")

    # Create Request
    request = ServeTTSRequest(
        text=processed_text,
        references=references,
        reference_id=None, # Local inference usually doesn't use this the same way as the API
        max_new_tokens=args.max_new_tokens,
        chunk_length=args.chunk_length,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        seed=args.seed,
        format="wav",
    )

    import soundfile as sf
    import numpy as np

    print("Generating speech...")
    
    # Run inference
    generator = inference_engine.inference(request)
    
    final_audio = None
    sample_rate = 44100 # Default fallback

    for chunk in generator:
        if chunk.error:
            print(f"Error during generation: {chunk.error}")
            return
            
        if chunk.code == "final":
            sample_rate, final_audio = chunk.audio
            break
            
    if final_audio is None:
         print("No audio generated.")
         return

    # Save output
    sf.write(args.output, final_audio, sample_rate)
        
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
