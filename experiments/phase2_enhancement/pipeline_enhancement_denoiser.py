import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path.cwd()))
from scripts.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Facebook Denoiser Enhancement")
    parser.add_argument("--audio-dir", default="raw_audio")
    parser.add_argument("--output-dir", default="experiments/phase2_enhancement/audio_denoiser")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = pretrained.dns64().to(device)
    model.eval()

    for audio_path in audio_dir.glob("*.wav"):
        print(f"Enhancing: {audio_path.name}")
        wav, sr = torchaudio.load(str(audio_path))
        wav = wav.to(device)
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)
        
        with torch.no_grad():
            enhanced = model(wav[None])[0]
            
        torchaudio.save(str(output_dir / audio_path.name), enhanced.cpu(), model.sample_rate)

if __name__ == "__main__":
    main()
