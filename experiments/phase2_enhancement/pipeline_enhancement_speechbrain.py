import torch
import torchaudio
from speechbrain.inference.enhancement import SpectralMaskEnhancement
import argparse
from pathlib import Path
import sys

# Add project root to path (works regardless of current working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="Phase 2: SpeechBrain Enhancement")
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--output-dir", default="experiments/phase2_enhancement/audio_speechbrain")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = []
    if args.file:
        audio_files.append(Path(args.file))
    elif args.audio_dir:
        audio_files = list(Path(args.audio_dir).glob("*.wav"))
    else:
        audio_files = list(Path("raw_audio").glob("*.wav"))

    device = get_device()
    device_str = str(device)
    
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank", 
        run_opts={"device": device_str}
    )

    for audio_path in audio_files:
        print(f"Enhancing: {audio_path.name}")
        noisy = enhance_model.load_audio(str(audio_path)).unsqueeze(0).to(device_str)
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]).to(device_str))
        
        torchaudio.save(str(output_dir / audio_path.name), enhanced.cpu(), 16000)

if __name__ == "__main__":
    main()
