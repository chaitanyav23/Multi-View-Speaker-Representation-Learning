import subprocess
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Phase 2: DeepFilterNet Enhancement (Module)")
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--output-dir", default="experiments/phase2_enhancement/audio_deepfilternet")
    parser.add_argument("--device", default=None, help="Force device (e.g. cpu)")
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

    for audio_path in audio_files:
        print(f"Enhancing: {audio_path.name}")
        cmd = ["python3", "-m", "df.enhance", str(audio_path), "--output-dir", str(output_dir)]
        if args.device:
            cmd.extend(["--device", args.device])
            
        subprocess.run(cmd, check=True)
        
        generated_path = output_dir / f"{audio_path.stem}_DeepFilterNet3.wav"
        if generated_path.exists():
            os.replace(generated_path, output_dir / audio_path.name)

if __name__ == "__main__":
    main()
