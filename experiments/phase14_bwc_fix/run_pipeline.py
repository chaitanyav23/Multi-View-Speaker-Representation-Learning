import os
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n>>> Running {description}...")
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full conversational timeline pipeline.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--language", type=str, default="en", help="Language for transcription")
    parser.add_argument("--output_dir", type=str, default="experiments/phase14_bwc_fix", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing files")
    args = parser.parse_args()

    audio_in = Path(args.audio)
    output_dir = Path(args.output_dir)
    language = args.language
    
    # Create directory structure
    subdirs = ["audio", "grounding", "manifests", "outputs", "diarization"]
    for d in subdirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    # Filenames
    stem = audio_in.stem
    audio_repaired = output_dir / "audio" / f"{stem}_repaired.wav"
    audio_normalized = output_dir / "audio" / f"{stem}_normalized.wav"
    grounding_json = output_dir / "grounding" / f"{stem}_grounding.json"
    oracle_rttm = output_dir / "grounding" / f"{stem}_oracle.rttm"
    words_json = output_dir / "outputs" / f"{stem}_aligned_words.json"
    msdd_rttm = output_dir / "diarization" / f"{stem}_msdd.rttm"
    final_json = output_dir / "outputs" / f"{stem}_final.json"
    final_txt = output_dir / "outputs" / f"{stem}_final.txt"

    scripts_dir = Path(__file__).parent

    # [Step 0] Audio Repair (Downsample & Mono)
    if args.overwrite or not audio_repaired.exists():
        run_command([
            "ffmpeg", "-y", "-i", str(audio_in),
            "-ar", "16000", "-ac", "1",
            str(audio_repaired)
        ], "Step 0: Audio Repair (Downsample/Mono)")
    else:
        print("Skipping Step 0: Output already exists.")

    # [Step 1] Normalization (LoudNorm)
    if args.overwrite or not audio_normalized.exists():
        run_command([
            "ffmpeg", "-y", "-i", str(audio_repaired),
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            str(audio_normalized)
        ], "Step 1: Normalization (LoudNorm)")
    else:
        print("Skipping Step 1: Output already exists.")

    # [Step 2] VAD Grounding
    if args.overwrite or not grounding_json.exists():
        run_command([
            "python", str(scripts_dir / "step2_vad_grounding.py"),
            "--audio_path", str(audio_normalized),
            "--output_json", str(grounding_json)
        ], "Step 2: VAD Grounding")
    else:
        print("Skipping Step 2: Output already exists.")

    # [Step 3] Oracle RTTM
    if args.overwrite or not oracle_rttm.exists():
        run_command([
            "python", str(scripts_dir / "step3_oracle_rttm.py"),
            "--input_json", str(grounding_json),
            "--output_rttm", str(oracle_rttm)
        ], "Step 3: Oracle RTTM")
    else:
        print("Skipping Step 3: Output already exists.")

    # [Step 4] WhisperX ASR
    if args.overwrite or not words_json.exists():
        run_command([
            "python", str(scripts_dir / "step4_whisperx.py"),
            "--audio_path", str(audio_normalized),
            "--grounding_json", str(grounding_json),
            "--output_json", str(words_json),
            "--language", language
        ], "Step 4: WhisperX ASR")
    else:
        print("Skipping Step 4: Output already exists.")

    # [Step 5] MSDD Speaker Discovery
    if args.overwrite or not msdd_rttm.exists():
        run_command([
            "python", str(scripts_dir / "step5_msdd.py"),
            "--audio_path", str(audio_normalized),
            "--oracle_rttm", str(oracle_rttm),
            "--output_rttm", str(msdd_rttm),
            "--manifest_dir", str(output_dir / "manifests"),
            "--msdd_out_dir", str(output_dir / "diarization" / "msdd_outputs")
        ], "Step 5: MSDD Speaker Discovery")
    else:
        print("Skipping Step 5: Output already exists.")

    # [Step 6] Fusion
    if args.overwrite or not final_json.exists():
        run_command([
            "python", str(scripts_dir / "step6_fusion.py"),
            "--words_json", str(words_json),
            "--msdd_rttm", str(msdd_rttm),
            "--output_json", str(final_json),
            "--output_txt", str(final_txt)
        ], "Step 6: Fusion")
    else:
        print("Skipping Step 6: Output already exists.")

    print(f"\nPipeline complete! Final results:")
    print(f"JSON: {final_json}")
    print(f"TXT:  {final_txt}")

if __name__ == "__main__":
    main()
