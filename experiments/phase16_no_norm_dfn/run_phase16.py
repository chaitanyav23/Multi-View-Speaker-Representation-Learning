import os
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n>>> Running {description}...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run Phase 16: No-Normalization + DFN Experiment.")
    parser.add_argument("--audio", type=str, default="raw_audio/karen.wav", help="Path to input audio file")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing files")
    args = parser.parse_args()

    audio_in = Path(args.audio)
    output_dir = Path("experiments/phase16_no_norm_dfn")
    
    # Create directory structure
    subdirs = [
        "audio", "grounding", "manifests", "outputs", "diarization", 
        "comparisons", "diagnostics", "reports", "visualizations",
        "diagnostics/vad", "diagnostics/asr", "diagnostics/diarization", "diagnostics/attribution", "diagnostics/conversational",
        "reports/vad", "reports/asr", "reports/diarization", "reports/attribution", "reports/conversational"
    ]
    for d in subdirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    audio_repaired = output_dir / "audio" / "karen_repaired.wav"
    audio_dfn = output_dir / "audio" / "karen_dfn.wav"
    scripts_dir = output_dir

    # [Step 0] Audio Repair
    if args.overwrite or not audio_repaired.exists():
        run_command([
            "ffmpeg", "-y", "-i", str(audio_in),
            "-ar", "16000", "-ac", "1",
            str(audio_repaired)
        ], "Step 0: Audio Repair (Downsample/Mono)")

    # [Step 1] DeepFilterNet Enhancement
    if args.overwrite or not audio_dfn.exists():
        print(f"\n>>> Running Step 1: DeepFilterNet Enhancement...")
        # DeepFilterNet might be slow, so we use subprocess.run
        cmd = [
            "python", "-m", "df.enhance", 
            "--output-dir", str(output_dir / "audio"),
            str(audio_repaired)
        ]
        subprocess.run(cmd, check=True)
        # DeepFilterNet3 usually appends suffix. Let's find and rename.
        dfn_out = output_dir / "audio" / "karen_repaired_DeepFilterNet3.wav"
        if dfn_out.exists():
            dfn_out.rename(audio_dfn)
        else:
            # Fallback if suffix is different
            found = list((output_dir / "audio").glob("*DeepFilterNet*.wav"))
            if found:
                found[0].rename(audio_dfn)
            else:
                print("Error: DFN output not found.")
                sys.exit(1)
    else:
        print("Skipping Step 1: DFN output already exists.")

    # Execute Steps 2 through 14
    steps = [
        ("step2_vad_grounding.py", "Step 2: VAD Grounding"),
        ("step3_oracle_rttm.py", "Step 3: Oracle RTTM"),
        ("step4_whisperx.py", "Step 4: WhisperX ASR"),
        ("step5_msdd.py", "Step 5: MSDD Speaker Discovery"),
        ("step6_fusion.py", "Step 6: Fusion"),
        ("step7_process_gt.py", "Step 7: Process Ground Truth"),
        ("step8_vad_eval.py", "Step 8: VAD Evaluation"),
        ("step9_asr_eval.py", "Step 9: ASR Evaluation"),
        ("step10_diar_eval.py", "Step 10: Diarization Evaluation"),
        ("step11_attribution_eval.py", "Step 11: Attribution Evaluation"),
        ("step12_13_conversational_eval.py", "Step 12/13: Conversational Evaluation"),
        ("step14_comparison.py", "Step 14: Comparison Analysis")
    ]

    for script, desc in steps:
        run_command(["python", str(scripts_dir / script)], desc)

    print(f"\nPhase 16 (No Norm + DFN) complete!")

if __name__ == "__main__":
    main()
