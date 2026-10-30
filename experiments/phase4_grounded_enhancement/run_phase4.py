import subprocess
import argparse
from pathlib import Path
import os
import shutil

def run_command(cmd, description):
    print(f"\n>>> {description}")
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Grounded Enhancement Study")
    parser.add_argument("--skip-enhance", action="store_true")
    args = parser.parse_args()

    phase_dir = Path("experiments/phase4_grounded_enhancement")
    norm_audio_dir = phase_dir / "audio_normalized"
    grounding_dir = phase_dir / "grounding_master"
    
    # Enhancers to test
    enhancers = ["speechbrain", "deepfilternet"]
    
    # 1. Baseline: Run Hardened Pipeline on Normalized Audio (No Enhancement)
    run_command([
        "python3", "run_final_hardened_pipeline.py",
        "--audio-dir", str(norm_audio_dir),
        "--grounding-dir", str(grounding_dir),
        "--output-dir", str(phase_dir / "outputs_baseline"),
        "--overwrite"
    ], "Running Baseline (Normalized, No Enhancement)")

    # 2. Run Enhancements
    if not args.skip_enhance:
        # SpeechBrain
        run_command([
            "python3", "experiments/phase2_enhancement/pipeline_enhancement_speechbrain.py",
            "--audio-dir", str(norm_audio_dir),
            "--output-dir", str(phase_dir / "audio_speechbrain")
        ], "Applying SpeechBrain Enhancement")
        
        # DeepFilterNet
        run_command([
            "python3", "experiments/phase2_enhancement/pipeline_enhancement_deepfilternet.py",
            "--audio-dir", str(norm_audio_dir),
            "--output-dir", str(phase_dir / "audio_deepfilternet")
        ], "Applying DeepFilterNet Enhancement")

    # 3. Run Pipeline on Enhanced Audio
    for enhancer in enhancers:
        audio_dir = phase_dir / f"audio_{enhancer}"
        output_dir = phase_dir / f"outputs_{enhancer}"
        
        run_command([
            "python3", "run_final_hardened_pipeline.py",
            "--audio-dir", str(audio_dir),
            "--grounding-dir", str(grounding_dir),
            "--output-dir", str(output_dir),
            "--overwrite"
        ], f"Running Pipeline on {enhancer.capitalize()} Enhanced Audio")

    # 4. Evaluation
    print("\n>>> Starting Phase 4 Evaluation")
    eval_dirs = ["outputs_baseline", "outputs_speechbrain", "outputs_deepfilternet"]
    for edir in eval_dirs:
        pred_dir = phase_dir / edir
        output_name = f"phase4_{edir.replace('outputs_', '')}"
        
        run_command([
            "python3", "evaluate_conversational_timeline.py",
            "--pred-dir", str(pred_dir),
            "--output-name", output_name
        ], f"Evaluating {output_name}")

    print("\n[PHASE 4 COMPLETE] Enhancement Study Finished.")

if __name__ == "__main__":
    main()
