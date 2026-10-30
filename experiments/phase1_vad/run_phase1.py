import subprocess
import argparse
from pathlib import Path
import pandas as pd

def run_command(cmd, description):
    print(f"\n>>> {description}")
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Phase 1: VAD Benchmarking Runner")
    parser.add_argument("--skip-vad", action="store_true", help="Skip VAD generation, run only eval")
    args = parser.parse_args()

    vads = ["silero", "pyannote"]
    
    if not args.skip_vad:
        for vad in vads:
            run_command([
                "python3", f"experiments/phase1_vad/pipeline_vad_{vad}.py"
            ], f"Generating Grounding for {vad.capitalize()}")

    for vad in vads:
        grounding_dir = f"experiments/phase1_vad/grounding_{vad}"
        output_dir = f"experiments/phase1_vad/outputs_{vad}"
        report_name = f"phase1_{vad}"
        
        # 1. Run Hardened Pipeline
        run_command([
            "python3", "run_final_hardened_pipeline.py",
            "--audio-dir", "raw_audio",
            "--grounding-dir", grounding_dir,
            "--output-dir", output_dir,
            "--overwrite"
        ], f"Running Hardened Pipeline with {vad.capitalize()} VAD")
        
        # 2. Evaluate
        run_command([
            "python3", "evaluate_conversational_timeline.py",
            "--pred-dir", output_dir,
            "--output-name", report_name
        ], f"Evaluating {vad.capitalize()} VAD Pipeline")

    print("\n[PHASE 1 COMPLETE] All VAD variants evaluated.")

if __name__ == "__main__":
    main()
