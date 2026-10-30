import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    print(f"\n>>> {description}")
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Enhancement Benchmarking Runner")
    parser.add_argument("--skip-enhance", action="store_true", help="Skip enhancement generation")
    args = parser.parse_args()

    enhancers = ["deepfilternet", "speechbrain"]
    
    if not args.skip_enhance:
        for enhancer in enhancers:
            run_command([
                "python3", f"experiments/phase2_enhancement/pipeline_enhancement_{enhancer}.py"
            ], f"Running Enhancement: {enhancer.capitalize()}")

    for enhancer in enhancers:
        audio_dir = f"experiments/phase2_enhancement/audio_{enhancer}"
        grounding_dir = f"experiments/phase2_enhancement/grounding_{enhancer}"
        output_dir = f"experiments/phase2_enhancement/outputs_{enhancer}"
        report_name = f"phase2_{enhancer}"
        
        # 1. Run VAD (Silero) on enhanced audio
        run_command([
            "python3", "custom_pipeline_stage1_vad.py",
            "--audio-dir", audio_dir,
            "--output-dir", grounding_dir
        ], f"Generating VAD Grounding for {enhancer.capitalize()}")

        # 2. Run Hardened Pipeline
        run_command([
            "python3", "run_final_hardened_pipeline.py",
            "--audio-dir", audio_dir,
            "--grounding-dir", grounding_dir,
            "--output-dir", output_dir,
            "--overwrite"
        ], f"Running Hardened Pipeline with {enhancer.capitalize()} Enhanced Audio")
        
        # 3. Evaluate
        run_command([
            "python3", "evaluate_conversational_timeline.py",
            "--pred-dir", output_dir,
            "--output-name", report_name
        ], f"Evaluating {enhancer.capitalize()} Enhancement Pipeline")

    print("\n[PHASE 2 COMPLETE] All enhancement variants evaluated.")

if __name__ == "__main__":
    main()
