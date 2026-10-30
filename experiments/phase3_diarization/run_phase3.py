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
    parser = argparse.ArgumentParser(description="Phase 3: Diarization Benchmarking Runner")
    parser.add_argument("--skip-nemo", action="store_true", help="Skip NeMo generation")
    args = parser.parse_args()

    # 1. Run NeMo MSDD
    if not args.skip_nemo:
        run_command([
            "python3", "experiments/phase3_diarization/pipeline_msdd.py"
        ], "Running NeMo MSDD Diarization")

    # 2. Run Baseline (Pyannote 3.1)
    run_command([
        "python3", "run_final_hardened_pipeline.py",
        "--audio-dir", "raw_audio",
        "--grounding-dir", "vad_custom_grounding",
        "--output-dir", "experiments/phase3_diarization/outputs_pyannote31",
        "--overwrite"
    ], "Running Pyannote 3.1 Baseline")
    
    # 3. Run MSDD Pipeline
    run_command([
        "python3", "experiments/phase3_diarization/run_hardened_with_custom_diar.py",
        "--audio-dir", "raw_audio",
        "--grounding-dir", "vad_custom_grounding",
        "--diar-json-dir", "experiments/phase3_diarization/outputs_msdd",
        "--output-dir", "experiments/phase3_diarization/outputs_msdd_final"
    ], "Merging ASR with MSDD Diarization")

    # 4. Evaluate
    run_command([
        "python3", "evaluate_conversational_timeline.py",
        "--pred-dir", "experiments/phase3_diarization/outputs_pyannote31",
        "--output-name", "phase3_pyannote31"
    ], "Evaluating Pyannote 3.1 Baseline")

    run_command([
        "python3", "evaluate_conversational_timeline.py",
        "--pred-dir", "experiments/phase3_diarization/outputs_msdd_final",
        "--output-name", "phase3_msdd"
    ], "Evaluating NeMo MSDD Pipeline")

    print("\n[PHASE 3 COMPLETE] Diarization Study Finished.")

if __name__ == "__main__":
    main()
