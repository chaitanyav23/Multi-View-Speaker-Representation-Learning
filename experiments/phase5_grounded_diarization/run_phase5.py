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
    parser = argparse.ArgumentParser(description="Phase 5: Grounded Diarization Study (Final)")
    parser.add_argument("--skip-diar", action="store_true")
    args = parser.parse_args()

    phase_dir = Path("experiments/phase5_grounded_diarization")
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    norm_audio_dir = Path("experiments/phase4_grounded_enhancement/audio_normalized")
    grounding_dir = Path("experiments/phase4_grounded_enhancement/grounding_master")
    oracle_rttm_dir = Path("experiments/phase5_audit/oracle_rttms")
    
    # 1. Ensure Oracle RTTMs are ready (from master grounding)
    run_command([
        "python3", "scripts/utils/convert_grounding_to_rttm.py",
        "--grounding-dir", str(grounding_dir),
        "--output-dir", str(oracle_rttm_dir)
    ], "Preparing Oracle RTTMs")

    # 2. Baseline: Pyannote 3.1 Auto
    run_command([
        "python3", "run_final_hardened_pipeline.py",
        "--audio-dir", str(norm_audio_dir),
        "--grounding-dir", str(grounding_dir),
        "--output-dir", str(phase_dir / "outputs_pyannote31_auto"),
        "--overwrite"
    ], "Running Pyannote 3.1 Auto Baseline")

    # 3. Baseline: Pyannote 3.1 Fixed-2 (for comparison on multi-speaker detection)
    run_command([
        "python3", "run_final_hardened_pipeline.py",
        "--audio-dir", str(norm_audio_dir),
        "--grounding-dir", str(grounding_dir),
        "--output-dir", str(phase_dir / "outputs_pyannote31_fixed2"),
        "--min-speakers", "2", "--max-speakers", "2",
        "--overwrite"
    ], "Running Pyannote 3.1 Fixed-2 Baseline")

    # 4. Advanced: NeMo MSDD (Oracle VAD)
    if not args.skip_diar:
        # Generate MSDD segments using the hardened oracle config
        run_command([
            "python3", "experiments/phase5_audit/pipeline_msdd_oracle.py",
            "--audio-dir", str(norm_audio_dir),
            "--oracle-rttm-dir", str(oracle_rttm_dir),
            "--output-dir", str(phase_dir / "diar_msdd_raw"),
            "--oracle-vad"
        ], "Generating NeMo MSDD Speaker Segments (Oracle VAD)")
        
        # Merge with Hardened Pipeline
        run_command([
            "python3", "experiments/phase3_diarization/run_hardened_with_custom_diar.py",
            "--audio-dir", str(norm_audio_dir),
            "--grounding-dir", str(grounding_dir),
            "--diar-json-dir", str(phase_dir / "diar_msdd_raw"),
            "--output-dir", str(phase_dir / "outputs_msdd")
        ], "Merging Hardened ASR with MSDD Diarization")

    # 5. Evaluation & Diagnostics
    print("\n>>> Starting Phase 5 Final Evaluation")
    variants = ["pyannote31_auto", "pyannote31_fixed2", "msdd"]
    for var in variants:
        pred_dir = phase_dir / f"outputs_{var}"
        if not pred_dir.exists(): continue
        
        output_name = f"phase5_{var}"
        
        # Timeline Eval (WER, DER, SA-WER, cpWER, etc.)
        run_command([
            "python3", "evaluate_conversational_timeline.py",
            "--pred-dir", str(pred_dir),
            "--output-name", output_name
        ], f"Evaluating {var} Pipeline")
        
        # Detailed Diagnostics (Miss, FA, Conf clips)
        run_command([
            "python3", "scripts/utils/diarization_diagnostics.py",
            "--audio-dir", str(norm_audio_dir),
            "--pred-dir", str(pred_dir),
            "--output-dir", str(phase_dir / f"diagnostics_{var}")
        ], f"Generating Diagnostics for {var}")

    print("\n[PHASE 5 COMPLETE] Final Diarization Study Finished.")

if __name__ == "__main__":
    main()
