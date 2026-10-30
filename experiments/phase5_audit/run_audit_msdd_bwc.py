import subprocess
import time
import argparse
from pathlib import Path
import json

def run_command(cmd, description):
    print(f"\n>>> {description}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        return False, duration
    print(f"COMPLETE (Runtime: {duration:.2f}s)")
    return True, duration

def main():
    phase_dir = Path("experiments/phase5_audit/msdd_bwc_test")
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = "experiments/phase5_audit/temp_audio"
    oracle_rttm_dir = "experiments/phase5_audit/oracle_rttms"
    grounding_dir = "experiments/phase4_grounded_enhancement/grounding_master"
    
    stages_timing = {}

    # Stage 1 & 2: RTTM & JSON
    success, dur = run_command([
        "python3", "experiments/phase5_audit/pipeline_msdd_oracle.py",
        "--audio-dir", audio_dir,
        "--oracle-rttm-dir", oracle_rttm_dir,
        "--output-dir", str(phase_dir / "msdd_raw"),
        "--oracle-vad"
    ], "Stage 1 & 2: NeMo MSDD RTTM & JSON Generation")
    stages_timing["Stage 1 & 2: RTTM/JSON"] = dur
    if not success: return

    # Verify RTTM
    rttm_path = phase_dir / "msdd_raw" / "bwc" / "pred_rttms" / "bwc.rttm"
    if not rttm_path.exists():
        print(f"ERROR: pred_rttms/bwc.rttm not found at {rttm_path}")
        return

    # Stage 3 & 4: Hardened ASR Merge & Transcript
    success, dur = run_command([
        "python3", "experiments/phase3_diarization/run_hardened_with_custom_diar.py",
        "--audio-dir", audio_dir,
        "--grounding-dir", grounding_dir,
        "--diar-json-dir", str(phase_dir / "msdd_raw"),
        "--output-dir", str(phase_dir / "outputs_msdd")
    ], "Stage 3 & 4: Hardened ASR Merge & Transcript")
    stages_timing["Stage 3 & 4: ASR Merge/Transcript"] = dur
    if not success: return

    # Stage 5: Evaluation
    success, dur = run_command([
        "python3", "evaluate_conversational_timeline.py",
        "--pred-dir", str(phase_dir / "outputs_msdd"),
        "--output-name", "phase5_audit_msdd_bwc"
    ], "Stage 5: DER Evaluation")
    stages_timing["Stage 5: Evaluation"] = dur

    # Stage 6: Diagnostics
    success, dur = run_command([
        "python3", "scripts/utils/diarization_diagnostics.py",
        "--audio-dir", audio_dir,
        "--pred-dir", str(phase_dir / "outputs_msdd"),
        "--output-dir", str(phase_dir / "diagnostics_msdd")
    ], "Stage 6: Diagnostics Generation")
    stages_timing["Stage 6: Diagnostics"] = dur

    print("\n" + "="*30)
    print("PHASE 5 MSDD AUDIT TIMING REPORT")
    print("="*30)
    for stage, t in stages_timing.items():
        print(f"{stage:40}: {t:.2f}s")
    
    with open(phase_dir / "audit_timing.json", "w") as f:
        json.dump(stages_timing, f, indent=2)

if __name__ == "__main__":
    main()
