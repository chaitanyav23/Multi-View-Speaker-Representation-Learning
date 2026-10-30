import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = BASE_DIR / "scripts"


def phase_env():
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/marston_phase19_mpl")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/marston_phase19_numba")
    env.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    env.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    env.setdefault("TORCH_HOME", str(project_root / ".cache" / "torch"))
    return env


def run_step(cmd, label):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    result = subprocess.run([sys.executable, *map(str, cmd)], cwd=Path(__file__).resolve().parents[2], env=phase_env())
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Phase 19 dual-path speech processing experiment.")
    parser.add_argument("--audio", type=Path, default=Path("raw_audio/bwc.wav"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    overwrite = ["--overwrite"] if args.overwrite else []
    run_step([SCRIPTS / "step01_preprocess.py", "--input", args.audio, *overwrite], "1 preprocess")
    run_step([SCRIPTS / "step02a_deepfilternet.py", *overwrite], "2 DeepFilterNet")
    run_step([SCRIPTS / "step02b_loudnorm.py", *overwrite], "3 normalization")
    run_step([SCRIPTS / "step03_vad.py", "--path-name", "pathA", "--input", BASE_DIR / "audio" / "bwc_dfn.wav", "--output", BASE_DIR / "grounding" / "pathA_vad.json", *overwrite], "4 VAD A")
    run_step([SCRIPTS / "step03_vad.py", "--path-name", "pathB", "--input", BASE_DIR / "audio" / "bwc_normalized.wav", "--output", BASE_DIR / "grounding" / "pathB_vad.json", *overwrite], "5 VAD B")
    run_step([SCRIPTS / "step04_vad_eval.py", "--label", "pathA", "--hypothesis", BASE_DIR / "grounding" / "pathA_vad.json", "--metrics-output", BASE_DIR / "metrics" / "pathA_vad_metrics.json", "--report-output", BASE_DIR / "reports" / "pathA_vad_report.md"], "6 VAD evaluation A")
    run_step([SCRIPTS / "step04_vad_eval.py", "--label", "pathB", "--hypothesis", BASE_DIR / "grounding" / "pathB_vad.json", "--metrics-output", BASE_DIR / "metrics" / "pathB_vad_metrics.json", "--report-output", BASE_DIR / "reports" / "pathB_vad_report.md"], "7 VAD evaluation B")
    run_step([SCRIPTS / "step05b_whisperx.py", *overwrite], "8 WhisperX")
    run_step([SCRIPTS / "step05a_msdd.py", *overwrite], "9 MSDD")
    run_step([SCRIPTS / "step06b_asr_eval.py"], "10 ASR evaluation")
    run_step([SCRIPTS / "step06a_diar_eval.py"], "11 Diarization evaluation")
    run_step([SCRIPTS / "step07_fusion.py"], "12 Fusion")
    run_step([SCRIPTS / "step08_fusion_eval.py"], "13 Fusion evaluation")
    run_step([SCRIPTS / "step09_aggregate_report.py"], "14/15 Metrics aggregation and final report")
    print("\nPhase 19 complete.")


if __name__ == "__main__":
    main()
