import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import ensure_dirs, recording_paths  # noqa: E402


def phase_env() -> dict[str, str]:
    project_root = BASE_DIR
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/marston_phase20_mpl")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/marston_phase20_numba")
    env.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    env.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    env.setdefault("TORCH_HOME", str(project_root / ".cache" / "torch"))
    return env


def run_step(cmd: list[Path | str], label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    result = subprocess.run([sys.executable, *map(str, cmd)], cwd=BASE_DIR, env=phase_env())
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Phase 20 dual-path improved-VAD speech processing experiment.")
    parser.add_argument("--file", required=True, help="Recording stem, e.g. traffic_stop for raw_audio/traffic_stop.wav")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    paths = recording_paths(args.file)
    overwrite = ["--overwrite"] if args.overwrite else []
    file_args = ["--file", paths.name]
    run_step([SCRIPTS / "step00_ground_truth.py", *file_args, *overwrite], "00 Ground Truth Conversion")
    run_step([SCRIPTS / "step01_preprocess.py", *file_args, *overwrite], "01 Audio Preprocessing")
    run_step([SCRIPTS / "step02a_deepfilternet.py", *file_args, *overwrite], "02 DeepFilterNet")
    run_step([SCRIPTS / "step02b_loudnorm.py", *file_args, *overwrite], "03 Loudness Normalization")
    run_step([SCRIPTS / "step03_vad.py", *file_args, "--path-name", "pathA", *overwrite], "04 Path A VAD")
    run_step([SCRIPTS / "step03_vad.py", *file_args, "--path-name", "pathB", *overwrite], "05 Path B VAD")
    run_step([SCRIPTS / "step04_vad_eval.py", *file_args, "--label", "pathA"], "06 Path A Evaluation")
    run_step([SCRIPTS / "step04_vad_eval.py", *file_args, "--label", "pathB"], "07 Path B Evaluation")
    run_step([SCRIPTS / "step05b_whisperx.py", *file_args, *overwrite], "08 WhisperX")
    run_step([SCRIPTS / "step05a_msdd.py", *file_args, *overwrite], "09 MSDD")
    run_step([SCRIPTS / "step06b_asr_eval.py", *file_args], "10 ASR Evaluation")
    run_step([SCRIPTS / "step06a_diar_eval.py", *file_args], "11 Diarization Evaluation")
    run_step([SCRIPTS / "step07_fusion.py", *file_args], "12 Fusion")
    run_step([SCRIPTS / "step08_fusion_eval.py", *file_args], "13 Fusion Evaluation")
    run_step([SCRIPTS / "step09_aggregate_report.py", *file_args], "14 Aggregate Report")
    run_step([SCRIPTS / "step10_compare_phase19.py", *file_args], "15 Phase19 Comparison")
    print("\nPhase 20 complete.")


if __name__ == "__main__":
    main()
