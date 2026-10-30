#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${PHASE20_ENV_DIR:-.venv_phase20}"
PYTHON_BIN="${PHASE20_PYTHON:-python3}"
SKIP_MODELS=0
VALIDATE_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-models)
      SKIP_MODELS=1
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    return 1
  fi
}

install_system_packages() {
  local missing=()
  for exe in ffmpeg ffprobe git gcc g++ cmake "$PYTHON_BIN"; do
    if ! command -v "$exe" >/dev/null 2>&1; then
      missing+=("$exe")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "Missing system dependencies: ${missing[*]}"
    if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y ffmpeg git build-essential cmake libsndfile1 python3 python3-venv python3-dev
    else
      echo "Install these packages manually: ffmpeg git build-essential cmake libsndfile1 python3 python3-venv python3-dev" >&2
      exit 1
    fi
  fi
}

activate_environment() {
  if [ ! -d "$ENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$ENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
}

install_python_packages() {
  python -m pip install --upgrade pip==26.1.2 setuptools==83.0.0 wheel==0.47.0

  # The published Phase20 result is a historical environment. A normal resolver
  # install upgrades/downgrades transitive packages and changes WhisperX output.
  # The lock file is therefore installed with --no-deps to preserve the exact
  # package set that reproduced commit 44e9bb7.
  python -m pip install --no-deps -r environment/requirements.lock

  set +e
  python -m pip check
  local pip_check_status=$?
  set -e
  if [ "$pip_check_status" -ne 0 ]; then
    echo
    echo "pip check reported the known historical conflicts documented in README.md."
    echo "Do not resolve these conflicts unless intentionally leaving the published environment."
  fi
}

export_runtime_environment() {
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/marston_phase20_mpl}"
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/marston_phase20_numba}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
  export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-$ROOT_DIR/.cache/torch}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT_DIR/.cache/huggingface/transformers}"
  mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE"
}

validate_imports() {
  python - <<'PY'
import importlib

modules = [
    "torch",
    "torchaudio",
    "pyannote.audio",
    "whisperx",
    "nemo.collections.asr",
    "df",
    "scipy",
    "numpy",
    "jiwer",
    "pandas",
    "soundfile",
    "ffmpeg",
]

failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append((name, repr(exc)))

if failed:
    for name, exc in failed:
        print(f"IMPORT FAILED: {name}: {exc}")
    raise SystemExit(1)

print("All Phase20 imports succeeded.")
PY
}

write_environment_report() {
  python - <<'PY'
from pathlib import Path
import json
import platform
import subprocess

import torch
import torchaudio
import whisperx
import faster_whisper
import ctranslate2
import pyannote.audio
import nemo
import transformers
import numpy
import scipy
import soundfile

root = Path.cwd()
env = root / "environment"
env.mkdir(exist_ok=True)

def run(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()

freeze = run(["python", "-m", "pip", "freeze", "--all"])
(env / "pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")
(env / "requirements_frozen.txt").write_text(freeze + "\n", encoding="utf-8")
(env / "python_version.txt").write_text(run(["python", "--version"]) + "\n", encoding="utf-8")
(env / "pip_version.txt").write_text(run(["python", "-m", "pip", "--version"]) + "\n", encoding="utf-8")
(env / "ffmpeg_version.txt").write_text(run(["ffmpeg", "-version"]) + "\n", encoding="utf-8")
(env / "gpu_information.txt").write_text(run(["nvidia-smi"]) + "\n", encoding="utf-8")
(env / "cuda_version.txt").write_text(
    f"NVIDIA-SMI CUDA Version: see gpu_information.txt\nTorch CUDA build: {torch.version.cuda}\nTorch package: {torch.__version__}\n",
    encoding="utf-8",
)
(env / "cudnn_version.txt").write_text(
    f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}\n",
    encoding="utf-8",
)
(env / "system_information.txt").write_text(
    "\n\n".join([run(["uname", "-a"]), run(["cat", "/etc/os-release"]), run(["lscpu"]), run(["free", "-h"])]),
    encoding="utf-8",
)

snapshot = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torchaudio": torchaudio.__version__,
    "whisperx": getattr(whisperx, "__version__", None),
    "faster_whisper": faster_whisper.__version__,
    "ctranslate2": ctranslate2.__version__,
    "pyannote_audio": pyannote.audio.__version__,
    "nemo": nemo.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "soundfile": soundfile.__version__,
}
(env / "environment_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

report = f"""# Phase20 Environment Report

- Python: `{snapshot['python']}`
- Torch: `{snapshot['torch']}`
- Torch CUDA: `{snapshot['torch_cuda']}`
- cuDNN: `{snapshot['cudnn']}`
- CUDA available: `{snapshot['cuda_available']}`
- GPU: `{snapshot['gpu']}`
- TorchAudio: `{snapshot['torchaudio']}`
- WhisperX: `{snapshot['whisperx']}`
- Faster-Whisper: `{snapshot['faster_whisper']}`
- ctranslate2: `{snapshot['ctranslate2']}`
- Pyannote Audio: `{snapshot['pyannote_audio']}`
- NeMo: `{snapshot['nemo']}`
- Transformers: `{snapshot['transformers']}`
- NumPy: `{snapshot['numpy']}`
- SciPy: `{snapshot['scipy']}`
- SoundFile: `{snapshot['soundfile']}`

The full package freeze is in `environment/pip_freeze.txt`.
"""
(env / "environment_report.md").write_text(report, encoding="utf-8")
PY
}

install_system_packages
activate_environment
export_runtime_environment

if [ "$VALIDATE_ONLY" -eq 0 ]; then
  install_python_packages
fi

validate_imports

if [ "$SKIP_MODELS" -eq 0 ]; then
  if [ -z "${HF_TOKEN:-}" ] && [ ! -d "$HF_HOME/hub/models--pyannote--voice-activity-detection" ]; then
    echo "HF_TOKEN is required for first download of gated Pyannote models." >&2
    exit 1
  fi
  python download_models.py
fi

write_environment_report

cat <<EOF

Phase20 environment is ready.

Activate:
  source $ENV_DIR/bin/activate

Run:
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export MPLCONFIGDIR=/tmp/marston_phase20_mpl
  export NUMBA_CACHE_DIR=/tmp/marston_phase20_numba
  export XDG_CACHE_HOME=\$PWD/.cache
  export HF_HOME=\$PWD/.cache/huggingface
  export TORCH_HOME=\$PWD/.cache/torch
  export TRANSFORMERS_CACHE=\$PWD/.cache/huggingface/transformers
  python run_phase20.py --overwrite
EOF
