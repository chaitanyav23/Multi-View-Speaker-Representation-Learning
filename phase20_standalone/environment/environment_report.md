# Phase20 Frozen Environment Report

Generated: 2026-07-14

This report describes the frozen environment that reproduced the original committed Phase20 results from commit `44e9bb7` and the final release validation run.

## Runtime

- Python: `3.10.12`
- OS: Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-1057-aws`
- GPU: Tesla T4, 15360 MiB, driver 580.159.04
- NVIDIA-SMI CUDA: `13.0`
- Torch: `2.8.0+cu128`
- Torch CUDA build: `12.8`
- cuDNN: `91301`
- TorchAudio: `2.8.0+cu128`
- FFmpeg: `4.4.2-0ubuntu0.22.04.1`

## Primary Python Packages

- `torch`: `2.8.0`
- `torchaudio`: `2.8.0`
- `whisperx`: `3.8.5`
- `faster-whisper`: `1.2.1`
- `ctranslate2`: `4.7.2`
- `pyannote-audio`: `4.0.4`
- `nemo-toolkit`: `2.7.3`
- `transformers`: `4.57.6`
- `numpy`: `2.2.6`
- `scipy`: `1.15.3`
- `soundfile`: `0.13.1`
- `jiwer`: `3.1.0`
- `DeepFilterNet`: `0.5.6`
- `DeepFilterLib`: `0.5.6`

Full package list is archived in `environment/pip_freeze.txt` and `environment/requirements_frozen.txt`.

## Required Environment Variables

| Variable | Value |
|---|---|
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `1` |
| `MPLCONFIGDIR` | `/tmp/marston_phase20_mpl` |
| `NUMBA_CACHE_DIR` | `/tmp/marston_phase20_numba` |
| `XDG_CACHE_HOME` | `$PWD/.cache` |
| `HF_HOME` | `$PWD/.cache/huggingface` |
| `TORCH_HOME` | `$PWD/.cache/torch` |
| `TRANSFORMERS_CACHE` | `$PWD/.cache/huggingface/transformers` |
| `HF_TOKEN` | `required for first model download if Pyannote gated models are not cached` |

## HuggingFace Authentication

Pyannote models are gated. `HF_TOKEN` must be set before first model download unless the exact model snapshots already exist in `HF_HOME`. The final validation used cached authenticated model snapshots listed in `environment/model_manifest.md`.

## Cache Directories

- HuggingFace: `.cache/huggingface`
- Torch: `.cache/torch`
- DeepFilterNet: `.cache/DeepFilterNet`
- Matplotlib: `/tmp/marston_phase20_mpl`
- Numba: `/tmp/marston_phase20_numba`

## Activation

```bash
source .venv_phase20/bin/activate
```

For the already recreated validation environment on this machine:

```bash
source /home/ubuntu/marston/.venv_phase20_44e9bb7/bin/activate
```

## Execution

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR=/tmp/marston_phase20_mpl
export NUMBA_CACHE_DIR=/tmp/marston_phase20_numba
export XDG_CACHE_HOME=$PWD/.cache
export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
export TRANSFORMERS_CACHE=$PWD/.cache/huggingface/transformers
python download_models.py
python run_phase20.py --overwrite
```

## Known Historical Conflicts

The original published environment can only be recreated by installing the lock file with `--no-deps`. Normal resolver installation changes package versions and reproduces the ASR regression. The expected `pip check` conflicts are:

- `deepfilternet 0.5.6` declares `numpy<2.0,>=1.22`, while the reproduced environment uses `numpy==2.2.6`.
- `wheel 0.47.0` declares `packaging>=24.0`, while the reproduced environment uses `packaging==23.2`.

These conflicts are documented and intentionally preserved for reproducibility.

## Models

See `environment/model_manifest.md` for exact revisions and hashes.
