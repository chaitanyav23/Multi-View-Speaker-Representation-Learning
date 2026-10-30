# Phase20 Reproducibility Guide

This guide describes how to reproduce the published Phase20 result from a clean clone.

## Hardware and Operating System

The validated run used:

| Item | Value |
|---|---:|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-1057-aws |
| GPU | NVIDIA Tesla T4 |
| GPU memory | 15360 MiB |
| NVIDIA driver | 580.159.04 |
| NVIDIA-SMI CUDA | 13.0 |
| RAM | See `environment/system_information.txt` |
| Storage | At least 30 GB free recommended for model caches and generated outputs |

CPU execution is not the published configuration. Use a CUDA-capable GPU for reproduction.

## Environment Recreation

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git build-essential cmake libsndfile1 python3 python3-venv python3-dev
```

Set HuggingFace authentication:

```bash
export HF_TOKEN=hf_your_token_here
```

Create the frozen environment and download models:

```bash
./install.sh
```

Activate the environment:

```bash
source .venv_phase20/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR=/tmp/marston_phase20_mpl
export NUMBA_CACHE_DIR=/tmp/marston_phase20_numba
export XDG_CACHE_HOME=$PWD/.cache
export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
export TRANSFORMERS_CACHE=$PWD/.cache/huggingface/transformers
```

## Verification Before Running

Validate imports:

```bash
python - <<'PY'
import torch, torchaudio, whisperx, pyannote.audio
import nemo.collections.asr
import scipy, numpy, jiwer, pandas, soundfile, ffmpeg
import df
print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())
PY
```

Expected key versions:

| Component | Version |
|---|---:|
| Python | 3.10.12 |
| Torch | 2.8.0+cu128 |
| TorchAudio | 2.8.0+cu128 |
| WhisperX | 3.8.5 |
| Faster-Whisper | 1.2.1 |
| ctranslate2 | 4.7.2 |
| Pyannote Audio | 4.0.4 |
| NeMo Toolkit | 2.7.3 |
| NumPy | 2.2.6 |

## Pipeline Execution

Run:

```bash
python run_phase20.py --overwrite
```

The expected execution time depends on the GPU and model cache state. On the validated Tesla T4 environment, model-warm runs should complete without manual intervention; first runs may take longer because HuggingFace and NeMo caches are populated.

## Expected Outputs

After completion, these files should exist:

| Output | Expected |
|---|---|
| `audio/bwc_dfn.wav` | DeepFilterNet-processed audio |
| `grounding/pathB_vad.json` | 41 VAD segments |
| `transcription/aligned_words.json` | 465 aligned words in 41 Whisper segments |
| `diarization/pathA_msdd.rttm` | NeMo MSDD diarization output |
| `fusion/attributed_words.json` | Speaker-attributed words |
| `metrics/asr_metrics.json` | WER near 0.324380 |
| `metrics/all_metrics.json` | Aggregate Phase20 metrics |

## Expected Metrics

| Metric | Expected value |
|---|---:|
| ASR WER | 0.324380 |
| Aligned words | 465 |
| Whisper segments | 41 |
| Path B VAD segments | 41 |
| Path B speech duration | 108.081 s |
| DER | 0.447059 |
| Speaker accuracy | 0.902743 |
| Timeline accuracy | 0.705312 |

Small formatting differences in logs are acceptable. Metric drift is not expected.

## Validation Steps

1. Confirm the active environment path with `which python`.
2. Confirm `python -m pip freeze --all` matches `environment/requirements_frozen.txt`.
3. Confirm `python download_models.py` completes.
4. Run `python run_phase20.py --overwrite`.
5. Compare `metrics/asr_metrics.json`, `metrics/diarization_metrics.json`, and `metrics/fusion_metrics.json` to `results_to_be_published.md`.
6. Confirm that logs show real DeepFilterNet, Pyannote, WhisperX, and NeMo execution.

## Known Reproducibility Constraints

- The lock file must be installed with `--no-deps`.
- The two expected `pip check` conflicts are documented in `README.md`.
- Pyannote model access requires HuggingFace authentication and accepted model terms.
- CUDA/GPU execution is the validated configuration.
- The model cache locations and artifact hashes are archived in `environment/model_manifest.md`.

## Failure Modes

| Symptom | Likely cause |
|---|---|
| WER near 0.719 and 369 aligned words | Dependency drift from the published environment |
| Pyannote load failure | Missing `HF_TOKEN` or unaccepted gated model terms |
| CPU-only run | Missing NVIDIA driver, incompatible CUDA, or inactive frozen venv |
| DeepFilterNet fallback | Missing `DeepFilterNet==0.5.6` or unavailable `deepFilter` command |

## Reproducibility Statement

The repository reproduces the original committed Phase20 result when executed with the archived frozen environment described in `environment/` and the model revisions listed in `environment/model_manifest.md`.
