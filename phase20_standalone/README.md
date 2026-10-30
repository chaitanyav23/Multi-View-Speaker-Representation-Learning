# Phase20 Standalone Reproducible Artifact

This repository is the final standalone Phase20 research artifact for `raw_audio/bwc.wav`. It preserves the Phase20 pipeline and evaluation logic, and freezes the historical software environment that reproduced the original committed Phase20 results from commit `44e9bb7`.

The pipeline uses real DeepFilterNet, Pyannote, WhisperX, and NeMo models when the frozen environment is active. Fallback implementations remain in the source for robustness, but the published run must not rely on them.

## Published Environment

| Component | Published version |
|---|---:|
| Python | 3.10.12 |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-1057-aws |
| GPU | NVIDIA Tesla T4, 15360 MiB |
| NVIDIA driver | 580.159.04 |
| NVIDIA-SMI CUDA | 13.0 |
| Torch | 2.8.0+cu128 |
| Torch CUDA build | 12.8 |
| TorchAudio | 2.8.0+cu128 |
| cuDNN | 91301 |
| WhisperX | 3.8.5 |
| Faster-Whisper | 1.2.1 |
| ctranslate2 | 4.7.2 |
| Pyannote Audio | 4.0.4 |
| NeMo Toolkit | 2.7.3 |
| Transformers | 4.57.6 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| SoundFile | 0.13.1 |
| FFmpeg | 4.4.2-0ubuntu0.22.04.1 |

Full package and system manifests are in `environment/`.

## Pipeline Architecture

```text
raw_audio/bwc.wav
  -> scripts/step01_preprocess.py
  -> audio/bwc_repaired.wav
  -> scripts/step02a_deepfilternet.py
  -> audio/bwc_dfn.wav
  -> scripts/step02b_loudnorm.py
  -> audio/bwc_normalized.wav
  -> scripts/step03_vad.py for Path A and Path B
  -> scripts/step04_vad_eval.py for Path A and Path B
  -> scripts/step05b_whisperx.py
  -> transcription/aligned_words.json
  -> scripts/step05a_msdd.py
  -> diarization/pathA_msdd.rttm
  -> scripts/step06b_asr_eval.py
  -> scripts/step06a_diar_eval.py
  -> scripts/step07_fusion.py
  -> scripts/step08_fusion_eval.py
  -> scripts/step09_aggregate_report.py
  -> scripts/step10_compare_phase19.py
```

## Directory Structure

| Path | Purpose |
|---|---|
| `run_phase20.py` | Pipeline orchestrator |
| `scripts/` | Stage scripts and shared metric utilities |
| `raw_audio/` | Input audio |
| `ground_truth_json/` | Word/text/speaker reference JSON |
| `ground_truth_rttm/` | Diarization reference RTTM |
| `audio/` | Generated audio artifacts |
| `grounding/` | VAD outputs and oracle RTTM |
| `diarization/` | NeMo MSDD outputs |
| `transcription/` | WhisperX aligned words |
| `fusion/` | Speaker-attributed words and transcript |
| `metrics/` | Per-stage and aggregate metric JSON |
| `reports/` | Validation and per-stage reports |
| `outputs/` | Stage metadata JSON |
| `phase19_reference/` | Phase19 reference artifacts for comparison |
| `environment/` | Frozen software, model, and system manifests |

## Installation

Ubuntu 22.04 is the validated platform.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git build-essential cmake libsndfile1 python3 python3-venv python3-dev
```

Set a HuggingFace token before first model download. Pyannote models are gated and require accepting the model terms on HuggingFace.

```bash
export HF_TOKEN=hf_your_token_here
```

Create the frozen environment, download models, validate imports, and regenerate the environment report:

```bash
./install.sh
```

The installer creates `.venv_phase20` by default. Override with:

```bash
PHASE20_ENV_DIR=/path/to/env ./install.sh
```

## Why `--no-deps` Is Required

The historical environment cannot be reproduced with a normal dependency resolver run. Resolver installation changes transitive package versions and changes WhisperX output. The release installer therefore runs:

```bash
python -m pip install --no-deps -r environment/requirements.lock
```

The expected `pip check` conflicts are:

- `DeepFilterNet==0.5.6` declares `numpy<2.0,>=1.22`, while the published environment uses `numpy==2.2.6`.
- `wheel==0.47.0` declares `packaging>=24.0`, while the published environment uses `packaging==23.2`.

These are archived historical conflicts. Do not resolve them unless intentionally leaving the published environment.

## Models

`download_models.py` warms or verifies these models:

| Model | Revision or package |
|---|---|
| Whisper large-v3 | `Systran/faster-whisper-large-v3`, revision `edaa852ec7e145841d8ffdb056a99866b5f0a478` |
| WhisperX alignment | `wav2vec2_fairseq_base_ls960_asr_ls960.pth` |
| Pyannote VAD | `pyannote/voice-activity-detection`, revision `8d678d7647d9c991934b2270b21fe390f9a8fea5` |
| DeepFilterNet | `DeepFilterNet==0.5.6`, DeepFilterNet3 assets |
| TitaNet Large | NeMo `titanet_large`, NeMo 2.7.3 cache |
| NeMo MSDD | NeMo `diar_msdd_telephonic`, NeMo 2.7.3 cache |

Exact cache locations and hashes are in `environment/model_manifest.md`.

## Activation

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

## Execution

Run the complete pipeline:

```bash
python run_phase20.py --overwrite
```

## Validation

The standalone implementation has been validated against the original Phase 20 implementation.

Using the published frozen environment, the standalone release reproduces

- VAD outputs
- WhisperX outputs
- MSDD outputs
- Speaker attribution outputs
- Published evaluation metrics

within normal floating-point precision.

Expected published ASR reproduction:

| Output | Expected value |
|---|---:|
| WER | 0.324380 |
| Aligned words | 465 |
| Whisper segments | 41 |
| Path B VAD segments | 41 |

## Evaluation and Metrics

The pipeline reports:

- VAD: precision, recall, F1, false alarm rate, miss rate.
- ASR: WER, substitutions, insertions, deletions, reference words, hypothesis words.
- Diarization: DER, JER, miss, false alarm, confusion.
- Fusion: cpWER, SA-WER, speaker accuracy, word attribution error, timeline accuracy.

Detailed scientific validation reports are in `reports/validation_*.md` and `reports/scientific_validation_report.md`.

## Outputs

| Stage | Output |
|---|---|
| DeepFilterNet | `audio/bwc_dfn.wav`, `outputs/step02a_deepfilternet.json` |
| Loudness normalization | `audio/bwc_normalized.wav`, `outputs/step02b_loudnorm.json` |
| VAD | `grounding/pathA_vad.json`, `grounding/pathB_vad.json` |
| WhisperX | `transcription/aligned_words.json` |
| NeMo MSDD | `diarization/pathA_msdd.rttm` |
| Fusion | `fusion/attributed_words.json`, `fusion/transcript.txt` |
| Metrics | `metrics/*.json` |
| Reports | `reports/*.md` |

## Reproducibility Files

| File | Contents |
|---|---|
| `environment/requirements.lock` | Exact install lock used by `install.sh` |
| `environment/requirements_frozen.txt` | Full `pip freeze --all` from the reproduced environment |
| `environment/pip_freeze.txt` | Full package freeze |
| `environment/environment_snapshot.json` | Machine-readable runtime snapshot |
| `environment/environment_hashes.json` | Hashes for environment files and model artifacts |
| `environment/package_hashes.txt` | Installed distribution RECORD hashes |
| `environment/dependency_tree.txt` | `python -m pip inspect` dependency metadata |
| `environment/model_manifest.md` | Model revisions, cache locations, hashes |
| `environment/environment_report.md` | Human-readable environment report |

## Troubleshooting

- `HF_TOKEN is required`: set `HF_TOKEN` and accept the Pyannote model terms on HuggingFace.
- `CUDA unavailable`: verify `nvidia-smi`, driver 580.159.04 or compatible, and Torch `2.8.0+cu128`.
- `pip check` reports two conflicts: expected for the historical published environment; do not resolve them.
- WhisperX produces 369 words or WER near 0.719: the environment drifted. Recreate from `environment/requirements.lock` with `--no-deps`.
- DeepFilterNet command missing: activate `.venv_phase20` and confirm `DeepFilterNet==0.5.6` and `DeepFilterLib==0.5.6`.

## Reproducing the Published Paper Results

```bash
git clone <repository-url>
cd phase20_standalone
export HF_TOKEN=hf_your_token_here
./install.sh
source .venv_phase20/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR=/tmp/marston_phase20_mpl
export NUMBA_CACHE_DIR=/tmp/marston_phase20_numba
export XDG_CACHE_HOME=$PWD/.cache
export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
export TRANSFORMERS_CACHE=$PWD/.cache/huggingface/transformers
python run_phase20.py --overwrite
```
