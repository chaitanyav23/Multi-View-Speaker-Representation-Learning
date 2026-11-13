# Reproducibility Guide: ConvoFuse CLI & Research Release

This guide provides the exact environment specifications, model verification commands, CLI invocations, output layout, and automated audit checks required to reproduce the **ConvoFuse** final research release in `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`.

---

## 1. Environment & Runtime Prerequisites

### 1.1 Hardware & OS Validated Configuration
- **OS**: Linux (Ubuntu 22.04 LTS, kernel `6.8.0-x86_64`)
- **GPU**: NVIDIA GeForce RTX 4090 (24.5 GB VRAM, CUDA Driver `595.71.05`)
- **Repository Root**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`

### 1.2 Python Environment
All CLI and test commands run inside the frozen virtual environment located at:

```bash
./research_stages/.venv_phase20/bin/python
```

Verify the Python environment:

```bash
./research_stages/.venv_phase20/bin/python --version
./research_stages/.venv_phase20/bin/python -c "import numpy, scipy, soundfile, pytest; print('Environment OK')"
```

### 1.3 Local Ollama & `DiarizationLM-Gemma-4-E4B-v1` Verification
ConvoFuse uses the local Ollama server (`http://127.0.0.1:11434`) serving `diarizationlm-gemma:latest` (quantized `Q4_K_M` GGUF derived from `diarizers-community/DiarizationLM-Gemma-4-E4B-v1`).

Verify that Ollama is running and the model is loaded:

```bash
curl -s http://127.0.0.1:11434/api/version
ollama list | grep diarizationlm-gemma
```

Expected output:
```text
{"version":"0.32.6"}
diarizationlm-gemma:latest    441ab67c5d8f    5.3 GB
```

---

## 2. Reproducing the CLI Runs

All commands should be executed from `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning` with `PYTHONPATH=.`.

### 2.1 Inspect CLI Help & Options

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli --help
```

### 2.2 Run a Single Benchmark Recording (`FROZEN_INPUT_REPLAY`)

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --recording R01_sara_sharif \
  --condition auto \
  --output-dir convofuse_outputs/single_r01_check
```

- `--condition auto` (default) selects the best validated multi-view diarization input per recording (`Stage 5B.4 E20` for `R01_sara_sharif` and `Stage 5A.1 Condition J` for `R02`–`R08`).
- You can also explicitly specify `--condition condition_j` to force Stage 5A.1 Condition J on `R01_sara_sharif`.

### 2.3 Reproduce the Complete `R01`–`R08` Final Release Package

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --all-recordings \
  --condition auto \
  --output-dir convofuse_outputs/final_release
```

> [!NOTE]
> **Warm vs. Cold Ollama Cache Note (`R01_sara_sharif`)**:
> If `diarizationlm-gemma:latest` is loaded from cold disk into GPU memory on the first request (`runtime ~ 6.1s`), prompt batch evaluation leaves word 29 (`"Alright."` at `9.409s–9.850s` on `R01`) under `SPEAKER_02` (`0` accepted changes, `11` turns). When the model is already warm in Ollama GPU memory (`runtime ~ 3.5s`), it deterministically transfers word 29 to `SPEAKER_01` (`1` accepted change, `13` turns). Both runs yield identical `TS-WDER = 20.08%`, `Lex-WDER = 19.38%`, and `cpWER = 52.30%`. Running a quick single-recording warmup (`--recording R01_sara_sharif`) before `--all-recordings` reproduces the exact warm `13`-turn `R01` output in `convofuse_outputs/final_release/`.

### 2.4 Run the Unconstrained Ablation (`--unconstrained`)

To evaluate raw `DiarizationLM-Gemma` speaker transfer without the `LG-TPST` locality gate (while retaining word/timestamp immutability and `UNK` isolation):

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --recording R04_piers_robinson \
  --unconstrained \
  --output-dir convofuse_outputs/ablation_unconstrained
```

### 2.5 Run Custom Recordings (`--mode custom_inputs` / `FULL_PROCESSING`)

For new recordings outside `R01`–`R08`, `--mode custom_inputs` runs the full ConvoFuse canonical audio verification, word-to-speaker attribution, overlap annotation, `LG-TPST` generative refinement, and chronological transcript reconstruction on user-supplied audio, aligned ASR words JSON, and RTTM diarization files:

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --mode custom_inputs \
  --recording CUSTOM_REC_01 \
  --audio-file /path/to/audio.wav \
  --asr-words-json /path/to/words.json \
  --rttm-file /path/to/diarization.rttm \
  --overlap-npz /path/to/optional_overlap.npz \
  --output-dir convofuse_outputs/custom_run
```

- **Scope & Limitations of Custom Mode**: `--mode custom_inputs` requires pre-extracted `--asr-words-json` and `--rttm-file` (plus `--audio-file`). End-to-end raw acoustic extraction (running `DeepFilterNet3`, `WhisperX Large-v3`, `TitaNet-Large`, and `WeSpeaker ResNet293-LM` from scratch on raw audio) is handled by the upstream stage runners in [`research_stages/stage1_data_prep/`](research_stages/stage1_data_prep/), [`research_stages/stage2_baseline/`](research_stages/stage2_baseline/), [`research_stages/stage3_dual_encoder/`](research_stages/stage3_dual_encoder/), and [`research_stages/stage5b/stage5b_4_fresh_diarization/`](research_stages/stage5b/stage5b_4_fresh_diarization/).

---

## 3. Output Directory Layout (`convofuse_outputs/final_release/`)

Running `--all-recordings --output-dir convofuse_outputs/final_release` produces the following deterministic structure:

```text
convofuse_outputs/final_release/
├── CORPUS_SUMMARY.json                         # Macro & per-recording WDER, cpWER, safeguard counts, integrity flags
├── CORPUS_INDEX.json                           # Recording index mapping R01-R08 to all 8 per-recording artifact paths
├── RELEASE_MANIFEST.json                       # Git commit, CLI args, input SHA-256s, and output file SHA-256s/sizes
├── R01_sara_sharif/
│   ├── R01_sara_sharif__transcript.txt         # Primary human-readable chronological conversation transcript
│   ├── R01_sara_sharif__transcript.json        # Primary full machine-readable ConvoFuseTranscriptBundle JSON
│   ├── R01_sara_sharif__words.json             # Word-level attribution, timestamps, overlap probs & LG-TPST status
│   ├── R01_sara_sharif__turns.json             # Baseline & refined chronological speaker turns
│   ├── R01_sara_sharif__speakers.json          # Per-speaker duration, turn/word counts & overlap summary
│   ├── R01_sara_sharif__comparison.json        # Baseline vs refined word diff & safeguard reason breakdown
│   ├── R01_sara_sharif__processing_manifest.json # Per-recording provenance, input SHA-256s & runtime telemetry
│   ├── R01_sara_sharif__evaluation_report.json # Post-inference TS-WDER, Lex-WDER, cpWER & oracle correction audit
│   └── raw_llm_completions/                    # Raw Ollama prompt/completion JSON logs per segment window
├── R02_dennis_skinner/ ...
├── R03_disabled_parking/ ...
├── R04_piers_robinson/ ...
├── R05_southwest_pilot/ ...
├── R06_rich_karen/ ...
├── R07_bwc_cheap_shot/ ...
└── R08_uk_parliament/ ...
```

---

## 4. Automated Verification & Smoke Test Suite

### 4.1 Run the ConvoFuse & Research Stage Pytest Suite

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/pytest \
  convofuse/tests/ \
  research_stages/stage6_generative_refinement/tests/ \
  research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/tests/ \
  research_stages/stage5b/stage5b_4_fresh_diarization/tests/ -v
```

### 4.2 Verify Cryptographic Provenance & Ground-Truth Isolation

Every `<rec_id>__processing_manifest.json` and `<rec_id>__evaluation_report.json` can be verified with a one-liner:

```bash
./research_stages/.venv_phase20/bin/python -c "
import json
from pathlib import Path
summary = json.loads(Path('convofuse_outputs/final_release/CORPUS_SUMMARY.json').read_text())
m = summary['macro_averages']
assert summary['recording_count'] == 8
assert m['all_word_sequences_preserved'] is True
assert m['all_timestamps_preserved'] is True
assert m['all_unk_invariance_verified'] is True
assert m['all_frozen_inputs_sha256_verified'] is True
print('All 8 recordings verified: 100% word, timestamp, UNK, and SHA-256 integrity.')
"
```

### 4.3 Optional Demonstration Interface (`app/`)

The [`app/`](app/) directory contains an optional, non-deliverable FastAPI/Streamlit interface ([`app/README.md`](app/README.md)). It is not required for reproducing any research result or CLI output, but its unit test suite can be verified independently using:

```bash
cd app && /home/priyanka/anaconda3/envs/cs698b_phase20/bin/pytest tests/ -v
```
