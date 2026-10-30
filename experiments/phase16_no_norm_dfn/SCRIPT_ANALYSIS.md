# Script Analysis: Phase 16

This document provides a detailed technical breakdown of the scripts utilized in Phase 16.

## Orchestration
### `run_phase16.py`
- **Purpose:** Central coordinator for the entire pipeline.
- **Workflow:** 
  1. Sets up directory structure.
  2. Executes Step 0 (FFmpeg) and Step 1 (DeepFilterNet).
  3. Sequentially invokes `step2_*.py` through `step14_*.py`.
- **Key Logic:** Handles file naming conventions (e.g., renaming DFN output) and ensures dependency satisfaction between steps.

## Pipeline Steps
### `step2_vad_grounding.py`
- **Technique:** Pyannote 3.1 VAD.
- **Innovation:** Skips normalization. Runs directly on DFN-enhanced audio.
- **Output:** `grounding/karen_grounding.json` containing speech segments.

### `step3_oracle_rttm.py`
- **Purpose:** Converts VAD segments into an RTTM format compatible with NeMo MSDD.
- **Detail:** Labels all detected speech as a single generic "speaker" to satisfy the `oracle_vad` requirement of the MSDD configuration while using the system's actual VAD boundaries.

### `step4_whisperx.py`
- **Mechanism:** Slices audio by VAD segments and transcribes using Whisper v3 Large.
- **Alignment:** Uses WhisperX `align` with Wav2Vec2 to produce word-level timestamps.
- **Params:** `compute_type="float16"`, `language="en"`, `condition_on_previous_text=False`.

### `step5_msdd.py`
- **Model:** NeMo Multi-scale Speaker Diarization Decoder.
- **Config:** Uses `diar_msdd_telephonic` checkpoint with TitaNet-L embeddings.
- **Multi-scale Windows:** Configured with 5 scale levels to capture fine-grained speaker changes.

### `step6_fusion.py`
- **Logic:** Speaker attribution.
- **Details:** Maps MSDD speakers to aligned words. Implements segment grouping with a 2-second gap threshold and speaker-change triggers.

## Evaluation and Comparison
### `step8_vad_eval.py`
- **Metrics:** Accuracy, Precision, Recall, F1, Miss, FA.
- **Diagnostics:** Generates `miss_regions.json` and `fa_regions.json` for fine-grained error localization.

### `step10_diar_eval.py`
- **Library:** `pyannote.metrics`.
- **Primary Metric:** DER (Diarization Error Rate).

### `step12_13_conversational_eval.py`
- **Metrics:** `cpWER` (concatenated minimum permutation Word Error Rate), `WDER` (Word Diarization Error Rate), `SA-WER` (Speaker-Attributed WER).
- **Function:** Evaluates the joint performance of ASR and Diarization.

### `step14_comparison.py`
- **Function:** Contrasts Phase 16 results against Phase 11A (BWC) baseline.
- **Output:** `comparisons/karen_vs_bwc.csv`.
