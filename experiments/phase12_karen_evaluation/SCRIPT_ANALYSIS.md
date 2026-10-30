# Phase 12 Script Analysis

This document provides a technical walkthrough of the Python implementation used for the Karen evaluation.

## 1. `step0_audio_validation.py`
**Purpose:** Detect discrepancies between file metadata and actual audio content.
- **Mechanism:** Compares `ffprobe` header duration (reported as ~7 mins) against `librosa` decoded duration (~52 seconds).
- **Critical Logic:** Flagged a "Massive discrepancy" which alerted the system to potential file corruption or truncation. This step is vital for pipeline stability in production environments.

## 2. `step1_normalization.py`
**Purpose:** Standardize audio for acoustic models.
- **Preprocessing:** Includes a `tmp_clean.mp3` step to rebuild the MP3 stream before decoding, handling the discovered corruption.
- **Normalization:** Uses `loudnorm` filter with Integrated Loudness (I) target of -20 LUFS.
- **Quality Control:** Computes MD5/SHA256 hashes and LUFS/RMS metrics for provenance tracking.

## 3. `step2_to_6.py` (Master Processing)
**Purpose:** Executes the front-end of the pipeline (VAD through Alignment).
- **VAD (Step 2-3):** Implements a HuggingFace-based Pyannote pipeline with a custom config patch to handle newer segmentation checkpoint formats.
- **Muting (Step 4):** Generates `karen_muted.wav` by applying a binary mask from the VAD segments. This isolates speech for diagnostic purposes.
- **ASR (Step 5):** Uses `whisperx.load_model("large-v3")`. Significantly, it transcribes on **slices** (short audio buffers) rather than the full file to prevent temporal drift and context hallucinations.
- **Alignment (Step 6):** Employs `whisperx.align` (Wav2Vec2) to provide the word-level granularity required for fusion.

## 4. `step7_to_10.py` (Diarization & Fusion)
**Purpose:** Neural speaker attribution and timeline construction.
- **MSDD (Step 7):** Configures NeMo `NeuralDiarizer`. Note the use of `oracle_num_speakers=False` and `max_num_speakers=10`. This tests the unsupervised clustering capability.
- **Speaker Analysis (Step 8):** Analyzes the resulting RTTM to count unique speaker labels and their relative talk-time. Found 2 speakers (Speaker 0 and Speaker 1).
- **Fusion (Step 9):** Implements word-level maximum overlap. If a word spans multiple diarization segments, the speaker with the longest overlap is selected.
- **Timeline (Step 10):** Formats the fused JSON into a human-readable text transcript with timestamps.

## 5. `step11_to_15.py` (Evaluation)
**Purpose:** Metric generation and error analysis.
- **VAD Evaluation:** Frame-level comparison at 100fps (10ms resolution).
- **Conversational Eval:** Invokes `evaluate_conversational_timeline.py` to compute WER, DER, and cpWER.
- **Report Generation:** Aggregates findings into the final phase report.
