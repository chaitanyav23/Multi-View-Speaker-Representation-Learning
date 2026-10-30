# Script Analysis: Phase 18 (Tich Karen)

The Phase 18 implementation is decomposed into 10 primary execution steps and several diagnostic scripts. This document provides a technical breakdown of each component.

## Core Pipeline Scripts

### `step1_preprocess.py`
- **Function**: Audio standardization.
- **Logic**: Loads audio using `librosa`, resamples to 16kHz, and applies RMS normalization to -20dB.
- **Output**: `audio/rich_karen_normalized.wav`.

### `step2_process_gt.py`
- **Function**: Ground Truth (GT) normalization.
- **Logic**: Parses raw `rich_karen.txt` to extract speaker labels and timestamps, converting them into a standardized JSON format for evaluation.
- **Output**: `manifests/gt_segments.json`.

### `step3_vad_grounding.py`
- **Function**: Automated VAD execution.
- **Logic**: Instantiates the Pyannote 3.1 pipeline. It modifies the segmentation parameters dynamically to ensure optimal sensitivity.
- **Output**: `grounding/vad_segments.json`.

### `step4_oracle_rttm.py`
- **Function**: Upper-bound baseline generation.
- **Logic**: Converts the Ground Truth segments into a standard RTTM format. This allows the system to run in "Oracle" mode (perfect VAD) to isolate diarizer performance.
- **Output**: `grounded/oracle.rttm`.

### `step5_whisperx.py`
- **Function**: ASR and Alignment.
- **Logic**: Runs Whisper `large-v3`. It utilizes the VAD segments from Step 3 (or GT in Oracle mode) to focus transcription on active speech regions. It then performs forced alignment for word-level precision.
- **Output**: `outputs/asr_aligned.json`.

### `step6_msdd.py`
- **Function**: Multi-scale Neural Diarization.
- **Logic**: Uses NeMo's `NeuralDiarizer`. It generates a manifest file pointing to the audio and VAD segments, then executes the MSDD model with `titanet_large` embeddings.
- **Output**: `diarization/msdd_output.rttm`.

### `step7_fusion.py`
- **Function**: Modality Fusion.
- **Logic**: Merges the temporal boundaries of WhisperX words with the speaker labels from the MSDD RTTM. It handles conflicts using a "majority-temporal-overlap" strategy.
- **Output**: `outputs/final_transcript.json` and `final_transcript.txt`.

### `step8_vad_eval.py`
- **Function**: VAD Metric Calculation.
- **Logic**: Compares `vad_segments.json` against `gt_segments.json`. Calculates Precision, Recall, and F1 at a 100ms tolerance.

### `step9_asr_eval.py`
- **Function**: ASR Metric Calculation.
- **Logic**: Calculates Word Error Rate (WER) and Word Information Loss (WIL) using `jiwer`. It normalizes text (removing punctuation and casing) before comparison.

### `step10_diar_eval.py`
- **Function**: Diarization Metric Calculation.
- **Logic**: Uses `pyannote.metrics` to calculate DER, Miss, False Alarm, and Confusion.

## Diagnostic Scripts

### `analyze_embeddings.py`
- **Function**: Speaker Embedding Visualization.
- **Logic**: Extracts d-vectors from the `titanet_large` model, performs dimensionality reduction (PCA/UMAP), and generates scatter plots colored by predicted speaker ID.

### `compute_conv_metrics.py`
- **Function**: Conversational Dynamics Analysis.
- **Logic**: Calculates metrics such as Inter-Speaker Silence, Overlap Percentage, and Turn-Taking frequency.
