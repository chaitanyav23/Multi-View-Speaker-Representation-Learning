# Phase 15: Script Analysis

## Primary Controller
### `run_phase15.py`
The orchestration script that manages the sequential execution of the pipeline.
- **Key Logic:** It encapsulates Step 0 (Audio Repair) directly using `subprocess` calls to `ffmpeg`. It then loops through a list of Python scripts (Steps 2-14), ensuring each completes before the next begins.
- **Role:** Handles directory initialization and centralized execution logging.

## Pipeline Components
### `step2_vad_grounding.py`
- **Model:** Pyannote 3.1.
- **Logic:** Loads the pretrained VAD pipeline from HuggingFace. It processes the audio to find speech regions and applies a `merge_gap` of 0.15s to bridge tiny pauses.
- **Output:** `karen_grounding.json`.

### `step3_oracle_rttm.py`
- **Logic:** Converts the VAD segments into a standard RTTM format. 
- **Purpose:** This file is required by NeMo MSDD. In this pipeline, it serves as the "proposal" for where speech exists, effectively forcing MSDD to only look at those regions.

### `step4_whisperx.py`
- **Model:** Whisper `large-v3`.
- **Logic:** Performs transcription segment-by-segment. This is more robust than full-file transcription as it avoids some hallucination issues at high timestamps.
- **Alignment:** Uses `whisperx.align` to find the exact start/end of each word.

### `step5_msdd.py`
- **Model:** NeMo `NeuralDiarizer` with MSDD.
- **Logic:** Configures the multiscale speaker embedding extractor. It uses the VAD-generated RTTM as `oracle_vad`. 
- **Hyper-parameters:** Uses `titanet_large` for embeddings and a 0.25s collar for evaluation during inference.

### `step6_fusion.py`
- **Logic:** The "Time-Overlap Attribution" engine. It takes the word list from Step 4 and the RTTM labels from Step 5. It calculates which speaker was active during each word's time interval.
- **Fallback:** If no speaker is active (gap in diarization), it looks for the nearest speaker within 0.5s.

## Evaluation Scripts
### `step8_vad_eval.py` to `step12_13_conversational_eval.py`
- **Logic:** These scripts compare the system outputs against processed ground truth.
- **Metrics:** WER (JIWER), DER (pyannote.metrics), and custom conversational metrics like cpWER (Concatenated Permutation WER) and WDER (Word Diarization Error Rate).

### `step14_comparison.py`
- **Logic:** Aggregates current results and compares them against a hardcoded baseline (BWC Phase 11A).
- **Purpose:** Quantifies the delta in performance to determine if the phase's changes (removing normalization) resulted in a net gain or loss.
