# Architecture: Phase 17 (No-Normalization)

## Pipeline Overview
The Phase 17 architecture is a sequential processing pipeline designed for high-fidelity evaluation of BWC audio. It deviates from the production-standard pipeline by omitting the `LoudNorm` stage to preserve original signal dynamics.

## Component Breakdown

### 1. Audio Pre-processing (Step 0)
- **Tool:** FFmpeg.
- **Operations:**
    - Resampling to 16,000 Hz.
    - Downmixing to Single Channel (Mono).
    - Format conversion to WAV (PCM 16-bit).
- **Rationale:** Standardizes input for neural models without modifying amplitude ratios or dynamic range.

### 2. Voice Activity Detection (Step 2)
- **Model:** Pyannote 3.1 (`pyannote/voice-activity-detection`).
- **Configuration:**
    - `merge_gap`: 0.15s (segments closer than this are fused).
    - `min_duration`: 0.1s (minimum segment length).
- **Execution:** Generates a "grounding" JSON containing binary speech/non-speech timestamps.

### 3. Automatic Speech Recognition (Step 4)
- **Engine:** WhisperX.
- **Model:** `large-v2`.
- **Optimization:** Utilizes VAD-guided transcription to reduce hallucination in silent/noisy regions.
- **Output:** Word-level timestamps with alignment scores.

### 4. Neural Diarization (Step 5)
- **Model:** NeMo Multi-Scale Diarization Decoder (MSDD).
- **Method:** 
    - Uses the grounding VAD from Step 2.
    - Performs speaker embedding extraction and clustering.
    - Refines boundaries using the multi-scale decoder.
- **Output:** RTTM (Rich Transcription Time Marked) file.

### 5. Fusion and Attribution (Step 6)
- **Logic:**
    - Assigns speakers to individual WhisperX words based on temporal overlap with MSDD segments.
    - **Fallback:** If no direct overlap, assigns to the nearest speaker turn within a 0.5s tolerance.
- **Clustering:** Groups words into conversational blocks based on speaker continuity and a 2.0s maximum inter-word gap.

## Evaluation Framework
The architecture includes a comprehensive diagnostic suite (Steps 8-14) that compares:
- **Lexical Accuracy:** WER (Word Error Rate).
- **Temporal Accuracy:** DER (Diarization Error Rate), Miss, False Alarm, Confusion.
- **Conversational Accuracy:** WDER (Word Diarization Error Rate), cpWER (Concatenated Permutation WER), SA-WER (Speaker-Attributed WER).
