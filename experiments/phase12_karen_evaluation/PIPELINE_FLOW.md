# Phase 12 Pipeline Flow

This document details the sequential data transformations within the Karen evaluation pipeline.

## 1. Input Layer
- **Raw File:** `raw_audio/karen.mp3`
- **Condition:** Corrupted header (claims 7:06, actual 0:52).
- **Transformation:** `ffmpeg` stream-copy to `tmp_clean.mp3` -> `loudnorm` filter -> `karen_normalized.wav` (16kHz, 1-channel).

## 2. Voice Activity Detection (VAD) Layer
- **Input:** `karen_normalized.wav`
- **Process:** Pyannote VAD inference.
- **Merge Logic:**
    - Any gap between segments < 0.15s is merged.
    - This creates "Grounding Segments" used to drive all subsequent stages.
- **Output:** `karen_vad.json`, `karen_system_vad.rttm`.

## 3. Transcription & Alignment Layer
- **Input:** Audio + VAD segments.
- **Process:**
    - **Slicing:** The audio is cut into buffers based on VAD timestamps.
    - **Inference:** WhisperX Large-v3 transcribes each slice.
    - **Alignment:** Wav2Vec2 aligns characters to the original audio waveform.
- **Output:** `karen_aligned.json`.

## 4. Neural Diarization Layer
- **Input:** Audio + System VAD RTTM.
- **Constraint:** `oracle_num_speakers=False`.
- **Process:**
    - Multiscale TitaNet embedding extraction.
    - Neural Diarization (MSDD) with telephonic weights.
    - Automatic clustering based on embedding similarity.
- **Output:** `karen_diarization.rttm`.

## 5. Fusion Layer (Timeline Construction)
- **Inputs:** Aligned ASR JSON + Diarization RTTM.
- **Logic:**
    - For each word $w \in \text{ASR}$:
        - Find all $s \in \text{MSDD}$ where $s.\text{interval} \cap w.\text{interval} \neq \emptyset$.
        - Assign $w.\text{speaker} = \text{argmax}(\text{overlap\_duration}(w, s))$.
- **Output:** `karen_fused.json`.

## 6. Evaluation Layer
- **Process:**
    - Compare `karen_fused.json` against `ground_truth/karen.txt`.
    - Note: GT contains 7 minutes of text; Audio contains 52 seconds.
- **Metrics:** WER, DER, cpWER, SA-WER, WDER.
- **Diagnostics:** Analysis of speaker counts and attribution errors.
