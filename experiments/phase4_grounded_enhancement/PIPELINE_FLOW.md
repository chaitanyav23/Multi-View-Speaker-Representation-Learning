# Phase 4 Pipeline Flow

## Sequential Processing Steps

1.  **Normalization (Librosa)**
    *   Target: -20dB Peak.
    *   Result: `audio_normalized/`

2.  **Selective Enhancement (Branching)**
    *   **Baseline:** Skip enhancement.
    *   **DFN Path:** Run DeepFilterNet3 on normalized audio.
    *   **SB Path:** Run SpeechBrain MetricGAN+ on normalized audio.

3.  **Grounded Processing (Hardened Pipeline)**
    *   **Fixed Input:** `grounding_master/*.json` (VAD segments).
    *   **ASR:** WhisperX Large-v3 transcribes the segments.
    *   **Diarization:** Pyannote 3.1 assigns speakers to the transcribed words.

4.  **Convergent Evaluation**
    *   Compute metrics for all three branches using the same ground truth.

## Pipeline Improvements
*   **Decoupled VAD:** By using a fixed Master Grounding, Phase 4 eliminated the variable of VAD Miss/FA from the enhancement study. This revealed that enhancement actually **improves diarization accuracy (WDER)** while **harming transcription accuracy (WER)**.
