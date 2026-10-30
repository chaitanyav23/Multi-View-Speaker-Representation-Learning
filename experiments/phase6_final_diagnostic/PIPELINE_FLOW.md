# Phase 6 Pipeline Flow

## Sequential Processing Steps

1.  **Loudness Normalization (Loudnorm)**
    *   Target: -20 LUFS.
    *   Result: Standardized volume profile.

2.  **Audio Masking (The "Muting Trick")**
    *   Input: LUFS audio + Pyannote VAD segments.
    *   Operation: Zero-out non-speech samples.
    *   Result: Cleaned signal for the diarizer.

3.  **Experimental Branching (Denoising)**
    *   **Branch A:** Baseline (Muted only).
    *   **Branch B:** DFN (Muted + Denoised).
    *   **Branch C:** SpeechBrain (Muted + Denoised).

4.  **Inference (ASR & Diarization)**
    *   ASR: Manual slice transcription on raw (muted) audio.
    *   Diarization: Pyannote 3.1 run on muted audio.

5.  **Audit Data Generation**
    *   `error_audit/`: Timestamps and labels for every single-second segment of audio, categorized by error type.

## Data Artifacts
*   `audio_preprocessed/`: Normalized and potentially denoised .wav files.
*   `visualizations/`: Mel-spectrograms showing VAD activity.
*   `v2/`: Refined audit scripts and metrics.
