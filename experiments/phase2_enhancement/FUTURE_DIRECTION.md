# Phase 2 Future Direction

## Remaining Limitations
*   **Destructive Preprocessing:** We still lack a way to handle extreme noise without harming transcription accuracy.
*   **Diarization Sensitivity:** Diarization remains the most fragile part of the pipeline, especially when audio quality is low.

## Next Phase: Phase 3 Diarization
Phase 3 will shift focus from audio enhancement to **Diarization Optimization**. Instead of trying to "fix" the audio, we will look for more robust diarization engines (like Pyannote 3.1 and MSDD) that can handle raw, noisy signals more effectively.

## Long-term Work
*   **Dual-Branch Processing:** Implementing a "Forked Pipeline" where VAD runs on DFN-enhanced audio to get better grounding, but ASR runs on the raw audio to get better transcription.
*   **Loudness Normalization:** Testing simple Peak or RMS normalization (LUFS) which is non-destructive but can fix gain-related issues for the models.
