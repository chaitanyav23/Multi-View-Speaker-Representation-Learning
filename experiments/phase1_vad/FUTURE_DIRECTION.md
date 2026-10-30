# Phase 1 Future Direction

## Remaining Limitations
1.  **Noise Sensitivity:** Pyannote's high FA rate in BWC indicates that it cannot distinguish between meaningful background speech (e.g., a distantly speaking suspect) and structured noise (e.g., rhythmic static).
2.  **Lack of Audio Enhancement:** Running VAD on raw, unnormalized audio leads to inconsistent performance when the speaker moves relative to the microphone.

## Next Phase: Phase 2 Enhancement
Phase 2 will address these limitations by introducing **Audio Enhancement** (Denoising and Normalization) *before* the VAD stage. Specifically, models like DeepFilterNet and SpeechBrain will be tested to see if they can reduce the False Alarms observed in Pyannote's output.

## Long-term Work
*   **Multi-Model Fusion:** Exploring a combination of Silero (for speed/safety) and Pyannote (for precision).
*   **Dynamic Thresholding:** Adjusting VAD sensitivity based on the estimated SNR of the audio file.
