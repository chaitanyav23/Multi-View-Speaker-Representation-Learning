# Phase 6 Conclusions

## Summary of Results
Phase 6 was a definitive "reality check" for the project. By performing a deep audit of the hardened pipeline, the team discovered that advanced audio features like LUFS-normalization and binary masking—while logically sound—cannot substitute for a robust diarization engine. The regression in DER confirmed that Phase 5's MSDD Oracle remains the project's high-water mark.

## Hypothesis Success
The hypothesis was **rejected**. The combination of LUFS-20 and Muting was *not* enough to allow Pyannote 3.1 to match the performance of NeMo MSDD. In fact, it introduced new failure modes related to gain-pumping and identity swaps.

## Key Takeaways
*   **MSDD Oracle is mandatory** for BWC datasets.
*   **The Muting Trick** is an effective tool for reducing False Alarms but should be used sparingly (with generous temporal buffers).
*   **LUFS-20** should be reserved for the *final output* audio, not used as a preprocessing step for ML inference.

## Preservation
The **Spectrogram Visualization** and **Frame-by-Frame Audit** scripts are preserved as the standard diagnostic suite for all future phases. The **Muting Trick** logic is moved to an optional experimental branch.
