# Phase 4 Inferences

## Scientific Findings
1.  **Diarization is Noise-Sensitive, ASR is Noise-Resilient:** Foundation models like Whisper are built to handle noise, but embedding models like Pyannote are not. Therefore, cleaning the audio helps the *labels* more than it helps the *text*.
2.  **The Foreman Effect:** In forensic audio, the speaker's identity is often more important than the exact word. Phase 4 proved that we can "trade" transcription accuracy for attribution accuracy.

## Engineering Findings
1.  **Normalization is the Foundation:** Peak normalization to -20dB significantly stabilized the ASR performance, providing the best baseline WER (0.29) achieved to date.
2.  **Master Grounding is the Benchmark Standard:** Using a fixed set of segments is the only way to perform an "apples-to-apples" comparison of audio quality changes.

## Architectural Bottlenecks
*   **Destructive Denoising:** DFN is too aggressive. We need a "lighter" touch for ASR.
*   **The Single-Audio Path:** We are still processing the entire pipeline on a single version of the audio (either Raw or Enhanced). The results suggest we need a **Dual-Audio Path**.
