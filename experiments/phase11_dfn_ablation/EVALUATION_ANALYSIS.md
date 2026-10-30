# Phase 11 Evaluation Analysis

## The "Perfect Isolation" Test (11C vs 11A)
The most mathematically significant comparison in this phase is between Pipeline 11C and Pipeline 11A. 
*   **The Setup:** Both pipelines were fed the exact same VAD timestamps. Both pipelines were assigned the exact same speaker labels from the exact same MSDD run. 
*   **The Variable:** The only difference was that 11A's WhisperX instance listened to the LUFS-Normalized audio, while 11C's WhisperX instance listened to the DFN-Enhanced audio.

## The Verdict on Neural Denoising
*   **Deletions Spike:** In 11A, WhisperX had 46 deleted words. In 11C, it had 93 deleted words. DFN literally erased nearly 50 words from the transcript that the ASR could otherwise hear.
*   **WER Degradation:** WER degraded from 32.4% (11A) to 37.8% (11C).
*   **Conclusion:** DFN intrinsically damages the acoustic features that transformer-based ASR engines rely on. It is not an artifact of bad VAD or bad fusion; the denoising itself is destructive.

## VAD "Shattering" (11B vs 11A)
Phase 11 also evaluated if DFN helps the VAD stage.
*   **Finding:** DFN caused the Pyannote VAD to generate 70 separate micro-segments, compared to 41 continuous segments on Normalized audio. 
*   **Reason:** DFN aggressively zeros out the natural pauses between words. Pyannote interprets these deep silences as the end of a turn. This "shattering" effect fragments the conversational context and artificially inflates the Diarization Error Rate (DER) during fusion.
