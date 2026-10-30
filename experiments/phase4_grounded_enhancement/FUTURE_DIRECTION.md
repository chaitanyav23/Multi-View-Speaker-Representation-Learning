# Phase 4 Future Direction

## Remaining Limitations
*   **Dual-Audio Implementation:** We currently cannot feed different audio versions to the ASR and Diarizer in a single pass.
*   **MSDD Still Missing:** Phase 4 used Pyannote 3.1 for diarization. We still haven't fixed the MSDD timeout issue discovered in Phase 3.

## Next Phase: Phase 5 Audit
Phase 5 will return to the **MSDD problem**. Armed with the knowledge that external grounding and clean audio help diarization, we will attempt to "Force-Ground" MSDD by injecting our Master Grounding segments directly as Oracle RTTMs.

## Long-term Work
*   **Soft Fusion:** Exploring ways to weighted-average the raw and enhanced audio to find a "sweet spot" for ASR.
*   **Automatic Gain Control (AGC):** Moving beyond simple Peak Normalization to RMS-based AGC for even more consistent volume profiles.
