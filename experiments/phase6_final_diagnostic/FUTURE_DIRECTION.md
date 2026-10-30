# Phase 6 Future Direction

## Remaining Limitations
*   **Identity Flipper:** We still have a 50/50 chance of mis-labeling the speakers in a two-person conversation.
*   **Segment Boundary Fragility:** Tight VAD segments are still "clipping" important word data.

## Next Phase: Phase 7 Optimal
Phase 7 will focus on building the **"Optimal" Pipeline**. This will synthesize the "Winners" from all previous phases:
1.  Peak Normalization (from Phase 4)
2.  Pyannote 3.1 Master VAD with temporal padding (to fix the clipping from Phase 6)
3.  Oracle MSDD (from Phase 5)
4.  Manual Sliced WhisperX ASR (from Phase 6)

## Long-term Work
*   **Identity Seeding:** Pre-loading the pipeline with a 3-second embedding of the "Officer" to force the diarizer to always assign that identity correctly.
*   **Contextual VAD:** Using a large language model to filter out VAD segments that contain only repetitive noise patterns.
