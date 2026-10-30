# Phase 5 Future Direction

## Remaining Limitations
*   **VAD False Alarms:** The high FA rate (22%) remains the biggest remaining bottleneck. Hallucinated segments confuse the diarizer and bloat the transcript.
*   **Complexity:** The pipeline now has five distinct stages (Normalize, VAD, ASR, Diarize, Fuse), making it difficult to maintain and monitor.

## Next Phase: Phase 6 Final Diagnostic
Phase 6 will focus on a **Final Diagnostic Audit**. We will perform a frame-by-frame error analysis of the Phase 5 output to identify the exact cause of every remaining error. We will also explore "Ablation Studies" to see if any stages can be simplified without losing accuracy.

## Long-term Work
*   **Custom VAD Training:** Training a small binary classifier to distinguish between "Police Siren" and "Human Voice" to reduce False Alarms.
*   **End-to-End Orchestrator:** Developing a single Python class that manages the entire Oracle MSDD flow to reduce file I/O overhead.
