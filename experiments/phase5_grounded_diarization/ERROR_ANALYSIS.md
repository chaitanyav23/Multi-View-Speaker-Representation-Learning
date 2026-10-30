# Phase 5 Error Analysis

## Remaining "Confusion" Errors
*   **Root Cause:** Even with Oracle VAD, MSDD occasionally swaps speaker IDs during moments of extreme overlapping speech (cross-talk).
*   **Symptom:** In the diagnostic clips, two speakers talking simultaneously are often both assigned to the dominant speaker's ID.

## Midpoint Misalignment
*   **Root Cause:** Because the ASR (WhisperX) and Diarization (MSDD) are run on different branches, there is a small "temporal jitter" (approx 50-100ms) between word boundaries and speaker segments.
*   **Impact:** Words occurring at the exact millisecond of a turn-swap can be "captured" by the wrong speaker's ID.

## High False Alarm (VAD Propagated)
*   Because Phase 5 uses a "Master Grounding" based on Pyannote 3.1, it inherits the **22% False Alarm rate** on BWC.
*   **Impact:** MSDD is forced to "diarize" segments that contain only noise (sirens), leading to hallucinated speakers being assigned to hallucinated words.

## TitaNet Embedding Drift
*   In one forensic case, the suspect's voice became extremely distorted due to a physical struggle. MSDD correctly separated the suspect initially but created a *third* speaker ID for the distorted section, failing to realize it was the same individual.
