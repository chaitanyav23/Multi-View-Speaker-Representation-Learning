# Phase 1 Error Analysis

## VAD Misses (False Negatives)
*   **Root Cause:** In the BWC dataset, "mumbled" speech or speech occurring simultaneously with loud environmental noise (sirens, wind) often fell below the VAD threshold.
*   **Silero specific:** Its energy-based sensitivity caused it to "drop" the end of sentences when the speaker's volume trailed off, leading to truncated transcriptions.

## VAD False Alarms (False Positives)
*   **Root Cause:** Highly non-stationary noise in BWC (e.g., slamming car doors, metallic clicks) triggered the VAD.
*   **Impact:** Downstream WhisperX ASR attempted to "force" words out of these noises, resulting in hallucinated high-frequency words (e.g., "you", "the", "thank you") that were not in the ground truth.

## Attribution Failures (Diarization)
*   **Overlap Errors:** Both models struggled when multiple speakers spoke simultaneously over a noisy background. The VAD would often create one large segment, but the downstream diarizer (Pyannote) would fail to distinguish the speakers due to low SNR.
*   **Grounding Misalignment:** If the VAD segment starts 100ms late, the first word of the sentence is often lost to the ASR, regardless of model quality.

## Fusion/Merging Failures
*   **Gap Errors:** Gaps between VAD segments sometimes occurred in the middle of a continuous sentence. If the gap was >500ms, the pipeline treated them as distinct conversational turns, breaking the narrative flow.
