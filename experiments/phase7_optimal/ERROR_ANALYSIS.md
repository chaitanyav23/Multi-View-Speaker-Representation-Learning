# Phase 7 Error Analysis

## Artifact Substitution (ASR)
*   **Symptom:** Common words being replaced by sound-alike non-words.
*   **Root Cause:** DeepFilterNet smoothing over high-frequency phonemes. This confirms the findings of Phase 2 but at a larger scale with more complex orchestration.

## Padding Overlap
*   **Symptom:** The end of one person's turn "bleeding" into the start of the next person's turn.
*   **Root Cause:** The 0.1s VAD padding, while good for recall, sometimes caused segments to overlap in time. The `merge_gap=0` logic in `run_soft_vad` combined these overlapping regions, creating long segments that confused the MSDD decoder.

## Diarization "Drift"
*   **Symptom:** Speaker IDs flipping during a segment.
*   **Root Cause:** Because the audio was DFN-enhanced, the speaker embeddings (TitaNet) were "flatter" and less distinct. MSDD's sigmoid threshold of 0.7 was sometimes too strict for these flattened embeddings, leading to "UNK" assignments.

## Context Guard Failures
*   **Symptom:** WhisperX repeating the same phrase (e.g., "thank you thank you") when the segment contained only static.
*   **Root Cause:** Using `condition_on_previous_text=True` on hallucinated False Alarm segments caused the model to use the previous text as a strong prior for the current noise.
