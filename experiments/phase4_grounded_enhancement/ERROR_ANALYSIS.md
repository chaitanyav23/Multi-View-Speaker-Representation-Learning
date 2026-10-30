# Phase 4 Error Analysis

## The "Silent Speaker" Error
*   **Symptom:** A speaker is correctly identified in some segments but "vanishes" in others.
*   **Root Cause:** DeepFilterNet occasionally identifies the quiet voice of a secondary speaker as "background noise" and suppresses it entirely. Because the word is gone, it cannot be transcribed or diarized, increasing the **Miss** rate.

## Hallucinated Identity Collapse
*   **Symptom:** Diarization metrics improve (WDER 0.05), but the total word count drops.
*   **Root Cause:** DFN "fixes" identity collapse by simply deleting the speech of the speakers it can't distinguish. The metrics look better because "Confusion" is low, but the transcript is missing data.

## Consonant Slurring
*   As seen in Phase 2, DFN-enhanced audio makes it difficult for WhisperX to distinguish between similar-sounding words (e.g., "fight" vs "might"), leading to high **Substitution** errors.

## Artifact Interaction
*   WhisperX's alignment stage (Wav2Vec2) was found to be more robust than its transcription stage (Large-v3). Even when the words were slightly robotic, the aligner was still able to place them relatively accurately in the timeline.
