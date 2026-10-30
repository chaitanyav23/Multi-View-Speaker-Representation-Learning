# Phase 10 Error Analysis

## The "Cross-Talk Erasure"
*   **Symptom:** Entire sentences spoken by Speaker B are attributed to Speaker A, even though the words were correctly transcribed.
*   **Root Cause:** WhisperX's internal segmentation algorithm grouped the overlapping sentences into a single temporal block. The `custom_fusion.py` script's majority vote logic then assigned the entire block to the dominant speaker.

## The Overlap Threshold
*   **Symptom:** Words assigned to `UNK` (Unknown).
*   **Root Cause:** If a word fell outside of the MSDD Oracle segments by more than the `tolerance` (0.5s), it was orphaned. This happened frequently at the trailing ends of long sentences where the VAD gate closed slightly before the speaker finished articulating.

## DeepFilterNet "Feature Erasure"
*   Phase 10B confirmed that DFN not only damages the high-frequency components needed for WhisperX, but it also damages the mid-frequency harmonics that NeMo's TitaNet uses to distinguish individual voices in noisy environments.
