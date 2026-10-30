# Phase 8 Future Direction

## Remaining Limitations
*   **The cpWER Bottleneck:** We are stuck at ~0.65 cpWER when using DFN for ASR.
*   **Identity Swap (Officer/Suspect):** We still lack a way to "anchor" the speaker labels to specific roles.

## Next Phase: Phase 9 Production
Phase 9 will be the culmination of the BWC research. It will implement the **Dual-Branch Fusion Architecture**:
1.  **Branch 1 (Diarization):** Uses LUFS-Normalized + DFN-Enhanced audio for VAD and Oracle MSDD (to maximize identity precision).
2.  **Branch 2 (ASR):** Uses LUFS-Normalized **RAW** audio for transcription (to maximize word-level clarity).
3.  **Synthesis:** Fuses the high-precision identities from Branch 1 with the high-clarity text from Branch 2.

## Long-term Work
*   **Speaker-Role Classifier:** Building a model to automatically identify which speaker is the "Officer" based on keywords (e.g., "Dispatch," "Copy that").
*   **Dynamic Context Guard:** Adapting the ASR buffer size based on the SNR of the audio segment.
