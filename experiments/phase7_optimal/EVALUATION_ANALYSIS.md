# Phase 7 Evaluation Analysis

## The "Total Context" Evaluation

### 1. Contextual ASR (condition_on_previous_text)
Phase 7 tested whether WhisperX's ability to see previous text would lower WER.
*   **Finding:** On DFN-enhanced audio, contextual prompting **failed to lower WER** significantly. The neural artifacts were too disruptive for the prompt to overcome.
*   **Finding:** However, the **readability** of the transcript improved (fewer mid-sentence breaks).

### 2. VAD Padding Impact
*   **Recall:** Improved significantly (from 0.70 to 0.77).
*   **Finding:** Adding 100ms of padding to the VAD segments effectively "recovered" the start of sentences that were previously being clipped. This is critical for forensic accuracy where the first word of a command (e.g., "STOP!") is vital.

## The cpWER Paradox
Phase 7 resulted in the highest **stability** but a mediocre **cpWER (0.65)**. This Paradox is explained by the destructive nature of DeepFilterNet on Whisper's internal attention mechanisms. Phase 7 effectively optimized the *wrong version* of the audio for the ASR stage.
