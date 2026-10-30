# Phase 10 Evaluation Analysis

## The "Custom Fusion" Regression

### 1. The Goal of Custom Fusion
The goal of `custom_fusion.py` was to handle edge cases where a single word spanned a speaker transition boundary. By calculating the exact duration of overlap, it theoretically provided a more accurate assignment than the simple midpoint check from Phase 5.

### 2. The Flaw
The evaluation revealed a severe cpWER regression (0.50) compared to the Phase 5 baseline (0.45). Analysis of the code and the resulting text files revealed that the `Majority Vote` logic at the end of `custom_fusion.py` was responsible.
*   **Mechanism:** WhisperX often groups several logical sentences into a single temporal "segment."
*   **Result:** If Speaker A talks for 3 seconds, and Speaker B interrupts for 1 second, WhisperX might group them. The majority vote sees 75% Speaker A, and assigns the *entire block* to Speaker A, effectively erasing Speaker B's turn.

### 3. Dual-Branch Discrepancy
Phase 10B tested the hypothesis that DFN audio would improve ASR clarity. However, the evaluation showed a WER of 0.37 for DFN vs 0.32 for Raw. This re-confirmed the Phase 7 findings: Neural denoising actively destroys the acoustic features Whisper needs, regardless of the pipeline architecture.
