# Phase 3 Evaluation Analysis

## The "Identity Collapse" Metric (DER vs. cpWER)

### 1. The DER Trap
*   In Phase 3, it was discovered that a "good" **DER** can be deceptive.
*   **Case:** A 2-speaker audio where the model assigns everyone to `SPEAKER_00`.
*   **Result:** The "Miss" rate remains high, but "Confusion" is 0. If the speakers are unevenly distributed (e.g., 90% speaker A, 10% speaker B), the DER might look low (e.g., 15%), despite the model failing to identify the second person entirely.

### 2. The cpWER Reveal
*   **cpWER** (Concatenated minimum-Permutation Word Error Rate) is the "truth" metric. 
*   If Identity Collapse occurs, the cpWER skyrockets (often >0.50) because the ground truth text for Speaker B has no predicted counterpart.
*   Phase 3 evaluation focused on using cpWER to detect when the diarizer failed to "separate" the conversation.

## MSDD Specific Metrics
*   **Multiscale weights:** Analyzing how different window sizes affect the model's ability to handle short interjections (e.g., "Yeah", "Okay") vs. long monologues.
*   **Sigmoid Threshold:** Measuring the tradeoff between "Diarization Miss" (threshold too high) and "Diarization False Alarm" (threshold too low, leading to speakers being "detected" in silence).
