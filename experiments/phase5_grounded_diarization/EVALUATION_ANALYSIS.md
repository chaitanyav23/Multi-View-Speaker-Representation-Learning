# Phase 5 Evaluation Analysis

## Benchmarking Identity Separation

### 1. Pyannote Auto (Identity Collapse Case)
*   **Behavior:** The model assigns everyone to `SPEAKER_00`.
*   **Result:** DER is artificially low (0.41), but cpWER is high (0.5054).
*   **Finding:** Agglomerative clustering is "too safe"—it prefers to group everyone together rather than risk a wrong split in noisy conditions.

### 2. MSDD Oracle (Success Case)
*   **Behavior:** The sequence decoder successfully separates the officer and suspect.
*   **Result:** DER is 0.4492 (higher than Auto), but **cpWER is 0.4582** (9.3% lower).
*   **Key Insight:** The DER is higher for MSDD because it attempts to find the "second speaker," which introduces some "Confusion" errors. However, because it *finds* the second speaker, the cpWER (which measures the usability of the text) improves dramatically.

## Diagnostic Metrics
Phase 5 introduced `diarization_diagnostics.py`, which provides:
*   **Overlap Miss Rate:** How often the model fails to detect two people speaking at once.
*   **Short Turn Attribution:** Accuracy on segments <1.0s.
*   **Identity Consistency:** Whether a speaker's ID changes throughout the file.

## Oracle VAD Sensitivity
The analysis proved that the quality of the "Oracle" VAD injected into MSDD is the primary driver of performance. If the VAD segment includes too much background noise, MSDD's clustering logic starts to drift.
