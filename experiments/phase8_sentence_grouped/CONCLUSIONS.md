# Phase 8 Conclusions

## Summary of Results
Phase 8 achieved a major technical milestone by optimizing **VAD Recall to 0.86** and **F1 to 0.84**. This was accomplished by implementing a 1.0s "Sentence Grouping" gap. However, the phase also confirmed a hard limit on performance: **DeepFilterNet is not suitable for the ASR stage.** Despite improved detection and diarization, the overall cpWER remained stagnant at 0.66.

## Hypothesis Success
The hypothesis was **confirmed** regarding VAD and DER improvements. The 1.0s gap successfully improved detection and provided more stable identities for MSDD.

## Key Takeaways
*   **Sentence-level grouping** is the most effective VAD strategy for BWC.
*   **Acoustic buffers (0.5s)** are essential for WhisperX Large-v3.
*   The **"Single-Path" DFN architecture** has been exhausted and will be abandoned in favor of a "Dual-Path" production model.

## Preservation
The **1.0s Merge Gap** and **0.5s ASR Buffer** settings are preserved as the standard for all future production code.
