# Results Analysis: Phase 13 Karen Generalization

This document analyzes the numerical results obtained during the Phase 13 benchmark on the Karen dataset.

## Comparative Performance (Karen vs. BWC Phase 11A)

| Metric | BWC (Phase 11A) | Karen (Phase 13) | Delta |
| :--- | :---: | :---: | :---: |
| **WER** | 0.3002 | 0.1467 | -0.1535 |
| **DER** | 0.4679 | 0.4070 | -0.0609 |
| **Miss** | 0.2878 | 0.0814 | -0.2064 |
| **False Alarm** | 0.0766 | 0.1512 | +0.0746 |
| **Confusion** | 0.1035 | 0.1745 | +0.0710 |
| **cpWER** | 0.5032 | 0.2685 | -0.2347 |
| **WDER** | 0.1136 | 0.1849 | +0.0713 |

## Analysis of ASR Performance
The most striking result is the 51% reduction in WER (from 0.30 to 0.14).
-   **Intelligibility:** The Karen recording features significantly better signal-to-noise ratio and less "masking" than the Body Worn Camera (BWC) data.
-   **Model Performance:** WhisperX Large-v3 demonstrates excellent generalization, achieving high accuracy on this new acoustic environment.
-   **Error Type:** ASR errors are dominated by **Deletions** (156) over Substitutions (41) and Insertions (33). This indicates the system is still missing words, potentially during short, overlapping, or low-volume turns.

## Analysis of Diarization Performance
While DER improved slightly (from 0.46 to 0.40), the *composition* of errors shifted significantly.
-   **Reduction in Miss Rate:** The miss rate dropped from 0.28 to 0.08. This is a direct result of using Pyannote 3.1 VAD with a sensitive configuration, ensuring that almost all speech is passed to the diarizer.
-   **Increase in False Alarm:** The FA rate doubled. The sensitive VAD is triggering on non-speech events (breaths, background thumps), which the diarizer then attempts to attribute to speakers.
-   **Increase in Confusion:** Confusion rose from 0.10 to 0.17. This is primarily due to **over-clustering**. MSDD discovered 6 speakers for a 4-speaker conversation. The extra identities (Speaker_5, Speaker_6) fragment the dialogue and increase confusion metrics.

## Conversational Quality
-   **cpWER at 26.85%** is a massive leap forward compared to the 50% baseline on BWC. This indicates that for the Karen dataset, the resulting transcript is highly usable for downstream tasks like summarization or analysis.
-   **WDER at 18.49%** confirms that approximately 1 in 5 words is attributed to the wrong speaker. This is the primary bottleneck for conversational accuracy.

## Summary
The pipeline generalizes successfully. The architecture is robust enough to yield a high-quality transcript on higher-quality audio without any dataset-specific tuning. However, the over-clustering in MSDD remains a technical hurdle for perfect speaker attribution.
