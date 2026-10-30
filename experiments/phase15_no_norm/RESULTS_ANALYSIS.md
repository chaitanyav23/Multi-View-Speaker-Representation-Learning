# Phase 15: Results Analysis

## Core Metrics Table
The following table summarizes the performance of the "No-Normalization" pipeline on the `karen.wav` dataset.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **WER** | 0.1435 | Excellent transcription accuracy. |
| **DER** | 0.4093 | High error in speaker labeling. |
| **cpWER** | 0.2647 | Significant degradation when speakers are attributed. |
| **WDER** | 0.1782 | ~18% of words assigned to wrong speaker. |
| **VAD F1** | 0.8967 | Strong VAD performance. |

## Detailed ASR Results
- **Ref Words:** 1568
- **Substitutions:** 49
- **Deletions:** 154
- **Insertions:** 22
- **Observation:** The high number of deletions (154) suggests that the VAD grounding might be cutting off some speech segments prematurely, or Whisper is struggling with certain low-volume words in the non-normalized audio.

## Detailed Diarization Results
- **Miss Rate:** 0.0820 (8.2%)
- **False Alarm:** 0.1554 (15.5%)
- **Confusion:** 0.1720 (17.2%)
- **Observation:** The Confusion rate is the primary driver of DER. This indicates that MSDD is frequently swapping speaker IDs, which is further reflected in the high WDER (17.82%).

## VAD Performance
- **Recall:** 0.9422
- **Precision:** 0.8554
- **False Alarm Rate:** 0.6889 (Frame-level)
- **Observation:** While the F1 score is high, the False Alarm rate suggests that the VAD is significantly over-segmenting or including a lot of non-speech background noise in its segments. However, the high Recall (94%) is good for ensuring most speech is captured for ASR.

## Comparison to BWC Phase 11A
Phase 15 (Karen) shows a significant improvement over the Phase 11A (BWC) baseline in almost all categories:
- **WER:** 0.1435 vs 0.3002 (~52% reduction in error)
- **DER:** 0.4093 vs 0.4679 (~12% reduction in error)
- **cpWER:** 0.2647 vs 0.5032 (~47% reduction in error)

This suggests that the "No-Normalization" approach on the Karen dataset is highly effective, though some of this gain is likely due to the inherent differences between the Karen and BWC datasets.
