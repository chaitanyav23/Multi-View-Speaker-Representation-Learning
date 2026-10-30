# Phase 4 Evaluation Analysis

## Primary Metrics: cpWER and WDER

### 1. The cpWER "Inversion"
Phase 4 provided a counter-intuitive result. While **WER** (transcription error) was higher for DeepFilterNet (0.36) than for the Baseline (0.29), the **cpWER** (conversational error) was lower for DeepFilterNet (0.47) than for the Baseline (0.50). 

**Interpretation:** 
The improvement in speaker attribution was so significant that it outweighed the cost of transcription artifacts. A slightly less accurate transcript with correct speaker labels is more "forensicly correct" than a high-accuracy transcript where all words are attributed to the wrong person.

### 2. WDER (Word Diarization Error Rate)
*   **Baseline (Norm only):** 0.1033
*   **DFN (Norm + Denoise):** 0.0558
*   **Reduction:** ~46% relative improvement.
*   **Insight:** Pyannote's embedding model is highly sensitive to background noise. Denoising provides a cleaner feature space for the speaker classifier.

## Error Tradeoff Matrix

| Action | Impact on WER | Impact on WDER | Impact on cpWER |
| :--- | :--- | :--- | :--- |
| **Normalization** | Neutral | Neutral | Neutral |
| **Denoising (DFN)**| **Worsens** | **Improves** | **Improves Overall** |
| **Denoising (SB)** | **Worsens** | **Worsens** | **Worsens Overall** |

## Metric Sensitivity
Phase 4 proved that **WDER** is the most sensitive metric to audio quality, while **WER** is the most sensitive to neural artifacts. The optimal forensic pipeline must balance these two.
