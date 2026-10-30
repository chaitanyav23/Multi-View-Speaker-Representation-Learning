# Phase 2 Results Analysis

## Comparative Performance (BWC Dataset)

| Metric | Phase 1 (Raw) | DeepFilterNet (DFN) | SpeechBrain (MetricGAN+) |
| :--- | :--- | :--- | :--- |
| **WER** | 0.3037 | 0.3781 | 0.4504 |
| **DER** | 0.4234 | 0.4261 | 0.5603 |
| **cpWER** | 0.4989 | 0.5203 | 0.5482 |
| **Miss** | 0.1784 | 0.1531 | 0.2889 |
| **FA** | 0.1746 | 0.2103 | 0.2282 |

## Key Findings

### 1. DFN is the "Best of the Worst"
DeepFilterNet3 performed significantly better than SpeechBrain, but still resulted in a **24% relative increase in WER** (0.30 -> 0.37). 

### 2. SpeechBrain Failure
The MetricGAN+ model performed very poorly on BWC data, resulting in a **WER of 0.45** and a massive **DER of 0.56**. It effectively destroyed the speaker characteristics needed for diarization, leading to chaotic speaker assignments.

### 3. VAD Resilience
Surprisingly, DFN slightly improved the **Miss rate (0.17 -> 0.15)**, suggesting it helped the VAD find speech segments. However, this gain was completely negated by the loss of clarity during the ASR stage.

### 4. Podcast Stability
Performance remained stable on the Podcast dataset, confirming that enhancement models do minimal damage to already clean audio but fail to "fix" forensic-grade noise without significant side effects.

## Experimental Conclusions
*   **ASR Sensitivity:** WhisperX (and Whisper in general) is trained on noisy data and is remarkably resilient to "natural" noise. However, it is **highly sensitive to "neural" noise** (artifacts introduced by denoising models).
*   **Stage-Specific Impact:** Enhancement helps VAD slightly but hurts ASR and Diarization significantly.
