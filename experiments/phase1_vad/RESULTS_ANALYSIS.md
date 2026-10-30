# Phase 1 Results Analysis

## Comparative Performance Table (BWC Dataset)

| Metric | Silero VAD (Baseline) | Pyannote v3.0 | Delta (Rel. %) |
| :--- | :--- | :--- | :--- |
| **WER** | 0.3037 | 0.3079 | +1.3% (Slightly Worse) |
| **DER** | 0.4234 | 0.4147 | -2.0% (Better) |
| **Miss** | 0.1784 | 0.1164 | -34.7% (Significantly Better) |
| **FA** | 0.1746 | 0.2299 | +31.6% (Worse) |
| **cpWER** | 0.4989 | 0.5139 | +3.0% (Slightly Worse) |

## Analysis of Findings

### 1. The "Miss" Advantage
Pyannote v3.0 demonstrated a superior ability to capture speech in the challenging Body-Worn Camera (BWC) dataset, reducing the **Miss rate by nearly 35%** compared to Silero. This indicates that Pyannote is much better at detecting low-volume or heavily masked speech in noisy environments.

### 2. The "False Alarm" Penalty
Conversely, Pyannote's permissive nature led to a **31.6% increase in False Alarms**. In practice, this meant the pipeline attempted to transcribe static, wind noise, or unintelligible background chatter, which contributed to "hallucinations" in the ASR output.

### 3. Impact on Conversational Metrics
Despite the better DER, Pyannote resulted in a slightly higher **cpWER (0.5139 vs 0.4989)**. This suggests that the False Alarms (hallucinated words) penalized the conversational WER more than the recovered speech improved it.

### 4. Stability in High-SNR (Podcast)
In the clean Podcast dataset, both models performed similarly (DER ~0.095, WER ~0.053), indicating that VAD choice is less critical when audio quality is high.

## Experimental Findings
*   Pyannote's segmentation is more granular and precise for overlapping speech.
*   Silero is more "safe" but misses critical dialogue in forensic (BWC) contexts.
*   The **Min-Cut** strategy was essential for Silero to prevent massive 60s+ segments which would crash the ASR stage or lead to context loss.
