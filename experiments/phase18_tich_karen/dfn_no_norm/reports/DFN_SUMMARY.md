# Phase 18: Denoised (DeepFilterNet) Pipeline Summary

To address the high False Alarm rate caused by background noise triggering the VAD, we ran a sub-experiment (`dfn_no_norm`) where the raw, unnormalized audio was passed through **DeepFilterNet (DFN)** before running Pyannote and WhisperX.

## Results Comparison

| Metric | 2. No-Norm (Baseline) | **5. DFN No-Norm** | Change |
| :--- | :--- | :--- | :--- |
| **VAD Total Speech** | 497.6s | **439.1s** | Closer to GT (393.0s) |
| **VAD False Alarm** | 0.3649 | **0.2824** | **-8.25% (Improvement)** |
| **VAD Missed Speech**| 0.1052 | **0.1705** | +6.53% (Worse) |
| **DER** | 0.6293 | **0.6004** | **-2.89% (Improvement)** |
| **Speaker Confusion**| 0.1592 | **0.1475** | -1.17% (Improvement) |
| **ASR WER** | 0.2289 | **0.3400** | **+11.11% (Severe Degradation)** |

---

## Key Insights

### 1. DFN Reduces False Alarms (Success)
DeepFilterNet successfully removed enough background noise (sirens, wind, handling noise) to reduce the total amount of "hallucinated" speech by nearly 60 seconds. This led to a direct drop in the Diarization False Alarm rate (from 36.5% down to 28.2%).

### 2. DFN Improves Speaker Clarity (Success)
Because the background noise was suppressed, the TitaNet embeddings were "cleaner." This resulted in a drop in Speaker Confusion (from 15.9% to 14.7%). The MSDD model had an easier time distinguishing the voices without the acoustic interference of the background environment.

### 3. DFN Destroys ASR Performance (Failure)
The critical tradeoff is in transcription accuracy. While DFN makes the audio "cleaner" for Voice Activity Detection and Embedding extraction, the aggressive filtering distorts the phonemes. WhisperX (Large-v3) struggled heavily with this artificial-sounding voice data, causing the Word Error Rate (WER) to skyrocket from **22% to 34%**.

---

## Final Recommendation for Phase 18 Architecture

We have uncovered a classic **Diarization vs. Transcription tradeoff**. 
* **Clean audio (DFN)** is best for VAD and Diarization (lower DER).
* **Raw audio (Normalized)** is best for WhisperX (lower WER).

### The "Dual-Path" Solution
To achieve the best of both worlds, the production pipeline should split the audio processing into two parallel paths:
1. **Path A (Diarization Path):** Apply DeepFilterNet -> Pyannote VAD -> NeMo MSDD.
2. **Path B (Transcription Path):** Apply RMS Normalization (No DFN) -> WhisperX.
3. **Fusion:** Merge the clean Diarization timestamps (Path A) with the accurate WhisperX transcripts (Path B).