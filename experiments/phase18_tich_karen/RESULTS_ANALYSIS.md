# Results Analysis: Phase 18 (Tich Karen)

## Comparative Results Table

| Variation | VAD F1 | ASR WER | DER | Key Driver of Error |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Normalized** | 0.7940 | 22.33% | 62.86% | False Alarms (VAD) |
| **No Normalization** | 0.7939 | 22.89% | 62.93% | No significant change |
| **Oracle Grounded** | 1.0000 | 41.18% | **18.10%** | Speaker Confusion |
| **ASR-Grounded** | 0.7586 | 25.15% | 69.66% | Missed Speech |

## Analysis of Outcomes

### 1. The VAD Ceiling
The most significant result is the **44.7% improvement in DER** when moving from Standard VAD to Oracle VAD. This confirms that the MSDD diarizer is highly competent at separating voices (18.1% DER is elite for noisy field recordings), but it is being overwhelmed by "non-speech" segments passed to it by the VAD.

### 2. Normalization Impact
The comparison between Normalized and No-Normalization shows that RMS normalization is a "safe" operation. It slightly improved WER (22.33% vs 22.89%) without degrading diarization. This suggests that the TitaNet embedding model is robust to volume scaling.

### 3. ASR-Grounded Failure
Using WhisperX word timestamps to guide diarization (ASR-Grounded) was expected to be more robust to noise. Instead, it performed worse (69.6% DER). This is because WhisperX often "skips" words in high-noise environments, leading to high Miss rates (16%) in the diarization evaluation. Pyannote remains the preferred VAD tool.

### 4. Oracle ASR WER Paradox
The Oracle-Grounded run showed a much higher WER (41.18%) than the Standard run (22.33%). 
- **Cause**: In Oracle mode, the system transcribes *exactly* the GT segments, including segments with extreme background noise where a human could transcribe but the model struggles. 
- **Inference**: WhisperX's internal VAD (used in the Standard run) is actually quite good at filtering out untranscribable audio, which artificially "improves" the WER score by ignoring the hardest segments.

## Summary
The current architecture (Phase 13 derived) is stable and robust for transcription, but its performance on noisy audio is capped by the VAD's inability to distinguish between environmental noise (sirens) and human speech.
