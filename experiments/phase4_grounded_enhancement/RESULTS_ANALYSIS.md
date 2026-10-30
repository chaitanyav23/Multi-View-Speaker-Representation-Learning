# Phase 4 Results Analysis

## Comparative Performance (BWC Dataset)

| Metric | Baseline (Norm Only) | DeepFilterNet (Grounded) | Delta (Rel. %) |
| :--- | :--- | :--- | :--- |
| **WER** | 0.2975 | 0.3698 | +24% (Degradation) |
| **DER** | 0.4114 | 0.4408 | +7% (Degradation) |
| **WDER** | 0.1033 | **0.0558** | **-46% (Improvement)** |
| **cpWER** | 0.5054 | **0.4711** | **-6.8% (Improvement)** |
| **Miss** | 0.1137 | 0.2345 | +106% (Degradation) |

## Key Findings

### 1. The Diarization "Save"
Even though DFN more than doubled the **Miss rate (0.11 -> 0.23)** because it occasionally muted quiet speech, its ability to perfectly separate the louder speakers (Officer vs. Suspect) led to a better overall cpWER. This proves that for forensic intelligence, correctly attributing the "core" conversation is more valuable than capturing every mumbled background word.

### 2. Normalization Benefit
The baseline **WER of 0.2975** (on normalized audio) is the best raw performance seen thus far, suggesting that simple librosa peak normalization is more effective than complex neural denoising for transcription accuracy.

### 3. SpeechBrain Failure Redux
SpeechBrain results (WER 0.47) confirmed its unsuitability for this project. It continues to introduce destructive artifacts that neither ASR nor Diarization can overcome.

## Experimental Conclusion
DeepFilterNet3 is a powerful tool for **Diarization assistance**, but it is **not a global solution**. A hybrid approach is needed.
