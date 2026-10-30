# Results Analysis: Phase 17 vs. Phase 11A

## Quantitative Comparison
The following table summarizes the performance delta between the LoudNorm-enabled baseline (P11A) and the No-Normalization experiment (P17).

| Metric | Baseline (P11A) | No-Norm (P17) | Delta | Trend |
| :--- | :--- | :--- | :--- | :--- |
| **WER** | 0.3002 | 0.3368 | +0.0366 | ❌ Regression |
| **DER** | 0.4679 | 0.4493 | -0.0186 | ✅ Improvement |
| **Miss** | 0.2878 | 0.2050 | -0.0828 | ✅ Strong Improvement |
| **FA** | 0.0766 | 0.1674 | +0.0908 | ❌ Regression |
| **Confusion**| 0.1035 | 0.0770 | -0.0265 | ✅ Improvement |
| **WDER** | 0.1136 | 0.1066 | -0.0070 | ✅ Slight Improvement |
| **cpWER** | 0.5032 | 0.4814 | -0.0218 | ✅ Improvement |

## Key Insights

### 1. The VAD Sensitivity Paradox
The most striking result is the 8.28% absolute reduction in **Miss Rate**. This confirms that normalization was inadvertently crushing low-amplitude speech (common in BWC audio where the officer might be whispering or the subject is distant). However, this gain is offset by a 9.08% increase in **False Alarms**, as the VAD now triggers on un-normalized background noise.

### 2. Lexical Degradation
The **WER** increase of 3.66% indicates that while more speech is being "captured" (lower miss rate), the quality of transcription for those segments is lower. Un-normalized audio likely presents a more challenging signal-to-noise ratio for the WhisperX acoustic model, leading to higher substitution errors.

### 3. Structural Diarization Gains
Interestingly, **DER** and **cpWER** both improved. This suggests that for BWC audio, the *structural* accuracy of knowing who spoke when is actually better on un-normalized audio. The decrease in **Confusion** (2.65%) indicates that speaker embeddings extracted from raw signals are more discriminative than those extracted from normalized (and potentially distorted) signals.

### 4. Attribution Stability
The improvement in **WDER** (11.36% -> 10.66%) shows that the speaker-to-word attribution logic is robust to the lack of normalization, and actually benefits slightly from the more sensitive VAD grounding.
