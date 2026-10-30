
# Phase 6 V2 Audit and Contamination Report

## 1. Executive Summary

This report documents the clean rerun of the Phase 6 experiment to investigate potential baseline contamination. The investigation confirms that the original Phase 6 "normalized" baseline was contaminated with enhanced audio (likely DeepFilterNet), which skewed the experimental results and led to incorrect conclusions.

**Conclusion:** The original Phase 6 results are INVALID for baseline comparison. The Phase 6 V2 results should be used as the definitive reference point for all future work.

---

## 2. Audio Provenance Investigation

We compared the original Phase 6 audio artifacts against a clean regeneration from `raw_audio/bwc.wav`.

### Metric Comparison (BWC Baseline)

| Metric | Original Phase 6 (`bwc_norm.wav`) | Clean V2 (`bwc_norm_v2.wav`) |
| :--- | :--- | :--- |
| **MD5** | `679e47f99a4c45255b9e9a6665e6d321` | `c0b66acf8b833b6f8515e524b0c9f6e0` |
| **Spectral Centroid** | 1732.82 Hz | 1309.61 Hz |
| **Spectral Roll-off** | 3234.90 Hz | 2593.17 Hz |
| **RMS** | 0.0728 | 0.1009 |
| **LUFS** | -20.49 dB | -19.64 dB |

**Finding:** The original `bwc_norm.wav` is acoustically much closer to `bwc_deepfilternet.wav` (Spectral Centroid: 1672 Hz) than to the actual normalized audio. This confirms that the original baseline was already enhanced, masking the true performance of the system on raw normalized audio.

---

## 3. Metric Evaluation (V2 Clean Rerun)

The following metrics represent the performance of the pipeline on clean, verified v2 audio.

### V2 Performance Summary

| Approach | WER | DER | VAD F1 | SA-WER | cpWER |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Clean)** | **0.3008** | **0.6064** | **0.8022** | **0.6133** | **0.7466** |
| **DeepFilterNet** | 0.4121 | 0.6647 | 0.7695 | 0.6738 | 0.7470 |
| **SpeechBrain** | 0.4922 | 0.7057 | 0.7114 | 0.7793 | 0.6699 |

**Crucial Observation:** On clean normalized audio, the system performs significantly BETTER (WER 0.30) than previously thought (WER 0.40). Furthermore, current enhancement methods (DeepFilterNet, SpeechBrain) actually DEGRADE performance relative to clean normalization in this specific BWC environment.

---

## 4. Contamination Impact Analysis

| Metric | Original Baseline | V2 Clean Baseline | Absolute Delta | Relative Delta (%) |
| :--- | :--- | :--- | :--- | :--- |
| **WER** | 0.4074 | 0.3008 | -0.1066 | **-26.17%** |
| **DER** | 0.6991 | 0.6064 | -0.0927 | **-13.26%** |
| **VAD F1** | 0.7729 | 0.8022 | +0.0293 | **+3.79%** |

### Statistical Assessment
The contamination caused a 26% relative increase in ASR error and a 13% relative increase in Diarization error for the reported baseline. This materially changed the scientific conclusions of Phase 6.

---

## 5. Conclusion and Recommendations

### 5.1 Did the original conclusions hold?
**NO.** 
*   **Original Conclusion:** Baseline is poor (WER 0.40), enhancement helps slightly or not at all.
*   **V2 Conclusion:** Baseline is relatively strong (WER 0.30), current enhancement methods are counter-productive and introduce artifacts that confuse the ASR/Diarization models.

### 5.2 Recommendations
1.  **Discard Original Phase 6:** All future comparisons must use the Phase 6 V2 baseline.
2.  **Normalization Priority:** Clean normalization (`loudnorm`) is superior to current neural enhancement for BWC audio.
3.  **Future Enhancer Research:** Any future enhancement experiments must be strictly validated against the V2 `loudnorm` baseline to ensure they do not introduce regression.
4.  **Audit Mandate:** All future phases must include the `run_audio_audit.py` step to prevent similar contamination events.

---
*Report generated on June 2, 2026.*
