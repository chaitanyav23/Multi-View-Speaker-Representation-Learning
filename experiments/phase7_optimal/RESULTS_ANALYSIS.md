# Phase 7 Results Analysis

## Benchmarking (BWC Dataset)

| Metric | Phase 5 (Breakthrough) | Phase 7 (Optimal Single-Path) | Delta (Rel. %) |
| :--- | :--- | :--- | :--- |
| **WER** | **0.2975** | 0.3802 | +27% (Degradation) |
| **DER** | 0.4492 | 0.6137 | +36% (Degradation) |
| **WDER** | 0.1059 | 0.1983 | +87% (Degradation) |
| **cpWER** | **0.4582** | 0.6595 | +44% (Degradation) |
| **VAD Recall**| 0.6975 | **0.7788** | **+11.6% (Improvement)** |

## Analysis of Results

### 1. The Cost of Enhancement
The decision to use **DeepFilterNet** as a global conditioner in Phase 7 resulted in a severe performance regression across all conversational metrics. While it improved **VAD Recall** (by cleaning the speech for the detector), it introduced artifacts that significantly confused the **ASR** and **Diarization** models.

### 2. MSDD Sensitivity to Artifacts
The **WDER (0.1983)** in Phase 7 was nearly double that of Phase 5. This indicates that NeMo MSDD, like Whisper, is highly sensitive to the spectral distortions introduced by neural denoising. It prefers "raw noise" over "neural artifacts."

### 3. VAD "Soft-Grounding" Success
The only metric that improved was **VAD Recall**. The combination of DFN and 0.1s padding ensured that more actual speech was captured in the grounding manifest.

## Experimental Conclusion
Phase 7 was a successful "stress test" that yielded a vital architectural insight: **The pipeline cannot be a single-path system.** The ASR and Diarization models must see raw (or only normalized) audio, while only the VAD should see enhanced audio.
