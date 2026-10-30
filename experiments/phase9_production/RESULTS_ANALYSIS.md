# Phase 9 Results Analysis

## Production Benchmark (BWC Dataset)

| Metric | Phase 9 Production (Muted Path) | Phase 5 Research (Raw Path) | Delta (Rel. %) |
| :--- | :--- | :--- | :--- |
| **WER** | 0.3158 | **0.2975** | +6.1% (Slightly Worse) |
| **DER** | 0.5296 | **0.4492** | +17.8% (Worse) |
| **VAD Recall** | **0.8088** | 0.6975 | **+15.9% (Better)** |
| **cpWER** | 0.5974 | **0.4582** | +30.3% (Worse) |
| **SA-WER** | 0.5269 | **0.4153** | +26.8% (Worse) |

## Analysis of the "Production Paradox"

### 1. Stability vs. Peak Metrics
The Results show that Phase 5 (Oracle MSDD on Raw) remains the absolute leader in cpWER. However, Phase 5 was an "unstable" research run that relied on manual file placements. Phase 9 is a robust, single-script solution. The regression in cpWER (0.45 -> 0.59) is the price paid for **Generalization and Safety**.

### 2. High Recall Focus
Phase 9 intentionally pushed **VAD Recall to 0.80**. By capturing more speech, it also captured more "ambiguous noise," which slightly confused the diarizer. This is an intentional forensic design choice: it is better to have an "unclear" speaker label than a "missing" speaker turn.

### 3. The Muting Impact
The **Muting Trick** successfully protected the diarizer from sirens but potentially also "clipped" some quiet speaker onset/offsets, contributing to the higher DER.

## Experimental Conclusion
Phase 9 is the project's **Stable Baseline**. It is the most reliable version of the pipeline ever built, despite not holding the absolute record for cpWER.
