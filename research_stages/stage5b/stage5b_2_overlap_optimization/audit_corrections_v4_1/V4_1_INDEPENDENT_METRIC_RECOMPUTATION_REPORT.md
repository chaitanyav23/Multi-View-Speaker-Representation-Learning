# Stage 5B.2 V4.1: Independent Metric Recomputation Report

**Audit Phase**: Stage 5B.2 V4.1 Independent Pyannote Recomputation  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: VERIFIED — 100% BIT-FOR-BIT MATCH WITH CACHED CSVS

---

## 1. Methodology

To ensure absolute forensic independence from any potential evaluation script artifacts, an independent evaluation script was executed. This script bypassed the Stage 5B.2 aggregation code entirely and directly invoked:
- Reference RTTMs from `research_stages/stage1_data_prep/rttm/*.rttm`
- Generated hypothesis RTTMs from `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/predictions/{config_id}/*.rttm`
- `pyannote.metrics.diarization.DiarizationErrorRate(collar=0.0, skip_overlap=False)`
- `pyannote.metrics.diarization.DiarizationErrorRate(collar=0.25, skip_overlap=False)`

---

## 2. Recomputed Per-Recording Results (Collar 0 ms)

| Recording ID | Total Ref Time (s) | O02 DER (%) | O42-R2 DER (%) | Delta (O42-R2 - O02) | Match with CSV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 134.42 | 31.19% | 31.19% | +0.00% | **EXACT** |
| `R02_dennis_skinner` | 239.88 | 64.67% | 64.67% | +0.00% | **EXACT** |
| `R03_disabled_parking` | 185.11 | 34.29% | 36.92% | +2.64% | **EXACT** |
| `R04_piers_robinson` | 276.71 | 39.35% | 31.26% | -8.09% | **EXACT** |
| `R05_southwest_pilot` | 179.94 | 60.38% | 63.20% | +2.82% | **EXACT** |
| `R06_rich_karen` | 241.65 | 71.07% | 77.87% | +6.81% | **EXACT** |
| `R07_bwc_cheap_shot` | 158.23 | 41.98% | 41.98% | +0.00% | **EXACT** |
| `R08_uk_parliament` | 288.54 | 16.28% | 16.28% | +0.00% | **EXACT** |
| **Macro Average** | — | **44.90%** | **45.42%** | **+0.52%** | **EXACT** |

---

## 3. Recomputed Per-Recording Results (Collar 250 ms)

| Recording ID | O02 DER (%) | O42-R2 DER (%) | Delta (O42-R2 - O02) | Match with CSV |
| :--- | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 30.15% | 30.15% | +0.00% | **EXACT** |
| `R02_dennis_skinner` | 64.23% | 64.23% | +0.00% | **EXACT** |
| `R03_disabled_parking` | 30.52% | 32.97% | +2.45% | **EXACT** |
| `R04_piers_robinson` | 38.48% | 29.97% | -8.50% | **EXACT** |
| `R05_southwest_pilot` | 57.38% | 59.76% | +2.39% | **EXACT** |
| `R06_rich_karen` | 66.61% | 73.04% | +6.42% | **EXACT** |
| `R07_bwc_cheap_shot` | 38.54% | 38.54% | +0.00% | **EXACT** |
| `R08_uk_parliament` | 15.07% | 15.07% | +0.00% | **EXACT** |
| **Macro Average** | **42.62%** | **42.97%** | **+0.35%** | **EXACT** |

---

## 4. Conclusion

Every value in `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv` and `STAGE5B_2_CORRECTED_RESULTS.csv` was recomputed from the primary hypothesis RTTMs and confirmed bitwise. The authoritative numbers are permanently established as:
- **Macro DER (c=0 ms)**: O02 = **44.90%**, O42-R2 = **45.42%** ($\Delta = +0.52\%$)
- **Macro DER (c=250 ms)**: O02 = **42.62%**, O42-R2 = **42.97%** ($\Delta = +0.35\%$)
