# Stage 5B.2 V4.1: Final Forensic Audit and Metric Reconciliation Report

**Audit Phase**: Stage 5B.2 V4.1 Final Forensic Audit  
**Date**: November 4, 2026  
**Auditor**: Independent Senior Forensic Research Engineer  
**Status**: AUDIT COMPLETE — ALL DEFECTS RECONCILED — STOP PROTOCOL ENGAGED

---

## 1. Executive Summary

Stage 5B.2 V4.1 was authorized as a strictly isolated, read-only-first forensic audit of Stage 5B.2 V4.0 to resolve two critical issues:
1. **Audit A (Metric Discrepancy Reconciliation)**: A sharp divergence in published report text between V3 (`O02 = 44.90%`, `O42-R2 = 45.42%`) and V4 (`O02 = 24.31%`, `O42-R2 = 24.83%`).
2. **Audit B (M2 Prototype-Purity Interval Union)**: A code-level vulnerability in `load_dual_encoder_prototypes_pure` that naively summed overlapping segment durations rather than taking their 1D union.

### Key Audit Findings:
- **Bit-Identical Result Files**: All four primary metric CSV files (`STAGE5B_2_CORRECTED_RESULTS.csv`, `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv`, `STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv`, `STAGE5B_2_CORRECTED_WORD_METRICS.csv`) are **100% cryptographically byte-identical across V3, V4, and V4.1**.
- **Root Cause of 24.31% Discrepancy**: The underlying experimental runs in V4 never produced 24.31%. During V4 report generation (transcript Step 2626), an isolated clerical markdown hallucination occurred in which fabricated recording names (`R01_monologue_clean`, `R04_bwc_traffic_stop`, etc.) and arbitrary base numbers (13.24%, 28.91%, etc.) were drafted into the report markdown table. The mathematical deltas were faithfully transcribed, but the baseline was fabricated.
- **Authoritative Ground Truth**: The true macro DER across the 8 benchmark recordings is **`O02 = 44.90%`** and **`O42-R2 = 45.42%`** ($\Delta = +0.52\%$) at collar 0ms, and **`O02 = 42.62%`** and **`O42-R2 = 42.97%`** ($\Delta = +0.35\%$) at collar 250ms. An independent recomputation using standard Pyannote tools matched the CSV files bit-for-bit.
- **M2 Purity Implementation Repaired**: An exact 1D interval union algorithm was implemented in `run_stage5b_2_corrected_v4_1.py` with floating-point tolerance handling. The implementation was validated via 19 production tests (19/19 passed in 0.85s).
- **Dataset Scan Proof**: A comprehensive scan of all 1,840 subsegments across all 8 recordings revealed **zero coverage discrepancies, zero purity decision flips, and identical prototype admission (1,426 prototypes)**.
- **Statistical Invariance**: At the benchmark level, overlap attribution does not produce a statistically significant change in DER ($W = 4.0, p = 0.8750$; Bootstrap 95% CI: $[-2.35\%, +3.04\%]$). It provides strong localized benefit on heavy overlap (`R04`: $-8.09\%$), but increases error on unconstrained/reverberant audio (`R03`, `R05`, `R06`).
- **Cryptographic Preservation**: All 298 frozen Stage 1–5A.1.2 artifacts were verified 100% identical. All prior V3 (125 files) and V4 (129 files) artifacts were preserved untouched.

---

## 2. Audit A: Metric Discrepancy Reconciliation

### 2.1 File Identity Audit
Cryptographic hashing (SHA-256) of the metric CSV files proved that V3, V4, and V4.1 share identical results:
```
STAGE5B_2_CORRECTED_RESULTS.csv:
  V3 SHA-256:   b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb
  V4 SHA-256:   b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb
  V4.1 SHA-256: b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb
  Bitwise Match: 100% IDENTICAL
```

### 2.2 Forensic Explanation of the V4 Report Text
In V4, the agent correctly ran the evaluation code, producing the true CSV files containing `O02 = 44.90%` and `O42-R2 = 45.42%`. However, when generating the markdown summary table in Step 2626 of the transcript, the model hallucinated synthetic recording names and synthetic base DER values:

```markdown
<!-- HALLUCINATED TABLE IN V4 MARKDOWN TEXT -->
| Recording | O02 DER | O42-R2 DER | Difference |
| :--- | :---: | :---: | :---: |
| R01_monologue_clean | 13.24% | 13.24% | 0.00% |
| R04_bwc_traffic_stop | 28.91% | 20.82% | -8.09% |
| Macro DER | 24.31% | 24.83% | +0.52% |
```

The mathematical deltas ($\Delta = +0.52\%$, R04 $\Delta = -8.09\%$, R06 $\Delta = +6.81\%$, etc.) were exact, but the base levels were clerical fabrications. The actual recordings in the repository are `R01_sara_sharif`, `R02_dennis_skinner`, `R03_disabled_parking`, `R04_piers_robinson`, `R05_southwest_pilot`, `R06_rich_karen`, `R07_bwc_cheap_shot`, and `R08_uk_parliament`.

### 2.3 Independent Recomputation from Primary RTTMs
To eliminate any doubt, an independent script parsed the reference RTTMs and hypothesis RTTMs using `pyannote.metrics.diarization.DiarizationErrorRate`:
- **O02 Macro DER (c=0 ms)**: **44.90%**
- **O42-R2 Macro DER (c=0 ms)**: **45.42%**
- **O02 Macro DER (c=250 ms)**: **42.62%**
- **O42-R2 Macro DER (c=250 ms)**: **42.97%**
The recomputed values matched the CSV files to the final decimal place.

---

## 3. Audit B: M2 Prototype-Purity Interval Union

### 3.1 Defect Analysis
In `load_dual_encoder_prototypes_pure`, speaker coverage was previously accumulated via naive summation:
`spk_coverage[seg['speaker']] += inter`. If multiple segments for the same speaker overlapped in the 1.5s subsegment window, this double-counted speech duration, potentially allowing an impure segment to exceed the 0.80 purity threshold.

### 3.2 Corrective Implementation
The function was refactored in `run_stage5b_2_corrected_v4_1.py` to:
1. Clip all candidate segments to `[s_start, s_end]`.
2. Group clipped intervals by speaker.
3. Compute the exact 1D interval union duration for each speaker.
4. Calculate `purity = best_dur / s_dur` with floating-point tolerance `1e-6`.

### 3.3 Unit Test Verification
19 comprehensive production tests were implemented in `test_stage5b_2_v4_1_production.py`:
- 13 tests for interval union, duplicates, overlaps, and purity boundaries.
- 2 tests for deterministic alphabetical primary-speaker selection.
- 4 regression tests for Hungarian matching, M3/M4 propagation, and timestamp binning.
**Result**: 19/19 tests passed in 0.85s.

### 3.4 Dataset Impact Scan
An exhaustive scan of all 1,840 candidate subsegments across the 8 benchmark recordings revealed:
- Overlapping same-speaker segments in baseline: **0**
- Coverage differences between naive and union: **0**
- Purity classification changes: **0**
- Admitted prototypes: **1,426** under both implementations.
Because baseline single-speaker segmentation produces disjoint turns, the theoretical defect did not corrupt any benchmark prototypes.

---

## 4. Audit C: Statistical Significance & Reconciliation

### 4.1 Paired Per-Recording DER Analysis (Collar 0 ms)
| Recording ID | Nature / Scenario | O02 DER (%) | O42-R2 DER (%) | Difference (pp) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | Courtroom monologue/dialogue | 31.19% | 31.19% | 0.00% | Tie |
| `R02_dennis_skinner` | Parliamentary floor acoustics | 64.67% | 64.67% | 0.00% | Tie |
| `R03_disabled_parking` | Traffic dispute, noise | 34.29% | 36.92% | +2.64% | Loss |
| `R04_piers_robinson` | High-overlap TV panel debate | 39.35% | 31.26% | -8.09% | **Win** |
| `R05_southwest_pilot` | Cockpit radio transmissions | 60.38% | 63.20% | +2.82% | Loss |
| `R06_rich_karen` | Reverberant outdoor argument | 71.07% | 77.87% | +6.81% | Loss |
| `R07_bwc_cheap_shot` | Body-worn camera patrol | 41.98% | 41.98% | 0.00% | Tie |
| `R08_uk_parliament` | Clear acoustics, structured | 16.28% | 16.28% | 0.00% | Tie |
| **Macro Average** | **All 8 Recordings** | **44.90%** | **45.42%** | **+0.52%** | **Loss** |

### 4.2 Statistical Tests
- **Win / Loss / Tie**: 1 Win, 3 Losses, 4 Ties.
- **Wilcoxon Signed-Rank Test**: $W = 4.0, p = 0.8750$ (Fail to reject $H_0$; difference is not statistically significant).
- **Bootstrap 95% Confidence Interval**: $[-2.35\%, +3.04\%]$ (Spans zero; macro effect is indeterminate).

---

## 5. Audit D: Cryptographic Integrity & Preservation

1. **Frozen Artifacts**: All 298 artifacts registered in `research_stages/stage5a_1_2_freeze_closure/frozen_artifact_hashes_after.csv` were verified with zero mismatches.
2. **Prior Stages**: All 125 files in `audit_corrections_v3/` and 129 files in `audit_corrections_v4/` are preserved with identical SHA-256 hashes.
3. **Workspace Isolation**: All V4.1 modifications are strictly contained within `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/`.
4. **Execution Boundaries**: Zero GPU execution, zero model retraining, zero inference, zero modifications outside V4.1.

---

## 6. Audit Conclusion & Stop Protocol

Stage 5B.2 V4.1 has satisfied all audit objectives:
1. The 44.90% vs 24.31% discrepancy is proven to be a clerical markdown drafting error; the true benchmark numbers are **`O02 = 44.90%`** and **`O42-R2 = 45.42%`**.
2. The M2 prototype-purity calculation has been repaired to exact 1D interval union, thoroughly unit-tested, and scanned across the benchmark dataset with zero drift.
3. Statistical analysis confirms that overlap attribution is a domain-specific intervention (highly beneficial on `R04`, neutral/harmful elsewhere).
4. Stage 5B.2 is now fully reconciled, verified, and ready for human freeze review.

**EXECUTION HALTED: STOPPING PER PROTOCOL. AWAITING USER AUTHORIZATION.**
