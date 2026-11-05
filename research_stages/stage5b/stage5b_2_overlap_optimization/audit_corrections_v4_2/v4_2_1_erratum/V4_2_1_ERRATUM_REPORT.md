# Stage 5B.2 V4.2.1: Documentation Erratum Report

**Audit Phase**: Stage 5B.2 V4.2.1 Documentation Erratum & Reconciliation  
**Date**: November 5, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE — ALL ERRATA RECONCILED

---

## 1. Executive Summary

This erratum report records the specific documentation corrections implemented during Stage 5B.2 V4.2.1:
1. **Correction of Bootstrap Mean Terminology**: Disentangled the observed sample mean difference (+0.5221 pp) from the empirical mean of the bootstrap resample replicates (+0.5027 pp).
2. **Standardization of Difference Units to Percentage Points (pp)**: Replaced percentage symbols (`%`) with percentage points (`pp`) for all metric differences and confidence intervals, avoiding ambiguity between absolute changes in DER and relative rate changes.
3. **Harmonization of Stop-Condition Wording**: Updated the stop condition language to prevent any apparent contradiction with the historical record of the 48 evaluations performed during V4.1.
4. **Preservation of Benchmark Ground Truth**: Confirmed that all underlying benchmark values, RTTMs, embeddings, and metric CSVs remain 100% unchanged.

---

## 2. Detailed Errata Corrections

### 2.1 Bootstrap Terminology Reconciliation
In previous drafts, `+0.5221%` was incorrectly described as the "Mean Resampled Difference".
- **Observed Paired Mean Difference**: **`+0.5221 percentage points`** (`+0.5221 pp`, $+0.005221$). This is the exact sample mean of the eight paired recording-level differences $(\text{DER}_{\text{O42-R2}} - \text{DER}_{\text{O02}})$.
- **Empirical Bootstrap Replicate Mean**: **`+0.5027 percentage points`** (`+0.5027 pp`, $+0.005027$). This is the arithmetic mean across the $B = 5,000$ resampled means under random seed `42`.
- **95% Percentile Confidence Interval ($B=5,000$, Seed=42)**:
  $$\mathbf{[-2.3501, \; +2.9050] \text{ percentage points}} \quad (\mathbf{[-2.35, \; +2.90] \text{ pp}})$$
- **Non-Parametric Hypothesis Test**: Wilcoxon signed-rank test confirmed at $W = 4.0, p = 0.8750$ across the $N_{\text{effective}} = 4$ non-zero paired differences.

### 2.2 Percentage-Point Notation Standardization
All reporting was updated to consistently use "percentage points" or "pp" for differences and intervals:
- Macro DER Difference (c=0 ms): **`+0.52 pp`** (`+0.5221 pp`)
- Macro DER Difference (c=250 ms): **`+0.35 pp`** (`+0.3444 pp`)
- Pooled DER Difference (c=0 ms): **`+0.60 pp`** (`+0.5980 pp`)
- `R04_piers_robinson` Improvement: **`-8.09 pp`** (`-8.0852 pp` at c=0 ms) and **`-8.50 pp`** (`-8.5048 pp` at c=250 ms)
- Paired Bootstrap 95% Confidence Interval: **`[-2.35, +2.90] pp`**

### 2.3 Updated Stop-Condition Phrasing
To maintain total consistency with the documented V4.1 execution record, the stop condition was updated from the overbroad phrasing ("No models have been trained or evaluated") to the exact, auditable statement:
> *"No model training, neural inference, embedding extraction, VAD inference or ASR inference occurred during V4.2/V4.2.1. The 48 downstream assignment/scoring evaluations performed during V4.1 remain transparently documented as a protocol-scope deviation."*

---

## 3. Benchmark Ground Truth Preservation

Zero benchmark numbers or evaluation files were modified:
- `O00` (Baseline C00 VAD): `46.42%` (Macro DER c=0 ms)
- `O02` (Baseline C62 Consensus VAD): `44.90%` (Macro DER c=0 ms) / `42.62%` (Macro DER c=250 ms)
- `O42-R2` (C62 Consensus VAD + M4 Conservative Overlap): `45.42%` (Macro DER c=0 ms) / `42.97%` (Macro DER c=250 ms)
- Direction of Change: 1 Win (`R04`), 3 Losses (`R03`, `R05`, `R06`), 4 Ties (`R01`, `R02`, `R07`, `R08`).
- The four primary metric CSVs and `V4_2_FINAL_METRIC_TABLE.csv` are bitwise unchanged.
