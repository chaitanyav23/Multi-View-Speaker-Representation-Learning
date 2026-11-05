# Stage 5B.2 V4.2: Bootstrap Reproduction and Statistical Reconciliation Report

**Audit Phase**: Stage 5B.2 V4.2 Statistical Reconciliation  
**Date**: November 4, 2026 (Updated for V4.2.1 Erratum, November 5, 2026)  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE & VERIFIED

---

## 1. Executive Summary

This report establishes the authoritative paired bootstrap confidence intervals and non-parametric hypothesis testing for the primary diarization comparison in Stage 5B.2:
- **Baseline**: `O02` (Single-label diarization with C62 consensus VAD, no overlap handling).
- **Intervention**: `O42-R2` (Integrated pipeline with C62 consensus VAD + M4 conservative contiguous overlap attribution).

The audit resolves the sample-count discrepancy between the protocol specification ($B = 5,000$) and the V4.1 execution log ($B = 10,000$), documents the exact resampling algorithm and random seed, eliminates pseudo-Bayesian phrasing, and confirms that the difference in upper bounds is purely Monte Carlo sampling variation in the empirical bootstrap distribution.

All difference metrics and confidence intervals are expressed strictly in percentage points (pp).

---

## 2. Input Data: Paired Per-Recording Observations

The eight recording-level DER observations are drawn directly from the authoritative result file [`STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv):

| Recording Index | Recording ID | O02 DER (c=0 ms) | O42-R2 DER (c=0 ms) | Paired Difference $\Delta_i$ (pp) | Direction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `R01_sara_sharif` | 31.1935% | 31.1935% | +0.0000 pp | Tie |
| 1 | `R02_dennis_skinner` | 64.6665% | 64.6665% | +0.0000 pp | Tie |
| 2 | `R03_disabled_parking` | 34.2865% | 36.9219% | +2.6354 pp | Loss |
| 3 | `R04_piers_robinson` | 39.3486% | 31.2634% | -8.0852 pp | **Win** |
| 4 | `R05_southwest_pilot` | 60.3814% | 63.2010% | +2.8196 pp | Loss |
| 5 | `R06_rich_karen` | 71.0671% | 77.8739% | +6.8067 pp | Loss |
| 6 | `R07_bwc_cheap_shot` | 41.9837% | 41.9837% | +0.0000 pp | Tie |
| 7 | `R08_uk_parliament` | 16.2821% | 16.2821% | +0.0000 pp | Tie |
| **Macro** | **Unweighted Mean (N=8)** | **44.9012%** | **45.4233%** | **+0.5221 pp** | **Loss** |

The eight paired differences sum to $+0.041767$, giving the observed paired mean difference of:
$$\bar{\Delta} = \frac{1}{8} \sum_{i=1}^8 \Delta_i = +0.005221 \quad (+0.5221 \text{ percentage points})$$

---

## 3. Protocol-Specified Bootstrap Reproduction ($B = 5,000$, Seed = 42)

### 3.1 Mathematical Specification
1. **Sampling Unit**: The complete recording pair $(\text{DER}_{i}^{\text{O02}}, \text{DER}_{i}^{\text{O42-R2}})$.
2. **Resample Count**: $B = 5,000$.
3. **Random Seed**: `42` (`np.random.seed(42)`).
4. **Resampling Mechanism**: For each $b \in \{1, \dots, B\}$, draw $N = 8$ indices with replacement uniformly from $\{0, \dots, 7\}$:
   $$\mathbf{idx}^{(b)} = (j_1, \dots, j_8), \quad j_k \sim \text{Uniform}(\{0, \dots, 7\})$$
5. **Replicate Mean**:
   $$\bar{\Delta}^{(b)} = \frac{1}{8} \sum_{k=1}^8 \Delta_{j_k}$$
6. **Percentile Interval**: Using standard linear interpolation (`np.percentile` method):
   $$\text{CI}_{95} = \left[ \text{Percentile}(\{\bar{\Delta}^{(b)}\}, 2.5), \; \text{Percentile}(\{\bar{\Delta}^{(b)}\}, 97.5) \right]$$

### 3.2 Numerical Results
- **Resamples**: $B = 5,000$
- **Observed Paired Mean Difference**: $+0.5221$ percentage points (+0.005221)
- **Bootstrap Replicate Mean**: $+0.5027$ percentage points (+0.005027) (empirical mean across the 5,000 bootstrap replicate means)
- **2.5th Percentile**: $-2.3501$ percentage points (-2.3501 pp)
- **97.5th Percentile**: $+2.9050$ percentage points (+2.9050 pp)
- **Authoritative 95% Bootstrap Confidence Interval ($B=5,000$)**:
  $$\mathbf{[-2.3501, \; +2.9050] \text{ percentage points}} \quad (\mathbf{[-2.35, \; +2.90] \text{ pp}})$$

---

## 4. Reconciliation with 10,000-Resample Result

To resolve the discrepancy between the $B=5,000$ protocol specification and the $B=10,000$ report in V4.1:

| Bootstrap Configuration | Seed | Resample Count $B$ | 2.5th Percentile (pp) | 97.5th Percentile (pp) | 95% Confidence Interval (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Protocol Authoritative** | **42** | **5,000** | **-2.3501 pp** | **+2.9050 pp** | **[-2.35, +2.90] pp** |
| V4.1 Reported | 42 | 10,000 | -2.3507 pp | +3.0424 pp | **[-2.35, +3.04] pp** |
| Monte Carlo Difference | — | $+5,000$ | $-0.0006$ pp | $+0.1374$ pp | — |

### Root Cause of Difference:
With an $N=8$ sample size, the discrete support of the mean difference contains $8^8 / 8! \approx 16,777,216 / 40,320$ distinct multiset compositions. In small samples, the extreme tails of the resampled distribution are sensitive to how frequently combinations containing multiple copies of the outlier point `R04` ($-8.09$ pp) or `R06` ($+6.81$ pp) are drawn. The slight widening from $+2.90$ pp to $+3.04$ pp is **pure Monte Carlo variance** inherent to pseudo-random resampling, not an experimental discrepancy or data mutation. Both intervals span zero.

---

## 5. Non-Parametric Hypothesis Test: Wilcoxon Signed-Rank

The non-parametric Wilcoxon signed-rank test was independently verified on the paired differences:
- **Null Hypothesis $H_0$**: The median paired difference between O42-R2 and O02 is zero.
- **Zero Differences (Ties)**: 4 recordings (`R01`, `R02`, `R07`, `R08`) exhibit $\Delta_i = 0.0000$ pp. Following standard Wilcoxon protocol, zero differences are excluded, yielding $N_{\text{effective}} = 4$.
- **Absolute Differences and Ranks**:
  1. $|\Delta_{\text{R03}}| = 2.6354$ pp $\to$ Rank 1 (sign: $+$)
  2. $|\Delta_{\text{R05}}| = 2.8196$ pp $\to$ Rank 2 (sign: $+$)
  3. $|\Delta_{\text{R06}}| = 6.8067$ pp $\to$ Rank 3 (sign: $+$)
  4. $|\Delta_{\text{R04}}| = 8.0852$ pp $\to$ Rank 4 (sign: $-$)
- **Rank Sums**:
  - $W^{+} = 1 + 2 + 3 = 6$
  - $W^{-} = 4$
  - Test statistic: $W = \min(W^{+}, W^{-}) = 4.0$
- **Exact Two-Tailed $p$-value**:
  $$p = 2 \times \frac{7}{16} = 0.8750$$
- **Decision**: Fail to reject $H_0$ ($p = 0.8750 \gg 0.05$).

---

## 6. Corrected Scientific Interpretation & Statistical Wording

In accordance with rigorous statistical reporting standards, all frequentist-to-Bayesian misinterpretations are formally retracted:
- **Retracted Statement**: *"The probability that the true effect is harmful is 60%."* (This is an invalid pseudo-Bayesian interpretation of a frequentist bootstrap distribution).
- **Authoritative Scientific Conclusion**:
  > *"The paired bootstrap confidence interval includes zero, and the Wilcoxon signed-rank test does not establish a statistically significant corpus-level difference."*
