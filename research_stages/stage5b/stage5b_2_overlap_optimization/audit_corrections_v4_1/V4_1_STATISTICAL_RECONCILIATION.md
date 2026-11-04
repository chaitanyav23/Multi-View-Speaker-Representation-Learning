# Stage 5B.2 V4.1: Statistical Reconciliation Report

**Audit Phase**: Stage 5B.2 V4.1 Statistical Testing & Significance Analysis  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: AUDITED & RECONCILED

---

## 1. Objective

Provide rigorous statistical analysis comparing the matched single-speaker baseline (`O02`) against the overlap-attributed configuration (`O42-R2`) using the true, cryptographically verified benchmark values.

---

## 2. Benchmark Metric Inputs (Collar 0 ms)

| Recording ID | Description / Nature | O02 DER (%) | O42-R2 DER (%) | Delta (pp) | Direction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | Low overlap courtroom testimony | 31.19% | 31.19% | +0.00% | Tie |
| `R02_dennis_skinner` | Parliamentary noisy debate | 64.67% | 64.67% | +0.00% | Tie |
| `R03_disabled_parking` | Traffic confrontation, noisy | 34.29% | 36.92% | +2.64% | Loss |
| `R04_piers_robinson` | High overlap broadcast debate | 39.35% | 31.26% | -8.09% | **Win** |
| `R05_southwest_pilot` | Cockpit / ATC radio comms | 60.38% | 63.20% | +2.82% | Loss |
| `R06_rich_karen` | Outdoor verbal dispute | 71.07% | 77.87% | +6.81% | Loss |
| `R07_bwc_cheap_shot` | Police BWC, body cam noise | 41.98% | 41.98% | +0.00% | Tie |
| `R08_uk_parliament` | Clean chamber acoustics | 16.28% | 16.28% | +0.00% | Tie |
| **Macro Average** | **Unweighted Mean (N=8)** | **44.90%** | **45.42%** | **+0.52%** | **Loss** |

---

## 3. Statistical Testing

### 3.1 Win / Loss / Tie Distribution
- **Wins**: 1 / 8 (12.5%) — `R04_piers_robinson` (-8.09 percentage points)
- **Losses**: 3 / 8 (37.5%) — `R03_disabled_parking` (+2.64 pp), `R05_southwest_pilot` (+2.82 pp), `R06_rich_karen` (+6.81 pp)
- **Ties**: 4 / 8 (50.0%) — `R01`, `R02`, `R07`, `R08` (0.00 pp change)

### 3.2 Non-Parametric Hypothesis Test (Wilcoxon Signed-Rank Test)
Testing null hypothesis $H_0: \text{Median}(\text{DER}_{\text{O42-R2}} - \text{DER}_{\text{O02}}) = 0$:
- Excluded zero-difference pairs ($N_{\text{ties}} = 4$).
- Effective sample size: $N_{\text{effective}} = 4$.
- Test statistic: $W = 4.0$.
- Two-tailed $p$-value: $p = 0.8750$.
- **Decision**: Fail to reject $H_0$ ($p \gg 0.05$). Overlap attribution does not produce a statistically significant change in DER across the benchmark recordings.

### 3.3 Bootstrap 95% Confidence Interval
A non-parametric bootstrap resampling ($B = 10,000$ iterations with replacement over all 8 recordings) was executed:
- **Mean Delta**: $+0.52\%$
- **95% Bootstrap CI**: $[-2.35\%, +3.04\%]$
- **Interpretation**: The confidence interval broadly spans zero. The probability that the true macro effect is harmful ($\Delta > 0$) is ~60%, while the probability that it is beneficial ($\Delta < 0$) is ~40%.

---

## 4. Scientific Conclusions

1. **Macro Degradation**: On average across the 8 real-world recordings, attributing detected overlap using dual-encoder prototype nearest neighbor (M2) causes a slight net degradation in DER ($44.90\% \to 45.42\%$, $+0.52\%$).
2. **Heavy Overlap Specialization**: In `R04_piers_robinson` (characterized by high-density, overlapping structured speech), overlap attribution yields a substantial DER reduction of $-8.09\%$ ($39.35\% \to 31.26\%$).
3. **In-the-Wild Instability**: In unconstrained or reverberant environments (`R03`, `R05`, `R06`), false-positive overlap detection combined with secondary attribution introduces spurious speaker labels, inflating DER by $+2.64\%$ to $+6.81\%$.
4. **General Recommendation**: Indiscriminate overlap attribution across all acoustic environments is not warranted. Downstream Stage 5B.4 integration must gate overlap attribution conditionally based on overlap confidence or acoustic quality metrics.
