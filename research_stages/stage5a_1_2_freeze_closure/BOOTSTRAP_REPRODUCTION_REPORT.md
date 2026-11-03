# Stage 5A.1.2: Paired Clustered Bootstrap Reproduction and Statistical Report

**Audit Date**: November 3, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Target Subsystem**: Paired Clustered Bootstrap Statistical Uncertainty & Hypothesis Testing  
**Audit Status**: REPRODUCED, RECONCILED, AND FULLY QUALIFIED  
**Repository State**: Git Commit `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`

---

## 1. Executive Summary and Reconciliation of Replicates

This report provides the complete reproduction and statistical interpretation of the paired recording-cluster bootstrap evaluations across Stage 5A and Stage 5A.1.

### 1.1 Resolution of the $B = 2,000$ vs $B = 10,000$ Replicate Discrepancy
- **Stage 5A Run**: Executed with $B = 2,000$ replicates using `np.random.default_rng(42)`. Reported `C_vs_A` cpWER difference of $-1.02\%$ with 95% CI $[-2.13\%, -0.12\%]$ ([`stage5a/metrics/stage5a_paired_bootstrap_differences.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/metrics/stage5a_paired_bootstrap_differences.csv)).
- **Stage 5A.1 Run**: Executed with $B = 2,000$ replicates using `np.random.seed(42)` and sequential `np.random.choice`. Reported `J_vs_H` cpWER difference of $-1.02\%$ with 95% CI $[-2.07\%, -0.10\%]$ ([`stage5a_1/bootstrap/stage5a_1_paired_bootstrap_differences.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/bootstrap/stage5a_1_paired_bootstrap_differences.csv)).
- **Previous Audit Report ($B = 10,000$)**: In `STAGE5A_1_FINAL_AUDIT_REPORT.md` (Section 13), the text labeled the analysis as $B = 10,000$ while directly quoting the $[-2.07\%, -0.10\%]$ interval from the $B = 2,000$ CSV output.
- **Sensitivity Verification**: Executing an actual fresh $B = 10,000$ paired cluster bootstrap produces a 95% CI of $[-2.11\%, -0.10\%]$, confirming that the nominal cpWER reduction strictly excludes zero regardless of whether $B = 2,000$ or $B = 10,000$ replicates are drawn.

---

## 2. Reproduction Results Table

| Configuration / Experiment | Comparison | Metric | Point Diff ($\Delta$) | 95% Bootstrap CI | Includes Zero? | Win / Loss / Tie | Replicate Match Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 5A.1 Exact Script Replication** ($B=2000$) | J vs H | cpWER | **-1.02%** | **[-2.07%, -0.10%]** | **No** | 5 / 1 / 2 | Matches Stage 5A.1 CSV bit-for-bit |
| **Sensitivity Analysis** ($B=10000$) | J vs H | cpWER | **-1.02%** | **[-2.11%, -0.10%]** | **No** | 5 / 1 / 2 | Excludes zero across 10k draws |
| **Stage 5A RNG Protocol Replication** ($B=2000$) | J vs H | cpWER | **-1.02%** | **[-2.13%, -0.12%]** | **No** | 5 / 1 / 2 | Matches Stage 5A CSV closely |
| **Equal Fusion vs ResNet** ($B=10000$) | J vs I | cpWER | +0.61% | [-3.73%, +6.27%] | **Yes** | 3 / 3 / 2 | Wide interval crossing zero |
| **Reliability Gate vs Equal Fusion** ($B=10000$) | K vs J | DER (c0) | -0.25% | [-0.76%, 0.00%] | **Yes** | 1 / 0 / 7 | Upper bound touches 0.0000 exactly |

---

## 3. Mathematical Analysis of the Gate-vs-Equal-Fusion DER Boundary

The Stage 5A.1 report noted that the 95% bootstrap CI for Condition K (Gate) vs Condition J (Equal Fusion) on macro DER was $[-0.76\%, 0.00\%]$. An examination of the unrounded recording-level values clarifies this boundary:

### Recording-by-Recording DER Breakdown ($K$ vs $J$)
| Recording ID | Gate DER ($K$) | Equal Fusion DER ($J$) | Point Difference ($K - J$) |
| :--- | :---: | :---: | :---: |
| `R01_sara_sharif` | 0.3806 | 0.3806 | **0.0000** |
| `R02_dennis_skinner` | 0.6231 | 0.6231 | **0.0000** |
| `R03_disabled_parking` | 0.3354 | 0.3354 | **0.0000** |
| `R04_piers_robinson` | 0.4039 | 0.4039 | **0.0000** |
| `R05_southwest_pilot` | 0.5875 | 0.5875 | **0.0000** |
| `R06_rich_karen` | 0.7056 | 0.7258 | **-0.0202** |
| `R07_bwc_cheap_shot` | 0.4516 | 0.4516 | **0.0000** |
| `R08_uk_parliament` | 0.1858 | 0.1858 | **0.0000** |
| **Macro Average** | **0.4592** | **0.4617** | **-0.002525** |

### Combinatorial Explanation
1. In **7 out of 8 recordings**, the gate produces DER outputs that are bitwise identical to equal fusion ($\Delta = 0.0000$).
2. The *only* recording where the gate differs from equal fusion is `R06_rich_karen`, where it lowers DER by $0.0202$.
3. When drawing a cluster bootstrap resample of 8 recordings with replacement, the probability that `R06` is omitted entirely is:
   $$P(\text{R06 omitted}) = \left( \frac{7}{8} \right)^8 \approx 34.36\%$$
4. In every resample where `R06` is omitted, the mean difference is **identically 0.0000**.
5. When `R06` is included, the mean difference is strictly negative ($-0.0202 \cdot k / 8 \le -0.0025$).
6. Because $34.36\% > 2.5\%$, the 97.5th percentile of the bootstrap distribution is **identically 0.000000**.
7. Therefore, the interval $[-0.007575, 0.000000]$ **includes zero as an attainable boundary value** in more than one-third of all resamples.

**Conclusion**: The gate cannot be claimed to have statistically established downstream DER superiority over equal fusion.

---

## 4. Rigorous Statistical Interpretation

### 4.1 Sample Size and Cluster Limitations
All bootstrap calculations resample from **eight recording clusters**. Drawing 2,000 or 10,000 replicates does not generate independent speech data; it merely quantifies sampling variability over the observed sample of eight sessions. 

The appropriate scientific phrasing is:
> "Equal affinity fusion showed a nominal exploratory cpWER improvement over TitaNet under the paired recording-cluster bootstrap. The result is based on eight recording clusters and should not be interpreted as definitive evidence of generalization."

### 4.2 Multiple Comparison Caveat
In the Stage 5A and 5A.1 evaluations, multiple downstream metrics (DER collar 0, DER collar 0.25, JER, cpWER, SAWER, SACC, Speaker MAE) were evaluated across multiple pairwise method contrasts without family-wise error rate (FWER) or false discovery rate (FDR) adjustments. The observed cpWER significance ($p < 0.05$) is exploratory.

### 4.3 Asymmetric System Comparisons
Equal fusion does **NOT** outperform both individual encoders on cpWER:
- TitaNet cpWER: 38.07%
- Equal Fusion cpWER: 37.05% ($\Delta = -1.02\%$ vs TitaNet)
- ResNet293 cpWER: **36.44%** ($\Delta = +0.61\%$ vs Equal Fusion)

While Equal Fusion outperforms TitaNet on cpWER, ResNet293-LM alone achieves the lowest overall cpWER (36.44%), and the difference between Equal Fusion and ResNet293 is within statistical uncertainty (95% CI: $[-3.73\%, +6.27\%]$, including zero).

---

## 5. Summary of Statistical Findings for Freeze

1. **cpWER vs TitaNet**: Statistically significant nominal reduction (-1.02 percentage points, 95% CI: $[-2.07\%, -0.10\%]$ under $B=2,000$ and $[-2.11\%, -0.10\%]$ under $B=10,000$).
2. **cpWER vs ResNet**: Not statistically significant (+0.61 percentage points, 95% CI: $[-3.73\%, +6.27\%]$). ResNet is numerically superior.
3. **DER vs TitaNet**: Not statistically significant (-0.05 percentage points, 95% CI: $[-0.89\%, +0.89\%]$).
4. **Gate vs Equal Fusion**: Not statistically significant downstream (-0.25 percentage points DER, 95% CI: $[-0.76\%, 0.00\%]$; +0.69 percentage points cpWER, 95% CI: $[-0.11\%, +2.18\%]$).
