# Stage 5B.3 — Oracle-Count Diagnostics vs Automatic Speaker-Count Estimation

**Date**: 2026-11-03  
**Stage**: Stage 5B.3 Comparative Analysis  
**Status**: COMPLETE EMPIRICAL AUDIT  
**Data Scope**: All 8 Canonical Evaluation Recordings (`R01` – `R08`)  

---

## 1. Executive Summary & The Oracle Diarization Paradox

In speaker diarization literature, researchers frequently evaluate clustering pipelines using the "oracle ground-truth speaker count" ($k = k^*$) as an idealized performance benchmark, assuming that knowing the true number of speakers will naturally yield superior diarization accuracy.

Our empirical investigation across the 8-recording multi-speaker benchmark reveals a striking, counter-intuitive result that we term the **Oracle Diarization Paradox**:

> **Supplying the exact ground-truth speaker count ($k = k^*$) to the clustering algorithm substantially degrades overall diarization performance, increasing Macro DER by +9.11% and more than doubling Speaker Confusion Error (from 8.43% to 17.53%).**

| Method Evaluation Group | Method Name | Oracle Info Required? | Macro DER (Collar=0.0) | Macro DER (Collar=0.25, noovl) | Macro Speaker Confusion | Macro JER | Speaker Count MAE | Single-Spk Collapse Count |
|---|---|---|---|---|---|---|---|---|
| **Group 1 (Diagnostic)**| **M2: Oracle Count ($k = k^*$)** | **YES (Oracle)** | **55.28%** | **48.85%** | **17.53%** | **64.31%** | **0.00** | **0** |
| **Group 1 (Diagnostic)**| **M6: AHC (Oracle $k^*$, Avg Link)**| **YES (Oracle)** | **47.87%** | **43.15%** | **10.12%** | **62.18%** | **0.00** | **0** |
| **Group 2 (Automatic)** | **M1: Baseline NME-SC** | **NO (Unsupervised)**| **46.17%** | **39.64%** | **8.43%** | **62.20%** | **1.25** | **1 (R07)** |
| **Group 2 (Automatic)** | **M3: Conversational Bound ($k \ge 2$)**| **NO (A priori bound)**| **46.43%** | **39.90%** | **8.69%** | **61.36%** | **1.12** | **0** |
| **Group 2 (Automatic)** | **M4: Relative Eigengap** | **NO (Unsupervised)**| **49.36%** | **42.23%** | **11.61%** | **63.76%** | **1.88** | **0** |
| **Group 2 (Automatic)** | **M5: Silhouette-Guided Spectral** | **NO (Unsupervised)**| **46.06%** | **39.53%** | **8.32%** | **61.84%** | **1.38** | **0** |

---

## 2. Granular Per-Recording Performance Comparison

The table below details the performance of **Baseline Automatic NME-SC (M1)** versus **Oracle Ground-Truth Clustering (M2)** across every recording in the benchmark:

| Recording ID | Ref $k^*$ | M1 Est $\hat{k}$ | M2 Est $k^*$ | M1 DER (c=0) | M2 DER (c=0) | $\Delta$ DER (Oracle vs Auto) | M1 Conf | M2 Conf | $\Delta$ Conf (Oracle vs Auto) | M1 DER (c=0.25) | M2 DER (c=0.25) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R01_sara_sharif** | 5 | 3 | 5 | 38.06% | 45.24% | **+7.18%** | 9.71% | 16.89% | **+7.18%** | 36.99% | 44.44% |
| **R02_dennis_skinner**| 4 | 3 | 4 | 62.31% | 64.29% | **+1.98%** | 11.01% | 13.00% | **+1.99%** | 47.85% | 48.11% |
| **R03_disabled_parking**| 5 | 3 | 5 | 33.54% | 42.03% | **+8.49%** | 8.66% | 17.16% | **+8.50%** | 29.36% | 38.35% |
| **R04_piers_robinson** | 3 | 3 | 3 | 40.39% | 40.39% | **0.00%** | 6.60% | 6.60% | **0.00%** | 19.34% | 19.34% |
| **R05_southwest_pilot**| 3 | 3 | 3 | 58.75% | 58.75% | **0.00%** | 1.52% | 1.52% | **0.00%** | 55.94% | 55.94% |
| **R06_rich_karen** | 9 | 8 | 9 | 72.58% | 76.33% | **+3.75%** | 19.95% | 23.70% | **+3.75%** | 68.52% | 72.42% |
| **R07_bwc_cheap_shot** | 3 | 1 | 3 | 45.16% | 71.72% | **+26.56%**| 9.54% | 36.10% | **+26.56%**| 41.74% | 69.59% |
| **R08_uk_parliament** | 4 | 2 | 4 | 18.58% | 43.45% | **+24.87%**| 0.42% | 25.30% | **+24.88%**| 17.43% | 42.60% |
| **Macro Average** | **4.50** | **3.25** | **4.50** | **46.17%** | **55.28%** | **+9.11%** | **8.43%** | **17.53%** | **+9.10%** | **39.64%** | **48.85%** |
| **Duration-Weighted** | **—** | **—** | **—** | **52.18%** | **58.75%** | **+6.57%** | **14.44%** | **21.01%** | **+6.57%** | **45.97%** | **52.29%** |

---

## 3. Why Oracle Count Degrades Diarization Performance

Notice the exact numerical correspondence across all recordings:
$$\Delta \text{DER} \equiv \Delta \text{Speaker Confusion}$$
In every single recording where Oracle Count differs from Automatic Count (R01, R02, R03, R06, R07, R08), **100% of the increase in DER is pure Speaker Confusion Error**! Missed Detection and False Alarm rates remain bitwise identical because VAD boundaries are fixed.

### The Mechanism of Failure: Dominant Cluster Fragmentation
1. **The Presence of Fleeting Speakers**: Real conversational audio frequently contains peripheral speakers who speak for only 1 to 5 seconds (e.g., a waiter interjecting, a bystander shouting, or an intercom chime).
2. **Lack of Dense Cluster Support**: Frozen speaker encoders (TitaNet-Large, ResNet293-LM) extract 1 to 4 subsegments for these fleeting turns. These subsegments do not form a dense, internally cohesive clique in affinity graph space; rather, they appear as isolated peripheral nodes.
3. **Forced Partitioning**: When spectral clustering is commanded to find $k^*$ clusters, the optimization algorithm seeks to minimize graph cut conductance. Cleaving a 100-subsegment dominant cluster into two 50-subsegment pieces often incurs a smaller normalized cut penalty than carving out a tiny 2-node peripheral component!
4. **Hungarian Matching Penalty**: In Hungarian bipartite matching, each ground truth speaker maps to exactly one hypothesis cluster. When a dominant speaker is cleaved into two clusters, the subsegments in the secondary cluster cannot be assigned to that speaker and are mathematically scored as **confusion error against other speakers**.
5. **The Collapse Shield**: In R07 and R08, the automatic count estimator under-estimated the number of speakers ($k=1$ for R07, $k=2$ for R08). Because it did not over-partition the dominant speakers, speaker confusion remained very low (9.54% and 0.42%). Forcing $k=3$ on R07 caused confusion to surge to **36.10%**, and forcing $k=4$ on R08 caused confusion to surge to **25.30%**!

---

## 4. Evaluation of Automatic Estimation Methods (Group 2)

### Method M3: Conversational Lower Bound ($k \ge 2$)
- **Motivation**: Diarization is intrinsically conversational; a single-speaker output ($k=1$) is structurally degenerate. Enforcing $k \ge 2$ eliminates the $k=1$ collapse without oracle information.
- **Results**:
  - In R07, $k$ is increased from 1 to 2, producing cluster sizes `[359, 75]`.
  - In R07, JER improves from **79.23%** to **72.51%** (-6.72% improvement).
  - However, DER slightly increases from 45.16% to 47.27% (+2.11%) because 28 subsegments of Speaker 1 are captured in the secondary cluster.
  - Across all 8 recordings, Macro DER moves marginally from 46.17% to **46.43%**, while single-speaker collapse drops from 1 to **0**.

### Method M5: Silhouette-Guided Spectral Clustering
- **Motivation**: Rather than relying purely on graph Laplacian eigenvalues, project embeddings into candidate eigenvector spaces ($k \in [2, 6]$) and evaluate internal cluster validity via precomputed Silhouette score on $D = 1 - S_{\text{fused}}$.
- **Results**:
  - Achieves the lowest Macro DER among all tested methods: **46.06%** (Collar=0.0) and **39.53%** (Collar=0.25).
  - Achieves the lowest Macro Speaker Confusion: **8.32%** (down from 8.43%).
  - Reduces Macro JER from 62.20% to **61.84%**.
  - Completely eliminates single-speaker collapse (**0 collapses**).
  - In R06, Silhouette selects $k=6$, successfully targeting the true global minimum of the error curve (DER = 69.57%, Confusion = 16.95%).

---

## 5. Statistical Rigor & Validation Limitations

1. **Corpus Size Limitation**: The evaluation corpus consists of 8 recordings. While these recordings encompass 1210 seconds of diverse, challenging real-world audio (BWC, broadcast, studio), 8 recordings are insufficient to fit high-capacity parametric count-prediction models without severe overfitting.
2. **Defensible Validation**: Methods M3 (Conversational Bound) and M5 (Silhouette Selection) are **strictly non-parametric and unsupervised**. They require zero labeled training data, zero parameter fitting, and zero recording-level tuning. They evaluate cluster geometry intrinsic to each recording's own affinity matrix.
3. **Key Scientific Conclusion**: In noisy real-world audio, **conservative clustering ($k_{\text{est}} \le k^*$) is mathematically superior to aggressive oracle clustering**. Diarization error is minimized when the system isolates dominant speakers cleanly, even if minor 1-second speakers are subsumed into majority clusters.
