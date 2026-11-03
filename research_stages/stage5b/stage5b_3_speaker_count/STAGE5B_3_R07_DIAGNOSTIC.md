# Stage 5B.3 — Forensic Diagnostic & Empirical Autopsy of Recording R07 (`R07_bwc_cheap_shot`)

**Date**: 2026-11-03  
**Stage**: Stage 5B.3 Forensic Investigation  
**Status**: COMPLETE EMPIRICAL AUDIT  
**Target Recording**: `R07_bwc_cheap_shot` (Law Enforcement Body-Worn Camera Altercation)  

---

## 1. Executive Summary & Core Verdict

The Stage 5B.0 preliminary report hypothesized that:
> "In R07, the fused affinity matrix cleanly separates speakers into partitions of [359, 75] and [215, 157, 62], and that NeMo's automatic eigengap estimator is 100% to blame for choosing $k=1$, reportedly inflating DER to 64.95% instead of 21.80%."

**Our rigorous independent audit has disproved this preliminary hypothesis with hard empirical data**:
1. **The Baseline Record**: Under frozen Stage 5A.1 Condition J, R07's DER is **45.16%** (collar=0.0) and **41.74%** (collar=0.25, noovl), with Speaker Confusion of **9.54%**. The previously cited 64.95% was an unverified exploratory figure.
2. **The Partitions are Real, but Fragmented**: The partitions `[359, 75]` ($k=2$) and `[215, 157, 62]` ($k=3$) are 100% mathematically reproducible. However, cross-tabulation against ground truth reveals that **they do NOT cleanly separate speakers**. Instead, forced clustering shatters the dominant Speaker 1 into multiple arbitrary clusters!
3. **Forcing Oracle Count Catastrophically Degrades DER**:
   - At $k=1$ (Automatic): DER = **45.16%**, Confusion = **9.54%**.
   - At $k=2$ (Forced): DER = **47.27%**, Confusion = **11.65%**.
   - At $k=3$ (True Oracle): DER = **71.72%**, Confusion = **36.10%** (+26.56% confusion increase!).
4. **The Root Cause**: The primary failure resides in the **acoustic representations within the affinity graph**, not merely the eigengap heuristic. Due to severe wind noise and audio clipping, Speaker 2's intra-speaker similarity is **0.3422**, whereas its inter-speaker similarity to Speaker 1 is **0.3292** (a separation margin of only $0.0130$). Speaker 3 has only 1 second of speech (2 subsegments). The affinity graph does not contain separated clusters.
5. **The Irony of the $k=1$ Collapse**: NeMo's $k=1$ selection, while technically an under-clustering failure, **actually shielded the downstream system from massive speaker confusion error**!

---

## 2. Granular Ground Truth & Subsegment Alignment

### 2.1 Audio & Ground Truth Profiles
- **Audio Duration**: 175.43 seconds (16 kHz mono WAV / DFN).
- **Total Speech Duration**: 123.00 seconds.
- **Reference Speakers ($k^* = 3$)**:
  - **Speaker 1**: 107.00 seconds ($87.0\%$ of total speech).
  - **Speaker 2**: 15.00 seconds ($12.2\%$ of total speech, fractured into brief interjections).
  - **Speaker 3**: 1.00 second ($0.8\%$ of total speech, a single distant command).
  - **Imbalance Ratio**: $107.0 : 1$ (Speaker 1 to Speaker 3).

### 2.2 Subsegment Breakdown ($N = 434$ Subsegments at Scale 4)
- Contiguous sliding windows (length $W=0.5$s, shift $S=0.25$s):
  - Clean `Speaker_1`: 305 subsegments
  - Partial boundary `Speaker_1`: 26 subsegments
  - Overlap containing `Speaker_1`: 12 subsegments
  - Clean `Speaker_2`: 31 subsegments
  - Partial boundary `Speaker_2`: 12 subsegments
  - Overlap containing `Speaker_2`: 7 subsegments
  - Clean `Speaker_3`: 2 subsegments (1.0 second total)
  - Non-Speech / Rounding Gaps: 39 subsegments

---

## 3. Reproduction & Cross-Tabulation of Cluster Partitions

We executed spectral clustering on the frozen Condition J fused affinity matrix $S_{\text{fused}} \in \mathbb{R}^{434 \times 434}$ with candidate cluster counts $k \in \{1, 2, 3\}$.

### Cross-Tabulation Matrix: Cluster Assignments vs Ground Truth Labels

```
===================================================================================================
k=1 (Automatic NeMo Baseline) -> Cluster Size: [434]
---------------------------------------------------------------------------------------------------
Cluster 0: 305 Spk1 | 26 Part_Spk1 | 12 Ovl_Spk1 | 31 Spk2 | 12 Part_Spk2 | 7 Ovl_Spk2 | 2 Spk3 | 39 NonSpk
Evaluation: DER(c=0)=45.16% | DER(c=0.25)=41.74% | Confusion=9.54% | Miss=21.50% | FA=14.12% | JER=79.23%
===================================================================================================

k=2 (Conversational Lower Bound) -> Cluster Sizes: [359, 75]
---------------------------------------------------------------------------------------------------
Cluster 0 (N=359): 277 Spk1 | 19 Part_Spk1 | 9 Ovl_Spk1 | 13 Spk2 | 7 Part_Spk2 | 3 Ovl_Spk2 | 1 Spk3 | 30 NonSpk
Cluster 1 (N= 75):  28 Spk1 |  7 Part_Spk1 | 3 Ovl_Spk1 | 18 Spk2 | 5 Part_Spk2 | 4 Ovl_Spk2 | 1 Spk3 |  9 NonSpk
Evaluation: DER(c=0)=47.27% | DER(c=0.25)=43.82% | Confusion=11.65% | Miss=21.50% | FA=14.12% | JER=72.51%
Analysis: Cluster 1 contains 28 subsegments of Spk1 and 18 subsegments of Spk2. It is an impure mixture!
===================================================================================================

k=3 (True Oracle Ground Truth) -> Cluster Sizes: [215, 157, 62]
---------------------------------------------------------------------------------------------------
Cluster 0 (N=215): 173 Spk1 | 11 Part_Spk1 | 7 Ovl_Spk1 |  4 Spk2 | 4 Part_Spk2 | 1 Ovl_Spk2 | 1 Spk3 | 14 NonSpk
Cluster 1 (N=157): 110 Spk1 |  8 Part_Spk1 | 2 Ovl_Spk1 | 15 Spk2 | 4 Part_Spk2 | 2 Ovl_Spk2 | 0 Spk3 | 16 NonSpk
Cluster 2 (N= 62):  22 Spk1 |  7 Part_Spk1 | 3 Ovl_Spk1 | 12 Spk2 | 4 Part_Spk2 | 4 Ovl_Spk2 | 1 Spk3 |  9 NonSpk
Evaluation: DER(c=0)=71.72% | DER(c=0.25)=69.59% | Confusion=36.10% | Miss=21.50% | FA=14.12% | JER=82.54%
Analysis: Catastrophic fragmentation! Speaker 1 is split into 173 (Clust 0), 110 (Clust 1), and 22 (Clust 2).
===================================================================================================
```

### Critical Finding on Diarization Matching Mechanics:
In NIST Hungarian matching between reference speakers and hypothesis clusters:
- In $k=3$, Cluster 0 (173 Spk1) is matched to `Speaker_1`.
- Cluster 1 (110 Spk1, 15 Spk2) is matched to `Speaker_2`.
- Cluster 2 (22 Spk1, 12 Spk2, 1 Spk3) is matched to `Speaker_3`.
- Consequently, all **110 subsegments of Speaker 1 in Cluster 1** and all **22 subsegments of Speaker 1 in Cluster 2** are counted as **Speaker Confusion Error**!
- That is why Confusion explodes from **9.54%** to **36.10%**, and DER increases from **45.16%** to **71.72%**!

---

## 4. Dissection of Affinity Graph Mechanics: Why Separability Fails

To establish whether the failure resides in spectral clustering or upstream in the representation space, we computed the pairwise cosine similarities directly on clean subsegments:

| Subsegment Pair Type | Sample Count | Raw TitaNet DFN Mean | Raw ResNet 16k Mono Mean | Condition J Fused Affinity Mean |
|---|---|---|---|---|
| **Speaker 1 Intra-Speaker** | 46,360 pairs | 4.8373 | 5.0365 | **0.5377 $\pm$ 0.19** |
| **Speaker 2 Intra-Speaker** | 465 pairs | 3.7488 | 3.9623 | **0.3422 $\pm$ 0.18** |
| **Speaker 1 vs Speaker 2 Inter-Speaker**| 9,455 pairs | 3.6550 | 3.7043 | **0.3292 $\pm$ 0.18** |
| **Speaker 1 vs Speaker 3 Inter-Speaker**| 610 pairs | 3.1475 | 2.9878 | **0.2245 $\pm$ 0.16** |
| **Speaker 2 vs Speaker 3 Inter-Speaker**| 62 pairs | 2.9260 | 2.6186 | **0.1814 $\pm$ 0.15** |

### Mathematical Proof of Representation Defect:
$$\Delta_{\text{separation}} = \mu(\text{Spk2 Intra}) - \mu(\text{Spk1-Spk2 Inter}) = 0.3422 - 0.3292 = \mathbf{0.0130}$$
- The separation margin between Speaker 2 and Speaker 1 is a negligible **0.0130**, while the standard deviation within Speaker 2 is **0.1812** (14 times larger than the margin!).
- Because the acoustic environment of R07 features heavy wind turbulence, microphone distortion, and physical clothing friction, the voice prints of Speaker 1 and Speaker 2 are acoustically smeared.
- In graph-theoretic terms, the Speaker 2 nodes are densely interconnected with Speaker 1 nodes. There is **no sparse cut** separating them in the affinity graph.

---

## 5. Mathematical Autopsy of NeMo Eigengap Analysis

In NeMo NME-SC, for neighborhood sparsification $p$:
$$\lambda_0 \le \lambda_1 \le \lambda_2 \le \dots \le \lambda_9$$
$$\text{gap}(k) = \lambda_{k+1} - \lambda_k, \quad k \in \{0, 1, \dots, 8\}$$

For R07 at $p=15$ ($p_{\text{hat}}$ selected by NME-SC):
- $\lambda_0 = 0.000$
- $\lambda_1 = 1.965 \implies \mathbf{\text{gap}(0) = 1.965}$
- $\lambda_2 = 2.615 \implies \text{gap}(1) = 0.650$
- $\lambda_3 = 3.123 \implies \text{gap}(2) = 0.508$
- $\lambda_4 = 3.489 \implies \text{gap}(3) = 0.366$

Because $\text{gap}(0) = 1.965$ is **three times larger** than any subsequent gap:
$$\text{argmax}_{k} \text{gap}(k) = 0 \implies k_{\text{est}} = 0 + 1 = \mathbf{1}$$

**Verdict**: The eigengap heuristic is mathematically reporting the truth: the graph has only one strongly connected community. The failure is not in the eigengap arithmetic; it is in the affinity matrix itself, which lacks the discriminative power to isolate Speaker 2 under high BWC noise.
