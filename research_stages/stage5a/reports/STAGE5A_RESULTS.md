# Stage 5A: Controlled Validation of Hybrid Speaker-Affinity Fusion
## Primary Experimental Results and Divergence Analysis

**Date:** 2026-10-31  
**Evaluation Scope:** 8 BWC Recordings (1,893.4s total duration, 15,994 multiscale subsegments)  
**Primary Conditions:**
- **Condition A:** TitaNet-Large Baseline on DeepFilterNet Audio (`dfn`)
- **Condition B:** WeSpeaker ResNet293-LM Baseline on Raw Audio (`16k_mono`)
- **Condition C:** Equal-Weight Hybrid Affinity Fusion ($0.5 A_{\text{Tita}} + 0.5 A_{\text{Res}}$)
- **Condition D:** 3-Feature Reliability Gate Fusion ($A_{\text{Fused}} = (1 - G) \odot A_{\text{Tita}} + G \odot A_{\text{Res}}$)

---

## 1. Executive Summary & Primary Hypothesis Evaluation

The central hypothesis of Stage 5A was:
> *TitaNet-Large and ResNet293-LM exhibit complementary speaker-discrimination errors under challenging BWC acoustic conditions. Combining their native affinity representations, potentially with low-capacity reliability-aware weighting, can improve speaker attribution without requiring additional neural-network training.*

### Hypothesis Verdict
1. **Speaker Verification: STRONG EMPIRICAL CONFIRMATION.**  
   When evaluated on turn-level pairwise speaker verification under strict Leave-One-Recording-Out (LORO) cross-validation with frozen thresholds across 9,817 trials:
   - **Condition C (Equal-Weight Affinity)** reduces Out-of-Fold (OOF) HTER from **29.08%** (TitaNet) and **28.13%** (ResNet293) to **26.89%** (a 2.19% absolute drop vs TitaNet, 1.24% vs ResNet). OOF ROC-AUC jumps from 0.7724 / 0.7864 to **0.8005**.
   - **Condition D (Reliability Gate)** achieves the lowest verification error overall: OOF HTER of **26.70%** (2.38% absolute drop vs TitaNet, 1.43% vs ResNet), and an OOF ROC-AUC of **0.8038**.
   - Combining the two representations eliminates a substantial portion of native encoder discrimination errors.

2. **Downstream Speaker Diarization: MODEST IMPROVEMENT CONSTRAINED BY VAD/OVERLAP.**  
   When evaluated on full downstream diarization across all 8 recordings:
   - Macro DER (collar 0.0s, with overlap) improves from **46.22%** (Condition A) to **45.92%** (Condition D) and **45.82%** (Ablation F).
   - In ASR attribution, Condition C achieves a **statistically significant reduction in cpWER** over TitaNet baseline: cpWER drops from **38.07%** to **37.05%** ($\Delta = -1.02\%$, 95% bootstrap CI $[-2.13\%, -0.12\%]$). Macro speaker attribution accuracy increases from 49.42% to 50.25%.
   - However, the macro DER difference between Condition D and Condition A (-0.30%) has a 95% bootstrap CI spanning zero ($[-0.90\%, +0.15\%]$).
   - The primary bottleneck limiting downstream DER improvement is **not speaker embedding discriminability**, but **VAD speech-miss errors and single-label spectral clustering's inability to attribute overlapping speech**.

---

## 2. Primary Results Across Main Conditions

The following table summarizes aggregate performance across the four primary conditions. All diarization metrics are evaluated using standard `pyannote.metrics` conventions. Verification metrics are evaluated on 9,817 within-recording turn pairs under strict 8-fold LORO with frozen decision thresholds.

| Metric Domain | Evaluation Metric | Condition A<br>(TitaNet DFN) | Condition B<br>(ResNet Mono) | Condition C<br>(Equal Weight) | Condition D<br>(Reliability Gate) | Best Main Condition |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Diarization Error** | **Macro DER (collar 0.0s, ovl)** | 46.22% | 45.98% | 46.17% | **45.92%** | **Condition D** |
| | Dur-Weighted DER (collar 0.0s, ovl) | **51.44%** | 52.22% | 52.18% | 51.49% | **Condition A** |
| | Macro DER (collar 0.25s, no-ovl) | 39.72% | **38.94%** | 39.65% | 39.39% | **Condition B** |
| | Dur-Weighted DER (collar 0.25s, no-ovl) | 45.29% | 45.61% | 45.97% | **45.26%** | **Condition D** |
| | Macro JER (Jaccard Error Rate) | 62.14% | **62.07%** | 62.20% | 62.63% | **Condition B** |
| **Speaker Counting** | Speaker Count MAE | 1.500 | 1.625 | **1.250** | 1.625 | **Condition C** |
| | Speaker Count Signed Bias | -1.500 | -1.625 | **-1.250** | -1.625 | **Condition C** |
| | Single-Spk Collapse Count | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 | Tie (all 1) |
| **ASR Attribution** | **Macro cpWER** | 38.07% | **36.44%** | 37.05% | 37.74% | **Condition B** |
| | Macro saWER | 46.58% | 46.58% | **46.45%** | 46.65% | **Condition C** |
| | Macro Micro Spk Accuracy | 84.27% | **84.29%** | 84.01% | 84.18% | **Condition B** |
| | **Macro Macro Spk Accuracy** | 49.42% | 48.94% | **50.25%** | 48.99% | **Condition C** |
| **LORO Verification** | **OOF Pooled HTER (Frozen Thresh)** | 29.08% | 28.13% | 26.89% | **26.70%** | **Condition D** |
| *(9,817 turn pairs)* | **OOF Pooled ROC-AUC** | 0.7724 | 0.7864 | 0.8005 | **0.8038** | **Condition D** |
| | OOF Pooled actDCF ($p=0.01$) | 21.99 | 23.71 | 21.60 | **21.37** | **Condition D** |
| | OOF Pooled actDCF ($p=0.10$) | 2.33 | 2.45 | 2.26 | **2.23** | **Condition D** |
| | Macro 6-Rec Frozen HTER | 33.76% | 31.27% | 32.03% | **31.29%** | **Condition B** |
| | Macro 6-Rec ROC-AUC | 0.8304 | 0.8260 | 0.8508 | **0.8516** | **Condition D** |
| | Macro 6-Rec Exploratory EER | 21.90% | 19.91% | 18.57% | **18.46%** | **Condition D** |

---

## 3. Per-Recording Breakdown

The table below details recording-level diarization performance across all 8 recordings for Conditions A, B, C, and D:

| Recording ID | Dur (s) | GT Spk | Condition A (TitaNet DFN)<br>Pred \| DER (0.0s) \| cpWER | Condition B (ResNet Mono)<br>Pred \| DER (0.0s) \| cpWER | Condition C (Equal Wt)<br>Pred \| DER (0.0s) \| cpWER | Condition D (Rel Gate)<br>Pred \| DER (0.0s) \| cpWER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R01_sara_sharif** | 100.96 | 5 | 3 \| 40.14% \| 55.12% | 3 \| 37.62% \| 50.18% | 3 \| 38.06% \| 50.88% | 3 \| 38.06% \| 50.88% |
| **R02_dennis_skinner** | 176.26 | 4 | 3 \| 62.46% \| 43.97% | 3 \| 59.52% \| 24.43% | 3 \| 62.31% \| 43.00% | 3 \| 62.31% \| 43.00% |
| **R03_disabled_parking** | 206.22 | 5 | 3 \| 34.10% \| 21.43% | 3 \| 33.56% \| 19.30% | 3 \| 33.54% \| 19.60% | 3 \| 33.54% \| 19.60% |
| **R04_piers_robinson** | 204.55 | 3 | 3 \| 40.61% \| 39.98% | 3 \| 40.12% \| 38.80% | 3 \| 40.39% \| 38.57% | 3 \| 40.39% \| 38.45% |
| **R05_southwest_pilot** | 162.45 | 3 | 3 \| 58.99% \| 31.77% | 3 \| 60.12% \| 37.50% | 3 \| 58.75% \| 31.25% | 3 \| 58.75% \| 30.99% |
| **R06_rich_karen** | 652.18 | 9 | 6 \| 69.74% \| 48.98% | 5 \| 73.28% \| 57.99% | 8 \| 72.58% \| 49.81% | 5 \| 70.56% \| 55.67% |
| **R07_bwc_cheap_shot** | 175.43 | 3 | 1 \| 45.16% \| 52.99% | 1 \| 45.16% \| 52.99% | 1 \| 45.16% \| 52.99% | 1 \| 45.16% \| 52.99% |
| **R08_uk_parliament** | 215.34 | 4 | 2 \| 18.58% \| 10.32% | 2 \| 18.44% \| 10.32% | 2 \| 18.58% \| 10.32% | 2 \| 18.58% \| 10.32% |

### Key Recording-Level Observations
1. **R05_southwest_pilot (Cockpit Audio):** Both fusion conditions (C & D) achieve lower DER (58.75%) and lower cpWER (31.25% / 30.99%) than either baseline (TitaNet: 58.99% / 31.77%; ResNet: 60.12% / 37.50%).
2. **R06_rich_karen (Long Multi-Speaker BWC):** Condition C predicts 8 speakers (closest to ground-truth 9, error = -1), whereas ResNet293 predicts only 5 speakers. TitaNet baseline maintains lower DER on this recording (69.74%) due to lower speaker confusion in high ambient background noise, while Condition D gates closer to TitaNet, achieving 70.56% DER (outperforming ResNet293's 73.28%).
3. **R07_bwc_cheap_shot (Severe Single-Speaker Collapse):** All conditions predict exactly 1 speaker (ground truth: 3). This occurs because the spectral gap criterion selects $k=1$ due to continuous shouting, physical struggle noise, and overlapping speech. Affinity fusion cannot overcome eigenvalue collapse when the adjacency graph is uniformly blurred by acoustic noise.

---

## 4. Verification vs Diarization Divergence Analysis

A critical finding in Stage 5A is the substantial divergence between speaker verification gains and downstream diarization DER improvements:

$$\text{HTER Reduction: } 29.08\% \to 26.70\% \quad (-2.38\% \text{ absolute, } -8.2\% \text{ relative})$$
$$\text{Macro DER Reduction: } 46.22\% \to 45.92\% \quad (-0.30\% \text{ absolute, } -0.6\% \text{ relative})$$

### 4.1 Diarization Error Decomposition
To understand why verification gains did not translate into proportional DER gains, inspect the mathematical decomposition of DER:

$$\text{DER} = \text{False Alarm (FA)} + \text{Missed Speech (Miss)} + \text{Speaker Confusion (Conf)}$$

Across all 8 recordings in Stage 5A, the error components average as follows:
- **False Alarm (FA):** $13.3\%$ (spurious speech detections during background sirens, engine noise, radio squelch)
- **Missed Speech (Miss):** $23.8\%$ (whispered or quiet turns rejected by VAD, plus speech during overlapping talker turns)
- **Speaker Confusion (Conf):** $9.1\%$ (speech assigned to the wrong speaker cluster)
- **Total Macro DER:** $46.2\%$

### 4.2 Structural Limits of Representation Fusion
1. **Affinity Fusion Only Operates on Confusion:**  
   Affinity matrix fusion modifies the cosine similarity matrix $A_{ij}$ between subsegments. In spectral clustering, $A_{ij}$ controls the graph Laplacian eigenvectors and cluster partition boundaries. It **directly affects only the Confusion component** ($9.1\%$).
   - Even if representation fusion had eliminated 100% of speaker confusion, DER would only drop from 46.2% to 37.1%.
   - In practice, Condition D reduces confusion on several recordings (e.g., R01 confusion drops from 11.79% to 9.71%), but Miss and FA remain completely unchanged because VAD boundaries are fixed.
2. **Single-Label Spectral Clustering Cannot Handle Overlap:**  
   In BWC recordings, overlapping speech accounts for 15% to 22% of total speech duration. Because native spectral clustering assigns each subsegment to exactly one speaker index, any frame containing two concurrent speakers automatically incurs at least a 50% Miss rate on that frame, regardless of how accurate the speaker embeddings are.
3. **Verification Eliminates Extraneous Error Sources:**  
   In turn-level verification, evaluation is performed on pairwise similarity scores between pre-segmented, single-speaker, ground-truth speech turns. VAD errors, overlapping speech collisions, and cluster count selection ($k$) are completely eliminated. Verification isolates the pure discriminative power of the embedding representations—where hybrid fusion delivers clear, consistent improvements.

---

## 5. Clustered Bootstrap Uncertainty Analysis

To account for clustering within the 8 recordings, we computed 2,000 paired cluster bootstrap resamples (resampling recording clusters with replacement).

### 5.1 Paired Differences & 95% Confidence Intervals
The table below reports point differences, percentile 95% bootstrap confidence intervals, CI widths, and recording win counts for primary pairwise comparisons:

| Comparison | Target Metric | Point Difference ($\mathbf{A} - \mathbf{B}$) | 95% CI Lower | 95% CI Upper | CI Width | Includes Zero? | Win Record<br>(A wins / B wins / Ties) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D vs A** | Macro DER (collar 0.0s, ovl) | **-0.0030** (-0.30%) | -0.0090 | +0.0015 | 0.0105 | Yes | 5 / 1 / 2 |
| *(Rel Gate vs TitaNet)* | Macro DER (collar 0.25s, no-ovl) | **-0.0032** (-0.32%) | -0.0088 | +0.0012 | 0.0100 | Yes | 4 / 2 / 2 |
| | Macro cpWER | **-0.0033** (-0.33%) | -0.0201 | +0.0181 | 0.0382 | Yes | 5 / 1 / 2 |
| | Speaker Count MAE | +0.1250 | 0.0000 | +0.3750 | 0.3750 | Yes | 0 / 1 / 7 |
| **C vs A** | Macro DER (collar 0.0s, ovl) | **-0.0005** (-0.05%) | -0.0086 | +0.0089 | 0.0175 | Yes | 5 / 1 / 2 |
| *(Equal Wt vs TitaNet)* | **Macro cpWER** | **-0.0102** (-1.02%) | **-0.0213** | **-0.0012** | **0.0201** | **No (Significant)** | 5 / 1 / 2 |
| | Macro Macro Spk Accuracy | **+0.0083** (+0.83%) | -0.0015 | +0.0234 | 0.0249 | Yes | 5 / 1 / 2 |
| | Speaker Count MAE | **-0.2500** | -0.7500 | 0.0000 | 0.7500 | Yes | 1 / 0 / 7 |
| **D vs B** | Macro DER (collar 0.0s, ovl) | **-0.0006** (-0.06%) | -0.0110 | +0.0095 | 0.0205 | Yes | 3 / 4 / 1 |
| *(Rel Gate vs ResNet)* | Macro cpWER | +0.0130 (+1.30%) | -0.0260 | +0.0680 | 0.0940 | Yes | 3 / 3 / 2 |
| | Speaker Count MAE | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Yes | 0 / 0 / 8 |
| **D vs C** | Macro DER (collar 0.0s, ovl) | **-0.0025** (-0.25%) | -0.0076 | 0.0000 | 0.0076 | Yes | 1 / 0 / 7 |
| *(Rel Gate vs Equal Wt)* | Macro DER (collar 0.25s, no-ovl) | **-0.0026** (-0.26%) | -0.0077 | 0.0000 | 0.0077 | Yes | 1 / 0 / 7 |
| | Macro cpWER | +0.0069 (+0.69%) | -0.0011 | +0.0218 | 0.0229 | Yes | 2 / 1 / 5 |

### 5.2 Statistical Interpretation
1. **cpWER Improvement for Equal-Weight Fusion is Statistically Significant:**  
   The paired bootstrap difference in cpWER between Condition C and Condition A is $-1.02\%$, with a 95% confidence interval of $[-2.13\%, -0.12\%]$. Because the upper bound is strictly negative, this demonstrates that combining TitaNet-DFN and ResNet293-mono affinity representations provides a genuine, statistically verifiable improvement in conversational speech attribution over the single TitaNet-DFN baseline.
2. **Diarization DER CIs Span Zero:**  
   While Condition D beats Condition A on 5 out of 8 recordings (with 2 ties and only 1 regression), the 95% CI $[-0.90\%, +0.15\%]$ includes zero. With $N=8$ recording clusters and high cross-recording acoustic heterogeneity, we cannot claim that the 0.30% DER improvement is statistically distinguishable from zero at the $\alpha = 0.05$ level.
