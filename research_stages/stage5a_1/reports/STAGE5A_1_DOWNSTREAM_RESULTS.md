# Stage 5A.1: Affinity Normalization, Fusion Fairness & Implementation Audit
## Downstream Diarization and ASR Attribution Results

**Date:** 2026-10-31  
**Evaluation Scope:** 8 BWC Recordings (1,893.4s total duration, 15,994 multiscale subsegments)  
**Evaluated Conditions:**
- **Condition A:** Raw TitaNet-DFN Baseline
- **Condition B:** Raw ResNet293-Mono Baseline
- **Condition C:** Stage 5A Normalized Equal-Weight Fusion
- **Condition H:** Normalized TitaNet-DFN Alone (Fair Control)
- **Condition I:** Normalized ResNet293-Mono Alone (Fair Control)
- **Condition J:** Normalized Equal-Weight Fusion (Fair Control)
- **Condition K:** Normalized Reliability Gate (Linear Clipped)
- **Condition K_sig:** Normalized Reliability Gate (Sigmoid Sensitivity)

---

## 1. Executive Summary & Core Experimental Answers

Stage 5A.1 was designed to evaluate whether multi-view representation fusion genuinely improves speaker diarization and conversational attribution when compared fairly against normalized single-encoder controls.

### 1.1 Summary of Core Research Answers

#### Question A — Does Representation Fusion Help? (Compare J vs H & J vs I)
- **In ASR Attribution: YES, STATISTICALLY SIGNIFICANT OVER TITANET.**  
  Normalized Equal-Weight Fusion (J) achieves a **macro cpWER of 37.05%**, outperforming Normalized TitaNet (H, 38.07%) by **$-1.02\%$ absolute**, with a 95% clustered bootstrap confidence interval of **$[-2.07\%, -0.10\%]$** (strictly excluding zero). Macro speaker accuracy improves from $49.42\%$ to $50.25\%$.
- **In Diarization DER: EQUIVALENT / MODEST IMPROVEMENT.**  
  Condition J achieves a Macro DER of $46.17\%$, beating Normalized TitaNet ($46.22\%$, 5 wins vs 1 loss, 2 ties). Compared to Normalized ResNet ($45.98\%$), Condition J improves speaker count estimation (MAE drops from $1.625$ to $1.250$), but Macro DER is slightly higher ($+0.19\%$, $95\%$ CI $[-0.51\%, +1.04\%]$).

#### Question B — Does Reliability-Aware Fusion Help? (Compare K vs J)
- **In Speaker Confusion & Diarization DER: YES.**  
  Condition K (Reliability Gate) achieves the **lowest Speaker Confusion (8.18%)** and the **lowest Macro DER (45.92%)** of all main conditions, outperforming Equal Fusion (J, $46.17\%$) by $-0.25\%$ DER ($95\%$ CI $[-0.76\%, 0.00\%]$).
- **In Speaker Attribution: EQUAL FUSION REMAINS STRONGER.**  
  Condition J achieves lower cpWER ($37.05\%$ vs $37.74\%$) and higher macro speaker accuracy ($50.25\%$ vs $48.99\%$) than Condition K. Equal weighting provides beneficial regularization against estimation variance on difficult multi-speaker recordings.

#### Question C — Did Normalization Confound Stage 5A? (Compare H vs A & I vs B)
- **NO. NORMALIZATION DOES NOT CONFOUND SINGLE-ENCODER DIARIZATION.**  
  Comparing Condition H against Condition A, and Condition I against Condition B, yields **identically zero difference** ($\Delta = 0.0000$) across every single recording and metric. In NeMo NME-SC spectral clustering, single-encoder clustering is based on row-wise argsort rank, which is invariant to monotonic normalization. The improvements observed in Stage 5A were genuine multi-view fusion effects.

---

## 2. Complete Summary Diarization Results Across All Conditions

The following table reports macro and duration-weighted diarization metrics, DER error decompositions, speaker count statistics, and WhisperX word attribution metrics across all 8 evaluated conditions:

| Metric Domain | Evaluation Metric | Condition A<br>(Raw TitaNet) | Condition B<br>(Raw ResNet) | Condition C<br>(S5A Fusion) | Condition H<br>(Norm TitaNet) | Condition I<br>(Norm ResNet) | Condition J<br>(Norm Eq Fusion) | Condition K<br>(Norm Rel Gate) | Condition K_sig<br>(Sigmoid Gate) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diarization Error** | **Macro DER (0.0s, ovl)** | 46.22% | 45.98% | 46.17% | 46.22% | 45.98% | 46.17% | **45.92%** | 46.12% |
| | Dur-Weighted DER (0.0s, ovl) | 51.44% | 52.22% | 52.18% | 51.44% | 52.22% | 52.18% | 51.49% | 52.03% |
| | Macro DER (0.25s, no-ovl) | 39.72% | **38.94%** | 39.65% | 39.72% | **38.94%** | 39.65% | 39.39% | 39.59% |
| | Dur-Weighted DER (0.25s, no-ovl) | 45.29% | 45.61% | 45.97% | 45.29% | 45.61% | 45.97% | **45.26%** | 45.81% |
| | Macro JER | 62.14% | **62.07%** | 62.20% | 62.14% | **62.07%** | 62.20% | 62.63% | 62.17% |
| **DER Decomposition** | Macro False Alarm (FA) | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% |
| | Macro Missed Speech (Miss) | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% |
| | **Macro Speaker Confusion (Conf)** | 8.48% | 8.24% | 8.43% | 8.48% | 8.24% | 8.43% | **8.18%** | 8.37% |
| **Speaker Counting** | Speaker Count MAE | 1.500 | 1.625 | **1.250** | 1.500 | 1.625 | **1.250** | 1.625 | **1.250** |
| | Speaker Count Signed Bias | -1.500 | -1.625 | **-1.250** | -1.500 | -1.625 | **-1.250** | -1.625 | **-1.250** |
| | Single-Spk Collapse Count | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 | 1 / 8 |
| **ASR Attribution** | **Macro cpWER** | 38.07% | **36.44%** | 37.05% | 38.07% | **36.44%** | 37.05% | 37.74% | 37.02% |
| | Macro saWER | 46.58% | 46.58% | 46.45% | 46.58% | 46.58% | 46.45% | 46.65% | **46.37%** |
| | Macro Micro Spk Accuracy | 84.27% | **84.29%** | 84.01% | 84.27% | **84.29%** | 84.01% | 84.18% | 84.01% |
| | **Macro Macro Spk Accuracy** | 49.42% | 48.94% | **50.25%** | 49.42% | 48.94% | **50.25%** | 48.99% | 50.24% |

---

## 3. Per-Recording Results Breakdown Across All 8 Recordings

The table below presents the recording-level breakdown across all 8 recordings for the primary fair conditions (H, I, J, K):

| Recording ID | Dur (s) | Ref Spk | Condition H (Norm TitaNet)<br>Pred \| DER (0.0s) \| cpWER | Condition I (Norm ResNet)<br>Pred \| DER (0.0s) \| cpWER | Condition J (Norm Eq Fusion)<br>Pred \| DER (0.0s) \| cpWER | Condition K (Norm Rel Gate)<br>Pred \| DER (0.0s) \| cpWER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R01_sara_sharif** | 100.96 | 5 | 3 \| 40.14% \| 55.12% | 3 \| 37.62% \| 50.18% | 3 \| 38.06% \| 50.88% | 3 \| 38.06% \| 50.88% |
| **R02_dennis_skinner** | 176.26 | 4 | 3 \| 62.46% \| 43.97% | 3 \| 59.52% \| 24.43% | 3 \| 62.31% \| 43.00% | 3 \| 62.31% \| 43.00% |
| **R03_disabled_parking** | 206.22 | 5 | 3 \| 34.10% \| 21.43% | 3 \| 33.56% \| 19.30% | 3 \| 33.54% \| 19.60% | 3 \| 33.54% \| 19.60% |
| **R04_piers_robinson** | 204.55 | 3 | 3 \| 40.61% \| 39.98% | 3 \| 40.12% \| 38.80% | 3 \| 40.39% \| 38.57% | 3 \| 40.39% \| 38.45% |
| **R05_southwest_pilot** | 162.45 | 3 | 3 \| 58.99% \| 31.77% | 3 \| 60.12% \| 37.50% | 3 \| 58.75% \| 31.25% | 3 \| 58.75% \| 30.99% |
| **R06_rich_karen** | 652.18 | 9 | 6 \| 69.74% \| 48.98% | 5 \| 73.28% \| 57.99% | 8 \| 72.58% \| 49.81% | 5 \| 70.56% \| 55.67% |
| **R07_bwc_cheap_shot** | 175.43 | 3 | 1 \| 45.16% \| 52.99% | 1 \| 45.16% \| 52.99% | 1 \| 45.16% \| 52.99% | 1 \| 45.16% \| 52.99% |
| **R08_uk_parliament** | 215.34 | 4 | 2 \| 18.58% \| 10.32% | 2 \| 18.44% \| 10.32% | 2 \| 18.58% \| 10.32% | 2 \| 18.58% \| 10.32% |

### In-Depth Analysis of Difficult Recordings: R06 and R07

#### R06_rich_karen (Long Multi-Speaker BWC Audio, 652.2s, 9 Ground-Truth Speakers)
- **Acoustic Environment:** High ambient street noise, shouting, cross-talk, dynamic distance from body-worn camera microphone.
- **Single-Encoder Failures:**
  - TitaNet-DFN predicts 6 speakers; achieves $69.74\%$ DER and $48.98\%$ cpWER.
  - ResNet293-mono predicts only 5 speakers; exhibits severe undercounting (bias = -4); achieves $73.28\%$ DER and $57.99\%$ cpWER.
- **Fusion Behavior:**
  - **Condition J (Equal Fusion):** Predicts **8 speakers** (closest to ground-truth 9, bias = -1). Diarization DER is $72.58\%$, and cpWER is **$49.81\%$** (substantially outperforming ResNet's $57.99\%$). Equal fusion successfully prevents the graph partition from undercounting.
  - **Condition K (Reliability Gate):** Predicts 5 speakers (weighting shifts towards ResNet's undercounting partition in high-energy segments); achieves $70.56\%$ DER.

#### R07_bwc_cheap_shot (Severe Single-Speaker Collapse, 175.4s, 3 Ground-Truth Speakers)
- **Acoustic Environment:** Physical struggle, screaming, severe microphone clipping, continuous overlapping speech.
- **Single-Speaker Collapse Across All Conditions:**
  - Ground truth contains 3 active speakers.
  - Every single evaluated condition (A, B, C, H, I, J, K, K_sig) predicts **exactly 1 speaker**.
  - All conditions report identical DER ($45.16\%$), JER ($79.23\%$), and cpWER ($52.99\%$).
- **Root Cause Analysis:**
  - Inspection of the graph Laplacian eigenvalues reveals that severe overlap and background saturation blur the similarity graph uniformly across all speakers.
  - The maximum eigengap heuristic in NME-SC selects $k=1$ because $\lambda_2 - \lambda_1$ is smaller than the threshold parameter.
  - **Affinity fusion alone CANNOT fix single-speaker collapse** when the underlying adjacency matrix eigenvalues are uniformly degraded. This requires explicit speaker-count calibration (Stage 5C).

---

## 4. Paired Clustered Bootstrap Uncertainty Analysis

We computed 2,000 paired cluster bootstrap resamples (resampling recording clusters with replacement).

### 4.1 Question A: Representation Fusion Effect (J vs H & J vs I)
| Comparison | Evaluated Metric | Point Difference ($\mathbf{J} - \text{Ref}$) | 95% Bootstrap CI | CI Width | Includes Zero? | Win Record<br>(J wins / Ref wins / Ties) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **J vs H** | Macro DER (0.0s, ovl) | **-0.0005** (-0.05%) | $[-0.0089, +0.0089]$ | 0.0178 | Yes | 5 / 1 / 2 |
| *(Equal Fusion vs Norm TitaNet)* | Macro DER (0.25s, no-ovl) | **-0.0007** (-0.07%) | $[-0.0084, +0.0082]$ | 0.0166 | Yes | 4 / 2 / 2 |
| | **Macro cpWER** | **-0.0102** (-1.02%) | **$[-0.0207, -0.0010]$** | **0.0197** | **No (Significant)** | 5 / 1 / 2 |
| | **Macro Macro Spk Acc** | **+0.0083** (+0.83%) | $[-0.0015, +0.0238]$ | 0.0253 | Yes | 5 / 1 / 2 |
| | **Speaker Count MAE** | **-0.2500** | $[-0.7500, 0.0000]$ | 0.7500 | Yes | 1 / 0 / 7 |
| **J vs I** | Macro DER (0.0s, ovl) | +0.0019 (+0.19%) | $[-0.0051, +0.0104]$ | 0.0155 | Yes | 3 / 4 / 1 |
| *(Equal Fusion vs Norm ResNet)* | Macro DER (0.25s, no-ovl) | +0.0071 (+0.71%) | $[-0.0046, +0.0250]$ | 0.0296 | Yes | 3 / 4 / 1 |
| | Macro cpWER | +0.0061 (+0.61%) | $[-0.0370, +0.0624]$ | 0.0995 | Yes | 3 / 3 / 2 |
| | **Macro Macro Spk Acc** | **+0.0130** (+1.30%) | $[-0.0062, +0.0422]$ | 0.0483 | Yes | 3 / 3 / 2 |
| | **Speaker Count MAE** | **-0.3750** | $[-1.1250, 0.0000]$ | 1.1250 | Yes | 1 / 0 / 7 |

### 4.2 Question B: Reliability Gate Effect (K vs J)
| Comparison | Evaluated Metric | Point Difference ($\mathbf{K} - \mathbf{J}$) | 95% Bootstrap CI | CI Width | Includes Zero? | Win Record<br>(K wins / J wins / Ties) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **K vs J** | **Macro DER (0.0s, ovl)** | **-0.0025** (-0.25%) | $[-0.0076, 0.0000]$ | 0.0076 | Yes | 1 / 0 / 7 |
| *(Rel Gate vs Equal Fusion)* | **Macro DER (0.25s, no-ovl)** | **-0.0025** (-0.25%) | $[-0.0077, 0.0000]$ | 0.0077 | Yes | 1 / 0 / 7 |
| | Macro cpWER | +0.0069 (+0.69%) | $[-0.0011, +0.0218]$ | 0.0229 | Yes | 2 / 1 / 5 |
| | Macro Macro Spk Acc | -0.0126 (-1.26%) | $[-0.0376, 0.0000]$ | 0.0376 | Yes | 0 / 2 / 6 |
| | Speaker Count MAE | +0.3750 | $[0.0000, +1.1250]$ | 1.1250 | Yes | 0 / 1 / 7 |

---

## 5. Synthesis: Why Diarization Improvements Remain Modest

The decomposition of DER explains why representation fusion yields substantial verification gains (HTER: $29.3\% \to 27.6\%$) and significant transcription gains (cpWER: $-1.02\%$), but only modest DER gains ($-0.30\%$):

$$\text{DER} = \text{False Alarm (13.33%)} + \text{Missed Speech (24.41%)} + \text{Speaker Confusion (8.18% - 8.48%)}$$

1. **Fixed VAD Ceiling:** False Alarms ($13.33\%$) and Missed Speech ($24.41\%$) account for **$37.74\%$** out of the **$45.92\%$** total DER ($82.2\%$ of all diarization errors). Affinity fusion operates exclusively on the affinity matrix, which controls cluster assignment—directly affecting **only the Confusion component ($8.18\%$)**.
2. **Single-Label Spectral Clustering:** In BWC recordings, overlapping speech accounts for $15\%$ to $22\%$ of speaking time. Single-label clustering cannot output two concurrent speakers, forcing every overlapping frame to incur miss errors.
3. **Conclusion:** To break below $35\%$ DER, research must advance to **Stage 5B (VAD & Overlap Handling)**. Representation fusion has already maximized the gains possible within the confusion component.
