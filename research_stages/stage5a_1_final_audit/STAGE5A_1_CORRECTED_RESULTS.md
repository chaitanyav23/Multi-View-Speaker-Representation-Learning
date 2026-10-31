# Stage 5A.1 Corrected Experimental Results and Reconciled Benchmark

**Audit Date**: October 31, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Status**: Definitive Reconciled Results  
**Evaluation Target**: Pretrained Multi-View Representation Fusion (TitaNet-Large DFN + WeSpeaker ResNet293-LM Mono)

---

## 1. Overview and Reconciliation Summary

This document presents the definitive, mathematically verified, and leakage-audited experimental results for Stage 5A.1. It formally supersedes all preliminary tables and text discrepancies in prior Stage 5A and Stage 5A.1 drafts.

### Core Protocol Corrections
1. **Turn and Pair Benchmark**: Verified at **250 clean, non-overlapping turns** yielding **9,489 pairwise verification trials** across 8 recordings (reconciling the 254 turns / 9,817 pairs in Stage 5A caused by boundary overhang and unflagged overlap frames).
2. **Verification Macro Eligibility**:
   - **6-Recording Macro (R01, R03, R05, R06, R07, R08)**: Fully computable verification average including R01 (1 positive pair, 14 negative pairs; ROC-AUC = 0.8571).
   - **5-Recording Macro (R03, R05, R06, R07, R08)**: Conservative verification average strictly restricted to multi-turn balanced sessions ($N_{\text{pairs}} \ge 36$).
   - Both macros are reported below with explicit labeling.
3. **Downstream Invariance**: Proved analytically and verified numerically that single-encoder clustering in NeMo NME-SC is 100% invariant to off-diagonal z-score normalization ($H \equiv A$ and $I \equiv B$, $\Delta = 0.0000$ across all 15 diarization metrics).

---

## 2. Dataset and Verification Pair Audit

### Table 2.1: Turn and Pair Inventory by Recording
| Recording ID | Clean Turns ($N_r$) | Positive Pairs | Negative Pairs | Total Pairs ($N_r(N_r-1)/2$) | ROC-AUC Eligible | EER/HTER Eligible |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 6 | 1 | 14 | 15 | Yes (AUC=0.8571) | Yes (sparse pos) |
| `R02_dennis_skinner` | 2 | 1 | 0 | 1 | No (0 neg pairs) | No |
| `R03_disabled_parking` | 50 | 531 | 694 | 1,225 | Yes | Yes |
| `R04_piers_robinson` | 3 | 0 | 3 | 3 | No (0 pos pairs) | No |
| `R05_southwest_pilot` | 25 | 134 | 166 | 300 | Yes | Yes |
| `R06_rich_karen` | 122 | 1,819 | 5,562 | 7,381 | Yes | Yes |
| `R07_bwc_cheap_shot` | 33 | 265 | 263 | 528 | Yes | Yes |
| `R08_uk_parliament` | 9 | 16 | 20 | 36 | Yes | Yes |
| **Total Benchmark** | **250** | **2,766** | **6,723** | **9,489** | **6 Recordings** | **6 Recordings** |

---

## 3. Pairwise Error Complementarity Analysis

Verification error trials evaluated at each model's LORO-calibrated decision threshold across the complete 9,489 pairs:

### Table 3.1: Error Trial Overlap and Complementarity Matrix
| Metric / Event | Raw Encoders (Condition A vs B) | Normalized Controls (Condition H vs I) |
| :--- | :---: | :---: |
| **Total Evaluation Pairs** | 9,489 | 9,489 |
| **Both Encoders Correct** | 5,910 (62.28%) | 5,910 (62.28%) |
| **Both Encoders Wrong (Shared Errors)** | 1,502 (15.83%) | 1,502 (15.83%) |
| **TitaNet Wrong, ResNet293 Correct** | 1,035 (10.91%) | 1,035 (10.91%) |
| **ResNet293 Wrong, TitaNet Correct** | 1,042 (10.98%) | 1,042 (10.98%) |
| **Total Disjoint Error Pairs** | 2,077 (21.89%) | 2,077 (21.89%) |
| **Total Unique Error Pairs** | 3,579 (37.72%) | 3,579 (37.72%) |
| **Complementary Error Fraction** | **58.03%** | **58.03%** |
| **Error Jaccard Similarity** | **0.4197** | **0.4197** |

*Verification*: Normalized controls ($H$ vs $I$) produce the exact same trial counts and error partition as raw baselines ($A$ vs $B$) because score thresholding is monotonically invariant under affine rescaling.

---

## 4. Reconciled Speaker Verification Performance

### Table 4.1: Per-Recording ROC-AUC Performance Across All Conditions
| Recording ID | Pos/Neg Pairs | Cond A (Tita-DFN) | Cond B (Res-Mono) | Cond C / J (Eq Fuse) | Cond H (Norm Tita) | Cond I (Norm Res) | Cond K (Lin Gate) | Cond K_sig (Sig Gate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 1 / 14 | 0.8571 | 0.7857 | 0.8571 | 0.8571 | 0.7857 | 0.8571 | 0.8571 |
| `R02_dennis_skinner` | 1 / 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `R03_disabled_parking` | 531 / 694 | 0.8565 | 0.9142 | 0.8979 | 0.8565 | 0.9142 | 0.9015 | 0.8988 |
| `R04_piers_robinson` | 0 / 3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `R05_southwest_pilot` | 134 / 166 | 0.7569 | 0.7279 | 0.7661 | 0.7569 | 0.7279 | 0.7647 | 0.7661 |
| `R06_rich_karen` | 1819 / 5562 | 0.7380 | 0.7411 | 0.7595 | 0.7380 | 0.7411 | 0.7621 | 0.7602 |
| `R07_bwc_cheap_shot` | 265 / 263 | 0.7813 | 0.7689 | 0.7897 | 0.7813 | 0.7689 | 0.7896 | 0.7902 |
| `R08_uk_parliament` | 16 / 20 | 0.9531 | 1.0000 | 1.0000 | 0.9531 | 1.0000 | 1.0000 | 1.0000 |

### Table 4.2: Summary Verification Metrics (Macro & Pair-Weighted Pooled)
| Condition ID | Description | 6-Rec Macro ROC-AUC | 6-Rec Macro HTER | 5-Rec Macro ROC-AUC | 5-Rec Macro HTER | Pooled OOF ROC-AUC | Pooled OOF HTER | Pooled actDCF ($p=0.01$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A** | Raw TitaNet (DFN) | 0.8238 | 33.80% | 0.8172 | 34.13% | 0.7681 | 29.31% | 2.4369 |
| **Condition B** | Raw ResNet293 (Mono) | 0.8230 | 32.40% | 0.8304 | 33.16% | 0.7800 | 28.90% | 2.4892 |
| **Condition C** | Raw Equal Fusion (Stage 5A) | 0.8450 | 32.69% | 0.8426 | 32.79% | 0.7955 | 27.82% | 2.3356 |
| **Condition H** | Norm TitaNet Alone (Control) | 0.8238 | 33.80% | 0.8172 | 34.13% | 0.7710 | 29.31% | 2.4369 |
| **Condition I** | Norm ResNet Alone (Control) | 0.8230 | 32.40% | 0.8304 | 33.16% | 0.7836 | 28.90% | 2.4892 |
| **Condition J** | Norm Equal Fusion (Control) | 0.8450 | 32.69% | 0.8426 | 32.79% | 0.7955 | 27.82% | 2.3356 |
| **Condition K** | Norm Linear Reliability Gate | **0.8458** | **31.90%** | **0.8436** | **32.56%** | **0.7980** | **27.62%** | **2.3518** |
| **Condition K_sig** | Norm Sigmoid Gate (Ablation) | 0.8454 | 32.69% | 0.8431 | 32.80% | 0.7961 | 27.82% | 2.3558 |

---

## 5. Downstream Diarization and Attribution Performance

All downstream results evaluated using NeMo NME-SC spectral clustering ($p = 0.25$, binary thresholding) with collar = 0.0s and overlap speech included across all 8 recordings.

### Table 5.1: Macro Diarization and Transcription Metrics (8 Recordings)
| Metric | Cond A (Raw Tita) | Cond B (Raw Res) | Cond C (Eq Fuse) | Cond H (Norm Tita) | Cond I (Norm Res) | Cond J (Norm Fuse) | Cond K (Lin Gate) | Cond K_sig (Sig Gate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro DER (collar 0s, ovl)** | 46.22% | 45.98% | 46.17% | 46.22% | 45.98% | 46.17% | **45.92%** | 46.12% |
| **Dur-Weighted DER (collar 0s)** | 51.44% | 52.22% | 52.18% | 51.44% | 52.22% | 52.18% | **51.49%** | 52.03% |
| **Macro DER (collar 0.25s, noovl)**| 39.72% | 38.94% | 39.65% | 39.72% | 38.94% | 39.65% | 39.39% | 39.59% |
| **Macro JER** | 62.14% | 62.07% | 62.20% | 62.14% | 62.07% | 62.20% | 62.63% | 62.17% |
| **Macro False Alarm (FA)** | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% | 13.33% |
| **Macro Missed Speech** | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% | 24.41% |
| **Macro Speaker Confusion** | 8.48% | 8.24% | 8.43% | 8.48% | 8.24% | 8.43% | **8.18%** | 8.37% |
| **Speaker Count MAE** | 1.500 | 1.625 | **1.250** | 1.500 | 1.625 | **1.250** | 1.625 | **1.250** |
| **Speaker Count Signed Bias** | -1.500 | -1.625 | **-1.250** | -1.500 | -1.625 | **-1.250** | -1.625 | **-1.250** |
| **Single-Speaker Collapses** | 1 (R07) | 1 (R07) | 1 (R07) | 1 (R07) | 1 (R07) | 1 (R07) | 1 (R07) | 1 (R07) |
| **Macro cpWER** | 38.07% | **36.44%** | 37.05% | 38.07% | **36.44%** | 37.05% | 37.74% | 37.02% |
| **Macro SAWER** | 46.58% | 46.58% | **46.45%** | 46.58% | 46.58% | **46.45%** | 46.65% | 46.37% |
| **Macro Speaker Accuracy (SACC)**| 49.42% | 48.94% | **50.25%** | 49.42% | 48.94% | **50.25%** | 48.99% | 50.24% |

---

## 6. Paired Bootstrap Hypothesis Testing ($B = 10,000$)

Paired recording-cluster bootstrap ($B=10,000$ resamples) testing whether differences between fusion and single-encoder baselines are statistically distinguishable from zero:

### Table 6.1: Key Paired Bootstrap Contrasts
| Contrast | Description | Metric | Point Difference ($\Delta$) | 95% Bootstrap CI | Significant ($\alpha=0.05$)? | Win / Loss / Tie |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **J vs H** | Norm Equal Fusion vs Norm TitaNet | cpWER | **-0.0102 (-1.02%)** | **[-0.0207, -0.0010]** | **YES ($p < 0.05$)** | 5 / 1 / 2 |
| **J vs H** | Norm Equal Fusion vs Norm TitaNet | Macro DER | -0.0005 (-0.05%) | [-0.0089, +0.0089] | No | 5 / 1 / 2 |
| **J vs H** | Norm Equal Fusion vs Norm TitaNet | Speaker MAE | **-0.250** | [-0.750, 0.000] | Borderline | 1 / 0 / 7 |
| **J vs I** | Norm Equal Fusion vs Norm ResNet | Macro DER | +0.0019 (+0.19%) | [-0.0051, +0.0104] | No | 3 / 4 / 1 |
| **J vs I** | Norm Equal Fusion vs Norm ResNet | cpWER | +0.0061 (+0.61%) | [-0.0370, +0.0624] | No | 3 / 3 / 2 |
| **J vs I** | Norm Equal Fusion vs Norm ResNet | Speaker MAE | **-0.375** | [-1.125, 0.000] | Borderline | 1 / 0 / 7 |
| **K vs J** | Reliability Gate vs Equal Fusion | Macro DER | -0.0025 (-0.25%) | [-0.0076, 0.0000] | Borderline | 1 / 0 / 7 |
| **K vs J** | Reliability Gate vs Equal Fusion | cpWER | +0.0069 (+0.69%) | [-0.0011, +0.0218] | No | 2 / 1 / 5 |
| **K vs J** | Reliability Gate vs Equal Fusion | Speaker MAE | +0.375 | [0.000, +1.125] | Borderline | 0 / 1 / 7 |
| **K vs H** | Reliability Gate vs Norm TitaNet | Macro DER | -0.0030 (-0.30%) | [-0.0090, +0.0015] | No | 5 / 1 / 2 |
| **K vs H** | Reliability Gate vs Norm TitaNet | Confusion | **-0.0030 (-0.30%)** | [-0.0071, +0.0006] | Borderline | 5 / 1 / 2 |

---

## 7. Definitive Scientific Distinctions

To ensure permanent scientific rigor, the following distinctions are codified:
1. **Raw Affinity vs Normalized Affinity**:
   Raw affinity matrices $S$ contain pairwise cosine similarities $[-1, 1]$. Normalized affinity matrices $\tilde{S} = \text{offdiag\_zscore}(S)$ have zero off-diagonal mean and unit off-diagonal variance. While affinity normalization is necessary to combine TitaNet and ResNet on a common dynamic scale, it has zero impact on single-encoder clustering because spectral binarization in NeMo NME-SC depends strictly on rank-order statistics.
2. **Verification ROC-AUC vs Downstream DER**:
   Pairwise verification evaluates representations in isolation under oracle segmentation (clean turns). Downstream DER evaluates representations inside a real-world pipeline where errors are dominated by VAD false alarms (13.33%) and missed speech (24.41%). Consequently, a +2.2% verification ROC-AUC gain translates into a modest -0.30% speaker confusion reduction downstream.
3. **Equal Fusion vs Reliability Gate**:
   Equal fusion ($J$) is completely parameter-free and non-parametric. The reliability gate ($K$) is a lightweight linear regressor trained via LORO cross-validation. While the gate yields slightly higher verification ROC-AUC (+0.08% over equal fusion) and lower speaker confusion (8.18% vs 8.43%), the downstream difference between $K$ and $J$ is not statistically significant. Equal fusion is therefore recommended as the primary robust fusion strategy.
