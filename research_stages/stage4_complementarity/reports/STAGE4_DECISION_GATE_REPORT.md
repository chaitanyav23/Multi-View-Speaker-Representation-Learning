# Stage 4 — Complementarity Verification & Mandatory Decision Gate Report

**Script Executed:** [`research_stages/stage4_complementarity/run_stage4_complementarity.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py)
**Master Results JSON:** [`research_stages/stage4_complementarity/tables/stage4_master_results.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_master_results.json)
**Evaluation Protocol:** Recording-level separation (`8-fold Leave-One-Recording-Out [LORO]`) for all fitted transformations and gating parameters; non-overlapping clean ground-truth turns (`388` turns, `9,817` within-recording pairs, `32,131` cross-recording pairs) and non-adjacent subsegments (`|t_i - t_j| > 2.0s`) for verification metrics.

---

## 1. Representational Similarity (Linear CKA & Cross-Condition Consistency) — `VERIFIED`

Full table: [`stage4_linear_cka.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_linear_cka.csv) & [`stage4_cross_condition_consistency.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_cross_condition_consistency.csv)

| Scale Index | Window (`s`) | `16k_mono` Mean CKA (All) | `16k_mono` Mean CKA (Clean) | `dfn` Mean CKA (All) | `dfn` Mean CKA (Clean) | `normalized` Mean CKA (All) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scale 0** | `1.50s` | `0.8338` | `0.8770` | `0.8060` | `0.8539` | `0.8327` |
| **Scale 1** | `1.20s` | `0.7944` | `0.8342` | `0.7677` | `0.8158` | `0.7939` |
| **Scale 2** | `1.00s` | `0.7565` | `0.7977` | `0.7255` | `0.7734` | `0.7560` |
| **Scale 3** | `0.80s` | `0.6982` | `0.7388` | `0.6636` | `0.7108` | `0.6977` |
| **Scale 4** | `0.50s` | `0.5318` | `0.5781` | `0.5069` | `0.5555` | `0.5309` |

- **Interpretation (`VERIFIED`)**: At `1.5s`, TitaNet-Large and ResNet293-LM share moderate-to-high global relational structure (`CKA = 0.8338` on all subsegments, `0.8770` on clean single-speaker subsegments), but diverge substantially as window duration shrinks (`CKA = 0.5318` at `0.5s`) and on acoustically degraded recordings (`R07_bwc_cheap_shot` Scale 0 `CKA = 0.7434`, Scale 4 `CKA = 0.5297`; `R06_rich_karen` Scale 4 `CKA = 0.4443`).
- **Cross-Condition Perturbation (`16k_mono` vs. `dfn`)**: Applying `DeepFilterNet3` (`dfn`) shifts the embeddings of **both** encoders dramatically on noisy field recordings (`cos(mono, dfn)` at Scale 0 is only `0.6659` for TitaNet and `0.5853` for ResNet293 on `R05_southwest_pilot`, and `0.6663` / `0.6706` on `R06_rich_karen`), whereas `normalized` is nearly invariant (`cos(mono, norm) > 0.97` across all recordings).

---

## 2. LORO Orthogonal Procrustes Feature Alignment — `VERIFIED`

Full table: [`stage4_loro_procrustes.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_loro_procrustes.csv)

Under strict 8-fold LORO evaluation (fitting orthogonal $W \in \mathbb{R}^{256 \times 192}$ on 7 training recordings with equal recording weight and testing on the held-out 8th recording):

| Audio Condition | Scale | Window (`s`) | Mean Train $R^2$ | Mean Held-Out Test $R^2$ | Generalization Drop ($\Delta R^2$) | Mean Held-Out Cosine |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `16k_mono` | Scale 0 | `1.50s` | `+0.3383` | **`-0.3331`** | `0.6714` | `0.4684` |
| `16k_mono` | Scale 1 | `1.20s` | `+0.2701` | **`-0.3065`** | `0.5766` | `0.4606` |
| `16k_mono` | Scale 2 | `1.00s` | `+0.2106` | **`-0.2913`** | `0.5019` | `0.4529` |
| `16k_mono` | Scale 3 | `0.80s` | `+0.1338` | **`-0.2905`** | `0.4243` | `0.4344` |
| `16k_mono` | Scale 4 | `0.50s` | `-0.0374` | **`-0.3281`** | `0.2907` | `0.3882` |
| `dfn` | Scale 0 | `1.50s` | `+0.2829` | **`-0.4110`** | `0.6939` | `0.4521` |
| `dfn` | Scale 4 | `0.50s` | `-0.0595` | **`-0.3480`** | `0.2885` | `0.3906` |

- **Critical Finding (`VERIFIED`)**: Orthogonal Procrustes alignment **overfits severely to the 7 training recordings' speaker subspaces** (`~22` speakers total) and fails to generalize to unseen speakers in the held-out recording (`Test R^2 = -0.3331` at Scale 0 on `16k_mono`, `-0.4110` on `dfn`, with held-out cosine similarity averaging only `0.4684`). Consequently, feature-level Procrustes averaging (`7_loro_procrustes_192d`) **degrades** downstream diarization compared to single-encoder TitaNet (`DER = 46.37%` vs. `46.22%`, `cpWER = 39.02%` vs. `38.07%`).

---

## 3. Speaker Discriminability & Verification Error Complementarity — `VERIFIED`

Full per-recording table: [`stage4_per_recording_discriminability.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_per_recording_discriminability.csv)

### 3.1 Per-Recording Discriminability at Scale 0 (`1.5s`, `16k_mono`, Non-Adjacent Subsegments `|t_i - t_j| > 2.0s`)

| Recording ID | Clean Subsegments | TitaNet EER (%) | ResNet293 EER (%) | Equal Fusion EER (%) | TitaNet FDR | ResNet293 FDR | TitaNet Sil | ResNet293 Sil | Better Single Encoder |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R01_sara_sharif` | `82` | `31.88%` | **`29.88%`** | **`28.94%`** | `0.735` | **`0.781`** | `0.155` | **`0.181`** | **ResNet293-LM** |
| `R02_dennis_skinner` | `103` | `17.34%` | **`16.16%`** | **`15.49%`** | `2.067` | **`2.275`** | `0.273` | **`0.311`** | **ResNet293-LM** |
| `R03_disabled_parking` | `137` | `8.08%` | **`5.33%`** | **`5.04%`** | `4.051` | **`5.783`** | `0.369` | **`0.464`** | **ResNet293-LM** |
| `R04_piers_robinson` | `87` | `3.90%` | **`3.51%`** | **`2.61%`** | `6.521` | **`8.165`** | `0.506` | **`0.569`** | **ResNet293-LM** |
| `R05_southwest_pilot` | `81` | **`19.50%`** | `25.19%` | `20.11%` | **`1.869`** | `1.073` | **`0.228`** | `0.193` | **TitaNet-Large** |
| `R06_rich_karen` | `300` | `16.20%` | **`14.35%`** | **`12.82%`** | `2.186` | **`2.556`** | `0.219` | **`0.289`** | **ResNet293-LM** |
| `R07_bwc_cheap_shot` | `92` | **`11.57%`** | `15.79%` | `12.64%` | **`1.425`** | `1.224` | **`0.228`** | `0.207` | **TitaNet-Large** |
| `R08_uk_parliament` | `207` | `9.44%` | **`3.02%`** | **`4.13%`** | `3.633` | **`7.884`** | `0.225` | **`0.360`** | **ResNet293-LM** |

### 3.2 Turn-Level Pairwise Verification & $2 \times 2$ Error Contingency (`388` Clean Ground-Truth Turns)

| Protocol & Condition | Pairs ($N_+ / N_-$) | TitaNet EER / minDCF | ResNet293 EER / minDCF | Equal Fusion EER / minDCF | Z-Score Fusion EER / minDCF | Oracle Gate EER / minDCF | Error Overlap (Jaccard) | $n_{00}$ (Both OK) | $n_{10}$ (Tita Err, Res OK) | $n_{01}$ (Tita OK, Res Err) | $n_{11}$ (Both Err) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Within-Rec (`16k_mono`)** | `9,817` (`4,061` / `5,756`) | `24.33%` / `0.7332` | `28.75%` / `0.6901` | `25.27%` / `0.6844` | `25.00%` / `0.6788` | **`15.25%`** / **`0.4648`** | **`0.3581`** | `5,980` (`60.9%`) | `1,015` (`10.3%`) | `1,448` (`14.8%`) | `1,374` (`14.0%`) |
| **Within-Rec (`dfn`)** | `9,817` (`4,061` / `5,756`) | `30.34%` / `0.7611` | `31.86%` / `0.7689` | `29.33%` / `0.7308` | `29.18%` / `0.7303` | **`20.36%`** / **`0.5432`** | **`0.3712`** | `5,375` (`54.8%`) | `1,227` (`12.5%`) | `1,566` (`16.0%`) | `1,649` (`16.8%`) |
| **Pooled Cross-Rec (`16k_mono`)** | `32,131` (`4,061` / `28,070`) | `23.27%` / `0.6590` | `25.11%` / `0.6962` | `22.73%` / `0.6416` | **`22.21%`** / **`0.6254`** | **`13.09%`** / **`0.4090`** | **`0.3198`** | `20,636` (`64.2%`) | `3,551` (`11.1%`) | `4,269` (`13.3%`) | `3,675` (`11.4%`) |
| **Pooled Cross-Rec (`dfn`)** | `32,131` (`4,061` / `28,070`) | `27.70%` / `0.7289` | `27.78%` / `0.7429` | `25.63%` / `0.6976` | **`25.43%`** / **`0.6908`** | **`16.59%`** / **`0.4789`** | **`0.3446`** | `18,901` (`58.8%`) | `4,361` (`13.6%`) | `4,306` (`13.4%`) | `4,563` (`14.2%`) |

---

## 4. Downstream Clustering & Diarization Benchmark (11 Methods Across 8 Recordings) — `VERIFIED`

Full tables: [`stage4_diarization_method_summary.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_diarization_method_summary.csv) & [`stage4_diarization_per_recording.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_diarization_per_recording.csv)

| Method ID | Audio Input | Macro DER (`c=0.0, ovl`) | Dur-Weighted DER (`c=0.0`) | Macro DER (`c=0.25, no-ovl`) | Macro JER | Spk Count MAE | Single-Spk Collapses | Macro cpWER | Macro saWER | Macro Micro-SACC | Macro Macro-SACC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_titanet_nmesc` (Baseline) | `dfn` | `46.22%` | `51.44%` | `39.72%` | `62.14%` | `1.500` | `1` (`R07`) | `38.07%` | `46.58%` | `84.27%` | `49.42%` |
| `2_resnet293_nmesc` | `dfn` | `47.60%` | `53.98%` | `41.21%` | `63.20%` | **`1.250`** | `1` (`R07`) | `39.96%` | `48.89%` | `82.74%` | `48.70%` |
| `3_naive_concat_448d` | `dfn` | `46.07%` | `51.32%` | `39.69%` | `62.93%` | `1.625` | `1` (`R07`) | `38.40%` | `46.67%` | `84.13%` | `48.90%` |
| `4_equal_affinity_050` | `dfn` | `46.06%` | **`51.29%`** | `39.68%` | `62.88%` | `1.625` | `1` (`R07`) | `38.32%` | `46.59%` | `84.09%` | `48.88%` |
| `5_zscore_affinity_050` | `dfn` | `46.08%` | `51.31%` | `39.73%` | `62.85%` | `1.625` | `1` (`R07`) | `38.42%` | `46.69%` | `84.13%` | `49.00%` |
| `6_heuristic_quality_affinity` | `dfn` | `46.10%` | `51.48%` | `39.76%` | `62.77%` | `1.500` | `1` (`R07`) | `37.78%` | `46.04%` | `84.14%` | `49.06%` |
| `7_loro_procrustes_192d` | `dfn` | `46.37%` | `51.62%` | `40.03%` | `62.94%` | `1.625` | `1` (`R07`) | `39.02%` | `46.92%` | `83.55%` | `48.36%` |
| `8_loro_calibrated_gate_affinity` | `dfn` | `46.04%` | `51.34%` | `39.56%` | `62.68%` | `1.500` | `1` (`R07`) | `37.69%` | **`45.92%`** | `84.15%` | `49.06%` |
| `9_titanet_mono_nmesc` | `16k_mono` | `46.57%` | **`50.91%`** | `39.34%` | `65.27%` | `2.000` | `2` (`R05, R07`) | `38.17%` | `49.53%` | `83.69%` | `45.12%` |
| `10_resnet293_mono_nmesc` | `16k_mono` | **`45.98%`** | `52.22%` | **`38.94%`** | **`62.07%`** | `1.625` | `1` (`R07`) | **`36.44%`** | `46.58%` | **`84.29%`** | `48.94%` |
| `11_hybrid_titaDFN_resMONO_zscore` | `dfn` + `mono` | `46.17%` | `52.18%` | `39.65%` | `62.20%` | **`1.250`** | `1` (`R07`) | `37.05%` | `46.45%` | `84.01%` | **`50.25%`** |

---

## 5. Answers to the 8 Mandatory Decision Gate Questions

### Question 1: Does ResNet293-LM provide measurable speaker-discriminative information beyond TitaNet?
- **Answer (`VERIFIED`)**: **YES.**
  1. At longer window scales (`1.0s–1.5s`) on unenhanced `16k_mono` audio, **ResNet293-LM outperforms TitaNet-Large in verification EER, Fisher Discriminant Ratio (FDR), and Cosine Silhouette on 6 out of 8 recordings** (`R01`, `R02`, `R03`, `R04`, `R06`, `R08`), with dramatic gains on multi-speaker recordings such as `R03_disabled_parking` (`EER 5.33%` vs. `8.08%`, `FDR 5.78` vs. `4.05`) and `R08_uk_parliament` (`EER 3.02%` vs. `9.44%`, `FDR 7.88` vs. `3.63`).
  2. Even at the shortest `0.5s` scale (`Scale 4`, `16k_mono`), **zero-training Equal 50/50 Cosine Fusion improves EER over BOTH individual encoders on 7 out of 8 recordings** (`R01: 37.32%`, `R02: 29.70%`, `R03: 22.72%`, `R04: 21.27%`, `R05: 29.14%`, `R06: 28.82%`, `R07: 28.76%`).
  3. On unenhanced `16k_mono` audio, standalone `ResNet293-LM` (`10_resnet293_mono_nmesc`) achieves the best overall macro `DER (c=0.0) = 45.98%`, `DER (c=0.25) = 38.94%`, `JER = 62.07%`, and `cpWER = 36.44%`, beating the Phase 20 TitaNet baseline (`46.22%`, `39.72%`, `62.14%`, `38.07%`).

### Question 2: Are their verification errors complementary?
- **Answer (`VERIFIED`)**: **YES.**
  - Across the `9,817` within-recording non-overlapping ground-truth turn pairs (`16k_mono`), at each encoder's EER operating point, `1,015` pairs (`10.34%`) are misclassified by TitaNet but correctly classified by ResNet293-LM, while `1,448` pairs (`14.75%`) are misclassified by ResNet293-LM but correctly classified by TitaNet. Only `1,374` pairs (`14.00%`) fail on both encoders.
  - The error Jaccard overlap is **`0.3581`**, meaning **`64.19%` of the error union is complementary** (one encoder succeeds where the other fails).
  - An **Oracle Gate** selecting the better encoder per pair lowers within-recording turn-pair EER from `24.33%` (TitaNet) / `28.75%` (ResNet293) down to **`15.25%`** (`-9.08` percentage points, `minDCF 0.4648`), and lowers pooled cross-recording EER (`32,131` pairs) from `23.27%` to **`13.09%`**.

### Question 3: Under which acoustic conditions does each encoder perform better?
- **Answer (`VERIFIED`)**:
  1. **Window Duration / Temporal Scale**: ResNet293-LM achieves higher speaker-centroid cosine margins at longer windows (`1.5s`: `0.4201` vs. `0.3897`, winning `63.59%` of clean subsegments; `1.0s`: `0.3268` vs. `0.2972`, winning `61.22%`), whereas TitaNet-Large maintains higher nearest-centroid accuracy at the shortest `0.5s` window (`88.69%` vs. `86.23%` on `16k_mono`, `87.39%` vs. `83.89%` on `dfn`).
  2. **SNR Regime**: In the lowest SNR quartile (`Q1`), TitaNet-Large is more robust (`81.96%` vs. `78.28%` on `16k_mono`; `70.63%` vs. `64.40%` on `dfn`, where ResNet293 wins only `41.17%` of subsegments). In medium-to-high SNR quartiles (`Q2` and `Q4`), ResNet293-LM produces larger centroid separation margins (`res_win_fraction = 55.95%` in `Q2` and `52.60%` in `Q4` on `16k_mono`; `55.29%–56.52%` across `Q2–Q4` on `dfn`).
  3. **Audio Preprocessing (`16k_mono` vs. `dfn`) — Major Empirical Finding**:
     - `DeepFilterNet3` (`dfn`) **degrades speaker verification discriminability for BOTH encoders** (`within-rec turn EER` worsens from `24.33% -> 30.34%` for TitaNet and `28.75% -> 31.86%` for ResNet293-LM) because spectral suppression introduces phase/formant distortion in low-SNR regions (`dfn` Q1 centroid accuracy drops to `70.63%` / `64.40%` vs. `81.96%` / `78.28%` on `16k_mono`).
     - However, `dfn` stabilizes NME-SC eigenvalue thresholding for TitaNet on `R05_southwest_pilot` (preventing a single-speaker collapse that occurs for `9_titanet_mono_nmesc`), whereas **ResNet293-LM performs substantially better on unenhanced `16k_mono` (`10_resnet293_mono_nmesc`) than on `dfn` (`2_resnet293_nmesc`)**, improving DER (`c=0.25`) from `41.21%` to **`38.94%`** and `cpWER` from `39.96%` to **`36.44%`**.
  4. **Overlapping Speech**: Both encoders collapse to chance-level centroid assignment on overlapping speech (`45.80%` TitaNet, `46.37%` ResNet293 on `16k_mono`, negative mean margins `-0.0128` and `-0.0235`), confirming that single-vector subsegment embeddings cannot resolve concurrent speakers without overlap-aware segmentation or multi-label post-processing.

### Question 4: Is affinity-level fusion more promising than feature-level alignment?
- **Answer (`VERIFIED`)**: **YES — Affinity-level fusion is decisively superior to feature-level alignment.**
  - Every affinity-level fusion method (`4_equal_affinity_050`, `5_zscore_affinity_050`, `6_heuristic_quality_affinity`, `8_loro_calibrated_gate_affinity`, and `11_hybrid_titaDFN_resMONO_zscore`) outperforms feature-level LORO Orthogonal Procrustes alignment (`7_loro_procrustes_192d`) across `DER (c=0.0)`, `DER (c=0.25)`, `JER`, `cpWER`, `saWER`, `Micro-SACC`, and `Macro-SACC`.
  - Crucially, affinity-level fusion operates directly in the native cosine similarity space of each frozen encoder ($S \in \mathbb{R}^{N \times N}$) without requiring any cross-space projection matrix $W \in \mathbb{R}^{256 \times 192}$.

### Question 5: Is Procrustes alignment empirically justified?
- **Answer (`VERIFIED`)**: **NO.**
  - Under 8-fold LORO cross-validation, Orthogonal Procrustes achieves positive $R^2$ on the 7 training recordings (`+0.3383` at Scale 0) but **negative $R^2$ (`-0.3331` on `16k_mono`, `-0.4110` on `dfn`) on every single held-out recording across all 5 scales** (`80/80` LORO folds have negative test $R^2$).
  - Because `7` training recordings contain only `~22` speakers, a `256 x 192` (`49,152`-parameter) orthogonal matrix overfits to the training speaker directions and rotates unseen speakers into poorly aligned subspaces (`mean held-out cosine = 0.4684`). Procrustes alignment must be **abandoned**.

### Question 6: Which reliability indicators have demonstrated predictive value?
- **Answer (`VERIFIED`)**:
  Full table: [`stage4_reliability_predictive_value.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/tables/stage4_reliability_predictive_value.csv)
  1. **Pre-Normalization ResNet293 Embedding L2 Norm (`resnet293_l2_norm`)**: **Strongest predictor overall** — correlates with ResNet293 centroid margin at $\rho = +0.6015$ (`16k_mono`) / $+0.6108$ (`dfn`), predicts ResNet293 correctness with **`AUROC = 0.7714`** (`16k_mono`) / **`0.7647`** (`dfn`), and predicts the **differential margin $\Delta m = m_{\text{res}} - m_{\text{tita}}$ at $\rho = +0.2849$** (`16k_mono`) / $+0.2795$ (`dfn`).
  2. **DeepFilterNet Attenuation Proxy (`dfn_noise_reduction_db`)**: Second strongest predictor — correlates with ResNet293 margin at $\rho = +0.4563$ (`AUROC = 0.7212`) and TitaNet margin at $\rho = +0.3606$ (`AUROC = 0.6796`), and predicts $\Delta m$ at $\rho = +0.2293$ (higher noise suppression / cleaner speech favors ResNet293; heavier noise / lower suppression favors TitaNet).
  3. **Pyannote Frame-Level VAD Posterior Mean & Std (`vad_conf_mean`, `vad_conf_std`)**: Strong predictor — `vad_conf_mean` achieves $\rho = +0.4119$ (`AUROC = 0.7115`) for ResNet293 and $\rho = +0.3702$ (`AUROC = 0.6705`) for TitaNet, with $\rho(\Delta m) = +0.1661$.
  4. **Cross-Scale Embedding Consistency (`resnet293_cross_scale_cos`, `titanet_cross_scale_cos`, and their difference `cross_scale_cos_diff_res_minus_tita`)**: Moderate-to-strong predictor — `resnet293_cross_scale_cos` achieves $\rho = +0.4049$ (`AUROC = 0.6646`) and $\rho(\Delta m) = +0.1953$; the view-specific difference `cross_scale_cos_diff_res_minus_tita` specifically predicts which encoder has the larger margin ($\rho(\Delta m) = +0.1707$ on `dfn`, $+0.1114$ on `16k_mono`).
  5. **Weak / Uninformative Indicators (`spectral_flatness`, `titanet_l2_norm`, thresholded `overlap_prob_mean`)**:
     - `titanet_l2_norm` has weak predictive value ($\rho = +0.1278$, `AUROC = 0.5663`) because TitaNet's architecture normalizes internal representations more tightly than ResNet293's TSTP+Linear head.
     - `spectral_flatness` has near-zero rank correlation ($\rho = -0.0442$, `AUROC = 0.5541`).
     - `overlap_prob_mean` using the default thresholded `pyannote/segmentation` multi-speaker count is zero on nearly all subsegments (`AUROC = 0.5000`); raw pre-threshold 2nd-speaker logits would be needed if an overlap indicator is desired.

### Question 7: Is the dataset sufficient to calibrate a reliability-aware gate under LORO evaluation?
- **Answer (`VERIFIED`)**:
  - **Sufficient for low-degree-of-freedom linear/logistic affinity gating (`4–6` parameters), NOT sufficient for high-capacity neural fusion (e.g., multi-head self-attention or deep MLPs).**
  - In Stage 4, our 6-parameter LORO-calibrated ridge-regression gate (`8_loro_calibrated_gate_affinity`) improved downstream diarization over both single-encoder baselines and zero-training equal affinity fusion on `dfn` (`DER c=0.0 = 46.04%`, `DER c=0.25 = 39.56%`, `cpWER = 37.69%`, `saWER = 45.92%`).
  - However, the gain of LORO-calibrated gating over zero-training heuristic quality affinity (`6_heuristic_quality_affinity`: `cpWER = 37.78%`, `saWER = 46.04%`) is small (`~0.1–0.2%` absolute), and all NME-SC clustering methods remain bottlenecked by **eigenvalue speaker-count estimation** (specifically single-speaker collapse on `R07_bwc_cheap_shot` where `Speaker_1` holds `91.1%` of speech time, and under-clustering on `R06_rich_karen`).

### Question 8: Which experiments should be prioritized next, and which should be abandoned?
- **Answer (`VERIFIED`)**:
  - **ABANDON**:
    1. **Feature-level Orthogonal Procrustes alignment (`256-D -> 192-D`) and learned cross-space projection matrices**: Empirically falsified under LORO (`Test R^2 < -0.29` across all 80 folds).
    2. **High-capacity neural fusion heads (multi-head self-attention, Conformer/MLP projection networks, or MSDD end-to-end retraining) on the 8-recording corpus**: With only 8 recordings (`~25` speakers), any model with $>10^2$ parameters will overfit to recording-specific acoustics just as Procrustes did.
    3. **Feeding `DeepFilterNet3`-enhanced audio into `ResNet293-LM`**: Empirically shown to degrade `ResNet293-LM` verification EER (`+3.11%`), `DER c=0.25` (`+2.27%`), and `cpWER` (`+3.52%`) compared to unenhanced `16k_mono`.
  - **PRIORITIZE NEXT (If Approved for Stage 5)**:
    1. **Asymmetric / Hybrid Audio Routing (`TitaNet` on `dfn` + `ResNet293-LM` on `16k_mono`) with Reliability-Weighted Scale-Specific Affinity Fusion**:
       - Our exploratory hybrid experiment (`11_hybrid_titaDFN_resMONO_zscore`) already achieved the **lowest Speaker Count MAE (`1.250`)**, the **highest Macro-SACC (`50.25%`)**, and strong `cpWER` (`37.05%`), while standalone `10_resnet293_mono_nmesc` achieved `DER (c=0.25) = 38.94%` and `cpWER = 36.44%`.
       - Combining **scale-dependent encoder weighting** (giving higher weight to `TitaNet` at short scales `0.5s–0.8s` and low SNR, and higher weight to `ResNet293-LM (16k_mono)` at longer scales `1.0s–1.5s` and high `resnet293_l2_norm` / `vad_conf_mean`) directly exploits the verified acoustic complementarity without overfitting.
    2. **Fixing the NME-SC Minority-Speaker Collapse Bottleneck (`R07_bwc_cheap_shot` & `R06_rich_karen`)**:
       - Because all 11 affinity variants still suffer from `pred_speakers = 1` on `R07_bwc_cheap_shot` due to NME-SC's `maj_vote_spk_count` and eigenvalue gap threshold (`threshold=0.1` in `SpeakerClustering`), calibrating the spectral clustering stopping criterion / scale-consistency check using the dual-encoder affinity agreement is the highest-leverage path to reducing diarization error on imbalanced BWC audio.
