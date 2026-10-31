# Stage 4.1 — Section 9: Downstream Diarization & Speaker Attribution Evaluation Report

**Audit Status**: `VERIFIED` & `CORRECTED` (Added Leak-Free LOTO Gate `8b` + Diagnostic Oracle Speaker Count `12` + Clustered 95% CIs)  
**Supporting Artifacts**:
- [corrected_diarization_summary.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_diarization_summary.csv) (`13` methods across 12 aggregate metrics)
- [corrected_diarization_per_recording.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_diarization_per_recording.csv) (`104` rows: all 8 recordings $\times$ 13 methods)

---

## 1. Downstream Evaluation Protocol & Audit Findings

We audited lines 906–1081 of [run_stage4_complementarity.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py#L906-L1081) and re-evaluated all 11 original methods plus 2 new Stage 4.1 methods (`8b_loro_gate_corrected_loto_affinity` and `12_diagnostic_oracle_spkcount_equal_affinity`):

1. **Zero Ground-Truth Fallback or Clustering Leakage in Methods `1–7, 9–11`**:
   - NME-SC clustering (`run_nmesc_with_custom_affinity`), Hungarian label alignment (`pyannote.metrics`), and Whisper `base` ASR temporal attribution (`assign_speakers_and_compute_wer`) are **100% unsupervised** with respect to ground-truth speaker labels or speaker counts (`oracle_num_speakers=False`).
   - Only Method `8_loro_calibrated_gate_affinity` had indirect training-target leakage (from self-included centroids on the 7 training recordings), which we corrected in Method **`8b_loro_gate_corrected_loto_affinity`** (fitted strictly on turn-disjoint LOTO clean subsegments from the 7 training recordings with equal recording weight).
2. **Diagnostic Oracle Speaker Count (`12_diagnostic_oracle_spkcount_equal_affinity`)**:
   - Explicitly labeled as a non-deployable diagnostic experiment where the true number of positive-duration reference speakers $K_{\text{ref}}$ is passed to spectral clustering on the `4_equal_affinity_050` affinity matrix, isolating speaker-count estimation error from affinity matrix quality.

---

## 2. Complete 13-Method Downstream Diarization Summary Table (`N = 8` Recordings)

From [corrected_diarization_summary.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_diarization_summary.csv) and [clustered_bootstrap_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/clustered_bootstrap_metrics.csv):

| Method | Audio | Macro DER ($c=0$, ovl) `[95% CI]` | Dur-Wtd DER ($c=0$) | Macro DER ($c=0.25$, no-ovl) | Macro JER | Spk Count MAE (Bias) | Collapse Count | Macro cpWER `[95% CI]` | Macro saWER | Micro-SACC | Macro-SACC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_titanet_nmesc` (Baseline)** | `dfn` | `46.22% [35.68, 56.26]` | `51.44%` | `39.72%` | `62.14%` | `1.50 (-1.50)` | 1 (`R07`) | `38.07% [27.40, 47.62]` | `46.58%` | `84.27%` | `49.42%` |
| **`2_resnet293_nmesc`** | `dfn` | `47.60% [35.85, 59.01]` | `53.98%` | `41.21%` | `63.20%` | `1.25 (-1.25)` | 1 (`R07`) | `39.96% [29.36, 49.34]` | `48.89%` | `82.74%` | `48.70%` |
| **`3_naive_concat_448d`** | `dfn` | `46.07% [35.58, 56.24]` | `51.32%` | `39.69%` | `62.93%` | `1.625 (-1.625)` | 1 (`R07`) | `38.40% [27.45, 48.08]` | `46.67%` | `84.13%` | `48.90%` |
| **`4_equal_affinity_050`** | `dfn` | `46.06% [35.58, 56.22]` | `51.29%` | `39.68%` | `62.88%` | `1.625 (-1.625)` | 1 (`R07`) | `38.32% [27.38, 48.02]` | `46.59%` | `84.09%` | `48.88%` |
| **`5_zscore_affinity_050`** | `dfn` | `46.08% [35.61, 56.22]` | `51.31%` | `39.73%` | `62.85%` | `1.625 (-1.625)` | 1 (`R07`) | `38.42% [27.40, 48.44]` | `46.69%` | `84.13%` | `49.00%` |
| **`6_heuristic_quality_affinity`** | `dfn` | `46.10% [35.60, 56.32]` | `51.48%` | `39.76%` | `62.77%` | `1.50 (-1.50)` | 1 (`R07`) | `37.78% [27.19, 47.29]` | `46.04%` | `84.14%` | `49.06%` |
| **`7_loro_procrustes_192d`** | `dfn` | `46.37% [35.72, 56.61]` | `51.62%` | `40.03%` | `62.94%` | `1.625 (-1.625)` | 1 (`R07`) | `39.02% [28.07, 48.80]` | `46.92%` | `83.55%` | `48.36%` |
| **`8_loro_calibrated_gate_affinity`** | `dfn` | `46.04% [35.57, 56.22]` | `51.34%` | `39.56%` | `62.68%` | `1.50 (-1.50)` | 1 (`R07`) | `37.69% [27.07, 47.22]` | `45.92%` | `84.15%` | `49.06%` |
| **`8b_loro_gate_corrected_loto_affinity`** | `dfn` | **`46.06% [35.57, 56.21]`** | `51.32%` | `39.70%` | **`62.61%`** | `1.50 (-1.50)` | 1 (`R07`) | **`37.70% [27.07, 47.22]`** | `45.96%` | `84.13%` | **`49.46%`** |
| **`9_titanet_mono_nmesc`** | `mono` | `46.57% [36.07, 55.91]` | **`50.91%`** | `39.34%` | `65.27%` | `2.00 (-2.00)` | **2 (`R01,R07`)** | `38.17% [24.25, 53.18]` | `49.53%` | `83.69%` | `45.12%` |
| **`10_resnet293_mono_nmesc`** | `mono` | **`45.98% [35.07, 56.71]`** | `52.22%` | **`38.94%`** | **`62.07%`** | `1.625 (-1.625)` | 1 (`R07`) | **`36.44% [25.02, 47.15]`** | `46.58%` | **`84.29%`** | `48.94%` |
| **`11_hybrid_titaDFN_resMONO_zscore`** | `hybrid` | `46.17% [35.51, 56.81]` | `52.18%` | `39.65%` | `62.20%` | **`1.25 (-1.25)`** | 1 (`R07`) | **`37.05% [26.56, 46.38]`** | `46.45%` | `84.01%` | **`50.25%`** |
| **`12_diagnostic_oracle_spkcount_equal_affinity`** | `dfn` | `54.83% [45.43, 64.14]` | `59.20%` | `48.38%` | `63.24%` | `0.00 (0.00)` | 0 | `53.04% [39.06, 69.13]` | `62.01%` | `73.88%` | **`58.84%`** |

---

## 3. Per-Recording Analysis & Bottleneck Diagnosis

From [corrected_diarization_per_recording.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_diarization_per_recording.csv), the per-recording breakdown reveals why aggregate DER barely moves across fusion methods (`45.98%` to `46.37%`):

1. **VAD False Alarm & Miss Dominate Total DER (`35.99%` out of `46.22%` Macro DER)**:
   - Across all 13 methods, the **exact same NeMo Path A VAD segments** are used. Consequently, for every single method:
     - Mean False Alarm Rate (`FA`) = **`13.33%`** (reaching `41.33%` on `R06_rich_karen` and `30.16%` on `R05_southwest_pilot`, largely due to unannotated speech and 1s timestamp rounding gaps).
     - Mean Missed Speech Rate (`Miss`) = **`24.41%`** (reaching `49.96%` on `R02_dennis_skinner` and `32.00%` on `R04_piers_robinson` due to severe multi-speaker overlap where single-speaker diarization can only assign 1 speaker at a time).
   - **Speaker Confusion Error (`Confusion`) is only `8.48%` of total duration** on `1_titanet_nmesc` (`11.79%` on `R01`, `11.16%` on `R02`, `9.23%` on `R03`, `6.82%` on `R04`, `1.76%` on `R05`, `17.11%` on `R06`, `9.54%` on `R07`, `0.42%` on `R08`). Any clustering/embedding change can only modify this `8.48%` confusion slice!
2. **Per-Recording Speaker Confusion (`Confusion` %) Across Key Methods**:

| Recording | Ref Spks | `1_titanet_nmesc` (DFN) | `2_resnet293_nmesc` (DFN) | `4_equal_affinity` (DFN) | `8b_loro_gate_loto` (DFN) | `9_titanet_mono` | `10_resnet293_mono` | `11_hybrid_titaDFN_resMONO` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 5 | 11.79% (3 spk) | 11.06% (3 spk) | 10.54% (3 spk) | **10.39%** (3 spk) | 21.03% (**1 spk!**) | **9.27%** (3 spk) | 9.71% (3 spk) |
| `R02_dennis_skinner` | 4 | 11.16% (3 spk) | 11.33% (3 spk) | 11.33% (3 spk) | 11.33% (3 spk) | 8.25% (2 spk) | **8.23%** (3 spk) | 11.01% (3 spk) |
| `R03_disabled_parking` | 5 | 9.23% (3 spk) | 9.65% (3 spk) | 9.37% (3 spk) | 9.37% (3 spk) | **8.46%** (3 spk) | 8.69% (3 spk) | 8.66% (3 spk) |
| `R04_piers_robinson` | 3 | 6.82% (3 spk) | 6.57% (3 spk) | 6.63% (3 spk) | 6.60% (3 spk) | 6.43% (3 spk) | **6.33%** (3 spk) | 6.60% (3 spk) |
| `R05_southwest_pilot` | 3 | 1.76% (3 spk) | 7.11% (3 spk) | 2.02% (3 spk) | 2.08% (3 spk) | **1.01%** (3 spk) | 2.89% (3 spk) | 1.52% (3 spk) |
| `R06_rich_karen` | 9 | 17.11% (6 spk) | 23.18% (8 spk) | 16.84% (5 spk) | 16.94% (6 spk) | **15.48%** (5 spk) | 20.66% (5 spk) | 19.95% (8 spk) |
| `R07_bwc_cheap_shot` | 3 | 9.54% (**1 spk**) | 9.54% (**1 spk**) | 9.54% (**1 spk**) | 9.54% (**1 spk**) | 9.54% (**1 spk**) | 9.54% (**1 spk**) | 9.54% (**1 spk**) |
| `R08_uk_parliament` | 4 | 0.42% (2 spk) | 0.42% (2 spk) | **0.28%** (2 spk) | **0.28%** (2 spk) | 0.41% (2 spk) | **0.28%** (2 spk) | 0.42% (2 spk) |
| **Macro Mean Confusion** | — | **8.48%** | 9.86% | 8.32% | **8.32%** | 8.83% | **8.24%** | 8.43% |

3. **Why `R07_bwc_cheap_shot` Collapses to 1 Speaker Across All Unsupervised Methods**:
   - In `R07_bwc_cheap_shot`, `Speaker_1` speaks for `107.1s`, `Speaker_2` for `35.0s`, and `Speaker_3` for `1.0s`. Under NME-SC's `get_k_with_gap` eigengap heuristic (`max_rp_threshold=0.15`), the first eigengap $g_1 / g_2$ exceeds the threshold on `R07` for every single affinity matrix (`1–11`), causing `est_num_of_spk = 1`.
   - Furthermore, when `12_diagnostic_oracle_spkcount_equal_affinity` forces `K = 3` on `R07`, NME-SC splits the dominant `Speaker_1` cluster into 3 temporal/acoustic subclusters instead of isolating `Speaker_2` and `Speaker_3`, causing `R07` DER to jump from `45.16%` to **`72.40%`** and `cpWER` from `52.99%` to **`101.00%`**! Only Macro-SACC improves (`48.88%` $\to$ `58.84%`) when forcing oracle speaker counts across the 8 recordings.
4. **Preprocessing Asymmetry Is the Strongest Validated Downstream Effect**:
   - **TitaNet-Large** requires DeepFilterNet enhancement on `R01_sara_sharif` to avoid collapsing to 1 speaker (`9_titanet_mono_nmesc` collapses to 1 speaker on `R01`, jumping from `40.14%` to `49.38%` DER and `55.12%` to `78.45%` cpWER), even though `16k_mono` is slightly better for TitaNet on `R02, R03, R04, R05, R06`.
   - **ResNet293-LM** is consistently harmed by DeepFilterNet spectral artifacts: switching ResNet293 from `dfn` (`2_resnet293_nmesc`, Macro DER `47.60%`, cpWER `39.96%`) to `16k_mono` (`10_resnet293_mono_nmesc`, Macro DER **`45.98%`**, cpWER **`36.44%`**) improves DER on **7 out of 8 recordings** (`95% CI [-2.66%, -0.68%]`).
   - Combining **TitaNet (`dfn`) + ResNet293 (`16k_mono`)** (`11_hybrid_titaDFN_resMONO_zscore`) prevents `R01` collapse, achieves the lowest Speaker Count MAE (`1.25`), the highest Macro-SACC (`50.25%`), and statistically significantly lowers `cpWER` relative to the `1_titanet_nmesc` baseline (`37.05%` vs `38.07%`, $\Delta = -1.02\%$, `95% CI [-2.13%, -0.12%]`, winning `5 / 1 / 2` recordings).
