# Stage 4.1 — Critical Issue E: LORO Procrustes & LORO Gate Implementation Audit

**Audit Status**: `CORRECTED`  
**Supporting Artifacts**:
- [corrected_loro_results.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_loro_results.csv) (`136` rows: 4 Procrustes variants across Scales 0 & 4, `16k_mono` & `dfn`, plus corrected LORO gate fold results)
- [stage4_1_master_audit.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/stage4_1_master_audit.json)

---

## 1. Source-Code Audit of Stage 4 LORO Procrustes (`run_stage4_complementarity.py` L572–610)

### 1.1 What Stage 4 Actually Executed
In [run_stage4_complementarity.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py#L579-L603), for each audio condition $c \in \{\text{16k\_mono}, \text{dfn}, \text{normalized}\}$, each scale $s \in \{0, 4\}$, and each held-out recording $r \in \{\text{R01}, \dots, \text{R08}\}$:
1. **Held-out exclusion**: Training matrices `X_tr` (`[N_tr, 256]`) and `Y_tr` (`[N_tr, 192]`) were assembled from `[r_o for r_o in RECORDINGS if r_o != rec_id]`. **The held-out recording was genuinely excluded from SVD fitting.**
2. **Subsegment-weighted (NOT equal-recording-weighted) SVD**:
   - Contrary to the Stage 4 report text claiming "equal recording weight", line 580 simply ran `np.vstack([caches[...][r_o][...] for r_o in RECORDINGS if r_o != rec_id])` without per-recording sample weights $w_i = 1 / N_{r_o}$.
   - Consequently, `R06_rich_karen` (`1,884 / 5,622 = 33.51%` of Scale 4 subsegments) contributed **35%–38%** of the Frobenius norm loss in every training fold where it was present, while `R01_sara_sharif` (`260` subsegments) contributed less than **5%**.
3. **Unfiltered training subsegments**: `X_tr` and `Y_tr` included all VAD subsegments (clean + `OVERLAP::` + `PARTIAL_BOUNDARY::` + `NON_SPEECH_OR_ROUNDING_GAP`).
4. **No mean centering before orthogonal SVD**: `fit_orthogonal_procrustes` computed `U, _, Vt = np.linalg.svd(X_train.T @ Y_train, full_matrices=False)` directly on L2-normalized embeddings on the unit hypersphere without centering.

### 1.2 Four-Variant LORO Procrustes Re-Evaluation
In [run_stage4_1_audit.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/run_stage4_1_audit.py#L860-L928), we evaluated 4 LORO Procrustes variants across all 8 folds, Scales 0 (`1.5s`) and 4 (`0.5s`), and conditions (`16k_mono`, `dfn`):
1. **`procrustes_stage4_subseg_weighted_all`**: Exact Stage 4 implementation (unweighted vstack, all subsegments).
2. **`procrustes_equal_rec_weighted_all`**: Weighted Procrustes SVD where each subsegment in training recording $r'$ receives weight $w_{r'} = 1 / N_{r'}$, giving each of the 7 training recordings exact $1/7$ weight.
3. **`procrustes_equal_rec_weighted_clean_only`**: Equal-recording-weighted Procrustes fitted exclusively on **clean single-speaker subsegments** (`4,234` at Scale 4; `1,090` at Scale 0).
4. **`procrustes_equal_rec_weighted_centered`**: Equal-recording-weighted Procrustes with training-set mean centering prior to SVD.

### 1.3 Summary of LORO Procrustes Results ([corrected_loro_results.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_loro_results.csv))

| Procrustes Variant | Audio Condition | Scale 0 (`1.5s`) Mean Train $R^2$ | Scale 0 (`1.5s`) Mean Held-Out Test $R^2$ | Scale 0 (`1.5s`) Held-Out Mean Cosine | Scale 4 (`0.5s`) Mean Train $R^2$ | Scale 4 (`0.5s`) Mean Held-Out Test $R^2$ | Scale 4 (`0.5s`) Held-Out Mean Cosine |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1. Stage 4 (Subseg-Weighted, All)`** | `16k_mono` | `+0.3383` | **`-0.3331`** | `0.4684` | `-0.0374` | **`-0.3281`** | `0.3882` |
| **`2. Equal-Rec-Weighted (All)`** | `16k_mono` | `+0.3104` | **`-0.3748`** | `0.4525` | `-0.0551` | **`-0.3436`** | `0.3805` |
| **`3. Equal-Rec-Weighted (Clean Only)`** | `16k_mono` | `+0.4174` | **`-0.4467`** | `0.4229` | `-0.0085` | **`-0.3657`** | `0.3698` |
| **`4. Equal-Rec-Weighted (Centered)`** | `16k_mono` | `+0.3169` | **`-0.3718`** | `0.4561` | `-0.0384` | **`-0.3299`** | `0.3773` |
| **`1. Stage 4 (Subseg-Weighted, All)`** | `dfn` | `+0.2829` | **`-0.4110`** | `0.4521` | `-0.0595` | **`-0.3480`** | `0.3906` |
| **`2. Equal-Rec-Weighted (All)`** | `dfn` | `+0.2529` | **`-0.4442`** | `0.4392` | `-0.0775` | **`-0.3663`** | `0.3820` |
| **`3. Equal-Rec-Weighted (Clean Only)`** | `dfn` | `+0.3659` | **`-0.5385`** | `0.4023` | `-0.0349` | **`-0.3893`** | `0.3711` |
| **`4. Equal-Rec-Weighted (Centered)`** | `dfn` | `+0.2614` | **`-0.4384`** | `0.4514` | `-0.0686` | **`-0.3606`** | `0.3822` |

**Scientific Conclusion on Linear Feature-Space Alignment**:
Across **all 8 folds, all 4 variants, both scales, and all audio conditions**, held-out LORO Procrustes $R^2$ is **strictly negative (`-0.198` to `-0.662`)**, and held-out cosine alignment stays low (`0.33–0.52`). Even when fitted exclusively on clean single-speaker segments with equal recording weights (which raises in-sample Scale-0 training $R^2$ to `+0.4174`), held-out test $R^2$ drops even further to `-0.4467` because a linear map between ResNet293 (`256-D`) and TitaNet (`192-D`) overfits to recording-specific speaker subspaces and fails to generalize to unseen speakers and recordings. **Vector-space linear projection between frozen TitaNet and ResNet293 is definitively falsified.**

---

## 2. Source-Code Audit & Correction of the LORO Reliability Gate

### 2.1 Flaws in Stage 4's LORO Gate (`run_stage4_complementarity.py` L884–904)
1. **Leaked Training Targets**: While Stage 4 held out recording `rec_id` during ridge regression fitting (`tr_rows = [r for r in rel_rows if r["rec_id"] != rec_id]`), the regression target `y_tr = delta_margin_res_minus_tita` was computed using **self-included speaker centroids** and included `OVERLAP::` and `PARTIAL_BOUNDARY::` subsegments (`5,290` rows).
2. **Unweighted Subsegment Fitting**: Stage 4 solved `np.linalg.solve(X_tr_aug.T @ X_tr_aug + 10.0 * I, X_tr_aug.T @ y_tr)` without inverse-recording-size weighting, giving `R06_rich_karen` ~35% of the regression weight.
3. **Hardcoded Sigmoid Scaling Factor (`5.0`)**: Line 902 mapped predicted margin differences to weights via `1.0 / (1.0 + np.exp(-5.0 * pred_delta))` with a fixed heuristic gain of `5.0`.

### 2.2 Corrected Leak-Free LORO Gate (`8b_loro_gate_corrected_loto_affinity`)
In [run_stage4_1_audit.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/run_stage4_1_audit.py#L933-L1055), we implemented a strictly leak-free LORO gate (`8b_loro_gate_corrected_loto_affinity`):
1. **Turn-Disjoint LOTO Targets**: Training targets $y_i = m_{\text{Res}}^{\text{LOTO}}(i) - m_{\text{Tita}}^{\text{LOTO}}(i)$ are computed strictly on **clean subsegments (`2,771` subsegments across the 5 LOTO-eligible recordings `R03, R05, R06, R07, R08`)** using turn-disjoint speaker prototypes $\mu_s^{(-t)}$.
2. **Equal-Recording Weighting**: Each training subsegment in recording $r'$ is weighted by $w_{r'} = 1 / N_{r'}^{\text{LOTO}}$, ensuring every training recording contributes equally to the ridge regression loss ($W = \text{diag}(w)$).
3. **Training-Only Feature Standardization**: Feature means and standard deviations (`resnet293_l2_norm`, `dfn_noise_reduction_db`, `vad_conf_mean`) are computed using the recording-weighted training distribution and frozen before evaluating held-out recording $r$.

### 2.3 Held-Out Gate Diagnostics & Downstream Impact ([corrected_loro_results.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/corrected_loro_results.csv))
- **Out-of-Fold Predictive Correlation ($r_{\text{pred, true}}$)**: Across the 5 LOTO-eligible held-out recordings (`R03, R05, R06, R07, R08`), the corrected LORO gate achieves positive held-out Pearson correlation with the true turn-disjoint $\Delta\text{margin}$ on all 5 recordings:
  - `R03_disabled_parking`: $r = +0.2109$
  - `R05_southwest_pilot`: $r = +0.1570$
  - `R06_rich_karen`: $r = +0.1797$
  - `R07_bwc_cheap_shot`: $r = +0.1476$
  - `R08_uk_parliament`: $r = +0.3766$ (Mean held-out $r = \mathbf{+0.2144}$).
- **Out-of-Fold $R^2$**: Because uncentered $\Delta\text{margin}$ means shift across recordings, uncalibrated MSE $R^2$ on held-out folds is slightly negative (`-0.0041` on `R06`, `-0.0196` on `R03`, `-0.0821` on `R07`), demonstrating that while the gate's **ranking direction** generalizes ($r > 0$ on every fold), its absolute scale shifts across recordings.
- **Downstream Diarization Comparison (`8` vs `8b`)**:
  - Stage 4 Leaked Gate (`8_loro_calibrated_gate_affinity`): Macro DER `46.04%` (`c=0`), `39.56%` (`c=0.25`), Macro JER `62.68%`, Macro cpWER `37.69%`, Macro-SACC `49.06%`.
  - Stage 4.1 Corrected LOTO Gate (`8b_loro_gate_corrected_loto_affinity`): Macro DER **`46.06%`** (`c=0`), **`39.70%`** (`c=0.25`), Macro JER **`62.61%`**, Macro cpWER **`37.70%`**, Macro-SACC **`49.46%`**.
  - Both gates produce identical speaker counts on all 8 recordings (`[3, 3, 3, 3, 3, 6, 1, 2]`, MAE `1.50`) and nearly identical DER/cpWER (`0.02%` DER difference, `0.01%` cpWER difference).
