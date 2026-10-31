# Stage 4.1 — Critical Issue B: Pairwise Evaluation & Threshold Leakage Report

**Audit Status**: `CORRECTED`  
**Supporting Artifacts**:
- [loro_verification_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_verification_metrics.csv) (Per-recording held-out verification results across all 8 folds, 5 methods, and 3 audio conditions)
- [loro_thresholds.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_thresholds.csv) (Training-fold Z-score parameters $\mu_{\text{tr}}, \sigma_{\text{tr}}$ and EER thresholds $\tau_{\text{tr}}$ for all 8 LORO folds)
- [loro_fusion_comparison.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_fusion_comparison.csv) (Held-out frozen-threshold HTER, actual DCF, ROC-AUC, and $2\times 2$ error contingency vs exploratory leaked EER)

---

## 1. Nature of the Threshold & Score-Normalization Leakage in Stage 4

In Stage 4 ([run_stage4_complementarity.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py#L323-L358)), `evaluate_turn_pairs_and_contingency` computed:
1. **In-Sample EER Thresholds**: `t_eer, t_thr = compute_eer_and_threshold(same, t_cos)` and `r_eer, r_thr = compute_eer_and_threshold(same, r_cos)` directly on the evaluation pairs (`9,817` pairs).
2. **In-Sample Z-Score Parameters**: `z_cos = 0.5 * ((t_cos - mean(t_cos))/std(t_cos) + (r_cos - mean(r_cos))/std(r_cos))` using the mean and standard deviation of the evaluation pairs.
3. **In-Sample $2\times 2$ Error Contingency & Oracle Gate**: Binary error indicators `t_err = (t_pred != same)` and `r_err = (r_pred != same)` were evaluated at the in-sample EER thresholds `t_thr` and `r_thr`.

While threshold-free ROC-AUC (or the exact continuous EER operating point on a single test set) is a standard descriptive summary of score separability, using in-sample thresholds to claim operational verification error rates, Z-score calibration gains, or $2\times 2$ error complementarity leaks test-set score distributions into the decision rule.

---

## 2. Corrected 8-Fold LORO Verification Protocol

In [run_stage4_1_audit.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/run_stage4_1_audit.py#L332-L542), we implemented a **strict 8-fold Leave-One-Recording-Out (LORO) verification protocol** on the `9,817` within-recording clean turn pairs:

For each held-out recording $r \in \{\text{R01}, \dots, \text{R08}\}$:
1. **Training Split**: Form the training pair set $\mathcal{P}_{\text{train}}^{(-r)}$ using only within-recording pairs from the other 7 recordings ($\mathcal{R} \setminus \{r\}$).
2. **Training-Only Score Normalization**: Fit $\mu_{T,\text{tr}}, \sigma_{T,\text{tr}}$ and $\mu_{R,\text{tr}}, \sigma_{R,\text{tr}}$ exclusively on $\mathcal{P}_{\text{train}}^{(-r)}$ and apply those frozen parameters to normalize scores in recording $r$:
   $$s_{\text{zscore}}(i,j) = \frac{1}{2}\left(\frac{s_T(i,j) - \mu_{T,\text{tr}}}{\sigma_{T,\text{tr}}} + \frac{s_R(i,j) - \mu_{R,\text{tr}}}{\sigma_{R,\text{tr}}}\right)$$
3. **Training-Only Reliability Gate**: Fit the 3-feature reliability gate (`resnet293_l2_norm`, `dfn_noise_reduction_db`, `vad_conf_mean`) exclusively on clean turn-disjoint LOTO subsegment margins from $\mathcal{R} \setminus \{r\}$ with equal recording weight, and freeze its parameters before scoring pairs in recording $r$.
4. **Frozen Decision Thresholds**: Select the EER decision threshold $\tau_{m,\text{tr}}^{(-r)}$ for each method $m$ strictly on $\mathcal{P}_{\text{train}}^{(-r)}$, and apply $\tau_{m,\text{tr}}^{(-r)}$ without modification to classify pairs in recording $r$.
5. **Held-Out Evaluation Metrics**:
   - **Frozen-Threshold FAR, FRR, and HTER**: $\text{HTER} = \frac{1}{2}(\text{FAR}(\tau_{\text{tr}}) + \text{FRR}(\tau_{\text{tr}}))$.
   - **Frozen-Threshold Actual DCF**: $\text{actDCF}(p_{\text{target}}) = C_{\text{miss}} \cdot \text{FRR}(\tau_{\text{tr}}) \cdot p_{\text{target}} + C_{\text{fa}} \cdot \text{FAR}(\tau_{\text{tr}}) \cdot (1 - p_{\text{target}}) / \min(C_{\text{miss}} p_{\text{target}}, C_{\text{fa}}(1-p_{\text{target}}))$ at $p_{\text{target}} \in \{0.01, 0.10\}$.
   - **Threshold-Free ROC-AUC**: Computed per recording and on out-of-fold (OOF) scores.
   - **Out-of-Fold $2\times 2$ Error Contingency Table**: Computed strictly from held-out binary decisions at frozen thresholds $\tau_{T,\text{tr}}^{(-r)}$ and $\tau_{R,\text{tr}}^{(-r)}$.

---

## 3. Corrected LORO Verification Results vs Exploratory Stage 4 Numbers

From [loro_fusion_comparison.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_fusion_comparison.csv):

| Audio Condition | Method | Exploratory In-Sample Pooled EER (Leaked) | Held-Out OOF Pooled FAR (Frozen $\tau_{\text{tr}}$) | Held-Out OOF Pooled FRR (Frozen $\tau_{\text{tr}}$) | Held-Out OOF Pooled HTER (Frozen $\tau_{\text{tr}}$) | Held-Out OOF Pooled ROC-AUC | Macro 6-Rec Held-Out HTER (Frozen $\tau_{\text{tr}}$) | Macro 6-Rec Held-Out ROC-AUC | Macro 6-Rec Exploratory EER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`16k_mono`** | **TitaNet-Large** | 24.33% | 14.49% | 32.64% | **23.57%** | **0.8285** | 31.88% | 0.8082 | 20.31% |
| **`16k_mono`** | **ResNet293-LM** | 28.75% | 23.62% | 32.64% | 28.13% | 0.7864 | 31.27% | 0.8260 | 19.91% |
| **`16k_mono`** | **Equal Cosine Fusion** | 25.46% | 18.20% | 30.98% | 24.59% | 0.8253 | **30.22%** | **0.8294** | **19.07%** |
| **`16k_mono`** | **Z-Score Fusion (Train-Cal)** | 25.54% | 18.31% | 30.91% | 24.61% | 0.8230 | 31.14% | 0.8180 | 19.37% |
| **`16k_mono`** | **Reliability Gate (LORO)** | 25.28% | 17.43% | 30.80% | 24.11% | 0.8265 | 31.14% | 0.8085 | 20.00% |
| **`dfn`** | **TitaNet-Large** | 30.34% | 21.85% | 36.32% | **29.08%** | 0.7724 | 33.76% | 0.8304 | 21.90% |
| **`dfn`** | **ResNet293-LM** | 31.86% | 28.27% | 34.65% | 31.46% | 0.7432 | **31.74%** | 0.8155 | 20.87% |
| **`dfn`** | **Equal Cosine Fusion** | 29.98% | 24.60% | 34.44% | 29.52% | 0.7735 | 33.03% | 0.8329 | 20.89% |
| **`dfn`** | **Z-Score Fusion (Train-Cal)** | 29.98% | 24.60% | 34.48% | 29.54% | 0.7779 | 33.08% | 0.8329 | 20.85% |
| **`dfn`** | **Reliability Gate (LORO)** | **29.81%** | 23.98% | 34.58% | 29.28% | **0.7788** | 32.79% | **0.8341** | **20.79%** |
| **`normalized`** | **TitaNet-Large** | 24.40% | 14.44% | 32.46% | **23.45%** | **0.8288** | 31.29% | 0.7986 | 28.28% |
| **`normalized`** | **ResNet293-LM** | 28.71% | 24.18% | 32.14% | 28.16% | 0.7858 | 31.54% | **0.8230** | 20.47% |
| **`normalized`** | **Equal Cosine Fusion** | 25.39% | 17.90% | 30.87% | 24.38% | 0.8252 | **29.89%** | 0.8061 | 20.15% |
| **`normalized`** | **Z-Score Fusion (Train-Cal)** | 25.50% | 18.07% | 30.94% | 24.51% | 0.8236 | 30.37% | 0.8068 | 19.98% |
| **`normalized`** | **Reliability Gate (LORO)** | 25.07% | 17.33% | 30.69% | 24.01% | 0.8276 | 30.75% | 0.8100 | **19.78%** |

---

## 4. Key Findings from Frozen-Threshold LORO Verification

1. **Severe Threshold Calibration Shift Between Recordings**:
   - Inspecting [loro_thresholds.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_thresholds.csv), when `R06_rich_karen` (`7,503` pairs) is in the training set (Folds `R01–R05, R07, R08`), the training EER threshold on `16k_mono` is $\tau_T \approx 0.143$ and $\tau_R \approx 0.235$.
   - However, when `R06_rich_karen` is held out (Fold `R06`), the remaining 7 recordings (`2,191` pairs) have much higher cosine similarities, yielding training thresholds $\tau_T = 0.2039$ and $\tau_R = 0.2744$.
   - Applying $\tau_T = 0.2039$ to held-out `R06_rich_karen` causes a massive spike in False Reject Rate ($\text{FRR} = 43.73\%$ vs $\text{FAR} = 8.96\%$). Conversely, applying the `R06`-dominated threshold ($\tau_T \approx 0.143$) to `R03_disabled_parking`, `R05_southwest_pilot`, or `R08_uk_parliament` causes high False Accept Rates ($\text{FAR} = 42.56\%$ on `R03`, $35.54\%$ on `R05`, and $100\%$ on `R08`!).
   - Consequently, **Macro 6-Recording Held-Out HTER (`30.22%–33.76%`) is `+10%` to `+12%` worse than Macro 6-Recording In-Sample Exploratory EER (`19.07%–21.90%`)**. Uncalibrated cosine similarity thresholds do NOT transfer cleanly across diverse BWC acoustic environments.
2. **Held-Out $2\times 2$ Error Contingency Confirms Genuine Error Diversity**:
   - Even under strictly frozen LORO thresholds ([loro_fusion_comparison.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_fusion_comparison.csv)), TitaNet and ResNet293 make substantially different errors on the `9,817` within-recording pairs:
     - On `16k_mono`: Both correct $n_{00} = 6,517$ (`66.38%`), TitaNet wrong / ResNet right $n_{10} = 726$ (`7.40%`), TitaNet right / ResNet wrong $n_{01} = 1,364$ (`13.89%`), Both wrong $n_{11} = 1,210$ (`12.33%`). Held-out error Jaccard is **`0.3667`** (`63.33%` of errors are single-encoder complementary errors), closely matching the exploratory Jaccard (`0.3827`).
     - On `dfn`: $n_{00} = 5,998$, $n_{10} = 863$, $n_{01} = 1,265$, $n_{11} = 1,691$, yielding held-out error Jaccard **`0.4428`** (`55.72%` complementary errors).
3. **Why Score Fusion Improves `R03` and `R07` but Not Pooled `16k_mono`**:
   - Per-recording results ([loro_verification_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/loro_verification_metrics.csv)) show that on `R03_disabled_parking` (`1,275` pairs, `16k_mono`), ResNet293 (`ROC-AUC = 0.9197, HTER = 15.02%`) outperforms TitaNet (`ROC-AUC = 0.9028, HTER = 24.52%`), and fusion achieves the highest ROC-AUC (`0.9239` with reliability gate, `0.9234` with equal fusion). On `R07_bwc_cheap_shot` (`561` pairs), fusion also slightly improves ROC-AUC (`0.7993` vs `0.7965`).
   - However, on `R06_rich_karen` (`7,503` pairs) and `R05_southwest_pilot` (`300` pairs), TitaNet is substantially stronger than ResNet293 (`ROC-AUC 0.8036` vs `0.7471` on `R06`; `0.8462` vs `0.7321` on `R05`), so averaging with ResNet293 slightly dilutes TitaNet on `16k_mono` (`OOF ROC-AUC 0.8265` for gate / `0.8253` for equal fusion vs `0.8285` for TitaNet alone).
   - On `dfn`, where TitaNet degrades (`OOF ROC-AUC 0.7724`), the LORO Reliability Gate slightly improves held-out OOF ROC-AUC to **`0.7788`** (and Macro 6-Rec ROC-AUC from `0.8304` to **`0.8341`**).
