# Stage 4.1 — Critical Issue D: Statistical Dependence & Clustered Uncertainty Analysis

**Audit Status**: `CORRECTED`  
**Supporting Artifacts**:
- [clustered_bootstrap_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/clustered_bootstrap_metrics.csv) (`116` rows: point estimates and 95% confidence intervals across all diarization methods and verification protocols)
- [paired_method_differences.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/paired_method_differences.csv) (`67` rows: paired differences $\Delta(\text{Method A} - \text{Method B})$, 95% CIs, zero-inclusion flags, and per-recording win/loss/tie counts)

---

## 1. Why Pair-Level Resampling Is Statistically Invalid

In speaker verification and diarization on conversational audio:
1. **`9,817` within-recording turn pairs are generated from only `254` turns across `8` recordings (`30` clean speakers)**. Because $\binom{N_r}{2}$ pairs are formed quadratically from $N_r$ turns, pairs sharing a turn $(i,j)$ and $(i,k)$ are algebraically dependent. Treating `9,817` pairs (or `1,251,231` subsegment pairs) as independent samples underestimates standard errors by an order of magnitude.
2. **Recording-level clustering (`R01–R08`) is the primary source of variance**: A single recording (`R06_rich_karen`) contains `123` clean turns and generates `7,503 / 9,817` (`76.43%`) of all within-recording turn pairs. Resampling individual pairs or turns without accounting for recording-level clustering masks the massive heterogeneity across recordings.

---

## 2. Three-Tier Clustered Bootstrap Methodology (`B = 2,000`, `seed = 42`)

In [run_stage4_1_audit.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/run_stage4_1_audit.py#L1063-L1294), we implemented three clustered bootstrap schemes with `B = 2,000` replicates (`np.random.default_rng(42)`):

1. **Recording-Level Cluster Bootstrap (`recording_cluster_bootstrap`)**:
   - Resamples `8` recordings with replacement from `R01–R08` (`R* = [r_1, ..., r_8]`).
   - Used for all downstream diarization metrics (`Macro DER c=0`, `Macro DER c=0.25`, `Macro JER`, `Macro cpWER`, `Macro saWER`, `Macro Micro-SACC`, `Macro Macro-SACC`, `Macro Speaker Count MAE`) and **Paired Recording-Level Differences** $\Delta = \text{Metric}(A) - \text{Metric}(B)$ evaluated on the exact same resampled recording draw.
2. **Within-Recording Turn-Clustered Bootstrap (`turn_clustered_bootstrap_within_rec`)**:
   - Within each recording $r$, resamples $N_r$ turns with replacement ($T_r^*$) and retains all valid off-diagonal pairs $(u_a, u_b)$ ($a < b, \text{turn}(u_a) \neq \text{turn}(u_b)$) induced by the resampled turns.
   - Captures turn-level sampling uncertainty conditional on the 8 fixed recordings.
3. **Two-Stage Hierarchical Recording-then-Turn Cluster Bootstrap (`hierarchical_recording_then_turn_bootstrap`)**:
   - Stage 1: Resamples `8` recordings with replacement from `R01–R08`.
   - Stage 2: Within each sampled recording, resamples its turns with replacement and forms the induced off-diagonal pairs.
   - Captures both cross-recording population variance and within-recording turn variance.

---

## 3. Verification EER Uncertainty & Paired Encoder Comparisons

From [clustered_bootstrap_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/clustered_bootstrap_metrics.csv) and [paired_method_differences.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/paired_method_differences.csv):

### A. Point Estimates and 95% Confidence Intervals for Within-Recording Turn EER (`9,817` pairs)

| Audio Condition | Method | Point Estimate EER | Turn-Clustered 95% CI (Fixed 8 Recs) | CI Width (Turn) | Hierarchical Rec+Turn 95% CI (Cross-Rec) | CI Width (Hierarchical) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`16k_mono`** | **TitaNet-Large** | **24.33%** | `[20.35%, 28.81%]` | 8.46% | `[16.24%, 30.86%]` | 14.62% |
| **`16k_mono`** | **ResNet293-LM** | **28.75%** | `[23.93%, 33.26%]` | 9.33% | `[14.42%, 35.46%]` | 21.04% |
| **`16k_mono`** | **Equal Cosine Fusion** | **25.46%** | `[20.91%, 30.02%]` | 9.11% | `[13.93%, 31.91%]` | 17.98% |
| **`dfn`** | **TitaNet-Large** | **30.34%** | `[25.84%, 34.23%]` | 8.39% | `[18.41%, 35.36%]` | 16.95% |
| **`dfn`** | **ResNet293-LM** | **31.86%** | `[27.59%, 35.60%]` | 8.01% | `[15.72%, 38.17%]` | 22.45% |
| **`dfn`** | **Equal Cosine Fusion** | **29.98%** | `[25.64%, 34.13%]` | 8.49% | `[16.49%, 36.38%]` | 19.89% |

### B. Paired Verification EER Differences $\Delta\text{EER}(A - B)$

| Audio Condition | Comparison ($A - B$) | Point Diff $\Delta\text{EER}$ | Turn-Clustered 95% CI | Turn CI Includes 0? | Hierarchical Rec+Turn 95% CI | Hierarchical CI Includes 0? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`16k_mono`** | `ResNet293 - TitaNet` | **+4.42%** | `[+1.22%, +7.44%]` | **No** (TitaNet better) | `[-3.77%, +7.61%]` | **Yes** |
| **`16k_mono`** | `Equal Fusion - TitaNet` | **+1.13%** | `[-1.10%, +3.08%]` | **Yes** | `[-3.68%, +3.03%]` | **Yes** |
| **`16k_mono`** | `Equal Fusion - ResNet293` | **-3.29%** | `[-5.03%, -1.65%]` | **No** (Fusion better) | `[-5.35%, +0.78%]` | **Yes** |
| **`dfn`** | `ResNet293 - TitaNet` | **+1.52%** | `[-1.61%, +4.53%]` | **Yes** | `[-5.61%, +5.81%]` | **Yes** |
| **`dfn`** | `Equal Fusion - TitaNet` | **-0.36%** | `[-2.15%, +1.83%]` | **Yes** | `[-3.79%, +2.91%]` | **Yes** |
| **`dfn`** | `Equal Fusion - ResNet293` | **-1.88%** | `[-3.39%, -0.05%]` | **No** (Fusion better) | `[-3.76%, +2.27%]` | **Yes** |

**Interpretation**:
- Conditional on the 8 fixed recordings (`turn_clustered_bootstrap_within_rec`), TitaNet is significantly better than ResNet293 on `16k_mono` (`+4.42%`, 95% CI `[+1.22%, +7.44%]`), and Equal Fusion is significantly better than ResNet293 on both `16k_mono` (`-3.29%`, 95% CI `[-5.03%, -1.65%]`) and `dfn` (`-1.88%`, 95% CI `[-3.39%, -0.05%]`), while indistinguishable from TitaNet (`-0.36%`, 95% CI `[-2.15%, +1.83%]` on `dfn`).
- However, under the **hierarchical recording-then-turn bootstrap**, **every 95% confidence interval spans zero**, because when `R06_rich_karen` is not drawn in a bootstrap replicate, `R03_disabled_parking` dominates the pair pool and reverses the ranking between ResNet293 (`13.88%` EER on `R03`) and TitaNet (`17.64%` EER on `R03`).

---

## 4. Downstream Diarization Paired Differences & Uncertainty (`N = 8` Recordings)

From [paired_method_differences.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/paired_method_differences.csv), comparing key downstream diarization methods against the Phase 20 baseline (`1_titanet_nmesc` on DFN, Macro DER `46.22%`, Macro cpWER `38.07%`):

| Comparison ($A$ vs $B = \text{1\_titanet\_nmesc}$) | $\Delta\text{Macro DER}_{c=0}$ `[95% CI]` | Includes 0? | Rec Wins ($A/B/\text{Tie}$) | $\Delta\text{Macro cpWER}$ `[95% CI]` | Includes 0? | Rec Wins ($A/B/\text{Tie}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2_resnet293_nmesc` (DFN)** | `+1.38% [-0.14%, +3.40%]` | Yes | 2 / 4 / 2 | `+1.89% [-0.68%, +5.84%]` | Yes | 2 / 3 / 3 |
| **`4_equal_affinity_050` (DFN)** | `-0.16% [-0.51%, +0.10%]` | Yes | 4 / 3 / 1 | `+0.25% [-1.18%, +2.02%]` | Yes | 4 / 2 / 2 |
| **`6_heuristic_quality_affinity` (DFN)** | `-0.13% [-0.52%, +0.14%]` | Yes | 3 / 4 / 1 | `-0.29% [-1.32%, +0.62%]` | Yes | 4 / 2 / 2 |
| **`7_loro_procrustes_192d` (DFN)** | `+0.15% [-0.32%, +0.67%]` | Yes | 3 / 4 / 1 | `+0.95% [-0.82%, +2.77%]` | Yes | 3 / 3 / 2 |
| **`8_loro_calibrated_gate_affinity` (DFN)** | `-0.18% [-0.52%, +0.08%]` | Yes | 5 / 2 / 1 | `-0.38% [-1.36%, +0.46%]` | Yes | 4 / 2 / 2 |
| **`8b_loro_gate_corrected_loto_affinity`** | `-0.16% [-0.55%, +0.11%]` | Yes | 4 / 3 / 1 | `-0.37% [-1.36%, +0.48%]` | Yes | 4 / 2 / 2 |
| **`10_resnet293_mono_nmesc` (`16k_mono`)** | `-0.25% [-1.50%, +1.12%]` | Yes | 5 / 2 / 1 | `-1.63% [-7.67%, +3.50%]` | Yes | 4 / 2 / 2 |
| **`11_hybrid_titaDFN_resMONO_zscore`** | `-0.05% [-0.86%, +0.89%]` | Yes | 5 / 1 / 2 | **`-1.02% [-2.13%, -0.12%]`** | **No** | **5 / 1 / 2** |

### Which Paired Differences Exclude Zero at the 95% Level?
Across all 67 paired tests in [paired_method_differences.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/paired_method_differences.csv), **only four downstream diarization comparisons exclude zero**:
1. **`10_resnet293_mono_nmesc` vs `2_resnet293_nmesc` (DFN)**:
   - $\Delta\text{Macro DER}_{c=0} = \mathbf{-1.63\%}$ (`95% CI [-2.66%, -0.68%]`, wins `7 / 0 / 1` recordings).
   - $\Delta\text{Macro DER}_{c=0.25} = \mathbf{-2.27\%}$ (`95% CI [-4.21%, -0.63%]`, wins `6 / 1 / 1` recordings).
   - **Finding**: Feeding unenhanced `16k_mono` audio to ResNet293-LM instead of DeepFilterNet-enhanced audio yields a statistically significant, consistent improvement across 7 of 8 recordings (the 8th is `R07` where all methods tie at 1 speaker).
2. **`8b_loro_gate_corrected_loto_affinity` vs `4_equal_affinity_050` on `cpWER`**:
   - $\Delta\text{Macro cpWER} = \mathbf{-0.62\%}$ (`95% CI [-1.69%, -0.03%]`, wins `3 / 0 / 5` recordings), driven primarily by preserving 6 clusters instead of 5 on `R06_rich_karen`.
3. **`11_hybrid_titaDFN_resMONO_zscore` vs `1_titanet_nmesc` on `cpWER`**:
   - $\Delta\text{Macro cpWER} = \mathbf{-1.02\%}$ (`95% CI [-2.13%, -0.12%]`, wins `5 / 1 / 2` recordings).
4. **`7_loro_procrustes_192d` vs `1_titanet_nmesc` on `Macro-SACC`**:
   - $\Delta\text{Macro-SACC} = \mathbf{-1.06\%}$ (`95% CI [-2.26%, -0.10%]`, loses `1 / 5 / 2` recordings — Procrustes significantly degrades speaker attribution accuracy).
5. **`12_diagnostic_oracle_spkcount_equal_affinity` vs `4_equal_affinity_050`**:
   - Forcing the ground-truth speaker count $K_{\text{ref}}$ into NME-SC **statistically significantly worsens** $\text{Macro DER}_{c=0}$ by **`+8.77%`** (`95% CI [+2.02%, +16.39%]`, loses `0 / 6 / 2` recordings) and $\text{Macro cpWER}$ by **`+14.72%`** (`95% CI [+2.83%, +27.82%]`).

**All DFN-only affinity fusion improvements over `1_titanet_nmesc` (`-0.16%` to `-0.18%` Macro DER, `-0.37%` to `-0.38%` Macro cpWER) have 95% confidence intervals that include zero (`[-0.55%, +0.11%]` for DER) and are statistically indistinguishable from noise at $N=8$ recordings.**
