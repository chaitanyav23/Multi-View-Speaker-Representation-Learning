# Stage 4.1 — Critical Issue C: Centroid Leakage & Reliability Indicator Audit

**Audit Status**: `CORRECTED`  
**Supporting Artifacts**:
- [reliability_loro_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/reliability_loro_metrics.csv) (Comparison of all 5 protocols across 12 indicators on `16k_mono` and `dfn`)
- [reliability_per_recording.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/reliability_per_recording.csv) (Per-recording Spearman correlations and raw AUROCs under the leak-free LOTO protocol)

---

## 1. Four Methodological Flaws Identified in Stage 4's Reliability Analysis

Our line-by-line audit of [run_stage4_complementarity.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_complementarity/run_stage4_complementarity.py#L697-L879) uncovered four distinct flaws in how Stage 4 evaluated reliability indicators:

### Flaw 1: Centroid Self-Inclusion & Same-Turn Leakage (`L702–711`)
In Stage 4, speaker centroids `t_centroids[s]` and `r_centroids[s]` were computed by averaging all clean Scale-4 (`0.5s`) subsegments of speaker `s` in that recording, **including the query subsegment `i` itself and adjacent overlapping `0.5s` subsegments (`0.25s` shift) from the exact same ground-truth turn**:
```python
t_centroids = {s: l2_normalize_rows(np.mean(t_emb[idx], axis=0, keepdims=True))[0] for s, idx in spk_to_clean_idx.items()}
```
When speaker `s` has $K$ clean subsegments in the recording, subsegment `i` contributes $1/K$ of its own target centroid, and adjacent subsegments from the same turn share 50% waveform overlap (`0.25s` step on `0.5s` window).

### Flaw 2: Inclusion of `OVERLAP::` and `PARTIAL_BOUNDARY::` Subsegments in the Evaluation Set (`L714–717`, `L835`)
While `spk_to_clean_idx` was restricted to clean subsegments (`4,234` subsegments), lines 714–717 stripped `"OVERLAP::"` and `"PARTIAL_BOUNDARY::"` prefixes via `raw_lbl.split("::")[-1]`, and line 835 (`rel_rows = [r for r in subseg_rel_rows if not np.isnan(r["tita_margin"])]`) **did not filter out `is_overlap` or `PARTIAL_BOUNDARY` subsegments**!  
Consequently, Stage 4 evaluated **`5,290` subsegments** (`4,937` after dropping `R02`'s 353 single-clean-speaker rows): `3,869` clean + `703` `OVERLAP::` + `353` `PARTIAL_BOUNDARY::` subsegments.

### Flaw 3: In-Sample Sign Rectification of AUROC (`L857–858`)
Stage 4 reported `auroc = max(auroc, 1.0 - auroc)` over all recordings combined, without checking whether the indicator's direction (sign) generalized from training recordings to held-out recordings.

### Flaw 4: `overlap_prob_mean` Collapse Bug (`step03_vad.py` L48 & `run_stage2_baseline.py` L121)
In [step03_vad.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step03_vad.py#L48), `VoiceActivityDetection._segmentation` registers a `pre_aggregation_hook` that takes `np.max(scores, axis=-1, keepdims=True)`, collapsing `pyannote/segmentation`'s 3-speaker output `[frames, 3]` to `[frames, 1]`. Therefore, in [run_stage2_baseline.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage2_baseline/run_stage2_baseline.py#L121), `raw_data.shape[-1] >= 2` was always `False`, setting `p_overlap = np.zeros_like(p_speech)`. Every single subsegment has `overlap_prob_mean == 0.0`, explaining why `overlap_prob_mean` had `NaN` Spearman correlation and `0.5000` AUROC.

---

## 2. Five-Protocol Ablation & Leak-Free LOTO Correction

To isolate the exact impact of each flaw and establish a leak-free benchmark, [run_stage4_1_audit.py](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/run_stage4_1_audit.py#L549-L853) evaluated 5 progressively stricter protocols:

1. **`1_stage4_leaked_all5290` (`N = 4,937`, 7 multi-speaker recs)**: Exact reproduction of Stage 4 (self-included centroids + includes `OVERLAP::` and `PARTIAL_BOUNDARY::` subsegments).
2. **`2_self_included_clean4234` (`N = 3,869`, 7 recs)**: Self-included centroids, but restricted strictly to clean single-speaker subsegments (`>=75%` purity, no overlap/boundary).
3. **`3_corrected_LOTO_clean_subsegs` (`N = 2,771`, 5 eligible recs `R03, R05, R06, R07, R08`)**: **Turn-Disjoint Leave-One-Turn-Out (LOTO)** at the subsegment level. For every clean subsegment $i$ belonging to ground-truth turn $t$ of speaker $s$, the target speaker prototype $\mu_{s}^{(-t)}$ is computed exclusively from clean subsegments of speaker $s$ in **other turns $t' \neq t$** within the same recording. Zero waveform overlap and zero self-inclusion exist between query subsegment $i$ and any prototype $\mu_{s}^{(-t)}$.
4. **`4_corrected_LOTO_clean_turns` (`N = 237` turns, 5 recs)**: **Turn-Level LOTO**. Each turn is represented by a single pooled embedding, and the target prototype is computed from the speaker's other turns $t' \neq t$.
5. **`5_corrected_split_enrollment_turns` (`N = 122` test turns, 5 recs)**: **Chronological Split Enrollment/Test**. The first 50% of each speaker's clean turns in a recording form a fixed enrollment prototype, and only the remaining 50% of turns are evaluated as held-out test queries.

Furthermore, for all 5 protocols, we replaced in-sample `max(auroc, 1 - auroc)` with **LORO Held-Out Sign-Calibrated AUROC**, where the indicator sign ($+1$ or $-1$) is determined strictly on the training recordings and applied to the held-out recording.

---

## 3. Side-by-Side Comparison: Leaked Stage 4 vs Corrected Turn-Disjoint LOTO

From [reliability_loro_metrics.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/reliability_loro_metrics.csv), here are the key indicators on `16k_mono` and `dfn`:

### A. Predicting ResNet293 Correctness & Margin (`16k_mono`)

| Indicator | Stage 4 Leaked (`N=4937`) $\rho(\text{Res margin})$ | Stage 4 Leaked In-Sample AUROC | **Corrected LOTO Subsegs (`N=2771`) $\rho(\text{Res margin})$** | **Corrected LOTO Subsegs Held-Out AUROC** | **Corrected LOTO Turns (`N=237`) $\rho(\text{Res margin})$** | **Corrected LOTO Turns Held-Out AUROC** | **Split Enrollment Turns (`N=122`) Held-Out AUROC** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`resnet293_l2_norm`** | +0.6015 | 0.7714 | **+0.6952** | **0.8419** | **+0.6641** | **0.7635** | **0.7584** |
| **`dfn_noise_reduction_db`** | +0.4563 | 0.7212 | **+0.5364** | **0.7821** | **+0.5218** | **0.7111** | **0.7291** |
| **`vad_conf_mean`** | +0.4119 | 0.7115 | **+0.4634** | **0.7499** | **+0.3526** | **0.6281** | **0.6189** |
| **`resnet293_cross_scale_cos`** | +0.4049 | 0.6646 | **+0.4746** | **0.7157** | **+0.4081** | **0.6278** | **0.6995** |
| **`titanet_cross_scale_cos`** | +0.3763 | 0.6779 | **+0.4217** | **0.7186** | **+0.3784** | **0.6145** | **0.6372** |
| **`vad_conf_std`** | -0.3624 | 0.6715 | **-0.3946** | **0.7017** | -0.1673 | 0.5090 | 0.5152 |
| **`titanet_l2_norm`** | +0.0685 | 0.5806 | +0.0385 | 0.5267 | -0.0196 | 0.4772 | 0.5384 |
| **`snr_local_db`** | +0.0742 | 0.5146 | +0.0873 | 0.3509 | +0.2608 | 0.4048 | 0.4141 |
| **`rms_dbfs`** | +0.0231 | 0.5419 | -0.1082 | 0.4022 | -0.1957 | 0.6010 | 0.6250 |
| **`spectral_flatness`** | -0.0125 | 0.5450 | +0.0879 | 0.4935 | +0.1049 | 0.4692 | 0.4271 |

### B. Predicting Differential Encoder Advantage ($\Delta\text{margin} = m_{\text{Res}} - m_{\text{Tita}}$ and "ResNet293 Wins Margin")

| Indicator (`16k_mono`) | Stage 4 Leaked $\rho(\Delta\text{margin})$ | **Corrected LOTO Subsegs (`N=2771`) $\rho(\Delta\text{margin})$** | **Corrected LOTO Subsegs Held-Out AUROC (Res Wins)** | **Corrected LOTO Turns (`N=237`) $\rho(\Delta\text{margin})$** | **Corrected LOTO Turns Held-Out AUROC (Res Wins)** | **Split Enrollment (`N=122`) Held-Out AUROC (Res Wins)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`resnet293_l2_norm`** | +0.2849 | **+0.3072** | **0.6544** | **+0.4745** | **0.7024** | **0.7305** |
| **`dfn_noise_reduction_db`** | +0.2293 | **+0.1986** | **0.6040** | **+0.2702** | **0.5934** | **0.6108** |
| **`resnet293_cross_scale_cos`** | +0.1953 | **+0.2188** | **0.6077** | **+0.2838** | **0.6253** | **0.6672** |
| **`vad_conf_mean`** | +0.1661 | **+0.1563** | **0.5911** | +0.0728 | 0.5012 | 0.5390 |
| **`cross_scale_cos_diff`** | +0.1114 | **+0.1564** | **0.5761** | +0.0787 | 0.5542 | 0.5574 |
| **`titanet_l2_norm`** | -0.0313 | -0.0457 | 0.5257 | -0.0568 | 0.4385 | 0.4267 |

---

## 4. Scientific Conclusions from the Corrected Reliability Audit

1. **The Top 3 Reliability Signals Are Genuine and Survive Strict Turn-Disjoint LOTO**:
   - Removing centroid self-inclusion and `OVERLAP::`/`PARTIAL_BOUNDARY::` contamination **actually strengthens** the Spearman correlation and held-out AUROC of the top three indicators at the clean subsegment level (`N = 2,771` across `R03, R05, R06, R07, R08`):
     - **`resnet293_l2_norm`**: $\rho(\text{Res margin}) = \mathbf{+0.6952}$ (`16k_mono`) and $\mathbf{+0.6916}$ (`dfn`), Held-Out Correctness AUROC = **`0.8419`** (`16k_mono`) and **`0.8463`** (`dfn`), and Held-Out "ResNet Wins Margin" AUROC = **`0.6544`** (subsegments), **`0.7024`** (LOTO turns), and **`0.7305`** (split enrollment turns).
     - **`dfn_noise_reduction_db`**: $\rho(\text{Res margin}) = \mathbf{+0.5364}$ (`16k_mono`) and $\mathbf{+0.5410}$ (`dfn`), Held-Out Correctness AUROC = **`0.7821`** (`16k_mono`) and **`0.7552`** (`dfn`).
     - **`vad_conf_mean`** and **`resnet293_cross_scale_cos`**: Held-Out Correctness AUROC = **`0.7499`** and **`0.7157`** on `16k_mono` subsegments (`0.7487` and `0.7320` on `dfn`).
2. **Per-Recording Consistency Verified ([reliability_per_recording.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage4_1/reliability_per_recording.csv))**:
   - Unlike `snr_local_db` and `rms_dbfs`—whose correlations flip signs across recordings (`-0.1424` on `R06` vs `+0.0579` on `R08`, causing their held-out sign-calibrated AUROC to drop below `0.50` to `0.3509–0.4022`)—**`resnet293_l2_norm`**, **`dfn_noise_reduction_db`**, **`vad_conf_mean`**, and **`resnet293_cross_scale_cos`** have **strictly positive Spearman correlations and AUROC $> 0.65$ in all 5 eligible recordings (`R03, R05, R06, R07, R08`)**.
3. **Why `resnet293_l2_norm` Is Informative While `titanet_l2_norm` Is Not**:
   - Pre-L2-normalization ResNet293 embedding magnitude (`resnet293_l2_norm`) acts as a natural acoustic concentration/confidence proxy that collapses under severe noise or short/corrupted frames ($\rho = +0.6952$), whereas TitaNet's attentive statistics pooling + batch-normalized linear projection constrains its pre-norm embedding norm near a narrow shell (`AUROC = 0.5267–0.5558`).
4. **Effect Size on Differential Advantage ($\Delta\text{margin}$) Is Real but Modest at the Subsegment Level**:
   - Because both encoders degrade under heavy noise ($\rho(\text{Tita margin}) = +0.5446$, $\rho(\text{Res margin}) = +0.6952$ for `resnet293_l2_norm`), the indicators primarily predict **overall acoustic quality / both-encoder correctness** (`AUROC 0.75–0.85`), and secondarily predict **which encoder has the larger margin** ($\rho(\Delta\text{margin}) = +0.3072$ on subsegments, $+0.4745$ on turns; Held-Out AUROC `0.60–0.73`).
