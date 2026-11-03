# Stage 5A.1.2 Final Freeze Record: Pretrained Multi-View Representation Fusion

**Date of Freeze**: November 3, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Freeze Decision**: **CONDITIONAL PASS — Freeze Approved with Qualifications**  
**Git Commit**: `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`  
**Target Repository**: [`Multi-View-Speaker-Representation-Learning`](https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning)

---

## 1. Scope of the Final Freeze

This freeze record formally supersedes all prior preliminary records and establishes the definitive, unalterable baseline of Stage 5A.1 representation-fusion findings. Following the evidence-backed resolution of all four audit inquiries in Stage 5A.1.2, this document codifies the exact specifications, verified metrics, approved claims, and prohibited claims before any Stage 5B architectural interventions begin.

---

## 2. Frozen Specifications and Architecture

### 2.1 Pretrained Encoders
1. **View 1 (Acoustic Noise Robust)**: NVIDIA TitaNet-Large
   - Checkpoint: `nvidia/speakerverification_titanet_large` (NeMo Toolkit v2.7.3)
   - Input Audio: DeepFilterNet3-enhanced (`DFN`) 16 kHz mono
   - Embedding Dimension: 192 (unit L2-normalized internally)
2. **View 2 (Conversational / Phonetic Robust)**: WeSpeaker ResNet293-LM
   - Checkpoint: VoxCeleb1+2 pretrained
   - Input Audio: Original 16 kHz mono (`16k_mono`)
   - Embedding Dimension: 256 (unit L2-normalized; raw Euclidean norm recorded prior to unit normalization)

### 2.2 Evaluation Protocol and Benchmark
- **Dataset**: 8 noisy conversational body-worn camera (BWC) audio recordings (`R01` through `R08`).
- **Turn Selection**: 50 ms tolerance-based boundary containment, with overlap exclusion and ground-truth speaker-label matching (**250 clean turns**).
- **Pairwise Verification**: Exhaustive intra-recording pairs ($N_r(N_r-1)/2$) summing to **9,489 pairwise trials** (2,766 positive, 6,723 negative).
- **Macro Formulations**:
  - **Six-Recording Eligible Macro**: `R01`, `R03`, `R05`, `R06`, `R07`, `R08` (includes R01's 1 pos / 14 neg pairs).
  - **Five-Recording Balanced-Session Sensitivity Macro**: `R03`, `R05`, `R06`, `R07`, `R08` (excludes R01).
  - `R02` and `R04` are excluded from all verification macros due to mathematically undefined ROC-AUC ($0/0$).

### 2.3 Frozen Affinity and Fusion Equations
1. **Affinity Normalization**:
   $$Z_{ij} = \frac{A_{ij} - \mu_{\text{offdiag}}}{\sigma_{\text{offdiag}}} \quad (i \ne j)$$
   $$\tilde{A} = \text{ScalerMinMax}\left( \text{fill\_diagonal}(\sigma(Z), 1.0) \right)$$
2. **Equal-Weight Affinity Fusion (Condition J)**:
   $$A_{\text{equal}} = 0.5 \cdot \tilde{A}_{\text{tita}} + 0.5 \cdot \tilde{A}_{\text{res}}$$
3. **Linear Clipped Reliability Gate (Condition K)**:
   $$\mathbf{x}_{rel} = [\text{resnet293\_l2\_norm}, \text{dfn\_noise\_reduction\_db}, \text{vad\_conf\_mean}]^T$$
   $$\tilde{\mathbf{x}} = \frac{\mathbf{x}_{rel} - \boldsymbol{\mu}_x}{\boldsymbol{\sigma}_x + \epsilon}$$
   $$\Delta_{\text{pred}} = \mathbf{w}^T \tilde{\mathbf{x}} + b$$
   $$g_{\text{res}} = \text{clip}(0.5 + 2.0 \cdot \Delta_{\text{pred}},\, 0.15,\, 0.85)$$
   $$G_{ij} = 0.5 \cdot (g_{\text{res}, i} + g_{\text{res}, j})$$
   $$A_{\text{gate}} = (1 - G) \odot \tilde{A}_{\text{tita}} + G \odot \tilde{A}_{\text{res}}$$

---

## 3. Definitive Accepted Performance Metrics

### 3.1 Verification Performance
| Condition / Method | 6-Rec Eligible Macro ROC-AUC | 6-Rec Eligible Macro HTER | 5-Rec Sensitivity Macro ROC-AUC | 5-Rec Sensitivity Macro HTER |
| :--- | :---: | :---: | :---: | :---: |
| Condition A: Raw TitaNet-DFN | 0.8238 | 33.80% | 0.8172 | 34.13% |
| Condition B: Raw ResNet293-Mono | 0.8230 | 32.40% | 0.8304 | 33.16% |
| Condition H: Norm TitaNet Alone (Control) | 0.8238 | 33.80% | 0.8172 | 34.13% |
| Condition I: Norm ResNet Alone (Control) | 0.8230 | 32.40% | 0.8304 | 33.16% |
| Condition J: Norm Equal Fusion (Control) | **0.8450** | 32.69% | **0.8426** | 32.79% |
| Condition K: Norm Linear Reliability Gate | **0.8458** | **31.90%** | **0.8436** | **32.56%** |

### 3.2 Downstream Diarization Performance (8 Recordings, collar=0.0s, overlap included)
| Metric | Condition A (Tita) | Condition B (Res) | Condition J (Equal Fusion) | Condition K (Gate) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro DER** | 46.22% | 45.98% | 46.17% | **45.92%** |
| **Macro Missed Speech** | 24.41% | 24.41% | 24.41% | 24.41% |
| **Macro False Alarm** | 13.33% | 13.33% | 13.33% | 13.33% |
| **Macro Speaker Confusion** | 8.48% | 8.24% | 8.43% | **8.18%** |
| **Speaker Count MAE** | 1.500 | 1.625 | **1.250** | 1.625 |
| **Macro cpWER** | 38.07% | **36.44%** | 37.05% | 37.74% |
| **Macro SAWER** | 46.58% | 46.58% | **46.45%** | 46.65% |
| **Single-Speaker Collapses** | 1 (R07) | 1 (R07) | 1 (R07) | 1 (R07) |

---

## 4. Approved Research Claims (Retained for Paper)

1. **Error Complementarity**: Pretrained TitaNet-Large and ResNet293-LM make disjoint errors on **58.03%** of failing pairwise verification trials ($2,077 / 3,579$, Error Jaccard = 0.4197) under LORO threshold calibration on noisy BWC audio.
2. **Verification Gain**: Training-free equal affinity fusion improves verification macro ROC-AUC by **+0.0212** over TitaNet and **+0.0220** over ResNet on the 6-recording macro (+0.0254 and +0.0122 on the 5-recording sensitivity macro).
3. **cpWER Improvement over TitaNet**: Equal affinity fusion significantly reduces downstream cpWER by **-1.02 percentage points** over TitaNet (95% CI: $[-2.07\%, -0.10\%]$ under $B=2,000$ and $[-2.11\%, -0.10\%]$ under $B=10,000$, $p < 0.05$).
4. **Baseline Fairness**: Single-encoder spectral clustering in NeMo NME-SC is mathematically and empirically invariant to monotonic offdiag_zscore_affinity normalization ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$), confirming baseline comparisons were strictly unconfounded.

---

## 5. Prohibited and Retired Claims (Explicitly Barred)

1. **NO Universal Generalization**: Do NOT claim that complementarity generalizes to unseen acoustic domains; evaluations are restricted to 8 BWC sessions.
2. **NO Gate Superiority Claim**: Do NOT claim that the reliability gate is definitively superior to equal fusion downstream; the difference ($\Delta_{\text{DER}} = -0.25\%$, 95% CI: $[-0.76\%, 0.00\%]$) touches zero and is not statistically significant.
3. **NO Claim of Fusion Outperforming ResNet on cpWER**: Do NOT claim that equal fusion beats both encoders on transcription; ResNet alone achieves lower cpWER (36.44% vs 37.05%).
4. **NO Claim of Solving VAD or Overlap**: Do NOT claim that representation fusion overcomes the 37.74% VAD overhead. With upstream VAD outputs held fixed, affinity fusion primarily affects speaker discrimination and cluster assignment; it cannot recover speech omitted by VAD or eliminate VAD false alarms.
5. **NO Claim of Solving Speaker Count Estimation**: All models collapse to 1 speaker on `R07_bwc_cheap_shot`. Representation fusion does not solve speaker-count collapse.

---

## 6. Freeze Status and Stage 5B Handoff

- **Freeze Status**: ACTIVE AND IMMUTABLE.
- **Stage 5A, 5A.1, and Final Audit Directories**: Locked; no modifications allowed.
- **Next Permitted Action**: Awaiting user instruction to plan Stage 5B (Upstream VAD recalibration, overlap speech detection, and speaker-count estimation).
