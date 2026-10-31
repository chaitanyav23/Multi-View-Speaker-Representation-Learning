# Stage 5A.1 Formal Freeze Record: Pretrained Multi-View Speaker Representation Fusion

**Date of Freeze**: October 31, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Status**: APPROVED — CONDITIONAL PASS (Freeze with Qualifications)  
**Git Commit**: `16d80ec3e98da93d9a8692df6dd9661723afe0c1`  
**Target Repository**: [`Multi-View-Speaker-Representation-Learning`](https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning)

---

## 1. Freeze Scope and Purpose

This document establishes the authoritative, immutable freeze of Stage 5A.1 experimental artifacts, source scripts, empirical metrics, and validated scientific claims. 

Following complete mathematical reconciliation, leakage auditing, and non-parametric bootstrap verification, the representation-fusion findings of Stage 5A.1 are frozen to provide a permanent, unalterable baseline before any Stage 5B architectural, VAD, or overlap interventions begin.

---

## 2. Frozen Artifact Inventory and Cryptographic Signatures

The following core files in `research_stages/stage5a_1/` are formally frozen:

| File Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `scripts/run_stage5a_1_controlled_eval.py` | `5ea3260c93f65ef9b9b4df4b7c96d345447048ff53acf4fb1487924f44bef9b9` | Definitive controlled evaluation pipeline |
| `scripts/audit_gate_features.py` | `9f50cbf0281fc7b1e3c656b1ab9ee566ac273ffc70f8246efdde069cf1be114d` | Feature distribution & SNR audit script |
| `scripts/audit_normalization.py` | `8eec7f737f340459320c51488ee9e454f5e9700da3bf41bbb98c0142aa42f0c8` | Off-diagonal normalization audit script |
| `verification/stage5a_1_verification_summary.csv` | `7ee148e5c12e1cc4c7c04d99a934d7f7b051e9bc72879f3c6bdd8a4df55b3721` | Summary verification performance metrics |
| `verification/stage5a_1_verification_per_recording.csv` | `cfead6a0e6fb0c7877e2a8cc78c4f41ed1e70e87cd16a0e709c67e1baf6be193` | Disaggregated per-recording verification results |
| `verification/stage5a_1_error_complementarity.csv` | `9886fd483e107bb6cb252e59b62971f074e87ca41644f488bbf42fa5b3bbe3f4` | Pairwise error complementarity counts |
| `downstream_metrics/stage5a_1_summary_diarization.csv` | `4b6c52a04a9e8b9826d766eae02fd07bc31e6a8775eed91fb1023e0e8edfcc98` | Summary downstream diarization metrics |
| `downstream_metrics/stage5a_1_per_recording_diarization.csv` | `b1ac5ead21668807d4b61df183e6aac0b8e8ec16d6603ba1fced873bd15a1924` | Per-recording diarization results |
| `bootstrap/stage5a_1_paired_bootstrap_differences.csv` | `d4b870048e61ace2302f0f032679791e1560b874f4b71bf8131de8a38482b4fb` | Paired bootstrap differences and 95% CIs |

---

## 3. Frozen Specifications and Protocol Definitions

### 3.1 Pretrained Encoders
1. **View 1 (Acoustic Noise Robust)**: NVIDIA TitaNet-Large
   - Checkpoint: `nvidia/speakerverification_titanet_large` (NeMo Toolkit v2.7.3)
   - Input Audio: DeepFilterNet3-enhanced (`DFN`) 16 kHz mono
   - Embedding Dimension: 192 (unit L2-normalized)
2. **View 2 (Conversational / Phonetic Robust)**: WeSpeaker ResNet293-LM
   - Checkpoint: Pretrained on VoxCeleb1+2 (WeSpeaker repository)
   - Input Audio: Original 16 kHz mono (`16k_mono`)
   - Embedding Dimension: 256 (unit L2-normalized; raw Euclidean norm logged prior to normalization)

### 3.2 Evaluation Benchmark
- **Dataset**: 8 noisy conversational body-worn camera (BWC) audio recordings (`R01` through `R08`).
- **Turn Filtering**: 250 verified clean, non-overlapping turns (meeting strict $\pm 50\text{ ms}$ boundary margin and speaker matching).
- **Verification Pairs**: Exhaustive intra-recording pairs ($N_r(N_r-1)/2$) summing to **9,489 pairwise trials** (2,766 positive, 6,723 negative).
- **Cross-Validation**: Leave-One-Recording-Out (LORO) with zero feature leakage.

### 3.3 Frozen Fusion Equations
1. **Affinity Normalization**:
   $$\tilde{S}_{ij} = \frac{S_{ij} - \mu_{\text{offdiag}}}{\sigma_{\text{offdiag}}} \quad (i \ne j), \quad \tilde{S}_{ii} = 1.0$$
2. **Equal-Weight Affinity Fusion (Condition J)**:
   $$S_{\text{equal}} = 0.5 \cdot \tilde{S}_{\text{tita}} + 0.5 \cdot \tilde{S}_{\text{res}}$$
3. **Linear Clipped Reliability Gate (Condition K)**:
   $$\Delta_{\text{pred}} = \mathbf{w}^T \mathbf{f} + b, \quad \mathbf{f} = [\text{snr}, \text{vad\_conf}, \|\mathbf{e}_{\text{tita}}\|_2, \|\mathbf{e}_{\text{res}}\|_2]^T$$
   $$g_{\text{res}} = \text{clip}(0.5 + 2.0 \cdot \Delta_{\text{pred}},\, 0.15,\, 0.85)$$
   $$S_{\text{gate}} = (1 - g_{\text{res}}) \cdot \tilde{S}_{\text{tita}} + g_{\text{res}} \cdot \tilde{S}_{\text{res}}$$

---

## 4. Final Accepted Metrics

The following metrics are certified as the official ground truth of Stage 5A.1:

### 4.1 Verification Performance
| Method / Condition | 6-Rec Macro ROC-AUC | 6-Rec Macro HTER | 5-Rec Macro ROC-AUC | 5-Rec Macro HTER | Pooled OOF ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TitaNet-DFN Alone (H) | 0.8238 | 33.80% | 0.8172 | 34.13% | 0.7710 |
| ResNet293-Mono Alone (I) | 0.8230 | 32.40% | 0.8304 | 33.16% | 0.7836 |
| Equal-Weight Fusion (J) | **0.8450** | 32.69% | **0.8426** | 32.79% | **0.7955** |
| Reliability Gate (K) | **0.8458** | **31.90%** | **0.8436** | **32.56%** | **0.7980** |

### 4.2 Downstream Diarization Performance (8 Recordings, collar=0.0s, overlap included)
| Method / Condition | Macro DER | Macro Confusion | Macro Miss | Macro FA | Macro cpWER | Macro SACC | Single-Spk Collapse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TitaNet-DFN Alone (H) | 46.22% | 8.48% | 24.41% | 13.33% | 38.07% | 49.42% | 1 (R07) |
| ResNet293-Mono Alone (I) | 45.98% | 8.24% | 24.41% | 13.33% | **36.44%** | 48.94% | 1 (R07) |
| Equal-Weight Fusion (J) | 46.17% | 8.43% | 24.41% | 13.33% | 37.05% | **50.25%** | 1 (R07) |
| Reliability Gate (K) | **45.92%** | **8.18%** | 24.41% | 13.33% | 37.74% | 48.99% | 1 (R07) |

---

## 5. Validated Scientific Findings (Retained for Publication)

1. **Substantial Encoder Error Complementarity**: Pretrained TitaNet-Large and ResNet293-LM make disjoint errors on **58.03%** of all failing pairwise trials ($1,035$ TitaNet-only errors vs $1,042$ ResNet-only errors across 9,489 pairs).
2. **Equal Affinity Fusion Improves Verification Accuracy**: Combining normalized affinity matrices without training improves verification ROC-AUC by **+0.0212** over TitaNet and **+0.0220** over ResNet on the 6-recording macro average (+0.0254 and +0.0122 on the 5-recording macro).
3. **Significant Attribution Improvement Over TitaNet**: Equal fusion achieves a statistically significant **-1.02% reduction in cpWER** over TitaNet (95% CI: $[-2.07\%, -0.10\%]$, $p < 0.05$).
4. **Fairness of Raw Baseline Comparison**: Single-encoder spectral clustering in NeMo NME-SC is invariant to monotonic affine normalization ($\Delta = 0.0000$), confirming that Stage 5A's baseline comparisons were strictly unconfounded.

---

## 6. Prohibited and Retired Claims (Explicitly Excluded)

The following claims are strictly unsupported by the evidence and must NOT be made in research papers, technical reports, or future documentation:
1. **NO Universal Generalization**: Do not claim that the observed complementarity generalizes universally across acoustic domains; the evaluation set comprises 8 localized BWC recordings.
2. **NO Definitive Superiority of the Gate**: Do not claim that the reliability gate definitively outperforms equal fusion; the downstream difference ($\Delta_{\text{DER}} = -0.25\%$, $\Delta_{\text{cpWER}} = +0.69\%$) is not statistically significant.
3. **NO "Training-Free" Gate Label**: The reliability gate requires LORO cross-validation regression training. Only Equal Fusion (Condition J) is parameter-free and training-free.
4. **NO End-to-End Equivalence**: Do not claim that pairwise verification improvements (+2.2% ROC-AUC) imply equivalent downstream diarization improvements; DER is overwhelmingly dominated by upstream VAD false alarms and missed speech ($37.74\%$).
5. **NO Claim of Solving Speaker Count Collapse**: Do not claim that representation fusion resolves speaker-count estimation; all models collapse to 1 speaker on `R07_bwc_cheap_shot`.

---

## 7. Execution Boundary and Handoff to Stage 5B

- **Freeze Status**: Formal Freeze in Effect.
- **Stage 5A / 5A.1 Files**: Locked; no further edits permitted.
- **Next Permitted Action**: Awaiting user approval to plan Stage 5B (Upstream VAD recalibration, overlap speech handling, and speaker-count estimation).
