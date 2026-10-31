# Stage 5A.1 Final Independent Audit Report: Representation-Fusion Reconciliation and Freeze Assessment

**Audit Date**: October 31, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Target Subsystem**: Stage 5A.1 Affinity Normalization, Fusion Fairness, and Evaluation Protocol  
**Repository**: [`Multi-View-Speaker-Representation-Learning`](https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning)  
**Git Commit**: `16d80ec3e98da93d9a8692df6dd9661723afe0c1`  
**Execution Environment**: Linux x86_64, Python 3.10.12, PyTorch 2.5.1+cu124, NeMo 2.7.3 (`.venv_phase20`)

---

## 1. Executive Summary

This report delivers the final independent code-level, result-level, and statistical audit of **Stage 5A.1 (Affinity Normalization, Fusion Fairness, and Controlled Evaluation)**. 

The audit was conducted to resolve methodological questions raised regarding Stage 5A: specifically, whether reported multi-view fusion gains were genuine or an artifact of comparing raw single-encoder affinities against z-score normalized fused affinities, whether data leakage existed in cross-validation routines, whether turn/pair counts were consistent across stages, and whether the proposed reliability gate provides statistically verifiable gains.

### Core Audit Verdict: CONDITIONAL PASS — Freeze with Qualifications
The evidence confirms that:
1. **The Normalization Confound is Non-Existent Downstream**: Single-encoder NeMo NME-SC spectral clustering is mathematically and numerically **100% invariant** to monotonic off-diagonal affine transformations ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$ across all recordings). The Stage 5A baseline comparisons were methodologically sound.
2. **Error Complementarity is Genuine**: Pretrained NVIDIA TitaNet-Large (on DeepFilterNet audio) and WeSpeaker ResNet293-LM (on original mono audio) exhibit **58.03% complementary error trials** (Error Jaccard similarity = 0.4197 across 9,489 pairwise trials).
3. **Equal Fusion Outperforms Both Single Encoders**: Equal-weight affinity fusion improves speaker verification macro ROC-AUC over both single encoders (+0.0212 vs TitaNet, +0.0220 vs ResNet on the 6-recording macro; +0.0254 vs TitaNet, +0.0122 vs ResNet on the 5-recording macro) and achieves a statistically significant **-1.02% reduction in cpWER** over TitaNet (95% CI: $[-2.07\%, -0.10\%]$, $p < 0.05$).
4. **Reliability Gating Provides Marginal, Non-Significant Verification Gains**: The LORO-calibrated linear gate yields a marginal verification ROC-AUC gain (+0.0008 over equal fusion) and slight confusion reduction (8.18% vs 8.43%), but downstream diarization differences between gate and equal fusion are statistically indistinguishable from zero.
5. **Downstream Error Ceiling**: Macro DER is dominated by upstream VAD false alarms (13.33%) and missed speech (24.41%), which together constitute **37.74% out of 46.17% total DER**. Affinity fusion operates strictly on speaker confusion ($\sim 8.2\%$).

Stage 5A.1 is approved for freeze under formal qualifications restricting all claims to the local evaluation domain.

---

## 2. Audit Scope and Methodology

The audit evaluated all components of the representation learning and affinity fusion pipeline across four levels:

```mermaid
flowchart TD
    subgraph L1["Level 1: Data Integrity & Ground Truth"]
        A["Pre-Audit Inventory & Hashes"] --> B["Turn & Subsegment Boundaries"]
        B --> C["Pair Generation (N choose 2)"]
    end
    subgraph L2["Level 2: Mathematical & Code Audit"]
        D["Normalization Functions"] --> E["Gate Model & Linear Mapping"]
        E --> F["Leakage Checks (LORO / LOTO)"]
    end
    subgraph L3["Level 3: Metric Recomputation"]
        G["Verification ROC-AUC & HTER"] --> H["Downstream DER, JER, cpWER"]
        H --> I["Error Complementarity Matrix"]
    end
    subgraph L4["Level 4: Statistical Verification"]
        J["Paired Bootstrap (B=10,000)"] --> K["Confidence Interval Verification"]
        K --> L["Scientific Claim Audit"]
    end
    L1 --> L2 --> L3 --> L4
```

### Methodological Principles
- **Zero-Modification Rule**: No pre-existing files under `research_stages/` were modified.
- **Zero-Inference Footprint**: No neural models were retrained; cached embeddings from Stage 3 were consumed directly.
- **Code-Level Verification**: All algorithms were traced to exact line numbers in repository source code.
- **Statistical Rigor**: All hypothesis tests evaluated with paired non-parametric bootstrapping ($B = 10,000$) clustered at the recording level.

---

## 3. Source Files and Artifacts Inspected

### Source Code Audited
- [`research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py): Primary controlled evaluation pipeline (lines 145–540).
- [`research_stages/stage5a_1/scripts/audit_gate_features.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/audit_gate_features.py): Feature distribution and alignment analysis.
- [`research_stages/stage5a_1/scripts/test_normalization_invariance.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/test_normalization_invariance.py): Normalization rank invariance verification.
- [`research_stages/stage5a/scripts/run_stage5a_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/run_stage5a_eval.py): Stage 5A baseline evaluation pipeline.
- [`research_stages/stage5a/scripts/fusion_models.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py): Affinity normalization and reliability gate implementation (lines 20–120).
- [`research_stages/stage3_dual_encoder/run_stage3_extraction.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage3_dual_encoder/run_stage3_extraction.py): Embedding extraction and raw norm logging (lines 550–575).

### Evaluation Artifacts Inspected
- Verification Summaries: `stage5a_1_verification_summary.csv`, `stage5a_1_verification_per_recording.csv`.
- Downstream Diarization: `stage5a_1_summary_diarization.csv`, `stage5a_1_per_recording_diarization.csv`.
- Bootstrap Statistics: `stage5a_1_paired_bootstrap_differences.csv`.
- Gate Feature Audit: `stage5a_1/gate_audit/gate_audit_results.json`.
- Complementarity Matrix: `stage5a_1_error_complementarity.csv`.

---

## 4. Dataset and Pair-Count Reconciliation

A central task of this audit was reconciling the turn and pair counts between Stage 5A (254 turns, 9,817 pairs) and Stage 5A.1 (250 turns, 9,489 pairs).

### Root-Cause Analysis of Turn Discrepancy
Automated segment-level diffing executed via [`audit_turn_discrepancy.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_final_audit/scripts/audit_turn_discrepancy.py) revealed that Stage 5A.1 added three strict quality filters in `build_hybrid_turn_records`:
1. **Strict Margin Tolerance**: `(st >= gst - 0.05) and (en <= gen + 0.05)`.
2. **Explicit Overlap Exclusion**: `not m["gt_is_overlap"]`.
3. **Speaker Identity Match**: `m["gt_label"] == g["speaker"]`.

Four turns accepted by Stage 5A were correctly rejected by Stage 5A.1:
1. `R03_disabled_parking`, turn segment 41 ([133.0s, 134.0s], Speaker 1): Subsegment 455 ends at 134.113s, hanging over by 63 ms beyond the 50 ms tolerance threshold ($134.113 > 134.050$).
2. `R06_rich_karen`, turn segment 66 ([288.0s, 289.0s], Speaker 7): Subsegment 526 ends at 289.077s, hanging over by 27 ms beyond tolerance ($289.077 > 289.050$).
3. `R06_rich_karen`, turn segment 85 ([334.0s, 335.0s], Speaker 4): Subsegment 669 has ground-truth overlap metadata `gt_is_overlap = True`. Stage 5A erroneously included it; Stage 5A.1 cleanly dropped it.
4. `R07_bwc_cheap_shot`, turn segment 44 ([135.0s, 136.0s], Speaker 1): Subsegment 373 ends at 136.054s, hanging over by 4 ms beyond tolerance ($136.054 > 136.050$).

### Pair Count Derivation
Within each recording $r$, pairwise trials are evaluated exhaustively across clean turns:
$$P_r = \frac{N_r (N_r - 1)}{2}$$

The 4 dropped turns reduced pair counts across 3 recordings:
- R03: 51 turns (1,275 pairs) $\rightarrow$ 50 turns (1,225 pairs), $\Delta = -50$ pairs.
- R06: 124 turns (7,626 pairs) $\rightarrow$ 122 turns (7,381 pairs), $\Delta = -245$ pairs.
- R07: 34 turns (561 pairs) $\rightarrow$ 33 turns (528 pairs), $\Delta = -33$ pairs.
- Total reduction: $50 + 245 + 33 = \mathbf{328\text{ pairs}}$ ($9,817 - 328 = 9,489$).

**Audit Verdict**: Stage 5A.1's filtering is scientifically superior. The 250-turn / 9,489-pair benchmark is adopted as the definitive frozen standard.

---

## 5. Verification Eligibility Correction

A documentation discrepancy occurred in Stage 5A.1 regarding whether 5 or 6 recordings were eligible for macro averaging:

```mermaid
flowchart LR
    A["8 Audio Recordings"] --> B["R02 (1 Pos, 0 Neg) - Ineligible"]
    A --> C["R04 (0 Pos, 3 Neg) - Ineligible"]
    A --> D["R01 (1 Pos, 14 Neg) - Eligible"]
    A --> E["5 Balanced Sessions (R03, R05, R06, R07, R08) - Eligible"]
    D --> F["6-Recording Macro (R01+R03+R05+R06+R07+R08)"]
    E --> G["5-Recording Macro (R03+R05+R06+R07+R08)"]
```

### Clarification and Reconciliation
- In `run_stage5a_1_controlled_eval.py`, a recording is flagged `eligible_for_eer = True` if $N_{\text{pos}} > 0$ and $N_{\text{neg}} > 0$.
- `R01_sara_sharif` has 1 positive pair and 14 negative pairs. ROC-AUC is mathematically well-defined ($0.8571$).
- The code included R01 in the macro average (`eligible_recordings_count = 6`), yielding:
  - TitaNet (A/H): ROC-AUC = **0.8238**
  - ResNet (B/I): ROC-AUC = **0.8230**
  - Equal Fusion (C/J): ROC-AUC = **0.8450**
  - Reliability Gate (K): ROC-AUC = **0.8458**
- However, narrative report sections described R01 as excluded and labeled this exact row as "Macro Average (5 Eligible)".
- Recomputing the macro average strictly excluding R01 gives:
  - TitaNet (A/H): ROC-AUC = **0.8172**
  - ResNet (B/I): ROC-AUC = **0.8304**
  - Equal Fusion (C/J): ROC-AUC = **0.8426**
  - Reliability Gate (K): ROC-AUC = **0.8436**

**Audit Verdict**: Under both definitions, the core conclusions remain completely unchanged. Equal fusion and reliability gating outperform both single encoders in both 6-recording and 5-recording macro formulations.

---

## 6. Normalization Implementation Audit

The Stage 5A.1 normalization function was audited in [`research_stages/stage5a/scripts/fusion_models.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py#L20-L45):

```python
def offdiag_zscore_affinity(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = S.shape[0]
    if n <= 1:
        return S.copy()
    mask = ~np.eye(n, dtype=bool)
    offdiag = S[mask]
    mu = float(np.mean(offdiag))
    sigma = float(np.std(offdiag))
    if sigma < eps:
        S_norm = S - mu
    else:
        S_norm = (S - mu) / sigma
    np.fill_diagonal(S_norm, 1.0)
    return S_norm
```

### Mathematical Properties Verified
1. **Zero Off-Diagonal Mean**: $\frac{1}{N(N-1)} \sum_{i \ne j} \tilde{S}_{ij} = 0$.
2. **Unit Off-Diagonal Variance**: $\frac{1}{N(N-1)} \sum_{i \ne j} \tilde{S}_{ij}^2 = 1$ (when $\sigma \ge \epsilon$).
3. **Preserved Self-Affinity**: Diagonal entries are explicitly clamped to $1.0$.
4. **Strict Off-Diagonal Monotonicity**: For all $i \ne j$ and $k \ne l$:
   $$S_{ij} > S_{kl} \iff \tilde{S}_{ij} > \tilde{S}_{kl}$$
   Because $\sigma > 0$, the transformation is a strictly monotonically increasing affine mapping on off-diagonals.

---

## 7. Normalization Fairness and Downstream Invariance

Stage 5A.1 was specifically commissioned to investigate whether normalizing affinity matrices gave fused models an unfair advantage over raw single encoders.

### Mathematical Proof of Invariance
In the NeMo NME-SC spectral clustering implementation (`nemo.collections.asr.parts.utils.clustering_utils`):
1. The affinity matrix $S \in \mathbb{R}^{N \times N}$ is binarized row-by-row using `mask_method='binary'`.
2. For each row $i$, the top $p$-percentile threshold is determined by sorting row entries:
   $$\text{ranks}_i = \text{argsort}(S_{i, :})$$
3. The top $K_i = \lfloor p \cdot N \rfloor$ neighbors are set to $1$, and the rest to $0$.
4. Because $\tilde{S}_{ij} = \frac{S_{ij} - \mu}{\sigma}$ is a strictly increasing monotonic transformation on off-diagonals, and the diagonal $S_{ii} = 1.0$ remains maximal, the permutation that sorts row $i$ is identical:
   $$\text{argsort}(\tilde{S}_{i, :}) \equiv \text{argsort}(S_{i, :})$$
5. Consequently, the binarized graph adjacency matrix $\mathcal{A}$, the normalized Laplacian $L_{\text{sym}}$, its eigenvectors, and the final k-means cluster assignments are **strictly identical**.

### Empirical Verification
Empirical comparison of Condition H (Normalized TitaNet) vs Condition A (Raw TitaNet), and Condition I (Normalized ResNet) vs Condition B (Raw ResNet) across all 8 recordings:
- Discrepancies across all recordings, all conditions, all 15 diarization metrics: **Exactly 0**.
- Point differences: $\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$, $\Delta_{\text{Confusion}} = 0.0000$.

**Audit Verdict**: The suspected normalization confound is decisively disproven. Stage 5A's comparison of raw single-encoder affinities against normalized fused affinities was completely fair because single-encoder performance is invariant to normalization.

---

## 8. Reliability Gate Implementation Audit

The executable reliability gate was inspected in [`research_stages/stage5a/scripts/fusion_models.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py#L96-L103).

### Audited Formulation
The executable implementation employs a piecewise linear clipped mapping:
$$\Delta_{\text{pred}} = w_1 \cdot \text{snr} + w_2 \cdot \text{vad\_conf} + w_3 \cdot \|\mathbf{e}_{\text{tita}}\|_2 + w_4 \cdot \|\mathbf{e}_{\text{res}}\|_2 + b$$
$$g_{\text{res}} = \text{clip}(0.5 + 2.0 \cdot \Delta_{\text{pred}},\, 0.15,\, 0.85)$$
$$S_{\text{gate}} = (1 - g_{\text{res}}) \cdot \tilde{S}_{\text{tita}} + g_{\text{res}} \cdot \tilde{S}_{\text{res}}$$

### Resolution of Documentation Discrepancy
Early design notes described a sigmoid mapping $\sigma(2.0 \cdot \Delta_{\text{pred}})$. Stage 5A.1 explicitly compared both mappings (`condition_K_norm_reliability_gate` vs `condition_K_sigmoid_gate`):
- Linear Clipped (K): Macro ROC-AUC = 0.8458, Macro HTER = 31.90%
- Sigmoid (K_sig): Macro ROC-AUC = 0.8454, Macro HTER = 32.69%

The linear clipped mapping avoids exponential operations, converges cleanly, and is retained as the authoritative frozen gate.

---

## 9. Feature Validity and Alignment

The gate consumes four acoustic reliability indicators. An audit was conducted on their distributions:
1. `wada_snr`: Estimated via WADA algorithm. Mean = 14.82 dB, Std = 6.41 dB. Well-distributed across noisy BWC audio.
2. `resnet293_l2_norm`: Raw Euclidean norm before L2 normalization. Traced to [`run_stage3_extraction.py:560`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage3_dual_encoder/run_stage3_extraction.py#L560). Valid indicator of acoustic clarity.
3. `titanet_l2_norm`: Unit normalized ($1.0 \pm 10^{-6}$) due to NeMo's internal normalization. Provides zero variance for gating.
4. `vad_conf_mean`: Mean = 0.9542, Variance = 0.0102. In clean speech LOTO turns, VAD confidence is saturated near 1.0, limiting its dynamic utility.

**Audit Verdict**: Gating relies primarily on WADA SNR and raw ResNet293 norm. Features are valid but modest in effective dimensionality.

---

## 10. LORO/LOTO Leakage Audit

A thorough code audit of cross-validation routines confirmed:
1. **Leave-One-Recording-Out (LORO)**: In [`run_stage5a_1_controlled_eval.py:270-340`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py#L270-L340), when recording $r$ is held out:
   - The gate model is fitted exclusively on $\{r' \mid r' \ne r\}$.
   - Decision thresholds $\tau$ for EER/HTER are computed exclusively on $\{r' \mid r' \ne r\}$.
   - Zero test-recording features or labels leak into training.
2. **Leave-One-Turn-Out (LOTO)**: When computing within-recording turn-level weights, features are aggregated over non-target turns.

**Audit Verdict**: Zero data leakage detected. The cross-validation protocol is completely clean.

---

## 11. Verification Protocol Audit

Verification trials are formed by pairing clean speaker turns within each recording. The distance metric is normalized cosine similarity. 
- Thresholding: Evaluated at frozen out-of-fold $\tau_{\text{frozen}}$ calibrated on training folds.
- Operating Point: Far/Frr balanced at EER operating point.

**Audit Verdict**: Protocol is strictly compliant with international speaker recognition standards (NIST SRE / ISO/IEC 19795).

---

## 12. Complementarity Analysis

The 9,489 verification trials were partitioned based on individual encoder correctness at their LORO thresholds:

```
Total Pairs: 9,489
├── Both Encoders Correct:         5,910 (62.28%)
└── Total Unique Errors:            3,579 (37.72%)
    ├── Both Encoders Wrong:        1,502 (15.83%) [Shared Errors]
    ├── TitaNet Wrong, ResNet OK:   1,035 (10.91%) [ResNet Complementary Win]
    └── ResNet Wrong, TitaNet OK:   1,042 (10.98%) [TitaNet Complementary Win]
```

- **Complementary Error Fraction**:
  $$\frac{1035 + 1042}{3579} = \frac{2077}{3579} = \mathbf{58.03\%}$$
- **Error Jaccard Similarity**:
  $$\frac{1502}{1035 + 1042 + 1502} = \frac{1502}{3579} = \mathbf{0.4197}$$

**Audit Verdict**: Complementarity is substantial and balanced. TitaNet rescues ResNet in 1,042 trials, while ResNet rescues TitaNet in 1,035 trials.

---

## 13. Statistical Audit

Paired bootstrap hypothesis testing ($B = 10,000$ resamples clustered by recording) established:
1. **Equal Fusion vs TitaNet (J vs H)**:
   - **cpWER**: $\Delta = \mathbf{-1.02\%}$, 95% CI: $[-2.07\%, -0.10\%]$ ($p < 0.05$, strictly excludes zero).
   - Macro DER: $\Delta = -0.05\%$, 95% CI: $[-0.89\%, +0.89\%]$ (includes zero).
2. **Equal Fusion vs ResNet (J vs I)**:
   - cpWER: $\Delta = +0.61\%$, 95% CI: $[-3.70\%, +6.24\%]$ (includes zero).
   - Macro DER: $\Delta = +0.19\%$, 95% CI: $[-0.51\%, +1.04\%]$ (includes zero).
3. **Reliability Gate vs Equal Fusion (K vs J)**:
   - Macro DER: $\Delta = -0.25\%$, 95% CI: $[-0.76\%, 0.00\%]$ (borderline, includes zero).
   - cpWER: $\Delta = +0.69\%$, 95% CI: $[-0.11\%, +2.18\%]$ (includes zero).

**Audit Verdict**: Equal fusion provides a statistically significant improvement in speaker-attributed word error rate (cpWER) over TitaNet. All other downstream differences between fusion variants are within statistical uncertainty bounds.

---

## 14. Downstream Diarization Reconciliation

The macro DER across all 8 recordings decomposes as follows:
$$\text{DER} = \text{False Alarm} + \text{Missed Speech} + \text{Speaker Confusion}$$
$$\text{DER}_{\text{macro}} = 13.33\% + 24.41\% + \text{Confusion} = \mathbf{37.74\%} + \text{Confusion}$$

- Condition A (TitaNet): Confusion = 8.48% $\rightarrow$ DER = 46.22%
- Condition B (ResNet): Confusion = 8.24% $\rightarrow$ DER = 45.98%
- Condition J (Equal Fusion): Confusion = 8.43% $\rightarrow$ DER = 46.17%
- Condition K (Reliability Gate): Confusion = **8.18%** $\rightarrow$ DER = **45.92%**

**Audit Verdict**: VAD false alarms and missed speech contribute 37.74% out of ~46% DER (81.7% of total error). Representation fusion reduces speaker confusion by 0.30%, but this gain is heavily compressed at the total DER level.

---

## 15. Environment and Reproducibility Audit

- Python: 3.10.12
- PyTorch: 2.5.1+cu124
- NeMo Toolkit: 2.7.3 (verified in `.venv_phase20`)
- WeSpeaker: Pretrained ResNet293-LM (frozen checkpoint)
- DeepFilterNet: Pretrained DFN3 (frozen checkpoint)
- Determinism: Random seeds fixed (`seed=42`). All scripts re-execute with identical numerical outputs.

---

## 16. Remaining Limitations

1. **Small Sample Size**: 8 recordings (250 clean turns). While adequate for proof-of-concept verification, cross-domain generalization cannot be claimed.
2. **Single-Speaker Collapse**: On R07, all models collapse to 1 speaker due to elevated acoustic similarity.
3. **Upstream VAD Bottleneck**: 37.74% DER error ceiling cannot be breached without improving speech activity detection and overlap handling.
4. **Gate Feature Satiation**: Unit normalization in TitaNet and saturated VAD confidence limit gate feature diversity.

---

## 17. Final Decision: CONDITIONAL PASS — Freeze with Qualifications

The evaluation methodology, code implementation, and core empirical results of Stage 5A.1 are **statistically sound, reproducible, and free of leakage**. 

Stage 5A.1 is awarded a **CONDITIONAL PASS**. Findings may be formally frozen subject to the qualifications specified in Section 18.

---

## 18. Explicit Recommendations for Representation-Fusion Freeze

### Claims Approved for Research Publication
1. TitaNet-Large (DFN) and WeSpeaker ResNet293-LM (Mono) exhibit substantial error complementarity (58.03% complementary error trials, Error Jaccard = 0.4197).
2. Equal affinity fusion improves speaker verification macro ROC-AUC over both single encoders (+0.0212 vs TitaNet, +0.0220 vs ResNet).
3. Equal affinity fusion significantly reduces downstream cpWER by -1.02% over TitaNet ($p < 0.05$).
4. Single-encoder spectral clustering is mathematically invariant to monotonic off-diagonal affinity normalization.

### Claims Explicitly Retired / Prohibited
1. Do NOT claim universal complementarity across unseen domains.
2. Do NOT claim definitive superiority of the reliability gate over equal fusion.
3. Do NOT claim that the gate is training-free.
4. Do NOT claim that oracle verification improvements translate directly into proportional end-to-end DER improvements.
5. Do NOT claim that representation fusion resolves speaker-count estimation or single-speaker collapse.
