# Stage 5A.1.2 Final Evidence Verification and Freeze Closure Report

**Audit Date**: November 3, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Target Subsystem**: Pretrained Multi-View Speaker Representation Fusion (TitaNet-Large DFN + WeSpeaker ResNet293-LM Mono)  
**Audit Scope**: Definitive Evidence Verification across Four Core Audit Areas  
**Audit Decision**: **CONDITIONAL PASS — Freeze Approved with Qualifications**  
**Repository State**: Git Commit `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`  
**Execution Environment**: Linux x86_64, Python 3.10.12, PyTorch 2.5.1+cu124, NeMo Toolkit 2.7.3 (`.venv_phase20`)

---

## 1. Executive Summary

This report delivers the final, evidence-backed resolution of all outstanding methodological and documentation questions identified in Stage 5A.1. Operating under strict non-destructive constraints (zero neural inference, zero embedding extraction, zero gate retraining), this closure audit independently examined source code, execution scripts, serialized metadata, runtime environments, and retained evaluation artifacts to reach definitive findings.

### 1.1 Summary of Core Resolutions
1. **Macro Verification Reconciliation**: Clarified the mathematical distinction between the **Six-Recording Eligible Macro** (including `R01_sara_sharif`'s 15 trials, ROC-AUC = 0.8571) and the **Five-Recording Balanced-Session Sensitivity Macro** (excluding R01 due to trial sparsity). Under both formulations, multi-view fusion outperforms both single encoders. Because raw pair-level held-out prediction tables were not serialized to disk during the original run, prediction-level reproduction is classified as **AGGREGATE CONSISTENCY ONLY**.
2. **Exact Reliability-Gate Feature Vector**: Resolved the discrepancy between earlier 3-feature and later 4-feature documentation. Source code inspection across extraction, training, and inference proves conclusively that the executed gate model uses strictly the **Three-Feature Vector**:
   $$\mathbf{x}_{rel} = [\text{resnet293\_l2\_norm}, \text{dfn\_noise\_reduction\_db}, \text{vad\_conf\_mean}]$$
   WADA SNR and TitaNet embedding norm were never passed to the active gate model.
3. **NME-SC Diagonal Invariance Proof**: Audited NeMo v2.7.3 (`offline_clustering.py`). Proved mathematically and verified empirically that while `torch.argsort` includes the diagonal, the sigmoid step in [`offdiag_zscore_affinity`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py#L20-L45) guarantees all off-diagonals are strictly $< 1.0$, and the subsequent diagonal reset to $1.0$ guarantees the diagonal element is strictly maximal (rank 0 descending). Thus, binary neighborhood masks and downstream diarization outputs are **100% invariant** ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$).
4. **Bootstrap Reconciliation and Statistical Interpretation**: Reconciled the $B = 2,000$ vs $B = 10,000$ replicate notation. The retained CSV was generated with $B = 2,000$ (95% CI: $[-2.07\%, -0.10\%]$), while an independent $B = 10,000$ run produces $[-2.11\%, -0.10\%]$. The unrounded upper bound for Gate vs Equal Fusion DER touches $0.0000$ exactly because 7 of 8 recordings have zero difference. All statistical claims are strictly qualified to the 8 recording clusters.

---

## 2. Audit A — Verification Macro and Held-Out Prediction Reproduction

### 2.1 Resolution of Macro Verification Definitions
In the 8-recording conversational BWC dataset, speaker turns were selected using **50 ms tolerance-based boundary containment, with overlap exclusion and ground-truth speaker-label matching**, yielding 250 clean turns and 9,489 pairwise trials.

Evaluating ROC-AUC requires both positive ($y=1$) and negative ($y=0$) pairwise trials within a recording:
- `R02_dennis_skinner`: 2 clean turns $\rightarrow$ 1 positive pair, 0 negative pairs. ROC-AUC is mathematically undefined ($0/0$ denominator). **Excluded from all verification macros**.
- `R04_piers_robinson`: 3 clean turns $\rightarrow$ 0 positive pairs, 3 negative pairs. ROC-AUC is mathematically undefined. **Excluded from all verification macros**.
- `R01_sara_sharif`: 6 clean turns $\rightarrow$ 1 positive pair, 14 negative pairs. ROC-AUC is mathematically well-defined ($0.8571$), but has minimal positive support (a single trial).

To eliminate ambiguity, both macro metrics are officially adopted and distinguished:
- **Six-Recording Eligible Macro**: Averages all recordings with mathematically defined ROC-AUC (`R01`, `R03`, `R05`, `R06`, `R07`, `R08`).
- **Five-Recording Balanced-Session Sensitivity Macro**: Averages multi-speaker sessions with balanced trial density ($N_{\text{pairs}} \ge 36$: `R03`, `R05`, `R06`, `R07`, `R08`), strictly excluding R01.

#### Table 2.1: Reconciled Verification Macro Performance
| Condition Code | Method Description | 6-Rec Eligible Macro ROC-AUC | 6-Rec Eligible Macro HTER | 5-Rec Sensitivity Macro ROC-AUC | 5-Rec Sensitivity Macro HTER |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Condition A** | Raw TitaNet-Large (DFN) | 0.8238 | 33.80% | 0.8172 | 34.13% |
| **Condition B** | Raw ResNet293-LM (Mono) | 0.8230 | 32.40% | 0.8304 | 33.16% |
| **Condition C** | Raw Equal Fusion (Stage 5A) | 0.8450 | 32.69% | 0.8426 | 32.79% |
| **Condition H** | Norm TitaNet Alone (Control) | 0.8238 | 33.80% | 0.8172 | 34.13% |
| **Condition I** | Norm ResNet Alone (Control) | 0.8230 | 32.40% | 0.8304 | 33.16% |
| **Condition J** | Norm Equal Fusion (Control) | **0.8450** | 32.69% | **0.8426** | 32.79% |
| **Condition K** | Norm Linear Reliability Gate | **0.8458** | **31.90%** | **0.8436** | **32.56%** |
| **Condition K_sig**| Norm Sigmoid Gate (Ablation) | 0.8454 | 32.69% | 0.8431 | 32.80% |

*Key Takeaway*: Under **both** macro definitions, equal affinity fusion and reliability gating improve ROC-AUC over both single encoders.

### 2.2 Independent Reproduction and Leakage Status
A recursive search across all directories in `research_stages/` confirmed that raw pair-level held-out prediction tables (e.g. trial-by-trial scores and predictions) were maintained in memory during the execution of [`run_stage5a_1_controlled_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py) and summarized into `stage5a_1_verification_per_recording.csv`, rather than serialized to disk.

- **Status**: Formally classified as **AGGREGATE CONSISTENCY ONLY — raw held-out prediction reproduction unavailable**.
- **Leakage Verification**: Inspection of lines 270–340 and 550–600 in `run_stage5a_1_controlled_eval.py` confirms that in each LORO fold, the held-out recording was strictly omitted from threshold selection and gate model training. Zero test-fold feature leakage occurred.

### 2.3 Complementarity Arithmetic Reproduction
The retained summary table [`stage5a_1_error_complementarity.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/verification/stage5a_1_error_complementarity.csv) was verified via independent arithmetic recalculation:
- Total pairwise trials: $9,489$
- Both encoders correct: $5,910$ ($62.28\%$)
- Shared errors ($E_T \cap E_R$): $1,502$ ($15.83\%$)
- TitaNet-only errors ($E_T \setminus E_R$): $1,035$ ($10.91\%$)
- ResNet-only errors ($E_R \setminus E_T$): $1,042$ ($10.98\%$)
- Total disjoint error pairs ($|E_T \triangle E_R|$): $1,035 + 1,042 = \mathbf{2,077}$
- Total unique error pairs ($|E_T \cup E_R|$): $2,077 + 1,502 = \mathbf{3,579}$

$$\text{Complementary Error Fraction} = \frac{|E_T \triangle E_R|}{|E_T \cup E_R|} = \frac{2077}{3579} = \mathbf{58.03\%}$$
$$\text{Error Jaccard Similarity} = \frac{|E_T \cap E_R|}{|E_T \cup E_R|} = \frac{1502}{3579} = \mathbf{0.4197}$$

The arithmetic matches the reported metrics exactly.

---

## 3. Audit B — Exact Reliability-Gate Feature Vector

### 3.1 Source-Code Audit and Discrepancy Resolution
Prior documentation contained a discrepancy between a 3-feature description and a 4-feature description citing WADA SNR and TitaNet embedding norm.

An audit tracing extraction, training, and inference across [`run_stage5a_1_controlled_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py) and [`fusion_models.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py) establishes that the active, executed gate model strictly employs the **Three-Feature Vector**:

$$\mathbf{x}_{rel} = \begin{bmatrix} \text{resnet293\_l2\_norm} \\ \text{dfn\_noise\_reduction\_db} \\ \text{vad\_conf\_mean} \end{bmatrix}$$

#### Evidence Points:
1. **Feature Vector Definition**: [`run_stage5a_1_controlled_eval.py:307`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py#L307):
   ```python
   gate_features = ["resnet293_l2_norm", "dfn_noise_reduction_db", "vad_conf_mean"]
   ```
2. **Training Assembly**: Lines 298–300 explicitly populate only these three keys in `loto_subseg_items`.
3. **Inference Assembly**: Lines 376–382 construct `meta_gate_items` with identical 3 keys.
4. **Candidate Feature Resolution**:
   - `wada_snr`: Never passed to the gate. It was evaluated in early exploratory scripts but omitted from the active pipeline.
   - `titanet_embedding_norm`: TitaNet embeddings are unit L2-normalized internally by NeMo ($1.0 \pm 10^{-6}$), giving zero variance. It was intentionally excluded from the gate model.

The complete feature trace is recorded in [`gate_feature_trace.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/gate_feature_trace.csv) and [`GATE_FEATURE_VECTOR_VERIFICATION.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/GATE_FEATURE_VECTOR_VERIFICATION.md).

---

## 4. Audit C — NME-SC Diagonal-Handling Proof

### 4.1 NeMo Clustering Implementation Inspection
In NeMo Toolkit v2.7.3 (`offline_clustering.py:309-350`), the function `getKneighborsConnections` binarizes the affinity matrix:
```python
sorted_matrix = torch.argsort(affinity_mat, dim=1, descending=True)[:, :p_value]
```
The diagonal elements are **included** in `torch.argsort`.

### 4.2 Complete Mathematical Invariance Proof
The normalization function in [`fusion_models.py:20-33`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py#L20-L33) executes:
```python
def offdiag_zscore_affinity(A: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
    off = A[mask]
    mu = torch.mean(off)
    sd = torch.clamp(torch.std(off), min=1e-6)
    Z = (A - mu) / sd
    Z_sig = torch.sigmoid(Z)
    Z_sig.fill_diagonal_(1.0)
    return ScalerMinMax(Z_sig)
```

1. **Off-Diagonal Order Preservation**: Because $\sigma > 0$, standardizing $Z_{ij} = \frac{A_{ij} - \mu}{\sigma}$ is strictly monotonic. Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ is strictly monotonically increasing. Hence:
   $$A_{ij} > A_{ik} \iff (Z_{\text{sig}})_{ij} > (Z_{\text{sig}})_{ik} \quad \forall j, k \ne i$$
2. **Diagonal Global Maximality**: The range of sigmoid on finite inputs is strictly $(0, 1)$. Therefore, every off-diagonal entry satisfies:
   $$(Z_{\text{sig}})_{ij} < 1.0 \quad \forall i \ne j$$
   Resetting the diagonal via `Z_sig.fill_diagonal_(1.0)` guarantees:
   $$(Z_{\text{sig}})_{ii} = 1.0 > (Z_{\text{sig}})_{ij} \quad \forall j \ne i$$
   The diagonal element is the strict global maximum of row $i$.
3. **ScalerMinMax Preserves Maximality**: `ScalerMinMax` maps $[v_{\min}, 1.0] \to [0, 1.0]$. The diagonal maps to $1.0$, all off-diagonals map into $[0, 1.0)$, and off-diagonal pairwise orderings are strictly maintained.
4. **Neighborhood Mask Identity**: Because the diagonal is maximal in both raw and normalized matrices, column index $i$ is placed at **rank 0 descending** in both matrices:
   $$\text{sorted\_matrix}_{i, 0} = i$$
   The remaining $p - 1$ indices select the top off-diagonal elements in identical descending order.
5. **Graph and Diarization Identity**: Binary masks $\mathcal{M}$, symmetrized graph matrices $\frac{\mathcal{M} + \mathcal{M}^T}{2}$, Laplacian eigenvectors, and final cluster assignments are **strictly identical**.

### 4.3 Empirical Verification
Script [`verify_nmesc_diagonal_invariance.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/verify_nmesc_diagonal_invariance.py) confirmed:
- Synthetic & real affinity matrices: Binary masks match 100%.
- Real recordings (`R01` through `R08`): Condition H $\equiv$ Condition A, Condition I $\equiv$ Condition B ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$).

**Verdict**: Invariance is **MATHEMATICALLY PROVEN AND EMPIRICALLY VERIFIED**.

---

## 5. Audit D — Paired Recording-Cluster Bootstrap Reproduction

### 5.1 Reconciliation of Replicate Counts
- In `run_stage5a_1_controlled_eval.py:755-760`, the execution parameter was `n_boot = 2000`, producing `[-0.0207, -0.0010]` in [`stage5a_1_paired_bootstrap_differences.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/bootstrap/stage5a_1_paired_bootstrap_differences.csv).
- The prior audit text cited $B = 10,000$ due to a narrative documentation typo.
- Running an independent fresh bootstrap with $B = 10,000$ (`seed=42`) via [`reproduce_paired_cluster_bootstrap.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/reproduce_paired_cluster_bootstrap.py) yields:
  - Point diff: **-1.02%**
  - 95% Bootstrap CI: **[-2.11%, -0.10%]** (strictly excludes zero).

### 5.2 Mathematical Boundary Analysis of Gate-vs-Equal-Fusion DER
The reported 95% CI for Gate vs Equal Fusion on macro DER was $[-0.76\%, 0.00\%]$.
An inspection of unrounded recording-level differences reveals:
- In **7 of 8 recordings**, Gate and Equal Fusion produce identical DER ($\Delta = 0.0000$).
- In `R06_rich_karen`, Gate lowers DER by $0.0202$.
- In a cluster bootstrap with replacement, the probability of drawing a resample that omits `R06` entirely is $(7/8)^8 \approx 34.36\%$.
- In every such resample, the sample difference is identically zero.
- Because $34.36\% > 2.5\%$, the 97.5th percentile is **identically 0.000000**.
- The interval touches zero as an attainable boundary value, confirming that the gate provides **no statistically established downstream advantage** over training-free equal fusion.

### 5.3 Rigorous Statistical Caveats
1. **Cluster Limitations**: All bootstrap evaluations resample from **8 recording clusters**.
   > "Equal affinity fusion showed a nominal exploratory cpWER improvement over TitaNet under the paired recording-cluster bootstrap. The result is based on eight recording clusters and should not be interpreted as definitive evidence of generalization."
2. **Multiple Comparisons**: Multiple metrics and contrasts were evaluated without FWER/FDR corrections.
3. **ResNet Superiority on cpWER**: Equal fusion (37.05% cpWER) improves over TitaNet (38.07%), but does **NOT** outperform ResNet293-LM alone (36.44%). The difference vs ResNet (+0.61%, 95% CI: $[-3.73\%, +6.27\%]$) is not statistically significant.

---

## 6. Required Wording Reconciliation Applied

In accordance with Section 8 of the audit instructions, the following formal phrasing standards are enforced across all documentation:
1. **Turn Selection**: "50 ms tolerance-based boundary containment, with overlap exclusion and ground-truth speaker-label matching."
2. **VAD Limitations**: "With upstream VAD outputs held fixed, affinity fusion primarily affects speaker discrimination and cluster assignment. It cannot recover speech omitted by VAD or independently eliminate VAD false alarms."
3. **VAD Error Contribution**: "The evaluated pipeline has a substantial VAD-associated error contribution under its fixed speech-activity hypotheses."
4. **Gate Result**: "The reliability-aware gate produces a marginal verification improvement, but no statistically established downstream advantage over training-free equal affinity fusion."
5. **Fusion Distinction**: Verification improves over both encoders; cpWER improves over TitaNet; cpWER does not improve over ResNet; downstream DER does not improve over the gate.

---

## 7. Final Freeze Decision

### Verdict: CONDITIONAL PASS — Freeze Approved with Qualifications

All four audit areas are fully resolved with concrete code-level and mathematical evidence. The representation-fusion findings of Stage 5A.1 are formally frozen under the qualifications detailed in [`STAGE5A_1_2_FINAL_FREEZE_RECORD.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/STAGE5A_1_2_FINAL_FREEZE_RECORD.md).
