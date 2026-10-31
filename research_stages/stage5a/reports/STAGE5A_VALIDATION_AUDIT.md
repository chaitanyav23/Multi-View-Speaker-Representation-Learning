# Stage 5A: Controlled Validation of Hybrid Speaker-Affinity Fusion
## Research Safeguards and Methodological Validation Audit

**Date:** 2026-10-31  
**Audit Scope:** Verification of All Experimental Protocols, Leakage Safeguards, and Implementation Integrity for Stage 5A  
**Auditor:** Senior Research Systems Auditor & Machine Learning Engineer

---

## 1. Compliance Audit Overview

Following the extensive findings of the Stage 4.1 Statistical Validity and Leakage Audit, Stage 5A implemented mandatory research safeguards to guarantee that all reported metrics are methodologically unimpeachable, free from data leakage, and statistically sound.

Every safeguard defined in Section 7 of the Stage 5A specification was enforced via software assertions in `run_stage5a_eval.py` and pre-execution validation checks in `validate_inputs_and_caches.py`.

---

## 2. Systematic Safeguard Compliance Checklist

| Item | Requirement / Safeguard | Implementation & Verification Evidence | Audit Verdict |
| :---: | :--- | :--- | :---: |
| **1.1** | **No Destruction of Prior Stages** | Verified that `research_stages/stage1_prep/`, `stage1b_dfn/`, `stage2_baseline/`, `stage3_extraction/`, `stage4_complementarity/`, and `stage4_1/` remained untouched. `git status --short` confirms only untracked files in `research_stages/stage5a/`. | **PASSED** |
| **1.2** | **No Re-extraction of Embeddings** | Reused existing cached embeddings from Stage 3 (`research_stages/stage3_extraction/cache/`). Zero neural network forward passes or embedding extractions were rerun. | **PASSED** |
| **2.1** | **Strict Cache Compatibility** | `validate_inputs_and_caches.py` verified identical temporal boundaries ($< 10^{-4}$s discrepancy), identical multiscale scales (1.5s, 1.25s, 1.0s, 0.75s, 0.5s), matching `subseg_idx`, 192-D (TitaNet) and 256-D (ResNet293) dimensions, zero NaNs, and exact L2 normalization. | **PASSED** |
| **3.1** | **Pretrained Model Freezing** | Both TitaNet-Large and WeSpeaker ResNet293-LM were strictly frozen. No backpropagation, gradient calculation, or fine-tuning was performed. | **PASSED** |
| **3.2** | **No High-Capacity Architecture** | No attention layers, MLPs, transformers, or neural networks were introduced. Gate is a linear ridge regressor ($\lambda = 0.05$) mapped through a fixed sigmoid. | **PASSED** |
| **4.1** | **Strict LORO Gate Isolation** | The reliability gate was fitted under strict 8-fold Leave-One-Recording-Out. When evaluating recording $k$, recording $k$'s data was completely excluded from training, feature scaling ($\mu, \sigma$), and weight estimation. | **PASSED** |
| **4.2** | **Turn-Disjoint LOTO Targets** | Training targets $\Delta\text{margin}$ were constructed using Leave-One-Turn-Out centroids from non-overlapping ground-truth turns ($\ge 1.5$s duration). Subsegments sharing a turn with the test subsegment were excluded from centroid calculation. | **PASSED** |
| **4.3** | **Equal Recording Weights** | Sample weights were normalized so each recording contributed equally ($1 / N_{\text{rec}}$) to the loss function, preventing long recordings (e.g. R06) from dominating. | **PASSED** |
| **4.4** | **No Downstream Metric Leakage** | Gate parameters were optimized strictly to predict representation margin differences ($\Delta\text{margin}$). No diarization metrics (DER, JER, cpWER) or spectral clustering parameters were accessible during gate fitting. | **PASSED** |
| **5.1** | **Frozen-Threshold Verification** | Pairwise verification thresholds were optimized exclusively on training folds to minimize training EER, then frozen and applied out-of-fold to test recordings. No threshold tuning on test folds. | **PASSED** |
| **5.2** | **Elimination of Leaked EER Reporting** | Exploratory full-dataset leaked EER is explicitly flagged as invalid/leaked in all tables; primary verification metrics are Out-of-Fold (OOF) HTER and actDCF under frozen thresholds. | **PASSED** |
| **6.1** | **Standard Evaluation Conventions** | Evaluated DER using `pyannote.metrics` with both standard conventions: (a) collar = 0.0s with overlap, (b) collar = 0.25s without overlap. Both macro-unweighted and duration-weighted scores are reported. | **PASSED** |
| **6.2** | **Ground-Truth Preservation** | Exactly 8 recordings evaluated without silent fallback or segment modification. Ground-truth annotations were parsed with strict timestamp validation. | **PASSED** |
| **6.3** | **WhisperX Attribution Integrity** | Word-level alignments matched Stage 1 Whisper transcripts. Word-attributed metrics (cpWER, saWER, micro/macro speaker accuracy) computed with Hungarian speaker matching. | **PASSED** |
| **7.1** | **Clustered Bootstrap Uncertainty** | 2,000 bootstrap resamples clustered by recording unit. Percentile 95% confidence intervals and confidence interval widths computed for all primary metrics and paired differences. | **PASSED** |
| **7.2** | **Statistical Honesty** | No claims of statistical significance where confidence intervals include zero. Positive cpWER significance explicitly supported by CI $[-2.13\%, -0.12\%]$. | **PASSED** |
| **8.1** | **Scope Boundary Enforcement** | Stopped strictly at Stage 5A boundary. Did not execute Stage 5B (VAD/overlap correction), Stage 5C (speaker-count calibration), or GenAI audio augmentation. | **PASSED** |

---

## 3. Deep-Dive Audit of Gate Independence and Leakage Prevention

### 3.1 Mathematical Formulation of Gate Training
The gate parameters $\mathbf{w}_k$ and $b_k$ for fold $k$ (where recording $R_k$ is held out) were fitted by minimizing:

$$\mathcal{L}(\mathbf{w}, b) = \sum_{r \neq k} \frac{1}{M_r} \sum_{i=1}^{M_r} \left( \Delta\text{margin}_i - (\mathbf{w}^T \tilde{\mathbf{x}}_i + b) \right)^2 + \lambda \Vert \mathbf{w} \Vert_2^2$$

where:
- $M_r$ is the number of clean LOTO subsegments in recording $r$.
- $\tilde{\mathbf{x}}_i = (\mathbf{x}_i - \boldsymbol{\mu}_{-k}) \oslash \boldsymbol{\sigma}_{-k}$, where $\boldsymbol{\mu}_{-k}$ and $\boldsymbol{\sigma}_{-k}$ are estimated strictly from $\bigcup_{r \neq k} \mathcal{S}_r$.
- $\lambda = 0.05$ (ridge penalty).

### 3.2 Verification of Complete Fold Separation
An audit of `run_stage5a_eval.py` (lines 142–185 and 273–322) confirms:
1. When recording $R_k$ is processed, only subsegments where `rec_id != R_k` are passed into `gate.fit()`.
2. The gate object is instantiated fresh for each fold and stored in `gates_by_condition[cond_name][held_out_rec]`.
3. During downstream evaluation of recording $R_k$, the affinity matrix is fused using `gate = gates_by_condition[cond_name][rec_id]`, which has never seen $R_k$.
4. The gate output is clipped strictly to $[0.15, 0.85]$, preventing degenerative single-encoder collapse.

---

## 4. Audit Conclusion

The execution of Stage 5A satisfies all research safeguards, reproducibility criteria, and statistical standards established for this project. All reported results represent unbiased, out-of-fold generalization metrics on the target evaluation set.
