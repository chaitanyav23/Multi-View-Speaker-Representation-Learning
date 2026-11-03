# Stage 5B.3 — Speaker-Count Estimation and Clustering Calibration: Experiment Protocol

**Date**: 2026-11-03  
**Stage**: Stage 5B.3  
**Status**: COMPLETE & VERIFIED  
**Working Directory**: `research_stages/stage5b/stage5b_3_speaker_count/`  
**Git Baseline Commit**: `c09dd6751f2b77a81672fb7cb0fb1c4b41a7ecb5`  
**Frozen Baseline Commit**: `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`  

---

## 1. Research Objectives & Primary Question

The objective of Stage 5B.3 is to rigorously investigate the mechanics of speaker-count estimation and clustering calibration across the 8-recording multi-speaker benchmark, focusing specifically on:
1. Why NVIDIA NeMo Normalized Maximum Eigengap Spectral Clustering (NME-SC) estimates $k=1$ for `R07_bwc_cheap_shot.wav` despite multiple ground truth speakers.
2. Distinguishing whether this failure is driven by:
   - Failure of the speaker-count estimator itself.
   - Failure of the affinity graph / embedding representation.
   - Failure of spectral clustering after the count is selected.
   - Interaction between these components.
3. Evaluating whether automatic speaker-count estimation can be improved without using oracle ground-truth knowledge.
4. Comparing automatic methods against oracle speaker-count upper bounds across all 8 recordings.

---

## 2. Experimental Boundaries & Safeguards

- **Strict Isolation**: All scripts, configurations, predictions, and metrics reside exclusively inside `research_stages/stage5b/stage5b_3_speaker_count/`.
- **Zero Retraining or Re-Inference**: Frozen pretrained encoders (NVIDIA TitaNet-Large and WeSpeaker ResNet293-LM), VAD segments, and ASR word timestamps are preserved without modification.
- **Representation Reuse**: All affinity matrices are reconstructed deterministically from the cached multiscale representations of Stage 3 (`research_stages/stage3_dual_encoder/cache/`), exactly matching the frozen Stage 5A.1 Condition J (Equal-Weight Fusion: $S_{\text{fused}} = 0.5 S_{\text{TitaNet}}^{\text{zscore}} + 0.5 S_{\text{ResNet}}^{\text{zscore}}$).
- **Evaluation Integrity**: NIST/pyannote standardized metrics are computed with collar $c=0.0$ (with overlap) and collar $c=0.25$ (no overlap), reporting overall DER, Missed Detection, False Alarm, Speaker Confusion, and JER.

---

## 3. Evaluated Methods

The study evaluates six distinct clustering and count-estimation strategies:

| Method ID | Method Name | Description | Count Selection Mechanism | Oracle Info Required? |
|---|---|---|---|---|
| **M1** | `nmesc_auto` | Baseline NeMo NME-SC | Sweeps $p \in [1, p_{\text{max}}]$, selects $p$ minimizing $g_p = \frac{p/N}{\max(\text{gap})}$, then picks $\text{argmax}(\text{gap}) + 1$. | **No** (Fully Unsupervised) |
| **M2** | `oracle_k` | Oracle Ground-Truth Count | Constrains spectral clustering to $k = k^*$ (Ground Truth Speaker Count). | **Yes** (Diagnostic Upper Bound) |
| **M3** | `floor_k2` | Conversational Lower Bound | Bounds candidate count to $k \in [2, k_{\text{max}}]$, disallowing single-speaker collapse. | **No** (A priori conversational prior) |
| **M4** | `rel_eigengap` | Relative Eigengap | Selects $k = \text{argmax}_{k \ge 2} \frac{\lambda_{k+1} - \lambda_k}{\lambda_k + \epsilon}$. | **No** (Relative Spectral Curvature) |
| **M5** | `silhouette` | Silhouette-Guided Spectral | Projects embeddings into top-$k$ eigenvectors, performs $k$-means ($k \in [2, 6]$), and chooses $k$ maximizing precomputed Silhouette score. | **No** (Internal Cluster Validity) |
| **M6** | `ahc_oracle` | Agglomerative Hierarchical Clustering | Average linkage hierarchical clustering on $D = 1 - S_{\text{fused}}$ with oracle $k=k^*$. | **Yes** (Diagnostic Reference) |

---

## 4. Execution Pipeline & Reproducibility Parameters

- **Random Seed**: Fixed to `42` across all PyTorch and NumPy random generators.
- **Device**: CUDA execution on local NVIDIA GeForce RTX 4090 GPU (24 GB VRAM) for matrix operations and eigensolvers; fallback to CPU.
- **Multiscale Weights**: $\mathbf{w} = [1.0, 2.0, 3.0, 3.0, 3.0]$ across temporal scales 0 to 4 (windows: 1.5s, 1.2s, 1.0s, 0.8s, 0.5s).
- **Execution Script**: `research_stages/stage5b/stage5b_3_speaker_count/scripts/run_stage5b_3_experiments.py`.
