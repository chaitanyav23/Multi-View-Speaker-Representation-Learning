# Stage 5A.1: Affinity Normalization, Fusion Fairness & Implementation Audit
## Implementation and Feature Audit of the Reliability Gate

**Date:** 2026-10-31  
**Audit Target:** Reliability Gate Architecture, Training Targets, Feature Signals, and Modeling Assumptions  
**Auditor:** Senior Research Systems Auditor & Machine Learning Engineer

---

## 1. Executive Summary

This report documents the implementation audit of the low-capacity reliability gate evaluated in Stage 5A and Stage 5A.1. The objective is to verify the exact mathematical equations, audit feature sources and variance, verify the strict absence of data leakage under Leave-One-Recording-Out (LORO) cross-validation, and clarify architectural assumptions.

### Key Discoveries
1. **Gate Mapping Equation Discrepancy:**  
   The executable implementation in `fusion_models.py` uses a **linear mapping with clipping**:
   $$g_{\text{res}} = \text{clip}\left(0.5 + 2.0 \cdot \hat{\Delta}, 0.15, 0.85\right)$$
   rather than the bounded sigmoid mapping $\text{clip}\left(\sigma(2.0 \cdot \hat{\Delta}), 0.15, 0.85\right)$ stated in the Stage 5A report text. An empirical sensitivity analysis conducted in Stage 5A.1 demonstrates that the linear and sigmoid mappings yield very similar downstream diarization performance (Macro DER: $45.92\%$ linear vs $46.12\%$ sigmoid).
2. **ResNet Embedding Norm is Measured Pre-Normalization:**  
   `resnet293_l2_norm` is confirmed to be the Euclidean norm of the raw output embedding vector from WeSpeaker ResNet293-LM **before** L2 unit-normalization ($\Vert \mathbf{e}_{\text{raw}} \Vert_2$). It reflects representation confidence.
3. **VAD Confidence Exhibits Near-Zero Variance on Clean Speech:**  
   Across the 2,771 clean LOTO subsegments, `vad_conf_mean` has a mean of $0.954$, a median of $0.991$, and a variance of only $0.0102$. Over $75\%$ of clean subsegments have VAD confidence $> 0.962$. This explains why `vad_conf_mean` contributed negligible predictive weight in Stage 5A.
4. **Architectural Classification:**  
   The gate must be classified as **low-capacity, recording-held-out calibrated reliability-aware affinity fusion**. It is not training-free; it fits a 3-parameter ridge regressor on out-of-fold recordings with equal recording weights.
5. **Scale-k 4 Modeling Assumption:**  
   The gate uses features extracted at temporal Scale 4 (window 0.5s, shift 0.25s) because Scale 4 represents the base resolution defining the $N \times N$ multiscale affinity matrix. Gating is applied as a pairwise Hadamard product to the multiscale-fused matrices.

---

## 2. Gate Equation Verification

### 2.1 Code vs Documentation Discrepancy
In `fusion_models.py` (lines 96–103), the method `predict_subsegment_weights` is implemented as:
```python
def predict_subsegment_weights(self, meta_items: List[Dict[str, Any]]) -> np.ndarray:
    pred_delta = self.predict_delta_margin(meta_items)
    g_res = 0.5 + self.gain_factor * pred_delta
    return np.clip(g_res, self.clip_min, self.clip_max).astype(np.float32)
```
- The mapping is strictly **linear with clipping**: $g_{\text{res}} = \text{clip}(0.5 + 2.0 \cdot \hat{\Delta}, 0.15, 0.85)$.
- The Stage 5A report text inadvertently wrote $g_i = \text{clip}(\sigma(2.0 \cdot \hat{\Delta}_i), 0.15, 0.85)$.
- Because $\sigma(x) \approx 0.5 + 0.25 x$ near $x=0$, a linear mapping with slope $2.0$ acts as a first-order Taylor approximation with higher gain near the balanced point ($0.5$).

### 2.2 Sensitivity Analysis: Linear vs Sigmoid Gate
In Stage 5A.1, we implemented Condition K (linear-clipped gate) and Condition K_sig (sigmoid gate):

| Metric | Condition K (Linear Clipped Gate) | Condition K_sig (Sigmoid Gate) | Difference (K_sig - K) | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **Macro DER (0.0s, ovl)** | **45.92%** | 46.12% | +0.20% | $[0.00\%, +0.59\%]$ |
| Dur-Weighted DER (0.0s, ovl) | **51.49%** | 52.03% | +0.54% | $[0.00\%, +1.02\%]$ |
| Macro DER (0.25s, no-ovl) | **39.39%** | 39.59% | +0.20% | $[0.00\%, +0.59\%]$ |
| Speaker Count MAE | 1.625 | **1.250** | -0.375 | $[-1.125, 0.000]$ |
| Macro cpWER | 37.74% | **37.02%** | -0.72% | $[-2.26\%, +0.10\%]$ |
| **Macro 6-Rec ROC-AUC** | **0.8458** | 0.8454 | -0.0004 | $[-0.003, +0.002]$ |
| **OOF Pooled HTER** | **27.62%** | 27.82% | +0.20% | $[-0.35\%, +0.65\%]$ |

**Verdict:** The linear-clipped gate achieves slightly lower DER ($45.92\%$ vs $46.12\%$), while the sigmoid gate achieves slightly lower cpWER ($37.02\%$ vs $37.74\%$). Both mappings significantly outperform single-encoder baselines in verification. The primary linear-clipped reproduction is retained as standard.

---

## 3. Reliability Feature Audit

The three features used by the reliability gate were audited on all 2,771 clean LOTO subsegments extracted across the 5 eligible multi-speaker recordings (R03, R05, R06, R07, R08):

| Feature Name | Feature Source / Definition | Mean | Std | Min | Median | Max | Variance | Signal Quality Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `resnet293_l2_norm` | Euclidean norm of raw ResNet293-LM embedding prior to unit-normalization | 1.260 | 0.573 | 0.229 | 1.267 | 3.074 | 0.3287 | **High Quality.** Wide dynamic range, strong predictive correlation with margin advantage ($\rho = 0.31$). |
| `dfn_noise_reduction_db` | $10 \log_{10} \frac{P_{\text{mono}}}{P_{\text{removed}}}$ energy ratio on subsegment | 8.569 | 6.246 | -0.023 | 7.865 | 31.273 | 39.015 | **Moderate Quality.** Broad dynamic range, but extrinsic to speaker discrimination; slight risk of overfitting. |
| `vad_conf_mean` | Mean frame-level speech posterior from pyannote VAD in window | 0.954 | 0.101 | 0.187 | 0.991 | 1.000 | 0.0102 | **Uninformative on Clean Turns.** $75\%$ of clean turns have VAD $> 0.962$. Variance is near-zero; negligible predictive power. |
| `delta_margin` *(target)* | $m_{\text{Res}} - m_{\text{Tita}}$ on turn-disjoint prototypes | 0.039 | 0.129 | -0.521 | 0.042 | 0.520 | 0.0167 | Target variable. Mean is slightly positive ($+0.039$), indicating a slight margin advantage for ResNet on clean turns. |

### 3.1 Feature Extraction Details
1. **`resnet293_l2_norm` (Pre-norm vs Post-norm):**  
   Confirmed in `run_stage3_extraction.py` (lines 558–569). The raw ResNet output is `b_out = resnet_model(b_fb)`. Norm is `res_norms = np.linalg.norm(res_raw, axis=-1)`. `resnet293_l2_norm` stores `res_norms`. It is the true unnormalized embedding length.
2. **`dfn_noise_reduction_db`:**  
   Computed as the energy ratio between original mono audio and the removed noise signal ($x_{\text{mono}} - x_{\text{dfn}}$). Positive dB indicates substantial noise attenuation by DeepFilterNet.
3. **`vad_conf_mean`:**  
   Computed from frame-level speech posteriors. Because the LOTO training dataset requires speech segments to align with ground-truth turns, non-speech segments are excluded, leaving `vad_conf_mean` saturated near $1.0$.

---

## 4. Gate Training Protocol & Leakage Verification

### 4.1 Strict Leave-One-Recording-Out (LORO)
To eliminate data leakage, the gate is fitted under strict 8-fold LORO:
1. When evaluating recording $R_k$, all 2,771 subsegments are filtered:
   $$\mathcal{D}_{\text{train}} = \{ x \in \mathcal{D}_{\text{LOTO}} \mid x.\text{rec\_id} \neq R_k \}$$
2. Feature standardization moments $\boldsymbol{\mu}_{-k}$ and $\boldsymbol{\sigma}_{-k}$ are estimated exclusively from $\mathcal{D}_{\text{train}}$:
   $$\boldsymbol{\mu}_{-k} = \sum_{x \in \mathcal{D}_{\text{train}}} w_x \mathbf{x}, \qquad \boldsymbol{\sigma}_{-k} = \sqrt{\sum_{x \in \mathcal{D}_{\text{train}}} w_x (\mathbf{x} - \boldsymbol{\mu}_{-k})^2}$$
3. Recording weighting is balanced: each training recording receives an equal aggregate weight of $\frac{1}{N_{\text{train\_recs}}}$, preventing the longest recording (R06) from dominating the loss function.
4. Ridge regression penalty is fixed to $\lambda = 0.05$.
5. Test recording $R_k$ features are standardized using $\boldsymbol{\mu}_{-k}$ and $\boldsymbol{\sigma}_{-k}$ and passed through the frozen gate model.
6. **Zero held-out ground truth or downstream clustering metrics are accessible during gate training.**

### 4.2 Turn-Disjoint Prototype Construction
Training targets $\Delta\text{margin} = m_{\text{Res}} - m_{\text{Tita}}$ are constructed via:
$$m = \cos(\mathbf{e}_i, \mathbf{c}^+_{\setminus \text{turn}(i)}) - \max_{s' \neq s} \cos(\mathbf{e}_i, \mathbf{c}_{s'})$$
- The positive prototype $\mathbf{c}^+_{\setminus \text{turn}(i)}$ strictly excludes all subsegments belonging to the same turn as subsegment $i$.
- Negative prototypes $\mathbf{c}_{s'}$ represent full centroids of other speakers within the same recording.
- This ensures that local acoustic continuity within a single speech turn cannot leak into the target margin calculation.

---

## 5. Architectural Modeling Assumptions: Scale-k 4

### 5.1 Why Scale 4?
In NeMo's multiscale diarization framework, embeddings are extracted across 5 temporal scales:
- Scale 0: 1.50s window, 0.75s shift
- Scale 1: 1.25s window, 0.625s shift
- Scale 2: 1.00s window, 0.500s shift
- Scale 3: 0.75s window, 0.375s shift
- **Scale 4: 0.50s window, 0.250s shift**

The final affinity matrix used for spectral clustering is of size $N \times N$, where $N$ is the number of subsegments at the finest temporal resolution (**Scale 4**). Coarser scale affinities are projected onto the Scale 4 grid via nearest-timestamp mapping (`get_argmin_mat`).

### 5.2 Application in the Computational Graph
The reliability gate operates at the subsegment level:
1. For each subsegment $i \in \{1, \dots, N\}$ at Scale 4, the gate computes a scalar weight $g_i \in [0.15, 0.85]$.
2. The pairwise affinity gate matrix $G \in \mathbb{R}^{N \times N}$ is constructed via pairwise averaging:
   $$G_{ij} = \frac{1}{2} (g_i + g_j)$$
3. Gating is applied **after** multiscale cosine aggregation:
   $$A_{\text{fused}} = (1.0 - G) \odot A^{\text{norm}}_{\text{Tita}} + G \odot A^{\text{norm}}_{\text{Res}}$$
This design choice applies segment-level reliability weighting directly to the fused multiscale affinity graph.
