# Stage 5A.1: Affinity Normalization, Fusion Fairness & Implementation Audit
## Mathematical Audit of Affinity Normalization

**Date:** 2026-10-31  
**Audit Target:** `offdiag_zscore_affinity(A)` and its interaction with NeMo `SpeakerClustering`  
**Auditor:** Senior Experimental Auditor & Speech Systems Engineer

---

## 1. Executive Summary

This audit evaluates the mathematical properties of the affinity normalization function `offdiag_zscore_affinity(A)` used in Stage 5A, investigates whether single-encoder baselines were unfairly disadvantaged by evaluating them on unnormalized affinity matrices, and examines the exact impact of normalization on NeMo's NME-SC spectral clustering algorithm.

### Key Discoveries
1. **The Confound Existed in Stage 5A:**  
   In Stage 5A, single-encoder baselines (Condition A: TitaNet-DFN, Condition B: ResNet293-mono) were evaluated on raw multiscale cosine affinity matrices with diagonal entries of $12.0$ and off-diagonal ranges of $[0, 12]$, whereas fusion conditions (Condition C: Equal Weight, Condition D: Reliability Gate) were evaluated on off-diagonal z-scored affinities with diagonal entries of $1.0$ and off-diagonal ranges of $[0, 1]$.
2. **Single-Encoder Diarization Invariance:**  
   Empirical and mathematical audits prove that for any single encoder, `offdiag_zscore_affinity(A)` is a strictly monotonic sigmoid transformation of off-diagonal similarities. Because NeMo NME-SC selects top-$p$ neighbors per row using rank-based sorting (`torch.argsort`), **normalizing single-encoder affinity matrices produces 100% identical clustering, identical speaker counts, and identical DER/JER/cpWER scores** across all 8 recordings.
3. **Multi-View Fusion Requires Normalization:**  
   While single-encoder clustering is invariant to monotonic normalization, multi-view linear fusion ($0.5 A_{\text{Tita}} + 0.5 A_{\text{Res}}$) is **not** invariant. Raw TitaNet and ResNet affinities exhibit different means, variances, and dynamic ranges across BWC acoustic conditions. Without normalization, the encoder with higher variance disproportionately dominates graph connectivity. Normalization is therefore necessary for balanced representation fusion.
4. **Classification:**  
   The normalization is classified as **unsupervised, per-recording transductive normalization**. It uses the distribution of similarities within the test recording without using ground-truth labels.

---

## 2. Mathematical Definition & Implementation

The normalization function is implemented in `fusion_models.py` as:

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

where NeMo's `ScalerMinMax` is:

$$\text{ScalerMinMax}(X) = \frac{X - \min(X)}{\max(X) - \min(X)}$$

### Step-by-Step Mathematical Flow
1. **Off-Diagonal Extraction:** For an $N \times N$ matrix $A$, extract the $N(N-1)$ off-diagonal entries:
   $$\mathcal{A}_{\text{off}} = \{ A_{ij} \mid i \neq j \}$$
2. **Moments Estimation:**
   $$\mu = \frac{1}{N(N-1)} \sum_{i \neq j} A_{ij}, \qquad \sigma = \max\left( \sqrt{\frac{1}{N(N-1)} \sum_{i \neq j} (A_{ij} - \mu)^2}, 10^{-6} \right)$$
3. **Standardization:**
   $$Z_{ij} = \frac{A_{ij} - \mu}{\sigma} \quad \forall i, j$$
4. **Sigmoid Non-Linearity:**
   $$S_{ij} = \sigma(Z_{ij}) = \frac{1}{1 + e^{-Z_{ij}}}$$
5. **Diagonal Re-initialization:**
   $$S_{ii} = 1.0 \quad \forall i$$
6. **Min-Max Rescaling:**
   Because $S_{ii} = 1.0$ and $S_{ij} = \sigma(Z_{ij}) < 1.0$ for finite $Z_{ij}$, $\max(S) = 1.0$. Let $v_{\min} = \min_{i \neq j} S_{ij}$. Then:
   $$A^{\text{norm}}_{ii} = \frac{1.0 - v_{\min}}{1.0 - v_{\min}} = 1.0$$
   $$A^{\text{norm}}_{ij} = \frac{S_{ij} - v_{\min}}{1.0 - v_{\min}} \in [0.0, 1.0) \quad \forall i \neq j$$

---

## 3. Systematic Audit of Matrix Properties

| Question | Evaluation Property | Verified Mathematical & Code Behavior | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Diagonal Exclusion** | `mask = ~torch.eye(n)` extracts strictly off-diagonal entries for computing $\mu$ and $\sigma$. Self-similarity values do not bias the distribution moments. | **CONFIRMED** |
| **2** | **Upper and Lower Triangular Entries** | `mask` includes all $N^2 - N$ non-diagonal elements. Both $(i, j)$ and $(j, i)$ are included in the moment calculations. | **CONFIRMED** |
| **3** | **Symmetrization Treatment** | Symmetrization is **not** performed within `offdiag_zscore_affinity`. If input $A$ is symmetric, output $A^{\text{norm}}$ is symmetric. If input $A$ is asymmetric, asymmetry is preserved. | **CONFIRMED** |
| **4** | **Diagonal Value Treatment** | Diagonal entries are set to $1.0$ prior to min-max scaling (`Z_sig.fill_diagonal_(1.0)`). Post min-max scaling, diagonal entries evaluate to exactly $1.0$. | **CONFIRMED** |
| **5** | **Zero/Near-Zero Variance Handling** | Standard deviation is clamped to a minimum of $10^{-6}$ (`torch.clamp(..., min=1e-6)`), preventing division by zero. | **CONFIRMED** |
| **6** | **Sigmoid Transformation** | Applied element-wise via `torch.sigmoid(Z)`. Maps standardized z-scores from $(-\infty, +\infty)$ monotonically into $(0, 1)$. | **CONFIRMED** |
| **7** | **Min-Max Scaling** | Applied after sigmoid via `ScalerMinMax(Z_sig)`. Maps the lowest off-diagonal value to $0.0$ and diagonal to $1.0$. | **CONFIRMED** |
| **8** | **Symmetry Guarantee** | Output is guaranteed symmetric **if and only if** the input matrix is symmetric. (Input matrices from `getMultiScaleCosAffinityMatrix` are symmetric within float32 tolerance $10^{-6}$). | **CONFIRMED** |
| **9** | **Boundedness** | Output values are strictly bounded in $[0.0, 1.0]$. Minimum off-diagonal is $0.0$; diagonal is $1.0$. | **CONFIRMED** |
| **10** | **Degenerate Behavior (NaNs/Infs)** | For $N \ge 2$, zero NaNs or Infs occur. For constant off-diagonal matrices, all off-diagonals map to $0.0$ and diagonal to $1.0$. | **CONFIRMED** |

---

## 4. Synthetic Verification Results

To confirm the numerical properties, 5 synthetic tests were executed (`audit_normalization.py`):

1. **Standard Symmetric Matrix ($4 \times 4$):**
   - Input shape: $[4, 4]$, Output shape: $[4, 4]$
   - Symmetry: `True` (`torch.allclose(out, out.T, atol=1e-6)`)
   - Diagonal values: $[1.0, 1.0, 1.0, 1.0]$
   - Value range: $[0.0, 1.0]$, Min off-diagonal: $0.0$, Max off-diagonal: $0.7426$
   - NaNs / Infs: `False`
2. **Constant Off-Diagonal Matrix ($3 \times 3$, all off-diagonal = 0.5):**
   - Output diagonal: $[1.0, 1.0, 1.0]$
   - Output off-diagonal: $[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]$
   - Symmetry: `True`, NaNs: `False`
3. **Near-Zero Variance Matrix ($\sigma < 10^{-6}$):**
   - Clamping prevented divergence; output diagonal = $1.0$, min value = $0.0$.
4. **Negative Off-Diagonal Matrix:**
   - Correctly mapped negative cosine similarities into $[0.0, 1.0]$ while preserving diagonal = $1.0$.
5. **Asymmetric Input Matrix:**
   - Input symmetric: `False` $\implies$ Output symmetric: `False`. Proves that explicit symmetrization is omitted.

---

## 5. Actual Cached Affinity Matrix Audit Across 8 Recordings

Auditing the actual multiscale affinity matrices across all 8 recordings produced the following statistics:

| Recording ID | Subsegs ($N$) | Raw TitaNet ($A_{\text{tita}}$)<br>Diag \| Off-Diag Mean (Std) | Raw ResNet ($A_{\text{res}}$)<br>Diag \| Off-Diag Mean (Std) | Norm TitaNet ($A_{\text{zt}}$)<br>Diag \| Off-Diag Mean (Std) | Norm ResNet ($A_{\text{zr}}$)<br>Diag \| Off-Diag Mean (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R01_sara_sharif** | 260 | 12.0 \| 4.14 (1.41) | 12.0 \| 3.88 (1.47) | 1.0 \| 0.46 (0.22) | 1.0 \| 0.44 (0.23) |
| **R02_dennis_skinner** | 511 | 12.0 \| 3.89 (1.13) | 12.0 \| 4.11 (1.47) | 1.0 \| 0.47 (0.21) | 1.0 \| 0.45 (0.22) |
| **R03_disabled_parking** | 696 | 12.0 \| 4.12 (1.34) | 12.0 \| 4.24 (1.36) | 1.0 \| 0.46 (0.22) | 1.0 \| 0.45 (0.22) |
| **R04_piers_robinson** | 684 | 12.0 \| 3.84 (1.08) | 12.0 \| 3.73 (1.40) | 1.0 \| 0.47 (0.21) | 1.0 \| 0.44 (0.23) |
| **R05_southwest_pilot** | 556 | 12.0 \| 5.86 (2.25) | 12.0 \| 5.58 (2.05) | 1.0 \| 0.47 (0.22) | 1.0 \| 0.46 (0.22) |
| **R06_rich_karen** | 2,168 | 12.0 \| 4.29 (1.29) | 12.0 \| 3.86 (1.48) | 1.0 \| 0.46 (0.22) | 1.0 \| 0.44 (0.23) |
| **R07_bwc_cheap_shot** | 586 | 12.0 \| 5.16 (1.93) | 12.0 \| 4.54 (1.78) | 1.0 \| 0.47 (0.22) | 1.0 \| 0.45 (0.23) |
| **R08_uk_parliament** | 711 | 12.0 \| 3.99 (1.31) | 12.0 \| 3.98 (1.44) | 1.0 \| 0.46 (0.22) | 1.0 \| 0.45 (0.23) |

### Key Observations
1. In raw multiscale affinity matrices, the diagonal is always exactly $12.0$ (sum of multiscale weights: $1.0 + 2.0 + 3.0 + 3.0 + 3.0 = 12.0$).
2. Across all recordings, raw off-diagonal similarities average between $3.7$ and $5.9$, with standard deviations up to $2.25$.
3. Post normalization, all 8 recordings exhibit stable off-diagonal distributions: mean $= 0.44 \text{ to } 0.47$, std $= 0.21 \text{ to } 0.23$, min $= 0.0$, max $\approx 0.995$, with diagonal $= 1.0$.

---

## 6. Single-Encoder Invariance Proof

In Stage 5A.1, we introduced Condition H (Normalized TitaNet-DFN alone) and Condition I (Normalized ResNet293-mono alone).

Comparing Condition H vs Condition A (Raw TitaNet) and Condition I vs Condition B (Raw ResNet) yielded **zero difference**:
$$\Delta \text{DER} = 0.0000 \quad \Delta \text{JER} = 0.0000 \quad \Delta \text{cpWER} = 0.0000 \quad \Delta \text{saWER} = 0.0000$$
All 8 recordings tied exactly on all metrics.

### Mathematical Proof of Invariance
In NeMo's `SpeakerClustering.forward_unit_infer`, the affinity graph is constructed via `getKneighborsConnections(affinity_mat, p_value)`:
```python
sorted_matrix = torch.argsort(affinity_mat, dim=1, descending=True)[:, :p_value]
binarized_affinity_mat[indices_row, indices_col] = 1.0
```
Let $f(x) = \text{ScalerMinMax}(\sigma((x - \mu)/\sigma))$ be the normalization function.
- For off-diagonal elements, $f(x)$ is strictly monotonically increasing because $\frac{d}{dx} \sigma\left(\frac{x - \mu}{\sigma}\right) > 0$.
- Therefore:
  $$A_{ik} > A_{ij} \iff f(A_{ik}) > f(A_{ij}) \quad \forall j, k \neq i$$
- The diagonal entry $A_{ii} = 12.0$ is strictly greater than all off-diagonal entries ($A_{ij} \le 11.46$). After normalization, $A^{\text{norm}}_{ii} = 1.0$ remains strictly greater than all normalized off-diagonal entries ($A^{\text{norm}}_{ij} \le 0.9987$).
- Consequently, for any row $i$:
  $$\text{argsort}(A_{i,:}) \equiv \text{argsort}(A^{\text{norm}}_{i,:})$$
- Because the row-wise ranking is identical, the top-$p$ neighbor connectivity graph, the normalized Laplacian $L_{\text{sym}}$, its eigenvalues, the eigengap heuristic, the estimated speaker count $k$, and the resulting spectral cluster assignments are **100% invariant to normalization for a single encoder**.

### Conclusion
**The difference observed in Stage 5A between fusion and single-encoder baselines was NOT caused by normalization confounding.** Single-encoder baselines achieve the exact same diarization performance whether raw or normalized. The performance differences observed in fusion represent genuine multi-view representation interactions.
