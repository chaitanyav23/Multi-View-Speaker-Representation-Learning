# Stage 5A.1.2: NME-SC Diagonal-Handling Proof and Normalization Invariance Audit

**Audit Date**: November 3, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Target Subsystem**: Normalized Maximum Eigengap Spectral Clustering (NME-SC) Diagonal Handling  
**Audit Status**: MATHEMATICALLY PROVEN AND EMPIRICALLY VERIFIED  
**NeMo Version**: `nemo_toolkit == 2.7.3` (`.venv_phase20`, PyTorch 2.5.1+cu124)  
**Implementation Source**: [`offline_clustering.py:309-360`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/lib/python3.10/site-packages/nemo/collections/asr/parts/utils/offline_clustering.py#L309-L360)

---

## 1. Executive Summary

A central inquiry in the Stage 5A.1 audit was whether single-encoder spectral clustering in NeMo NME-SC is invariant under monotonic affinity normalization. While off-diagonal affine transformations preserve off-diagonal ordering, a complete mathematical proof requires examining **how the matrix diagonal is treated during row-wise neighborhood selection**.

This audit establishes both mathematically and empirically that:
1. NeMo NME-SC **includes the diagonal** in row-wise sorting (`torch.argsort(affinity_mat, dim=1, descending=True)`).
2. The normalization function ([`offdiag_zscore_affinity`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py#L20-L45)) applies a sigmoid transformation followed by an explicit diagonal reset to $1.0$ and MinMax scaling.
3. Because the sigmoid function maps all real numbers strictly into the open interval $(0, 1)$, **every off-diagonal entry is strictly less than 1.0** prior to diagonal reset.
4. Setting the diagonal to $1.0$ guarantees that **the diagonal element remains the strict global maximum of its row** ($A_{ii} = 1.0 > A_{ij} \, \forall j \ne i$).
5. Consequently, the diagonal element occupies **rank 0 descending** in both raw and normalized matrices. The remaining $p - 1$ neighbors are chosen exclusively from off-diagonal elements in identical order.
6. The resulting binary neighborhood masks, symmetrized adjacency matrices, Laplacian eigenbases, and final cluster assignments are **strictly identical** ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$).

---

## 2. NeMo Runtime Implementation Inspection

The runtime clustering pipeline was inspected in `.venv_phase20`:
- **File**: `nemo/collections/asr/parts/utils/offline_clustering.py`
- **Class**: `SpeakerClustering` (lines 1124–1350)
- **Functions**:
  - `getKneighborsConnections(affinity_mat, p_value, mask_method='binary')` (lines 309–350)
  - `getAffinityGraphMat(affinity_mat_raw, p_value)` (lines 352–360)
  - `ScalerMinMax(X)` (lines 65–75)

### Exact Executed Source Code for Neighborhood Binarization
```python
def getKneighborsConnections(affinity_mat: torch.Tensor, p_value: int, mask_method: str = 'binary') -> torch.Tensor:
    dim = affinity_mat.shape
    binarized_affinity_mat = torch.zeros_like(affinity_mat).half()
    sorted_matrix = torch.argsort(affinity_mat, dim=1, descending=True)[:, :p_value]
    binarized_affinity_mat[sorted_matrix.T, torch.arange(affinity_mat.shape[0])] = (
        torch.ones(1).to(affinity_mat.device).half()
    )
    indices_row = sorted_matrix[:, :p_value].flatten()
    indices_col = torch.arange(dim[1]).repeat(p_value, 1).T.flatten()
    if mask_method == 'binary' or mask_method is None:
        binarized_affinity_mat[indices_row, indices_col] = (
            torch.ones(indices_row.shape[0]).to(affinity_mat.device).half()
        )
    ...
    return binarized_affinity_mat
```

### Key Implementation Findings
1. **No Diagonal Masking**: The diagonal is **not** masked or excluded before `torch.argsort`.
2. **Descending Sort**: `torch.argsort(..., descending=True)` selects the largest `p_value` entries per row.
3. **Binary Assignment**: Selected indices are set to $1.0$, while all other entries remain $0.0$.
4. **Symmetrization**: `getAffinityGraphMat` explicitly symmetrizes the binarized graph via:
   $$A_{\text{sym}} = 0.5 \cdot (X + X^T)$$

---

## 3. Mathematical Proof of Invariance

Let $A \in \mathbb{R}^{N \times N}$ be the raw multiscale cosine affinity matrix.
By construction from unit embeddings:
- $A_{ii} = 1.0$ for all $i \in \{1, \dots, N\}$.
- $A_{ij} \in [-1, 1]$ for all $i \ne j$. In distinct audio subsegments, $A_{ij} < 1.0$.

### Step 1: Off-Diagonal Standardizing and Sigmoid Transformation
The normalization function computes:
$$\mu = \frac{1}{N(N-1)} \sum_{i \ne j} A_{ij}, \quad \sigma = \sqrt{\frac{1}{N(N-1)} \sum_{i \ne j} (A_{ij} - \mu)^2}$$
$$Z_{ij} = \frac{A_{ij} - \mu}{\sigma} \quad (i \ne j)$$
Because $\sigma > 0$, the mapping $A_{ij} \mapsto Z_{ij}$ is a strictly increasing monotonic affine transformation on off-diagonals:
$$A_{ij} > A_{ik} \iff Z_{ij} > Z_{ik} \quad \forall j, k \ne i$$

Next, the standard logistic sigmoid is applied:
$$(Z_{\text{sig}})_{ij} = \sigma(Z_{ij}) = \frac{1}{1 + e^{-Z_{ij}}}$$
Because $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z)) > 0$ for all $z \in \mathbb{R}$, $\sigma$ is strictly monotonically increasing.
Therefore:
$$A_{ij} > A_{ik} \iff (Z_{\text{sig}})_{ij} > (Z_{\text{sig}})_{ik} \quad \forall j, k \ne i$$
Crucially, for any finite real value $Z_{ij} \in \mathbb{R}$:
$$(Z_{\text{sig}})_{ij} \in (0, 1) \implies (Z_{\text{sig}})_{ij} < 1.0 \quad \forall i \ne j$$

### Step 2: Diagonal Reset
The function explicitly executes:
```python
Z_sig.fill_diagonal_(1.0)
```
Setting $(Z_{\text{sig}})_{ii} = 1.0$.
Because all off-diagonal entries satisfy $(Z_{\text{sig}})_{ij} < 1.0$, the diagonal entry satisfies:
$$(Z_{\text{sig}})_{ii} = 1.0 > (Z_{\text{sig}})_{ij} \quad \forall j \ne i$$
**Result**: The diagonal entry $(Z_{\text{sig}})_{ii}$ is the strict, unique global maximum of row $i$.

### Step 3: MinMax Scaling
NeMo's `ScalerMinMax` computes:
$$v_{\max} = \max_{k, l} (Z_{\text{sig}})_{kl} = 1.0, \quad v_{\min} = \min_{k, l} (Z_{\text{sig}})_{kl} > 0$$
$$A_{\text{norm}} = \frac{Z_{\text{sig}} - v_{\min}}{1.0 - v_{\min}}$$
Since $1.0 - v_{\min} > 0$, this is a strictly positive affine transformation $y = a \cdot x + b$ ($a > 0$).
- For diagonal entries:
  $$(A_{\text{norm}})_{ii} = \frac{1.0 - v_{\min}}{1.0 - v_{\min}} = 1.0$$
- For off-diagonal entries:
  $$(A_{\text{norm}})_{ij} = \frac{(Z_{\text{sig}})_{ij} - v_{\min}}{1.0 - v_{\min}} < 1.0$$
- Off-diagonal ordering is preserved:
  $$(A_{\text{norm}})_{ij} > (A_{\text{norm}})_{ik} \iff A_{ij} > A_{ik} \quad \forall j, k \ne i$$
- The diagonal entry remains the strictly maximal element of row $i$:
  $$(A_{\text{norm}})_{ii} = 1.0 > (A_{\text{norm}})_{ij} \quad \forall j \ne i$$

### Step 4: Row-Wise argsort and Top-p Connection Invariance
In `getKneighborsConnections(affinity_mat, p_value)`:
```python
sorted_matrix = torch.argsort(affinity_mat, dim=1, descending=True)[:, :p_value]
```
1. In both $A_{\text{raw}}$ and $A_{\text{norm}}$, the diagonal element is strictly maximal in row $i$.
2. Therefore, column index $i$ is guaranteed to be placed at rank 0 (the first column of `sorted_matrix`):
   $$\text{sorted\_matrix}_{i, 0} = i \quad \text{for both } A_{\text{raw}} \text{ and } A_{\text{norm}}$$
3. The remaining $p - 1$ columns of row $i$ in `sorted_matrix` ($\text{sorted\_matrix}_{i, 1:p}$) select the indices corresponding to the largest $p - 1$ off-diagonal values.
4. Because $(A_{\text{norm}})_{ij}$ and $A_{\text{raw}, ij}$ have strictly identical pairwise orderings on off-diagonals, the indices corresponding to the top $p - 1$ off-diagonal entries are **IDENTICAL**.
5. Therefore:
   $$\text{sorted\_matrix}(A_{\text{norm}}) \equiv \text{sorted\_matrix}(A_{\text{raw}})$$
6. The resulting binary neighborhood masks are identical:
   $$\mathcal{M}(A_{\text{norm}}) \equiv \mathcal{M}(A_{\text{raw}})$$
7. Symmetrization $0.5 \cdot (\mathcal{M} + \mathcal{M}^T)$, graph Laplacian computation $L_{\text{sym}} = I - D^{-1/2} \mathcal{M}_{\text{sym}} D^{-1/2}$, eigenvector decomposition, and k-means clustering yield **identical cluster assignments**.

Q.E.D.

---

## 4. Empirical Verification Across All Recordings

The script [`verify_nmesc_diagonal_invariance.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/verify_nmesc_diagonal_invariance.py) executed this exact chain on synthetic and recording affinity matrices:

| Property Verified | Raw Matrix ($A_{\text{raw}}$) | Normalized Matrix ($A_{\text{norm}}$) | Status |
| :--- | :---: | :---: | :---: |
| Diagonal values identically 1.0 | True ($1.0 \pm 0.0$) | True ($1.0 \pm 0.0$) | VERIFIED |
| Maximum off-diagonal value | 0.2614 | 0.9734 | VERIFIED |
| Off-diagonal strictly $< 1.0$ | True | True | VERIFIED |
| Diagonal rank in descending argsort | Rank 0 (Maximal) | Rank 0 (Maximal) | VERIFIED |
| Top-$p$ neighbor indices match | Reference | 100% Identical | VERIFIED |
| Binary connection matrix match | Reference | 100% Identical | VERIFIED |
| Symmetrized graph matrix match | Reference | 100% Identical | VERIFIED |

Across all 8 real-world recordings in Stage 5A.1:
- Condition H (Normalized TitaNet) vs Condition A (Raw TitaNet): **0 discrepancies across all 8 recordings and 15 diarization metrics** ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$).
- Condition I (Normalized ResNet) vs Condition B (Raw ResNet): **0 discrepancies across all 8 recordings and 15 diarization metrics** ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$).

**Audit Verdict**: Mathematical invariance is complete and fully proven. Stage 5A's comparison of raw single encoders against normalized fusion was unconfounded.
