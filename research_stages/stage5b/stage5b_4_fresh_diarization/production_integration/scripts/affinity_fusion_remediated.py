#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Affinity Fusion Module
Computes multi-scale cosine affinity matrices, off-diagonal z-score normalization,
and Condition J equal-weight fusion.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


MULTISCALE_WEIGHTS = [0.225, 0.225, 0.225, 0.225, 0.100]


def scaler_min_max(X: torch.Tensor) -> torch.Tensor:
    """Min-max scaler mapping tensor values to [0, 1]."""
    x_min = torch.min(X)
    x_max = torch.max(X)
    diff = x_max - x_min
    if diff < 1e-12:
        return torch.zeros_like(X)
    return (X - x_min) / diff


def offdiag_zscore_affinity(A: torch.Tensor) -> torch.Tensor:
    """
    Z-score standardizes off-diagonal entries of an affinity matrix A and maps to [0, 1].
    Diagonal entries are fixed to 1.0. Exact replica of Stage 5A.1 fusion_models.py.
    """
    n = A.shape[0]
    if n <= 1:
        return torch.ones_like(A)

    mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
    off = A[mask]
    mu = torch.mean(off)
    sd = torch.clamp(torch.std(off), min=1e-6)
    Z = (A - mu) / sd
    Z_sig = torch.sigmoid(Z)
    Z_sig.fill_diagonal_(1.0)
    return scaler_min_max(Z_sig)


def compute_cosine_affinity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Computes pair-wise cosine similarity matrix from embeddings."""
    norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    norm_embs = embeddings / norms
    A = torch.matmul(norm_embs, norm_embs.t())
    A = torch.clamp(A, -1.0, 1.0)
    A.fill_diagonal_(1.0)
    return A


def compute_multiscale_cosine_affinity(
    scale_embeddings: Dict[int, torch.Tensor],
    scale_subsegments: Dict[int, List[Dict[str, float]]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds the base-scale affinity matrix using multi-scale embeddings and temporal alignment.
    Base scale is scale 4 (0.50s window).
    """
    base_scale = 4
    if base_scale not in scale_embeddings or base_scale not in scale_subsegments:
        raise KeyError("Base scale 4 missing from embeddings or subsegments")

    base_subsegs = scale_subsegments[base_scale]
    n_base = len(base_subsegs)
    if n_base == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=device)

    base_mids = np.array([(s["start"] + s["end"]) / 2.0 for s in base_subsegs])
    A_total = torch.zeros((n_base, n_base), dtype=torch.float32, device=device)
    total_weight = 0.0

    for scale_k, weight in enumerate(MULTISCALE_WEIGHTS):
        if scale_k not in scale_embeddings or scale_k not in scale_subsegments:
            continue

        embs = scale_embeddings[scale_k].to(device)
        subsegs = scale_subsegments[scale_k]

        if scale_k == base_scale:
            A_k = compute_cosine_affinity_matrix(embs)
        else:
            scale_starts = np.array([s["start"] for s in subsegs])
            scale_ends = np.array([s["end"] for s in subsegs])
            scale_mids = (scale_starts + scale_ends) / 2.0
            mapping = np.zeros(n_base, dtype=np.int64)

            for b_idx, b_mid in enumerate(base_mids):
                contains = (scale_starts <= b_mid + 1e-5) & (b_mid <= scale_ends + 1e-5)
                if np.any(contains):
                    c_indices = np.where(contains)[0]
                    closest = c_indices[np.argmin(np.abs(scale_mids[c_indices] - b_mid))]
                    mapping[b_idx] = closest
                else:
                    mapping[b_idx] = np.argmin(np.abs(scale_mids - b_mid))

            mapped_embs = embs[mapping]
            A_k = compute_cosine_affinity_matrix(mapped_embs)

        A_total += weight * A_k
        total_weight += weight

    if total_weight > 0:
        A_total = A_total / total_weight

    A_total.fill_diagonal_(1.0)
    return A_total


def compute_condition_j_fusion(
    A_titanet: torch.Tensor,
    A_resnet: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Condition J equal-weight fusion of normalized affinity matrices:
    A_eq = 0.5 * A_zt + 0.5 * A_zr
    """
    if A_titanet.shape != A_resnet.shape:
        raise ValueError(
            f"Shape mismatch between TitaNet {A_titanet.shape} and ResNet {A_resnet.shape}"
        )

    A_zt = offdiag_zscore_affinity(A_titanet)
    A_zr = offdiag_zscore_affinity(A_resnet)

    A_eq = 0.5 * A_zt + 0.5 * A_zr
    A_eq.fill_diagonal_(1.0)

    assert torch.all(torch.isfinite(A_eq)), "A_eq contains NaN or Inf"
    return A_eq, A_zt, A_zr
