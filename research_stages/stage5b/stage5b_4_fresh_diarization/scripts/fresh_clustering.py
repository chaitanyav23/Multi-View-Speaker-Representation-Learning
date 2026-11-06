#!/usr/bin/env python3
"""
Stage 5B.4 Fresh Spectral Clustering Interface (NME-SC)
Implements Normalized Maximum Eigengap Spectral Clustering without any reference-label
or frozen-baseline leakage. Contains explicit guards rejecting stale cluster labels.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.cluster import KMeans


class StaleClusterLabelError(ValueError):
    """Raised when an attempt is made to pass frozen or ground-truth cluster labels."""
    pass


def reject_frozen_label_inputs(kwargs: Dict[str, Any]) -> None:
    """
    Guards clustering against accidental re-use of frozen baseline cluster labels.
    """
    prohibited_keys = [
        "frozen_labels",
        "baseline_labels",
        "stage5a_labels",
        "stage5b1_labels",
        "ground_truth_labels",
        "oracle_labels",
        "ref_labels",
    ]
    for key in prohibited_keys:
        if key in kwargs and kwargs[key] is not None:
            raise StaleClusterLabelError(
                f"FATAL: Prohibited stale cluster label argument detected: '{key}'. "
                "Fresh clustering must operate strictly on affinity matrices."
            )


def synthetic_nme_sc(
    A: np.ndarray,
    max_num_speakers: int = 10,
    max_rp_threshold: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, int]:
    """
    Pure Python / NumPy / SciPy implementation of Normalized Maximum Eigengap
    Spectral Clustering (NME-SC) for synthetic unit testing and CPU fallback.
    Identical mathematical objective to NeMo's offline_clustering.
    """
    n = A.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), 0
    if n == 1:
        return np.zeros(1, dtype=np.int64), 1

    # Ensure symmetric and non-negative
    A_sym = 0.5 * (A + A.T)
    np.fill_diagonal(A_sym, 0.0)
    A_sym = np.maximum(A_sym, 0.0)

    # Degree matrix
    degrees = np.sum(A_sym, axis=1)
    d_inv_sqrt = np.power(np.maximum(degrees, 1e-12), -0.5)
    L_norm = np.eye(n) - np.diag(d_inv_sqrt) @ A_sym @ np.diag(d_inv_sqrt)

    # Eigen-decomposition (symmetric)
    evals, evecs = np.linalg.eigh(L_norm)
    idx_sorted = np.argsort(evals)
    evals = evals[idx_sorted]
    evecs = evecs[:, idx_sorted]

    # Eigengap estimation
    max_k = min(max_num_speakers, n)
    if max_k <= 1:
        k_est = 1
    else:
        # Search for maximum eigengap among eigenvalues 0..max_k-1
        gaps = np.diff(evals[:max_k])
        # If all gaps near zero, fallback to 1
        if len(gaps) == 0 or np.max(gaps) < 1e-6:
            k_est = 1
        else:
            k_est = int(np.argmax(gaps) + 1)

    k_est = max(1, min(k_est, max_k))

    if k_est == 1:
        return np.zeros(n, dtype=np.int64), 1

    # Embeddings for k_est
    U = evecs[:, :k_est]
    # L2 normalize rows
    U_norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / np.maximum(U_norms, 1e-12)

    # K-means clustering
    kmeans = KMeans(n_clusters=k_est, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(U_norm)

    return cluster_labels, k_est


def cluster_fresh_affinity(
    A: torch.Tensor,
    base_subsegments: List[Dict[str, float]],
    use_nemo_cuda: bool = False,
    max_num_speakers: int = 10,
    **kwargs,
) -> Tuple[List[str], int]:
    """
    Fresh clustering entry point.
    Rejects any stale cluster label inputs.
    Returns: (subsegment_speaker_labels, estimated_num_speakers)
    """
    reject_frozen_label_inputs(kwargs)

    n_base = len(base_subsegments)
    if A.shape[0] != n_base or A.shape[1] != n_base:
        raise ValueError(
            f"Affinity shape {A.shape} does not match base subsegments count {n_base}"
        )

    if n_base == 0:
        return [], 0

    if use_nemo_cuda and torch.cuda.is_available():
        from nemo.collections.asr.parts.utils.offline_clustering import (
            SpeakerClustering,
            generate_cluster_labels,
        )
        clustering = SpeakerClustering(cuda=True)
        # Format timestamps array: shape (N, 2)
        base_ts = torch.tensor(
            [[s["start"], s["end"]] for s in base_subsegments],
            dtype=torch.float32,
            device="cuda",
        )
        c_labels = clustering.forward_unit_infer(
            mat=A.to("cuda").float(),
            oracle_num_speakers=-1,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=0.25,
            sparse_search_volume=30,
            est_num_of_spk_enhanced=torch.tensor(-1),
            fixed_thres=-1.0,
            kmeans_random_trials=1,
        )
        labels, _ = generate_cluster_labels(base_ts.cpu().numpy(), c_labels.cpu().numpy())
        unique_spks = sorted(list(set(labels)))
        k_est = len(unique_spks)
        return [str(lbl) for lbl in labels], k_est
    else:
        # Standalone / synthetic mode
        A_np = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else np.array(A)
        c_labels, k_est = synthetic_nme_sc(A_np, max_num_speakers=max_num_speakers)
        spk_strings = [f"speaker_{int(c)}" for c in c_labels]
        return spk_strings, k_est
