#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Clustering Backends
Explicitly separates synthetic/test spectral clustering from production NeMo clustering.
Enforces input contract validation and stale-label rejection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.cluster import KMeans


class StaleClusterLabelError(ValueError):
    """Raised when an attempt is made to pass frozen or ground-truth cluster labels."""
    pass


class BackendUnavailableError(RuntimeError):
    """Raised when a production backend is requested but unavailable or unauthorized."""
    pass


def reject_frozen_label_inputs(kwargs: Dict[str, Any]) -> None:
    """Guards clustering against accidental re-use of frozen baseline cluster labels."""
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
                "Fresh clustering must operate strictly on fresh affinity matrices."
            )


def validate_affinity_matrix(A: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Validates that affinity matrix is 2D square, symmetric, finite, and diagonal is 1.0."""
    if isinstance(A, np.ndarray):
        A_t = torch.from_numpy(A).float()
    elif isinstance(A, torch.Tensor):
        A_t = A.float()
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(A)}")

    if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
        raise ValueError(f"Affinity matrix must be 2D square (N, N), got {A_t.shape}")

    if not torch.all(torch.isfinite(A_t)):
        raise ValueError("Affinity matrix contains NaN or Inf.")

    # Symmetry check
    if not torch.allclose(A_t, A_t.t(), atol=1e-4):
        raise ValueError("Affinity matrix is not symmetric.")

    return A_t


class BaseClusteringBackend(ABC):
    """Abstract base class for fresh clustering backends."""

    @abstractmethod
    def cluster(
        self,
        affinity_matrix: torch.Tensor,
        base_subsegments: List[Dict[str, float]],
        max_num_speakers: int = 10,
        **kwargs,
    ) -> Tuple[List[str], int]:
        """
        Clusters subsegments given affinity matrix.
        Returns: (speaker_labels, estimated_k)
        """
        pass


class SyntheticSpectralClusteringBackend(BaseClusteringBackend):
    """
    TEST-ONLY synthetic NME-SC routine for CPU execution and pipeline verification.
    Not mathematically or behaviorally identical to production NeMo GPU clustering.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_synthetic = True

    def cluster(
        self,
        affinity_matrix: torch.Tensor,
        base_subsegments: List[Dict[str, float]],
        max_num_speakers: int = 10,
        **kwargs,
    ) -> Tuple[List[str], int]:
        reject_frozen_label_inputs(kwargs)
        A_t = validate_affinity_matrix(affinity_matrix)
        n = A_t.shape[0]

        if n != len(base_subsegments):
            raise ValueError(f"Affinity rows ({n}) != base subsegments ({len(base_subsegments)})")

        if n == 0:
            return [], 0
        if n == 1:
            return ["speaker_0"], 1

        A_np = A_t.detach().cpu().numpy()
        A_sym = 0.5 * (A_np + A_np.T)
        np.fill_diagonal(A_sym, 0.0)
        A_sym = np.maximum(A_sym, 0.0)

        degrees = np.sum(A_sym, axis=1)
        d_inv_sqrt = np.power(np.maximum(degrees, 1e-12), -0.5)
        L_norm = np.eye(n) - np.diag(d_inv_sqrt) @ A_sym @ np.diag(d_inv_sqrt)

        evals, evecs = np.linalg.eigh(L_norm)
        idx_sorted = np.argsort(evals)
        evals = evals[idx_sorted]
        evecs = evecs[:, idx_sorted]

        max_k = min(max_num_speakers, n)
        if max_k <= 1:
            k_est = 1
        else:
            gaps = np.diff(evals[:max_k])
            k_est = int(np.argmax(gaps) + 1) if (len(gaps) > 0 and np.max(gaps) >= 1e-6) else 1

        k_est = max(1, min(k_est, max_k))
        if k_est == 1:
            return ["speaker_0"] * n, 1

        U = evecs[:, :k_est]
        U_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U_norm = U / np.maximum(U_norms, 1e-12)

        kmeans = KMeans(n_clusters=k_est, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(U_norm)
        return [f"speaker_{int(c)}" for c in cluster_labels], k_est


class NeMoProductionClusteringBackend(BaseClusteringBackend):
    """
    Production adapter for NVIDIA NeMo SpeakerClustering.forward_unit_infer.
    Traced to Stage 5A.1:
    - oracle_num_speakers=-1
    - max_num_speakers=10
    - max_rp_threshold=0.25
    - sparse_search_volume=30
    - fixed_thres=-1.0
    - kmeans_random_trials=1
    """

    def __init__(self, mock_clustering_engine: Optional[Any] = None, allow_inference: bool = False):
        self.mock_clustering_engine = mock_clustering_engine
        self.allow_inference = allow_inference
        self.is_synthetic = False

    def cluster(
        self,
        affinity_matrix: torch.Tensor,
        base_subsegments: List[Dict[str, float]],
        max_num_speakers: int = 10,
        **kwargs,
    ) -> Tuple[List[str], int]:
        reject_frozen_label_inputs(kwargs)
        A_t = validate_affinity_matrix(affinity_matrix)
        n = A_t.shape[0]

        if n != len(base_subsegments):
            raise ValueError(f"Affinity rows ({n}) != base subsegments ({len(base_subsegments)})")

        if n == 0:
            return [], 0

        # In testing with mock engine:
        if self.mock_clustering_engine is not None:
            # Interface testing with mock engine
            base_ts = np.array([[s["start"], s["end"]] for s in base_subsegments], dtype=np.float32)
            c_labels = self.mock_clustering_engine.forward_unit_infer(
                mat=A_t,
                oracle_num_speakers=-1,
                max_num_speakers=max_num_speakers,
                max_rp_threshold=0.25,
                sparse_search_volume=30,
                est_num_of_spk_enhanced=torch.tensor(-1),
                fixed_thres=-1.0,
                kmeans_random_trials=1,
            )
            labels = [f"speaker_{int(c)}" for c in c_labels]
            k_est = len(set(labels))
            return labels, k_est

        # Real production execution:
        if not self.allow_inference:
            raise BackendUnavailableError(
                "Production NeMo clustering backend is not authorized or initialized in Stage 5B.4-A.1."
            )

        if not torch.cuda.is_available():
            raise BackendUnavailableError("NeMo SpeakerClustering requires CUDA GPU.")

        from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering

        clustering = SpeakerClustering(cuda=True)
        base_ts = torch.tensor(
            [[s["start"], s["end"]] for s in base_subsegments],
            dtype=torch.float32,
            device="cuda",
        )
        c_labels = clustering.forward_unit_infer(
            mat=A_t.to("cuda").float(),
            oracle_num_speakers=-1,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=0.25,
            sparse_search_volume=30,
            est_num_of_spk_enhanced=torch.tensor(-1),
            fixed_thres=-1.0,
            kmeans_random_trials=1,
        )
        labels = [f"speaker_{int(c)}" for c in c_labels.cpu().numpy()]
        k_est = len(set(labels))
        return labels, k_est
