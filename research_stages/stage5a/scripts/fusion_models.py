#!/usr/bin/env python3
"""
Stage 5A Fusion Models and Affinity Construction Module:
Implements Multiscale Affinity Builders, Off-Diagonal Z-Score Normalization,
and Low-Capacity LORO-Calibrated Reliability Gates (Full 3-feature & Ablations).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from nemo.collections.asr.parts.utils.offline_clustering import getMultiScaleCosAffinityMatrix
from nemo.collections.asr.parts.utils.offline_clustering import ScalerMinMax


def offdiag_zscore_affinity(A: torch.Tensor) -> torch.Tensor:
    """
    Z-score standardizes off-diagonal entries of an affinity matrix A and maps to [0, 1].
    Diagonal entries are fixed to 1.0.
    """
    n = A.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
    off = A[mask]
    mu = torch.mean(off)
    sd = torch.clamp(torch.std(off), min=1e-6)
    Z = (A - mu) / sd
    Z_sig = torch.sigmoid(Z)
    Z_sig.fill_diagonal_(1.0)
    return ScalerMinMax(Z_sig)


class ReliabilityGate:
    """
    Low-capacity ridge-regression gate trained strictly under Leave-One-Recording-Out (LORO)
    with equal recording weights and turn-disjoint LOTO targets.
    """

    def __init__(
        self,
        features: List[str],
        regularization_lambda: float = 0.05,
        clip_min: float = 0.15,
        clip_max: float = 0.85,
        gain_factor: float = 2.0,
    ) -> None:
        self.features = list(features)
        self.regularization_lambda = regularization_lambda
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.gain_factor = gain_factor

        self.mu_x: Optional[np.ndarray] = None
        self.sd_x: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None

    def fit(self, training_items: List[Dict[str, Any]]) -> "ReliabilityGate":
        """
        Fits gate parameters strictly on training items using inverse-recording-size weighting.
        """
        assert len(self.features) > 0, "Cannot fit gate with 0 features!"
        tr_recs = sorted(set(x["rec_id"] for x in training_items))
        rec_counts = {r: sum(1 for x in training_items if x["rec_id"] == r) for r in tr_recs}

        # Equal recording weighting: each training recording gets equal total mass
        weights = np.array(
            [1.0 / (len(tr_recs) * rec_counts[x["rec_id"]]) for x in training_items],
            dtype=np.float64,
        )
        weights = weights / np.sum(weights)

        X_tr = np.array([[x[f] for f in self.features] for x in training_items], dtype=np.float64)
        y_tr = np.array([x["delta_margin"] for x in training_items], dtype=np.float64)

        self.mu_x = np.sum(X_tr * weights[:, None], axis=0)
        var_x = np.sum(((X_tr - self.mu_x) ** 2) * weights[:, None], axis=0)
        self.sd_x = np.maximum(np.sqrt(var_x), 1e-6)

        Xz = (X_tr - self.mu_x) / self.sd_x
        self.b = float(np.sum(y_tr * weights))
        yc = y_tr - self.b

        W_diag = weights[:, None]
        reg_eye = self.regularization_lambda * np.eye(Xz.shape[1], dtype=np.float64)
        self.w = np.linalg.solve((Xz * W_diag).T @ Xz + reg_eye, (Xz * W_diag).T @ yc)
        return self

    def predict_delta_margin(self, items: List[Dict[str, Any]]) -> np.ndarray:
        """
        Predicts margin difference (res_margin - tita_margin) for test items.
        """
        assert self.w is not None and self.mu_x is not None, "Gate must be fitted before predict!"
        X_te = np.array([[x[f] for f in self.features] for x in items], dtype=np.float64)
        Xz = (X_te - self.mu_x) / self.sd_x
        return (Xz @ self.w + self.b).astype(np.float32)

    def predict_subsegment_weights(self, meta_items: List[Dict[str, Any]]) -> np.ndarray:
        """
        Predicts ResNet relative affinity weight g_res in [clip_min, clip_max] for each subsegment.
        """
        pred_delta = self.predict_delta_margin(meta_items)
        g_res = 0.5 + self.gain_factor * pred_delta
        return np.clip(g_res, self.clip_min, self.clip_max).astype(np.float32)

    def build_affinity_gate_matrix(self, meta_items: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        """
        Constructs pairwise gate matrix G_{ij} = 0.5 * (g_i + g_j).
        """
        g_res = self.predict_subsegment_weights(meta_items)
        G_mat = 0.5 * (g_res[:, None] + g_res[None, :])
        return torch.from_numpy(G_mat).to(device).float()

    def get_params_dict(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "regularization_lambda": float(self.regularization_lambda),
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
            "gain_factor": float(self.gain_factor),
            "mu_x": [float(v) for v in self.mu_x] if self.mu_x is not None else None,
            "sd_x": [float(v) for v in self.sd_x] if self.sd_x is not None else None,
            "weights": [float(v) for v in self.w] if self.w is not None else None,
            "bias": float(self.b) if self.b is not None else None,
        }
