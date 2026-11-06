#!/usr/bin/env python3
"""Tests for Stage 5B.4 Affinity Construction and Condition J Equal Fusion."""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from affinity_fusion import (
    compute_cosine_affinity_matrix,
    offdiag_zscore_affinity,
    compute_condition_j_fusion,
    scaler_min_max,
)


def test_cosine_affinity_matrix_properties():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((15, 192)).astype(np.float32)
    embs = torch.from_numpy(raw)

    A = compute_cosine_affinity_matrix(embs)
    assert A.shape == (15, 15)
    # Symmetry
    assert torch.allclose(A, A.t(), atol=1e-5)
    # Diagonal = 1.0
    assert torch.allclose(torch.diag(A), torch.ones(15))
    # Bounded in [-1, 1]
    assert torch.all(A >= -1.0) and torch.all(A <= 1.0)


def test_offdiag_zscore_normalization():
    # Construct an asymmetric raw affinity
    A = torch.tensor([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ], dtype=torch.float32)

    A_z = offdiag_zscore_affinity(A)
    assert A_z.shape == (3, 3)
    # Diagonal must be 1.0
    assert torch.allclose(torch.diag(A_z), torch.ones(3))
    # Values bounded in [0, 1]
    assert torch.all(A_z >= 0.0) and torch.all(A_z <= 1.0)


def test_condition_j_equal_fusion():
    A_t = torch.tensor([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.3],
        [0.1, 0.3, 1.0],
    ], dtype=torch.float32)

    A_r = torch.tensor([
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.5],
        [0.2, 0.5, 1.0],
    ], dtype=torch.float32)

    A_eq, A_zt, A_zr = compute_condition_j_fusion(A_t, A_r)
    assert A_eq.shape == (3, 3)
    # Diagonal must be 1.0
    assert torch.allclose(torch.diag(A_eq), torch.ones(3))
    # Must be exact equal weight 0.5 * A_zt + 0.5 * A_zr
    expected_offdiag = 0.5 * A_zt + 0.5 * A_zr
    expected_offdiag.fill_diagonal_(1.0)
    assert torch.allclose(A_eq, expected_offdiag, atol=1e-5)


def test_dimension_mismatch_rejection():
    A1 = torch.eye(4)
    A2 = torch.eye(5)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_condition_j_fusion(A1, A2)
