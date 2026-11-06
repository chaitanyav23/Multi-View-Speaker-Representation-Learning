#!/usr/bin/env python3
"""Tests for Task 8: Separation of Synthetic and Production Clustering Backends."""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_backends import (
    SyntheticSpectralClusteringBackend,
    NeMoProductionClusteringBackend,
    StaleClusterLabelError,
    BackendUnavailableError,
    validate_affinity_matrix,
)


def test_stale_label_rejection():
    backend = SyntheticSpectralClusteringBackend()
    A = torch.eye(4)
    subsegs = [{"start": 0.0, "end": 0.5}] * 4

    with pytest.raises(StaleClusterLabelError, match="Prohibited stale cluster label"):
        backend.cluster(A, subsegs, frozen_labels=["spk0", "spk1", "spk0", "spk1"])

    with pytest.raises(StaleClusterLabelError, match="Prohibited stale cluster label"):
        backend.cluster(A, subsegs, ground_truth_labels=["A", "B", "A", "B"])


def test_affinity_matrix_validation():
    # Asymmetric matrix
    bad_A = torch.tensor([[1.0, 0.9], [0.2, 1.0]])
    with pytest.raises(ValueError, match="not symmetric"):
        validate_affinity_matrix(bad_A)

    # NaN in matrix
    nan_A = torch.eye(3)
    nan_A[0, 1] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_affinity_matrix(nan_A)

    # Non-square matrix
    rect_A = torch.ones(3, 4)
    with pytest.raises(ValueError, match="2D square"):
        validate_affinity_matrix(rect_A)


def test_synthetic_backend_partitioning():
    backend = SyntheticSpectralClusteringBackend(random_state=42)
    # 2 speakers (6 segments: 3 for spk1, 3 for spk2)
    A = torch.full((6, 6), 0.05)
    A[:3, :3] = 0.90
    A[3:, 3:] = 0.90
    A.fill_diagonal_(1.0)

    subsegs = [{"start": float(i), "end": float(i) + 0.5} for i in range(6)]
    labels, k_est = backend.cluster(A, subsegs)

    assert k_est == 2
    assert len(labels) == 6
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_nemo_production_adapter_contract():
    class MockNeMoClusteringEngine:
        def __init__(self):
            self.last_kwargs = {}

        def forward_unit_infer(self, **kwargs):
            self.last_kwargs = kwargs
            # Return dummy 0 and 1 labels
            return torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    mock_engine = MockNeMoClusteringEngine()
    adapter = NeMoProductionClusteringBackend(mock_clustering_engine=mock_engine)

    A = torch.eye(4)
    subsegs = [{"start": float(i), "end": float(i) + 0.5} for i in range(4)]

    labels, k_est = adapter.cluster(A, subsegs, max_num_speakers=10)
    assert k_est == 2
    assert labels == ["speaker_0", "speaker_0", "speaker_1", "speaker_1"]

    # Verify Stage 5A.1 parameter invariants passed to NeMo
    passed = mock_engine.last_kwargs
    assert passed["oracle_num_speakers"] == -1
    assert passed["max_num_speakers"] == 10
    assert passed["max_rp_threshold"] == 0.25
    assert passed["sparse_search_volume"] == 30
    assert passed["fixed_thres"] == -1.0


def test_nemo_production_adapter_unauthorized_rejection():
    # Calling production clustering without mock and without authorization
    adapter = NeMoProductionClusteringBackend(mock_clustering_engine=None, allow_inference=False)
    A = torch.eye(2)
    subsegs = [{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0}]

    with pytest.raises(BackendUnavailableError, match="not authorized or initialized"):
        adapter.cluster(A, subsegs)
