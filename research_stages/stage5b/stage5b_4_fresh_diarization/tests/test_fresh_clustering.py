#!/usr/bin/env python3
"""Tests for Stage 5B.4 Fresh Spectral Clustering Interface."""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fresh_clustering import (
    StaleClusterLabelError,
    reject_frozen_label_inputs,
    synthetic_nme_sc,
    cluster_fresh_affinity,
)


def test_stale_label_rejection_guard():
    with pytest.raises(StaleClusterLabelError, match="Prohibited stale cluster label"):
        reject_frozen_label_inputs({"frozen_labels": ["spk_0", "spk_1"]})

    with pytest.raises(StaleClusterLabelError, match="Prohibited stale cluster label"):
        reject_frozen_label_inputs({"stage5a_labels": [0, 1, 0]})

    with pytest.raises(StaleClusterLabelError, match="Prohibited stale cluster label"):
        reject_frozen_label_inputs({"ground_truth_labels": ["A", "B"]})

    # Legitimate kwargs must pass cleanly
    reject_frozen_label_inputs({"max_num_speakers": 10, "random_state": 42})


def test_synthetic_nme_sc_speaker_separation():
    # Construct a 2-speaker block-diagonal affinity matrix (8 segments total)
    # Speaker 1: segments 0..3 (high intra-affinity 0.85)
    # Speaker 2: segments 4..7 (high intra-affinity 0.85)
    # Inter-speaker: low affinity 0.05
    A = np.full((8, 8), 0.05, dtype=np.float32)
    A[:4, :4] = 0.85
    A[4:, 4:] = 0.85
    np.fill_diagonal(A, 1.0)

    labels, k_est = synthetic_nme_sc(A, max_num_speakers=5)
    assert k_est == 2
    # Segments 0..3 must have identical label
    assert len(set(labels[:4])) == 1
    # Segments 4..7 must have identical label
    assert len(set(labels[4:])) == 1
    # Labels between the two groups must differ
    assert labels[0] != labels[4]


def test_cluster_fresh_affinity_interface():
    # 4 segments
    A = torch.tensor([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0],
    ], dtype=torch.float32)

    subsegs = [
        {"start": 0.0, "end": 0.5},
        {"start": 0.25, "end": 0.75},
        {"start": 1.0, "end": 1.5},
        {"start": 1.25, "end": 1.75},
    ]

    spk_labels, k_est = cluster_fresh_affinity(A, subsegs, use_nemo_cuda=False)
    assert k_est == 2
    assert len(spk_labels) == 4
    assert spk_labels[0] == spk_labels[1]
    assert spk_labels[2] == spk_labels[3]
    assert spk_labels[0] != spk_labels[2]
