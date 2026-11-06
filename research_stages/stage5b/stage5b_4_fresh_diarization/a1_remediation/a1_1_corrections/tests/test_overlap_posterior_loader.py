#!/usr/bin/env python3
"""Tests for Task 4: Overlap Posterior Input Integration."""

import numpy as np
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from overlap_posterior_loader import (
    OverlapPosteriorLoader,
    OverlapPosteriorError,
)
from path_utils import get_stage2_posteriors_dir


def test_load_real_stage2_baseline_overlap_posteriors():
    loader = OverlapPosteriorLoader()
    post_dir = get_stage2_posteriors_dir()
    if not (post_dir / "R01_sara_sharif_16k_mono_posteriors.npz").exists():
        pytest.skip("Stage 2 baseline posterior files not found.")

    times, p_ovl = loader.load_overlap_posteriors("R01_sara_sharif")
    assert len(times) > 0
    assert len(times) == len(p_ovl)
    assert np.all(p_ovl >= 0.0)
    assert np.all(p_ovl <= 1.0)


def test_missing_posterior_file(tmp_path):
    loader = OverlapPosteriorLoader(posteriors_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        loader.load_overlap_posteriors("missing_rec")


def test_missing_keys_in_npz(tmp_path):
    loader = OverlapPosteriorLoader(posteriors_dir=tmp_path)
    npz_path = tmp_path / "test_rec_16k_mono_posteriors.npz"
    # Missing p_overlap
    np.savez(str(npz_path), frame_times=np.array([0.0, 0.1, 0.2]))

    with pytest.raises(KeyError, match="missing required key 'p_overlap'"):
        loader.load_overlap_posteriors("test_rec")


def test_dimension_mismatch(tmp_path):
    loader = OverlapPosteriorLoader(posteriors_dir=tmp_path)
    npz_path = tmp_path / "test_rec_16k_mono_posteriors.npz"
    np.savez(
        str(npz_path),
        frame_times=np.array([0.0, 0.1, 0.2, 0.3]),
        p_overlap=np.array([0.1, 0.2]),  # length 2 vs 4
    )

    with pytest.raises(OverlapPosteriorError, match="Array length mismatch"):
        loader.load_overlap_posteriors("test_rec")


def test_out_of_bounds_probabilities(tmp_path):
    loader = OverlapPosteriorLoader(posteriors_dir=tmp_path)
    npz_path = tmp_path / "test_rec_16k_mono_posteriors.npz"
    np.savez(
        str(npz_path),
        frame_times=np.array([0.0, 0.1]),
        p_overlap=np.array([0.1, 1.5]),  # 1.5 > 1.0
    )

    with pytest.raises(OverlapPosteriorError, match="outside valid probability range"):
        loader.load_overlap_posteriors("test_rec")


def test_nan_probabilities(tmp_path):
    loader = OverlapPosteriorLoader(posteriors_dir=tmp_path)
    npz_path = tmp_path / "test_rec_16k_mono_posteriors.npz"
    np.savez(
        str(npz_path),
        frame_times=np.array([0.0, 0.1]),
        p_overlap=np.array([0.1, np.nan]),
    )

    with pytest.raises(OverlapPosteriorError, match="contains NaN or Inf"):
        loader.load_overlap_posteriors("test_rec")
