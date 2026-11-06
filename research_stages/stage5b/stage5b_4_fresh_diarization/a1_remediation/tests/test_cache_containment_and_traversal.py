#!/usr/bin/env python3
"""Tests for Task 5: Enforce Cache-Root Containment and Safe Path Components."""

import pytest
import os
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from isolated_embedding_cache import (
    IsolatedEmbeddingCacheManager,
    CacheIdentity,
    CacheSecurityError,
    validate_path_component,
    PROHIBITED_STAGE3_CACHE,
    CACHE_SCHEMA_VERSION,
)


def test_path_component_validation():
    # Valid components
    assert validate_path_component("rec", "R01_sara_sharif") == "R01_sara_sharif"
    assert validate_path_component("scale", 4) == "4"
    assert validate_path_component("vad", "C62") == "C62"

    # Rejection of path traversal attempts
    with pytest.raises(CacheSecurityError, match="Path traversal sequence"):
        validate_path_component("rec", "../R01")

    with pytest.raises(CacheSecurityError, match="Path traversal sequence"):
        validate_path_component("rec", "R01/sara")

    with pytest.raises(CacheSecurityError, match="Path traversal sequence"):
        validate_path_component("rec", "R01\\sara")

    with pytest.raises(CacheSecurityError, match="invalid characters"):
        validate_path_component("rec", "R01;rm")

    with pytest.raises(CacheSecurityError, match="invalid characters"):
        validate_path_component("rec", "R01$sara")

    with pytest.raises(CacheSecurityError, match="empty"):
        validate_path_component("rec", "   ")


def test_prohibited_stage3_cache_rejection():
    # Direct attempt to use Stage 3 cache
    with pytest.raises(CacheSecurityError, match="prohibited Stage 3 cache"):
        IsolatedEmbeddingCacheManager(cache_root=PROHIBITED_STAGE3_CACHE, allow_custom_root=True)

    # Subdirectory of Stage 3 cache
    with pytest.raises(CacheSecurityError, match="prohibited Stage 3 cache"):
        IsolatedEmbeddingCacheManager(cache_root=PROHIBITED_STAGE3_CACHE / "dfn" / "R01", allow_custom_root=True)


def test_cache_root_containment_rejection(tmp_path):
    # Attempting to use a directory completely outside Stage 5B.4 without allow_custom_root
    outside_dir = tmp_path / "outside_workspace"
    outside_dir.mkdir()
    with pytest.raises(CacheSecurityError, match="escapes permitted Stage 5B.4 workspace"):
        IsolatedEmbeddingCacheManager(cache_root=outside_dir, allow_custom_root=False)


def test_traversal_via_symlink_rejection(tmp_path):
    # Setup allowed custom root
    root = tmp_path / "safe_cache"
    root.mkdir()
    mgr = IsolatedEmbeddingCacheManager(cache_root=root, allow_custom_root=True)

    # Create target outside
    outside_target = tmp_path / "evil_target"
    outside_target.mkdir()

    # Symlink inside root pointing outside
    symlink_dir = root / "symlink_escape"
    os.symlink(outside_target, symlink_dir)

    with pytest.raises(CacheSecurityError, match="escapes cache root"):
        mgr._assert_contained(symlink_dir / "test.npz")


def test_atomic_write_and_containment(tmp_path):
    root = tmp_path / "safe_cache"
    root.mkdir()
    mgr = IsolatedEmbeddingCacheManager(cache_root=root, allow_custom_root=True)

    identity = CacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        recording_id="R01",
        vad_config_id="C00",
        vad_intervals_sha256="1234567890abcdef",
        encoder_name="titanet_large",
        encoder_checkpoint_sha256="abcdef1234567890",
        audio_source_id="dfn",
        audio_preprocessing_id="raw_16k",
        scale_index=4,
        window_sec=0.50,
        shift_sec=0.25,
        sample_rate=16000,
        frontend_config_version="nemo_v1",
    )

    embs = np.ones((2, 192), dtype=np.float32)
    norms = np.ones(2, dtype=np.float32)
    starts = np.array([0.0, 0.25], dtype=np.float64)
    ends = np.array([0.5, 0.75], dtype=np.float64)

    saved = mgr.save_embeddings(identity, embs, norms, starts, ends)
    assert saved.exists()
    # Confirm it is inside root
    assert str(root) in str(saved.resolve())
