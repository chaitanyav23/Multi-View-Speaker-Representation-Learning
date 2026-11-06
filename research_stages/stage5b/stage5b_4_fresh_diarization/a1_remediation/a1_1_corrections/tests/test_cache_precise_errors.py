#!/usr/bin/env python3
"""Tests for Task 5: Precise Cache Error Handling and Counter Accounting."""

import json
import numpy as np
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from path_utils import get_stage5b_4_dir
A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from cache_manager_remediated import (
    IsolatedEmbeddingCacheManagerRemediated,
    CacheIdentity,
    CacheSecurityError,
    CacheValidationError,
    CACHE_SCHEMA_VERSION,
)


def make_sample_identity(recording_id="rec01", vad_hash="abcd"*16, encoder_hash="1234"*16, scale=0):
    return CacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        recording_id=recording_id,
        vad_config_id="C00",
        vad_intervals_sha256=vad_hash,
        encoder_name="titanet_large",
        encoder_checkpoint_sha256=encoder_hash,
        audio_source_id="dfn",
        audio_preprocessing_id="raw_16k",
        scale_index=scale,
        window_sec=1.5,
        shift_sec=0.75,
        sample_rate=16000,
        frontend_config_version="nemo_msdd_v1",
    )


def test_cache_hit_and_miss(tmp_path):
    mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path, allow_custom_root=True)
    ident = make_sample_identity()

    # 1. Initially empty -> MISS
    res, status = mgr.load_or_status(ident, expected_dim=192)
    assert status == "MISS"
    assert res is None

    # 2. Save valid embeddings
    embs = np.ones((4, 192), dtype=np.float32)
    norms = np.ones(4, dtype=np.float32)
    starts = np.array([0.0, 0.75, 1.5, 2.25])
    ends = np.array([1.5, 2.25, 3.0, 3.75])
    mgr.save_embeddings(ident, embs, norms, starts, ends)

    # 3. Query again -> HIT
    res, status = mgr.load_or_status(ident, expected_dim=192)
    assert status == "HIT"
    assert res is not None
    loaded_embs, loaded_norms = res
    assert loaded_embs.shape == (4, 192)


def test_cache_invalidation_on_checkpoint_change(tmp_path):
    mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path, allow_custom_root=True)
    ident1 = make_sample_identity(encoder_hash="aaaa" * 16)

    # Save under ident1
    embs = np.ones((2, 192), dtype=np.float32)
    mgr.save_embeddings(
        ident1, embs, np.ones(2, dtype=np.float32), np.array([0.0, 1.0]), np.array([1.0, 2.0])
    )

    # Query under different checkpoint hash -> should be INVALIDATION because cached metadata hash mismatches
    ident2 = make_sample_identity(encoder_hash="bbbb" * 16)
    res, status = mgr.load_or_status(ident2, expected_dim=192)
    assert status == "INVALIDATION"
    assert res is None


def test_cache_invalidation_on_corrupt_file(tmp_path):
    mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path, allow_custom_root=True)
    ident = make_sample_identity()

    # Create dummy corrupt file at expected path
    entry_dir = mgr.get_entry_dir(ident.vad_config_id, ident.recording_id)
    entry_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = entry_dir / mgr.build_filename(ident)
    corrupt_file.write_bytes(b"corrupt non-zip binary junk")

    # Query -> INVALIDATION (not MISS!)
    res, status = mgr.load_or_status(ident, expected_dim=192)
    assert status == "INVALIDATION"
    assert res is None


def test_cache_security_violation_raises_immediately(tmp_path):
    mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path, allow_custom_root=True)
    
    # Path traversal in recording_id
    with pytest.raises(CacheSecurityError, match="Path traversal"):
        make_sample_identity(recording_id="../escape")
