#!/usr/bin/env python3
"""Tests for Stage 5B.4 VAD-Aware Isolated Embedding Cache."""

import pytest
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from embedding_cache import (
    CacheIdentity,
    EmbeddingCacheManager,
    CACHE_SCHEMA_VERSION,
    PROHIBITED_STAGE3_CACHE,
)
from embedding_extractor import generate_synthetic_embeddings


def create_mock_identity(
    vad_id: str = "C62",
    scale_idx: int = 4,
    vad_hash: str = "abcdef0123456789",
    ckpt_hash: str = "450f6bd279a3f783",
) -> CacheIdentity:
    return CacheIdentity(
        schema_version=CACHE_SCHEMA_VERSION,
        recording_id="R01_test",
        vad_config_id=vad_id,
        vad_intervals_sha256=vad_hash,
        encoder_name="titanet_large",
        encoder_checkpoint_sha256=ckpt_hash,
        audio_source_id="dfn",
        audio_preprocessing_id="raw_16k",
        scale_index=scale_idx,
        window_sec=0.50,
        shift_sec=0.25,
        sample_rate=16000,
        frontend_config_version="nemo_msdd_v1",
    )


def test_cache_hit_on_same_identity(tmp_path):
    mgr = EmbeddingCacheManager(cache_root=tmp_path / "cache")
    identity = create_mock_identity()

    n_subs = 10
    embs, norms = generate_synthetic_embeddings(n_subs, "titanet_large")
    starts = np.linspace(0.0, 5.0, n_subs)
    ends = starts + 0.50

    saved_path = mgr.save_embeddings(identity, embs, norms, starts, ends)
    assert saved_path.exists()

    expected_subs = [{"start": starts[i], "end": ends[i]} for i in range(n_subs)]
    loaded_embs, loaded_norms = mgr.load_embeddings(
        identity, expected_dim=192, expected_subsegments=expected_subs
    )

    assert loaded_embs.shape == (10, 192)
    assert np.allclose(loaded_embs, embs)
    assert np.allclose(loaded_norms, norms)


def test_cache_invalidation_on_altered_identity(tmp_path):
    mgr = EmbeddingCacheManager(cache_root=tmp_path / "cache")
    id1 = create_mock_identity(vad_id="C00", vad_hash="1111111111111111")
    id2 = create_mock_identity(vad_id="C50", vad_hash="2222222222222222")

    embs, norms = generate_synthetic_embeddings(5, "titanet_large")
    starts = np.linspace(0.0, 2.0, 5)
    ends = starts + 0.50

    mgr.save_embeddings(id1, embs, norms, starts, ends)

    # Looking up id2 must raise FileNotFoundError (distinct directory and filename)
    with pytest.raises(FileNotFoundError):
        mgr.load_embeddings(id2, expected_dim=192)


def test_cache_invalidation_on_checkpoint_hash_change(tmp_path):
    mgr = EmbeddingCacheManager(cache_root=tmp_path / "cache")
    id1 = create_mock_identity(ckpt_hash="hash_v1")
    id2 = create_mock_identity(ckpt_hash="hash_v2")

    embs, norms = generate_synthetic_embeddings(5, "titanet_large")
    starts = np.linspace(0.0, 2.0, 5)
    ends = starts + 0.50

    mgr.save_embeddings(id1, embs, norms, starts, ends)

    # Directly pointing to file with mismatched metadata identity
    # Modify metadata inside npz to simulate stale checkpoint
    file_path = mgr.get_entry_dir(id1.vad_config_id, id1.recording_id) / mgr.build_filename(id1)
    # Trying to load id2 against id1 path
    with pytest.raises(ValueError, match="Cache identity mismatch"):
        mgr.load_embeddings(id2, expected_dim=192)


def test_rejection_of_nan_and_inf(tmp_path):
    mgr = EmbeddingCacheManager(cache_root=tmp_path / "cache")
    identity = create_mock_identity()

    embs, norms = generate_synthetic_embeddings(5, "titanet_large")
    embs[2, 50] = np.nan
    starts = np.linspace(0.0, 2.0, 5)
    ends = starts + 0.50

    with pytest.raises(ValueError, match="NaN or Inf"):
        mgr.save_embeddings(identity, embs, norms, starts, ends)


def test_rejection_of_wrong_embedding_dimension(tmp_path):
    mgr = EmbeddingCacheManager(cache_root=tmp_path / "cache")
    identity = create_mock_identity()

    embs, norms = generate_synthetic_embeddings(5, "titanet_large")
    starts = np.linspace(0.0, 2.0, 5)
    ends = starts + 0.50

    mgr.save_embeddings(identity, embs, norms, starts, ends)

    # Requesting dim=256 on a 192d embedding matrix
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        mgr.load_embeddings(identity, expected_dim=256)


def test_prohibited_stage3_cache_rejection():
    with pytest.raises(PermissionError, match="prohibited Stage 3 cache"):
        EmbeddingCacheManager(cache_root=PROHIBITED_STAGE3_CACHE)
