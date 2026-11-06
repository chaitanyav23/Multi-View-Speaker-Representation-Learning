#!/usr/bin/env python3
"""
Tests for Task 10: Synthetic End-to-End Integration Suite.
Exercises the entire fresh diarization pipeline in a controlled temporary directory:
Synthetic waveform -> VAD posteriors -> subsegmentation -> cache miss -> mock extraction
-> caching -> multiscale affinity -> Condition J fusion -> clustering -> overlap attribution
-> RTTM output -> evaluation -> second run cache hit verification.
Zero real model checkpoints or real audio accessed.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from orchestrator import FreshDiarizationOrchestrator
from isolated_embedding_cache import IsolatedEmbeddingCacheManager
from vad_adapter_remediated import VADAdapterRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters import SyntheticEvaluationAdapter
from feature_extraction_interface import MockTitaNetModel, MockWeSpeakerModel
from rttm_adapter_remediated import read_rttm_file


def create_synthetic_test_environment(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    cache_mgr = IsolatedEmbeddingCacheManager(cache_root=cache_root, allow_custom_root=True)
    vad_adapter = VADAdapterRemediated()
    clustering = SyntheticSpectralClusteringBackend(random_state=42)
    evaluation = SyntheticEvaluationAdapter()

    tita_model = MockTitaNetModel(seed=42)
    wes_model = MockWeSpeakerModel(seed=100)

    orchestrator = FreshDiarizationOrchestrator(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering,
        evaluation_adapter=evaluation,
        titanet_model=tita_model,
        wespeaker_model=wes_model,
        titanet_checkpoint_sha256="450f6bd279a3f783687dbd15814fd0f1",
        wespeaker_checkpoint_sha256="e7b8447ea84345e21af34a3985a43826",
    )
    return orchestrator, cache_root, output_dir


def test_synthetic_end_to_end_pipeline_and_cache_reuse(tmp_path):
    orchestrator, cache_root, output_dir = create_synthetic_test_environment(tmp_path)

    # 1. Generate synthetic waveforms (5 seconds = 80,000 samples @ 16kHz)
    rng = np.random.default_rng(42)
    dfn_wav = rng.standard_normal(80000).astype(np.float32)
    mono_wav = rng.standard_normal(80000).astype(np.float32)

    # 2. Synthetic posteriors: 2 speech turns: [0.5s - 2.0s] and [2.5s - 4.5s]
    # Frame resolution: 0.02s (250 frames total)
    f_times = np.linspace(0.01, 5.0, 250)
    p_speech = np.zeros(250, dtype=np.float32)
    p_speech[25:100] = 0.90   # 0.5s - 2.0s
    p_speech[125:225] = 0.90  # 2.5s - 4.5s

    p_ovl = np.zeros(250, dtype=np.float32)
    p_ovl[50:60] = 0.80       # Short overlap event

    posteriors = (f_times, p_speech, p_ovl)

    # Reference RTTM for evaluation
    ref_rttm = tmp_path / "ref.rttm"
    ref_rttm.write_text("SPEAKER syn_rec 1 0.50 1.50 <NA> <NA> ref_spk0 <NA> <NA>\n")

    # =========================================================================
    # RUN 1: Cache Miss on both encoders
    # =========================================================================
    stats_run1 = orchestrator.process_recording(
        condition_id="E00",
        recording_id="syn_rec",
        dfn_waveform=dfn_wav,
        mono_waveform=mono_wav,
        posteriors_override=posteriors,
        sample_rate=16000,
        output_dir=output_dir,
        reference_rttm_path=ref_rttm,
    )

    # All 5 scales should produce cache misses for both encoders
    assert stats_run1.titanet_cache_misses == 5
    assert stats_run1.wespeaker_cache_misses == 5
    assert stats_run1.titanet_cache_hits == 0
    assert stats_run1.wespeaker_cache_hits == 0
    assert stats_run1.extracted_embeddings > 0
    assert stats_run1.output_rttm_path is not None
    assert Path(stats_run1.output_rttm_path).exists()

    rttm_segs_run1 = read_rttm_file(Path(stats_run1.output_rttm_path))
    assert len(rttm_segs_run1) > 0

    # =========================================================================
    # RUN 2: Cache Hit on both encoders (Identical config)
    # =========================================================================
    stats_run2 = orchestrator.process_recording(
        condition_id="E00",
        recording_id="syn_rec",
        dfn_waveform=dfn_wav,
        mono_waveform=mono_wav,
        posteriors_override=posteriors,
        sample_rate=16000,
        output_dir=output_dir,
        reference_rttm_path=ref_rttm,
    )

    # Run 2 must have 100% cache hits and ZERO newly extracted embeddings
    assert stats_run2.titanet_cache_hits == 5
    assert stats_run2.wespeaker_cache_hits == 5
    assert stats_run2.titanet_cache_misses == 0
    assert stats_run2.wespeaker_cache_misses == 0
    assert stats_run2.extracted_embeddings == 0

    # Output RTTM must be identical
    rttm_segs_run2 = read_rttm_file(Path(stats_run2.output_rttm_path))
    assert len(rttm_segs_run1) == len(rttm_segs_run2)
    for s1, s2 in zip(rttm_segs_run1, rttm_segs_run2):
        assert s1["start"] == s2["start"]
        assert s1["end"] == s2["end"]
        assert s1["speaker"] == s2["speaker"]


def test_cache_invalidation_on_changed_checkpoint_hash(tmp_path):
    orchestrator, cache_root, output_dir = create_synthetic_test_environment(tmp_path)

    rng = np.random.default_rng(42)
    dfn_wav = rng.standard_normal(48000).astype(np.float32)
    mono_wav = rng.standard_normal(48000).astype(np.float32)

    f_times = np.linspace(0.01, 3.0, 150)
    p_speech = np.zeros(150, dtype=np.float32)
    p_speech[20:120] = 0.90
    posteriors = (f_times, p_speech, np.zeros(150, dtype=np.float32))

    # Populate cache with original hash
    orchestrator.process_recording(
        condition_id="E00",
        recording_id="rec_inval",
        dfn_waveform=dfn_wav,
        mono_waveform=mono_wav,
        posteriors_override=posteriors,
        output_dir=output_dir,
    )

    # Change TitaNet checkpoint hash in orchestrator
    orchestrator.titanet_ckpt_hash = "ffffffffffffffffffffffffffffffff"

    stats_changed = orchestrator.process_recording(
        condition_id="E00",
        recording_id="rec_inval",
        dfn_waveform=dfn_wav,
        mono_waveform=mono_wav,
        posteriors_override=posteriors,
        output_dir=output_dir,
    )

    # TitaNet cache entries must mismatch and be re-extracted (5 misses)
    assert stats_changed.titanet_cache_misses == 5
    # WeSpeaker was unchanged, so it should hit the cache!
    assert stats_changed.wespeaker_cache_hits == 5
