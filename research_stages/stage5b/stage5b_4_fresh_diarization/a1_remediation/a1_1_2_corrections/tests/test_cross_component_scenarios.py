#!/usr/bin/env python3
"""
Test Suite: Cross-Component Scenarios A through F
Tests the full integration of hardened components across 6 specific end-to-end scenarios:
- Scenario A: Production checkpoint preflight (fails before model loading without verified checkpoints)
- Scenario B: Synthetic checkpoint mode (runs with synthetic mocks, cannot activate production inference)
- Scenario C: RTTM path traversal (malicious recording_id or condition_id rejected before file creation)
- Scenario D: cpWER parity on synthetic end-to-end evaluation
- Scenario E: Overlap posterior rejection (malformed, misaligned, or provenance-incomplete posterior data rejected)
- Scenario F: Valid synthetic end-to-end path (runs completely on synthetic fixtures without reading real audio, loading weights, or GPU)
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from checkpoint_preflight import (
    ExecutionMode,
    CheckpointPreflightError,
    VerifiedCheckpoint,
    run_production_checkpoint_preflight,
)
from output_path_guard import (
    validate_final_rttm_path,
    validate_recording_id,
    validate_condition_id,
    OutputSecurityError,
)
from overlap_posterior_loader import (
    OverlapPosteriorLoaderStrict,
    OverlapPosteriorError,
    PYANNET_FRAME_STEP_SEC,
    PYANNET_FIRST_FRAME_CENTER,
)
from cpwer_adapter import evaluate_cpwer, attribute_words_to_segments, rttm_to_segments
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from vad_adapter_remediated import VADAdapterRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter, ProductionEvaluationAdapterRemediated
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated, PipelineExecutionStats
from feature_extraction_interface import MockTitaNetModel, MockWeSpeakerModel


@pytest.fixture
def synthetic_orchestrator(tmp_path):
    """Sets up an orchestrator in SYNTHETIC execution mode with mock backends."""
    cache_dir = tmp_path / "cache"
    vad_dir = tmp_path / "vad"
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=cache_dir, allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=vad_dir)
    clustering_backend = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()
    overlap_loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=tmp_path / "overlap",
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    tita_model = MockTitaNetModel(return_dim=192)
    wes_model = MockWeSpeakerModel(return_dim=256)

    orch = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering_backend,
        evaluation_adapter=eval_adapter,
        overlap_loader=overlap_loader,
        titanet_model=tita_model,
        wespeaker_model=wes_model,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_allowed_output_roots=[tmp_path],
    )
    return orch, tmp_path


# =========================================================================
# Scenario A: Production checkpoint preflight
# =========================================================================

def test_scenario_a_production_preflight_fails_without_checkpoints(tmp_path):
    """Verify that in PRODUCTION mode, preflight fails immediately if checkpoints are missing or unverified."""
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path / "cache", allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=tmp_path)
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    # 1. Missing checkpoint paths in production mode raises CheckpointPreflightError
    with pytest.raises(CheckpointPreflightError, match="Production preflight failure for 'titanet_large'"):
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=clustering,
            evaluation_adapter=eval_adapter,
            execution_mode=ExecutionMode.PRODUCTION,
        )

    # 2. Checkpoint with placeholder hash raises CheckpointPreflightError
    fake_ckpt = tmp_path / "model.nemo"
    fake_ckpt.write_bytes(b"model data")
    with pytest.raises(CheckpointPreflightError, match="is an unauthorized placeholder hash"):
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=clustering,
            evaluation_adapter=eval_adapter,
            titanet_checkpoint_path=fake_ckpt,
            titanet_checkpoint_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            wespeaker_checkpoint_path=fake_ckpt,
            wespeaker_checkpoint_sha256="1111111111111111111111111111111111111111111111111111111111111111",
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_checkpoint_roots=[tmp_path],
        )

    # 3. Checkpoint with hash mismatch raises CheckpointPreflightError
    real_hash = hashlib.sha256(b"model data").hexdigest()
    wrong_hash = hashlib.sha256(b"other content").hexdigest()
    with pytest.raises(CheckpointPreflightError, match="Hash mismatch for 'titanet_large'"):
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=clustering,
            evaluation_adapter=eval_adapter,
            titanet_checkpoint_path=fake_ckpt,
            titanet_checkpoint_sha256=wrong_hash,
            wespeaker_checkpoint_path=fake_ckpt,
            wespeaker_checkpoint_sha256=real_hash,
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_checkpoint_roots=[tmp_path],
        )


# =========================================================================
# Scenario B: Synthetic checkpoint mode
# =========================================================================

def test_scenario_b_synthetic_mode_isolation(synthetic_orchestrator):
    """Verify that synthetic mode runs with mock descriptors and cannot be confused with production."""
    orch, _ = synthetic_orchestrator
    assert orch.execution_mode == ExecutionMode.SYNTHETIC
    assert orch.verified_titanet is not None
    assert orch.verified_titanet.is_synthetic is True
    assert orch.verified_wespeaker is not None
    assert orch.verified_wespeaker.is_synthetic is True

    # Attempting to use synthetic mock descriptors in production mode preflight must fail
    synth_ckpt = orch.verified_titanet
    with pytest.raises(CheckpointPreflightError, match="Preflight validation failed"):
        run_production_checkpoint_preflight(
            titanet_path=synth_ckpt.verified_path,
            titanet_expected_sha256=synth_ckpt.computed_sha256,
            wespeaker_path=synth_ckpt.verified_path,
            wespeaker_expected_sha256=synth_ckpt.computed_sha256,
        )


# =========================================================================
# Scenario C: RTTM path traversal
# =========================================================================

def test_scenario_c_path_traversal_rejection(synthetic_orchestrator):
    """Verify malicious recording_id or condition_id are rejected before any file operation."""
    orch, tmp_path = synthetic_orchestrator
    sr = 16000
    dummy_audio = np.zeros(sr * 2, dtype=np.float32)

    # 1. Directory traversal in recording_id
    with pytest.raises(OutputSecurityError, match="Illegal characters or path traversal in recording_id"):
        orch.process_recording(
            condition_id="E00",
            recording_id="../../etc/passwd",
            dfn_waveform=dummy_audio,
            mono_waveform=dummy_audio,
            output_dir=tmp_path / "out",
        )

    # 2. Path separators in condition_id
    with pytest.raises(OutputSecurityError, match="Path traversal characters detected in condition_id"):
        orch.process_recording(
            condition_id="E00/../escape",
            recording_id="R01_synthetic",
            dfn_waveform=dummy_audio,
            mono_waveform=dummy_audio,
            output_dir=tmp_path / "out",
        )

    # 3. Direct validate_final_rttm_path path traversal escape check
    with pytest.raises(OutputSecurityError, match="Illegal characters or path traversal in recording_id"):
        validate_final_rttm_path(
            output_dir=tmp_path / "out",
            recording_id="../R01_test",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_allowed_roots=[tmp_path],
        )


# =========================================================================
# Scenario D: cpWER parity on synthetic end-to-end evaluation
# =========================================================================

def test_scenario_d_cpwer_parity_end_to_end(tmp_path):
    """Verify cpWER adapter evaluates synthetic hypothesis RTTM with exact Phase 20 word attribution parity."""
    # Build hypothesis RTTM
    hyp_rttm = tmp_path / "hyp.rttm"
    with open(hyp_rttm, "w") as f:
        f.write("SPEAKER rec1 1 0.000 2.000 <NA> <NA> SPK1 <NA> <NA>\n")
        f.write("SPEAKER rec1 1 2.000 2.000 <NA> <NA> SPK2 <NA> <NA>\n")

    # Build reference aligned words
    aligned_words = [
        {"word": "hello", "start": 0.2, "end": 0.8},
        {"word": "world", "start": 1.0, "end": 1.5},
        {"word": "testing", "start": 2.2, "end": 2.8},
        {"word": "diarization", "start": 3.0, "end": 3.6},
    ]
    words_json = tmp_path / "aligned_words.json"
    with open(words_json, "w") as f:
        json.dump(aligned_words, f)

    # Build reference transcript
    ref_transcript = {
        "segments": [
            {"speaker": "REF_A", "text": "hello world"},
            {"speaker": "REF_B", "text": "testing diarization"},
        ]
    }
    transcript_json = tmp_path / "ref_transcript.json"
    with open(transcript_json, "w") as f:
        json.dump(ref_transcript, f)

    # Evaluate via cpwer_adapter directly
    metrics = evaluate_cpwer(
        hypothesis_rttm=hyp_rttm,
        aligned_words_json=words_json,
        reference_transcript_json=transcript_json,
    )

    assert "cpwer" in metrics
    assert metrics["cpwer"] == 0.0  # Perfect attribution match: SPK1 -> REF_A, SPK2 -> REF_B
    assert metrics["reference_words"] == 4
    assert metrics["hypothesis_words"] == 4

    # Now verify via ProductionEvaluationAdapterRemediated
    # Create matching reference RTTM
    ref_rttm = tmp_path / "ref.rttm"
    with open(ref_rttm, "w") as f:
        f.write("SPEAKER rec1 1 0.000 2.000 <NA> <NA> REF_A <NA> <NA>\n")
        f.write("SPEAKER rec1 1 2.000 2.000 <NA> <NA> REF_B <NA> <NA>\n")

    prod_eval = ProductionEvaluationAdapterRemediated()
    adapter_metrics = prod_eval.evaluate(
        hypothesis_rttm=hyp_rttm,
        reference_rttm=ref_rttm,
        aligned_words_json=words_json,
        reference_transcript_json=transcript_json,
        k_hat=2,
    )
    assert adapter_metrics["cpwer"] == 0.0
    assert "der_collar0" in adapter_metrics
    assert "der_collar025" in adapter_metrics


# =========================================================================
# Scenario E: Overlap posterior rejection
# =========================================================================

def test_scenario_e_overlap_posterior_rejection(synthetic_orchestrator):
    """Verify malformed or misaligned overlap posteriors are rejected before attribution."""
    orch, tmp_path = synthetic_orchestrator
    sr = 16000
    dummy_audio = np.zeros(sr * 2, dtype=np.float32)

    # 1. Stride violation in override timestamps
    bad_stride_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100, dtype=np.float64) * 0.010
    speech_probs = np.full(100, 0.9, dtype=np.float32)
    bad_ovl_probs = np.full(100, 0.1, dtype=np.float32)

    with pytest.raises(OverlapPosteriorError, match="stride error exceeds tolerance"):
        orch.process_recording(
            condition_id="E01",  # E01 uses M4 overlap
            recording_id="R01_synthetic",
            dfn_waveform=dummy_audio,
            mono_waveform=dummy_audio,
            posteriors_override=(bad_stride_times, speech_probs, bad_ovl_probs),
            expected_duration=2.0,
        )

    # 2. Probability bounds violation (>1.0)
    good_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    bad_ovl_bounds = np.full(100, 1.5, dtype=np.float32)

    with pytest.raises(OverlapPosteriorError, match="outside valid probability range"):
        orch.process_recording(
            condition_id="E01",
            recording_id="R01_synthetic",
            dfn_waveform=dummy_audio,
            mono_waveform=dummy_audio,
            posteriors_override=(good_times, speech_probs, bad_ovl_bounds),
            expected_duration=100 * PYANNET_FRAME_STEP_SEC,
        )

    # 3. M4 condition without overlap probabilities provided raises OverlapPosteriorError
    with pytest.raises(OverlapPosteriorError, match="requires M4 overlap posteriors"):
        orch.process_recording(
            condition_id="E01",
            recording_id="R01_synthetic",
            dfn_waveform=dummy_audio,
            mono_waveform=dummy_audio,
            posteriors_override=(good_times, speech_probs, None),
            expected_duration=100 * PYANNET_FRAME_STEP_SEC,
        )


# =========================================================================
# Scenario F: Valid synthetic end-to-end path
# =========================================================================

def test_scenario_f_valid_synthetic_end_to_end_path(synthetic_orchestrator):
    """Verify complete valid execution in synthetic mode on synthetic fixtures without real checkpoints or GPU."""
    orch, tmp_path = synthetic_orchestrator
    sr = 16000
    duration_sec = 4.0
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)

    # Generate distinct synthetic audio signals (sum of tones)
    dfn_audio = (np.sin(2 * np.pi * 220 * t) + 0.1 * np.random.randn(n_samples)).astype(np.float32)
    mono_audio = (np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(n_samples)).astype(np.float32)

    # Generate synthetic posteriors matching PyanNet parameters
    n_frames = int(duration_sec / PYANNET_FRAME_STEP_SEC)
    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(n_frames, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    # Active speech between 0.5s and 3.5s
    p_speech = np.zeros(n_frames, dtype=np.float32)
    speech_mask = (f_times >= 0.5) & (f_times <= 3.5)
    p_speech[speech_mask] = 0.95

    # Overlap between 1.5s and 2.5s
    p_overlap = np.zeros(n_frames, dtype=np.float32)
    ovl_mask = (f_times >= 1.5) & (f_times <= 2.5)
    p_overlap[ovl_mask] = 0.85

    out_dir = tmp_path / "output_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run Condition E01 (VAD Oracle / M4 Overlap)
    stats = orch.process_recording(
        condition_id="E01",
        recording_id="R01_synthetic",
        dfn_waveform=dfn_audio,
        mono_waveform=mono_audio,
        posteriors_override=(f_times, p_speech, p_overlap),
        sample_rate=sr,
        output_dir=out_dir,
        expected_duration=duration_sec,
    )

    assert isinstance(stats, PipelineExecutionStats)
    assert stats.condition_id == "E01"
    assert stats.recording_id == "R01_synthetic"
    assert stats.vad_interval_count > 0
    assert stats.execution_mode == "synthetic"
    assert stats.provenance_verified is True
    assert stats.output_rttm_path is not None

    final_rttm_path = Path(stats.output_rttm_path)
    assert final_rttm_path.exists()
    assert final_rttm_path.stat().st_size > 0

    # Verify RTTM content is valid
    lines = final_rttm_path.read_text().strip().splitlines()
    assert len(lines) > 0
    for line in lines:
        parts = line.split()
        assert parts[0] == "SPEAKER"
        assert parts[1] == "R01_synthetic"
        dur = float(parts[4])
        assert dur > 0.0

    # Run Condition E00 (VAD Oracle / M0 No Overlap) to verify second condition
    stats_e00 = orch.process_recording(
        condition_id="E00",
        recording_id="R01_synthetic",
        dfn_waveform=dfn_audio,
        mono_waveform=mono_audio,
        posteriors_override=(f_times, p_speech, None),
        sample_rate=sr,
        output_dir=out_dir,
        expected_duration=duration_sec,
    )
    assert stats_e00.condition_id == "E00"
    assert stats_e00.overlap_method == "M0_no_overlap"
    assert Path(stats_e00.output_rttm_path).exists()
