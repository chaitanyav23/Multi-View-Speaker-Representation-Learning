#!/usr/bin/env python3
"""
Test Suite: Pipeline Validation State Propagation (Correction 4)
Validates that:
- structurally_valid, provenance_verified, and production_eligible remain distinct throughout the pipeline.
- Downstream components (orchestrator, overlap attribution, execution stats) inspect production_eligible.
- M0 runs without requiring overlap posterior provenance.
- M4 fails closed if production_eligible is False.
- Synthetic fixtures cannot be promoted to production_eligible.
- Unrecorded fields (such as mono source audio hash) fail-close production M4 while preserving structural validity.
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

from path_utils import get_stage5b_4_dir
A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    verify_encoder_checkpoint_strict,
    create_synthetic_checkpoint_fixture,
    CheckpointPreflightError,
)
from datetime import datetime, timezone

from model_factory import (
    ControlledModelFactory,
    VerifiedModelHandle,
    ModelFactoryError,
    _generate_handle_signature,
)
from overlap_provenance import (
    PYANNET_FRAME_STEP_SEC,
    PYANNET_FIRST_FRAME_CENTER,
    PYANNET_SAMPLE_RATE,
    PYANNET_MODEL_CHECKPOINT_SHA256,
    OverlapProvenanceError,
    StructuralValidationState,
    ProvenanceValidationState,
    OverallValidationState,
    ValidatedPosteriorContainer,
    OverlapProvenanceVerifier,
)
from overlap_posterior_loader import (
    OverlapPosteriorLoaderStrict,
    validate_frame_timestamps_strict,
    validate_posterior_values_strict,
)
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter
from vad_adapter_remediated import VADAdapterRemediated
from orchestrator_remediated import (
    FreshDiarizationOrchestratorRemediated,
    PipelineExecutionStats,
)
from feature_extraction_interface import MockTitaNetModel, MockWeSpeakerModel


def _make_mock_posterior_npz(path: Path, n_frames: int = 100):
    times = PYANNET_FIRST_FRAME_CENTER + np.arange(n_frames, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    probs = (np.sin(np.linspace(0, np.pi, n_frames)) ** 2).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_times=times, p_overlap=probs)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    return times, probs, sha, size


@pytest.fixture
def test_env(tmp_path):
    env_dir = tmp_path / "pipeline_test_env"
    env_dir.mkdir()
    cache_dir = env_dir / "cache"
    cache_dir.mkdir()
    outputs_dir = env_dir / "outputs"
    outputs_dir.mkdir()
    manifest_dir = env_dir / "manifests"
    manifest_dir.mkdir()

    rec_id = "R01_mock"
    dfn_path = cache_dir / f"{rec_id}_dfn_overlap.npz"
    mono_path = cache_dir / f"{rec_id}_mono_overlap.npz"
    times, probs, dfn_sha, dfn_size = _make_mock_posterior_npz(dfn_path, n_frames=120)
    _, _, mono_sha, mono_size = _make_mock_posterior_npz(mono_path, n_frames=120)

    # Valid manifest
    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            f"{rec_id}__dfn": {
                "recording_id": rec_id,
                "stream_name": "dfn",
                "posterior_file_rel": str(dfn_path),
                "posterior_file_sha256": dfn_sha,
                "posterior_file_size_bytes": dfn_size,
                "source_audio_rel": f"audio/{rec_id}_dfn.wav",
                "source_audio_sha256": hashlib.sha256(b"authoritative_source_audio_mock").hexdigest(),
                "source_sample_rate": 16000,
                "source_duration_sec": float(times[-1] + PYANNET_FRAME_STEP_SEC / 2),
                "model_architecture": "pyannote/segmentation",
                "model_checkpoint_sha256": PYANNET_MODEL_CHECKPOINT_SHA256,
                "generation_config_id": "pyannote_3_1_continuous_top2",
                "timestamp_convention": "sliding_window_middle_centering",
                "frame_step_sec": PYANNET_FRAME_STEP_SEC,
                "first_frame_center_sec": PYANNET_FIRST_FRAME_CENTER,
                "posterior_key": "p_overlap",
                "posterior_semantics": "continuous_top2_second_highest",
                "unverified_fields": [],
            },
            f"{rec_id}__mono": {
                "recording_id": rec_id,
                "stream_name": "mono",
                "posterior_file_rel": str(mono_path),
                "posterior_file_sha256": mono_sha,
                "posterior_file_size_bytes": mono_size,
                "source_audio_rel": f"audio/{rec_id}_mono.wav",
                "source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST",
                "source_sample_rate": 16000,
                "source_duration_sec": float(times[-1] + PYANNET_FRAME_STEP_SEC / 2),
                "model_architecture": "pyannote/segmentation",
                "model_checkpoint_sha256": PYANNET_MODEL_CHECKPOINT_SHA256,
                "generation_config_id": "pyannote_3_1_continuous_top2",
                "timestamp_convention": "sliding_window_middle_centering",
                "frame_step_sec": PYANNET_FRAME_STEP_SEC,
                "first_frame_center_sec": PYANNET_FIRST_FRAME_CENTER,
                "posterior_key": "p_overlap",
                "posterior_semantics": "continuous_top2_second_highest",
                "unverified_fields": ["source_audio_sha256"],
            },
        },
    }
    manifest_file = manifest_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    return {
        "env_dir": env_dir,
        "cache_dir": cache_dir,
        "outputs_dir": outputs_dir,
        "manifest_file": manifest_file,
        "rec_id": rec_id,
        "times": times,
        "probs": probs,
        "dfn_path": dfn_path,
        "mono_path": mono_path,
    }


def test_1_state_separation_dataclasses():
    """Verify that structural, provenance, and overall states are distinctly represented."""
    s_state = StructuralValidationState(
        structurally_valid=True,
        first_frame_sec=0.03096875,
        last_frame_sec=1.70159375,
        frame_step_sec=PYANNET_FRAME_STEP_SEC,
        frame_count=100,
        min_probability=0.0,
        max_probability=1.0,
        validation_errors=[],
    )
    p_state = ProvenanceValidationState(
        provenance_verified=False,
        recording_id="R01",
        stream_name="mono",
        posterior_file=Path("/tmp/test.npz"),
        posterior_sha256="abc",
        file_size_bytes=100,
        source_audio_file=None,
        source_audio_sha256=None,
        source_duration_sec=None,
        source_sample_rate=16000,
        model_architecture="pyannote/segmentation",
        model_checkpoint_sha256=None,
        generation_config_id=None,
        timestamp_convention="sliding_window_middle_centering",
        unverified_fields=["source_audio_sha256"],
        provenance_errors=[],
    )
    reasons = []
    if not p_state.provenance_verified:
        reasons.append(f"Unverified provenance fields: {p_state.unverified_fields}")
    overall = OverallValidationState(
        structurally_valid=s_state.structurally_valid,
        provenance_verified=p_state.provenance_verified,
        production_eligible=False,
        is_synthetic=False,
        reasons=reasons,
    )

    assert overall.structurally_valid is True
    assert overall.provenance_verified is False
    assert overall.production_eligible is False
    assert "Unverified provenance fields: ['source_audio_sha256']" in overall.reasons


def test_2_synthetic_fixture_cannot_be_production_eligible(test_env):
    """Synthetic fixtures are structurally valid but never provenance_verified or production_eligible."""
    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=test_env["cache_dir"],
        execution_mode=ExecutionMode.SYNTHETIC,
        provenance_manifest_path=test_env["manifest_file"],
    )
    container = loader.load_and_validate(
        recording_id=test_env["rec_id"],
        stream="dfn",
        override_arrays=(test_env["times"], test_env["probs"]),
        is_synthetic=True,
    )
    assert container.structural_state.structurally_valid is True
    assert container.provenance_state.provenance_verified is False
    assert container.overall_state.production_eligible is False
    assert container.is_synthetic is True


def test_3_unrecorded_mono_source_audio_fails_production_eligibility(test_env):
    """Verify that unrecorded mono source audio hash fails closed for production eligibility."""
    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=test_env["cache_dir"],
        execution_mode=ExecutionMode.PRODUCTION,
        provenance_manifest_path=test_env["manifest_file"],
    )
    container = loader.load_and_validate(
        recording_id=test_env["rec_id"],
        stream="mono",
    )
    assert container.structural_state.structurally_valid is True
    assert container.provenance_state.provenance_verified is False
    assert "source_audio_sha256" in container.provenance_state.unverified_fields
    assert container.overall_state.production_eligible is False


def test_4_orchestrator_preserves_distinct_states_in_stats(test_env):
    """Orchestrator records structurally_valid, provenance_verified, and production_eligible in stats."""
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(test_env["cache_dir"] / "embs", allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=test_env["cache_dir"] / "vad")
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    # Synthetic orchestrator
    tita_ckpt = create_synthetic_checkpoint_fixture("titanet_large", test_env["env_dir"] / "tita.nemo")
    wes_ckpt = create_synthetic_checkpoint_fixture("wespeaker_resnet293", test_env["env_dir"] / "wes.onnx")
    tita_handle = ControlledModelFactory.build_synthetic_model(tita_ckpt, MockTitaNetModel())
    wes_handle = ControlledModelFactory.build_synthetic_model(wes_ckpt, MockWeSpeakerModel())

    orch = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering,
        evaluation_adapter=eval_adapter,
        titanet_handle=tita_handle,
        wespeaker_handle=wes_handle,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=test_env["outputs_dir"],
    )

    # 1. Test Condition M0 (overlap_method == M0_no_overlap)
    # Synthetic multi-tone waveform
    audio_len = 16000 * 3  # 3 seconds
    wav = np.sin(2 * np.pi * 440 * np.linspace(0, 3, audio_len)).astype(np.float32)

    speech_probs = np.ones(len(test_env["times"]), dtype=np.float32)
    posteriors_override = (test_env["times"], speech_probs, None)

    stats_m0 = orch.process_recording(
        condition_id="E00",
        recording_id=test_env["rec_id"],
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=posteriors_override,
        expected_duration=None,
    )

    # M0 succeeds without overlap posterior provenance
    assert stats_m0.overlap_method == "M0_no_overlap"
    assert stats_m0.structurally_valid is True
    assert stats_m0.provenance_verified is False
    assert stats_m0.production_eligible is False
    assert stats_m0.output_rttm_path is not None
    assert Path(stats_m0.output_rttm_path).exists()

    # 2. Test Condition M4 in Synthetic Mode
    posteriors_override_m4 = (test_env["times"], speech_probs, test_env["probs"])
    stats_m4 = orch.process_recording(
        condition_id="E21",
        recording_id=test_env["rec_id"],
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=posteriors_override_m4,
        expected_duration=None,
    )
    # Synthetic M4 allows execution for testing but reflects correct flags
    assert stats_m4.overlap_method == "M4_conservative_abstention"
    assert stats_m4.structurally_valid is True
    assert stats_m4.provenance_verified is False
    assert stats_m4.production_eligible is False


def test_5_production_m4_fails_closed_on_unverified_posterior(test_env):
    """In PRODUCTION execution mode, Condition M4 strictly refuses to run on unverified posterior."""
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(test_env["cache_dir"] / "embs_prod", allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=test_env["cache_dir"] / "vad_prod")
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    # Create dummy real checkpoint files in permitted root
    permitted_root = test_env["env_dir"] / "checkpoints"
    permitted_root.mkdir()
    tita_path = permitted_root / "titanet_production.nemo"
    wes_path = permitted_root / "wespeaker_production.onnx"
    tita_path.write_bytes(b"A" * 64)
    wes_path.write_bytes(b"B" * 64)

    tita_sha = hashlib.sha256(tita_path.read_bytes()).hexdigest()
    wes_sha = hashlib.sha256(wes_path.read_bytes()).hexdigest()

    # Use verify_encoder_checkpoint_strict with custom_permitted_roots
    v_tita = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=tita_path,
        expected_sha256=tita_sha,
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_roots=[permitted_root],
    )
    v_wes = verify_encoder_checkpoint_strict(
        encoder_name="wespeaker_resnet293",
        checkpoint_path=wes_path,
        expected_sha256=wes_sha,
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_roots=[permitted_root],
    )

    # To avoid loading real model weights, create handle directly with mock model for unit test
    # but with PRODUCTION execution mode and signature
    ts = datetime.now(timezone.utc).isoformat()
    sig_tita = _generate_handle_signature(
        encoder_name="titanet_large",
        checkpoint_sha256=v_tita.computed_sha256,
        creation_timestamp=ts,
        is_synthetic=False,
    )
    tita_handle = VerifiedModelHandle(
        encoder_name="titanet_large",
        verified_checkpoint=v_tita,
        model_instance=MockTitaNetModel(),
        creation_timestamp=ts,
        is_synthetic=False,
        _handle_signature=sig_tita,
    )
    sig_wes = _generate_handle_signature(
        encoder_name="wespeaker_resnet293",
        checkpoint_sha256=v_wes.computed_sha256,
        creation_timestamp=ts,
        is_synthetic=False,
    )
    wes_handle = VerifiedModelHandle(
        encoder_name="wespeaker_resnet293",
        verified_checkpoint=v_wes,
        model_instance=MockWeSpeakerModel(),
        creation_timestamp=ts,
        is_synthetic=False,
        _handle_signature=sig_wes,
    )

    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=test_env["cache_dir"],
        execution_mode=ExecutionMode.PRODUCTION,
        provenance_manifest_path=test_env["manifest_file"],
    )

    orch = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering,
        evaluation_adapter=eval_adapter,
        overlap_loader=loader,
        titanet_handle=tita_handle,
        wespeaker_handle=wes_handle,
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_checkpoint_roots=[permitted_root],
    )

    audio_len = 16000 * 3
    wav = np.sin(2 * np.pi * 440 * np.linspace(0, 3, audio_len)).astype(np.float32)
    speech_probs = np.ones(len(test_env["times"]), dtype=np.float32)
    # Synthetic override posterior has is_synthetic=True -> production_eligible=False
    posteriors_override = (test_env["times"], speech_probs, test_env["probs"])

    with pytest.raises(OverlapProvenanceError) as exc_info:
        orch.process_recording(
            condition_id="E21",
            recording_id=test_env["rec_id"],
            dfn_waveform=wav,
            mono_waveform=wav,
            posteriors_override=posteriors_override,
            expected_duration=None,
        )

    assert "Posterior is not production eligible" in str(exc_info.value)
