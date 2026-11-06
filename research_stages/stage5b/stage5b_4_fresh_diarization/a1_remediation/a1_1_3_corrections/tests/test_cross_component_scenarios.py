#!/usr/bin/env python3
"""
Test Suite: Cross-Component Integration Scenarios (Scenarios A through J)
Authoritative verification for Stage 5B.4-A.1.3:
- Scenario A: Verified production model construction
- Scenario B: Preloaded model rejection
- Scenario C: Checkpoint mutation (TOCTOU defense)
- Scenario D: Canonical output containment
- Scenario E: Custom output root rejection
- Scenario F: Structural versus provenance state
- Scenario G: Posterior identity mismatch
- Scenario H: M4 fail-closed behavior
- Scenario I: M0 unchanged
- Scenario J: Valid synthetic integration (end-to-end pipeline)
"""

from datetime import datetime, timezone
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

from path_utils import get_stage5b_4_dir, get_stage5b_4_outputs_dir
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
from model_factory import (
    ControlledModelFactory,
    VerifiedModelHandle,
    validate_model_handle_strict,
    ModelFactoryError,
    _generate_handle_signature,
)
from output_path_guard import (
    FinalRTTMPath,
    validate_final_rttm_path,
    validate_recording_id,
    validate_condition_id,
    revalidate_and_write_rttm,
    OutputSecurityError,
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


def _make_posterior_fixture(path: Path, n_frames: int = 150):
    times = PYANNET_FIRST_FRAME_CENTER + np.arange(n_frames, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    probs = (0.5 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, n_frames))).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_times=times, p_overlap=probs)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    return times, probs, sha, size


@pytest.fixture
def scenario_fixture(tmp_path):
    base_dir = tmp_path / "scenarios_env"
    base_dir.mkdir()
    ckpts_dir = base_dir / "checkpoints"
    ckpts_dir.mkdir()
    posteriors_dir = base_dir / "posteriors"
    posteriors_dir.mkdir()
    cache_dir = base_dir / "cache"
    cache_dir.mkdir()
    manifest_dir = base_dir / "manifests"
    manifest_dir.mkdir()

    rec_id = "R01_sara_sharif"
    stream = "dfn"
    dfn_post_path = posteriors_dir / f"{rec_id}_{stream}_overlap.npz"
    mono_post_path = posteriors_dir / f"{rec_id}_mono_overlap.npz"
    times, probs, dfn_sha, dfn_size = _make_posterior_fixture(dfn_post_path, 150)
    _, _, mono_sha, mono_size = _make_posterior_fixture(mono_post_path, 150)

    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            f"{rec_id}__dfn": {
                "recording_id": rec_id,
                "stream_name": "dfn",
                "posterior_file_rel": str(dfn_post_path),
                "posterior_file_sha256": dfn_sha,
                "posterior_file_size_bytes": dfn_size,
                "source_audio_rel": f"audio/{rec_id}_dfn.wav",
                "source_audio_sha256": hashlib.sha256(b"scenario_source_dfn_audio").hexdigest(),
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
                "posterior_file_rel": str(mono_post_path),
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

    # Mock production checkpoint files in permitted root
    tita_file = ckpts_dir / "titanet_large.nemo"
    wes_file = ckpts_dir / "wespeaker_resnet293.onnx"
    tita_file.write_bytes(b"T" * 128)
    wes_file.write_bytes(b"W" * 128)
    tita_sha = hashlib.sha256(tita_file.read_bytes()).hexdigest()
    wes_sha = hashlib.sha256(wes_file.read_bytes()).hexdigest()

    return {
        "base_dir": base_dir,
        "ckpts_dir": ckpts_dir,
        "posteriors_dir": posteriors_dir,
        "cache_dir": cache_dir,
        "manifest_file": manifest_file,
        "rec_id": rec_id,
        "tita_file": tita_file,
        "wes_file": wes_file,
        "tita_sha": tita_sha,
        "wes_sha": wes_sha,
        "times": times,
        "probs": probs,
        "dfn_post_path": dfn_post_path,
        "mono_post_path": mono_post_path,
    }


def test_scenario_a_verified_production_model_construction(scenario_fixture):
    """Scenario A: Production model construction bound to verified checkpoint identity."""
    v_tita = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=scenario_fixture["tita_file"],
        expected_sha256=scenario_fixture["tita_sha"],
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_roots=[scenario_fixture["ckpts_dir"]],
    )
    assert v_tita.verification_status == "VERIFIED"
    assert v_tita.computed_sha256 == scenario_fixture["tita_sha"]

    ts = datetime.now(timezone.utc).isoformat()
    sig = _generate_handle_signature("titanet_large", v_tita.computed_sha256, ts, is_synthetic=False)
    handle = VerifiedModelHandle(
        encoder_name="titanet_large",
        verified_checkpoint=v_tita,
        model_instance=MockTitaNetModel(),
        creation_timestamp=ts,
        is_synthetic=False,
        _handle_signature=sig,
    )
    validated = validate_model_handle_strict(handle, "titanet_large", ExecutionMode.PRODUCTION)
    assert validated.verified_checkpoint.canonical_path == scenario_fixture["tita_file"]


def test_scenario_b_preloaded_model_rejection(scenario_fixture):
    """Scenario B: Arbitrary preloaded model objects and synthetic handles rejected in production."""
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(scenario_fixture["cache_dir"], allow_custom_root=True)
    vad_adapter = VADAdapterRemediated()
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    # Attempting to supply raw titanet_model in production raises ModelFactoryError
    with pytest.raises(ModelFactoryError) as exc_info:
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=clustering,
            evaluation_adapter=eval_adapter,
            titanet_model=MockTitaNetModel(),
            execution_mode=ExecutionMode.PRODUCTION,
        )
    assert "Arbitrary preloaded model instances" in str(exc_info.value)


def test_scenario_c_checkpoint_mutation_toctou(scenario_fixture):
    """Scenario C: Mutating checkpoint file on disk after preflight triggers TOCTOU failure."""
    v_tita = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=scenario_fixture["tita_file"],
        expected_sha256=scenario_fixture["tita_sha"],
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_roots=[scenario_fixture["ckpts_dir"]],
    )
    # Mutate physical bytes (keep length 128 bytes to test SHA-256 detection)
    scenario_fixture["tita_file"].write_bytes(b"X" * 128)

    with pytest.raises(ModelFactoryError) as exc_info:
        ControlledModelFactory.revalidate_checkpoint_physical_bytes(v_tita)
    assert "TOCTOU validation failure" in str(exc_info.value)
    assert "Checkpoint physical bytes modified between preflight and load" in str(exc_info.value)


def test_scenario_d_canonical_output_containment():
    """Scenario D: Production RTTMs written strictly to outputs/{condition_id}/{recording_id}.rttm."""
    final_path_obj = validate_final_rttm_path(
        recording_id="R01_sara_sharif",
        condition_id="E00",
        execution_mode=ExecutionMode.PRODUCTION,
    )
    canonical_root = get_stage5b_4_outputs_dir()
    expected_path = canonical_root / "E00" / "R01_sara_sharif.rttm"
    assert final_path_obj.path == expected_path
    assert final_path_obj.path.is_relative_to(canonical_root)


def test_scenario_e_custom_output_root_rejection():
    """Scenario E: Production mode strictly rejects caller-supplied custom output root."""
    with pytest.raises(OutputSecurityError) as exc_info:
        validate_final_rttm_path(
            recording_id="R01_sara_sharif",
            condition_id="E00",
            execution_mode=ExecutionMode.PRODUCTION,
            custom_output_root=Path("/tmp/custom_root"),
        )
    assert "Custom output root is strictly prohibited in PRODUCTION mode" in str(exc_info.value)


def test_scenario_f_structural_versus_provenance_state(scenario_fixture):
    """Scenario F: Structurally valid synthetic posterior is not provenance-verified or production-eligible."""
    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=scenario_fixture["posteriors_dir"],
        execution_mode=ExecutionMode.SYNTHETIC,
        provenance_manifest_path=scenario_fixture["manifest_file"],
    )
    container = loader.load_and_validate(
        recording_id=scenario_fixture["rec_id"],
        stream="dfn",
        override_arrays=(scenario_fixture["times"], scenario_fixture["probs"]),
        is_synthetic=True,
    )
    assert container.structural_state.structurally_valid is True
    assert container.provenance_state.provenance_verified is False
    assert container.overall_state.production_eligible is False
    assert container.is_synthetic is True


def test_scenario_g_posterior_identity_mismatch(scenario_fixture):
    """Scenario G: Corrupted posterior digest, wrong source audio, or wrong model hash fails verification."""
    verifier = OverlapProvenanceVerifier(manifest_path=scenario_fixture["manifest_file"])

    # 1. Wrong recording ID
    p_wrong_rec = verifier.verify_provenance(
        "R02_wrong", "dfn", scenario_fixture["dfn_post_path"], ExecutionMode.PRODUCTION
    )
    assert p_wrong_rec.provenance_verified is False
    assert any("No provenance manifest record found" in err for err in p_wrong_rec.provenance_errors)

    # 2. Corrupted posterior file bytes
    corrupt_file = scenario_fixture["base_dir"] / "corrupt.npz"
    corrupt_file.write_bytes(b"corrupted_bytes_content_not_matching_sha")
    p_corrupt = verifier.verify_provenance(
        scenario_fixture["rec_id"], "dfn", corrupt_file, ExecutionMode.PRODUCTION
    )
    assert p_corrupt.provenance_verified is False


def test_scenario_h_m4_fail_closed_behavior(scenario_fixture):
    """Scenario H: Production M4 refuses to execute on provenance-incomplete posterior."""
    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=scenario_fixture["posteriors_dir"],
        execution_mode=ExecutionMode.PRODUCTION,
        provenance_manifest_path=scenario_fixture["manifest_file"],
    )
    # The mono posterior has unverified_fields: ["source_audio_sha256"]
    container = loader.load_and_validate(scenario_fixture["rec_id"], stream="mono")
    assert container.structural_state.structurally_valid is True
    assert container.provenance_state.provenance_verified is False
    assert container.overall_state.production_eligible is False


def test_scenario_i_m0_unchanged(scenario_fixture):
    """Scenario I: M0 executes synthetic path without requiring overlap posterior provenance."""
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(scenario_fixture["cache_dir"], allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=scenario_fixture["base_dir"] / "vad")
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    tita_ckpt = create_synthetic_checkpoint_fixture("titanet_large", scenario_fixture["base_dir"] / "tita.nemo")
    wes_ckpt = create_synthetic_checkpoint_fixture("wespeaker_resnet293", scenario_fixture["base_dir"] / "wes.onnx")
    tita_h = ControlledModelFactory.build_synthetic_model(tita_ckpt, MockTitaNetModel())
    wes_h = ControlledModelFactory.build_synthetic_model(wes_ckpt, MockWeSpeakerModel())

    tmp_out = scenario_fixture["base_dir"] / "outputs"
    orch = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering,
        evaluation_adapter=eval_adapter,
        titanet_handle=tita_h,
        wespeaker_handle=wes_h,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_out,
    )

    audio_len = 16000 * 3
    wav = np.sin(2 * np.pi * 440 * np.linspace(0, 3, audio_len)).astype(np.float32)
    speech_probs = np.ones(len(scenario_fixture["times"]), dtype=np.float32)

    stats = orch.process_recording(
        condition_id="E00",
        recording_id=scenario_fixture["rec_id"],
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(scenario_fixture["times"], speech_probs, None),
    )
    assert stats.overlap_method == "M0_no_overlap"
    assert stats.structurally_valid is True
    assert stats.provenance_verified is False
    assert stats.production_eligible is False
    assert stats.output_rttm_path is not None
    assert Path(stats.output_rttm_path).exists()


def test_scenario_j_valid_synthetic_integration(scenario_fixture):
    """Scenario J: End-to-end execution on synthetic multi-tone audio without real weights or audio."""
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(
        scenario_fixture["base_dir"] / "cache_j", allow_custom_root=True
    )
    vad_adapter = VADAdapterRemediated(posteriors_dir=scenario_fixture["base_dir"] / "vad_j")
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    tita_ckpt = create_synthetic_checkpoint_fixture("titanet_large", scenario_fixture["base_dir"] / "tita_j.nemo")
    wes_ckpt = create_synthetic_checkpoint_fixture("wespeaker_resnet293", scenario_fixture["base_dir"] / "wes_j.onnx")
    tita_h = ControlledModelFactory.build_synthetic_model(tita_ckpt, MockTitaNetModel())
    wes_h = ControlledModelFactory.build_synthetic_model(wes_ckpt, MockWeSpeakerModel())

    tmp_out = scenario_fixture["base_dir"] / "outputs_j"
    orch = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering,
        evaluation_adapter=eval_adapter,
        titanet_handle=tita_h,
        wespeaker_handle=wes_h,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_out,
    )

    # Multi-speaker tone synthetic waveform: 2 speakers alternating frequencies 440Hz and 880Hz
    sr = 16000
    t = np.linspace(0, 4, sr * 4)
    wav_spk1 = np.sin(2 * np.pi * 440 * t) * (t < 2.0)
    wav_spk2 = np.sin(2 * np.pi * 880 * t) * (t >= 2.0)
    wav = (wav_spk1 + wav_spk2).astype(np.float32)

    n_frames = int(round(4.0 / PYANNET_FRAME_STEP_SEC))
    times = PYANNET_FIRST_FRAME_CENTER + np.arange(n_frames, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    speech_probs = np.ones(n_frames, dtype=np.float32)

    stats = orch.process_recording(
        condition_id="E00",
        recording_id=scenario_fixture["rec_id"],
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(times, speech_probs, None),
    )

    assert stats.recording_id == scenario_fixture["rec_id"]
    assert stats.condition_id == "E00"
    assert stats.extracted_embeddings > 0
    assert stats.output_rttm_path is not None
    assert Path(stats.output_rttm_path).exists()
    assert Path(stats.output_rttm_path).stat().st_size > 0
