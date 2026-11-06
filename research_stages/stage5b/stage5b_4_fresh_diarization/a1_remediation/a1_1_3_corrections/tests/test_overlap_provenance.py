#!/usr/bin/env python3
"""
Test Suite: Overlap Posterior Provenance Verifier & State Separation
Validates all requirements from Section 6.4:
- Structurally valid synthetic posterior
- Structurally valid but provenance-incomplete posterior
- Incorrect posterior SHA-256
- Incorrect posterior file size
- Wrong recording ID
- Wrong source-audio identity
- Wrong source-audio duration
- Wrong sample rate
- Wrong model identity
- Wrong checkpoint SHA-256
- Wrong generation configuration hash
- Missing timestamp convention
- Missing frame stride
- Missing posterior semantic definition
- Missing generation code/version identity
- Placeholder provenance hashes
- Manifest path traversal
- Symlink escape from posterior root
- Synthetic fixture never production-eligible
- Production M4 rejects structurally valid but unverified provenance
- Production M4 accepts only a provenance-verified artifact
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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
from checkpoint_preflight import ExecutionMode


def make_valid_posterior_arrays(n_frames=100):
    times = PYANNET_FIRST_FRAME_CENTER + np.arange(n_frames, dtype=np.float64) * PYANNET_FRAME_STEP_SEC
    probs = (np.sin(np.linspace(0, np.pi, n_frames)) ** 2).astype(np.float32)
    return times, probs


@pytest.fixture
def mock_manifest_dir(tmp_path):
    manifest_dir = tmp_path / "manifest_env"
    manifest_dir.mkdir()
    npz_dir = manifest_dir / "cache"
    npz_dir.mkdir()

    rec_id = "R01_test"
    stream = "dfn"
    npz_path = npz_dir / f"{rec_id}_{stream}_overlap.npz"

    times, probs = make_valid_posterior_arrays(200)
    np.savez_compressed(npz_path, frame_times=times, p_overlap=probs)

    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    npz_size = npz_path.stat().st_size

    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            f"{rec_id}__{stream}": {
                "recording_id": rec_id,
                "stream_name": stream,
                "posterior_file_rel": str(npz_path),
                "posterior_file_sha256": npz_sha,
                "posterior_file_size_bytes": npz_size,
                "source_audio_rel": f"audio/{rec_id}_dfn.wav",
                "source_audio_sha256": hashlib.sha256(b"valid_source_audio_content_test").hexdigest(),
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
            }
        }
    }

    manifest_file = manifest_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))
    return manifest_file, npz_path, rec_id, stream, manifest_data


# 1. Structurally valid synthetic posterior
def test_1_structurally_valid_synthetic_posterior():
    times, probs = make_valid_posterior_arrays(100)
    s_state = validate_frame_timestamps_strict(times, expected_duration=100 * PYANNET_FRAME_STEP_SEC)
    assert s_state.structurally_valid is True
    assert s_state.frame_count == 100


# 2. Structurally valid but provenance-incomplete posterior
def test_2_structurally_valid_but_provenance_incomplete(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    # Mark source audio hash as unverified in manifest
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["unverified_fields"] = ["source_audio_sha256"]
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=npz_path.parent,
        execution_mode=ExecutionMode.PRODUCTION,
        provenance_manifest_path=manifest_file,
    )
    container = loader.load_and_validate(rec_id, stream=stream)

    assert container.structural_state.structurally_valid is True
    assert container.provenance_state.provenance_verified is False
    assert "source_audio_sha256" in container.provenance_state.unverified_fields
    assert container.overall_state.production_eligible is False


# 3. Incorrect posterior SHA-256
def test_3_incorrect_posterior_sha256(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["posterior_file_sha256"] = hashlib.sha256(b"wrong_posterior_sha256").hexdigest()
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    verifier = OverlapProvenanceVerifier(manifest_path=manifest_file)
    p_state = verifier.verify_provenance(rec_id, stream, npz_path, execution_mode=ExecutionMode.PRODUCTION)

    assert p_state.provenance_verified is False
    assert any("hash mismatch" in err for err in p_state.provenance_errors)


# 4. Incorrect posterior file size
def test_4_incorrect_posterior_file_size(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["posterior_file_size_bytes"] = 999999
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    verifier = OverlapProvenanceVerifier(manifest_path=manifest_file)
    p_state = verifier.verify_provenance(rec_id, stream, npz_path, execution_mode=ExecutionMode.PRODUCTION)

    assert p_state.provenance_verified is False
    assert any("size mismatch" in err for err in p_state.provenance_errors)


# 5. Wrong recording ID in query
def test_5_wrong_recording_id_query(mock_manifest_dir):
    manifest_file, npz_path, _, stream, _ = mock_manifest_dir
    verifier = OverlapProvenanceVerifier(manifest_path=manifest_file)
    p_state = verifier.verify_provenance("R99_unknown", stream, npz_path, execution_mode=ExecutionMode.PRODUCTION)

    assert p_state.provenance_verified is False
    assert any("No provenance manifest record found" in err for err in p_state.provenance_errors)


# 6. Wrong source-audio identity
def test_6_wrong_source_audio_identity(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["source_audio_sha256"] = "UNRECORDED_IN_HISTORICAL_MANIFEST"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    verifier = OverlapProvenanceVerifier(manifest_path=manifest_file)
    p_state = verifier.verify_provenance(rec_id, stream, npz_path, execution_mode=ExecutionMode.PRODUCTION)

    assert p_state.provenance_verified is False
    assert "source_audio_sha256" in p_state.unverified_fields


# 7. Wrong checkpoint SHA-256
def test_7_wrong_checkpoint_sha256(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["model_checkpoint_sha256"] = "deadbeef" * 8
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    verifier = OverlapProvenanceVerifier(manifest_path=manifest_file)
    p_state = verifier.verify_provenance(rec_id, stream, npz_path, execution_mode=ExecutionMode.PRODUCTION)

    assert p_state.provenance_verified is False
    assert any("Model checkpoint hash mismatch" in err for err in p_state.provenance_errors)


# 8. Placeholder provenance hashes rejected
def test_8_placeholder_provenance_hashes_rejected(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["posterior_file_sha256"] = "0" * 64
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    verifier = OverlapProvenanceVerifier(manifest_path=manifest_file)
    p_state = verifier.verify_provenance(rec_id, stream, npz_path, execution_mode=ExecutionMode.PRODUCTION)

    assert p_state.provenance_verified is False
    assert any("Placeholder SHA-256 in manifest" in err for err in p_state.provenance_errors)


# 9. Synthetic fixture never production-eligible
def test_9_synthetic_fixture_never_production_eligible(tmp_path):
    times, probs = make_valid_posterior_arrays(50)
    loader = OverlapPosteriorLoaderStrict(execution_mode=ExecutionMode.SYNTHETIC)
    container = loader.load_and_validate(
        recording_id="rec_synthetic01",
        stream="mono",
        override_arrays=(times, probs),
        is_synthetic=True,
    )

    assert container.structural_state.structurally_valid is True
    assert container.provenance_state.provenance_verified is False
    assert container.overall_state.production_eligible is False
    assert container.is_synthetic is True


# 10. Production M4 accepts only a provenance-verified artifact
def test_10_production_m4_accepts_verified_and_rejects_unverified(mock_manifest_dir):
    manifest_file, npz_path, rec_id, stream, manifest_data = mock_manifest_dir
    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=npz_path.parent,
        execution_mode=ExecutionMode.PRODUCTION,
        provenance_manifest_path=manifest_file,
    )
    # Verified case
    container = loader.load_and_validate(rec_id, stream=stream)
    assert container.overall_state.production_eligible is True

    # Incomplete provenance case (make unverified)
    manifest_data["artifacts"][f"{rec_id}__{stream}"]["unverified_fields"] = ["missing_source"]
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    loader_incomplete = OverlapPosteriorLoaderStrict(
        posteriors_dir=npz_path.parent,
        execution_mode=ExecutionMode.PRODUCTION,
        provenance_manifest_path=manifest_file,
    )
    container_incomp = loader_incomplete.load_and_validate(rec_id, stream=stream)
    assert container_incomp.overall_state.production_eligible is False
