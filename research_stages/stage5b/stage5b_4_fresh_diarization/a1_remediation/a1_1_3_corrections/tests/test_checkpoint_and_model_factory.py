#!/usr/bin/env python3
"""
Test Suite: Checkpoint Preflight Verification & Controlled Model Factory
Validates all 14 requirements from Section 4.2.E:
1. Missing TitaNet checkpoint.
2. Missing WeSpeaker checkpoint.
3. Incorrect checkpoint digest.
4. Checkpoint changed after initial verification (TOCTOU).
5. Checkpoint path replaced by symlink.
6. Checkpoint path replaced by a different file.
7. Arbitrary preloaded model injection in production.
8. Synthetic model handle passed to production.
9. Production model factory receiving an unverified checkpoint.
10. Model handle bound to the wrong encoder identity.
11. Model handle bound to a different checkpoint digest.
12. Correct verified checkpoint passed to the factory.
13. Correct model handle returned by a synthetic factory.
14. Revalidation failure immediately before loading.
"""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    CheckpointPreflightError,
    verify_encoder_checkpoint_strict,
    create_synthetic_checkpoint_fixture,
    run_production_checkpoint_preflight,
    compute_file_sha256,
)
from model_factory import (
    VerifiedModelHandle,
    ControlledModelFactory,
    validate_model_handle_strict,
    ModelFactoryError,
)
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from vad_adapter_remediated import VADAdapterRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter
from feature_extraction_interface import MockTitaNetModel, MockWeSpeakerModel


# 1. Missing TitaNet checkpoint in production
def test_1_missing_titanet_checkpoint_in_production(tmp_path):
    with pytest.raises(CheckpointPreflightError, match="Checkpoint path must be explicitly provided"):
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=None,
            expected_sha256=hashlib.sha256(b"dummy").hexdigest(),
            execution_mode=ExecutionMode.PRODUCTION,
        )


# 2. Missing WeSpeaker checkpoint in production
def test_2_missing_wespeaker_checkpoint_in_production(tmp_path):
    with pytest.raises(CheckpointPreflightError, match="Checkpoint path must be explicitly provided"):
        verify_encoder_checkpoint_strict(
            encoder_name="wespeaker_resnet293",
            checkpoint_path=None,
            expected_sha256=hashlib.sha256(b"dummy").hexdigest(),
            execution_mode=ExecutionMode.PRODUCTION,
        )


# 3. Incorrect checkpoint digest
def test_3_incorrect_checkpoint_digest(tmp_path):
    ckpt = tmp_path / "model.bin"
    ckpt.write_bytes(b"actual model bytes")
    wrong_hash = hashlib.sha256(b"different bytes").hexdigest()

    with pytest.raises(CheckpointPreflightError, match="Hash mismatch"):
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=ckpt,
            expected_sha256=wrong_hash,
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[tmp_path],
        )


# 4. Checkpoint changed after initial verification (TOCTOU)
def test_4_checkpoint_changed_after_initial_verification(tmp_path):
    ckpt = tmp_path / "model.bin"
    initial_bytes = b"initial valid checkpoint content"
    ckpt.write_bytes(initial_bytes)
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    v_ckpt = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=ckpt,
        expected_sha256=initial_hash,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_permitted_roots=[tmp_path],
    )

    # Mutate physical bytes keeping exact same length (32 bytes)
    mutated_bytes = b"mutated content with same length"
    assert len(mutated_bytes) == len(initial_bytes)
    ckpt.write_bytes(mutated_bytes)

    # Factory revalidation must detect the hash mutation and fail closed before loading
    with pytest.raises(ModelFactoryError, match="modified between preflight and load"):
        ControlledModelFactory.build_synthetic_model(v_ckpt, MockTitaNetModel())


# 5. Checkpoint path replaced by symlink escaping authorized root
def test_5_checkpoint_path_replaced_by_symlink(tmp_path):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    danger_root = tmp_path / "danger"
    danger_root.mkdir()

    danger_file = danger_root / "evil.bin"
    danger_file.write_bytes(b"evil model payload")

    ckpt_link = safe_root / "model.bin"
    os.symlink(danger_file, ckpt_link)

    with pytest.raises(CheckpointPreflightError, match="escapes authorized checkpoint roots"):
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=ckpt_link,
            expected_sha256=hashlib.sha256(b"evil model payload").hexdigest(),
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[safe_root],
        )


# 6. Checkpoint path replaced by a different file
def test_6_checkpoint_replaced_by_different_file(tmp_path):
    ckpt = tmp_path / "model.bin"
    ckpt.write_bytes(b"original content")
    orig_hash = hashlib.sha256(b"original content").hexdigest()

    v_ckpt = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=ckpt,
        expected_sha256=orig_hash,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_permitted_roots=[tmp_path],
    )

    # Replace file with completely different content/size
    ckpt.unlink()
    ckpt.write_bytes(b"a completely different replacement file of different length")

    with pytest.raises(ModelFactoryError, match="size changed between preflight and load|modified"):
        ControlledModelFactory.build_synthetic_model(v_ckpt, MockTitaNetModel())


# 7. Arbitrary preloaded model injection in production rejected
def test_7_arbitrary_preloaded_model_injection_in_production(tmp_path):
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path / "cache", allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=tmp_path)
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    # Passing raw model instances (MockTitaNetModel) to production orchestrator must raise ModelFactoryError
    with pytest.raises(ModelFactoryError, match="Arbitrary preloaded model instances.*strictly prohibited in PRODUCTION"):
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=clustering,
            evaluation_adapter=eval_adapter,
            titanet_model=MockTitaNetModel(),
            wespeaker_model=MockWeSpeakerModel(),
            execution_mode=ExecutionMode.PRODUCTION,
        )


# 8. Synthetic model handle passed to production rejected
def test_8_synthetic_model_handle_passed_to_production(tmp_path):
    ckpt_file = tmp_path / "model.bin"
    v_ckpt = create_synthetic_checkpoint_fixture("titanet_large", ckpt_file)
    synth_handle = ControlledModelFactory.build_synthetic_model(v_ckpt, MockTitaNetModel())

    assert synth_handle.is_synthetic is True

    # Validating a synthetic handle in production mode must fail closed
    with pytest.raises(ModelFactoryError, match="Synthetic model handle.*passed into PRODUCTION execution"):
        validate_model_handle_strict(
            handle=synth_handle,
            expected_encoder="titanet_large",
            execution_mode=ExecutionMode.PRODUCTION,
        )


# 9. Production model factory receiving an unverified checkpoint
def test_9_production_factory_receiving_unverified_checkpoint():
    # Attempting to forge a VerifiedCheckpoint directly without valid signature
    with pytest.raises(CheckpointPreflightError, match="FORGERY DETECTED"):
        VerifiedCheckpoint(
            encoder_name="titanet_large",
            canonical_path=Path("/fake/path"),
            expected_sha256="0" * 64,
            computed_sha256="0" * 64,
            file_size_bytes=100,
            verification_timestamp=datetime.now(timezone.utc).isoformat(),
            verification_status="FORGED",
            authorized_root=Path("/fake"),
            execution_mode=ExecutionMode.PRODUCTION,
            _security_signature="invalid_forged_signature",
        )


# 10. Model handle bound to the wrong encoder identity
def test_10_model_handle_bound_to_wrong_encoder(tmp_path):
    ckpt_file = tmp_path / "wespeaker.bin"
    v_ckpt = create_synthetic_checkpoint_fixture("wespeaker_resnet293", ckpt_file)
    handle = ControlledModelFactory.build_synthetic_model(v_ckpt, MockWeSpeakerModel())

    # Asking for 'titanet_large' when handle is 'wespeaker_resnet293'
    with pytest.raises(ModelFactoryError, match="Encoder identity mismatch"):
        validate_model_handle_strict(
            handle=handle,
            expected_encoder="titanet_large",
            execution_mode=ExecutionMode.SYNTHETIC,
        )


# 11. Model handle bound to a different checkpoint digest
def test_11_model_handle_bound_to_different_digest(tmp_path):
    ckpt_file = tmp_path / "model.bin"
    v_ckpt = create_synthetic_checkpoint_fixture("titanet_large", ckpt_file)

    # Trying to tamper with the handle's bound checkpoint or signature
    with pytest.raises(ModelFactoryError, match="FORGERY DETECTED"):
        VerifiedModelHandle(
            encoder_name="titanet_large",
            verified_checkpoint=v_ckpt,
            model_instance=MockTitaNetModel(),
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            is_synthetic=True,
            _handle_signature="tampered_signature",
        )


# 12. Correct verified checkpoint passed to the factory succeeds
def test_12_correct_verified_checkpoint_passed_to_factory(tmp_path):
    ckpt_file = tmp_path / "valid.bin"
    v_ckpt = create_synthetic_checkpoint_fixture("titanet_large", ckpt_file)

    handle = ControlledModelFactory.build_synthetic_model(v_ckpt, MockTitaNetModel())
    assert isinstance(handle, VerifiedModelHandle)
    assert handle.encoder_name == "titanet_large"
    assert handle.verified_checkpoint.canonical_path == ckpt_file.resolve()
    assert handle.is_synthetic is True


# 13. Correct model handle returned by a synthetic factory
def test_13_correct_model_handle_returned_by_synthetic_factory(tmp_path):
    ckpt_tita = tmp_path / "tita.bin"
    ckpt_wes = tmp_path / "wes.bin"
    v_tita = create_synthetic_checkpoint_fixture("titanet_large", ckpt_tita)
    v_wes = create_synthetic_checkpoint_fixture("wespeaker_resnet293", ckpt_wes)

    h_tita = ControlledModelFactory.build_synthetic_model(v_tita, MockTitaNetModel(return_dim=192))
    h_wes = ControlledModelFactory.build_synthetic_model(v_wes, MockWeSpeakerModel(return_dim=256))

    assert h_tita.model.return_dim == 192
    assert h_wes.model.return_dim == 256
    assert h_tita.encoder_name == "titanet_large"
    assert h_wes.encoder_name == "wespeaker_resnet293"


# 14. Revalidation failure immediately before loading
def test_14_revalidation_failure_file_disappears(tmp_path):
    ckpt_file = tmp_path / "temp.bin"
    v_ckpt = create_synthetic_checkpoint_fixture("titanet_large", ckpt_file)

    # Delete file immediately after preflight
    ckpt_file.unlink()

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure: Checkpoint file disappeared"):
        ControlledModelFactory.build_synthetic_model(v_ckpt, MockTitaNetModel())
