#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.1 Remediation: Overlap Posterior Provenance Verifier & State Schema
Defines strict separation between:
1. Structural validity (array shape, stride, centering, monotonicity, probability range)
2. Provenance verification (byte-level source audio hash, posterior digest, physical model checkpoint, generation parameters)
3. Production eligibility (requires structural validity AND independent provenance verification)

Remediates Finding 2 and Finding 3:
- Byte-level SHA-256 verification of actual source-audio files on disk.
- Byte-level SHA-256 verification of actual PyanNet model checkpoint on disk.
- Validation of generation configuration, timestamp convention, and posterior semantics against
  a trusted, independently established specification (TRUSTED_PYANNET_POSTERIOR_SPEC).
- Fails closed on any incomplete, corrupt, unrecorded, mismatched, or placeholder provenance.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from path_utils import (
    get_repo_root,
    get_stage5b_2_overlap_cache_dir,
    get_stage2_posteriors_dir,
    get_a1_3_1_dir,
)
from checkpoint_preflight import ExecutionMode, compute_file_sha256, is_placeholder_hash

PYANNET_FRAME_STEP_SEC = 0.016875       # 270 samples at 16 kHz
PYANNET_FIRST_FRAME_CENTER = 0.03096875 # Center of first sliding window
PYANNET_SAMPLE_RATE = 16000
PYANNET_MODEL_CHECKPOINT_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"

# Trusted, version-controlled specification for Stage 2 PyanNet overlap posteriors (Finding 3)
TRUSTED_PYANNET_POSTERIOR_SPEC = {
    "schema_version": "1.0",
    "model_architecture": "pyannote/segmentation",
    "model_checkpoint_sha256": PYANNET_MODEL_CHECKPOINT_SHA256,
    "generation_config_id": "pyannote_3_1_continuous_top2",
    "posterior_semantics": "continuous_top2_second_highest",
    "timestamp_convention": "sliding_window_middle_centering",
    "frame_step_sec": PYANNET_FRAME_STEP_SEC,
    "first_frame_center_sec": PYANNET_FIRST_FRAME_CENTER,
    "sample_rate": PYANNET_SAMPLE_RATE,
    "generation_code_version": "stage2_baseline_v1",
}


class OverlapProvenanceError(Exception):
    """Raised when overlap posterior data or provenance fails verification."""
    pass


@dataclass(frozen=True)
class StructuralValidationState:
    """Represents purely acoustic / mathematical validity of posterior arrays."""
    structurally_valid: bool
    first_frame_sec: float
    last_frame_sec: float
    frame_step_sec: float
    frame_count: int
    min_probability: float
    max_probability: float
    validation_errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProvenanceValidationState:
    """Represents cryptographic, model, source audio, and generation provenance."""
    provenance_verified: bool
    recording_id: str
    stream_name: str
    posterior_file: Path
    posterior_sha256: str
    file_size_bytes: int
    source_audio_file: Optional[Path]
    source_audio_sha256: Optional[str]
    source_duration_sec: Optional[float]
    source_sample_rate: Optional[int]
    model_architecture: Optional[str]
    model_checkpoint_sha256: Optional[str]
    generation_config_id: Optional[str]
    timestamp_convention: str
    unverified_fields: List[str]
    provenance_errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["posterior_file"] = str(self.posterior_file)
        if self.source_audio_file:
            d["source_audio_file"] = str(self.source_audio_file)
        return d


@dataclass(frozen=True)
class OverallValidationState:
    """Overall multi-state validation status."""
    structurally_valid: bool
    provenance_verified: bool
    production_eligible: bool
    is_synthetic: bool
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedPosteriorContainer:
    """
    Immutable container carrying the validated arrays and explicit multi-state validation.
    Downstream components inspect overall_state.production_eligible.
    """
    frame_times: np.ndarray
    p_overlap: np.ndarray
    structural_state: StructuralValidationState
    provenance_state: ProvenanceValidationState
    overall_state: OverallValidationState
    is_synthetic: bool

    def __post_init__(self):
        # Prevent inconsistent or forged states
        if self.overall_state.production_eligible:
            if not self.structural_state.structurally_valid:
                raise OverlapProvenanceError("Invalid state: production_eligible cannot be True if structurally_valid is False.")
            if not self.provenance_state.provenance_verified:
                raise OverlapProvenanceError("Invalid state: production_eligible cannot be True if provenance_verified is False.")
            if self.is_synthetic:
                raise OverlapProvenanceError("Invalid state: Synthetic posterior can NEVER be production_eligible.")
            if len(self.provenance_state.unverified_fields) > 0:
                raise OverlapProvenanceError("Invalid state: production_eligible cannot be True with unverified fields.")


class OverlapProvenanceVerifier:
    """
    Independent cryptographic verifier for overlap posteriors against authoritative manifests.
    Remediates Findings 2 and 3:
    - Validates manifest metadata against TRUSTED_PYANNET_POSTERIOR_SPEC.
    - Directly hashes source audio files on disk (when recorded).
    - Directly hashes PyanNet model checkpoint files on disk.
    - Explicitly preserves historical unrecorded mono limitation.
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        source_audio_root: Optional[Path] = None,
        pyannet_checkpoint_path: Optional[Path] = None,
    ):
        self.manifest_path = manifest_path or (
            get_a1_3_1_dir() / "manifests" / "overlap_posterior_provenance_manifest.json"
        )
        self.source_audio_root = Path(source_audio_root).resolve() if source_audio_root else get_repo_root()
        self.pyannet_checkpoint_path = Path(pyannet_checkpoint_path).resolve() if pyannet_checkpoint_path else (
            get_repo_root() / "whisperx" / "assets" / "pytorch_model.bin"
        )
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def _load_manifest(self) -> Dict[str, Any]:
        if self._manifest_cache is not None:
            return self._manifest_cache
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Overlap provenance manifest not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._manifest_cache = data
        return data

    def verify_provenance(
        self,
        recording_id: str,
        stream_name: str,
        posterior_path: Path,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
        is_synthetic: bool = False,
        source_audio_override: Optional[Path] = None,
        pyannet_checkpoint_override: Optional[Path] = None,
    ) -> ProvenanceValidationState:
        """
        Independently verifies posterior file on disk against the authoritative provenance manifest
        and trusted configuration specification.
        Performs byte-level hashing of posterior, source audio, and model checkpoint.
        Fails closed on missing fields, corrupted checksums, size mismatches, or placeholder hashes.
        """
        errors: List[str] = []
        unverified: List[str] = []

        if is_synthetic or execution_mode == ExecutionMode.SYNTHETIC:
            # Synthetic fixtures are never provenance-verified
            unverified.append("synthetic_fixture_no_historical_provenance")
            return ProvenanceValidationState(
                provenance_verified=False,
                recording_id=recording_id,
                stream_name=stream_name,
                posterior_file=posterior_path,
                posterior_sha256="synthetic_fixture",
                file_size_bytes=posterior_path.stat().st_size if posterior_path.exists() else 0,
                source_audio_file=None,
                source_audio_sha256=None,
                source_duration_sec=None,
                source_sample_rate=PYANNET_SAMPLE_RATE,
                model_architecture="synthetic_mock",
                model_checkpoint_sha256=None,
                generation_config_id="synthetic",
                timestamp_convention="synthetic",
                unverified_fields=unverified,
                provenance_errors=["Synthetic fixture cannot have verified production provenance."],
            )

        # 1. Load authoritative manifest
        manifest_data = self._load_manifest()
        artifacts = manifest_data.get("artifacts", {})
        key = f"{recording_id}__{stream_name}"

        if key not in artifacts:
            errors.append(f"No provenance manifest record found for '{key}'.")
            return ProvenanceValidationState(
                provenance_verified=False,
                recording_id=recording_id,
                stream_name=stream_name,
                posterior_file=posterior_path,
                posterior_sha256="UNKNOWN",
                file_size_bytes=0,
                source_audio_file=None,
                source_audio_sha256=None,
                source_duration_sec=None,
                source_sample_rate=None,
                model_architecture=None,
                model_checkpoint_sha256=None,
                generation_config_id=None,
                timestamp_convention="unknown",
                unverified_fields=["manifest_record_missing"],
                provenance_errors=errors,
            )

        entry = artifacts[key]

        # 2. Schema and Configuration Verification against TRUSTED_PYANNET_POSTERIOR_SPEC (Finding 3)
        schema_version = manifest_data.get("schema_version")
        if schema_version != TRUSTED_PYANNET_POSTERIOR_SPEC["schema_version"]:
            errors.append(
                f"Unsupported manifest schema_version '{schema_version}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['schema_version']}'."
            )

        # Model Architecture
        model_arch = entry.get("model_architecture")
        if model_arch != TRUSTED_PYANNET_POSTERIOR_SPEC["model_architecture"]:
            errors.append(
                f"Model architecture mismatch: manifest has '{model_arch}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['model_architecture']}'."
            )

        # Generation Config ID
        gen_config = entry.get("generation_config_id")
        if gen_config != TRUSTED_PYANNET_POSTERIOR_SPEC["generation_config_id"]:
            errors.append(
                f"Generation config mismatch: manifest has '{gen_config}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['generation_config_id']}'."
            )

        # Posterior Semantics
        post_semantics = entry.get("posterior_semantics")
        if post_semantics != TRUSTED_PYANNET_POSTERIOR_SPEC["posterior_semantics"]:
            errors.append(
                f"Posterior semantics mismatch: manifest has '{post_semantics}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['posterior_semantics']}'."
            )

        # Timestamp Convention
        ts_convention = entry.get("timestamp_convention")
        if ts_convention != TRUSTED_PYANNET_POSTERIOR_SPEC["timestamp_convention"]:
            errors.append(
                f"Timestamp convention mismatch: manifest has '{ts_convention}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['timestamp_convention']}'."
            )

        # Frame step and first frame center
        frame_step = entry.get("frame_step_sec")
        if frame_step is None or abs(frame_step - TRUSTED_PYANNET_POSTERIOR_SPEC["frame_step_sec"]) > 1e-6:
            errors.append(
                f"Frame step mismatch: manifest has '{frame_step}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['frame_step_sec']}'."
            )

        first_center = entry.get("first_frame_center_sec")
        if first_center is None or abs(first_center - TRUSTED_PYANNET_POSTERIOR_SPEC["first_frame_center_sec"]) > 1e-6:
            errors.append(
                f"First frame center mismatch: manifest has '{first_center}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['first_frame_center_sec']}'."
            )

        # Sample Rate
        sample_rate = entry.get("source_sample_rate")
        if sample_rate != TRUSTED_PYANNET_POSTERIOR_SPEC["sample_rate"]:
            errors.append(
                f"Sample rate mismatch: manifest has '{sample_rate}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['sample_rate']}'."
            )

        # Generation Code Version
        code_ver = entry.get("generation_code_version", TRUSTED_PYANNET_POSTERIOR_SPEC["generation_code_version"])
        if code_ver != TRUSTED_PYANNET_POSTERIOR_SPEC["generation_code_version"]:
            errors.append(
                f"Generation code version mismatch: manifest has '{code_ver}', expected '{TRUSTED_PYANNET_POSTERIOR_SPEC['generation_code_version']}'."
            )

        # 3. Posterior file existence, size, and SHA-256 verification
        p_raw = Path(posterior_path)
        if p_raw.is_symlink():
            errors.append(f"Symlink escape: posterior path is a symlink: {p_raw}")

        p_path = p_raw.resolve()
        if not p_path.exists():
            errors.append(f"Posterior file does not exist at {p_path}.")
            actual_size = 0
            computed_sha = "MISSING_FILE"
        elif not p_path.is_file():
            errors.append(f"Posterior target is not a regular file: {p_path}.")
            actual_size = 0
            computed_sha = "NOT_A_FILE"
        else:
            actual_size = p_path.stat().st_size
            computed_sha = compute_file_sha256(p_path)

        expected_size = entry.get("posterior_file_size_bytes")
        if expected_size is not None and actual_size != expected_size:
            errors.append(
                f"Posterior file size mismatch: expected {expected_size} bytes, got {actual_size} bytes."
            )

        expected_sha = entry.get("posterior_file_sha256", "").strip().lower()
        if is_placeholder_hash(expected_sha):
            errors.append(f"Placeholder SHA-256 in manifest for posterior '{key}': {expected_sha}")
        elif computed_sha != expected_sha:
            errors.append(
                f"Posterior file hash mismatch: computed {computed_sha} != expected {expected_sha}."
            )

        # 4. PyanNet Checkpoint Physical Byte Verification (Finding 2)
        pyannet_ckpt = pyannet_checkpoint_override or self.pyannet_checkpoint_path
        if pyannet_ckpt.is_symlink():
            errors.append(f"Symlink escape: PyanNet checkpoint is a symlink: {pyannet_ckpt}")
        if not pyannet_ckpt.exists():
            errors.append(f"PyanNet checkpoint file does not exist at {pyannet_ckpt}.")
            computed_pyannet_sha = "MISSING_CHECKPOINT"
        elif not pyannet_ckpt.is_file():
            errors.append(f"PyanNet checkpoint is not a regular file: {pyannet_ckpt}.")
            computed_pyannet_sha = "NOT_A_FILE"
        else:
            computed_pyannet_sha = compute_file_sha256(pyannet_ckpt)

        manifest_model_sha = entry.get("model_checkpoint_sha256", "").strip().lower()
        if is_placeholder_hash(manifest_model_sha):
            errors.append(f"Placeholder model checkpoint SHA-256 in manifest: {manifest_model_sha}")
        elif manifest_model_sha != TRUSTED_PYANNET_POSTERIOR_SPEC["model_checkpoint_sha256"]:
            errors.append(
                f"Model checkpoint hash in manifest '{manifest_model_sha}' does not match trusted specification '{TRUSTED_PYANNET_POSTERIOR_SPEC['model_checkpoint_sha256']}'."
            )
        elif computed_pyannet_sha != TRUSTED_PYANNET_POSTERIOR_SPEC["model_checkpoint_sha256"]:
            errors.append(
                f"PyanNet physical checkpoint bytes hash mismatch: computed {computed_pyannet_sha} != expected {TRUSTED_PYANNET_POSTERIOR_SPEC['model_checkpoint_sha256']}."
            )

        # 5. Source Audio Physical Byte Verification (Finding 2)
        source_audio_sha = entry.get("source_audio_sha256")
        source_audio_rel = entry.get("source_audio_rel")
        resolved_audio_file: Optional[Path] = None

        if not source_audio_sha or source_audio_sha == "UNRECORDED_IN_HISTORICAL_MANIFEST":
            unverified.append("source_audio_sha256")
            errors.append(f"Historical source audio SHA-256 is unrecorded for '{key}'.")
        elif is_placeholder_hash(source_audio_sha):
            errors.append(f"Placeholder source audio SHA-256 in manifest: {source_audio_sha}")
        else:
            # Physical verification of source audio bytes
            raw_audio_path: Optional[Path] = None
            if source_audio_override:
                raw_audio_path = Path(source_audio_override)
            elif source_audio_rel:
                # Path traversal / escape check
                if ".." in Path(source_audio_rel).parts:
                    errors.append(f"Path traversal detected in source_audio_rel: {source_audio_rel}")
                else:
                    raw_audio_path = self.source_audio_root / source_audio_rel
            
            if raw_audio_path is None:
                errors.append(f"Source audio file path not specified in manifest for '{key}'.")
                unverified.append("source_audio_file")
            else:
                if raw_audio_path.is_symlink():
                    errors.append(f"Symlink escape: source audio file is a symlink: {raw_audio_path}")
                resolved_audio_file = raw_audio_path.resolve()
                if not resolved_audio_file.exists():
                    errors.append(f"Source audio file does not exist at {resolved_audio_file}.")
                    unverified.append("source_audio_file_missing")
                elif not resolved_audio_file.is_file():
                    errors.append(f"Source audio target is not a regular file: {resolved_audio_file}.")
                    unverified.append("source_audio_not_file")
                else:
                    computed_audio_sha = compute_file_sha256(resolved_audio_file)
                    if computed_audio_sha != source_audio_sha.strip().lower():
                        errors.append(
                            f"Source audio file hash mismatch: computed {computed_audio_sha} != expected {source_audio_sha}."
                        )

        # 6. Check if unverified fields were explicitly noted in entry
        for field in entry.get("unverified_fields", []):
            if field not in unverified:
                unverified.append(field)

        provenance_verified = (len(errors) == 0 and len(unverified) == 0)

        return ProvenanceValidationState(
            provenance_verified=provenance_verified,
            recording_id=recording_id,
            stream_name=stream_name,
            posterior_file=p_path,
            posterior_sha256=computed_sha,
            file_size_bytes=actual_size,
            source_audio_file=resolved_audio_file,
            source_audio_sha256=source_audio_sha,
            source_duration_sec=entry.get("source_duration_sec"),
            source_sample_rate=sample_rate or PYANNET_SAMPLE_RATE,
            model_architecture=model_arch,
            model_checkpoint_sha256=manifest_model_sha,
            generation_config_id=gen_config,
            timestamp_convention=ts_convention or "sliding_window_middle_centering",
            unverified_fields=unverified,
            provenance_errors=errors,
        )
