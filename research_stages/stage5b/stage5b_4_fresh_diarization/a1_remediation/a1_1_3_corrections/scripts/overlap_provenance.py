#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3 Remediation: Overlap Posterior Provenance Verifier & State Schema
Defines strict separation between:
1. Structural validity (array shape, stride, centering, monotonicity, probability range)
2. Provenance verification (source audio identity, posterior digest, model checkpoint, generation parameters)
3. Production eligibility (requires structural validity AND independent provenance verification)

Fails closed on any incomplete, corrupt, unrecorded, or placeholder provenance.
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
    get_a1_3_dir,
)
from checkpoint_preflight import ExecutionMode, compute_file_sha256, is_placeholder_hash

PYANNET_FRAME_STEP_SEC = 0.016875       # 270 samples at 16 kHz
PYANNET_FIRST_FRAME_CENTER = 0.03096875 # Center of first sliding window
PYANNET_SAMPLE_RATE = 16000
PYANNET_MODEL_CHECKPOINT_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"


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
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or (
            get_a1_3_dir() / "manifests" / "overlap_posterior_provenance_manifest.json"
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
    ) -> ProvenanceValidationState:
        """
        Independently verifies posterior file on disk against the authoritative provenance manifest.
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

        # 2. Path and file existence
        p_path = Path(posterior_path).resolve()
        if not p_path.exists():
            errors.append(f"Posterior file does not exist at {p_path}.")
            actual_size = 0
            computed_sha = "MISSING_FILE"
        else:
            actual_size = p_path.stat().st_size
            computed_sha = compute_file_sha256(p_path)

        # 3. File size verification
        expected_size = entry.get("posterior_file_size_bytes")
        if expected_size is not None and actual_size != expected_size:
            errors.append(
                f"Posterior file size mismatch: expected {expected_size} bytes, got {actual_size} bytes."
            )

        # 4. Posterior file SHA-256 verification
        expected_sha = entry.get("posterior_file_sha256", "").strip().lower()
        if is_placeholder_hash(expected_sha):
            errors.append(f"Placeholder SHA-256 in manifest for posterior '{key}': {expected_sha}")
        elif computed_sha != expected_sha:
            errors.append(
                f"Posterior file hash mismatch: computed {computed_sha} != expected {expected_sha}."
            )

        # 5. Model checkpoint hash verification
        model_sha = entry.get("model_checkpoint_sha256")
        if not model_sha:
            unverified.append("model_checkpoint_sha256")
        elif model_sha != PYANNET_MODEL_CHECKPOINT_SHA256:
            errors.append(
                f"Model checkpoint hash mismatch: manifest has {model_sha}, authoritative PyanNet is {PYANNET_MODEL_CHECKPOINT_SHA256}."
            )

        # 6. Source audio hash verification
        source_audio_sha = entry.get("source_audio_sha256")
        if not source_audio_sha or source_audio_sha == "UNRECORDED_IN_HISTORICAL_MANIFEST":
            unverified.append("source_audio_sha256")
        elif is_placeholder_hash(source_audio_sha):
            errors.append(f"Placeholder source audio SHA-256 in manifest: {source_audio_sha}")

        # 7. Check if unverified fields were explicitly noted in entry
        for field in entry.get("unverified_fields", []):
            if field not in unverified:
                unverified.append(field)

        provenance_verified = len(errors) == 0 and len(unverified) == 0

        return ProvenanceValidationState(
            provenance_verified=provenance_verified,
            recording_id=recording_id,
            stream_name=stream_name,
            posterior_file=p_path,
            posterior_sha256=computed_sha,
            file_size_bytes=actual_size,
            source_audio_file=Path(entry["source_audio_rel"]) if "source_audio_rel" in entry else None,
            source_audio_sha256=source_audio_sha,
            source_duration_sec=entry.get("source_duration_sec"),
            source_sample_rate=entry.get("source_sample_rate", PYANNET_SAMPLE_RATE),
            model_architecture=entry.get("model_architecture"),
            model_checkpoint_sha256=model_sha,
            generation_config_id=entry.get("generation_config_id"),
            timestamp_convention=entry.get("timestamp_convention", "sliding_window_middle_centering"),
            unverified_fields=unverified,
            provenance_errors=errors,
        )
