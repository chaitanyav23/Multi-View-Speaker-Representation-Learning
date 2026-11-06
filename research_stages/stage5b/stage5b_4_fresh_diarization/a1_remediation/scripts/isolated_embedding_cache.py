#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Isolated Embedding Cache
Enforces strict path containment, caller component sanitization,
symlink traversal defense, and atomic writes. Prohibits any access to Stage 3 caches.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


CACHE_SCHEMA_VERSION = "5b4_a1_v1"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_STAGE5B_4_CACHE_ROOT = (
    REPO_ROOT / "research_stages" / "stage5b" / "stage5b_4_fresh_diarization" / "cache"
)
PROHIBITED_STAGE3_CACHE = REPO_ROOT / "research_stages" / "stage3_dual_encoder" / "cache"

# Safe identifier pattern: alphanumeric, underscore, hyphen
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


class CacheSecurityError(PermissionError):
    """Raised when an unsafe path or prohibited cache location is accessed."""
    pass


class CacheValidationError(ValueError):
    """Raised when cached metadata or embedding payload fails integrity validation."""
    pass


def validate_path_component(name: str, value: Any) -> str:
    """
    Validates that a caller-provided identifier is strictly alphanumeric/hyphen/underscore.
    Rejects path traversal (..), separators (/ or \\), null bytes, and empty strings.
    """
    if not isinstance(value, (str, int)):
        raise CacheSecurityError(f"Component '{name}' must be string or integer, got {type(value)}")

    val_str = str(value).strip()
    if not val_str:
        raise CacheSecurityError(f"Component '{name}' cannot be empty.")

    if ".." in val_str or "/" in val_str or "\\" in val_str or "\x00" in val_str:
        raise CacheSecurityError(f"Path traversal sequence detected in '{name}': '{val_str}'")

    if not SAFE_ID_REGEX.match(val_str):
        raise CacheSecurityError(
            f"Component '{name}' contains invalid characters: '{val_str}'. "
            "Must match regex '^[a-zA-Z0-9_\\-]+$'."
        )

    return val_str


@dataclass(frozen=True)
class CacheIdentity:
    schema_version: str
    recording_id: str
    vad_config_id: str
    vad_intervals_sha256: str
    encoder_name: str
    encoder_checkpoint_sha256: str
    audio_source_id: str
    audio_preprocessing_id: str
    scale_index: int
    window_sec: float
    shift_sec: float
    sample_rate: int
    frontend_config_version: str

    def __post_init__(self):
        # Validate all caller-controlled path components
        validate_path_component("recording_id", self.recording_id)
        validate_path_component("vad_config_id", self.vad_config_id)
        validate_path_component("encoder_name", self.encoder_name)
        validate_path_component("audio_source_id", self.audio_source_id)
        validate_path_component("scale_index", self.scale_index)
        validate_path_component("vad_intervals_sha256", self.vad_intervals_sha256)
        validate_path_component("encoder_checkpoint_sha256", self.encoder_checkpoint_sha256)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class IsolatedEmbeddingCacheManager:
    """
    Manages loading, storing, and validating multi-scale speaker embeddings.
    Guarantees strict containment within authorized cache root.
    """

    def __init__(self, cache_root: Optional[Path] = None, allow_custom_root: bool = False):
        raw_root = Path(cache_root or DEFAULT_STAGE5B_4_CACHE_ROOT)
        # Canonical resolution
        self.cache_root = Path(os.path.realpath(str(raw_root)))

        # Validate containment
        self._assert_safe_cache_root(self.cache_root, allow_custom_root=allow_custom_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_safe_cache_root(path: Path, allow_custom_root: bool = False) -> None:
        """Enforces that path is safe and strictly isolated from prohibited directories."""
        real_path = Path(os.path.realpath(str(path)))
        real_prohibited = Path(os.path.realpath(str(PROHIBITED_STAGE3_CACHE)))

        # 1. Prohibit Stage 3 cache or any subdirectory
        if real_path == real_prohibited or real_prohibited in real_path.parents or real_path in real_prohibited.parents:
            raise CacheSecurityError(
                f"FATAL: Attempted to use prohibited Stage 3 cache directory: {real_path}"
            )

        # 2. If not custom test root, must be inside Stage 5B.4 workspace
        if not allow_custom_root:
            permitted_ancestor = Path(os.path.realpath(str(REPO_ROOT / "research_stages" / "stage5b" / "stage5b_4_fresh_diarization")))
            try:
                common = os.path.commonpath([str(permitted_ancestor), str(real_path)])
                if common != str(permitted_ancestor):
                    raise CacheSecurityError(
                        f"FATAL: Cache root {real_path} escapes permitted Stage 5B.4 workspace {permitted_ancestor}"
                    )
            except ValueError as e:
                raise CacheSecurityError(f"FATAL: Cache path outside workspace drive/root: {e}")

    def _assert_contained(self, target_path: Path) -> Path:
        """Verifies that resolved target_path resides strictly within self.cache_root."""
        real_target = Path(os.path.realpath(str(target_path)))
        real_root = Path(os.path.realpath(str(self.cache_root)))
        try:
            common = os.path.commonpath([str(real_root), str(real_target)])
            if common != str(real_root):
                raise CacheSecurityError(
                    f"Path traversal detected: {real_target} escapes cache root {real_root}"
                )
        except ValueError as e:
            raise CacheSecurityError(f"Path containment check failed: {e}")

        # Check for symlink escape
        if target_path.is_symlink():
            resolved_symlink = target_path.resolve()
            common_sym = os.path.commonpath([str(real_root), str(resolved_symlink)])
            if common_sym != str(real_root):
                raise CacheSecurityError(
                    f"Symlink traversal detected: {target_path} links to {resolved_symlink}"
                )

        return real_target

    def get_entry_dir(self, vad_config_id: str, recording_id: str) -> Path:
        """Directory for specific VAD and recording with component validation."""
        clean_vad = validate_path_component("vad_config_id", vad_config_id)
        clean_rec = validate_path_component("recording_id", recording_id)

        target_dir = self.cache_root / clean_vad / clean_rec
        self._assert_contained(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def build_filename(self, identity: CacheIdentity) -> str:
        """Constructs canonical readable filename with validated components."""
        vad_hash_16 = identity.vad_intervals_sha256[:16]
        return (
            f"{identity.encoder_name}__{identity.audio_source_id}__"
            f"scale{identity.scale_index}__{vad_hash_16}.npz"
        )

    def save_embeddings(
        self,
        identity: CacheIdentity,
        embeddings: np.ndarray,
        l2_norms: np.ndarray,
        subsegment_starts: np.ndarray,
        subsegment_ends: np.ndarray,
    ) -> Path:
        """
        Atomically saves embeddings and metadata.
        """
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise CacheValidationError(f"Embeddings must be 2D array, got {type(embeddings)}")

        if not np.all(np.isfinite(embeddings)):
            raise CacheValidationError("Embeddings contain NaN or Infinite values.")

        n_rows, dim = embeddings.shape
        if len(l2_norms) != n_rows or len(subsegment_starts) != n_rows or len(subsegment_ends) != n_rows:
            raise CacheValidationError("Vector lengths mismatch subsegment count.")

        target_dir = self.get_entry_dir(identity.vad_config_id, identity.recording_id)
        filename = self.build_filename(identity)
        final_path = target_dir / filename
        self._assert_contained(final_path)

        metadata_json = identity.to_json()

        # Atomic write via mkstemp inside the validated target_dir
        temp_fd, temp_path_str = tempfile.mkstemp(
            prefix="cache_atomic_", suffix=".npz", dir=str(target_dir)
        )
        os.close(temp_fd)
        temp_path = Path(temp_path_str)

        try:
            np.savez_compressed(
                str(temp_path),
                embeddings=embeddings.astype(np.float32),
                l2_norms=l2_norms.astype(np.float32),
                subsegment_starts=subsegment_starts.astype(np.float64),
                subsegment_ends=subsegment_ends.astype(np.float64),
                metadata_json=metadata_json,
            )
            # Atomic replace
            temp_path.replace(final_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return final_path

    def load_embeddings(
        self,
        identity: CacheIdentity,
        expected_dim: int,
        expected_subsegments: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads embeddings and validates complete identity and timestamps.
        Returns: (embeddings, l2_norms)
        """
        target_dir = self.get_entry_dir(identity.vad_config_id, identity.recording_id)
        filename = self.build_filename(identity)
        target_path = target_dir / filename
        self._assert_contained(target_path)

        if not target_path.exists():
            raise FileNotFoundError(f"Cache miss: {target_path}")

        try:
            data = np.load(str(target_path))
        except Exception as e:
            raise CacheValidationError(f"Corrupt cache file {target_path}: {e}")

        # 1. Metadata JSON validation
        if "metadata_json" not in data:
            raise CacheValidationError(f"Cache entry missing metadata_json: {target_path}")

        cached_meta = json.loads(str(data["metadata_json"]))
        current_meta = asdict(identity)

        # 2. Strict identity equality
        for k, v in current_meta.items():
            if cached_meta.get(k) != v:
                raise CacheValidationError(
                    f"Cache identity mismatch on field '{k}': expected {v}, cached {cached_meta.get(k)}"
                )

        # 3. Embedding matrix validation
        embeddings = data["embeddings"]
        l2_norms = data["l2_norms"]
        starts = data["subsegment_starts"]
        ends = data["subsegment_ends"]

        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise CacheValidationError(
                f"Feature dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}"
            )

        if not np.all(np.isfinite(embeddings)):
            raise CacheValidationError("Cached embeddings contain NaN or Infinite values.")

        # 4. Timestamp alignment validation
        if expected_subsegments is not None:
            if len(starts) != len(expected_subsegments):
                raise CacheValidationError(
                    f"Subsegment count mismatch: expected {len(expected_subsegments)}, cached {len(starts)}"
                )
            for i, sub in enumerate(expected_subsegments):
                if not np.isclose(starts[i], sub["start"], atol=1e-4):
                    raise CacheValidationError(
                        f"Subsegment {i} start mismatch: expected {sub['start']}, cached {starts[i]}"
                    )
                if not np.isclose(ends[i], sub["end"], atol=1e-4):
                    raise CacheValidationError(
                        f"Subsegment {i} end mismatch: expected {sub['end']}, cached {ends[i]}"
                    )

        return embeddings, l2_norms
