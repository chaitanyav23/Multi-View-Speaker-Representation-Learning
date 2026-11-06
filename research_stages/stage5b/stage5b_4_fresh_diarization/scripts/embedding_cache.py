#!/usr/bin/env python3
"""
Stage 5B.4 VAD-Aware Isolated Embedding Cache
Implements cryptographic cache validation, atomic writes, and schema verification.
Enforces complete isolation from Stage 3 cache directories.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from embedding_extractor import validate_embedding_matrix


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE5B_4_CACHE_ROOT = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_4_fresh_diarization" / "cache"
PROHIBITED_STAGE3_CACHE = REPO_ROOT / "research_stages" / "stage3_dual_encoder" / "cache"

CACHE_SCHEMA_VERSION = "5b4_v1"


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

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class EmbeddingCacheManager:
    """
    Manages loading, storing, and validating multi-scale speaker embeddings.
    Guarantees isolation from historical Stage 3 caches.
    """

    def __init__(self, cache_root: Optional[Path] = None):
        self.cache_root = Path(cache_root or STAGE5B_4_CACHE_ROOT).resolve()
        # Verify cache root does not overlap with Stage 3
        self._assert_safe_cache_root(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_safe_cache_root(path: Path) -> None:
        """Strictly prohibit using Stage 3 cache directory."""
        resolved_path = path.resolve()
        resolved_prohibited = PROHIBITED_STAGE3_CACHE.resolve()
        if resolved_prohibited in resolved_path.parents or resolved_path == resolved_prohibited:
            raise PermissionError(
                f"FATAL: Attempted to use prohibited Stage 3 cache directory: {resolved_path}"
            )

    def get_entry_dir(self, vad_config_id: str, recording_id: str) -> Path:
        """Directory for specific VAD and recording."""
        d = self.cache_root / vad_config_id / recording_id
        self._assert_safe_cache_root(d)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def build_filename(self, identity: CacheIdentity) -> str:
        """Constructs canonical readable filename with truncated hash."""
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
        Atomically saves embeddings and metadata to an isolated cache entry.
        """
        # Validate inputs
        validate_embedding_matrix(embeddings, expected_dim=embeddings.shape[1], expected_count=len(subsegment_starts))
        if len(l2_norms) != len(subsegment_starts) or len(subsegment_ends) != len(subsegment_starts):
            raise ValueError("Mismatched metadata vector lengths")

        target_dir = self.get_entry_dir(identity.vad_config_id, identity.recording_id)
        filename = self.build_filename(identity)
        final_path = target_dir / filename
        self._assert_safe_cache_root(final_path)

        metadata_dict = asdict(identity)
        metadata_json = identity.to_json()

        # Atomic write via named temporary file in the same directory
        temp_fd, temp_path_str = tempfile.mkstemp(
            prefix="cache_write_", suffix=".npz", dir=str(target_dir)
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

        if not target_path.exists():
            raise FileNotFoundError(f"Cache miss: {target_path}")

        try:
            data = np.load(str(target_path))
        except Exception as e:
            raise ValueError(f"Corrupt cache file {target_path}: {e}")

        # 1. Metadata JSON validation
        if "metadata_json" not in data:
            raise ValueError(f"Cache entry missing metadata_json: {target_path}")

        cached_meta = json.loads(str(data["metadata_json"]))
        current_meta = asdict(identity)

        # 2. Strict identity equality
        for k, v in current_meta.items():
            if cached_meta.get(k) != v:
                raise ValueError(
                    f"Cache identity mismatch on field '{k}': "
                    f"expected {v}, cached {cached_meta.get(k)}"
                )

        # 3. Embedding matrix validation
        embeddings = data["embeddings"]
        l2_norms = data["l2_norms"]
        starts = data["subsegment_starts"]
        ends = data["subsegment_ends"]

        validate_embedding_matrix(embeddings, expected_dim=expected_dim, expected_count=len(starts))

        # 4. Timestamp alignment validation against expected subsegments
        if expected_subsegments is not None:
            if len(starts) != len(expected_subsegments):
                raise ValueError(
                    f"Subsegment count mismatch: expected {len(expected_subsegments)}, cached {len(starts)}"
                )
            for i, sub in enumerate(expected_subsegments):
                if not np.isclose(starts[i], sub["start"], atol=1e-4):
                    raise ValueError(
                        f"Subsegment {i} start mismatch: expected {sub['start']}, cached {starts[i]}"
                    )
                if not np.isclose(ends[i], sub["end"], atol=1e-4):
                    raise ValueError(
                        f"Subsegment {i} end mismatch: expected {sub['end']}, cached {ends[i]}"
                    )

        return embeddings, l2_norms
