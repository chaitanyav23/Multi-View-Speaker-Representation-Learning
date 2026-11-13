#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.1 Remediation: Isolated Embedding Cache Manager
Strictly isolates Stage 5B.4 embedding cache from Stage 3 cache.
Defends against cache collisions across alternative VAD segmentations.
Provides distinct classification and tracking for:
- Cache HIT
- Cache MISS
- Cache INVALIDATION
- Security violations
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from path_utils import get_stage5b_4_cache_dir, get_stage5b_4_dir, get_research_stages_dir
from checkpoint_preflight import ExecutionMode

A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
import sys
if str(A1_SCRIPTS) not in sys.path:
    sys.path.append(str(A1_SCRIPTS))

from isolated_embedding_cache import (
    CacheIdentity,
    CacheSecurityError,
    CacheValidationError,
    IsolatedEmbeddingCacheManager,
    CACHE_SCHEMA_VERSION,
)


class IsolatedEmbeddingCacheManagerRemediated(IsolatedEmbeddingCacheManager):
    """
    Enhanced IsolatedEmbeddingCacheManager ensuring distinct status codes:
    'HIT', 'MISS', 'INVALIDATION', and raising CacheSecurityError on isolation violations.
    """

    def __init__(self, cache_root: Optional[Path] = None, allow_custom_root: bool = False):
        self.cache_root = Path(cache_root or get_stage5b_4_cache_dir()).resolve()
        self._assert_safe_cache_root_remediated(self.cache_root, allow_custom_root=allow_custom_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_safe_cache_root_remediated(path: Path, allow_custom_root: bool = False) -> None:
        """Enforces that path is safe and strictly isolated from prohibited directories."""
        real_path = Path(os.path.realpath(str(path)))
        prohibited_stage3 = get_research_stages_dir() / "stage3_dual_encoder" / "cache"
        real_prohibited = Path(os.path.realpath(str(prohibited_stage3)))

        # 1. Prohibit Stage 3 cache or any subdirectory
        if real_path == real_prohibited or real_prohibited in real_path.parents or real_path in real_prohibited.parents:
            raise CacheSecurityError(
                f"FATAL: Attempted to use prohibited Stage 3 cache directory: {real_path}"
            )

        # 2. If not custom test root, must be inside Stage 5B.4 workspace
        if not allow_custom_root:
            permitted_ancestor = Path(os.path.realpath(str(get_stage5b_4_dir())))
            try:
                common = os.path.commonpath([str(permitted_ancestor), str(real_path)])
                if common != str(permitted_ancestor):
                    raise CacheSecurityError(
                        f"FATAL: Cache root {real_path} escapes permitted Stage 5B.4 workspace {permitted_ancestor}"
                    )
            except ValueError:
                raise CacheSecurityError(
                    f"FATAL: Cache root {real_path} is on a different drive/volume than {permitted_ancestor}"
                )

    def get_cache_path(self, identity: CacheIdentity) -> Path:
        """Returns the canonical path for an embedding cache file and asserts containment."""
        target_dir = self.get_entry_dir(identity.vad_config_id, identity.recording_id)
        filename = self.build_filename(identity)
        target_path = target_dir / filename
        return self._assert_contained(target_path)

    def load_or_status(
        self,
        identity: CacheIdentity,
        expected_dim: int,
        expected_subsegments: List[Dict[str, Any]],
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
        """
        Loads embeddings and returns a tuple of ((embeddings, norms), status).
        Status is strictly one of:
        - 'HIT': Cache exists and completely valid.
        - 'MISS': Cache does not exist on disk.
        - 'INVALIDATION': Cache exists but schema, dimension, count, or boundaries are invalid.
        """
        npz_path = self.get_cache_path(identity)
        if not npz_path.exists():
            return None, "MISS"

        try:
            embs, norms = self.load_embeddings(identity, expected_dim, expected_subsegments)
            return (embs, norms), "HIT"
        except CacheValidationError:
            return None, "INVALIDATION"
        except Exception:
            return None, "INVALIDATION"
