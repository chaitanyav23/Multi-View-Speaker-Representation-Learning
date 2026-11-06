#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Cache Manager with Precise Error Handling
Re-exports and wraps IsolatedEmbeddingCacheManager with robust path resolution
and precise error categorization (miss vs. invalidation vs. security violation).
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from path_utils import get_stage5b_4_cache_dir, get_stage5b_4_dir, get_research_stages_dir

# Import base cache implementation from A.1
A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from isolated_embedding_cache import (
    CACHE_SCHEMA_VERSION,
    CacheIdentity,
    CacheSecurityError,
    CacheValidationError,
    IsolatedEmbeddingCacheManager as BaseCacheManager,
    validate_path_component,
)


class IsolatedEmbeddingCacheManagerRemediated(BaseCacheManager):
    """
    Subclass of IsolatedEmbeddingCacheManager using centralized path resolution
    and providing explicit query operations that distinguish misses from corruptions.
    """

    def __init__(self, cache_root: Optional[Path] = None, allow_custom_root: bool = False):
        root = Path(cache_root or get_stage5b_4_cache_dir())
        self.cache_root = Path(os.path.realpath(str(root)))
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
            except ValueError as e:
                raise CacheSecurityError(f"FATAL: Cache path outside workspace drive/root: {e}")

    def load_or_status(
        self,
        identity: CacheIdentity,
        expected_dim: int,
        expected_subsegments: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
        """
        Loads embeddings with precise error classification.

        Returns:
            ((embeddings, l2_norms), "HIT") on cache hit
            (None, "MISS") on FileNotFoundError
            (None, "INVALIDATION") on CacheValidationError (corruption or identity mismatch)

        Raises:
            CacheSecurityError: Immediately if an escape or prohibited path is attempted.
        """
        try:
            embs, norms = self.load_embeddings(
                identity=identity,
                expected_dim=expected_dim,
                expected_subsegments=expected_subsegments,
            )
            return (embs, norms), "HIT"
        except FileNotFoundError:
            return None, "MISS"
        except CacheValidationError:
            return None, "INVALIDATION"
        except CacheSecurityError:
            # Security violations must never be treated as misses or invalidations
            raise
