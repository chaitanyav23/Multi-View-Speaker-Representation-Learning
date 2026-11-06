#!/usr/bin/env python3
"""
Stage 5B.4-A.1.2 Remediation: Path Resolution Utilities
Centralized repository root and subsystem path resolution.
Guarantees consistent, non-duplicated paths regardless of caller location or nesting depth.
"""

from pathlib import Path
from typing import Optional


def get_repo_root(start_path: Optional[Path] = None) -> Path:
    """
    Finds repository root by searching upwards from start_path (or __file__)
    for the '.git' directory or marker.
    Falls back to locating 'research_stages' and 'phase20_standalone'.
    """
    if start_path is None:
        current = Path(__file__).resolve().parent
    else:
        current = Path(start_path).resolve()
        if current.is_file():
            current = current.parent

    # Traverse upward searching for .git
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent.resolve()

    # Fallback: look for marker 'research_stages' where parent is repo root
    for parent in [current] + list(current.parents):
        if (parent / "research_stages").is_dir() and (parent / "phase20_standalone").is_dir():
            return parent.resolve()

    raise RuntimeError(
        f"Could not determine repository root from starting path: {start_path or __file__}"
    )


def get_research_stages_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to research_stages directory without duplication."""
    return get_repo_root(root) / "research_stages"


def get_stage5b_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to research_stages/stage5b directory."""
    return get_research_stages_dir(root) / "stage5b"


def get_stage5b_4_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to Stage 5B.4 workspace."""
    return get_stage5b_dir(root) / "stage5b_4_fresh_diarization"


def get_stage5b_4_cache_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to Stage 5B.4 isolated embedding cache root."""
    return get_stage5b_4_dir(root) / "cache"


def get_stage5b_4_output_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to Stage 5B.4 authoritative outputs root."""
    return get_stage5b_4_dir(root) / "outputs"


def get_stage2_posteriors_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to Stage 2 baseline posteriors directory."""
    return get_research_stages_dir(root) / "stage2_baseline" / "posteriors"


def get_stage5b_2_overlap_cache_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to Stage 5B.2 overlap cache directory."""
    return get_stage5b_dir(root) / "stage5b_2_overlap_optimization" / "overlap_cache"


def get_phase20_scripts_dir(root: Optional[Path] = None) -> Path:
    """Returns absolute path to phase20_standalone/scripts directory."""
    return get_repo_root(root) / "phase20_standalone" / "scripts"
