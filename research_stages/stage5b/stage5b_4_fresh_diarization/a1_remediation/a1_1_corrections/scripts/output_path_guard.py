#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Output Path Guard
Enforces strict containment of output RTTMs, manifests, and reports
within the authorized Stage 5B.4 workspace (or explicit test roots).
Defends against directory traversal, symlink escapes, and prohibited overwrites
of upstream/downstream historical stages (Stages 1-3, 5A, 5B.1-5B.3).
"""

import os
from pathlib import Path
from typing import List, Optional

from path_utils import get_stage5b_4_dir, get_research_stages_dir


class OutputSecurityError(PermissionError):
    """Raised when an output path violates boundary containment or targets forbidden stages."""
    pass


PROHIBITED_STAGE_NAMES = [
    "stage1_data_prep",
    "stage2_baseline",
    "stage3_dual_encoder",
    "stage5a",
    "stage5b_1_vad_optimization",
    "stage5b_2_overlap_optimization",
    "stage5b_3_speaker_count",
]


def validate_output_path(
    output_path: Path,
    allowed_roots: Optional[List[Path]] = None,
    allow_creation: bool = True,
) -> Path:
    """
    Validates that output_path resolves strictly within allowed_roots.
    Default allowed root is research_stages/stage5b/stage5b_4_fresh_diarization.

    Raises:
        OutputSecurityError: On path traversal, symlink escape, or prohibited targets.
    """
    if allowed_roots is None:
        allowed_roots = [get_stage5b_4_dir()]

    # Canonicalize allowed roots
    resolved_roots = [Path(r).resolve() for r in allowed_roots]

    # Resolve target output path
    p = Path(output_path)
    
    # Check for raw string traversal patterns before resolution
    raw_str = str(output_path)
    if "\x00" in raw_str:
        raise OutputSecurityError("Null byte detected in output path.")

    # In case the directory does not exist yet, find the nearest existing parent to check symlinks
    resolved_target = p.resolve()
    stages_root = get_research_stages_dir().resolve()

    # 1. Prohibit targeting any frozen or historical stages
    for prohibited in PROHIBITED_STAGE_NAMES:
        if prohibited in resolved_target.parts:
            raise OutputSecurityError(
                f"Output path escapes into prohibited stage '{prohibited}': {resolved_target}"
            )

    # 2. Strict containment check within allowed roots
    is_contained = any(
        resolved_target == root or resolved_target.is_relative_to(root)
        for root in resolved_roots
    )
    if not is_contained:
        allowed_str = ", ".join(str(r) for r in resolved_roots)
        raise OutputSecurityError(
            f"Output path containment violation: '{resolved_target}' is outside allowed roots: [{allowed_str}]"
        )

    # 3. Symlink inspection on existing path components
    check_curr = p
    while check_curr != check_curr.parent:
        if check_curr.is_symlink():
            link_target = check_curr.resolve()
            if not any(link_target == r or link_target.is_relative_to(r) for r in resolved_roots):
                raise OutputSecurityError(
                    f"Symlink escape detected: {check_curr} points outside allowed workspace to {link_target}"
                )
        check_curr = check_curr.parent

    # 4. Optional directory creation
    if allow_creation:
        resolved_target.mkdir(parents=True, exist_ok=True)

    return resolved_target
