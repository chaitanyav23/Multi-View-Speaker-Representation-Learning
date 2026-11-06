#!/usr/bin/env python3
"""
Stage 5B.4 Fresh Diarization Main Orchestrator
CLI entry point coordinating VAD interval extraction, dynamic subsegmentation,
dual-encoder embedding extraction with isolated cache, Condition J fusion,
fresh NME-SC clustering, overlap attribution, and downstream evaluation.

CRITICAL EXECUTION SAFETY MECHANISM:
The 'run' mode is strictly guarded by '--authorize-inference'.
In Stage 5B.4-A, model inference is strictly PROHIBITED.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure local script directory is on path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from condition_registry import CONDITIONS, BENCHMARK_RECORDINGS, validate_registry
from embedding_cache import EmbeddingCacheManager, CACHE_SCHEMA_VERSION
from vad_adapter import VADAdapter
from subsegment_adapter import generate_all_scales


SAFETY_BANNER = """
================================================================================
  STAGE 5B.4 FRESH DIARIZATION ORCHESTRATOR
  Execution Safety Barrier Active: Model inference and audio extraction
  require explicit '--authorize-inference' authorization.
================================================================================
"""


def mode_validate_config() -> int:
    """Validates condition registry and parameter consistency."""
    print("[Validate Config] Checking Stage 5B.4 Condition Registry...")
    try:
        validate_registry()
        print(f"  [OK] Registry valid: 6 conditions registered ({list(CONDITIONS.keys())})")
        print(f"  [OK] Benchmark recordings valid: 8 recordings registered ({BENCHMARK_RECORDINGS})")
        return 0
    except Exception as e:
        print(f"  [FAIL] Registry validation failed: {e}", file=sys.stderr)
        return 1


def mode_validate_cache() -> int:
    """Validates isolated cache setup and prohibited path isolation."""
    print("[Validate Cache] Checking Stage 5B.4 Embedding Cache...")
    try:
        cache_mgr = EmbeddingCacheManager()
        print(f"  [OK] Isolated cache root: {cache_mgr.cache_root}")
        print(f"  [OK] Cache schema version: {CACHE_SCHEMA_VERSION}")
        print("  [OK] Stage 3 prohibited path check passed.")
        return 0
    except Exception as e:
        print(f"  [FAIL] Cache validation failed: {e}", file=sys.stderr)
        return 1


def mode_test_synthetic() -> int:
    """Executes synthetic non-inference pipeline verification."""
    print("[Test Synthetic] Running non-inference synthetic pipeline verification...")
    from affinity_fusion import compute_condition_j_fusion, offdiag_zscore_affinity
    from fresh_clustering import cluster_fresh_affinity
    from overlap_adapter import apply_overlap_attribution
    from rttm_adapter import subsegments_to_single_speaker_segments
    import torch
    import numpy as np

    try:
        # 1. Synthetic VAD intervals
        mock_vad = [
            {"start": 0.0, "end": 2.5, "duration": 2.5},
            {"start": 3.0, "end": 6.0, "duration": 3.0},
        ]
        # 2. Dynamic subsegments
        all_scales = generate_all_scales(mock_vad)
        base_subsegs = all_scales[4]
        n_base = len(base_subsegs)
        print(f"  [OK] Generated {n_base} base-scale subsegments")

        # 3. Synthetic affinities
        rng = np.random.default_rng(42)
        mock_a1 = torch.from_numpy(rng.uniform(0.1, 0.9, (n_base, n_base)).astype(np.float32))
        mock_a2 = torch.from_numpy(rng.uniform(0.1, 0.9, (n_base, n_base)).astype(np.float32))
        mock_a1 = 0.5 * (mock_a1 + mock_a1.t())
        mock_a2 = 0.5 * (mock_a2 + mock_a2.t())

        # 4. Condition J fusion
        A_eq, _, _ = compute_condition_j_fusion(mock_a1, mock_a2)
        print("  [OK] Condition J equal fusion verified")

        # 5. Fresh clustering
        labels, k_est = cluster_fresh_affinity(A_eq, base_subsegs, use_nemo_cuda=False)
        print(f"  [OK] Fresh clustering verified: {k_est} speakers estimated")

        # 6. Single-speaker segments
        single_segs = subsegments_to_single_speaker_segments(base_subsegs, labels)
        print(f"  [OK] RTTM segments assembled: {len(single_segs)} segments")

        # 7. Overlap attribution (M4)
        mock_f_times = np.linspace(0.0, 6.0, 300)
        mock_p_ovl = np.zeros(300, dtype=np.float32)
        mock_p_ovl[100:120] = 0.85  # Overlap event between 2.0s and 2.4s

        final_segs, stats = apply_overlap_attribution(
            single_speaker_segments=single_segs,
            p_overlap=mock_p_ovl,
            frame_times=mock_f_times,
            method_id="M4_conservative_abstention",
        )
        print(f"  [OK] Overlap attribution verified: stats={stats}")

        print("  [SUCCESS] All synthetic pipeline steps executed cleanly with zero model inference.")
        return 0
    except Exception as e:
        print(f"  [FAIL] Synthetic pipeline test failed: {e}", file=sys.stderr)
        return 1


def mode_run_inference(args: argparse.Namespace) -> int:
    """
    Executes real inference pipeline.
    PROTECTED: Must be explicitly authorized via --authorize-inference.
    """
    if not args.authorize_inference:
        print(
            "ERROR: Execution of 'run' mode is BLOCKED!\n"
            "Stage 5B.4-A does NOT authorize real model inference or benchmark execution.\n"
            "To run inference, the project owner must provide separate authorization\n"
            "and pass the explicit '--authorize-inference' flag.",
            file=sys.stderr,
        )
        raise PermissionError(
            "Model inference execution is strictly unauthorized in Stage 5B.4-A."
        )

    # Even if authorized in future stages, check required options
    if not args.recordings or not args.conditions:
        print("ERROR: --recordings and --conditions must be explicitly specified.", file=sys.stderr)
        return 1

    print("[RUN MODE] Authorized inference starting for:")
    print(f"  Recordings: {args.recordings}")
    print(f"  Conditions: {args.conditions}")
    # Inference logic will execute here once authorized in Stage 5B.4-B / Phase B
    return 0


def main() -> None:
    print(SAFETY_BANNER)

    parser = argparse.ArgumentParser(
        description="Stage 5B.4 Fresh Diarization Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["validate-config", "validate-cache", "test-synthetic", "run"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--authorize-inference",
        action="store_true",
        default=False,
        help="Explicit authorization flag required to execute model inference in 'run' mode.",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="List of recording IDs to process (e.g. R01_sara_sharif).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="List of condition IDs to process (e.g. E00 E01 E21).",
    )

    args = parser.parse_args()

    if args.mode == "validate-config":
        sys.exit(mode_validate_config())
    elif args.mode == "validate-cache":
        sys.exit(mode_validate_cache())
    elif args.mode == "test-synthetic":
        sys.exit(mode_test_synthetic())
    elif args.mode == "run":
        try:
            sys.exit(mode_run_inference(args))
        except PermissionError as pe:
            print(f"[ACCESS DENIED] {pe}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
