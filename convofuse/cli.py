"""
ConvoFuse Command-Line Interface, Sequential Corpus Runner & Release Packager.

Supports:
1. FROZEN_INPUT_REPLAY (`--mode frozen_replay`, default):
   - Replays audited Stage 1 audio metadata, Stage 2 WhisperX aligned words,
     Stage 5 dual-encoder RTTMs (`E20` for R01 or `Condition_J` / `Condition_K` for R01-R08),
     and Stage 5B.2 frame-level overlap posteriors through ground-truth-isolated
     DiarizationLM-Gemma refinement (`LG-TPST` by default; `--unconstrained` for ablation)
     and chronological transcript reconstruction.
2. CUSTOM_INPUTS / FULL_PROCESSING (`--mode custom_inputs`):
   - Processes a custom audio/video file (`--audio-file`) alongside pre-extracted
     aligned ASR words (`--asr-words-json`) and diarization RTTM (`--rttm-file`),
     performing canonical 16 kHz mono + EBU R128 loudness normalization, word-speaker
     attribution, overlap metadata enrichment, DiarizationLM-Gemma refinement, and
     complete transcript reconstruction without requiring the optional web application.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from convofuse.audio_diarization_adapter import (
    BENCHMARK_RECORDING_MAP,
    REPO_ROOT,
    compute_sha256,
    load_asr_words,
    parse_rttm_file,
)
from convofuse.contracts import ConvoFuseExecutionMode, ConvoFuseTranscriptBundle
from convofuse.pipeline import ConvoFusePipeline

BENCHMARK_RECORDING_IDS: List[str] = [
    "R01_sara_sharif",
    "R02_dennis_skinner",
    "R03_disabled_parking",
    "R04_piers_robinson",
    "R05_southwest_pilot",
    "R06_rich_karen",
    "R07_bwc_cheap_shot",
    "R08_uk_parliament",
]

REQUIRED_RECORDING_ARTIFACTS: Tuple[str, ...] = (
    "transcript.txt",
    "transcript.json",
    "words.json",
    "turns.json",
    "speakers.json",
    "comparison.json",
    "processing_manifest.json",
)


def _get_git_metadata() -> Dict[str, str]:
    commit = "unknown"
    branch = "unknown"
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True)
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=str(REPO_ROOT), text=True
            )
            .strip()
        )
    except Exception:
        pass
    return {"git_commit": commit, "git_branch": branch}


def _verify_and_hash_recording_outputs(rec_out_dir: Path, recording_id: str) -> Dict[str, Dict[str, Any]]:
    """Verify all required per-recording artifacts exist, are non-empty, and compute SHA-256 digests."""
    artifact_hashes: Dict[str, Dict[str, Any]] = {}
    for suffix in REQUIRED_RECORDING_ARTIFACTS:
        fname = f"{recording_id}__{suffix}"
        fpath = rec_out_dir / fname
        if not fpath.exists() or not fpath.is_file():
            raise RuntimeError(f"Missing required output artifact for {recording_id}: {fpath}")
        size_bytes = fpath.stat().st_size
        if size_bytes <= 0:
            raise RuntimeError(f"Empty output artifact for {recording_id}: {fpath}")
        artifact_hashes[fname] = {
            "relative_path": f"{recording_id}/{fname}",
            "size_bytes": size_bytes,
            "sha256": compute_sha256(fpath),
        }

    eval_fname = f"{recording_id}__evaluation_report.json"
    eval_fpath = rec_out_dir / eval_fname
    if eval_fpath.exists() and eval_fpath.is_file() and eval_fpath.stat().st_size > 0:
        artifact_hashes[eval_fname] = {
            "relative_path": f"{recording_id}/{eval_fname}",
            "size_bytes": eval_fpath.stat().st_size,
            "sha256": compute_sha256(eval_fpath),
        }
    return artifact_hashes


def _summarize_bundle_row(
    bundle: ConvoFuseTranscriptBundle,
    rec_out_dir: Path,
    elapsed: float,
) -> Dict[str, Any]:
    eval_rep = bundle.evaluation_report or {}
    base_m = eval_rep.get("baseline_metrics", {})
    ref_m = eval_rep.get("refined_metrics", {})
    outcomes = eval_rep.get("refinement_outcomes", {})
    integrity = eval_rep.get("integrity_verification", {})
    telem = bundle.processing_manifest.get("refinement_telemetry", {})

    base_ts_wder = float(base_m.get("wder_timestamp", {}).get("wder_assigned", 0.0)) * 100.0
    ref_ts_wder = float(ref_m.get("wder_timestamp", {}).get("wder_assigned", 0.0)) * 100.0
    base_lex_wder = float(base_m.get("wder_lexical", {}).get("wder_lexical", 0.0)) * 100.0
    ref_lex_wder = float(ref_m.get("wder_lexical", {}).get("wder_lexical", 0.0)) * 100.0
    base_cpwer = float(base_m.get("cpwer", 0.0)) * 100.0
    ref_cpwer = float(ref_m.get("cpwer", 0.0)) * 100.0

    word_seq_ok = bool(integrity.get("word_sequence_intact", True))
    ts_seq_ok = bool(integrity.get("timestamp_sequence_intact", True))
    unk_ok = bool(integrity.get("unk_invariance_verified", True))
    sha_ok = bool(bundle.processing_manifest.get("frozen_inputs_sha256_verified", True))

    if not (word_seq_ok and ts_seq_ok and unk_ok and sha_ok):
        raise RuntimeError(
            f"Integrity check failed for {bundle.recording_id}: "
            f"word_seq={word_seq_ok}, ts_seq={ts_seq_ok}, unk={unk_ok}, sha256={sha_ok}"
        )

    return {
        "recording_id": bundle.recording_id,
        "duration_sec": bundle.duration_sec,
        "execution_mode": bundle.execution_mode,
        "diarization_condition": bundle.diarization_condition,
        "diarization_backend": bundle.diarization_backend,
        "refinement_model": bundle.refinement_model,
        "word_count": len(bundle.words),
        "baseline_turn_count": len(bundle.baseline_turns),
        "refined_turn_count": len(bundle.turns),
        "overlapping_turn_count": sum(1 for t in bundle.turns if t.has_overlapping_speech),
        "overlapping_word_count": sum(
            1 for w in bundle.words if w.overlap_metadata.is_overlapping_speech
        ),
        "unk_word_count": sum(1 for w in bundle.words if w.refined_speaker_id == "UNK"),
        "speaker_inventory": list(bundle.speaker_inventory),
        "baseline_wder_ts_pct": round(base_ts_wder, 2),
        "refined_wder_ts_pct": round(ref_ts_wder, 2),
        "delta_wder_ts_pp": round(ref_ts_wder - base_ts_wder, 2),
        "baseline_wder_lex_pct": round(base_lex_wder, 2),
        "refined_wder_lex_pct": round(ref_lex_wder, 2),
        "delta_wder_lex_pp": round(ref_lex_wder - base_lex_wder, 2),
        "baseline_cpwer_pct": round(base_cpwer, 2),
        "refined_cpwer_pct": round(ref_cpwer, 2),
        "delta_cpwer_pp": round(ref_cpwer - base_cpwer, 2),
        "accepted_corrections": outcomes.get("accepted_corrections", 0),
        "rejected_proposals": outcomes.get("rejected_proposals", 0),
        "abstentions": outcomes.get("abstentions", 0),
        "unchanged_words": outcomes.get("unchanged_words", 0),
        "correction_precision": outcomes.get("correction_precision"),
        "beneficial_corrections": outcomes.get("beneficial_corrections", 0),
        "harmful_corrections": outcomes.get("harmful_corrections", 0),
        "neutral_corrections": outcomes.get("neutral_corrections", 0),
        "reasons_breakdown": outcomes.get("reasons_breakdown", {}),
        "word_sequence_intact": word_seq_ok,
        "timestamp_sequence_intact": ts_seq_ok,
        "unk_invariance_verified": unk_ok,
        "frozen_inputs_sha256_verified": sha_ok,
        "llm_runtime_sec": round(float(telem.get("total_runtime_sec", 0.0)), 4),
        "prompt_tokens": int(telem.get("total_prompt_tokens", 0)),
        "completion_tokens": int(telem.get("total_completion_tokens", 0)),
        "total_pipeline_sec": round(elapsed, 4),
        "output_dir": str(rec_out_dir),
    }


def write_release_packaging(
    output_dir: Path,
    summary: Dict[str, Any],
    bundles: Sequence[ConvoFuseTranscriptBundle],
) -> Dict[str, str]:
    """
    Writes Phase 6 release packaging files into `output_dir`:
    - `CORPUS_SUMMARY.json`: Per-recording and macro evaluation summary.
    - `CORPUS_INDEX.json`: Clean corpus index linking all processed recordings and artifacts.
    - `RELEASE_MANIFEST.json`: Provenance, condition, model, timestamp, input SHA-256, and output SHA-256 manifest.
    """
    output_dir = output_dir.resolve()
    git_meta = _get_git_metadata()
    now_iso = datetime.now(timezone.utc).isoformat()

    corpus_index_entries: List[Dict[str, Any]] = []
    manifest_recordings: List[Dict[str, Any]] = []

    for bundle in bundles:
        rec_id = bundle.recording_id
        rec_out_dir = output_dir / rec_id
        artifact_hashes = _verify_and_hash_recording_outputs(rec_out_dir, rec_id)

        corpus_index_entries.append(
            {
                "recording_id": rec_id,
                "duration_sec": bundle.duration_sec,
                "diarization_condition": bundle.diarization_condition,
                "refinement_model": bundle.refinement_model,
                "speaker_inventory": list(bundle.speaker_inventory),
                "word_count": len(bundle.words),
                "turn_count": len(bundle.turns),
                "overlapping_turn_count": sum(
                    1 for t in bundle.turns if t.has_overlapping_speech
                ),
                "unk_word_count": sum(
                    1 for w in bundle.words if w.refined_speaker_id == "UNK"
                ),
                "primary_transcript_txt": f"{rec_id}/{rec_id}__transcript.txt",
                "primary_transcript_json": f"{rec_id}/{rec_id}__transcript.json",
                "words_json": f"{rec_id}/{rec_id}__words.json",
                "turns_json": f"{rec_id}/{rec_id}__turns.json",
                "speakers_json": f"{rec_id}/{rec_id}__speakers.json",
                "comparison_json": f"{rec_id}/{rec_id}__comparison.json",
                "processing_manifest_json": f"{rec_id}/{rec_id}__processing_manifest.json",
                "evaluation_report_json": (
                    f"{rec_id}/{rec_id}__evaluation_report.json"
                    if f"{rec_id}__evaluation_report.json" in artifact_hashes
                    else None
                ),
            }
        )

        manifest_recordings.append(
            {
                "recording_id": rec_id,
                "execution_mode": bundle.execution_mode,
                "diarization_condition": bundle.diarization_condition,
                "diarization_backend": bundle.diarization_backend,
                "refinement_model": bundle.refinement_model,
                "created_at": bundle.processing_manifest.get("created_at", now_iso),
                "ground_truth_isolation_enforced": bundle.processing_manifest.get(
                    "ground_truth_isolation_enforced", True
                ),
                "frozen_inputs_sha256_verified": bundle.processing_manifest.get(
                    "frozen_inputs_sha256_verified", True
                ),
                "frozen_input_sha256_hashes": bundle.processing_manifest.get(
                    "input_sha256_hashes", {}
                ),
                "output_artifacts": artifact_hashes,
            }
        )

    summary_json_path = output_dir / "CORPUS_SUMMARY.json"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    corpus_index = {
        "project": "Multi-View-Speaker-Representation-Learning",
        "framework": "ConvoFuse",
        "created_at": now_iso,
        "recording_count": len(corpus_index_entries),
        "recordings": corpus_index_entries,
    }
    corpus_index_path = output_dir / "CORPUS_INDEX.json"
    corpus_index_path.write_text(
        json.dumps(corpus_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    release_manifest = {
        "project": "Multi-View-Speaker-Representation-Learning",
        "framework": "ConvoFuse",
        "release_created_at": now_iso,
        "git_commit": git_meta["git_commit"],
        "git_branch": git_meta["git_branch"],
        "python_executable": sys.executable,
        "condition_preference": summary.get("condition_preference"),
        "use_locality_gating": summary.get("use_locality_gating"),
        "recording_count": len(manifest_recordings),
        "corpus_summary_sha256": compute_sha256(summary_json_path),
        "corpus_index_sha256": compute_sha256(corpus_index_path),
        "recordings": manifest_recordings,
    }
    release_manifest_path = output_dir / "RELEASE_MANIFEST.json"
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "corpus_summary_json": str(summary_json_path),
        "corpus_index_json": str(corpus_index_path),
        "release_manifest_json": str(release_manifest_path),
    }


def run_corpus_benchmark(
    recording_ids: List[str],
    condition_preference: str,
    use_locality_gating: bool,
    output_dir: Path,
    pipeline: Optional[ConvoFusePipeline] = None,
) -> Dict[str, Any]:
    if not recording_ids:
        raise ValueError("recording_ids must not be empty.")

    active_pipeline = pipeline or ConvoFusePipeline(use_locality_gating=use_locality_gating)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_recording_results: List[Dict[str, Any]] = []
    bundles: List[ConvoFuseTranscriptBundle] = []
    t_corpus_start = time.perf_counter()

    for rec_id in recording_ids:
        canonical_id = BENCHMARK_RECORDING_MAP.get(rec_id, rec_id)
        if canonical_id not in BENCHMARK_RECORDING_IDS:
            raise ValueError(
                f"Unknown benchmark recording_id '{rec_id}'. "
                f"Expected one of {BENCHMARK_RECORDING_IDS} or R01..R08."
            )
        rec_out_dir = output_dir / canonical_id
        rec_out_dir.mkdir(parents=True, exist_ok=True)
        mode_name = "LG-TPST" if use_locality_gating else "Unconstrained-TPST"
        print(
            f"[ConvoFuse] Starting {canonical_id} "
            f"(condition_preference={condition_preference}, mode={mode_name})..."
        )
        t0 = time.perf_counter()
        bundle = active_pipeline.run_frozen_replay(
            recording_id=canonical_id,
            condition_preference=condition_preference,
            enable_refinement=True,
            use_locality_gating=use_locality_gating,
            output_dir=rec_out_dir,
        )
        elapsed = time.perf_counter() - t0
        row = _summarize_bundle_row(bundle, rec_out_dir, elapsed)
        per_recording_results.append(row)
        bundles.append(bundle)
        print(
            f"[ConvoFuse] Finished {canonical_id} in {elapsed:.2f}s | "
            f"TS-WDER: {row['baseline_wder_ts_pct']:.2f}% -> {row['refined_wder_ts_pct']:.2f}% "
            f"(Δ={row['delta_wder_ts_pp']:+.2f}pp) | "
            f"Lex-WDER: {row['baseline_wder_lex_pct']:.2f}% -> {row['refined_wder_lex_pct']:.2f}% "
            f"(Δ={row['delta_wder_lex_pp']:+.2f}pp) | "
            f"cpWER: {row['baseline_cpwer_pct']:.2f}% -> {row['refined_cpwer_pct']:.2f}% "
            f"(Δ={row['delta_cpwer_pp']:+.2f}pp) | "
            f"acc={row['accepted_corrections']}, rej={row['rejected_proposals']}, abs={row['abstentions']}"
        )

    if len(per_recording_results) != len(recording_ids):
        raise RuntimeError(
            f"Incomplete corpus run: expected {len(recording_ids)} recordings, "
            f"completed {len(per_recording_results)}"
        )

    total_corpus_sec = time.perf_counter() - t_corpus_start
    n = len(per_recording_results)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording_count": n,
        "condition_preference": condition_preference,
        "use_locality_gating": use_locality_gating,
        "total_corpus_sec": round(total_corpus_sec, 4),
        "macro_averages": {
            "mean_baseline_wder_ts_pct": round(
                sum(r["baseline_wder_ts_pct"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_refined_wder_ts_pct": round(
                sum(r["refined_wder_ts_pct"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_delta_wder_ts_pp": round(
                sum(r["delta_wder_ts_pp"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_baseline_wder_lex_pct": round(
                sum(r["baseline_wder_lex_pct"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_refined_wder_lex_pct": round(
                sum(r["refined_wder_lex_pct"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_delta_wder_lex_pp": round(
                sum(r["delta_wder_lex_pp"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_baseline_cpwer_pct": round(
                sum(r["baseline_cpwer_pct"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_refined_cpwer_pct": round(
                sum(r["refined_cpwer_pct"] for r in per_recording_results) / max(n, 1), 2
            ),
            "mean_delta_cpwer_pp": round(
                sum(r["delta_cpwer_pp"] for r in per_recording_results) / max(n, 1), 2
            ),
            "total_accepted_corrections": sum(
                r["accepted_corrections"] for r in per_recording_results
            ),
            "total_rejected_proposals": sum(
                r["rejected_proposals"] for r in per_recording_results
            ),
            "total_abstentions": sum(r["abstentions"] for r in per_recording_results),
            "total_unchanged_words": sum(r["unchanged_words"] for r in per_recording_results),
            "total_beneficial_corrections": sum(
                r["beneficial_corrections"] for r in per_recording_results
            ),
            "total_harmful_corrections": sum(
                r["harmful_corrections"] for r in per_recording_results
            ),
            "total_neutral_corrections": sum(
                r["neutral_corrections"] for r in per_recording_results
            ),
            "all_word_sequences_preserved": all(
                r["word_sequence_intact"] for r in per_recording_results
            ),
            "all_timestamps_preserved": all(
                r["timestamp_sequence_intact"] for r in per_recording_results
            ),
            "all_unk_invariance_verified": all(
                r["unk_invariance_verified"] for r in per_recording_results
            ),
            "all_frozen_inputs_sha256_verified": all(
                r["frozen_inputs_sha256_verified"] for r in per_recording_results
            ),
        },
        "recordings": per_recording_results,
    }

    write_release_packaging(output_dir=output_dir, summary=summary, bundles=bundles)
    return summary


def run_custom_inputs_cli(
    recording_id: str,
    audio_file: Path,
    asr_words_json: Path,
    rttm_file: Path,
    overlap_npz: Optional[Path],
    use_locality_gating: bool,
    output_dir: Path,
    pipeline: Optional[ConvoFusePipeline] = None,
) -> Dict[str, Any]:
    """
    Execute ConvoFuse FULL_PROCESSING mode from CLI given a media file, aligned ASR words JSON,
    and diarization RTTM (without requiring the optional FastAPI/Streamlit app).
    """
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio/video file not found: {audio_file}")
    if not asr_words_json.exists():
        raise FileNotFoundError(f"ASR aligned words JSON not found: {asr_words_json}")
    if not rttm_file.exists():
        raise FileNotFoundError(f"Diarization RTTM file not found: {rttm_file}")

    active_pipeline = pipeline or ConvoFusePipeline(use_locality_gating=use_locality_gating)
    output_dir = output_dir.resolve()
    rec_out_dir = output_dir / recording_id
    rec_out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = rec_out_dir / "_work"

    raw_words = load_asr_words(asr_words_json)
    diar_segments = parse_rttm_file(rttm_file)

    t0 = time.perf_counter()
    bundle = active_pipeline.run_custom_or_uploaded(
        recording_id=recording_id,
        filename=audio_file.name,
        source_media_path=audio_file,
        raw_words=raw_words,
        diar_segments=diar_segments,
        work_dir=work_dir,
        execution_mode=ConvoFuseExecutionMode.FULL_PROCESSING,
        diarization_condition="Custom_Full_Processing",
        diarization_backend="Custom_RTTM_Input",
        overlap_npz_path=overlap_npz,
        enable_refinement=True,
        use_locality_gating=use_locality_gating,
        output_dir=rec_out_dir,
    )
    elapsed = time.perf_counter() - t0
    row = _summarize_bundle_row(bundle, rec_out_dir, elapsed)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording_count": 1,
        "condition_preference": "Custom_Full_Processing",
        "use_locality_gating": use_locality_gating,
        "total_corpus_sec": round(elapsed, 4),
        "recordings": [row],
    }
    write_release_packaging(output_dir=output_dir, summary=summary, bundles=[bundle])
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ConvoFuse CLI Runner — Multi-View Speaker Representation Learning & Generative Refinement"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frozen_replay", "custom_inputs"],
        default="frozen_replay",
        help="Execution mode: 'frozen_replay' (default, R01-R08 audited artifacts) or 'custom_inputs' (new media + ASR JSON + RTTM)",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default="R01_sara_sharif",
        help="Recording ID (e.g., R01 or R01_sara_sharif for frozen_replay, or custom ID for custom_inputs)",
    )
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="Run all 8 benchmark recordings R01..R08 sequentially (frozen_replay mode)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="auto",
        help="Diarization condition preference (auto, E20, Condition_J, Condition_K)",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Disable LG-TPST locality gating (run unconstrained TPST ablation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for exported transcripts, artifacts, and release manifest",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="[custom_inputs mode] Path to input audio/video recording",
    )
    parser.add_argument(
        "--asr-words-json",
        type=str,
        default=None,
        help="[custom_inputs mode] Path to aligned ASR words JSON",
    )
    parser.add_argument(
        "--rttm-file",
        type=str,
        default=None,
        help="[custom_inputs mode] Path to predicted speaker diarization RTTM",
    )
    parser.add_argument(
        "--overlap-npz",
        type=str,
        default=None,
        help="[custom_inputs mode] Optional path to frame-level overlap posterior .npz",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    try:
        if args.mode == "custom_inputs":
            if not args.audio_file or not args.asr_words_json or not args.rttm_file:
                parser.error(
                    "--mode custom_inputs requires --audio-file, --asr-words-json, and --rttm-file"
                )
            run_custom_inputs_cli(
                recording_id=args.recording,
                audio_file=Path(args.audio_file).resolve(),
                asr_words_json=Path(args.asr_words_json).resolve(),
                rttm_file=Path(args.rttm_file).resolve(),
                overlap_npz=Path(args.overlap_npz).resolve() if args.overlap_npz else None,
                use_locality_gating=not args.unconstrained,
                output_dir=out_dir,
            )
        else:
            target_ids = (
                list(BENCHMARK_RECORDING_IDS) if args.all_recordings else [args.recording]
            )
            run_corpus_benchmark(
                recording_ids=target_ids,
                condition_preference=args.condition,
                use_locality_gating=not args.unconstrained,
                output_dir=out_dir,
            )
    except Exception as exc:
        print(f"[ConvoFuse ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
