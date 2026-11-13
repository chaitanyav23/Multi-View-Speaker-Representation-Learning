"""
ConvoFuse Command-Line Interface & Sequential Corpus Runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from convofuse.audio_diarization_adapter import BENCHMARK_RECORDING_MAP, REPO_ROOT
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


def run_corpus_benchmark(
    recording_ids: List[str],
    condition_preference: str,
    use_locality_gating: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    pipeline = ConvoFusePipeline(use_locality_gating=use_locality_gating)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_recording_results: List[Dict[str, Any]] = []
    t_corpus_start = time.perf_counter()

    for rec_id in recording_ids:
        canonical_id = BENCHMARK_RECORDING_MAP.get(rec_id, rec_id)
        rec_out_dir = output_dir / canonical_id
        rec_out_dir.mkdir(parents=True, exist_ok=True)
        mode_name = "LG-TPST" if use_locality_gating else "Unconstrained-TPST"
        print(
            f"[ConvoFuse] Starting {canonical_id} "
            f"(condition_preference={condition_preference}, mode={mode_name})..."
        )
        t0 = time.perf_counter()
        bundle = pipeline.run_frozen_replay(
            recording_id=canonical_id,
            condition_preference=condition_preference,
            enable_refinement=True,
            use_locality_gating=use_locality_gating,
            output_dir=rec_out_dir,
        )
        elapsed = time.perf_counter() - t0
        eval_rep = bundle.evaluation_report or {}
        base_m = eval_rep.get("baseline_metrics", {})
        ref_m = eval_rep.get("refined_metrics", {})
        deltas = eval_rep.get("metric_deltas", {})
        outcomes = eval_rep.get("refinement_outcomes", {})
        integrity = eval_rep.get("integrity_verification", {})
        telem = bundle.processing_manifest.get("refinement_telemetry", {})

        base_ts_wder = float(base_m.get("wder_timestamp", {}).get("wder_assigned", 0.0)) * 100.0
        ref_ts_wder = float(ref_m.get("wder_timestamp", {}).get("wder_assigned", 0.0)) * 100.0
        base_lex_wder = float(base_m.get("wder_lexical", {}).get("wder_lexical", 0.0)) * 100.0
        ref_lex_wder = float(ref_m.get("wder_lexical", {}).get("wder_lexical", 0.0)) * 100.0
        base_cpwer = float(base_m.get("cpwer", 0.0)) * 100.0
        ref_cpwer = float(ref_m.get("cpwer", 0.0)) * 100.0

        row = {
            "recording_id": canonical_id,
            "duration_sec": bundle.duration_sec,
            "diarization_condition": bundle.diarization_condition,
            "refinement_model": bundle.refinement_model,
            "word_count": len(bundle.words),
            "baseline_turn_count": len(bundle.baseline_turns),
            "refined_turn_count": len(bundle.turns),
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
            "word_sequence_intact": integrity.get("word_sequence_intact", True),
            "timestamp_sequence_intact": integrity.get("timestamp_sequence_intact", True),
            "unk_invariance_verified": integrity.get("unk_invariance_verified", True),
            "frozen_inputs_sha256_verified": bundle.processing_manifest.get("frozen_inputs_sha256_verified", True),
            "llm_runtime_sec": round(float(telem.get("total_runtime_sec", 0.0)), 4),
            "prompt_tokens": int(telem.get("total_prompt_tokens", 0)),
            "completion_tokens": int(telem.get("total_completion_tokens", 0)),
            "total_pipeline_sec": round(elapsed, 4),
            "output_dir": str(rec_out_dir),
        }
        per_recording_results.append(row)
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

    total_corpus_sec = time.perf_counter() - t_corpus_start
    n = len(per_recording_results)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording_count": n,
        "condition_preference": condition_preference,
        "use_locality_gating": use_locality_gating,
        "total_corpus_sec": round(total_corpus_sec, 4),
        "macro_averages": {
            "mean_baseline_wder_ts_pct": round(sum(r["baseline_wder_ts_pct"] for r in per_recording_results) / max(n, 1), 2),
            "mean_refined_wder_ts_pct": round(sum(r["refined_wder_ts_pct"] for r in per_recording_results) / max(n, 1), 2),
            "mean_delta_wder_ts_pp": round(sum(r["delta_wder_ts_pp"] for r in per_recording_results) / max(n, 1), 2),
            "mean_baseline_wder_lex_pct": round(sum(r["baseline_wder_lex_pct"] for r in per_recording_results) / max(n, 1), 2),
            "mean_refined_wder_lex_pct": round(sum(r["refined_wder_lex_pct"] for r in per_recording_results) / max(n, 1), 2),
            "mean_delta_wder_lex_pp": round(sum(r["delta_wder_lex_pp"] for r in per_recording_results) / max(n, 1), 2),
            "mean_baseline_cpwer_pct": round(sum(r["baseline_cpwer_pct"] for r in per_recording_results) / max(n, 1), 2),
            "mean_refined_cpwer_pct": round(sum(r["refined_cpwer_pct"] for r in per_recording_results) / max(n, 1), 2),
            "mean_delta_cpwer_pp": round(sum(r["delta_cpwer_pp"] for r in per_recording_results) / max(n, 1), 2),
            "total_accepted_corrections": sum(r["accepted_corrections"] for r in per_recording_results),
            "total_rejected_proposals": sum(r["rejected_proposals"] for r in per_recording_results),
            "total_abstentions": sum(r["abstentions"] for r in per_recording_results),
            "total_beneficial_corrections": sum(r["beneficial_corrections"] for r in per_recording_results),
            "total_harmful_corrections": sum(r["harmful_corrections"] for r in per_recording_results),
            "all_word_sequences_preserved": all(r["word_sequence_intact"] for r in per_recording_results),
            "all_timestamps_preserved": all(r["timestamp_sequence_intact"] for r in per_recording_results),
            "all_unk_invariance_verified": all(r["unk_invariance_verified"] for r in per_recording_results),
            "all_frozen_inputs_sha256_verified": all(r["frozen_inputs_sha256_verified"] for r in per_recording_results),
        },
        "recordings": per_recording_results,
    }

    summary_json_path = output_dir / "CORPUS_SUMMARY.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvoFuse CLI Runner")
    parser.add_argument("--recording", type=str, default="R01_sara_sharif", help="Recording ID (e.g., R01_sara_sharif)")
    parser.add_argument("--all-recordings", action="store_true", help="Run all 8 benchmark recordings R01..R08 sequentially")
    parser.add_argument("--condition", type=str, default="auto", help="Diarization condition preference (auto, E20, Condition_J, Condition_K)")
    parser.add_argument("--unconstrained", action="store_true", help="Disable LG-TPST locality gating (run unconstrained TPST)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for exported artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    if args.all_recordings:
        run_corpus_benchmark(
            recording_ids=list(BENCHMARK_RECORDING_IDS),
            condition_preference=args.condition,
            use_locality_gating=not args.unconstrained,
            output_dir=out_dir,
        )
    else:
        run_corpus_benchmark(
            recording_ids=[args.recording],
            condition_preference=args.condition,
            use_locality_gating=not args.unconstrained,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
