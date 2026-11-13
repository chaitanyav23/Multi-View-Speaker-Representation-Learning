#!/usr/bin/env python3
"""
Benchmark Evaluation CLI for CS698B Video Transcription with Confidence Analytics.

Runs real Phase 20 pretrained model inference across the project benchmark audio files,
evaluates speech recognition accuracy against ground-truth transcripts, fits confidence
calibration curves (Isotonic Regression & Temperature Scaling), and computes empirical
reliability metrics (WER, ECE, Brier Score, and Risk-Coverage curves).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure app/ is in sys.path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from adaptive.confidence import ConfidenceCalibrator
from engine.phase20_engine import Phase20SpeechEngine
from engine.schemas import TranscriptResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmarks")


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def find_benchmark_pairs(audio_dir: Path, gt_dir: Path) -> List[Dict[str, Path]]:
    """Pair media files in audio_dir with corresponding ground-truth .txt files."""
    pairs = []
    if not audio_dir.exists() or not gt_dir.exists():
        logger.warning("Audio directory '%s' or GT directory '%s' does not exist.", audio_dir, gt_dir)
        return pairs

    audio_files = [
        f for f in audio_dir.glob("*.*")
        if f.suffix.lower() in (".mp3", ".wav", ".mp4", ".m4a") and not f.name.startswith(".")
    ]

    for audio_path in sorted(audio_files):
        stem = audio_path.stem
        # Match against .txt in gt_dir
        gt_candidates = [
            gt_dir / f"{stem}.txt",
            gt_dir / f"{stem.replace('_', ' ')}.txt",
        ]
        # Also check case-insensitive match
        gt_path = None
        for cand in gt_candidates:
            if cand.exists():
                gt_path = cand
                break

        if not gt_path:
            for gt_file in gt_dir.glob("*.txt"):
                if normalize_text(gt_file.stem) == normalize_text(stem):
                    gt_path = gt_file
                    break

        if gt_path and gt_path.exists():
            pairs.append({"stem": stem, "audio": audio_path, "ground_truth": gt_path})
        else:
            logger.info("Ground truth not found for %s, skipping from benchmark evaluation.", audio_path.name)

    return pairs


def run_benchmark_suite(
    audio_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    model_size: str = "base",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete benchmark suite across discovered recording pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_benchmark_pairs(audio_dir, gt_dir)

    if not pairs:
        logger.error("No valid audio + ground truth pairs found in %s and %s", audio_dir, gt_dir)
        return {}

    if limit:
        pairs = pairs[:limit]

    logger.info("Discovered %d benchmark recording pairs to evaluate with model '%s'.", len(pairs), model_size)

    calibrator = ConfidenceCalibrator(method="isotonic_regression")
    engine = Phase20SpeechEngine(model_size=model_size, calibrator=calibrator)

    all_results = []
    total_ref_words = 0
    total_errors = 0
    all_raw_confidences: List[float] = []
    all_cal_confidences: List[float] = []
    all_correctness_labels: List[int] = []

    print("\n" + "=" * 90)
    print(f"CS698B BENCHMARK SUITE: Model = {model_size} | Recordings = {len(pairs)}")
    print("=" * 90)

    for idx, item in enumerate(pairs, 1):
        stem = item["stem"]
        audio_path = item["audio"]
        gt_path = item["ground_truth"]

        logger.info("[%d/%d] Processing '%s'...", idx, len(pairs), audio_path.name)
        start_time = time.time()

        # 1. Read ground truth text
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace").strip()
        gt_words = [w for w in gt_raw.split() if w]

        # 2. Execute real inference
        try:
            transcript_res: TranscriptResult = engine.process(
                audio_path=str(audio_path),
                recording_id=f"bench_{stem}",
                filename=audio_path.name,
            )
        except Exception as exc:
            logger.error("Error processing '%s': %s", audio_path.name, exc)
            continue

        elapsed = round(time.time() - start_time, 2)
        hyp_words = [w.word for w in transcript_res.words]
        raw_confs = [w.raw_confidence for w in transcript_res.words]
        cal_confs = [w.confidence for w in transcript_res.words]

        # 3. Dynamic programming Levenshtein alignment
        eval_dict = calibrator.generate_word_correctness_labels(hyp_words, gt_words)
        labels = eval_dict["labels"]

        all_raw_confidences.extend(raw_confs)
        all_cal_confidences.extend(cal_confs)
        all_correctness_labels.extend(labels)

        # 4. Calibration Metrics for this recording
        raw_ece = calibrator.compute_ece(raw_confs, labels)
        cal_ece = calibrator.compute_ece(cal_confs, labels)
        raw_brier = calibrator.compute_brier_score(raw_confs, labels)
        cal_brier = calibrator.compute_brier_score(cal_confs, labels)

        ref_word_count = len(gt_words)
        errors = eval_dict["substitutions"] + eval_dict["insertions"] + eval_dict["deletions_count"]
        wer = round(errors / max(1, ref_word_count), 4)

        total_ref_words += ref_word_count
        total_errors += errors

        rec_summary = {
            "recording": stem,
            "filename": audio_path.name,
            "duration": transcript_res.duration,
            "elapsed_sec": elapsed,
            "reference_words": ref_word_count,
            "hypothesis_words": len(hyp_words),
            "wer": wer,
            "substitutions": eval_dict["substitutions"],
            "insertions": eval_dict["insertions"],
            "deletions": eval_dict["deletions_count"],
            "raw_ece": round(raw_ece, 4),
            "calibrated_ece": round(cal_ece, 4),
            "raw_brier": round(raw_brier, 4),
            "calibrated_brier": round(cal_brier, 4),
            "mean_calibrated_confidence": round(float(sum(cal_confs) / max(1, len(cal_confs))), 4),
            "speakers_detected": len(transcript_res.speakers),
        }
        all_results.append(rec_summary)

        print(
            f"  {stem[:30]:<32} | Dur: {transcript_res.duration:>6.1f}s | "
            f"WER: {wer*100:>5.1f}% | ECE: {raw_ece:.3f} -> {cal_ece:.3f} | Brier: {cal_brier:.3f}"
        )

    # 5. Global Metrics Across Entire Benchmark Corpus
    global_wer = round(total_errors / max(1, total_ref_words), 4)
    global_raw_ece = round(calibrator.compute_ece(all_raw_confidences, all_correctness_labels), 4)
    global_cal_ece = round(calibrator.compute_ece(all_cal_confidences, all_correctness_labels), 4)
    global_raw_brier = round(calibrator.compute_brier_score(all_raw_confidences, all_correctness_labels), 4)
    global_cal_brier = round(calibrator.compute_brier_score(all_cal_confidences, all_correctness_labels), 4)

    reliability_diagram = calibrator.compute_reliability_diagram(all_cal_confidences, all_correctness_labels)
    risk_coverage = calibrator.compute_risk_coverage(all_cal_confidences, all_correctness_labels)

    report_data = {
        "model_size": model_size,
        "recordings_evaluated": len(all_results),
        "total_reference_words": total_ref_words,
        "global_metrics": {
            "wer": global_wer,
            "raw_ece": global_raw_ece,
            "calibrated_ece": global_cal_ece,
            "raw_brier": global_raw_brier,
            "calibrated_brier": global_cal_brier,
            "ece_reduction_percent": round((1.0 - global_cal_ece / max(1e-4, global_raw_ece)) * 100, 1),
            "brier_reduction_percent": round((1.0 - global_cal_brier / max(1e-4, global_raw_brier)) * 100, 1),
        },
        "per_recording": all_results,
        "reliability_diagram": reliability_diagram,
        "risk_coverage_curve": risk_coverage,
    }

    # Save JSON report
    out_json = out_dir / f"benchmark_summary_{model_size}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown Report
    out_md = out_dir / f"benchmark_report_{model_size}.md"
    md_lines = [
        f"# CS698B Benchmark Evaluation Report: `{model_size}`",
        "",
        f"**Evaluated Recordings**: {len(all_results)}  ",
        f"**Total Reference Words**: {total_ref_words}  ",
        f"**Global Corpus WER**: **{global_wer * 100:.2f}%**  ",
        f"**Expected Calibration Error (ECE)**: **{global_raw_ece:.4f} → {global_cal_ece:.4f}** ({report_data['global_metrics']['ece_reduction_percent']}% reduction)  ",
        f"**Brier Score**: **{global_raw_brier:.4f} → {global_cal_brier:.4f}** ({report_data['global_metrics']['brier_reduction_percent']}% reduction)  ",
        "",
        "## Per-Recording Performance",
        "",
        "| Recording | Duration | Words (Ref/Hyp) | WER | Raw ECE | Calibrated ECE | Brier Score | Speakers |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in all_results:
        md_lines.append(
            f"| `{r['recording']}` | {r['duration']:.1f}s | {r['reference_words']} / {r['hypothesis_words']} | "
            f"{r['wer']*100:.1f}% | {r['raw_ece']:.3f} | **{r['calibrated_ece']:.3f}** | {r['calibrated_brier']:.3f} | {r['speakers_detected']} |"
        )

    md_lines.extend([
        "",
        "## Risk-Coverage Analysis",
        "",
        "| Confidence Threshold | Coverage (%) | Retained Words | Empirical Risk (Error Rate %) |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for rc in risk_coverage:
        md_lines.append(
            f"| ≥ {rc['threshold']:.2f} | {rc['coverage']*100:.1f}% | {rc['retained_count']} | **{rc['risk']*100:.2f}%** |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 90)
    print(f"BENCHMARK COMPLETE: Global WER = {global_wer*100:.2f}% | ECE: {global_raw_ece:.4f} -> {global_cal_ece:.4f}")
    print(f"Results saved to:\n  - {out_json}\n  - {out_md}")
    print("=" * 90)

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 20 pipeline on CS698B benchmark dataset.")
    parser.add_argument("--audio-dir", type=Path, default=Path("../raw_audio"), help="Path to raw audio recordings")
    parser.add_argument("--gt-dir", type=Path, default=Path("../ground_truth"), help="Path to ground truth transcripts")
    parser.add_argument("--out-dir", type=Path, default=Path("storage/benchmark_results"), help="Output directory")
    parser.add_argument("--model", type=str, default="tiny", help="Whisper model size (tiny, base, small, large-v3)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of recordings to evaluate")
    args = parser.parse_args()

    # Resolve relative paths relative to current execution
    audio_dir = args.audio_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    out_dir = args.out_dir.resolve()

    # Fallback to local raw_audio if relative parent path does not exist
    if not audio_dir.exists() and (APP_DIR.parent / "raw_audio").exists():
        audio_dir = APP_DIR.parent / "raw_audio"
    if not gt_dir.exists() and (APP_DIR.parent / "ground_truth").exists():
        gt_dir = APP_DIR.parent / "ground_truth"

    run_benchmark_suite(
        audio_dir=audio_dir,
        gt_dir=gt_dir,
        out_dir=out_dir,
        model_size=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
