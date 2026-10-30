"""
scripts/evaluate_alignment.py
=============================
Evaluate alignment quality using ground-truth timestamps and predicted
WhisperX alignment outputs. Also computes CER for the aligned text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Tuple

from scripts.utils.logger import get_logger

PROJECT_ROOT = Path.cwd()

GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
ALIGNMENTS_DIR = PROJECT_ROOT / "alignments"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

GT_PATTERN = re.compile(
    r"\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]"
)
_BRACKET_PATTERN = re.compile(r"\[[^\]]+\]")
_SPEAKER_PATTERN = re.compile(r"\bSpeaker\s+\d+\b", flags=re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9']+")


def _ensure_dirs() -> None:
    for d in (REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _hms_to_seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def _parse_ground_truth(gt_file: Path) -> list[dict]:
    segments = []
    with gt_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = GT_PATTERN.search(line)
            if not match:
                continue
            h1, m1, s1, h2, m2, s2 = match.groups()
            start = _hms_to_seconds(h1, m1, s1)
            end = _hms_to_seconds(h2, m2, s2)
            if end <= start:
                continue
            text = ""
            if "]" in line:
                text = line.split("]:", 1)[-1].strip()
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                }
            )
    return segments


def _load_alignment(alignment_file: Path) -> list[dict]:
    with alignment_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    segments = []
    for seg in payload.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end <= start:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "text": seg.get("text", ""),
            }
        )
    return segments


def _intersection(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    return max(0.0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))


def _union(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    return max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])


def _iou(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    inter = _intersection(seg1, seg2)
    if inter <= 0:
        return 0.0
    return inter / _union(seg1, seg2)


def _compute_segment_metrics(
    gt_segments: list[dict],
    pred_segments: list[dict],
    overlap_threshold: float = 0.1,
) -> dict:
    matched_gt = 0
    matched_pred = 0
    ious: list[float] = []
    boundary_errors: list[float] = []

    coverage_numerator = 0.0
    coverage_denominator = 0.0

    gt_bounds = [(seg["start"], seg["end"]) for seg in gt_segments]
    pred_bounds = [(seg["start"], seg["end"]) for seg in pred_segments]

    for gt in gt_bounds:
        gt_len = gt[1] - gt[0]
        coverage_denominator += gt_len

        best_iou = 0.0
        best_pred = None
        overlap_duration = 0.0

        for pred in pred_bounds:
            overlap_duration += _intersection(gt, pred)
            current_iou = _iou(gt, pred)
            if current_iou > best_iou:
                best_iou = current_iou
                best_pred = pred

        coverage_numerator += min(overlap_duration, gt_len)

        if best_iou >= overlap_threshold and best_pred is not None:
            matched_gt += 1
            ious.append(best_iou)
            start_err = abs(gt[0] - best_pred[0])
            end_err = abs(gt[1] - best_pred[1])
            boundary_errors.append((start_err + end_err) / 2)

    for pred in pred_bounds:
        best_iou = 0.0
        for gt in gt_bounds:
            best_iou = max(best_iou, _iou(gt, pred))
        if best_iou >= overlap_threshold:
            matched_pred += 1

    segment_recall = matched_gt / len(gt_bounds) if gt_bounds else 0.0
    segment_precision = matched_pred / len(pred_bounds) if pred_bounds else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    boundary_error = (
        sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0.0
    )
    speech_coverage = (
        coverage_numerator / coverage_denominator
        if coverage_denominator > 0
        else 0.0
    )

    return {
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "coverage_numerator": coverage_numerator,
        "coverage_denominator": coverage_denominator,
        "iou_sum": sum(ious),
        "iou_count": len(ious),
        "boundary_error_sum": sum(boundary_errors),
        "boundary_error_count": len(boundary_errors),
        "speech_coverage": round(float(speech_coverage), 4),
        "segment_recall": round(float(segment_recall), 4),
        "segment_precision": round(float(segment_precision), 4),
        "mean_iou": round(float(mean_iou), 4),
        "boundary_error_sec": round(float(boundary_error), 4),
    }


def _normalize_text(text: str) -> str:
    cleaned = _BRACKET_PATTERN.sub(" ", text)
    cleaned = _SPEAKER_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace(":", " ")
    cleaned = cleaned.lower()
    cleaned = _NON_WORD_PATTERN.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _edit_stats(
    reference: Iterable[str],
    hypothesis: Iterable[str],
) -> Tuple[int, int, int, int, int]:
    ref_items = list(reference)
    hyp_items = list(hypothesis)

    ref_len = len(ref_items)
    hyp_len = len(hyp_items)

    dp: list[list[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(hyp_len + 1)]
        for _ in range(ref_len + 1)
    ]

    for i in range(1, ref_len + 1):
        dp[i][0] = (i, 0, 0, i)

    for j in range(1, hyp_len + 1):
        dp[0][j] = (j, 0, j, 0)

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_items[i - 1] == hyp_items[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            sub_cost, sub_s, sub_i, sub_d = dp[i - 1][j - 1]
            ins_cost, ins_s, ins_i, ins_d = dp[i][j - 1]
            del_cost, del_s, del_i, del_d = dp[i - 1][j]

            candidates = [
                (sub_cost + 1, sub_s + 1, sub_i, sub_d),
                (ins_cost + 1, ins_s, ins_i + 1, ins_d),
                (del_cost + 1, del_s, del_i, del_d + 1),
            ]
            dp[i][j] = min(candidates, key=lambda t: (t[0], t[1], t[2], t[3]))

    total_errors, subs, ins, dels = dp[ref_len][hyp_len]
    return total_errors, subs, ins, dels, ref_len


def _cer_rate(errors: int, ref_len: int, hyp_len: int) -> float:
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0
    return errors / ref_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate alignment quality and compute CER."
    )
    parser.add_argument(
        "--ground-truth",
        default=str(GROUND_TRUTH_DIR),
        help="Directory containing ground truth .txt files.",
    )
    parser.add_argument(
        "--alignments",
        default=str(ALIGNMENTS_DIR),
        help="Directory containing WhisperX alignment .json files.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPORTS_DIR / "alignment_eval.csv"),
        help="CSV output path for alignment metrics.",
    )
    args = parser.parse_args()

    _ensure_dirs()

    log = get_logger(
        name="bwc_pipeline.alignment_eval",
        level="INFO",
        log_dir=LOGS_DIR,
        log_filename="alignment_eval.log",
    )

    gt_dir = Path(args.ground_truth)
    align_dir = Path(args.alignments)
    output_csv = Path(args.output_csv)

    alignment_files = sorted(align_dir.glob("*_aligned.json"))
    if not alignment_files:
        log.warning("No alignment files found in %s", align_dir)
        return

    rows = []

    total_gt_segments = 0
    total_pred_segments = 0
    total_matched_gt = 0
    total_matched_pred = 0
    total_coverage_num = 0.0
    total_coverage_den = 0.0
    total_iou = 0.0
    total_iou_count = 0
    total_boundary_error = 0.0
    total_boundary_count = 0

    total_cer_errors = 0
    total_cer_ref = 0
    total_cer_hyp = 0
    total_cer_subs = 0
    total_cer_ins = 0
    total_cer_dels = 0

    for alignment_path in alignment_files:
        stem = alignment_path.name.replace("_aligned.json", "")
        gt_path = gt_dir / f"{stem}.txt"

        if not gt_path.exists():
            log.warning("Missing ground truth for %s", alignment_path.name)
            continue

        gt_segments = _parse_ground_truth(gt_path)
        pred_segments = _load_alignment(alignment_path)

        metrics = _compute_segment_metrics(gt_segments, pred_segments)

        gt_text = " ".join(seg["text"] for seg in gt_segments)
        pred_text = " ".join(seg["text"] for seg in pred_segments)
        normalized_gt = _normalize_text(gt_text).replace(" ", "")
        normalized_pred = _normalize_text(pred_text).replace(" ", "")

        cer_errors, cer_subs, cer_ins, cer_dels, cer_ref = _edit_stats(
            normalized_gt, normalized_pred
        )
        cer_hyp = len(normalized_pred)
        cer = _cer_rate(cer_errors, cer_ref, cer_hyp)

        rows.append(
            {
                "file": alignment_path.name,
                "gt_segments": len(gt_segments),
                "pred_segments": len(pred_segments),
                "matched_gt": metrics["matched_gt"],
                "matched_pred": metrics["matched_pred"],
                "speech_coverage": metrics["speech_coverage"],
                "segment_recall": metrics["segment_recall"],
                "segment_precision": metrics["segment_precision"],
                "mean_iou": metrics["mean_iou"],
                "boundary_error_sec": metrics["boundary_error_sec"],
                "cer": round(cer, 4),
                "cer_ref_chars": cer_ref,
                "cer_hyp_chars": cer_hyp,
                "cer_substitutions": cer_subs,
                "cer_insertions": cer_ins,
                "cer_deletions": cer_dels,
            }
        )

        log.info(
            "Alignment %s: coverage=%.4f recall=%.4f precision=%.4f mean_iou=%.4f "
            "boundary_err=%.4f cer=%.4f",
            alignment_path.name,
            metrics["speech_coverage"],
            metrics["segment_recall"],
            metrics["segment_precision"],
            metrics["mean_iou"],
            metrics["boundary_error_sec"],
            cer,
        )

        total_gt_segments += len(gt_segments)
        total_pred_segments += len(pred_segments)
        total_matched_gt += metrics["matched_gt"]
        total_matched_pred += metrics["matched_pred"]

        total_coverage_num += metrics["coverage_numerator"]
        total_coverage_den += metrics["coverage_denominator"]

        total_iou += metrics["iou_sum"]
        total_iou_count += metrics["iou_count"]

        total_boundary_error += metrics["boundary_error_sum"]
        total_boundary_count += metrics["boundary_error_count"]

        total_cer_errors += cer_errors
        total_cer_ref += cer_ref
        total_cer_hyp += cer_hyp
        total_cer_subs += cer_subs
        total_cer_ins += cer_ins
        total_cer_dels += cer_dels

    if not rows:
        log.warning("No matching alignment/ground truth pairs found.")
        return

    overall_recall = (
        total_matched_gt / total_gt_segments if total_gt_segments else 0.0
    )
    overall_precision = (
        total_matched_pred / total_pred_segments if total_pred_segments else 0.0
    )
    overall_coverage = (
        total_coverage_num / total_coverage_den if total_coverage_den > 0 else 0.0
    )
    overall_mean_iou = total_iou / total_iou_count if total_iou_count else 0.0
    overall_boundary = (
        total_boundary_error / total_boundary_count if total_boundary_count else 0.0
    )
    overall_cer = _cer_rate(total_cer_errors, total_cer_ref, total_cer_hyp)

    rows.append(
        {
            "file": "OVERALL",
            "gt_segments": total_gt_segments,
            "pred_segments": total_pred_segments,
            "matched_gt": total_matched_gt,
            "matched_pred": total_matched_pred,
            "speech_coverage": round(float(overall_coverage), 4),
            "segment_recall": round(float(overall_recall), 4),
            "segment_precision": round(float(overall_precision), 4),
            "mean_iou": round(float(overall_mean_iou), 4),
            "boundary_error_sec": round(float(overall_boundary), 4),
            "cer": round(float(overall_cer), 4),
            "cer_ref_chars": total_cer_ref,
            "cer_hyp_chars": total_cer_hyp,
            "cer_substitutions": total_cer_subs,
            "cer_insertions": total_cer_ins,
            "cer_deletions": total_cer_dels,
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "gt_segments",
                "pred_segments",
                "matched_gt",
                "matched_pred",
                "speech_coverage",
                "segment_recall",
                "segment_precision",
                "mean_iou",
                "boundary_error_sec",
                "cer",
                "cer_ref_chars",
                "cer_hyp_chars",
                "cer_substitutions",
                "cer_insertions",
                "cer_deletions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info("Saved alignment metrics → %s", output_csv)


if __name__ == "__main__":
    main()
