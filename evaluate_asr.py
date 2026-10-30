"""
scripts/evaluate_asr.py
=======================
Compute word error rate (WER) between transcripts/ and ground_truth/.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Tuple

from scripts.utils.logger import get_logger

PROJECT_ROOT = Path.cwd()

TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


_BRACKET_PATTERN = re.compile(r"\[[^\]]+\]")
_SPEAKER_PATTERN = re.compile(r"\bSpeaker\s+\d+\b", flags=re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9']+")


def _ensure_dirs() -> None:
    for d in (REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _normalize_text(text: str) -> str:
    cleaned = _BRACKET_PATTERN.sub(" ", text)
    cleaned = _SPEAKER_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace(":", " ")
    cleaned = cleaned.lower()
    cleaned = _NON_WORD_PATTERN.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _load_words(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    normalized = _normalize_text(raw)
    if not normalized:
        return []
    return normalized.split()


def _wer_stats(
    reference: Iterable[str],
    hypothesis: Iterable[str],
) -> Tuple[int, int, int, int, int]:
    ref_words = list(reference)
    hyp_words = list(hypothesis)

    ref_len = len(ref_words)
    hyp_len = len(hyp_words)

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
            if ref_words[i - 1] == hyp_words[j - 1]:
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


def _wer_rate(errors: int, ref_len: int, hyp_len: int) -> float:
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0
    return errors / ref_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER for transcripts vs. ground truth."
    )
    parser.add_argument(
        "--transcripts",
        default=str(TRANSCRIPTS_DIR),
        help="Directory containing transcript .txt files.",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(GROUND_TRUTH_DIR),
        help="Directory containing ground truth .txt files.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPORTS_DIR / "wer_eval.csv"),
        help="CSV output path for WER results.",
    )
    args = parser.parse_args()

    _ensure_dirs()

    log = get_logger(
        name="bwc_pipeline.wer",
        level="INFO",
        log_dir=LOGS_DIR,
        log_filename="wer_eval.log",
    )

    transcripts_dir = Path(args.transcripts)
    ground_truth_dir = Path(args.ground_truth)
    output_csv = Path(args.output_csv)

    transcript_files = sorted(transcripts_dir.glob("*.txt"))
    if not transcript_files:
        log.warning("No transcript files found in %s", transcripts_dir)
        return

    results = []
    total_errors = 0
    total_ref_words = 0
    total_hyp_words = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0

    for transcript_path in transcript_files:
        gt_path = ground_truth_dir / transcript_path.name
        if not gt_path.exists():
            log.warning("Missing ground truth for %s", transcript_path.name)
            continue

        hyp_words = _load_words(transcript_path)
        ref_words = _load_words(gt_path)

        errors, subs, ins, dels, ref_len = _wer_stats(ref_words, hyp_words)
        hyp_len = len(hyp_words)
        wer = _wer_rate(errors, ref_len, hyp_len)

        results.append(
            {
                "file": transcript_path.name,
                "ref_words": ref_len,
                "hyp_words": hyp_len,
                "substitutions": subs,
                "insertions": ins,
                "deletions": dels,
                "wer": round(wer, 4),
            }
        )

        log.info(
            "WER %s: %.4f (S=%d I=%d D=%d, N=%d)",
            transcript_path.name,
            wer,
            subs,
            ins,
            dels,
            ref_len,
        )

        total_errors += errors
        total_ref_words += ref_len
        total_hyp_words += hyp_len
        total_subs += subs
        total_ins += ins
        total_dels += dels

    if not results:
        log.warning("No matching transcript/ground truth pairs found.")
        return

    overall_wer = _wer_rate(total_errors, total_ref_words, total_hyp_words)
    log.info(
        "Overall WER: %.4f (S=%d I=%d D=%d, N=%d)",
        overall_wer,
        total_subs,
        total_ins,
        total_dels,
        total_ref_words,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "ref_words",
                "hyp_words",
                "substitutions",
                "insertions",
                "deletions",
                "wer",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
        writer.writerow(
            {
                "file": "OVERALL",
                "ref_words": total_ref_words,
                "hyp_words": total_hyp_words,
                "substitutions": total_subs,
                "insertions": total_ins,
                "deletions": total_dels,
                "wer": round(overall_wer, 4),
            }
        )

    log.info("Saved WER CSV → %s", output_csv)


if __name__ == "__main__":
    main()
