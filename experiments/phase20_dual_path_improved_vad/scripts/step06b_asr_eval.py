import argparse
from pathlib import Path

from phase19_common import BASE_DIR, asr_metrics, read_json, setup_logging, text_from_gt, text_from_words, write_json, write_report


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 6B: evaluate ASR.")
    parser.add_argument("--words", type=Path, default=BASE_DIR / "transcription" / "aligned_words.json")
    parser.add_argument("--reference", type=Path, default=Path("ground_truth_json/bwc.json"))
    parser.add_argument("--metrics-output", type=Path, default=BASE_DIR / "metrics" / "asr_metrics.json")
    parser.add_argument("--report-output", type=Path, default=BASE_DIR / "reports" / "asr_report.md")
    args = parser.parse_args()
    logger = setup_logging("step06b_asr_eval")
    metrics = asr_metrics(text_from_gt(args.reference), text_from_words(read_json(args.words)))
    write_json(args.metrics_output, metrics)
    write_report(args.report_output, "ASR Report", [("Inputs", {"reference": args.reference, "words": args.words}), ("Metrics", metrics)])
    logger.info("Saved ASR metrics to %s", args.metrics_output)


if __name__ == "__main__":
    main()
