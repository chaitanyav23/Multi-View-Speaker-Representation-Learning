import argparse
from pathlib import Path

from phase19_common import BASE_DIR, diarization_metrics, setup_logging, write_json, write_report


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 6A: evaluate diarization.")
    parser.add_argument("--hypothesis", type=Path, default=BASE_DIR / "diarization" / "pathA_msdd.rttm")
    parser.add_argument("--reference", type=Path, default=Path("ground_truth_rttm/bwc.rttm"))
    parser.add_argument("--metrics-output", type=Path, default=BASE_DIR / "metrics" / "diarization_metrics.json")
    parser.add_argument("--report-output", type=Path, default=BASE_DIR / "reports" / "diarization_report.md")
    args = parser.parse_args()
    logger = setup_logging("step06a_diar_eval")
    metrics = diarization_metrics(args.reference, args.hypothesis)
    write_json(args.metrics_output, metrics)
    write_report(args.report_output, "Diarization Report", [("Inputs", {"reference": args.reference, "hypothesis": args.hypothesis}), ("Metrics", metrics)])
    logger.info("Saved diarization metrics to %s", args.metrics_output)


if __name__ == "__main__":
    main()
