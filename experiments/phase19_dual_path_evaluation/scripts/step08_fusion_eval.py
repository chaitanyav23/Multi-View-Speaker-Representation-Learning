import argparse
from pathlib import Path

from phase19_common import BASE_DIR, audio_duration, fusion_metrics, setup_logging, write_json, write_report


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 8: evaluate fusion/conversational metrics.")
    parser.add_argument("--attributed", type=Path, default=BASE_DIR / "fusion" / "attributed_words.json")
    parser.add_argument("--reference", type=Path, default=Path("ground_truth_json/bwc.json"))
    parser.add_argument("--audio", type=Path, default=BASE_DIR / "audio" / "bwc_repaired.wav")
    parser.add_argument("--metrics-output", type=Path, default=BASE_DIR / "metrics" / "fusion_metrics.json")
    parser.add_argument("--report-output", type=Path, default=BASE_DIR / "reports" / "fusion_report.md")
    args = parser.parse_args()
    logger = setup_logging("step08_fusion_eval")
    metrics = fusion_metrics(args.reference, args.attributed, audio_duration(args.audio))
    write_json(args.metrics_output, metrics)
    write_report(args.report_output, "Fusion Report", [("Inputs", {"reference": args.reference, "attributed": args.attributed}), ("Metrics", metrics)])
    logger.info("Saved fusion metrics to %s", args.metrics_output)


if __name__ == "__main__":
    main()
