import argparse
from pathlib import Path

from phase19_common import BASE_DIR, audio_duration, load_segments_json, setup_logging, vad_metrics, write_json, write_report


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 4: evaluate VAD against ground truth segments.")
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("ground_truth_json/bwc.json"))
    parser.add_argument("--audio", type=Path, default=BASE_DIR / "audio" / "bwc_repaired.wav")
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    logger = setup_logging(f"step04_{args.label}_vad_eval")
    duration = audio_duration(args.audio)
    metrics = vad_metrics(load_segments_json(args.reference), load_segments_json(args.hypothesis), duration)
    write_json(args.metrics_output, metrics)
    write_report(args.report_output, f"{args.label} VAD Report", [("Inputs", {"reference": args.reference, "hypothesis": args.hypothesis, "duration_sec": duration}), ("Metrics", metrics)])
    logger.info("Saved VAD metrics to %s", args.metrics_output)


if __name__ == "__main__":
    main()
