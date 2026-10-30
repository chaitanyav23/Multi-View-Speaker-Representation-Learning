import argparse

from common import BASE_DIR, add_file_arg, audio_duration, load_segments_json, recording_paths, setup_logging, stage_metadata, vad_metrics, write_json, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 4: evaluate VAD against ground truth segments.")
    add_file_arg(parser)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    if args.label == "pathA":
        hypothesis = paths.path_a_vad
        metrics_output = paths.path_a_vad_metrics
        report_output = paths.path_a_vad_report
    elif args.label == "pathB":
        hypothesis = paths.path_b_vad
        metrics_output = paths.path_b_vad_metrics
        report_output = paths.path_b_vad_report
    else:
        raise ValueError("--label must be pathA or pathB")
    logger = setup_logging(f"{paths.name}_step04_{args.label}_vad_eval")
    duration = audio_duration(paths.repaired_audio)
    metrics = vad_metrics(load_segments_json(paths.ground_truth_json), load_segments_json(hypothesis), duration)
    write_json(metrics_output, metrics)
    write_report(report_output, f"{args.label} VAD Report", [("Inputs", {"reference": paths.ground_truth_json, "hypothesis": hypothesis, "duration_sec": duration}), ("Metrics", metrics)])
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step04_{args.label}_vad_eval.json", stage_metadata(f"{args.label}_vad_eval", [paths.ground_truth_json, hypothesis, paths.repaired_audio], [metrics_output, report_output], metrics))
    logger.info("Saved VAD metrics to %s", metrics_output)


if __name__ == "__main__":
    main()
