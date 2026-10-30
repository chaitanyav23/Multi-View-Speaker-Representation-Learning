import argparse

from common import BASE_DIR, add_file_arg, diarization_metrics, recording_paths, setup_logging, stage_metadata, write_json, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 6A: evaluate diarization.")
    add_file_arg(parser)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step06a_diar_eval")
    metrics = diarization_metrics(paths.ground_truth_rttm, paths.diarization_rttm)
    write_json(paths.diar_metrics, metrics)
    write_report(paths.diar_report, "Diarization Report", [("Inputs", {"reference": paths.ground_truth_rttm, "hypothesis": paths.diarization_rttm}), ("Metrics", metrics)])
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step06a_diar_eval.json", stage_metadata("diarization_eval", [paths.ground_truth_rttm, paths.diarization_rttm], [paths.diar_metrics, paths.diar_report], metrics))
    logger.info("Saved diarization metrics to %s", paths.diar_metrics)


if __name__ == "__main__":
    main()
