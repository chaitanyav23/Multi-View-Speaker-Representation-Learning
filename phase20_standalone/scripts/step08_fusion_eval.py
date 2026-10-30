import argparse

from common import BASE_DIR, add_file_arg, audio_duration, fusion_metrics, recording_paths, setup_logging, stage_metadata, write_json, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 8: evaluate fusion/conversational metrics.")
    add_file_arg(parser)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step08_fusion_eval")
    metrics = fusion_metrics(paths.ground_truth_json, paths.attributed_words, audio_duration(paths.repaired_audio))
    write_json(paths.fusion_metrics, metrics)
    write_report(paths.fusion_report, "Fusion Report", [("Inputs", {"reference": paths.ground_truth_json, "attributed": paths.attributed_words}), ("Metrics", metrics)])
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step08_fusion_eval.json", stage_metadata("fusion_eval", [paths.ground_truth_json, paths.attributed_words, paths.repaired_audio], [paths.fusion_metrics, paths.fusion_report], metrics))
    logger.info("Saved fusion metrics to %s", paths.fusion_metrics)


if __name__ == "__main__":
    main()
