import argparse

from common import BASE_DIR, add_file_arg, asr_metrics, read_json, recording_paths, setup_logging, stage_metadata, text_from_gt, text_from_words, write_json, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 6B: evaluate ASR.")
    add_file_arg(parser)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step06b_asr_eval")
    metrics = asr_metrics(text_from_gt(paths.ground_truth_json), text_from_words(read_json(paths.aligned_words)))
    write_json(paths.asr_metrics, metrics)
    write_report(paths.asr_report, "ASR Report", [("Inputs", {"reference": paths.ground_truth_json, "words": paths.aligned_words}), ("Metrics", metrics)])
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step06b_asr_eval.json", stage_metadata("asr_eval", [paths.ground_truth_json, paths.aligned_words], [paths.asr_metrics, paths.asr_report], metrics))
    logger.info("Saved ASR metrics to %s", paths.asr_metrics)


if __name__ == "__main__":
    main()
