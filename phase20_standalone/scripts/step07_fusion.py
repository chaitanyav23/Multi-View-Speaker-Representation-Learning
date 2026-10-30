import argparse

from common import BASE_DIR, add_file_arg, attributed_words, group_words, read_json, recording_paths, rttm_to_segments, setup_logging, stage_metadata, write_json, write_transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 7: maximum-overlap speaker attribution fusion.")
    add_file_arg(parser)
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step07_fusion")
    words = attributed_words(read_json(paths.aligned_words), rttm_to_segments(paths.diarization_rttm))
    segments = group_words(words)
    payload = {"file": f"{paths.name}.wav", "method": "maximum_overlap", "words": words, "segments": segments}
    write_json(paths.attributed_words, payload)
    write_transcript(paths.fusion_transcript, segments)
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step07_fusion.json", stage_metadata("fusion", [paths.aligned_words, paths.diarization_rttm], [paths.attributed_words, paths.fusion_transcript], {"words": len(words), "segments": len(segments)}))
    logger.info("Saved fusion outputs to %s and %s", paths.attributed_words, paths.fusion_transcript)


if __name__ == "__main__":
    main()
