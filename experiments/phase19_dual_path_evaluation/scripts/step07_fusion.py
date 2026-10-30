import argparse
from pathlib import Path

from phase19_common import BASE_DIR, attributed_words, group_words, read_json, rttm_to_segments, setup_logging, stage_metadata, write_json, write_transcript


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 7: maximum-overlap speaker attribution fusion.")
    parser.add_argument("--words", type=Path, default=BASE_DIR / "transcription" / "aligned_words.json")
    parser.add_argument("--diarization", type=Path, default=BASE_DIR / "diarization" / "pathA_msdd.rttm")
    parser.add_argument("--output-json", type=Path, default=BASE_DIR / "fusion" / "attributed_words.json")
    parser.add_argument("--output-txt", type=Path, default=BASE_DIR / "fusion" / "transcript.txt")
    args = parser.parse_args()
    logger = setup_logging("step07_fusion")
    words = attributed_words(read_json(args.words), rttm_to_segments(args.diarization))
    segments = group_words(words)
    payload = {"file": "bwc.wav", "method": "maximum_overlap", "words": words, "segments": segments}
    write_json(args.output_json, payload)
    write_transcript(args.output_txt, segments)
    write_json(BASE_DIR / "outputs" / "step07_fusion.json", stage_metadata("fusion", [args.words, args.diarization], [args.output_json, args.output_txt], {"words": len(words), "segments": len(segments)}))
    logger.info("Saved fusion outputs to %s and %s", args.output_json, args.output_txt)


if __name__ == "__main__":
    main()
