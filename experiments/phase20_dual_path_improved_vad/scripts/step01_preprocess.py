import argparse
from pathlib import Path

from phase19_common import BASE_DIR, run_command, setup_logging, stage_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 1: repair/resample audio to mono 16 kHz.")
    parser.add_argument("--input", type=Path, default=Path("raw_audio/bwc.wav"))
    parser.add_argument("--output", type=Path, default=BASE_DIR / "audio" / "bwc_repaired.wav")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("step01_preprocess")
    if not args.input.exists():
        raise FileNotFoundError(f"Input audio does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", args.output)
    else:
        run_command(["ffmpeg", "-y", "-i", args.input, "-ar", "16000", "-ac", "1", args.output], logger, "preprocess")
    write_json(BASE_DIR / "outputs" / "step01_preprocess.json", stage_metadata("preprocess", [args.input], [args.output]))


if __name__ == "__main__":
    main()
