import argparse
from pathlib import Path

from phase19_common import BASE_DIR, run_command, setup_logging, stage_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 2B: loudness normalize transcription path to -20 LUFS.")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "audio" / "bwc_repaired.wav")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "audio" / "bwc_normalized.wav")
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("step02b_loudnorm")
    if args.output.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", args.output)
    else:
        filt = f"loudnorm=I={args.target_lufs}:TP=-1.5:LRA=11"
        run_command(["ffmpeg", "-y", "-i", args.input, "-af", filt, "-ar", "16000", "-ac", "1", args.output], logger, "loudness normalization")
    write_json(BASE_DIR / "outputs" / "step02b_loudnorm.json", stage_metadata("loudnorm", [args.input], [args.output], {"target_lufs": args.target_lufs}))


if __name__ == "__main__":
    main()
