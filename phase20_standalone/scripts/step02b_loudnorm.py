import argparse

from common import BASE_DIR, add_file_arg, recording_paths, run_command, setup_logging, stage_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 2B: loudness normalize transcription path to -20 LUFS.")
    add_file_arg(parser)
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step02b_loudnorm")
    if paths.normalized_audio.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", paths.normalized_audio)
    else:
        filt = f"loudnorm=I={args.target_lufs}:TP=-1.5:LRA=11"
        run_command(["ffmpeg", "-y", "-i", paths.repaired_audio, "-af", filt, "-ar", "16000", "-ac", "1", paths.normalized_audio], logger, "loudness normalization")
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step02b_loudnorm.json", stage_metadata("loudnorm", [paths.repaired_audio], [paths.normalized_audio], {"target_lufs": args.target_lufs}))


if __name__ == "__main__":
    main()
