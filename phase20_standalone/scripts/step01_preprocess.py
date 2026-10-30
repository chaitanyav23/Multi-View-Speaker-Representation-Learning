import argparse

from common import BASE_DIR, add_file_arg, recording_paths, run_command, setup_logging, stage_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 1: repair/resample audio to mono 16 kHz.")
    add_file_arg(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step01_preprocess")
    if not paths.raw_audio.exists():
        raise FileNotFoundError(f"Input audio does not exist: {paths.raw_audio}")
    if paths.repaired_audio.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", paths.repaired_audio)
    else:
        run_command(["ffmpeg", "-y", "-i", paths.raw_audio, "-ar", "16000", "-ac", "1", paths.repaired_audio], logger, "preprocess")
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step01_preprocess.json", stage_metadata("preprocess", [paths.raw_audio], [paths.repaired_audio]))


if __name__ == "__main__":
    main()
