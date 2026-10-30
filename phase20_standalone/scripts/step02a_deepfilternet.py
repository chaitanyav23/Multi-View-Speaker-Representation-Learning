import argparse
import shutil

from common import BASE_DIR, add_file_arg, recording_paths, run_command, setup_logging, stage_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 2A: DeepFilterNet enhancement for diarization path.")
    add_file_arg(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step02a_deepfilternet")
    if paths.dfn_audio.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", paths.dfn_audio)
        evaluation = {"fallback": False}
    else:
        temp_dir = BASE_DIR / "audio" / "dfn_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("deepFilter"):
            run_command(["deepFilter", paths.repaired_audio, "-o", temp_dir], logger, "DeepFilterNet")
            candidates = sorted(temp_dir.glob("*.wav"))
            if not candidates:
                raise RuntimeError(f"DeepFilterNet did not produce a wav in {temp_dir}")
            candidates[0].replace(paths.dfn_audio)
            evaluation = {"fallback": False}
        else:
            logger.warning("deepFilter executable not found; copying repaired audio to %s", paths.dfn_audio)
            shutil.copy2(paths.repaired_audio, paths.dfn_audio)
            evaluation = {
                "fallback": True,
                "reason": "deepFilter executable not found",
            }
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step02a_deepfilternet.json", stage_metadata("deepfilternet", [paths.repaired_audio], [paths.dfn_audio], evaluation))


if __name__ == "__main__":
    main()
