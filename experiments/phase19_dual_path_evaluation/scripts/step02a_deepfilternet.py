import argparse
from pathlib import Path

from phase19_common import BASE_DIR, run_command, setup_logging, stage_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 2A: DeepFilterNet enhancement for diarization path.")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "audio" / "bwc_repaired.wav")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "audio" / "bwc_dfn.wav")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("step02a_deepfilternet")
    if args.output.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", args.output)
    else:
        temp_dir = BASE_DIR / "audio" / "dfn_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        run_command(["deepFilter", args.input, "-o", temp_dir], logger, "DeepFilterNet")
        candidates = sorted(temp_dir.glob("*.wav"))
        if not candidates:
            raise RuntimeError(f"DeepFilterNet did not produce a wav in {temp_dir}")
        candidates[0].replace(args.output)
    write_json(BASE_DIR / "outputs" / "step02a_deepfilternet.json", stage_metadata("deepfilternet", [args.input], [args.output]))


if __name__ == "__main__":
    main()
