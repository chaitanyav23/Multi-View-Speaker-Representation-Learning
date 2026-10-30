import argparse
from pathlib import Path

import torch
import yaml
from pyannote.audio import Pipeline
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub

from phase19_common import BASE_DIR, audio_duration, load_hf_token, setup_logging, stage_metadata, write_json


def load_pipeline(token):
    model_id = "pyannote/voice-activity-detection"
    config_path = download_from_hf_hub(model_id, AssetFileName.Pipeline, token=token)
    if config_path is None:
        raise RuntimeError(f"Could not load {model_id}. Check HF token/model access.")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    segmentation = config.get("pipeline", {}).get("params", {}).get("segmentation")
    if isinstance(segmentation, str) and "@" in segmentation:
        checkpoint, revision = segmentation.split("@", 1)
        config["pipeline"]["params"]["segmentation"] = {"checkpoint": checkpoint, "revision": revision}
    return Pipeline.from_pretrained(config, token=token)


def merge_segments(segments, merge_gap=0.15, min_duration=0.1):
    merged = []
    for seg in segments:
        if not merged or seg["start"] - merged[-1]["end"] >= merge_gap:
            merged.append(dict(seg))
        else:
            merged[-1]["end"] = seg["end"]
    return [s for s in merged if s["end"] - s["start"] >= min_duration]


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 3: Pyannote 3.1 VAD.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-name", required=True, choices=["pathA", "pathB"])
    parser.add_argument("--merge-gap", type=float, default=0.15)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = setup_logging(f"step03_{args.path_name}_vad")
    if args.output.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", args.output)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Pyannote VAD on %s", device)
    pipeline = load_pipeline(load_hf_token()).to(torch.device(device))
    vad = pipeline(str(args.input))
    raw = [{"start": round(s.start, 3), "end": round(s.end, 3)} for s in vad.get_timeline().support()]
    segments = merge_segments(raw, args.merge_gap, args.min_duration)
    payload = {
        "file": args.input.name,
        "path": args.path_name,
        "model": "pyannote/voice-activity-detection",
        "duration": audio_duration(args.input),
        "raw_count": len(raw),
        "merged_count": len(segments),
        "total_speech_duration": round(sum(s["end"] - s["start"] for s in segments), 3),
        "segments": segments,
    }
    write_json(args.output, payload)
    write_json(BASE_DIR / "outputs" / f"step03_{args.path_name}_vad.json", stage_metadata(f"{args.path_name}_vad", [args.input], [args.output], {"segments": len(segments)}))
    logger.info("Saved %d VAD segments to %s", len(segments), args.output)


if __name__ == "__main__":
    main()
