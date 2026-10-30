import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
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


def run_pyannote(pipeline, audio_path):
    file = {"audio": str(audio_path), "uri": audio_path.stem}
    probabilities = pipeline._segmentation(file)
    annotation = pipeline._binarize(probabilities)
    annotation.uri = file["uri"]
    annotation = annotation.rename_labels({label: "SPEECH" for label in annotation.labels()})
    return probabilities, annotation


def extract_probability_curve(probabilities):
    data = np.asarray(probabilities.data)
    if data.ndim == 3:
        data = data.reshape((-1, data.shape[-1]))
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    values = np.max(data, axis=-1)
    frames = probabilities.sliding_window
    times = np.asarray([frames[i].middle for i in range(len(values))], dtype=float)
    return {"times": times, "probabilities": values.astype(float)}


def extract_raw_segments(annotation):
    return [{"start": round(s.start, 3), "end": round(s.end, 3)} for s in annotation.get_timeline().support()]


def merge_segments(segments, merge_gap):
    if not segments:
        return [], 0
    merged = [dict(segments[0])]
    merge_count = 0
    for seg in segments[1:]:
        if seg["start"] - merged[-1]["end"] < merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merge_count += 1
        else:
            merged.append(dict(seg))
    return merged, merge_count


def remove_short_segments(segments, min_duration):
    kept = [s for s in segments if s["end"] - s["start"] >= min_duration]
    return kept, len(segments) - len(kept)


def _smooth(values, frame_count):
    if frame_count <= 1:
        return values
    kernel = np.ones(frame_count, dtype=float) / frame_count
    return np.convolve(values, kernel, mode="same")


def find_probability_minimum(segment, curve, search_start, search_end, minimum_local_window):
    start = segment["start"] + search_start
    end = min(segment["start"] + search_end, segment["end"])
    times = curve.get("times")
    probs = curve.get("probabilities")
    if times is None or probs is None or len(times) == 0:
        return None
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        return None
    idxs = np.where(mask)[0]
    window_times = times[idxs]
    window_probs = probs[idxs]
    if len(window_probs) < 3:
        return float(window_times[int(np.argmin(window_probs))])
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    local_frames = max(1, int(round(minimum_local_window / max(frame_step, 1e-6))))
    smoothed = _smooth(window_probs, local_frames)
    local_minima = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            local_minima.append(i)
    if not local_minima:
        best = int(np.argmin(smoothed))
    else:
        best = min(local_minima, key=lambda i: smoothed[i])
    return float(window_times[best])


def find_rms_minimum(segment, audio_path, search_start, search_end, minimum_local_window):
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start_time = segment["start"] + search_start
    end_time = min(segment["start"] + search_end, segment["end"])
    start = max(0, int(start_time * sample_rate))
    end = min(len(audio), int(end_time * sample_rate))
    if end <= start:
        return None
    frame = max(1, int(minimum_local_window * sample_rate))
    best_time = None
    best_energy = None
    for offset in range(start, max(start + 1, end - frame + 1), max(1, frame // 4)):
        chunk = audio[offset : min(end, offset + frame)]
        if len(chunk) == 0:
            continue
        energy = float(np.sqrt(np.mean(np.square(chunk))))
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_time = (offset + len(chunk) / 2) / sample_rate
    return best_time


def recursive_split(segment, curve, audio_path, args, depth=0, operations=None):
    if operations is None:
        operations = []
    duration = segment["end"] - segment["start"]
    if duration <= args.max_segment_duration:
        return [segment], operations, depth
    split_at = find_probability_minimum(segment, curve, args.split_search_start, args.split_search_end, args.minimum_local_window)
    method = "probability"
    if split_at is None:
        split_at = find_rms_minimum(segment, audio_path, args.split_search_start, args.split_search_end, args.minimum_local_window)
        method = "rms_fallback"
    if split_at is None or split_at <= segment["start"] or split_at >= segment["end"]:
        split_at = min(segment["start"] + args.max_segment_duration, segment["end"])
        method = "duration_fallback"
    left = {"start": round(segment["start"], 3), "end": round(split_at, 3)}
    right = {"start": round(split_at, 3), "end": round(segment["end"], 3)}
    operations.append(
        {
            "input_start": round(segment["start"], 3),
            "input_end": round(segment["end"], 3),
            "input_duration": round(duration, 3),
            "split_at": round(split_at, 3),
            "method": method,
            "depth": depth,
        }
    )
    left_parts, operations, left_depth = recursive_split(left, curve, audio_path, args, depth + 1, operations)
    right_parts, operations, right_depth = recursive_split(right, curve, audio_path, args, depth + 1, operations)
    return left_parts + right_parts, operations, max(left_depth, right_depth)


def merge_final_chunks(segments, merge_gap, merge_max_duration):
    if not segments:
        return [], 0
    merged = [dict(segments[0])]
    merge_count = 0
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        merged_duration = seg["end"] - merged[-1]["start"]
        if gap < merge_gap and merged_duration <= merge_max_duration:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merge_count += 1
        else:
            merged.append(dict(seg))
    return merged, merge_count


def segment_stats(segments):
    durations = [round(s["end"] - s["start"], 3) for s in segments]
    return {
        "average_segment_duration": float(np.mean(durations)) if durations else 0.0,
        "maximum_segment_duration": max(durations) if durations else 0.0,
        "minimum_segment_duration": min(durations) if durations else 0.0,
        "total_speech_duration": round(sum(durations), 3),
        "duration_histogram": {
            "0-5": sum(1 for d in durations if 0 <= d < 5),
            "5-10": sum(1 for d in durations if 5 <= d < 10),
            "10-15": sum(1 for d in durations if 10 <= d < 15),
            "15-20": sum(1 for d in durations if 15 <= d < 20),
            "20-25": sum(1 for d in durations if 20 <= d < 25),
            "25-30": sum(1 for d in durations if 25 <= d <= 30),
            ">30": sum(1 for d in durations if d > 30),
        },
    }


def save_results(args, payload, curve, metadata):
    write_json(args.output, payload)
    probability_path = args.output.with_name(args.output.stem + "_probabilities.json")
    write_json(
        probability_path,
        {
            "file": args.input.name,
            "path": args.path_name,
            "times": [round(float(t), 4) for t in curve["times"]],
            "probabilities": [round(float(p), 6) for p in curve["probabilities"]],
        },
    )
    metadata["probability_file"] = str(probability_path)
    write_json(BASE_DIR / "outputs" / f"step03_{args.path_name}_vad.json", metadata)


def main():
    parser = argparse.ArgumentParser(description="Phase 20 Step 3: improved probability-guided Pyannote VAD.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-name", required=True, choices=["pathA", "pathB"])
    parser.add_argument("--merge-gap", type=float, default=0.15)
    parser.add_argument("--min-duration", type=float, default=0.10)
    parser.add_argument("--max-segment-duration", type=float, default=30.0)
    parser.add_argument("--merge-max-duration", type=float, default=28.0)
    parser.add_argument("--split-search-start", type=float, default=15.0)
    parser.add_argument("--split-search-end", type=float, default=30.0)
    parser.add_argument("--minimum-local-window", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = setup_logging(f"step03_{args.path_name}_vad")
    started = time.time()
    if args.output.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", args.output)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Pyannote VAD on %s", device)
    pipeline = load_pipeline(load_hf_token()).to(torch.device(device))
    probabilities, annotation = run_pyannote(pipeline, args.input)
    curve = extract_probability_curve(probabilities)
    raw = extract_raw_segments(annotation)
    merged, initial_merges = merge_segments(raw, args.merge_gap)
    filtered, removed_short = remove_short_segments(merged, args.min_duration)
    long_before = sum(1 for s in filtered if s["end"] - s["start"] > args.max_segment_duration)
    split_segments = []
    split_operations = []
    recursive_depth = 0
    for seg in filtered:
        parts, split_operations, depth = recursive_split(seg, curve, args.input, args, 0, split_operations)
        split_segments.extend(parts)
        recursive_depth = max(recursive_depth, depth)
    final_segments, final_merges = merge_final_chunks(sorted(split_segments, key=lambda s: s["start"]), args.merge_gap, args.merge_max_duration)
    stats_before = segment_stats(filtered)
    stats_after = segment_stats(final_segments)
    long_after = sum(1 for s in final_segments if s["end"] - s["start"] > args.max_segment_duration)
    processing_time = time.time() - started
    payload = {
        "file": args.input.name,
        "model": "pyannote/voice-activity-detection",
        "path": args.path_name,
        "duration": audio_duration(args.input),
        "raw_count": len(raw),
        "merged_count": len(merged),
        "filtered_count": len(filtered),
        "split_count": len(split_segments),
        "final_count": len(final_segments),
        "removed_short_segments": removed_short,
        "split_operations": split_operations,
        "recursive_depth": recursive_depth,
        "pre_split_duration_histogram": stats_before["duration_histogram"],
        "post_split_duration_histogram": stats_after["duration_histogram"],
        **{k: v for k, v in stats_after.items() if k != "duration_histogram"},
        "segments": final_segments,
    }
    metadata_eval = {
        "number_of_merges": initial_merges + final_merges,
        "initial_merges": initial_merges,
        "final_merges": final_merges,
        "number_of_splits": len(split_operations),
        "long_segments_detected": long_before,
        "long_segments_after_processing": long_after,
        "average_chunk_duration": stats_after["average_segment_duration"],
        "maximum_chunk_duration": stats_after["maximum_segment_duration"],
        "minimum_chunk_duration": stats_after["minimum_segment_duration"],
        "processing_time": processing_time,
    }
    metadata = stage_metadata(f"{args.path_name}_improved_vad", [args.input], [args.output], metadata_eval)
    save_results(args, payload, curve, metadata)
    logger.info("Saved %d final VAD segments to %s", len(final_segments), args.output)
    logger.info("Splits=%d max_duration=%.3fs long_after=%d", len(split_operations), stats_after["maximum_segment_duration"], long_after)


if __name__ == "__main__":
    main()
