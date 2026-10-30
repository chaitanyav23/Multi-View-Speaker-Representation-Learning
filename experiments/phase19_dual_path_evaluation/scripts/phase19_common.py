import argparse
import datetime as _dt
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import jiwer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
MPL_DIR = Path("/tmp/marston_phase19_mpl")
CACHE_DIR = PROJECT_ROOT / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/marston_phase19_numba")
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))


def ensure_dirs():
    for name in [
        "audio",
        "outputs",
        "reports",
        "metrics",
        "grounding",
        "diarization",
        "transcription",
        "fusion",
        "evaluation",
        "scripts",
        "logs",
        "manifests",
    ]:
        (BASE_DIR / name).mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(stage: str) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(LOG_DIR / f"{stage}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def env_for_subprocess():
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(MPL_DIR))
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/marston_phase19_numba")
    env.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    env.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
    env.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
    return env


def run_command(cmd, logger: logging.Logger, description: str):
    logger.info("Running %s: %s", description, " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env_for_subprocess(), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_report(path: Path, title: str, sections):
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", ""])
        if isinstance(body, dict):
            lines.extend([f"- `{k}`: {v}" for k, v in body.items()])
        elif isinstance(body, list):
            lines.extend([f"- {item}" for item in body])
        else:
            lines.append(str(body))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Could not probe {path}")
    return float(result.stdout.strip())


def load_segments_json(path: Path):
    data = read_json(path)
    return data.get("segments", data)


def rttm_to_segments(path: Path):
    segments = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                dur = float(parts[4])
                if dur <= 0:
                    continue
                segments.append({"start": start, "end": start + dur, "speaker": parts[7]})
    return segments


def write_oracle_rttm(vad_json: Path, output_rttm: Path, file_id: str):
    segments = load_segments_json(vad_json)
    output_rttm.parent.mkdir(parents=True, exist_ok=True)
    with output_rttm.open("w") as f:
        for seg in segments:
            start = float(seg["start"])
            dur = float(seg["end"]) - start
            if dur <= 0:
                continue
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> speaker <NA> <NA>\n")


def activity_mask(segments, duration: float, resolution=0.01):
    mask = np.zeros(int(np.ceil(duration / resolution)) + 1, dtype=bool)
    for seg in segments:
        start = max(0, int(float(seg["start"]) / resolution))
        end = min(len(mask), int(np.ceil(float(seg["end"]) / resolution)))
        if end > start:
            mask[start:end] = True
    return mask


def vad_metrics(reference_segments, hypothesis_segments, duration: float, resolution=0.01):
    ref = activity_mask(reference_segments, duration, resolution)
    hyp = activity_mask(hypothesis_segments, duration, resolution)
    n = min(len(ref), len(hyp))
    ref = ref[:n]
    hyp = hyp[:n]
    tp = int(np.logical_and(ref, hyp).sum())
    tn = int(np.logical_and(~ref, ~hyp).sum())
    fp = int(np.logical_and(~ref, hyp).sum())
    fn = int(np.logical_and(ref, ~hyp).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm": fp / (tn + fp) if tn + fp else 0.0,
        "miss": fn / (tp + fn) if tp + fn else 0.0,
        "tp_frames": tp,
        "tn_frames": tn,
        "fp_frames": fp,
        "fn_frames": fn,
        "resolution_sec": resolution,
    }


def normalize_text(text: str) -> str:
    return jiwer.transforms.Compose(
        [
            jiwer.transforms.ToLowerCase(),
            jiwer.transforms.RemovePunctuation(),
            jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
            jiwer.transforms.RemoveMultipleSpaces(),
            jiwer.transforms.Strip(),
        ]
    )(text or "")


def text_from_gt(gt_json: Path) -> str:
    data = read_json(gt_json)
    return " ".join(seg.get("text", "") for seg in data.get("segments", []))


def text_from_words(words) -> str:
    return " ".join(w.get("word", "") for w in words if w.get("word"))


def asr_metrics(reference_text: str, hypothesis_text: str):
    ref = normalize_text(reference_text)
    hyp = normalize_text(hypothesis_text)
    res = jiwer.process_words(ref, hyp)
    return {
        "wer": res.wer,
        "substitutions": res.substitutions,
        "insertions": res.insertions,
        "deletions": res.deletions,
        "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()),
    }


def diarization_metrics(reference_rttm: Path, hypothesis_rttm: Path):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    def load(path):
        ann = Annotation()
        for seg in rttm_to_segments(path):
            ann[Segment(seg["start"], seg["end"])] = seg["speaker"]
        return ann

    reference = load(reference_rttm)
    hypothesis = load(hypothesis_rttm)
    der = DiarizationErrorRate()
    details = der(reference, hypothesis, detailed=True)
    total = details.get("total", 0.0) or 0.0
    try:
        jer = JaccardErrorRate()(reference, hypothesis)
    except Exception:
        jer = None
    return {
        "der": details.get("diarization error rate", 0.0),
        "miss": details.get("missed detection", 0.0) / total if total else 0.0,
        "false_alarm": details.get("false alarm", 0.0) / total if total else 0.0,
        "confusion": details.get("confusion", 0.0) / total if total else 0.0,
        "jer": jer,
        "total_reference_time": total,
    }


def max_overlap_speaker(word, diar_segments, tolerance=0.5):
    if word.get("start") is None or word.get("end") is None:
        return "UNK"
    w_start = float(word["start"])
    w_end = float(word["end"])
    overlaps = []
    for seg in diar_segments:
        overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
        if overlap > 0:
            overlaps.append((overlap, seg["speaker"]))
    if overlaps:
        return max(overlaps)[1]
    midpoint = (w_start + w_end) / 2
    nearest = None
    best = float("inf")
    for seg in diar_segments:
        dist = min(abs(midpoint - seg["start"]), abs(midpoint - seg["end"]))
        if dist < best:
            best = dist
            nearest = seg["speaker"]
    return nearest if nearest and best <= tolerance else "UNK"


def attributed_words(words, diar_segments):
    output = []
    for word in words:
        item = {
            "word": word.get("word"),
            "start": word.get("start"),
            "end": word.get("end"),
            "confidence": word.get("confidence", word.get("score")),
            "speaker": max_overlap_speaker(word, diar_segments),
        }
        output.append(item)
    return output


def group_words(words, max_gap=2.0):
    grouped = []
    current = None
    for word in words:
        if not word.get("word"):
            continue
        if current is None:
            current = {
                "speaker": word.get("speaker", "UNK"),
                "start": word.get("start"),
                "end": word.get("end"),
                "text": word["word"],
                "words": [word],
            }
            continue
        gap = 0 if word.get("start") is None or current.get("end") is None else word["start"] - current["end"]
        if word.get("speaker") == current.get("speaker") and gap <= max_gap:
            current["end"] = word.get("end")
            current["text"] += " " + word["word"]
            current["words"].append(word)
        else:
            grouped.append(current)
            current = {
                "speaker": word.get("speaker", "UNK"),
                "start": word.get("start"),
                "end": word.get("end"),
                "text": word["word"],
                "words": [word],
            }
    if current:
        grouped.append(current)
    return grouped


def fmt_ts(seconds):
    if seconds is None:
        seconds = 0
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def write_transcript(path: Path, segments):
    speaker_names = {}
    next_idx = 1
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNK")
        if speaker not in speaker_names and speaker != "UNK":
            speaker_names[speaker] = f"Speaker_{next_idx}"
            next_idx += 1
        label = speaker_names.get(speaker, "Speaker_Unknown")
        lines.append(f"{label} [{fmt_ts(seg.get('start'))} - {fmt_ts(seg.get('end'))}]")
        lines.append(seg.get("text", ""))
        lines.append("")
    path.write_text("\n".join(lines))


def optimal_speaker_mapping(pairs):
    sys_spks = sorted({p[0] for p in pairs})
    ref_spks = sorted({p[1] for p in pairs if p[1] != "UNK"})
    if not sys_spks or not ref_spks:
        return {}
    counts = {(s, r): 0 for s in sys_spks for r in ref_spks}
    for sys_spk, ref_spk in pairs:
        if ref_spk != "UNK":
            counts[(sys_spk, ref_spk)] += 1
    best_map = {}
    best_score = -1
    for ref_perm in permutations(ref_spks, min(len(sys_spks), len(ref_spks))):
        mapping = dict(zip(sys_spks, ref_perm))
        score = sum(counts.get((s, r), 0) for s, r in mapping.items())
        if score > best_score:
            best_score = score
            best_map = mapping
    return best_map


def gt_speaker_at_time(gt_segments, start, end):
    if start is None or end is None:
        return "UNK"
    mid = (float(start) + float(end)) / 2
    best = "UNK"
    best_overlap = 0
    for seg in gt_segments:
        overlap = min(float(end), float(seg["end"])) - max(float(start), float(seg["start"]))
        if overlap > best_overlap or (best_overlap == 0 and seg["start"] <= mid <= seg["end"]):
            best_overlap = overlap
            best = seg.get("speaker", "UNK")
    return best


def compute_cpwer(gt_segments, sys_segments):
    gt_by_spk = {}
    sys_by_spk = {}
    for seg in gt_segments:
        gt_by_spk.setdefault(seg.get("speaker", "UNK"), "")
        gt_by_spk[seg.get("speaker", "UNK")] += " " + seg.get("text", "")
    for seg in sys_segments:
        sys_by_spk.setdefault(seg.get("speaker", "UNK"), "")
        sys_by_spk[seg.get("speaker", "UNK")] += " " + seg.get("text", "")
    refs = [normalize_text(v) for v in gt_by_spk.values() if normalize_text(v)]
    hyps = [normalize_text(v) for v in sys_by_spk.values() if normalize_text(v)]
    if not refs:
        return 0.0
    if not hyps:
        return 1.0
    total_err = 0
    total_words = 0
    for ref in refs:
        best = None
        for hyp in hyps:
            res = jiwer.process_words(ref, hyp)
            err = res.substitutions + res.insertions + res.deletions
            best = err if best is None else min(best, err)
        total_err += best or 0
        total_words += len(ref.split())
    return total_err / total_words if total_words else 0.0


def fusion_metrics(gt_json: Path, attributed_path: Path, duration: float):
    gt = read_json(gt_json)
    sys_data = read_json(attributed_path)
    gt_segments = gt.get("segments", [])
    words = sys_data.get("words", [])
    sys_segments = sys_data.get("segments", [])
    pairs = []
    for word in words:
        ref = gt_speaker_at_time(gt_segments, word.get("start"), word.get("end"))
        pairs.append((word.get("speaker", "UNK"), ref))
    mapping = optimal_speaker_mapping(pairs)
    total = sum(1 for _, ref in pairs if ref != "UNK")
    correct = sum(1 for sys_spk, ref in pairs if ref != "UNK" and mapping.get(sys_spk) == ref)
    speaker_accuracy = correct / total if total else None
    cpwer = compute_cpwer(gt_segments, sys_segments)
    wder = 1 - speaker_accuracy if speaker_accuracy is not None else None
    gt_turns = activity_mask(gt_segments, duration)
    sys_turns = activity_mask(sys_segments, duration)
    n = min(len(gt_turns), len(sys_turns))
    timeline_accuracy = float((gt_turns[:n] == sys_turns[:n]).sum() / n) if n else None
    return {
        "cpwer": cpwer,
        "sawer": cpwer,
        "wder": wder,
        "speaker_accuracy": speaker_accuracy,
        "word_attribution_error": wder,
        "turn_accuracy": speaker_accuracy,
        "timeline_accuracy": timeline_accuracy,
        "words_evaluated": total,
        "correct_attributions": correct,
        "speaker_mapping": mapping,
    }


def load_hf_token():
    from dotenv import dotenv_values

    paths = [
        PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform" / ".env",
        PROJECT_ROOT / ".env",
    ]
    for path in paths:
        if path.exists():
            token = dotenv_values(path).get("HF_TOKEN")
            if token:
                return token
    raise RuntimeError("HF_TOKEN not found in .env")


def load_external_whisperx_utils():
    path = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform" / "utils.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("phase19_whisperx_utils", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Phase 19 base directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser


def stage_metadata(stage, inputs, outputs, evaluation=None):
    return {
        "stage": stage,
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "inputs": [str(p) for p in inputs],
        "outputs": [str(p) for p in outputs],
        "evaluation": evaluation or {},
    }


def copy_or_replace(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy(src, dst)
