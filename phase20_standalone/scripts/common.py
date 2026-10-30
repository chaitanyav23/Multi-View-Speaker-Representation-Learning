import argparse
import datetime as _dt
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from speaker_mapping import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
MPL_DIR = Path("/tmp/marston_phase20_mpl")
CACHE_DIR = PROJECT_ROOT / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/marston_phase20_numba")
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))


def ensure_dirs():
    for name in [
        "raw_audio",
        "ground_truth",
        "ground_truth_json",
        "ground_truth_rttm",
        "audio",
        "outputs",
        "reports",
        "metrics",
        "grounding",
        "diarization",
        "transcripts",
        "transcription",
        "fusion",
        "evaluation",
        "scripts",
        "logs",
        "manifests",
    ]:
        (BASE_DIR / name).mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RecordingPaths:
    """Canonical standalone paths for one recording name."""

    name: str
    raw_audio: Path
    ground_truth_txt: Path
    ground_truth_json: Path
    ground_truth_rttm: Path
    repaired_audio: Path
    dfn_audio: Path
    normalized_audio: Path
    path_a_vad: Path
    path_b_vad: Path
    path_a_vad_metrics: Path
    path_b_vad_metrics: Path
    path_a_vad_report: Path
    path_b_vad_report: Path
    whisperx_transcript: Path
    aligned_words: Path
    diarization_rttm: Path
    asr_metrics: Path
    diar_metrics: Path
    fusion_metrics: Path
    asr_report: Path
    diar_report: Path
    fusion_report: Path
    attributed_words: Path
    fusion_transcript: Path
    all_metrics: Path
    pipeline_metrics: Path
    final_report: Path
    final_pipeline_report: Path
    phase19_comparison_json: Path
    phase20_vs_phase19_report: Path
    phase19_comparison_report: Path


def validate_recording_name(name: str) -> str:
    """Validate a recording stem supplied through --file."""

    if not name or Path(name).name != name or Path(name).suffix:
        raise ValueError("--file must be a filename stem such as 'traffic_stop', not a path or extension")
    return name


def recording_paths(name: str, base_dir: Path = BASE_DIR) -> RecordingPaths:
    """Return every pipeline path derived from one recording stem."""

    stem = validate_recording_name(name)
    return RecordingPaths(
        name=stem,
        raw_audio=base_dir / "raw_audio" / f"{stem}.wav",
        ground_truth_txt=base_dir / "ground_truth" / f"{stem}.txt",
        ground_truth_json=base_dir / "ground_truth_json" / f"{stem}.json",
        ground_truth_rttm=base_dir / "ground_truth_rttm" / f"{stem}.rttm",
        repaired_audio=base_dir / "audio" / f"{stem}_repaired.wav",
        dfn_audio=base_dir / "audio" / f"{stem}_dfn.wav",
        normalized_audio=base_dir / "audio" / f"{stem}_normalized.wav",
        path_a_vad=base_dir / "grounding" / f"{stem}_pathA_vad.json",
        path_b_vad=base_dir / "grounding" / f"{stem}_pathB_vad.json",
        path_a_vad_metrics=base_dir / "metrics" / f"{stem}_pathA_vad_metrics.json",
        path_b_vad_metrics=base_dir / "metrics" / f"{stem}_pathB_vad_metrics.json",
        path_a_vad_report=base_dir / "reports" / f"{stem}_pathA_vad_report.md",
        path_b_vad_report=base_dir / "reports" / f"{stem}_pathB_vad_report.md",
        whisperx_transcript=base_dir / "transcripts" / f"{stem}_whisperx.json",
        aligned_words=base_dir / "transcripts" / f"{stem}_aligned.json",
        diarization_rttm=base_dir / "diarization" / f"{stem}.rttm",
        asr_metrics=base_dir / "metrics" / f"{stem}_asr_metrics.json",
        diar_metrics=base_dir / "metrics" / f"{stem}_diar_metrics.json",
        fusion_metrics=base_dir / "metrics" / f"{stem}_fusion_metrics.json",
        asr_report=base_dir / "reports" / f"{stem}_asr_report.md",
        diar_report=base_dir / "reports" / f"{stem}_diar_report.md",
        fusion_report=base_dir / "reports" / f"{stem}_fusion_report.md",
        attributed_words=base_dir / "fusion" / f"{stem}_attributed_words.json",
        fusion_transcript=base_dir / "fusion" / f"{stem}_transcript.txt",
        all_metrics=base_dir / "metrics" / f"{stem}_all_metrics.json",
        pipeline_metrics=base_dir / "metrics" / f"{stem}_pipeline_metrics.json",
        final_report=base_dir / "reports" / f"{stem}_final_report.md",
        final_pipeline_report=base_dir / "reports" / f"{stem}_final_pipeline_report.md",
        phase19_comparison_json=base_dir / "metrics" / f"{stem}_phase20_vs_phase19_comparison.json",
        phase20_vs_phase19_report=base_dir / "reports" / f"{stem}_phase20_vs_phase19_comparison.md",
        phase19_comparison_report=base_dir / "reports" / f"{stem}_phase19_comparison.md",
    )


def add_file_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the required recording stem argument used by standalone steps."""

    parser.add_argument("--file", required=True, help="Recording stem, e.g. traffic_stop for raw_audio/traffic_stop.wav")
    return parser


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
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/marston_phase20_numba")
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
    import math

    mask = [False] * (int(math.ceil(duration / resolution)) + 1)
    for seg in segments:
        start = max(0, int(float(seg["start"]) / resolution))
        end = min(len(mask), int(math.ceil(float(seg["end"]) / resolution)))
        if end > start:
            for idx in range(start, end):
                mask[idx] = True
    return mask


def vad_metrics(reference_segments, hypothesis_segments, duration: float, resolution=0.01):
    ref = activity_mask(reference_segments, duration, resolution)
    hyp = activity_mask(hypothesis_segments, duration, resolution)
    n = min(len(ref), len(hyp))
    ref = ref[:n]
    hyp = hyp[:n]
    tp = sum(1 for r, h in zip(ref, hyp) if r and h)
    tn = sum(1 for r, h in zip(ref, hyp) if not r and not h)
    fp = sum(1 for r, h in zip(ref, hyp) if not r and h)
    fn = sum(1 for r, h in zip(ref, hyp) if r and not h)
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
    try:
        import jiwer

        return jiwer.transforms.Compose(
            [
                jiwer.transforms.ToLowerCase(),
                jiwer.transforms.RemovePunctuation(),
                jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
                jiwer.transforms.RemoveMultipleSpaces(),
                jiwer.transforms.Strip(),
            ]
        )(text or "")
    except ModuleNotFoundError:
        from speaker_mapping import normalize_text as _normalize_text

        return _normalize_text(text or "")


def text_from_gt(gt_json: Path) -> str:
    data = read_json(gt_json)
    return " ".join(seg.get("text", "") for seg in data.get("segments", []))


def text_from_words(words) -> str:
    return " ".join(w.get("word", "") for w in words if w.get("word"))


def asr_metrics(reference_text: str, hypothesis_text: str):
    ref = normalize_text(reference_text)
    hyp = normalize_text(hypothesis_text)
    from speaker_mapping import edit_distance_counts

    counts = edit_distance_counts(ref.split() if ref else [], hyp.split() if hyp else [])
    return {
        "wer": counts["errors"] / len(ref.split()) if ref.split() else 0.0,
        "substitutions": counts["substitutions"],
        "insertions": counts["insertions"],
        "deletions": counts["deletions"],
        "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()),
    }


def diarization_metrics(reference_rttm: Path, hypothesis_rttm: Path):
    try:
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
    except ModuleNotFoundError:
        pass

    from speaker_mapping import optimal_speaker_mapping

    reference_segments = rttm_to_segments(reference_rttm)
    hypothesis_segments = rttm_to_segments(hypothesis_rttm)
    if not reference_segments:
        return {"der": 0.0, "miss": 0.0, "false_alarm": 0.0, "confusion": 0.0, "jer": None, "total_reference_time": 0.0}

    resolution = 0.01
    duration = max([seg["end"] for seg in reference_segments + hypothesis_segments] or [0.0])

    def active_at(segments, frame):
        start = frame * resolution
        end = start + resolution
        return {
            seg["speaker"]
            for seg in segments
            if min(end, seg["end"]) - max(start, seg["start"]) > 0
        }

    frames = int(duration / resolution) + 1
    pairs = []
    for idx in range(frames):
        ref_active = active_at(reference_segments, idx)
        hyp_active = active_at(hypothesis_segments, idx)
        if len(ref_active) == 1 and len(hyp_active) == 1:
            pairs.append((next(iter(hyp_active)), next(iter(ref_active))))
    mapping = optimal_speaker_mapping(pairs)

    ref_speech = miss = false_alarm = confusion = 0
    for idx in range(frames):
        ref_active = active_at(reference_segments, idx)
        hyp_active = {mapping.get(s, s) for s in active_at(hypothesis_segments, idx)}
        if ref_active:
            ref_speech += 1
            if not hyp_active:
                miss += 1
            elif ref_active.isdisjoint(hyp_active):
                confusion += 1
        elif hyp_active:
            false_alarm += 1

    total = ref_speech * resolution
    denom = ref_speech or 1
    miss_rate = miss / denom
    fa_rate = false_alarm / denom
    conf_rate = confusion / denom
    ref_speakers = sorted({seg["speaker"] for seg in reference_segments})
    hyp_speakers = sorted({mapping.get(seg["speaker"], seg["speaker"]) for seg in hypothesis_segments})
    jaccards = []
    for speaker in ref_speakers:
        ref_frames = {idx for idx in range(frames) if speaker in active_at(reference_segments, idx)}
        best = 0.0
        for hyp_speaker in hyp_speakers:
            hyp_frames = {idx for idx in range(frames) if hyp_speaker in {mapping.get(s, s) for s in active_at(hypothesis_segments, idx)}}
            union = ref_frames | hyp_frames
            if union:
                best = max(best, len(ref_frames & hyp_frames) / len(union))
        jaccards.append(best)
    jer = 1.0 - (sum(jaccards) / len(jaccards)) if jaccards else None
    return {
        "der": miss_rate + fa_rate + conf_rate,
        "miss": miss_rate,
        "false_alarm": fa_rate,
        "confusion": conf_rate,
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
    """
    Compute optimal one-to-one mapping from system speakers
    to reference speakers using the Hungarian algorithm.

    Parameters
    ----------
    pairs : list[(system_speaker, reference_speaker)]

    Returns
    -------
    dict
        system speaker -> reference speaker
    """

    import numpy as np

    sys_spks = sorted({s for s, _ in pairs if s != "UNK"})
    ref_spks = sorted({r for _, r in pairs if r != "UNK"})

    if not sys_spks or not ref_spks:
        return {}

    score = np.zeros((len(sys_spks), len(ref_spks)), dtype=int)

    sys_index = {s: i for i, s in enumerate(sys_spks)}
    ref_index = {r: j for j, r in enumerate(ref_spks)}

    for sys_spk, ref_spk in pairs:
        if sys_spk == "UNK" or ref_spk == "UNK":
            continue

        score[
            sys_index[sys_spk],
            ref_index[ref_spk]
        ] += 1

    rows, cols = linear_sum_assignment(-score)

    mapping = {}

    for r, c in zip(rows, cols):
        mapping[sys_spks[r]] = ref_spks[c]

    return mapping

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
    """
    Concatenated minimum-permutation WER (cpWER).

    Reference speakers and hypothesis speakers are matched
    using minimum-cost bipartite matching (Hungarian algorithm).
    """

    import jiwer
    import numpy as np

    gt_by_spk = {}
    sys_by_spk = {}

    for seg in gt_segments:
        spk = seg.get("speaker", "UNK")
        gt_by_spk.setdefault(spk, "")
        gt_by_spk[spk] += " " + seg.get("text", "")

    for seg in sys_segments:
        spk = seg.get("speaker", "UNK")
        sys_by_spk.setdefault(spk, "")
        sys_by_spk[spk] += " " + seg.get("text", "")

    refs = []
    hyps = []

    for txt in gt_by_spk.values():
        txt = normalize_text(txt)
        if txt:
            refs.append(txt)

    for txt in sys_by_spk.values():
        txt = normalize_text(txt)
        if txt:
            hyps.append(txt)

    if len(refs) == 0:
        return 0.0

    if len(hyps) == 0:
        return 1.0

    n = max(len(refs), len(hyps))

    while len(refs) < n:
        refs.append("")

    while len(hyps) < n:
        hyps.append("")

    cost = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            res = jiwer.process_words(refs[i], hyps[j])

            cost[i, j] = (
                res.substitutions
                + res.insertions
                + res.deletions
            )

    rows, cols = linear_sum_assignment(cost)

    total_errors = cost[rows, cols].sum()

    total_ref_words = sum(len(r.split()) for r in refs)

    if total_ref_words == 0:
        return 0.0

    return total_errors / total_ref_words

def compute_sawer(gt_segments, sys_segments, speaker_mapping):
    """
    Speaker-attributed WER.

    System speakers are first renamed according
    to the optimal speaker mapping.

    Ordinary WER is then computed speaker-by-speaker.
    """

    import jiwer

    gt_by_spk = {}
    sys_by_spk = {}

    for seg in gt_segments:
        spk = seg.get("speaker", "UNK")
        gt_by_spk.setdefault(spk, "")
        gt_by_spk[spk] += " " + seg.get("text", "")

    for seg in sys_segments:

        sys_spk = seg.get("speaker", "UNK")

        ref_spk = speaker_mapping.get(sys_spk, sys_spk)

        sys_by_spk.setdefault(ref_spk, "")
        sys_by_spk[ref_spk] += " " + seg.get("text", "")

    speakers = sorted(
        set(gt_by_spk.keys()) |
        set(sys_by_spk.keys())
    )

    total_errors = 0
    total_words = 0

    for spk in speakers:

        ref = normalize_text(gt_by_spk.get(spk, ""))

        hyp = normalize_text(sys_by_spk.get(spk, ""))

        res = jiwer.process_words(ref, hyp)

        total_errors += (
            res.substitutions
            + res.insertions
            + res.deletions
        )

        total_words += len(ref.split())

    if total_words == 0:
        return 0.0

    return total_errors / total_words

def speaker_timeline(segments, duration, resolution=0.01):
    """
    Create a frame-level speaker timeline.

    Each frame contains the active speaker label or "UNK".
    """

    import numpy as np

    n_frames = int(np.ceil(duration / resolution)) + 1

    timeline = np.full(n_frames, "UNK", dtype=object)

    for seg in segments:

        speaker = seg.get("speaker", "UNK")

        start = max(
            0,
            int(float(seg["start"]) / resolution)
        )

        end = min(
            n_frames,
            int(np.ceil(float(seg["end"]) / resolution))
        )

        if end > start:
            timeline[start:end] = speaker

    return timeline

def fusion_metrics(gt_json: Path, attributed_path: Path, duration: float):
    """
    Description
    -----------
    Compatibility wrapper for the standalone Phase 20 fusion evaluator.

    Mathematical definition
    -----------------------
    Delegates to ``scripts/fusion_metrics.py``, which computes cpWER, SA-WER,
    Speaker Accuracy, Word Attribution Error, and Timeline Accuracy.

    Equation
    --------
    metrics = fusion_metrics_module(gt_json, attributed_path, duration)

    Inputs
    ------
    gt_json:
        Ground-truth JSON path.
    attributed_path:
        Fusion attributed JSON path.
    duration:
        Recording duration in seconds.

    Outputs
    -------
    dict
        Public fusion metric JSON schema.

    Complexity
    ----------
    Same as ``fusion_metrics.fusion_metrics``.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394; CHiME-7 DASR;
    ESPnet diarization-attributed ASR evaluation practice.

    Implementation notes
    --------------------
    This keeps existing callers unchanged while the implementation lives in
    focused standalone modules.
    """

    from fusion_metrics import fusion_metrics as _fusion_metrics

    return _fusion_metrics(gt_json, attributed_path, duration)


def load_hf_token():
    from dotenv import dotenv_values

    paths = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform" / ".env",
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
    spec = importlib.util.spec_from_file_location("phase20_whisperx_utils", str(path))
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
