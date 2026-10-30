import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Ground Truth Parsing
# ============================================================

GT_PATTERN = re.compile(
    r"\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]"
)


def hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_ground_truth(gt_file):
    """
    Parse:

    Speaker 1 [00:00:00 - 00:00:10]: ...
    """

    segments = []

    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:

            match = GT_PATTERN.search(line)

            if not match:
                continue

            h1, m1, s1, h2, m2, s2 = match.groups()

            start = hms_to_seconds(h1, m1, s1)
            end = hms_to_seconds(h2, m2, s2)

            if end <= start:
                continue

            segments.append((float(start), float(end)))

    return segments


# ============================================================
# VAD JSON Parsing
# ============================================================

def load_vad_segments(vad_json):

    with open(vad_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []

    for seg in data["segments"]:

        start = float(seg["start"])
        end = float(seg["end"])

        if end > start:
            segments.append((start, end))

    duration = float(data["duration_sec"])

    return segments, duration


# ============================================================
# Frame Label Conversion
# ============================================================

def segments_to_frame_labels(
    segments,
    duration_sec,
    fps=100
):
    """
    fps=100 -> 10 ms resolution
    """

    n_frames = int(duration_sec * fps)

    labels = np.zeros(n_frames, dtype=np.uint8)

    for start, end in segments:

        s_idx = max(0, int(start * fps))
        e_idx = min(n_frames, int(end * fps))

        labels[s_idx:e_idx] = 1

    return labels


# ============================================================
# Segment Utilities
# ============================================================

def intersection(seg1, seg2):

    s1, e1 = seg1
    s2, e2 = seg2

    return max(
        0,
        min(e1, e2) - max(s1, s2)
    )


def union(seg1, seg2):

    s1, e1 = seg1
    s2, e2 = seg2

    return max(e1, e2) - min(s1, s2)


def iou(seg1, seg2):

    inter = intersection(seg1, seg2)

    if inter <= 0:
        return 0.0

    return inter / union(seg1, seg2)


# ============================================================
# Segment-Level Metrics
# ============================================================

def compute_segment_metrics(
    gt_segments,
    pred_segments,
    overlap_threshold=0.1,
):
    """
    Segment Recall
    Segment Precision
    Mean IoU
    Boundary Error
    Speech Coverage
    """

    matched_gt = 0
    matched_pred = 0

    ious = []
    boundary_errors = []

    coverage_numerator = 0.0
    coverage_denominator = 0.0

    # --------------------------------------------------------
    # GT coverage
    # --------------------------------------------------------

    for gt in gt_segments:

        gt_len = gt[1] - gt[0]

        coverage_denominator += gt_len

        best_iou = 0.0
        best_pred = None

        overlap_duration = 0.0

        for pred in pred_segments:

            overlap_duration += intersection(
                gt,
                pred
            )

            current_iou = iou(
                gt,
                pred
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_pred = pred

        coverage_numerator += min(
            overlap_duration,
            gt_len
        )

        if best_iou >= overlap_threshold:

            matched_gt += 1

            ious.append(best_iou)

            start_err = abs(
                gt[0] - best_pred[0]
            )

            end_err = abs(
                gt[1] - best_pred[1]
            )

            boundary_errors.append(
                (start_err + end_err) / 2
            )

    # --------------------------------------------------------
    # Segment Precision
    # --------------------------------------------------------

    for pred in pred_segments:

        best_iou = 0

        for gt in gt_segments:

            best_iou = max(
                best_iou,
                iou(gt, pred)
            )

        if best_iou >= overlap_threshold:
            matched_pred += 1

    segment_recall = (
        matched_gt / len(gt_segments)
        if gt_segments
        else 0
    )

    segment_precision = (
        matched_pred / len(pred_segments)
        if pred_segments
        else 0
    )

    mean_iou = (
        np.mean(ious)
        if ious
        else 0
    )

    boundary_error = (
        np.mean(boundary_errors)
        if boundary_errors
        else 0
    )

    speech_coverage = (
        coverage_numerator /
        coverage_denominator
        if coverage_denominator > 0
        else 0
    )

    return {
        "speech_coverage": round(float(speech_coverage), 4),
        "segment_recall": round(float(segment_recall), 4),
        "segment_precision": round(float(segment_precision), 4),
        "mean_iou": round(float(mean_iou), 4),
        "boundary_error_sec": round(float(boundary_error), 4),
    }


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_vad(
    gt_file,
    vad_json,
    fps=100,
):
    """
    Main entrypoint
    """

    gt_segments = parse_ground_truth(gt_file)

    pred_segments, duration_sec = load_vad_segments(
        vad_json
    )

    y_true = segments_to_frame_labels(
        gt_segments,
        duration_sec,
        fps=fps,
    )

    y_pred = segments_to_frame_labels(
        pred_segments,
        duration_sec,
        fps=fps,
    )

    # --------------------------------------------------------
    # Frame Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    miss_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    segment_metrics = compute_segment_metrics(
        gt_segments,
        pred_segments,
    )

    results = {
        "gt_file": Path(gt_file).name,
        "vad_file": Path(vad_json).name,

        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),

        "miss_rate": round(float(miss_rate), 4),
        "false_alarm_rate": round(
            float(false_alarm_rate),
            4
        ),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

        "gt_segments": len(gt_segments),
        "pred_segments": len(pred_segments),
    }

    results.update(segment_metrics)

    return results


# ============================================================
# CSV Export
# ============================================================

def append_vad_eval_row(
    video_id,
    metrics,
    csv_path,
):

    row = {
        "video_id": video_id,
        **metrics
    }

    csv_path = Path(csv_path)

    if csv_path.exists():

        df = pd.read_csv(csv_path)

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True
        )

    else:

        df = pd.DataFrame([row])

    df.to_csv(
        csv_path,
        index=False
    )


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    metrics = evaluate_vad(
        gt_file="ground_truth/bwc.txt",
        vad_json="vad_outputs/bwc_vad.json",
    )

    print(json.dumps(
        metrics,
        indent=2
        ))
