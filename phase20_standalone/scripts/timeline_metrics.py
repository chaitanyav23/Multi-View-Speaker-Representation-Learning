"""
Frame-level speaker timeline metrics for Phase 20 fusion evaluation.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


DEFAULT_FRAME_RESOLUTION = 0.01


def speaker_timeline(
    segments: Iterable[Mapping[str, object]],
    duration: float,
    resolution: float = DEFAULT_FRAME_RESOLUTION,
    speaker_mapping: Mapping[str, str] | None = None,
) -> list[set[str]]:
    """
    Description
    -----------
    Convert timed speaker segments into a frame-level active-speaker timeline.

    Mathematical definition
    -----------------------
    For each frame k covering time interval [k*r, (k+1)*r), store the set of
    speakers whose segments overlap the frame.  Hypothesis speaker IDs may be
    renamed through a mapping before insertion.

    Equation
    --------
    T[k] = {m(s): segment(s) intersects [k r, (k+1) r)}

    Inputs
    ------
    segments:
        Timed segments with ``start``, ``end``, and ``speaker`` fields.
    duration:
        Recording duration in seconds.
    resolution:
        Frame size in seconds.
    speaker_mapping:
        Optional hypothesis-to-reference speaker mapping.

    Outputs
    -------
    list[set[str]]
        Active speaker set for every frame.  Empty frames have an empty set.

    Complexity
    ----------
    O(F + A), where F is number of frames and A is total segment-frame
    activations.

    Reference paper
    ---------------
    CHiME-7 DASR evaluates timed multi-speaker outputs; MeetEval's
    time-constrained work motivates frame/time-aware scoring for meeting
    transcription.

    Implementation notes
    --------------------
    Sets make overlapping speakers explicit instead of overwriting earlier
    speakers.  This supports arbitrary speaker IDs and overlapping speech while
    preserving the project's 10 ms default frame resolution.
    """

    if duration < 0:
        raise ValueError("duration must be non-negative")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    n_frames = int(math.ceil(duration / resolution)) + 1
    timeline = [set() for _ in range(n_frames)]
    mapping = speaker_mapping or {}
    for segment in segments:
        if segment.get("start") is None or segment.get("end") is None:
            continue
        start = max(0, int(float(segment["start"]) / resolution))
        end = min(n_frames, int(math.ceil(float(segment["end"]) / resolution)))
        if end <= start:
            continue
        speaker = str(segment.get("speaker", "UNK") or "UNK")
        speaker = mapping.get(speaker, speaker)
        if speaker == "UNK":
            continue
        for frame in range(start, end):
            timeline[frame].add(speaker)
    return timeline


def timeline_accuracy(
    reference_segments: Iterable[Mapping[str, object]],
    hypothesis_segments: Iterable[Mapping[str, object]],
    duration: float,
    resolution: float = DEFAULT_FRAME_RESOLUTION,
    speaker_mapping: Mapping[str, str] | None = None,
) -> float | None:
    """
    Description
    -----------
    Compute frame-level speaker timeline accuracy.

    Mathematical definition
    -----------------------
    Build reference and hypothesis active-speaker sets at the project frame
    resolution, optionally rename hypothesis speakers, and compute the fraction
    of frames with exactly matching speaker sets.

    Equation
    --------
    TimelineAccuracy = (1/F) sum_k 1[T_ref[k] = T_hyp[k]]

    Inputs
    ------
    reference_segments:
        Timed reference speaker segments.
    hypothesis_segments:
        Timed hypothesis speaker segments.
    duration:
        Recording duration in seconds.
    resolution:
        Frame size in seconds.  Defaults to 0.01 seconds.
    speaker_mapping:
        Optional hypothesis-to-reference speaker mapping.

    Outputs
    -------
    float | None
        Fraction of matching frames, or None if no frames are available.

    Complexity
    ----------
    O(F + A_ref + A_hyp), where F is frame count and A are segment-frame
    activations.

    Reference paper
    ---------------
    CHiME-7 DASR and MeetEval emphasize timed multi-speaker evaluation.  This
    project's timeline metric is a frame-level accuracy analogue for the
    speaker-attribution pipeline.

    Implementation notes
    --------------------
    The previous implementation compared raw labels and overwrote overlaps.
    This refactor supports different speaker IDs by applying the optimal
    mapping and supports overlap by comparing active-speaker sets.
    """

    reference = speaker_timeline(reference_segments, duration, resolution)
    hypothesis = speaker_timeline(hypothesis_segments, duration, resolution, speaker_mapping)
    n = min(len(reference), len(hypothesis))
    if n == 0:
        return None
    correct = sum(1 for idx in range(n) if reference[idx] == hypothesis[idx])
    return correct / n
