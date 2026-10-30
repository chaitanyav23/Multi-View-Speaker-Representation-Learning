"""
Public fusion evaluation API for Phase 20.

The caller-facing function remains ``fusion_metrics(gt_json, attributed_json,
duration)``.  Internally it delegates cpWER, SA-WER, attribution, and timeline
metrics to focused local modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

from attribution_metrics import speaker_accuracy_stats, word_speaker_pairs
from cpwer import cpwer
from sawer import sawer
from speaker_mapping import concatenate_by_speaker, merge_mappings, transcript_speaker_mapping
from timeline_metrics import timeline_accuracy


def read_json(path: Path):
    """
    Description
    -----------
    Read a JSON file for fusion metric evaluation.

    Mathematical definition
    -----------------------
    Not a metric; this is the identity load operation from serialized JSON to
    Python objects.

    Equation
    --------
    data = JSON^{-1}(bytes(path))

    Inputs
    ------
    path:
        JSON file path.

    Outputs
    -------
    object
        Parsed JSON payload.

    Complexity
    ----------
    O(B), where B is the number of bytes in the file.

    Reference paper
    ---------------
    MeetEval supports JSON SegLST inputs for meeting transcription scoring.

    Implementation notes
    --------------------
    Kept local to avoid importing ``common.py`` and creating a circular import
    when ``common.fusion_metrics`` delegates to this module.
    """

    with Path(path).open() as f:
        return json.load(f)


def segments_from_payload(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """
    Description
    -----------
    Extract speaker-attributed segments from a project JSON payload.

    Mathematical definition
    -----------------------
    Not a scoring equation; returns the top-level segments sequence when
    present, otherwise an empty sequence.

    Equation
    --------
    segments(payload) = payload["segments"] if present else []

    Inputs
    ------
    payload:
        Ground-truth or attributed JSON object.

    Outputs
    -------
    list[Mapping[str, object]]
        Segment dictionaries.

    Complexity
    ----------
    O(1) plus list materialization.

    Reference paper
    ---------------
    MeetEval's SegLST format represents meeting transcripts as JSON segment
    lists containing words/speaker/timing fields.

    Implementation notes
    --------------------
    Phase 20 stores ground truth and fusion output as objects with a
    ``segments`` array.  Missing arrays are treated as empty transcripts to
    support empty-input tests.
    """

    return list(payload.get("segments", []) or [])


def words_from_payload(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """
    Description
    -----------
    Extract word-level attributed output from a project JSON payload.

    Mathematical definition
    -----------------------
    Not a metric; returns the sequence over which speaker-attribution accuracy
    is evaluated.

    Equation
    --------
    words(payload) = payload["words"] if present else []

    Inputs
    ------
    payload:
        Fusion attributed JSON object.

    Outputs
    -------
    list[Mapping[str, object]]
        Word dictionaries.

    Complexity
    ----------
    O(1) plus list materialization.

    Reference paper
    ---------------
    CHiME-7 DASR and MeetEval scoring use timed words or segments to evaluate
    speaker-attributed recognition.

    Implementation notes
    --------------------
    Empty word arrays make speaker accuracy undefined, matching previous Phase
    20 behavior, while WER metrics can still be computed from segments.
    """

    return list(payload.get("words", []) or [])


def fusion_metrics(gt_json, attributed_json, duration) -> Dict[str, object]:
    """
    Description
    -----------
    Compute Phase 20 fusion evaluation metrics from reference and attributed
    JSON files without changing the public API.

    Mathematical definition
    -----------------------
    The returned metrics are: official cpWER; SA-WER after optimal speaker
    mapping; Speaker Accuracy from mapped word-level speaker pairs; Word
    Attribution Error as 1 - Speaker Accuracy; and frame-level Timeline
    Accuracy over mapped active-speaker sets.

    Equation
    --------
    {
      cpWER = E_cp / N_ref,
      SA-WER = sum_s lev(R_s, H'_s) / N_ref,
      SpeakerAccuracy = correct_words / evaluated_words,
      WordAttributionError = 1 - SpeakerAccuracy,
      TimelineAccuracy = (1/F) sum_k 1[T_ref[k] = T_hyp[k]]
    }

    Inputs
    ------
    gt_json:
        Path to ground-truth JSON with ``segments``.
    attributed_json:
        Path to fusion output JSON with ``segments`` and ``words``.
    duration:
        Audio duration in seconds.

    Outputs
    -------
    dict
        Public Phase 20 fusion schema:
        ``cpwer``, ``sawer``, ``speaker_accuracy``,
        ``word_attribution_error``, ``timeline_accuracy``,
        ``words_evaluated``, ``correct_attributions``, and
        ``speaker_mapping``.

    Complexity
    ----------
    Dominated by cpWER and transcript fallback mapping:
    O(S^2 R H + S^3 + W G + F), where S is speaker count, R/H are typical
    per-speaker transcript lengths, W is word count, G is reference segment
    count, and F is frame count.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394; MeetEval
    implementation notes on cpWER's Hungarian assignment; ESPnet and CHiME-7
    DASR diarization-attributed ASR scoring methodology.

    Implementation notes
    --------------------
    cpWER follows the literature exactly: concatenate by speaker, compute
    Levenshtein costs, insert empty streams for count mismatch, and solve the
    one-to-one assignment with Hungarian matching.  Speaker Accuracy is kept
    mathematically identical to the prior project implementation.  Removed
    legacy attribution fields are intentionally absent from computation and
    JSON.
    """

    gt_payload = read_json(Path(gt_json))
    attributed_payload = read_json(Path(attributed_json))

    gt_segments = segments_from_payload(gt_payload)
    sys_segments = segments_from_payload(attributed_payload)
    words = words_from_payload(attributed_payload)

    pairs = word_speaker_pairs(gt_segments, words)
    attribution_mapping = speaker_accuracy_stats(pairs)["speaker_mapping"]

    transcript_mapping, _ = transcript_speaker_mapping(
        concatenate_by_speaker(gt_segments),
        concatenate_by_speaker(sys_segments),
    )
    speaker_mapping = merge_mappings(attribution_mapping, transcript_mapping)

    attribution = speaker_accuracy_stats(pairs, speaker_mapping)
    return {
        "cpwer": cpwer(gt_segments, sys_segments),
        "sawer": sawer(gt_segments, sys_segments, speaker_mapping),
        "speaker_accuracy": attribution["speaker_accuracy"],
        "word_attribution_error": attribution["word_attribution_error"],
        "timeline_accuracy": timeline_accuracy(gt_segments, sys_segments, float(duration), speaker_mapping=speaker_mapping),
        "words_evaluated": attribution["words_evaluated"],
        "correct_attributions": attribution["correct_attributions"],
        "speaker_mapping": attribution["speaker_mapping"],
    }
