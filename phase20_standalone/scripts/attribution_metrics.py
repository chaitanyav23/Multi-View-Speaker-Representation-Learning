"""
Word-level speaker attribution metrics for Phase 20 fusion evaluation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

from speaker_mapping import optimal_speaker_mapping


def reference_speaker_at_time(reference_segments: Iterable[Mapping[str, object]], start, end) -> str:
    """
    Description
    -----------
    Select the reference speaker associated with a word time span.

    Mathematical definition
    -----------------------
    For a word interval [a,b], choose the reference segment speaker with
    maximum positive temporal overlap.  If no segment overlaps, choose the
    segment containing the midpoint.  Otherwise return ``UNK``.

    Equation
    --------
    speaker(w) = speaker(argmax_s max(0, min(b,e_s) - max(a,t_s)))

    Inputs
    ------
    reference_segments:
        Reference segments with ``start``, ``end``, and ``speaker`` fields.
    start:
        Word start time in seconds.
    end:
        Word end time in seconds.

    Outputs
    -------
    str
        Reference speaker ID or ``UNK``.

    Complexity
    ----------
    O(S), where S is the number of reference segments.

    Reference paper
    ---------------
    CHiME-7 DASR evaluation and MeetEval assume timed speaker-labelled
    references for diarization-attributed scoring.

    Implementation notes
    --------------------
    This preserves the existing Phase 20 mathematical definition used by
    ``common.py`` before the refactor.  Overlap is the primary criterion; the
    midpoint fallback handles zero-duration reference annotations already
    present in the project data.
    """

    if start is None or end is None:
        return "UNK"
    start_f = float(start)
    end_f = float(end)
    midpoint = (start_f + end_f) / 2.0
    best = "UNK"
    best_overlap = 0.0
    for segment in reference_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        overlap = min(end_f, seg_end) - max(start_f, seg_start)
        if overlap > best_overlap or (best_overlap == 0 and seg_start <= midpoint <= seg_end):
            best_overlap = overlap
            best = str(segment.get("speaker", "UNK") or "UNK")
    return best


def word_speaker_pairs(reference_segments: Iterable[Mapping[str, object]], words: Iterable[Mapping[str, object]]) -> List[Tuple[str, str]]:
    """
    Description
    -----------
    Build evaluated hypothesis/reference speaker pairs for word attribution.

    Mathematical definition
    -----------------------
    For each timed hypothesis word with non-UNK speaker h, assign a reference
    speaker r by temporal overlap and keep the pair (h,r) when r is not UNK.

    Equation
    --------
    P = {(h_w, r_w): h_w != UNK and r_w = speaker_ref(time(w)) != UNK}

    Inputs
    ------
    reference_segments:
        Timed reference segments.
    words:
        Hypothesis word dictionaries with ``speaker``, ``start``, and ``end``.

    Outputs
    -------
    list[tuple[str, str]]
        Evaluated ``(hypothesis_speaker, reference_speaker)`` pairs.

    Complexity
    ----------
    O(W S), where W is the number of words and S is reference segment count.

    Reference paper
    ---------------
    CHiME-7 DASR and ESPnet diarization-attributed ASR scoring rely on timed
    word/segment attribution to compare speaker-labelled hypotheses against
    speaker-labelled references.

    Implementation notes
    --------------------
    The filtering and pairing rules are intentionally unchanged from the
    previous Phase 20 implementation so speaker accuracy remains comparable.
    """

    reference_list = list(reference_segments)
    pairs = []
    for word in words:
        hyp = str(word.get("speaker", "UNK") or "UNK")
        if hyp == "UNK":
            continue
        ref = reference_speaker_at_time(reference_list, word.get("start"), word.get("end"))
        if ref == "UNK":
            continue
        pairs.append((hyp, ref))
    return pairs


def speaker_accuracy_stats(pairs: Iterable[Tuple[str, str]], speaker_mapping: Mapping[str, str] | None = None) -> Dict[str, object]:
    """
    Description
    -----------
    Compute Phase 20 word-level speaker attribution accuracy.

    Mathematical definition
    -----------------------
    Given evaluated pairs (h,r), find or use a one-to-one mapping m from
    hypothesis to reference speakers and count a word as correct when m(h)=r.

    Equation
    --------
    SpeakerAccuracy = |{(h,r) in P: m(h)=r}| / |P|

    Inputs
    ------
    pairs:
        Evaluated ``(hypothesis_speaker, reference_speaker)`` pairs.
    speaker_mapping:
        Optional precomputed mapping.  If omitted, the optimal mapping
        maximizing correct attributions is computed.

    Outputs
    -------
    dict
        ``speaker_accuracy``, ``word_attribution_error``, ``words_evaluated``,
        ``correct_attributions``, and ``speaker_mapping``.

    Complexity
    ----------
    O(P + S^3), where P is evaluated words and S is speaker count.

    Reference paper
    ---------------
    The one-to-one assignment follows the permutation-invariant speaker
    matching principle described for meeting WER in MeetEval, von Neumann et
    al., CHiME 2023 / arXiv:2307.11394.

    Implementation notes
    --------------------
    This keeps the original Phase 20 speaker-accuracy definition exactly.  Word
    Attribution Error is intentionally defined as 1 - Speaker Accuracy.
    """

    pair_list = list(pairs)
    mapping = dict(speaker_mapping) if speaker_mapping is not None else optimal_speaker_mapping(pair_list)
    total = len(pair_list)
    correct = sum(1 for hyp, ref in pair_list if mapping.get(hyp) == ref)
    speaker_accuracy = correct / total if total else None
    word_attribution_error = 1.0 - speaker_accuracy if speaker_accuracy is not None else None
    return {
        "speaker_accuracy": speaker_accuracy,
        "word_attribution_error": word_attribution_error,
        "words_evaluated": total,
        "correct_attributions": correct,
        "speaker_mapping": dict(sorted(mapping.items())),
    }


def word_attribution_error(speaker_accuracy: float | None) -> float | None:
    """
    Description
    -----------
    Compute Word Attribution Error from speaker attribution accuracy.

    Mathematical definition
    -----------------------
    Word Attribution Error is the complement of Speaker Accuracy.

    Equation
    --------
    WAE = 1 - SpeakerAccuracy

    Inputs
    ------
    speaker_accuracy:
        Speaker attribution accuracy or None when undefined.

    Outputs
    -------
    float | None
        Complement error or None when speaker accuracy is undefined.

    Complexity
    ----------
    O(1).

    Reference paper
    ---------------
    CHiME-7 DASR style speaker-attributed scoring.  This project explicitly
    defines Word Attribution Error as this complement.

    Implementation notes
    --------------------
    Only the project-defined complement metric is returned in the output schema.
    """

    return 1.0 - speaker_accuracy if speaker_accuracy is not None else None
