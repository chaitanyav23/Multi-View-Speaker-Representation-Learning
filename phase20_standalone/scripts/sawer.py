"""
Speaker Attributed WER for Phase 20 fusion evaluation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

from speaker_mapping import concatenate_by_speaker, edit_distance_counts, safe_rate


def sawer_stats(
    reference_segments: Iterable[Mapping[str, object]],
    hypothesis_segments: Iterable[Mapping[str, object]],
    speaker_mapping: Mapping[str, str],
) -> Dict[str, object]:
    """
    Description
    -----------
    Compute Speaker Attributed Word Error Rate after renaming hypothesis
    speakers with an optimal one-to-one mapping.

    Mathematical definition
    -----------------------
    Let m be the selected hypothesis-to-reference speaker mapping.  Rename
    every hypothesis stream h to m(h), concatenate by speaker, compute standard
    WER edit counts independently per reference speaker, aggregate counts, and
    normalize by total reference words.

    Equation
    --------
    SA-WER = sum_s lev(R_s, H'_s) / sum_s |R_s|, where H'_{m(h)} = H_h

    Inputs
    ------
    reference_segments:
        Reference speaker-attributed segments.
    hypothesis_segments:
        Hypothesis speaker-attributed segments.
    speaker_mapping:
        Hypothesis speaker ID to reference speaker ID.

    Outputs
    -------
    dict
        ``sawer``, edit counts, reference length, and hypothesis length.

    Complexity
    ----------
    O(T + sum_s |R_s||H'_s|), where T is total segment-token work.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Eq. (1)-(3);
    CHiME-7 DASR scoring uses speaker-attributed WER variants to evaluate
    recognition and attribution jointly.

    Implementation notes
    --------------------
    The mapping step mirrors the permutation-invariant speaker assignment used
    for meeting WER.  Unlike a MeetEval wrapper, this function performs local
    renaming and local Levenshtein accumulation, so the standalone repository
    has no runtime dependency on MeetEval.
    """

    reference = concatenate_by_speaker(reference_segments)
    raw_hypothesis = concatenate_by_speaker(hypothesis_segments)
    hypothesis: Dict[str, list[str]] = {}
    for hyp_speaker, words in raw_hypothesis.items():
        mapped = speaker_mapping.get(hyp_speaker, hyp_speaker)
        hypothesis.setdefault(mapped, [])
        hypothesis[mapped].extend(words)

    speakers = sorted(set(reference) | set(hypothesis))
    totals = {"substitutions": 0, "insertions": 0, "deletions": 0, "correct": 0, "errors": 0}
    total_ref_words = sum(len(words) for words in reference.values())
    total_hyp_words = sum(len(words) for words in hypothesis.values())

    for speaker in speakers:
        counts = edit_distance_counts(reference.get(speaker, []), hypothesis.get(speaker, []))
        for key in totals:
            totals[key] += counts[key]

    rate = safe_rate(totals["errors"], total_ref_words)
    return {
        "sawer": 0.0 if rate is None else rate,
        "errors": totals["errors"],
        "substitutions": totals["substitutions"],
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "reference_words": total_ref_words,
        "hypothesis_words": total_hyp_words,
    }


def sawer(
    reference_segments: Iterable[Mapping[str, object]],
    hypothesis_segments: Iterable[Mapping[str, object]],
    speaker_mapping: Mapping[str, str],
) -> float:
    """
    Description
    -----------
    Return only the Speaker Attributed WER scalar.

    Mathematical definition
    -----------------------
    Aggregate per-speaker edit errors after speaker renaming and divide by the
    number of reference words.

    Equation
    --------
    SA-WER = (S + I + D) / N_ref

    Inputs
    ------
    reference_segments:
        Reference segments.
    hypothesis_segments:
        Hypothesis segments.
    speaker_mapping:
        Hypothesis-to-reference speaker mapping.

    Outputs
    -------
    float
        SA-WER.  When there are no reference words, returns 0.0.

    Complexity
    ----------
    O(T + sum_s |R_s||H'_s|).

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394; CHiME-7 DASR
    evaluation methodology for diarization-attributed ASR.

    Implementation notes
    --------------------
    This scalar wrapper keeps the public call site compact while ``sawer_stats``
    exposes count diagnostics for tests and release checks.
    """

    return float(sawer_stats(reference_segments, hypothesis_segments, speaker_mapping)["sawer"])
