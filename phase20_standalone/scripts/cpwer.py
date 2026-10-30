"""
Concatenated minimum-permutation WER for Phase 20 fusion evaluation.

This module implements cpWER locally.  It follows the MeetEval paper's
definition but does not import or call MeetEval at runtime.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

from speaker_mapping import concatenate_by_speaker, edit_distance_counts, linear_sum_assignment, safe_rate


def cpwer_stats(reference_segments: Iterable[Mapping[str, object]], hypothesis_segments: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """
    Description
    -----------
    Compute official Concatenated minimum-Permutation Word Error Rate
    statistics for diarization-style speaker-attributed transcripts.

    Mathematical definition
    -----------------------
    For every reference speaker i and hypothesis speaker j, concatenate all
    words for that speaker and compute c(i,j) = lev(R_i, H_j).  Pad the
    smaller speaker set with empty streams, solve the one-to-one assignment
    with minimum total cost, and divide the selected edit distance by the
    total number of reference words.

    Equation
    --------
    cpWER = min_pi sum_i lev(concat_u R_{pi(i),u}, concat_v H_{i,v})
            / sum_i |concat_u R_{i,u}|

    Inputs
    ------
    reference_segments:
        Reference segments containing ``speaker`` and ``text`` fields.
    hypothesis_segments:
        Hypothesis segments containing ``speaker`` and ``text`` fields.

    Outputs
    -------
    dict
        ``cpwer``, edit counts, reference length, hypothesis length, and the
        chosen ``assignment`` as ``(reference_speaker, hypothesis_speaker)``
        pairs.  Missing speakers are represented by None.

    Complexity
    ----------
    O(S^2 R H + S^3), where S is the padded speaker count and R/H are typical
    per-speaker word counts.  The S^3 term is Hungarian assignment.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., "MeetEval: A Toolkit for Computation of Word
    Error Rates for Meeting Transcription Systems", CHiME 2023, Section 3.2,
    Eq. (3).  The same cpWER convention is used by CHiME-7 DASR and ESPnet
    meeting-recognition recipes.

    Implementation notes
    --------------------
    The paper states that empty streams are inserted for over- or
    under-estimated speakers and that MeetEval uses the Hungarian algorithm to
    avoid factorial permutation search.  This implementation does exactly that
    with local Levenshtein counts and SciPy's Hungarian solver, with no
    MeetEval runtime dependency.
    """

    reference = concatenate_by_speaker(reference_segments)
    hypothesis = concatenate_by_speaker(hypothesis_segments)
    ref_keys = sorted(reference)
    hyp_keys = sorted(hypothesis)
    size = max(len(ref_keys), len(hyp_keys))
    total_ref_words = sum(len(words) for words in reference.values())
    total_hyp_words = sum(len(words) for words in hypothesis.values())

    if size == 0:
        return {
            "cpwer": 0.0,
            "errors": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "reference_words": 0,
            "hypothesis_words": 0,
            "assignment": (),
        }

    cost = [[0 for _ in range(size)] for _ in range(size)]
    pair_counts = {}
    for i in range(size):
        ref_speaker = ref_keys[i] if i < len(ref_keys) else None
        ref_words = reference.get(ref_speaker, []) if ref_speaker is not None else []
        for j in range(size):
            hyp_speaker = hyp_keys[j] if j < len(hyp_keys) else None
            hyp_words = hypothesis.get(hyp_speaker, []) if hyp_speaker is not None else []
            counts = edit_distance_counts(ref_words, hyp_words)
            cost[i][j] = counts["errors"]
            pair_counts[(i, j)] = counts

    rows, cols = linear_sum_assignment(cost)
    totals = {"substitutions": 0, "insertions": 0, "deletions": 0, "correct": 0, "errors": 0}
    assignment = []
    for row, col in zip(rows, cols):
        counts = pair_counts[(row, col)]
        for key in totals:
            totals[key] += counts[key]
        ref_speaker = ref_keys[row] if row < len(ref_keys) else None
        hyp_speaker = hyp_keys[col] if col < len(hyp_keys) else None
        assignment.append((ref_speaker, hyp_speaker))

    assignment.sort(key=lambda item: ("" if item[0] is None else item[0], "" if item[1] is None else item[1]))
    rate = safe_rate(totals["errors"], total_ref_words)
    return {
        "cpwer": 0.0 if rate is None else rate,
        "errors": totals["errors"],
        "substitutions": totals["substitutions"],
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "reference_words": total_ref_words,
        "hypothesis_words": total_hyp_words,
        "assignment": tuple(assignment),
    }


def cpwer(reference_segments: Iterable[Mapping[str, object]], hypothesis_segments: Iterable[Mapping[str, object]]) -> float:
    """
    Description
    -----------
    Return only the cpWER scalar for the supplied speaker-attributed
    transcripts.

    Mathematical definition
    -----------------------
    This is the selected edit distance from the minimum-permutation assignment
    divided by total reference words.

    Equation
    --------
    cpWER = E_cp / N_ref

    Inputs
    ------
    reference_segments:
        Reference speaker-attributed segments.
    hypothesis_segments:
        Hypothesis speaker-attributed segments.

    Outputs
    -------
    float
        cpWER.  When there are no reference words, returns 0.0.

    Complexity
    ----------
    O(S^2 R H + S^3).

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Section 3.2.

    Implementation notes
    --------------------
    Thin public scalar wrapper around ``cpwer_stats`` for callers that do not
    need edit-count diagnostics.
    """

    return float(cpwer_stats(reference_segments, hypothesis_segments)["cpwer"])
