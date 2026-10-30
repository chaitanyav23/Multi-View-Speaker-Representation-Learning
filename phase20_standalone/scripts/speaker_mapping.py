"""
Speaker assignment utilities for Phase 20 fusion evaluation.

The functions in this module are self-contained and intentionally do not
import MeetEval.  They implement the same assignment principle used for
cpWER-style meeting evaluation: build a rectangular speaker-pair objective,
pad missing streams implicitly, and solve the one-to-one assignment with the
Hungarian algorithm.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


TokenSeq = Sequence[str]


def linear_sum_assignment(cost_matrix: Sequence[Sequence[float]]) -> Tuple[List[int], List[int]]:
    """
    Description
    -----------
    Solve a square linear-sum assignment problem with the Hungarian algorithm.

    Mathematical definition
    -----------------------
    Given a cost matrix C, find a one-to-one assignment pi minimizing the sum
    of selected costs.

    Equation
    --------
    pi* = argmin_pi sum_i C[i, pi(i)]

    Inputs
    ------
    cost_matrix:
        Square matrix of finite numeric costs.

    Outputs
    -------
    tuple[list[int], list[int]]
        Row indices and assigned column indices.

    Complexity
    ----------
    O(S^3), where S is the matrix dimension.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Section 3.2,
    notes cpWER is solved with Hungarian assignment rather than factorial
    permutation enumeration.

    Implementation notes
    --------------------
    This is a local implementation of the shortest augmenting path form of
    the Hungarian algorithm for minimization.  It avoids a SciPy runtime
    dependency while preserving the assignment method specified by the
    literature.
    """

    n = len(cost_matrix)
    if n == 0:
        return [], []
    if any(len(row) != n for row in cost_matrix):
        raise ValueError("cost_matrix must be square")

    cost = [[float(value) for value in row] for row in cost_matrix]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return list(range(n)), assignment


def normalize_text(text: str) -> str:
    """
    Description
    -----------
    Normalize transcript text before word-level edit-distance scoring.

    Mathematical definition
    -----------------------
    Let x be a text string.  The normalization function N(x) lowercases x,
    removes punctuation, collapses whitespace, and strips boundary spaces.

    Equation
    --------
    N(x) = strip(collapse_space(remove_punctuation(lower(x))))

    Inputs
    ------
    text:
        Transcript string.  None-like inputs are treated as empty strings by
        callers before tokenization.

    Outputs
    -------
    str
        Normalized text.

    Complexity
    ----------
    O(|x|), where |x| is the number of input characters.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Section 3.

    Implementation notes
    --------------------
    The literature defines WER over word sequences and leaves text
    normalization to the scoring setup.  This project previously normalized
    case, punctuation, and whitespace before WER; this helper preserves that
    behavior without adding any dependency on MeetEval or external projects.
    """

    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def words_from_text(text: str) -> List[str]:
    """
    Description
    -----------
    Convert transcript text into the normalized word sequence used by all
    fusion WER metrics.

    Mathematical definition
    -----------------------
    For normalized text N(x), return the ordered token sequence split on a
    single space.

    Equation
    --------
    W(x) = split_space(N(x))

    Inputs
    ------
    text:
        Transcript string.

    Outputs
    -------
    list[str]
        Normalized tokens.  Empty text returns an empty list.

    Complexity
    ----------
    O(|x|).

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Eq. (1)-(3).

    Implementation notes
    --------------------
    cpWER and SA-WER are edit-distance metrics over words.  Keeping
    tokenization centralized guarantees that cpWER, SA-WER, and mapping costs
    are scored on the same sequence representation.
    """

    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def concatenate_by_speaker(segments: Iterable[Mapping[str, object]]) -> Dict[str, List[str]]:
    """
    Description
    -----------
    Concatenate segment transcripts independently for every speaker.

    Mathematical definition
    -----------------------
    For speaker s with ordered segments U_s = (u_1, ..., u_n), construct
    C_s = W(u_1) o ... o W(u_n), where o denotes sequence concatenation.

    Equation
    --------
    C_s = concat_{u in U_s} W(text(u))

    Inputs
    ------
    segments:
        Iterable of dictionaries with at least optional ``speaker`` and
        ``text`` fields.

    Outputs
    -------
    dict[str, list[str]]
        Speaker ID to concatenated word sequence.  Empty speaker transcripts
        are retained.

    Complexity
    ----------
    O(T), where T is the total number of transcript tokens.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Section 3.2.

    Implementation notes
    --------------------
    This directly follows cpWER's diarization-style input assumption: group by
    speaker, then concatenate all utterances for that speaker before computing
    pairwise WER costs.  The function does not inspect paths or global project
    state, keeping the repository standalone.
    """

    grouped: Dict[str, List[str]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker", "UNK") or "UNK")
        grouped.setdefault(speaker, [])
        grouped[speaker].extend(words_from_text(str(segment.get("text", "") or "")))
    return grouped


def edit_distance_counts(reference: TokenSeq, hypothesis: TokenSeq) -> Dict[str, int]:
    """
    Description
    -----------
    Compute Levenshtein edit counts between two word sequences.

    Mathematical definition
    -----------------------
    The Levenshtein distance is the minimum number of substitutions,
    insertions, and deletions required to transform the hypothesis sequence
    into the reference sequence, with correct matches carrying zero cost.

    Equation
    --------
    L[i,j] = min(
        L[i-1,j-1] + 0 if r_i = h_j else L[i-1,j-1] + 1,
        L[i,j-1] + 1,
        L[i-1,j] + 1
    )

    Inputs
    ------
    reference:
        Reference word sequence.
    hypothesis:
        Hypothesis word sequence.

    Outputs
    -------
    dict[str, int]
        Counts for ``substitutions``, ``insertions``, ``deletions``,
        ``correct``, and total ``errors``.

    Complexity
    ----------
    O(RH) time and O(RH) memory, where R and H are reference and hypothesis
    sequence lengths.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Eq. (1)-(2).

    Implementation notes
    --------------------
    This is a local Wagner-Fischer dynamic program.  Tie-breaking prefers
    correct/substitution, then deletion, then insertion, making outputs
    deterministic while leaving the total Levenshtein distance unchanged.
    """

    r_len = len(reference)
    h_len = len(hypothesis)
    costs = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    back = [[""] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(1, r_len + 1):
        costs[i][0] = i
        back[i][0] = "D"
    for j in range(1, h_len + 1):
        costs[0][j] = j
        back[0][j] = "I"

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                candidates = [(costs[i - 1][j - 1], "C")]
            else:
                candidates = [(costs[i - 1][j - 1] + 1, "S")]
            candidates.append((costs[i - 1][j] + 1, "D"))
            candidates.append((costs[i][j - 1] + 1, "I"))
            best_cost, best_op = min(candidates, key=lambda item: (item[0], "CSDI".index(item[1])))
            costs[i][j] = best_cost
            back[i][j] = best_op

    counts = Counter()
    i, j = r_len, h_len
    while i > 0 or j > 0:
        op = back[i][j]
        counts[op] += 1
        if op in {"C", "S"}:
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            j -= 1
        else:
            raise RuntimeError(f"Invalid edit backtrace op {op!r}")

    substitutions = counts["S"]
    insertions = counts["I"]
    deletions = counts["D"]
    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "correct": counts["C"],
        "errors": substitutions + insertions + deletions,
    }


def optimal_speaker_mapping(pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """
    Description
    -----------
    Compute the optimal one-to-one mapping from hypothesis speakers to
    reference speakers from word-level attribution pairs.

    Mathematical definition
    -----------------------
    Let n(h, r) be the number of evaluated words attributed to hypothesis
    speaker h whose reference speaker is r.  Find a one-to-one mapping m that
    maximizes sum_h n(h, m(h)).

    Equation
    --------
    m* = argmax_m sum_{h in H} n(h, m(h))

    Inputs
    ------
    pairs:
        Iterable of ``(hypothesis_speaker, reference_speaker)`` observations.
        ``UNK`` speakers are ignored.

    Outputs
    -------
    dict[str, str]
        Hypothesis speaker ID to reference speaker ID.

    Complexity
    ----------
    O(P + S^3), where P is the number of pairs and S is
    max(number of hypothesis speakers, number of reference speakers).

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Section 3.
    ESPnet and CHiME-7 DASR recipes use equivalent permutation-invariant
    speaker matching for diarization-attributed scoring.

    Implementation notes
    --------------------
    Hungarian assignment minimizes cost, so the count matrix is negated.  This
    preserves the existing Phase 20 speaker-accuracy definition while making
    the one-to-one constraint explicit and deterministic for arbitrary speaker
    counts.
    """

    clean_pairs = [(str(h), str(r)) for h, r in pairs if h != "UNK" and r != "UNK"]
    hyp_speakers = sorted({h for h, _ in clean_pairs})
    ref_speakers = sorted({r for _, r in clean_pairs})
    if not hyp_speakers or not ref_speakers:
        return {}

    size = max(len(hyp_speakers), len(ref_speakers))
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    hyp_index = {speaker: i for i, speaker in enumerate(hyp_speakers)}
    ref_index = {speaker: j for j, speaker in enumerate(ref_speakers)}

    for hyp, ref in clean_pairs:
        matrix[hyp_index[hyp]][ref_index[ref]] -= 1

    rows, cols = linear_sum_assignment(matrix)
    mapping: Dict[str, str] = {}
    for row, col in zip(rows, cols):
        if row < len(hyp_speakers) and col < len(ref_speakers):
            mapping[hyp_speakers[row]] = ref_speakers[col]
    return mapping


def transcript_speaker_mapping(
    reference_by_speaker: Mapping[str, TokenSeq],
    hypothesis_by_speaker: Mapping[str, TokenSeq],
) -> Tuple[Dict[str, str], Tuple[Tuple[str | None, str | None], ...]]:
    """
    Description
    -----------
    Compute a cpWER-style one-to-one speaker mapping from concatenated
    transcripts.

    Mathematical definition
    -----------------------
    Let c(r, h) = lev(R_r, H_h).  Missing reference or hypothesis streams are
    represented by empty sequences.  Find the assignment with minimum total
    cost.

    Equation
    --------
    a* = argmin_a sum_{(r,h) in a} lev(R_r, H_h)

    Inputs
    ------
    reference_by_speaker:
        Reference speaker ID to word sequence.
    hypothesis_by_speaker:
        Hypothesis speaker ID to word sequence.

    Outputs
    -------
    tuple
        ``(hypothesis_to_reference_mapping, assignment)``.  Assignment entries
        are ``(reference_speaker_or_None, hypothesis_speaker_or_None)``.

    Complexity
    ----------
    O(S^2 R H + S^3), where S is the padded speaker count and R/H are typical
    per-speaker word counts.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Eq. (3).

    Implementation notes
    --------------------
    This is the exact cpWER assignment objective described in the literature.
    It inserts empty streams for over- and under-estimated speakers and uses
    the Hungarian algorithm rather than enumerating all permutations.
    """

    ref_keys = sorted(reference_by_speaker)
    hyp_keys = sorted(hypothesis_by_speaker)
    size = max(len(ref_keys), len(hyp_keys))
    if size == 0:
        return {}, ()

    cost = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        ref_words = reference_by_speaker.get(ref_keys[i], []) if i < len(ref_keys) else []
        for j in range(size):
            hyp_words = hypothesis_by_speaker.get(hyp_keys[j], []) if j < len(hyp_keys) else []
            cost[i][j] = edit_distance_counts(ref_words, hyp_words)["errors"]

    rows, cols = linear_sum_assignment(cost)
    assignment = []
    mapping: Dict[str, str] = {}
    for row, col in zip(rows, cols):
        ref = ref_keys[row] if row < len(ref_keys) else None
        hyp = hyp_keys[col] if col < len(hyp_keys) else None
        assignment.append((ref, hyp))
        if ref is not None and hyp is not None:
            mapping[hyp] = ref

    assignment.sort(key=lambda item: ("" if item[0] is None else item[0], "" if item[1] is None else item[1]))
    return mapping, tuple(assignment)


def merge_mappings(primary: Mapping[str, str], fallback: Mapping[str, str]) -> Dict[str, str]:
    """
    Description
    -----------
    Merge a preferred speaker mapping with a fallback transcript mapping.

    Mathematical definition
    -----------------------
    For each hypothesis speaker h, return primary(h) when defined, otherwise
    fallback(h).

    Equation
    --------
    m(h) = m_primary(h) if h in domain(m_primary), else m_fallback(h)

    Inputs
    ------
    primary:
        Preferred mapping.
    fallback:
        Fallback mapping.

    Outputs
    -------
    dict[str, str]
        Combined deterministic mapping.

    Complexity
    ----------
    O(S), where S is the number of mapped hypothesis speakers.

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394.

    Implementation notes
    --------------------
    Speaker accuracy in Phase 20 is defined on word-attribution pairs, while
    cpWER has a transcript-distance assignment.  This helper keeps the public
    mapping compatible with the existing attribution metric and still supports
    empty/no-timing cases through transcript fallback.
    """

    merged = dict(fallback)
    merged.update(primary)
    return dict(sorted(merged.items()))


def safe_rate(numerator: float, denominator: float) -> float | None:
    """
    Description
    -----------
    Safely divide metric counts.

    Mathematical definition
    -----------------------
    Return numerator / denominator when denominator is non-zero, otherwise
    undefined.

    Equation
    --------
    rate = n / d if d > 0 else None

    Inputs
    ------
    numerator:
        Count or score numerator.
    denominator:
        Count denominator.

    Outputs
    -------
    float | None
        Ratio or None when undefined.

    Complexity
    ----------
    O(1).

    Reference paper
    ---------------
    MeetEval, von Neumann et al., CHiME 2023 / arXiv:2307.11394, Eq. (1).

    Implementation notes
    --------------------
    WER-style denominators are total reference words; attribution accuracy
    denominators are evaluated words.  Returning None for undefined accuracy
    preserves Phase 20's prior speaker-accuracy behavior.
    """

    return numerator / denominator if denominator else None
