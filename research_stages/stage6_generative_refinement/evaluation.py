"""
Stage 6 — Section 5: Evaluation & Comparison Framework
Evaluates original vs. LLM-refined diarization outputs on:
1. Word Diarization Error Rate (WDER) — both timestamp-aligned (Phase 20 WAE parity
   and strict UNK-inclusive) and lexical-aligned (Shafey et al., 2019 / DiarizationLM).
2. Concatenated minimum-Permutation Word Error Rate (cpWER) — Phase 20 parity.
3. Proposal, acceptance, rejection, and abstention accounting.
4. Cryptographic and structural integrity of original ASR words and timestamps.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schemas import OrchestratedWord, ReconciliationReason, RefinedWordOutput

# Ensure Phase 20 standalone scripts are importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE20_SCRIPTS = _REPO_ROOT / "phase20_standalone" / "scripts"
if str(_PHASE20_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PHASE20_SCRIPTS))

from attribution_metrics import (
    reference_speaker_at_time,
    speaker_accuracy_stats,
    word_speaker_pairs,
)
from common import group_words as phase20_group_words
from cpwer import cpwer_stats as phase20_cpwer_stats
from speaker_mapping import linear_sum_assignment, words_from_text


def _hash_word_sequence(words: Sequence[str]) -> str:
    canonical = json.dumps(list(words), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_timestamp_sequence(timestamps: Sequence[Tuple[Optional[float], Optional[float]]]) -> str:
    canonical = json.dumps(
        [[round(s, 6) if s is not None else None, round(e, 6) if e is not None else None] for s, e in timestamps]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_timestamp_wder(
    word_dicts: Sequence[Mapping[str, Any]],
    reference_segments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Computes Timestamp-Aligned Word Diarization Error Rate (WDER_time).

    Mathematical Definition:
    ------------------------
    For each ASR word w_i with time interval [start_i, end_i], let r_i be the
    reference speaker with maximum positive temporal overlap in `reference_segments`
    (or midpoint containment fallback). Let h_i be the hypothesis speaker assigned
    to w_i.

    1. `wder_assigned` (Phase 20 Word Attribution Error parity):
       Evaluated over words where both h_i != 'UNK' and r_i != 'UNK'.
       Finds optimal 1-to-1 speaker mapping pi* via Hungarian assignment maximizing
       sum_i I(pi(h_i) == r_i), then:
           WDER_assigned = 1 - (correct_attributions / evaluated_assigned_words)

    2. `wder_all_ref` (Strict Timestamp WDER including UNK):
       Evaluated over all words falling in reference speech (r_i != 'UNK').
       Any word with h_i == 'UNK' during reference speech is counted as an error:
           WDER_all_ref = 1 - (correct_attributions / total_ref_timed_words)
    """
    pairs_assigned = word_speaker_pairs(reference_segments, word_dicts)
    acc_stats = speaker_accuracy_stats(pairs_assigned)

    mapping: Dict[str, str] = dict(acc_stats["speaker_mapping"])
    correct_assigned = int(acc_stats["correct_attributions"])
    eval_assigned = int(acc_stats["words_evaluated"])
    wder_assigned = (
        round(float(acc_stats["word_attribution_error"]), 4)
        if acc_stats["word_attribution_error"] is not None
        else 0.0
    )

    # Also count words that overlapped a valid reference speaker but had h_i == "UNK"
    total_ref_timed = 0
    unassigned_in_ref = 0
    for w in word_dicts:
        ref_spk = reference_speaker_at_time(reference_segments, w.get("start"), w.get("end"))
        if ref_spk != "UNK":
            total_ref_timed += 1
            hyp_spk = str(w.get("speaker", "UNK") or "UNK")
            if hyp_spk == "UNK":
                unassigned_in_ref += 1

    wder_all_ref = (
        round(1.0 - (correct_assigned / total_ref_timed), 4)
        if total_ref_timed > 0
        else 0.0
    )

    return {
        "wder_assigned": wder_assigned,
        "speaker_accuracy_assigned": (
            round(float(acc_stats["speaker_accuracy"]), 4)
            if acc_stats["speaker_accuracy"] is not None
            else 0.0
        ),
        "wder_all_ref": wder_all_ref,
        "words_evaluated_assigned": eval_assigned,
        "words_evaluated_all_ref": total_ref_timed,
        "unassigned_words_in_ref_speech": unassigned_in_ref,
        "correct_attributions": correct_assigned,
        "incorrect_attributions_assigned": eval_assigned - correct_assigned,
        "speaker_mapping": mapping,
    }


def compute_lexical_wder(
    word_dicts: Sequence[Mapping[str, Any]],
    reference_segments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Computes Lexical-Aligned Word Diarization Error Rate (WDER_lexical) following
    Shafey et al. (2019) and Wang et al. (2024, DiarizationLM).

    Mathematical Definition:
    ------------------------
    1. Flatten reference segments into an ordered sequence of normalized tokens with
       their ground-truth speaker labels: R = [(r_token_1, r_spk_1), ..., (r_token_N, r_spk_N)].
    2. Flatten hypothesis words into an ordered sequence of normalized tokens with
       their hypothesis speaker labels: H = [(h_token_1, h_spk_1), ..., (h_token_M, h_spk_M)].
    3. Compute standard Levenshtein word-level edit alignment between [t for t, _ in R]
       and [t for t, _ in H].
    4. Collect all aligned word pairs A = {(i, j)} corresponding to lexical Correct (C)
       and lexical Substitution (S) operations (|A| = C + S).
    5. Find optimal 1-to-1 speaker mapping pi* over {(h_spk_j, r_spk_i) : (i, j) in A}
       using Hungarian assignment, and compute:
           WDER_lexical = (C_spk_err + S_spk_err) / (C + S)
    """
    ref_tokens_with_spk: List[Tuple[str, str]] = []
    for seg in reference_segments:
        spk = str(seg.get("speaker", "UNK") or "UNK")
        for tok in words_from_text(str(seg.get("text", "") or "")):
            if tok:
                ref_tokens_with_spk.append((tok, spk))

    hyp_tokens_with_spk: List[Tuple[str, str]] = []
    for w in word_dicts:
        spk = str(w.get("speaker", "UNK") or "UNK")
        for tok in words_from_text(str(w.get("word", "") or "")):
            if tok:
                hyp_tokens_with_spk.append((tok, spk))

    r_len = len(ref_tokens_with_spk)
    h_len = len(hyp_tokens_with_spk)
    if r_len == 0 or h_len == 0:
        return {
            "wder_lexical": 0.0,
            "aligned_words_c_plus_s": 0,
            "lexical_correct_c": 0,
            "lexical_substitutions_s": 0,
            "speaker_errors_on_c": 0,
            "speaker_errors_on_s": 0,
            "total_speaker_errors": 0,
            "speaker_mapping": {},
        }

    # Dynamic programming Levenshtein alignment
    costs = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    back = [[""] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(1, r_len + 1):
        costs[i][0] = i
        back[i][0] = "D"
    for j in range(1, h_len + 1):
        costs[0][j] = j
        back[0][j] = "I"

    for i in range(1, r_len + 1):
        r_tok = ref_tokens_with_spk[i - 1][0]
        for j in range(1, h_len + 1):
            h_tok = hyp_tokens_with_spk[j - 1][0]
            if r_tok == h_tok:
                candidates = [(costs[i - 1][j - 1], "C")]
            else:
                candidates = [(costs[i - 1][j - 1] + 1, "S")]
            candidates.append((costs[i - 1][j] + 1, "D"))
            candidates.append((costs[i][j - 1] + 1, "I"))
            best_cost, best_op = min(candidates, key=lambda item: (item[0], "CSDI".index(item[1])))
            costs[i][j] = best_cost
            back[i][j] = best_op

    # Backtrace to extract (ref_idx, hyp_idx, op) for C and S
    aligned_pairs: List[Tuple[int, int, str]] = []
    i, j = r_len, h_len
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("C", "S"):
            aligned_pairs.append((i - 1, j - 1, op))
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            j -= 1
        else:
            break
    aligned_pairs.reverse()

    if not aligned_pairs:
        return {
            "wder_lexical": 0.0,
            "aligned_words_c_plus_s": 0,
            "lexical_correct_c": 0,
            "lexical_substitutions_s": 0,
            "speaker_errors_on_c": 0,
            "speaker_errors_on_s": 0,
            "total_speaker_errors": 0,
            "speaker_mapping": {},
        }

    # Compute optimal 1-to-1 Hungarian speaker mapping over non-UNK aligned pairs
    spk_pairs = [
        (hyp_tokens_with_spk[h_idx][1], ref_tokens_with_spk[r_idx][1])
        for r_idx, h_idx, _ in aligned_pairs
        if hyp_tokens_with_spk[h_idx][1] != "UNK" and ref_tokens_with_spk[r_idx][1] != "UNK"
    ]
    acc_info = speaker_accuracy_stats(spk_pairs)
    mapping: Dict[str, str] = dict(acc_info["speaker_mapping"])

    c_count = 0
    s_count = 0
    c_spk_err = 0
    s_spk_err = 0

    for r_idx, h_idx, op in aligned_pairs:
        r_spk = ref_tokens_with_spk[r_idx][1]
        h_spk = hyp_tokens_with_spk[h_idx][1]
        mapped_h = mapping.get(h_spk) if h_spk != "UNK" else "UNK"
        is_spk_err = (mapped_h != r_spk)
        if op == "C":
            c_count += 1
            if is_spk_err:
                c_spk_err += 1
        elif op == "S":
            s_count += 1
            if is_spk_err:
                s_spk_err += 1

    total_aligned = c_count + s_count
    total_spk_err = c_spk_err + s_spk_err
    wder_lex = round(total_spk_err / total_aligned, 4) if total_aligned > 0 else 0.0

    return {
        "wder_lexical": wder_lex,
        "aligned_words_c_plus_s": total_aligned,
        "lexical_correct_c": c_count,
        "lexical_substitutions_s": s_count,
        "speaker_errors_on_c": c_spk_err,
        "speaker_errors_on_s": s_spk_err,
        "total_speaker_errors": total_spk_err,
        "speaker_mapping": mapping,
    }


def evaluate_speaker_attributed_words(
    word_dicts: Sequence[Mapping[str, Any]],
    reference_segments: Optional[Sequence[Mapping[str, Any]]] = None,
    turn_max_gap: float = 2.0,
) -> Dict[str, Any]:
    """
    Computes grouped conversational turns, cpWER, and both WDER definitions
    for a sequence of attributed words [{'word', 'start', 'end', 'speaker'}, ...].
    """
    grouped_turns = phase20_group_words(list(word_dicts), max_gap=turn_max_gap)
    result: Dict[str, Any] = {
        "word_count": len(word_dicts),
        "grouped_turn_count": len(grouped_turns),
    }

    if reference_segments is not None:
        cp_stats = phase20_cpwer_stats(reference_segments, grouped_turns)
        result["cpwer"] = round(float(cp_stats["cpwer"]), 4)
        result["cpwer_details"] = {
            "cpwer": round(float(cp_stats["cpwer"]), 4),
            "errors": int(cp_stats["errors"]),
            "substitutions": int(cp_stats["substitutions"]),
            "insertions": int(cp_stats["insertions"]),
            "deletions": int(cp_stats["deletions"]),
            "reference_words": int(cp_stats["reference_words"]),
            "hypothesis_words": int(cp_stats["hypothesis_words"]),
            "assignment": [list(pair) for pair in cp_stats["assignment"]],
            "hyp_turn_count": len(grouped_turns),
            "ref_segment_count": len(reference_segments),
        }
        result["wder_timestamp"] = compute_timestamp_wder(word_dicts, reference_segments)
        result["wder_lexical"] = compute_lexical_wder(word_dicts, reference_segments)

    return result


def evaluate_refinement_comparison(
    original_words: Sequence[OrchestratedWord],
    refined_words: Sequence[RefinedWordOutput],
    reference_segments: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive comparison framework evaluating:
    1. Original diarization output (baseline).
    2. LLM-refined diarization output.
    3. Correction counts (proposed, accepted, rejected, abstained).
    4. Transcript word sequence and timestamp cryptographic integrity.
    """
    orig_word_list = [w.word for w in original_words]
    ref_word_list = [w.word for w in refined_words]
    orig_ts_list = [(w.start, w.end) for w in original_words]
    ref_ts_list = [(w.start, w.end) for w in refined_words]

    orig_words_sha256 = _hash_word_sequence(orig_word_list)
    ref_words_sha256 = _hash_word_sequence(ref_word_list)
    orig_ts_sha256 = _hash_timestamp_sequence(orig_ts_list)
    ref_ts_sha256 = _hash_timestamp_sequence(ref_ts_list)

    word_seq_intact = (orig_word_list == ref_word_list) and (orig_words_sha256 == ref_words_sha256)
    ts_seq_intact = (orig_ts_list == ref_ts_list) and (orig_ts_sha256 == ref_ts_sha256)
    baseline_recoverable = (
        [w.original_speaker_id for w in original_words]
        == [w.original_speaker_id for w in refined_words]
    )

    # Convert to Phase 20 word dicts for baseline vs refined evaluation
    baseline_word_dicts = [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "confidence": w.asr_confidence,
            "speaker": w.original_speaker_id,
        }
        for w in refined_words
    ]
    refined_word_dicts = [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "confidence": w.asr_confidence,
            "speaker": w.refined_speaker_id,
        }
        for w in refined_words
    ]

    baseline_eval = evaluate_speaker_attributed_words(
        baseline_word_dicts, reference_segments=reference_segments
    )
    refined_eval = evaluate_speaker_attributed_words(
        refined_word_dicts, reference_segments=reference_segments
    )

    # Telemetry counts
    proposed_count = sum(1 for w in refined_words if w.correction_proposed)
    accepted_count = sum(1 for w in refined_words if w.correction_accepted)
    abstained_count = sum(1 for w in refined_words if w.abstained)
    reason_counts = Counter(w.reconciliation_reason for w in refined_words)

    rejection_reasons = {
        ReconciliationReason.REJECTED_MALFORMED_JSON.value,
        ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value,
        ReconciliationReason.REJECTED_INVALID_SPEAKER_ID.value,
        ReconciliationReason.REJECTED_MISSING_WORDS.value,
        ReconciliationReason.REJECTED_EXTRA_WORDS.value,
        ReconciliationReason.REJECTED_REORDERED_OR_MUTATED_WORDS.value,
        ReconciliationReason.REJECTED_WORD_TEXT_MISMATCH.value,
        ReconciliationReason.REJECTED_LOW_CONFIDENCE.value,
    }
    rejected_count = sum(
        1 for w in refined_words
        if w.reconciliation_reason in rejection_reasons or (w.correction_proposed and not w.correction_accepted and not w.abstained)
    )

    deltas: Dict[str, Any] = {}
    if reference_segments is not None:
        deltas = {
            "delta_cpwer": round(refined_eval["cpwer"] - baseline_eval["cpwer"], 4),
            "delta_wder_timestamp_assigned": round(
                refined_eval["wder_timestamp"]["wder_assigned"]
                - baseline_eval["wder_timestamp"]["wder_assigned"],
                4,
            ),
            "delta_wder_timestamp_all_ref": round(
                refined_eval["wder_timestamp"]["wder_all_ref"]
                - baseline_eval["wder_timestamp"]["wder_all_ref"],
                4,
            ),
            "delta_wder_lexical": round(
                refined_eval["wder_lexical"]["wder_lexical"]
                - baseline_eval["wder_lexical"]["wder_lexical"],
                4,
            ),
        }

    return {
        "integrity_checks": {
            "word_sequence_integrity_verified": word_seq_intact,
            "timestamp_integrity_verified": ts_seq_intact,
            "original_transcript_recoverable": baseline_recoverable,
            "original_words_sha256": orig_words_sha256,
            "refined_words_sha256": ref_words_sha256,
            "original_timestamps_sha256": orig_ts_sha256,
            "refined_timestamps_sha256": ref_ts_sha256,
        },
        "refinement_telemetry": {
            "total_words": len(refined_words),
            "proposed_corrections_count": proposed_count,
            "accepted_corrections_count": accepted_count,
            "rejected_corrections_count": rejected_count,
            "abstentions_count": abstained_count,
            "reasons_breakdown": dict(sorted(reason_counts.items())),
        },
        "baseline_metrics": baseline_eval,
        "refined_metrics": refined_eval,
        "metric_deltas": deltas,
    }
