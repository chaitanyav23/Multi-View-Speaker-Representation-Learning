"""
Stage 6 — Step D: Constrained Parser and TPST-Inspired Reconciliation
Implements deterministic validation, monotonic sequence alignment, and
Transcript-Preserving Speaker Transfer (TPST) reconciliation.

Why Hungarian matching is NOT used for word alignment:
Hungarian matching solves an unconstrained permutation assignment that does not
enforce sequential monotonicity (p_1 < p_2 <=> q_1 < q_2). In conversational
transcripts with repeated tokens (e.g. "you", "the", "yes"), permutation matching
can cross-align tokens across turns or mask reordered words. Instead, this module
enforces strict monotonic order and index-locked lexical alignment, rejecting
reordered, missing, extra, or mutated words deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schemas import (
    ConfidenceStatus,
    ContextWindow,
    OrchestratedWord,
    ReconciliationReason,
    RefinedWordOutput,
)


_CONFIDENCE_RANK = {
    ConfidenceStatus.HIGH.value: 3,
    ConfidenceStatus.MEDIUM.value: 2,
    ConfidenceStatus.LOW.value: 1,
    ConfidenceStatus.UNCERTAIN.value: 0,
    ConfidenceStatus.BASELINE_PRESERVED.value: 0,
}


def normalize_lexical_token(token: str) -> str:
    """Normalizes case and boundary punctuation for TPST surface-form comparison."""
    t = (token or "").strip().lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", t).strip()


def strip_optional_markdown_fences(raw_text: str) -> str:
    """Strips optional ```json ... ``` fences if emitted by an LLM."""
    s = (raw_text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return s


@dataclass(frozen=True)
class ReconcilerConfig:
    """
    Configuration for the Constrained Parser & TPST Reconciler.
    - strict_word_text_match: When True, requires exact character-for-character equality
      between the LLM decision's `word` and the original ASR `word`. When False, allows
      minor punctuation/case normalization via TPST while still transferring only the
      speaker label onto the original unmodified ASR word.
    - reject_window_on_length_mismatch: When True, if the LLM returns extra words
      (len(decisions) > len(window.words)), all words in the window are rejected with
      REJECTED_EXTRA_WORDS. When False, only extra/unaligned entries are rejected.
    - reject_window_on_reordering: When True, if any word_index or lexical swap is
      detected in the window, affected words (or the window) are rejected with
      REJECTED_REORDERED_OR_MUTATED_WORDS.
    - min_confidence_to_accept: Minimum confidence level ("LOW", "MEDIUM", "HIGH")
      required to accept a speaker reassignment.
    - allow_reassigning_unk: Whether the LLM is allowed to assign a valid speaker ID
      from available_speaker_ids to a word originally labeled "UNK".
    """
    strict_word_text_match: bool = True
    reject_window_on_extra_words: bool = True
    reject_window_on_reordering: bool = True
    min_confidence_to_accept: str = ConfidenceStatus.MEDIUM.value
    allow_reassigning_unk: bool = True


@dataclass
class WindowReconciliationReport:
    """Audit telemetry for a single ContextWindow's reconciliation."""
    window_id: str
    is_valid_json: bool = True
    is_valid_schema: bool = True
    expected_word_count: int = 0
    received_decision_count: int = 0
    missing_word_indices: List[int] = field(default_factory=list)
    extra_decisions: List[Dict[str, Any]] = field(default_factory=list)
    reordered_word_indices: List[int] = field(default_factory=list)
    mismatched_word_indices: List[int] = field(default_factory=list)
    invalid_speaker_word_indices: List[int] = field(default_factory=list)
    accepted_word_indices: List[int] = field(default_factory=list)
    abstained_word_indices: List[int] = field(default_factory=list)
    window_rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "is_valid_json": self.is_valid_json,
            "is_valid_schema": self.is_valid_schema,
            "expected_word_count": self.expected_word_count,
            "received_decision_count": self.received_decision_count,
            "missing_word_indices": list(self.missing_word_indices),
            "extra_decisions_count": len(self.extra_decisions),
            "extra_decisions": list(self.extra_decisions),
            "reordered_word_indices": list(self.reordered_word_indices),
            "mismatched_word_indices": list(self.mismatched_word_indices),
            "invalid_speaker_word_indices": list(self.invalid_speaker_word_indices),
            "accepted_word_indices": list(self.accepted_word_indices),
            "abstained_word_indices": list(self.abstained_word_indices),
            "window_rejection_reason": self.window_rejection_reason,
        }


class TPSTReconciler:
    """
    Deterministic Constrained Parser and Transcript-Preserving Speaker Transfer (TPST)
    Reconciler. Validates LLM responses per window and reconciles them onto the
    immutable original word sequence.
    """

    def __init__(self, config: ReconcilerConfig | None = None) -> None:
        self.config = config or ReconcilerConfig()

    def reconcile_window(
        self,
        window: ContextWindow,
        raw_llm_output: str,
    ) -> Tuple[Dict[int, RefinedWordOutput], WindowReconciliationReport]:
        """
        Validates and reconciles a single ContextWindow's LLM output.
        Returns a mapping {word_index: RefinedWordOutput} for all words in `window.words`
        and a detailed `WindowReconciliationReport`.
        """
        report = WindowReconciliationReport(
            window_id=window.window_id,
            expected_word_count=len(window.words),
        )
        valid_speakers = set(window.available_speaker_ids)

        # 1. Parse JSON
        cleaned = strip_optional_markdown_fences(raw_llm_output)
        try:
            payload = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            report.is_valid_json = False
            report.window_rejection_reason = ReconciliationReason.REJECTED_MALFORMED_JSON.value
            return (
                self._preserve_all_words_in_window(
                    window=window,
                    reason=ReconciliationReason.REJECTED_MALFORMED_JSON.value,
                    note="Malformed JSON in LLM response; baseline speaker preserved.",
                ),
                report,
            )

        # 2. Validate top-level schema
        if not isinstance(payload, dict) or "decisions" not in payload or not isinstance(payload["decisions"], list):
            report.is_valid_schema = False
            report.window_rejection_reason = ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value
            return (
                self._preserve_all_words_in_window(
                    window=window,
                    reason=ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value,
                    note="Invalid JSON schema (missing 'decisions' list); baseline speaker preserved.",
                ),
                report,
            )

        raw_decisions: List[Any] = payload["decisions"]
        report.received_decision_count = len(raw_decisions)

        for item in raw_decisions:
            if not isinstance(item, dict) or "word" not in item:
                report.is_valid_schema = False
                report.window_rejection_reason = ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value
                return (
                    self._preserve_all_words_in_window(
                        window=window,
                        reason=ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value,
                        note="Decision item missing required fields; baseline speaker preserved.",
                    ),
                    report,
                )

        # 3. Check for Reordered Indices or Positional Swaps
        expected_indices = [w.word_index for w in window.words]
        expected_index_set = set(expected_indices)
        word_by_index = {w.word_index: w for w in window.words}

        # Extract word_index if provided; otherwise infer positional only if lengths match
        parsed_indices: List[Optional[int]] = []
        for pos, d in enumerate(raw_decisions):
            if "word_index" in d and isinstance(d["word_index"], int) and not isinstance(d["word_index"], bool):
                parsed_indices.append(int(d["word_index"]))
            elif len(raw_decisions) == len(window.words):
                parsed_indices.append(expected_indices[pos])
            else:
                parsed_indices.append(None)

        # Detect non-monotonic (reordered) indices
        reordered_detected = False
        prev_idx: Optional[int] = None
        for pos, idx_val in enumerate(parsed_indices):
            if idx_val is not None:
                if prev_idx is not None and idx_val < prev_idx:
                    reordered_detected = True
                    if idx_val in expected_index_set:
                        report.reordered_word_indices.append(idx_val)
                    if prev_idx in expected_index_set and prev_idx not in report.reordered_word_indices:
                        report.reordered_word_indices.append(prev_idx)
                prev_idx = idx_val

        # Also detect positional lexical reordering when indices appear monotonic
        # (e.g., LLM kept word_index 0, 1 but swapped word strings "We", "are," -> "are,", "We")
        if len(raw_decisions) == len(window.words):
            for pos in range(len(window.words) - 1):
                w_curr = window.words[pos]
                w_next = window.words[pos + 1]
                d_curr_word = str(raw_decisions[pos].get("word", ""))
                d_next_word = str(raw_decisions[pos + 1].get("word", ""))
                if (
                    w_curr.word != w_next.word
                    and d_curr_word == w_next.word
                    and d_next_word == w_curr.word
                ):
                    reordered_detected = True
                    for r_idx in (w_curr.word_index, w_next.word_index):
                        if r_idx not in report.reordered_word_indices:
                            report.reordered_word_indices.append(r_idx)

        if reordered_detected and self.config.reject_window_on_reordering:
            report.window_rejection_reason = ReconciliationReason.REJECTED_REORDERED_OR_MUTATED_WORDS.value
            return (
                self._preserve_all_words_in_window(
                    window=window,
                    reason=ReconciliationReason.REJECTED_REORDERED_OR_MUTATED_WORDS.value,
                    note="Reordered words detected in LLM response; rejecting window to preserve transcript integrity.",
                ),
                report,
            )

        # 4. Detect Extra Words (len > expected, out-of-window indices, or duplicate indices)
        seen_indices: set[int] = set()
        aligned_decision_by_index: Dict[int, Dict[str, Any]] = {}

        for pos, d in enumerate(raw_decisions):
            idx_val = parsed_indices[pos]
            if idx_val is None or idx_val not in expected_index_set or idx_val in seen_indices:
                report.extra_decisions.append(d)
                continue
            seen_indices.add(idx_val)
            aligned_decision_by_index[idx_val] = d

        if len(raw_decisions) > len(window.words) or len(report.extra_decisions) > 0:
            if self.config.reject_window_on_extra_words:
                report.window_rejection_reason = ReconciliationReason.REJECTED_EXTRA_WORDS.value
                return (
                    self._preserve_all_words_in_window(
                        window=window,
                        reason=ReconciliationReason.REJECTED_EXTRA_WORDS.value,
                        note=(
                            f"Extra or duplicate words detected in LLM response "
                            f"(expected {len(window.words)}, got {len(raw_decisions)}); "
                            f"preserving baseline assignments."
                        ),
                    ),
                    report,
                )

        # 5. Per-word validation and TPST speaker transfer
        reconciled_map: Dict[int, RefinedWordOutput] = {}
        min_conf_rank = _CONFIDENCE_RANK.get(self.config.min_confidence_to_accept, 2)

        for w in window.words:
            w_idx = w.word_index

            # Check if word was missing from LLM response
            if w_idx not in aligned_decision_by_index:
                report.missing_word_indices.append(w_idx)
                reconciled_map[w_idx] = self._build_preserved_word(
                    word=w,
                    window_id=window.window_id,
                    reason=ReconciliationReason.REJECTED_MISSING_WORDS.value,
                    note="Word omitted from LLM response; original speaker assignment preserved.",
                    proposed_speaker_id=None,
                    correction_proposed=False,
                    abstained=False,
                )
                continue

            # Check if word was flagged as reordered
            if w_idx in report.reordered_word_indices:
                reconciled_map[w_idx] = self._build_preserved_word(
                    word=w,
                    window_id=window.window_id,
                    reason=ReconciliationReason.REJECTED_REORDERED_OR_MUTATED_WORDS.value,
                    note="Word order violation detected; original speaker assignment preserved.",
                    proposed_speaker_id=None,
                    correction_proposed=False,
                    abstained=False,
                )
                continue

            d = aligned_decision_by_index[w_idx]
            llm_word_str = str(d.get("word", ""))

            # Lexical alignment check (strict vs TPST normalized)
            if self.config.strict_word_text_match:
                lexical_match = (llm_word_str == w.word)
            else:
                lexical_match = (
                    llm_word_str == w.word
                    or (
                        normalize_lexical_token(llm_word_str) != ""
                        and normalize_lexical_token(llm_word_str) == normalize_lexical_token(w.word)
                    )
                )

            if not lexical_match:
                report.mismatched_word_indices.append(w_idx)
                reconciled_map[w_idx] = self._build_preserved_word(
                    word=w,
                    window_id=window.window_id,
                    reason=ReconciliationReason.REJECTED_WORD_TEXT_MISMATCH.value,
                    note=(
                        f"LLM mutated word text ({llm_word_str!r} != original {w.word!r}); "
                        f"rejecting proposal and preserving original word and speaker."
                    ),
                    proposed_speaker_id=d.get("proposed_speaker_id"),
                    correction_proposed=True,
                    abstained=False,
                )
                continue

            abstain_flag = bool(d.get("abstain", False))
            raw_proposed_spk = d.get("proposed_speaker_id")
            proposed_spk = str(raw_proposed_spk).strip() if raw_proposed_spk is not None else None
            raw_conf = str(d.get("confidence", ConfidenceStatus.UNCERTAIN.value)).upper().strip()
            if raw_conf not in _CONFIDENCE_RANK:
                raw_conf = ConfidenceStatus.UNCERTAIN.value
            reasoning_str = str(d.get("reasoning", "")).strip()

            # Explicit LLM abstention
            if abstain_flag or proposed_spk is None or proposed_spk == "":
                report.abstained_word_indices.append(w_idx)
                reconciled_map[w_idx] = RefinedWordOutput(
                    word_index=w.word_index,
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    asr_confidence=w.asr_confidence,
                    original_speaker_id=w.original_speaker_id,
                    refined_speaker_id=w.original_speaker_id,
                    correction_proposed=False,
                    correction_accepted=False,
                    abstained=True,
                    proposed_speaker_id=proposed_spk,
                    reconciliation_reason=ReconciliationReason.ABSTAINED_BY_LLM.value,
                    reasoning_summary=reasoning_str or "LLM abstained; baseline speaker preserved.",
                    confidence_status=raw_conf,
                    context_window_id=window.window_id,
                    attribution_status=w.attribution_status,
                    is_overlapping_speech=w.is_overlapping_speech,
                    overlapping_speakers=w.overlapping_speakers,
                )
                continue

            # No change proposed (proposed speaker equals original speaker)
            if proposed_spk == w.original_speaker_id:
                reconciled_map[w_idx] = RefinedWordOutput(
                    word_index=w.word_index,
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    asr_confidence=w.asr_confidence,
                    original_speaker_id=w.original_speaker_id,
                    refined_speaker_id=w.original_speaker_id,
                    correction_proposed=False,
                    correction_accepted=False,
                    abstained=False,
                    proposed_speaker_id=proposed_spk,
                    reconciliation_reason=ReconciliationReason.UNCHANGED_ORIGINAL_CONFIRMED.value,
                    reasoning_summary=reasoning_str or "Original speaker assignment confirmed.",
                    confidence_status=raw_conf,
                    context_window_id=window.window_id,
                    attribution_status=w.attribution_status,
                    is_overlapping_speech=w.is_overlapping_speech,
                    overlapping_speakers=w.overlapping_speakers,
                )
                continue

            # Validate proposed speaker ID against available_speaker_ids
            if proposed_spk not in valid_speakers:
                report.invalid_speaker_word_indices.append(w_idx)
                reconciled_map[w_idx] = self._build_preserved_word(
                    word=w,
                    window_id=window.window_id,
                    reason=ReconciliationReason.REJECTED_INVALID_SPEAKER_ID.value,
                    note=(
                        f"Proposed speaker ID {proposed_spk!r} is not in valid available_speaker_ids "
                        f"{sorted(valid_speakers)}; baseline speaker preserved."
                    ),
                    proposed_speaker_id=proposed_spk,
                    correction_proposed=True,
                    abstained=False,
                )
                continue

            # Check if original was UNK and policy forbids reassigning UNK
            if w.original_speaker_id == "UNK" and not self.config.allow_reassigning_unk:
                reconciled_map[w_idx] = self._build_preserved_word(
                    word=w,
                    window_id=window.window_id,
                    reason=ReconciliationReason.ABSTAINED_BY_LLM.value,
                    note="Original word was UNK and allow_reassigning_unk=False; preserving UNK.",
                    proposed_speaker_id=proposed_spk,
                    correction_proposed=True,
                    abstained=True,
                )
                continue

            # Validate confidence threshold
            conf_rank = _CONFIDENCE_RANK.get(raw_conf, 0)
            if conf_rank < min_conf_rank:
                reconciled_map[w_idx] = self._build_preserved_word(
                    word=w,
                    window_id=window.window_id,
                    reason=ReconciliationReason.REJECTED_LOW_CONFIDENCE.value,
                    note=(
                        f"Proposed speaker {proposed_spk!r} has confidence {raw_conf!r} below "
                        f"minimum threshold {self.config.min_confidence_to_accept!r}; preserving baseline."
                    ),
                    proposed_speaker_id=proposed_spk,
                    correction_proposed=True,
                    abstained=False,
                    confidence_status=raw_conf,
                )
                continue

            # Validated speaker transfer! Copy ONLY the validated speaker ID onto the
            # immutable original word text and timestamps.
            report.accepted_word_indices.append(w_idx)
            reconciled_map[w_idx] = RefinedWordOutput(
                word_index=w.word_index,
                word=w.word,
                start=w.start,
                end=w.end,
                asr_confidence=w.asr_confidence,
                original_speaker_id=w.original_speaker_id,
                refined_speaker_id=proposed_spk,
                correction_proposed=True,
                correction_accepted=True,
                abstained=False,
                proposed_speaker_id=proposed_spk,
                reconciliation_reason=ReconciliationReason.ACCEPTED_VALIDATED_CORRECTION.value,
                reasoning_summary=reasoning_str or f"Validated transfer from {w.original_speaker_id} to {proposed_spk}.",
                confidence_status=raw_conf,
                context_window_id=window.window_id,
                attribution_status=w.attribution_status,
                is_overlapping_speech=w.is_overlapping_speech,
                overlapping_speakers=w.overlapping_speakers,
            )

        return reconciled_map, report

    def reconcile_all_windows(
        self,
        original_words: Sequence[OrchestratedWord],
        windows: Sequence[ContextWindow],
        raw_outputs_by_window: Dict[str, str],
    ) -> Tuple[List[RefinedWordOutput], List[WindowReconciliationReport]]:
        """
        Reconciles all windows across a recording and merges overlapping window
        decisions deterministically:
        - Each word index is governed primarily by the window whose `core_word_indices`
          contains that word index.
        - Enforces post-condition assertions verifying that the output word sequence
          and timestamps are 100% bit-identical to `original_words`.
        """
        primary_outputs: Dict[int, RefinedWordOutput] = {}
        window_reports: List[WindowReconciliationReport] = []

        for window in windows:
            raw_json = raw_outputs_by_window.get(window.window_id, "")
            win_map, win_report = self.reconcile_window(window, raw_json)
            window_reports.append(win_report)

            core_set = set(window.core_word_indices)
            for w_idx, refined_item in win_map.items():
                if w_idx in core_set or w_idx not in primary_outputs:
                    primary_outputs[w_idx] = refined_item

        final_sequence: List[RefinedWordOutput] = []
        for w in original_words:
            if w.word_index in primary_outputs:
                final_sequence.append(primary_outputs[w.word_index])
            else:
                final_sequence.append(
                    self._build_preserved_word(
                        word=w,
                        window_id="unwindowed",
                        reason=ReconciliationReason.NO_LLM_PROPOSAL.value,
                        note="No context window covered word; baseline preserved.",
                        proposed_speaker_id=None,
                        correction_proposed=False,
                        abstained=True,
                    )
                )

        # Post-condition invariant verification: original transcript and timestamps must be intact
        self._verify_transcript_and_timestamp_invariants(original_words, final_sequence)
        return final_sequence, window_reports

    @staticmethod
    def _verify_transcript_and_timestamp_invariants(
        original_words: Sequence[OrchestratedWord],
        refined_words: Sequence[RefinedWordOutput],
    ) -> None:
        if len(original_words) != len(refined_words):
            raise RuntimeError(
                f"TPST invariant violation: word count changed from {len(original_words)} "
                f"to {len(refined_words)}"
            )
        for orig, ref in zip(original_words, refined_words):
            if orig.word_index != ref.word_index:
                raise RuntimeError(
                    f"TPST invariant violation: word_index mismatch ({orig.word_index} != {ref.word_index})"
                )
            if orig.word != ref.word:
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: "
                    f"word mutated from {orig.word!r} to {ref.word!r}"
                )
            if orig.start != ref.start or orig.end != ref.end:
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: "
                    f"timestamps mutated from ({orig.start}, {orig.end}) to ({ref.start}, {ref.end})"
                )
            if orig.original_speaker_id != ref.original_speaker_id:
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: "
                    f"original_speaker_id mutated from {orig.original_speaker_id!r} to {ref.original_speaker_id!r}"
                )

    def _preserve_all_words_in_window(
        self,
        window: ContextWindow,
        reason: str,
        note: str,
    ) -> Dict[int, RefinedWordOutput]:
        result: Dict[int, RefinedWordOutput] = {}
        for w in window.words:
            result[w.word_index] = self._build_preserved_word(
                word=w,
                window_id=window.window_id,
                reason=reason,
                note=note,
                proposed_speaker_id=None,
                correction_proposed=False,
                abstained=False,
            )
        return result

    @staticmethod
    def _build_preserved_word(
        word: OrchestratedWord,
        window_id: str,
        reason: str,
        note: str,
        proposed_speaker_id: Optional[str],
        correction_proposed: bool,
        abstained: bool,
        confidence_status: str = ConfidenceStatus.BASELINE_PRESERVED.value,
    ) -> RefinedWordOutput:
        return RefinedWordOutput(
            word_index=word.word_index,
            word=word.word,
            start=word.start,
            end=word.end,
            asr_confidence=word.asr_confidence,
            original_speaker_id=word.original_speaker_id,
            refined_speaker_id=word.original_speaker_id,
            correction_proposed=correction_proposed,
            correction_accepted=False,
            abstained=abstained,
            proposed_speaker_id=proposed_speaker_id,
            reconciliation_reason=reason,
            reasoning_summary=note,
            confidence_status=confidence_status,
            context_window_id=window_id,
            attribution_status=word.attribution_status,
            is_overlapping_speech=word.is_overlapping_speech,
            overlapping_speakers=word.overlapping_speakers,
        )
