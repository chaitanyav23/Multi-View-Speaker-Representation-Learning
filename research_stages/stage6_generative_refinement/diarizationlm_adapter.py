"""Stage 6.1 — DiarizationLM-Gemma Adapter & Official TPST Utilities.

Implements the official DiarizationLM (Wang et al., 2024) hypothesis formatting,
speaker label mapping, completion truncation, and Transcript-Preserving Speaker Transfer
(TPST / LG-TPST), adapted for diarizers-community/DiarizationLM-Gemma-4-E4B-v1.

Audited against:
1. Multiple speakers (1..10)
2. Repeated words and conversational backchannels
3. Overlapping turns and ambiguous alignments
4. Strict word sequence and timestamp preservation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import string
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import urllib.request

import numpy as np
from scipy import optimize

from .schemas import (
    ConfidenceStatus,
    OrchestratedWord,
    ReconciliationReason,
    RefinedWordOutput,
)

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]


@dataclass(frozen=True)
class PromptOptions:
  """Options for DiarizationLM prompt formatting."""
  prompt_prefix: str = "<|turn>user\n"
  prompt_suffix: str = " --> <turn|>\n<|turn>model\n"
  completion_suffix: str = " [eod]"
  speaker_prefix: str = "<speaker:"
  speaker_suffix: str = ">"


def normalize_text_for_alignment(text: str) -> str:
  """Normalize text for Levenshtein word alignment."""
  text_lower = (text or "").lower().strip()
  words = text_lower.split()
  new_words = []
  for word in words:
    new_word = word
    for punc in PUNCTUATIONS:
      replaced = new_word.replace(punc, "")
      if len(replaced.split()) != 1:
        continue
      new_word = replaced
    if new_word:
      new_words.append(new_word)
  return " ".join(new_words)


def levenshtein_with_edits(
    ref: str,
    hyp: str,
) -> Tuple[int, List[Tuple[int, int]]]:
  """Computes Levenshtein alignment between whitespace-delimited word sequences.

  Returns (distance, align) where align is a list of (ref_pos, hyp_pos) pairs.
  Deletions are (i, -1), insertions are (-1, j), substitutions/matches are (i, j).
  """
  s1 = ref.split()
  s2 = hyp.split()
  n1 = len(s1)
  n2 = len(s2)
  costs = np.zeros((n1 + 1, n2 + 1), dtype=np.int32)
  backptr = np.zeros((n1 + 1, n2 + 1), dtype=np.int32)

  # 0: correct, 1: substitution, 2: insertion, 3: deletion
  for i in range(n1 + 1):
    costs[i, 0] = i
  for j in range(n2):
    costs[0, j + 1] = j + 1
    for i in range(n1):
      ins = costs[i + 1, j] + 1
      del_ = costs[i, j + 1] + 1
      sub = costs[i, j] + (s1[i] != s2[j])
      min_cost = min(ins, del_, sub)
      costs[i + 1, j + 1] = min_cost
      if min_cost == ins:
        backptr[i + 1, j + 1] = 2
      elif min_cost == del_:
        backptr[i + 1, j + 1] = 3
      elif s1[i] == s2[j]:
        backptr[i + 1, j + 1] = 0
      else:
        backptr[i + 1, j + 1] = 1

  align: List[Tuple[int, int]] = []
  i = n1
  j = n2
  while i > 0 or j > 0:
    ed_op = 0
    if i >= 0 and j >= 0:
      ed_op = int(backptr[i, j])
    if i >= 0 and j < 0:
      ed_op = 3
    if i < 0 and j >= 0:
      ed_op = 2
    if ed_op == 2:
      align.append((-1, j - 1))
      j -= 1
    elif ed_op == 3:
      align.append((i - 1, -1))
      i -= 1
    else:
      align.append((i - 1, j - 1))
      i -= 1
      j -= 1

  align.reverse()
  return int(costs[n1, n2]), align


def truncate_suffix_and_tailing_text(text: str, suffix: str) -> str:
  """Official DiarizationLM truncation: removes trailing text following the suffix."""
  if suffix and suffix in text:
    return text[: text.find(suffix)]
  return text


def extract_text_and_spk(
    completions: str,
    po: PromptOptions = PromptOptions(),
    skip_meaningless_speaker: bool = True,
) -> Tuple[str, str]:
  """Official DiarizationLM extraction: extracts plain text and parallel speaker sequence."""
  spk = "1"
  previous_spk = "1"
  result_text: List[str] = []
  result_spk: List[str] = []

  for word in completions.split():
    if word.startswith(po.speaker_prefix):
      if not word.endswith(po.speaker_suffix):
        word += po.speaker_suffix
      extracted_spk = word[len(po.speaker_prefix) : -len(po.speaker_suffix)]
      try:
        spk_int = int(extracted_spk)
        if not extracted_spk or spk_int < 0 or spk_int > 20:
          raise ValueError(f"Unexpected speaker ID: {extracted_spk}")
        spk = extracted_spk
        previous_spk = spk
      except ValueError as exc:
        if skip_meaningless_speaker:
          spk = previous_spk
        else:
          raise exc
    else:
      result_text.append(word)
      result_spk.append(spk)

  return " ".join(result_text), " ".join(result_spk)


def get_aligned_hyp_speakers(
    hyp_text: str,
    ref_text: str,
    ref_spk: str,
) -> str:
  """Aligns ref_text (LLM completion) to hyp_text, applying alignment to ref_spk."""
  _, align = levenshtein_with_edits(
      normalize_text_for_alignment(ref_text),
      normalize_text_for_alignment(hyp_text),
  )
  ref_spk_list = ref_spk.split()
  hyp_spk_align: List[str] = []

  for i, j in align:
    if i == -1:
      hyp_spk_align.append("-1")
    elif j == -1:
      continue
    else:
      if i < len(ref_spk_list):
        hyp_spk_align.append(ref_spk_list[i])
      else:
        hyp_spk_align.append("-1")

  return " ".join(hyp_spk_align)


def get_oracle_speakers(hyp_spk: str, hyp_spk_align: str) -> List[int]:
  """Maps aligned hypothesis speakers onto target sequence, filling unaligned tokens."""
  hyp_spk_list = [int(x) for x in hyp_spk.split()]
  hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]
  n = len(hyp_spk_list)
  if n == 0:
    return []

  max_spk = max(
      max(hyp_spk_list, default=1),
      max(hyp_spk_align_list, default=1),
      1,
  )
  cost_matrix = np.zeros((max_spk, max_spk), dtype=np.float64)
  for aligned, original in zip(hyp_spk_align_list, hyp_spk_list):
    if aligned > 0 and original > 0:
      cost_matrix[aligned - 1, original - 1] += 1

  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix, maximize=True)
  col_map = {r + 1: c + 1 for r, c in zip(row_ind, col_ind)}

  hyp_spk_oracle = hyp_spk_list.copy()
  for i in range(n):
    align_val = hyp_spk_align_list[i] if i < len(hyp_spk_align_list) else -1
    if align_val <= 0:
      if hyp_spk_list[i] <= 0:
        hyp_spk_oracle[i] = hyp_spk_oracle[i - 1] if i > 0 else 1
      else:
        hyp_spk_oracle[i] = hyp_spk_list[i]
      continue
    hyp_spk_oracle[i] = col_map.get(align_val, align_val)

  return hyp_spk_oracle


def transcript_preserving_speaker_transfer(
    src_text: str,
    src_spk: str,
    tgt_text: str,
    tgt_spk: str,
) -> str:
  """Applies source (LLM completion) speakers to target (hypothesis) words via TPST.

  Aligns completion text to hypothesis text monotonically using Levenshtein distance,
  transferring LLM speaker labels directly while preserving original hypothesis words.
  """
  tgt_words = tgt_text.split()
  tgt_spk_list = tgt_spk.split()
  src_words = src_text.split()
  src_spk_list = src_spk.split()

  if len(tgt_words) != len(tgt_spk_list):
    raise ValueError(f"Target length mismatch: {len(tgt_words)} words vs {len(tgt_spk_list)} speakers")
  if len(src_words) != len(src_spk_list):
    raise ValueError(f"Source length mismatch: {len(src_words)} words vs {len(src_spk_list)} speakers")

  tgt_spk_align = get_aligned_hyp_speakers(
      hyp_text=tgt_text,
      ref_text=src_text,
      ref_spk=src_spk,
  )
  align_list = tgt_spk_align.split()
  final_spks = []
  for orig, al in zip(tgt_spk_list, align_list):
    if al == "-1" or not al:
      final_spks.append(orig)
    else:
      final_spks.append(al)

  return " ".join(final_spks)


def apply_locality_gated_speaker_transfer(
    hyp_spk: Sequence[str],
    llm_spk: Sequence[str],
    max_local_span_words: int = 5,
    absorb_ghost_speakers: bool = True,
) -> List[str]:
  """Locality-Gated filter (LG-TPST): accepts LLM modifications on short spans (1..5 words)."""
  if len(hyp_spk) != len(llm_spk):
    raise ValueError("hyp_spk and llm_spk must have the same length")
  n = len(hyp_spk)
  if n == 0:
    return []

  raw_spk = list(llm_spk)
  if absorb_ghost_speakers:
    counts: Dict[str, int] = {}
    for s in raw_spk:
      counts[s] = counts.get(s, 0) + 1
    ghost_thresh = min(12, max(2, int(0.01 * n)))
    if len(counts) > 2:
      valid_spks = {s for s, c in counts.items() if c > ghost_thresh}
      if len(valid_spks) >= 2:
        for idx in range(n):
          if raw_spk[idx] not in valid_spks:
            prev_s = raw_spk[idx - 1] if idx > 0 else ""
            next_s = ""
            for j in range(idx + 1, n):
              if raw_spk[j] in valid_spks:
                next_s = raw_spk[j]
                break
            if prev_s and prev_s in valid_spks:
              raw_spk[idx] = prev_s
            elif next_s and next_s in valid_spks:
              raw_spk[idx] = next_s

  gated_spk = list(hyp_spk)
  i = 0
  while i < n:
    if raw_spk[i] != hyp_spk[i]:
      j = i
      while j < n and raw_spk[j] != hyp_spk[j]:
        j += 1
      span_len = j - i
      if span_len <= max_local_span_words:
        for k in range(i, j):
          gated_spk[k] = raw_spk[k]
      i = j
    else:
      i += 1
  return gated_spk


@dataclass
class DiarizationLMPromptSegment:
  """A segment of the transcript formatted for DiarizationLM inference."""
  segment_id: int
  start_word_index: int
  end_word_index: int
  words: List[OrchestratedWord]
  hypothesis_text: str
  full_prompt: str
  spk_to_tag_map: Dict[str, str]
  tag_to_spk_map: Dict[str, str]


class DiarizationLMGemmaAdapter:
  """Adapter connecting Stage 6 OrchestratedWord sequences to DiarizationLM-Gemma."""

  def __init__(
      self,
      po: PromptOptions = PromptOptions(),
      max_chars_per_segment: int = 4000,
      use_locality_gating: bool = True,
      max_local_span_words: int = 5,
  ) -> None:
    self.po = po
    self.max_chars_per_segment = max_chars_per_segment
    self.use_locality_gating = use_locality_gating
    self.max_local_span_words = max_local_span_words

  def build_prompt_segments(
      self,
      words: Sequence[OrchestratedWord],
      available_speaker_ids: Sequence[str],
  ) -> List[DiarizationLMPromptSegment]:
    """Partitions the word sequence into ~4,000 character segments with deterministic speaker tags."""
    if not words:
      return []

    # Deterministic mapping of project speaker IDs to <speaker:1>, <speaker:2>, ...
    sorted_speakers = sorted(set(available_speaker_ids))
    spk_to_tag: Dict[str, str] = {}
    tag_to_spk: Dict[str, str] = {}

    idx = 1
    for spk in sorted_speakers:
      if spk and spk != "UNK":
        spk_to_tag[spk] = str(idx)
        tag_to_spk[str(idx)] = spk
        idx += 1

    # Map UNK to the first speaker or speaker "1"
    spk_to_tag["UNK"] = "1"

    segments: List[DiarizationLMPromptSegment] = []
    current_words: List[OrchestratedWord] = []
    current_tokens: List[str] = []
    current_chars = 0
    seg_start_idx = 0
    seg_id = 0

    prev_spk_tag = ""

    for i, w in enumerate(words):
      spk_tag = spk_to_tag.get(w.original_speaker_id, "1")
      token_str = ""
      if spk_tag != prev_spk_tag:
        token_str += f"{self.po.speaker_prefix}{spk_tag}{self.po.speaker_suffix} "
        prev_spk_tag = spk_tag
      token_str += w.word + " "

      token_len = len(token_str)
      if current_chars + token_len > self.max_chars_per_segment and current_words:
        # Flush current segment
        hyp_str = "".join(current_tokens).strip()
        full_prompt = f"{self.po.prompt_prefix}{hyp_str}{self.po.prompt_suffix}"
        segments.append(
            DiarizationLMPromptSegment(
                segment_id=seg_id,
                start_word_index=seg_start_idx,
                end_word_index=i - 1,
                words=list(current_words),
                hypothesis_text=hyp_str,
                full_prompt=full_prompt,
                spk_to_tag_map=dict(spk_to_tag),
                tag_to_spk_map=dict(tag_to_spk),
            )
        )
        seg_id += 1
        current_words = []
        current_tokens = []
        current_chars = 0
        seg_start_idx = i
        prev_spk_tag = ""
        # Re-emit speaker tag at start of new segment
        token_str = f"{self.po.speaker_prefix}{spk_tag}{self.po.speaker_suffix} {w.word} "
        prev_spk_tag = spk_tag
        token_len = len(token_str)

      current_words.append(w)
      current_tokens.append(token_str)
      current_chars += token_len

    if current_words:
      hyp_str = "".join(current_tokens).strip()
      full_prompt = f"{self.po.prompt_prefix}{hyp_str}{self.po.prompt_suffix}"
      segments.append(
          DiarizationLMPromptSegment(
              segment_id=seg_id,
              start_word_index=seg_start_idx,
              end_word_index=len(words) - 1,
              words=list(current_words),
              hypothesis_text=hyp_str,
              full_prompt=full_prompt,
              spk_to_tag_map=dict(spk_to_tag),
              tag_to_spk_map=dict(tag_to_spk),
          )
      )

    return segments

  def reconcile_segment(
      self,
      segment: DiarizationLMPromptSegment,
      raw_completion: str,
  ) -> Tuple[List[RefinedWordOutput], Dict[str, Any]]:
    """Transfers LLM completion onto segment words using official DiarizationLM TPST."""
    cleaned = truncate_suffix_and_tailing_text(
        raw_completion, self.po.completion_suffix
    )
    cleaned = truncate_suffix_and_tailing_text(cleaned, "<turn|>").strip()

    llm_text, llm_spk = extract_text_and_spk(cleaned, po=self.po)
    hyp_text, hyp_spk = extract_text_and_spk(
        segment.hypothesis_text, po=self.po
    )

    hyp_words = hyp_text.split()
    n_expected = len(segment.words)

    # Apply TPST transfer
    try:
      raw_transferred_spk = transcript_preserving_speaker_transfer(
          src_text=llm_text,
          src_spk=llm_spk,
          tgt_text=hyp_text,
          tgt_spk=hyp_spk,
      )
      if self.use_locality_gating:
        final_spk_tags = apply_locality_gated_speaker_transfer(
            hyp_spk=hyp_spk.split(),
            llm_spk=raw_transferred_spk.split(),
            max_local_span_words=self.max_local_span_words,
            absorb_ghost_speakers=True,
        )
      else:
        final_spk_tags = raw_transferred_spk.split()
      transfer_success = True
      error_msg = None
    except Exception as exc:
      transfer_success = False
      error_msg = str(exc)
      final_spk_tags = hyp_spk.split()

    refined_outputs: List[RefinedWordOutput] = []
    proposed_count = 0
    accepted_count = 0
    abstained_count = 0

    for idx, (word_obj, final_tag) in enumerate(
        zip(segment.words, final_spk_tags)
    ):
      orig_spk = word_obj.original_speaker_id
      proposed_spk = segment.tag_to_spk_map.get(final_tag, orig_spk)

      if not transfer_success:
        refined_outputs.append(
            RefinedWordOutput(
                word_index=word_obj.word_index,
                word=word_obj.word,
                start=word_obj.start,
                end=word_obj.end,
                asr_confidence=word_obj.asr_confidence,
                original_speaker_id=orig_spk,
                refined_speaker_id=orig_spk,
                correction_proposed=False,
                correction_accepted=False,
                abstained=True,
                proposed_speaker_id=orig_spk,
                reconciliation_reason=ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value,
                reasoning_summary=f"TPST alignment error: {error_msg}; baseline preserved.",
                confidence_status=ConfidenceStatus.BASELINE_PRESERVED.value,
                context_window_id=f"seg_{segment.segment_id:03d}",
                attribution_status=word_obj.attribution_status,
                is_overlapping_speech=word_obj.is_overlapping_speech,
                overlapping_speakers=word_obj.overlapping_speakers,
            )
        )
        abstained_count += 1
        continue

      if proposed_spk != orig_spk:
        proposed_count += 1
        # Validated in available speakers?
        if proposed_spk in segment.tag_to_spk_map.values():
          accepted_count += 1
          refined_outputs.append(
              RefinedWordOutput(
                  word_index=word_obj.word_index,
                  word=word_obj.word,
                  start=word_obj.start,
                  end=word_obj.end,
                  asr_confidence=word_obj.asr_confidence,
                  original_speaker_id=orig_spk,
                  refined_speaker_id=proposed_spk,
                  correction_proposed=True,
                  correction_accepted=True,
                  abstained=False,
                  proposed_speaker_id=proposed_spk,
                  reconciliation_reason=ReconciliationReason.ACCEPTED_VALIDATED_CORRECTION.value,
                  reasoning_summary="DiarizationLM TPST transfer proposed turn-boundary correction.",
                  confidence_status=ConfidenceStatus.HIGH.value,
                  context_window_id=f"seg_{segment.segment_id:03d}",
                  attribution_status=word_obj.attribution_status,
                  is_overlapping_speech=word_obj.is_overlapping_speech,
                  overlapping_speakers=word_obj.overlapping_speakers,
              )
          )
        else:
          abstained_count += 1
          refined_outputs.append(
              RefinedWordOutput(
                  word_index=word_obj.word_index,
                  word=word_obj.word,
                  start=word_obj.start,
                  end=word_obj.end,
                  asr_confidence=word_obj.asr_confidence,
                  original_speaker_id=orig_spk,
                  refined_speaker_id=orig_spk,
                  correction_proposed=True,
                  correction_accepted=False,
                  abstained=True,
                  proposed_speaker_id=proposed_spk,
                  reconciliation_reason=ReconciliationReason.REJECTED_INVALID_SPEAKER_ID.value,
                  reasoning_summary=f"Proposed speaker {proposed_spk!r} not in available speaker inventory.",
                  confidence_status=ConfidenceStatus.UNCERTAIN.value,
                  context_window_id=f"seg_{segment.segment_id:03d}",
                  attribution_status=word_obj.attribution_status,
                  is_overlapping_speech=word_obj.is_overlapping_speech,
                  overlapping_speakers=word_obj.overlapping_speakers,
              )
          )
      else:
        abstained_count += 1
        refined_outputs.append(
            RefinedWordOutput(
                word_index=word_obj.word_index,
                word=word_obj.word,
                start=word_obj.start,
                end=word_obj.end,
                asr_confidence=word_obj.asr_confidence,
                original_speaker_id=orig_spk,
                refined_speaker_id=orig_spk,
                correction_proposed=False,
                correction_accepted=False,
                abstained=True,
                proposed_speaker_id=orig_spk,
                reconciliation_reason=ReconciliationReason.UNCHANGED_ORIGINAL_CONFIRMED.value,
                reasoning_summary="DiarizationLM confirmed baseline speaker assignment.",
                confidence_status=ConfidenceStatus.BASELINE_PRESERVED.value,
                context_window_id=f"seg_{segment.segment_id:03d}",
                attribution_status=word_obj.attribution_status,
                is_overlapping_speech=word_obj.is_overlapping_speech,
                overlapping_speakers=word_obj.overlapping_speakers,
            )
        )

    telemetry = {
        "segment_id": segment.segment_id,
        "word_count": len(segment.words),
        "transfer_success": transfer_success,
        "proposed_count": proposed_count,
        "accepted_count": accepted_count,
        "abstained_count": abstained_count,
        "error_msg": error_msg,
    }
    return refined_outputs, telemetry
