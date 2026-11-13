"""
ConvoFuse — Ground-Truth-Isolated Generative Speaker Refinement (Part 4).

Integrates `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (Q4_K_M via local Ollama)
with strict Transcript-Preserving Speaker Transfer (TPST), speaker inventory validation,
UNK preservation, explicit rejection reasons, and measured telemetry.

STRICT GROUND-TRUTH ISOLATION INVARIANT (Part 4.1 & 4.5):
No class or function in this module accepts reference transcripts, ground-truth speaker
labels, reference timestamps, or evaluation metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.request

from .contracts import (
    ConvoFuseWord,
    RefinementDecision,
    RejectionOrAbstentionReason,
    SpeakerIdentityMapper,
    WordProvenance,
)
from research_stages.stage6_generative_refinement.diarizationlm_adapter import (
    PromptOptions,
    levenshtein_with_edits,
    normalize_text_for_alignment,
    truncate_suffix_and_tailing_text,
)


@dataclass(frozen=True)
class SegmentRefinementTelemetry:
    """Measured telemetry for a single DiarizationLM-Gemma prompt segment (Part 4.3)."""
    segment_id: str
    recording_id: str
    start_word_index: int
    end_word_index: int
    word_count: int
    model_identifier: str
    ollama_model_name: str
    quantization: str
    provider_version: str
    prompt_configuration: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    done: bool
    done_reason: str
    runtime_sec: float
    prompt_text: str
    raw_completion: str
    cleaned_completion: str
    accepted_count: int
    rejected_count: int
    abstained_count: int
    unchanged_count: int
    rejection_reasons_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerativeRefinementResult:
    """Complete output of ground-truth-isolated generative refinement across a recording."""
    recording_id: str
    model_identifier: str
    ollama_model_name: str
    quantization: str
    provider_version: str
    total_runtime_sec: float
    total_prompt_tokens: int
    total_completion_tokens: int
    segment_count: int
    refined_words: List[ConvoFuseWord]
    segment_telemetries: List[SegmentRefinementTelemetry]
    summary_counts: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "model_identifier": self.model_identifier,
            "ollama_model_name": self.ollama_model_name,
            "quantization": self.quantization,
            "provider_version": self.provider_version,
            "total_runtime_sec": round(self.total_runtime_sec, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "segment_count": self.segment_count,
            "summary_counts": dict(self.summary_counts),
            "segment_telemetries": [s.to_dict() for s in self.segment_telemetries],
        }


def extract_text_and_spk_strict(
    completion_text: str,
    po: PromptOptions = PromptOptions(),
) -> Tuple[List[str], List[str]]:
    """
    Strict parser for DiarizationLM completion strings (Part 4.4):
    - Never silently maps malformed or missing tags to a previous valid speaker.
    - Words appearing before any `<speaker:...>` tag receive `"MISSING_TAG"`.
    - Malformed tags (e.g., `<speaker:abc>`, `<speaker:>`, unclosed `<speaker:1`)
      assign `"MALFORMED_TAG"` to subsequent words until a valid tag appears.
    - Syntactically valid integer tags `<speaker:N>` assign `"N"` (which is later
      checked against the recording's speaker inventory `1..K`).
    """
    current_tag = "MISSING_TAG"
    words_out: List[str] = []
    tags_out: List[str] = []

    for raw_token in completion_text.split():
        # Check if token looks like a speaker tag (starts with `<speaker` or matches `<speaker:...`)
        if raw_token.startswith(po.speaker_prefix) or raw_token.startswith("<speaker"):
            if not (
                raw_token.startswith(po.speaker_prefix)
                and raw_token.endswith(po.speaker_suffix)
                and len(raw_token) > len(po.speaker_prefix) + len(po.speaker_suffix)
            ):
                current_tag = f"MALFORMED_TAG:{raw_token}"
                continue
            inner = raw_token[len(po.speaker_prefix) : -len(po.speaker_suffix)].strip()
            if not re.fullmatch(r"\d+", inner):
                current_tag = f"MALFORMED_TAG:{raw_token}"
                continue
            current_tag = inner
        else:
            words_out.append(raw_token)
            tags_out.append(current_tag)

    return words_out, tags_out


def align_completion_tags_to_hypothesis(
    hyp_words: Sequence[str],
    llm_words: Sequence[str],
    llm_tags: Sequence[str],
) -> List[Optional[str]]:
    """
    Monotonically aligns `llm_words` (from completion) to `hyp_words` (original ASR words)
    via Levenshtein edit distance and transfers `llm_tags` to each hypothesis word index.
    Returns a list of length `len(hyp_words)` where unaligned (omitted) words have `None`.
    """
    if not hyp_words:
        return []
    if not llm_words:
        return [None] * len(hyp_words)

    # Normalize each word individually while keeping 1-to-1 correspondence with hyp_words / llm_words
    norm_hyp: List[str] = []
    for w in hyp_words:
        nw = normalize_text_for_alignment(w)
        norm_hyp.append(nw.replace(" ", "") if nw else w.lower())

    norm_llm: List[str] = []
    for w in llm_words:
        nw = normalize_text_for_alignment(w)
        norm_llm.append(nw.replace(" ", "") if nw else w.lower())

    _, edits = levenshtein_with_edits(" ".join(norm_llm), " ".join(norm_hyp))

    aligned_tags: List[Optional[str]] = []
    for llm_idx, hyp_idx in edits:
        if hyp_idx == -1:
            # Deletion in hypothesis (extra word inserted by LLM): ignore so original word count is preserved
            continue
        elif llm_idx == -1:
            # Insertion in hypothesis (word omitted by LLM completion)
            aligned_tags.append(None)
        else:
            if 0 <= llm_idx < len(llm_tags):
                aligned_tags.append(llm_tags[llm_idx])
            else:
                aligned_tags.append(None)

    # Guarantee exact length match
    if len(aligned_tags) < len(hyp_words):
        aligned_tags.extend([None] * (len(hyp_words) - len(aligned_tags)))
    elif len(aligned_tags) > len(hyp_words):
        aligned_tags = aligned_tags[: len(hyp_words)]

    return aligned_tags


class OllamaDiarizationLMClient:
    """
    HTTP client for local Ollama server running `diarizationlm-gemma` (`DiarizationLM-Gemma-4-E4B-v1` Q4_K_M).
    Records genuine measured telemetry from the Ollama API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        ollama_model_name: str = "diarizationlm-gemma",
        model_identifier: str = "diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
        quantization: str = "Q4_K_M",
        temperature: float = 0.0,
        num_predict: int = 2048,
        timeout_sec: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ollama_model_name = ollama_model_name
        self.model_identifier = model_identifier
        self.quantization = quantization
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout_sec = timeout_sec
        self._cached_version: Optional[str] = None

    def get_provider_version(self) -> str:
        """Query real Ollama server version."""
        if self._cached_version is not None:
            return self._cached_version
        try:
            req = urllib.request.Request(f"{self.base_url}/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ver = str(data.get("version", "unknown"))
                self._cached_version = f"ollama-{ver}"
                return self._cached_version
        except Exception:
            return "ollama-unavailable"

    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        """Execute completion request against Ollama `/api/generate` and return measured metrics."""
        stop_tokens = ["<turn|>", "[eod]"]
        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_k": 1,
                "seed": 42,
                "num_predict": self.num_predict,
                "stop": stop_tokens,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        elapsed_sec = time.perf_counter() - t0

        return {
            "raw_completion": str(resp_data.get("response", "")),
            "done": bool(resp_data.get("done", False)),
            "done_reason": str(resp_data.get("done_reason", "stop" if resp_data.get("done") else "unknown")),
            "prompt_tokens": int(resp_data.get("prompt_eval_count", 0)),
            "completion_tokens": int(resp_data.get("eval_count", 0)),
            "runtime_sec": round(float(elapsed_sec), 4),
            "ollama_total_duration_ns": int(resp_data.get("total_duration", 0)),
            "prompt_configuration": {
                "temperature": self.temperature,
                "top_k": 1,
                "seed": 42,
                "num_predict": self.num_predict,
                "stop": stop_tokens,
            },
            "full_api_response": resp_data,
        }


class ConvoFuseGenerativeRefiner:
    """
    Ground-Truth-Isolated Stage 6 Generative Speaker Refinement Engine (Part 4).
    """

    FORBIDDEN_GT_KEYS = {
        "reference_transcript",
        "reference_segments",
        "reference_rttm",
        "ground_truth",
        "gt_segments",
        "gt_labels",
        "reference_path",
        "reference_json_path",
        "evaluation_metrics",
    }

    def __init__(
        self,
        client: Optional[OllamaDiarizationLMClient] = None,
        po: PromptOptions = PromptOptions(),
        max_chars_per_segment: int = 3600,
        max_words_per_segment: int = 300,
        use_locality_gating: bool = True,
        max_local_span_words: int = 5,
    ) -> None:
        self.client = client or OllamaDiarizationLMClient()
        self.po = po
        self.max_chars_per_segment = max_chars_per_segment
        self.max_words_per_segment = max_words_per_segment
        self.use_locality_gating = use_locality_gating
        self.max_local_span_words = max_local_span_words

    @classmethod
    def _enforce_no_ground_truth_kwargs(cls, kwargs: Dict[str, Any]) -> None:
        forbidden = cls.FORBIDDEN_GT_KEYS.intersection(kwargs.keys())
        if forbidden:
            raise TypeError(
                f"Ground-truth isolation violation: ConvoFuseGenerativeRefiner forbids "
                f"ground-truth/reference parameters {sorted(forbidden)}"
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

    def build_segments(
        self,
        words: Sequence[ConvoFuseWord],
        speaker_mapper: SpeakerIdentityMapper,
        recording_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Partitions `words` for a single recording into bounded DiarizationLM prompt segments.
        - Maps known speakers `SPEAKER_00..` to `<speaker:1>..<speaker:K>`.
        - Maps `UNK` to `<speaker:0>` so `UNK` is never conflated with `<speaker:1>`.
        """
        if not words:
            return []

        segments: List[Dict[str, Any]] = []
        cur_words: List[ConvoFuseWord] = []
        cur_tokens: List[str] = []
        cur_chars = 0
        seg_idx = 0
        prev_tag = ""

        for w in words:
            if w.baseline_speaker_id == "UNK":
                spk_tag = "0"
            else:
                spk_tag = speaker_mapper.canonical_to_dlm_tag.get(w.baseline_speaker_id, "0")

            piece = ""
            if spk_tag != prev_tag:
                piece += f"{self.po.speaker_prefix}{spk_tag}{self.po.speaker_suffix} "
                prev_tag = spk_tag
            piece += f"{w.word} "

            if (
                cur_words
                and (
                    cur_chars + len(piece) > self.max_chars_per_segment
                    or len(cur_words) >= self.max_words_per_segment
                )
            ):
                hyp_str = "".join(cur_tokens).strip()
                full_prompt = f"{self.po.prompt_prefix}{hyp_str}{self.po.prompt_suffix}"
                segments.append(
                    {
                        "segment_id": f"{recording_id}__seg_{seg_idx:03d}",
                        "segment_index": seg_idx,
                        "recording_id": recording_id,
                        "words": list(cur_words),
                        "hypothesis_text": hyp_str,
                        "full_prompt": full_prompt,
                    }
                )
                seg_idx += 1
                cur_words = []
                cur_tokens = []
                cur_chars = 0
                piece = f"{self.po.speaker_prefix}{spk_tag}{self.po.speaker_suffix} {w.word} "
                prev_tag = spk_tag

            cur_words.append(w)
            cur_tokens.append(piece)
            cur_chars += len(piece)

        if cur_words:
            hyp_str = "".join(cur_tokens).strip()
            full_prompt = f"{self.po.prompt_prefix}{hyp_str}{self.po.prompt_suffix}"
            segments.append(
                {
                    "segment_id": f"{recording_id}__seg_{seg_idx:03d}",
                    "segment_index": seg_idx,
                    "recording_id": recording_id,
                    "words": list(cur_words),
                    "hypothesis_text": hyp_str,
                    "full_prompt": full_prompt,
                }
            )

        return segments

    def reconcile_segment(
        self,
        segment: Dict[str, Any],
        speaker_mapper: SpeakerIdentityMapper,
        completion_data: Dict[str, Any],
    ) -> Tuple[List[ConvoFuseWord], SegmentRefinementTelemetry]:
        """
        Reconciles a single segment's DiarizationLM completion onto immutable `ConvoFuseWord` items
        enforcing all Part 4.4 rules:
        - UNK must remain UNK.
        - Out-of-inventory tags rejected (`REJECTED_OUT_OF_INVENTORY_TAG`).
        - Missing tags rejected (`REJECTED_MISSING_TAG`).
        - Malformed tags rejected (`REJECTED_MALFORMED_TAG`).
        - Truncated completions rejected (`REJECTED_TRUNCATED_COMPLETION`).
        - Incomplete transcript alignments rejected (`REJECTED_INCOMPLETE_ALIGNMENT`).
        - Simultaneous overlap speaker changes rejected (`REJECTED_UNSUPPORTED_OVERLAP_CORRECTION`).
        - Wholesale span flips (> max_local_span_words) abstained via LG-TPST (`ABSTAINED_LOCALITY_GATE`).
        """
        seg_words: List[ConvoFuseWord] = segment["words"]
        seg_id: str = segment["segment_id"]
        rec_id: str = segment["recording_id"]

        raw_completion = str(completion_data.get("raw_completion", ""))
        done = bool(completion_data.get("done", True))
        done_reason = str(completion_data.get("done_reason", "stop"))
        is_truncated = (not done) or (done_reason.lower() in ("length", "max_tokens", "truncated"))

        cleaned = truncate_suffix_and_tailing_text(raw_completion, self.po.completion_suffix)
        cleaned = truncate_suffix_and_tailing_text(cleaned, "<turn|>").strip()

        hyp_word_strings = [w.word for w in seg_words]
        llm_words, llm_tags = extract_text_and_spk_strict(cleaned, po=self.po)
        aligned_tags = align_completion_tags_to_hypothesis(hyp_word_strings, llm_words, llm_tags)

        # First pass: resolve each word's raw proposed canonical speaker or rejection reason
        resolved_proposals: List[Tuple[Optional[str], Optional[str]]] = []
        for w, tag_opt in zip(seg_words, aligned_tags):
            if is_truncated:
                resolved_proposals.append((None, RejectionOrAbstentionReason.REJECTED_TRUNCATED_COMPLETION.value))
                continue
            if tag_opt is None:
                resolved_proposals.append((None, RejectionOrAbstentionReason.REJECTED_INCOMPLETE_ALIGNMENT.value))
                continue
            # If baseline word is UNK, UNK must remain UNK (Part 2.3 & Part 4.4)
            if w.baseline_speaker_id == "UNK":
                if tag_opt == "0":
                    resolved_proposals.append(("UNK", RejectionOrAbstentionReason.ABSTAINED_PRESERVE_UNK.value))
                else:
                    resolved_proposals.append((None, RejectionOrAbstentionReason.REJECTED_UNKNOWN_SPEAKER_MAPPING.value))
                continue

            prop_spk, err_reason = speaker_mapper.resolve_dlm_tag(tag_opt)
            if err_reason is not None:
                resolved_proposals.append((None, err_reason))
            else:
                resolved_proposals.append((prop_spk, None))

        # Second pass: identify contiguous spans of valid speaker change proposals for LG-TPST
        n = len(seg_words)
        is_valid_change = [
            (prop_spk is not None and err_r is None and prop_spk != seg_words[i].baseline_speaker_id)
            for i, (prop_spk, err_r) in enumerate(resolved_proposals)
        ]
        exceeds_locality_span = [False] * n
        if self.use_locality_gating:
            i = 0
            while i < n:
                if is_valid_change[i]:
                    j = i
                    while j < n and is_valid_change[j]:
                        j += 1
                    span_len = j - i
                    if span_len > self.max_local_span_words:
                        for k in range(i, j):
                            exceeds_locality_span[k] = True
                    i = j
                else:
                    i += 1

        refined_out: List[ConvoFuseWord] = []
        accepted_count = 0
        rejected_count = 0
        abstained_count = 0
        unchanged_count = 0
        reason_counts: Dict[str, int] = {}

        def _inc_reason(r: str) -> None:
            reason_counts[r] = reason_counts.get(r, 0) + 1

        for idx, (w, (prop_spk, err_reason)) in enumerate(zip(seg_words, resolved_proposals)):
            base_spk = w.baseline_speaker_id
            new_prov = WordProvenance(
                attribution_method=w.provenance.attribution_method if w.provenance else "max_overlap_with_0.5s_midpoint_collar",
                attribution_status=w.provenance.attribution_status if w.provenance else "SINGLE_SPEAKER_OVERLAP",
                nearest_segment_distance_sec=w.provenance.nearest_segment_distance_sec if w.provenance else None,
                raw_rttm_speaker_id=w.provenance.raw_rttm_speaker_id if w.provenance else base_spk,
                diarization_backend=w.provenance.diarization_backend if w.provenance else "ConvoFuse-DualEncoder",
                diarization_condition=w.provenance.diarization_condition if w.provenance else "default",
                context_segment_id=seg_id,
                refinement_model=self.client.model_identifier,
                execution_mode=w.provenance.execution_mode if w.provenance else "FROZEN_INPUT_REPLAY",
            )

            # Case 1: UNK preserved without modification attempt
            if err_reason == RejectionOrAbstentionReason.ABSTAINED_PRESERVE_UNK.value:
                abstained_count += 1
                _inc_reason(err_reason)
                refined_out.append(
                    ConvoFuseWord(
                        word_index=w.word_index,
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        baseline_speaker_id=base_spk,
                        refined_speaker_id="UNK",
                        asr_confidence=w.asr_confidence,
                        calibrated_confidence=w.calibrated_confidence,
                        alignment_score=w.alignment_score,
                        alignment_status=w.alignment_status,
                        overlap_metadata=w.overlap_metadata,
                        provenance=new_prov,
                        refinement_decision=RefinementDecision.ABSTAINED.value,
                        reconciliation_reason=err_reason,
                        proposed_speaker_id="UNK",
                        reasoning_note="UNK speaker preserved per ConvoFuse UNK invariance contract.",
                    )
                )
                continue

            # Case 2: Explicit validation rejection (malformed, missing, out-of-inventory, truncated, incomplete alignment, UNK mapping)
            if err_reason is not None:
                rejected_count += 1
                _inc_reason(err_reason)
                refined_out.append(
                    ConvoFuseWord(
                        word_index=w.word_index,
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        baseline_speaker_id=base_spk,
                        refined_speaker_id=base_spk,
                        asr_confidence=w.asr_confidence,
                        calibrated_confidence=w.calibrated_confidence,
                        alignment_score=w.alignment_score,
                        alignment_status=w.alignment_status,
                        overlap_metadata=w.overlap_metadata,
                        provenance=new_prov,
                        refinement_decision=RefinementDecision.REJECTED_PROPOSAL.value,
                        reconciliation_reason=err_reason,
                        proposed_speaker_id=str(aligned_tags[idx]) if aligned_tags[idx] is not None else None,
                        reasoning_note=f"Rejected LLM proposal ({err_reason}); baseline speaker {base_spk} preserved.",
                    )
                )
                continue

            # Case 3: Proposed speaker equals baseline speaker -> UNCHANGED
            if prop_spk == base_spk:
                unchanged_count += 1
                reason_str = RejectionOrAbstentionReason.UNCHANGED_BASELINE_CONFIRMED.value
                _inc_reason(reason_str)
                refined_out.append(
                    ConvoFuseWord(
                        word_index=w.word_index,
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        baseline_speaker_id=base_spk,
                        refined_speaker_id=base_spk,
                        asr_confidence=w.asr_confidence,
                        calibrated_confidence=w.calibrated_confidence,
                        alignment_score=w.alignment_score,
                        alignment_status=w.alignment_status,
                        overlap_metadata=w.overlap_metadata,
                        provenance=new_prov,
                        refinement_decision=RefinementDecision.UNCHANGED.value,
                        reconciliation_reason=reason_str,
                        proposed_speaker_id=prop_spk,
                        reasoning_note="DiarizationLM-Gemma confirmed baseline speaker assignment.",
                    )
                )
                continue

            # Case 4: Proposed speaker differs from baseline on simultaneous multi-speaker overlap -> REJECTED
            if w.overlap_metadata.is_overlapping_speech:
                rejected_count += 1
                reason_str = RejectionOrAbstentionReason.REJECTED_UNSUPPORTED_OVERLAP_CORRECTION.value
                _inc_reason(reason_str)
                refined_out.append(
                    ConvoFuseWord(
                        word_index=w.word_index,
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        baseline_speaker_id=base_spk,
                        refined_speaker_id=base_spk,
                        asr_confidence=w.asr_confidence,
                        calibrated_confidence=w.calibrated_confidence,
                        alignment_score=w.alignment_score,
                        alignment_status=w.alignment_status,
                        overlap_metadata=w.overlap_metadata,
                        provenance=new_prov,
                        refinement_decision=RefinementDecision.REJECTED_PROPOSAL.value,
                        reconciliation_reason=reason_str,
                        proposed_speaker_id=prop_spk,
                        reasoning_note=(
                            "Word lies in a simultaneous multi-speaker overlap interval; lexical LLM "
                            "reassignment rejected to avoid unsupported simultaneous-overlap claim."
                        ),
                    )
                )
                continue

            # Case 5: Proposed change span exceeds locality gate -> ABSTAINED
            if exceeds_locality_span[idx]:
                abstained_count += 1
                reason_str = RejectionOrAbstentionReason.ABSTAINED_LOCALITY_GATE.value
                _inc_reason(reason_str)
                refined_out.append(
                    ConvoFuseWord(
                        word_index=w.word_index,
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        baseline_speaker_id=base_spk,
                        refined_speaker_id=base_spk,
                        asr_confidence=w.asr_confidence,
                        calibrated_confidence=w.calibrated_confidence,
                        alignment_score=w.alignment_score,
                        alignment_status=w.alignment_status,
                        overlap_metadata=w.overlap_metadata,
                        provenance=new_prov,
                        refinement_decision=RefinementDecision.ABSTAINED.value,
                        reconciliation_reason=reason_str,
                        proposed_speaker_id=prop_spk,
                        reasoning_note=(
                            f"Proposed speaker change span exceeds locality gate ({self.max_local_span_words} words); "
                            f"abstained to preserve acoustic turn assignment."
                        ),
                    )
                )
                continue

            # Case 6: Valid local turn-boundary speaker correction -> ACCEPTED_CORRECTION
            accepted_count += 1
            reason_str = RejectionOrAbstentionReason.ACCEPTED_VALIDATED_TRANSFER.value
            _inc_reason(reason_str)
            refined_out.append(
                ConvoFuseWord(
                    word_index=w.word_index,
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    baseline_speaker_id=base_spk,
                    refined_speaker_id=prop_spk or base_spk,
                    asr_confidence=w.asr_confidence,
                    calibrated_confidence=w.calibrated_confidence,
                    alignment_score=w.alignment_score,
                    alignment_status=w.alignment_status,
                    overlap_metadata=w.overlap_metadata,
                    provenance=new_prov,
                    refinement_decision=RefinementDecision.ACCEPTED_CORRECTION.value,
                    reconciliation_reason=reason_str,
                    proposed_speaker_id=prop_spk,
                    reasoning_note=(
                        f"Accepted TPST local boundary transfer from {base_spk} to {prop_spk}."
                    ),
                )
            )

        telemetry = SegmentRefinementTelemetry(
            segment_id=seg_id,
            recording_id=rec_id,
            start_word_index=seg_words[0].word_index if seg_words else 0,
            end_word_index=seg_words[-1].word_index if seg_words else 0,
            word_count=len(seg_words),
            model_identifier=self.client.model_identifier,
            ollama_model_name=self.client.ollama_model_name,
            quantization=self.client.quantization,
            provider_version=self.client.get_provider_version(),
            prompt_configuration=dict(completion_data.get("prompt_configuration", {})),
            prompt_tokens=int(completion_data.get("prompt_tokens", 0)),
            completion_tokens=int(completion_data.get("completion_tokens", 0)),
            done=done,
            done_reason=done_reason,
            runtime_sec=float(completion_data.get("runtime_sec", 0.0)),
            prompt_text=segment["full_prompt"],
            raw_completion=raw_completion,
            cleaned_completion=cleaned,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            abstained_count=abstained_count,
            unchanged_count=unchanged_count,
            rejection_reasons_breakdown=reason_counts,
        )
        return refined_out, telemetry

    def refine_words(
        self,
        words: Sequence[ConvoFuseWord],
        speaker_mapper: SpeakerIdentityMapper,
        recording_id: str,
        raw_log_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any,
    ) -> GenerativeRefinementResult:
        """
        Executes ground-truth-isolated DiarizationLM-Gemma inference and TPST reconciliation.
        Raises TypeError if any reference/ground-truth keyword argument is supplied.
        """
        self._enforce_no_ground_truth_kwargs(kwargs)

        segments = self.build_segments(words, speaker_mapper, recording_id)
        if raw_log_dir is not None:
            Path(raw_log_dir).mkdir(parents=True, exist_ok=True)

        all_refined: List[ConvoFuseWord] = []
        telemetries: List[SegmentRefinementTelemetry] = []
        t0_all = time.perf_counter()

        for idx, seg in enumerate(segments):
            if progress_cb:
                frac = (idx / max(1, len(segments)))
                progress_cb(
                    0.75 + 0.12 * frac,
                    f"GENERATIVE_REFINEMENT: DiarizationLM-Gemma segment {idx + 1}/{len(segments)}",
                )

            try:
                comp_data = self.client.generate_completion(seg["full_prompt"])
            except Exception as exc:
                # Safe abstention fallback if provider errors on a segment
                comp_data = {
                    "raw_completion": "",
                    "done": False,
                    "done_reason": f"provider_error: {exc}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "runtime_sec": 0.0,
                    "prompt_configuration": {
                        "temperature": self.client.temperature,
                        "num_predict": self.client.num_predict,
                    },
                }

            if raw_log_dir is not None:
                seg_file = Path(raw_log_dir) / f"{seg['segment_id']}__raw.json"
                seg_file.write_text(
                    json.dumps(
                        {
                            "segment_id": seg["segment_id"],
                            "recording_id": recording_id,
                            "prompt": seg["full_prompt"],
                            "completion_data": {
                                k: v for k, v in comp_data.items() if k != "full_api_response"
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            ref_seg_words, tele = self.reconcile_segment(seg, speaker_mapper, comp_data)
            all_refined.extend(ref_seg_words)
            telemetries.append(tele)

        total_runtime = time.perf_counter() - t0_all

        # Verify strict post-condition invariants (word text, ordering, timestamps, count, overlap metadata)
        self.verify_invariants(words, all_refined)

        total_prompt_tok = sum(t.prompt_tokens for t in telemetries)
        total_comp_tok = sum(t.completion_tokens for t in telemetries)
        total_accepted = sum(t.accepted_count for t in telemetries)
        total_rejected = sum(t.rejected_count for t in telemetries)
        total_abstained = sum(t.abstained_count for t in telemetries)
        total_unchanged = sum(t.unchanged_count for t in telemetries)

        agg_reasons: Dict[str, int] = {}
        for t in telemetries:
            for r_key, r_cnt in t.rejection_reasons_breakdown.items():
                agg_reasons[r_key] = agg_reasons.get(r_key, 0) + r_cnt

        return GenerativeRefinementResult(
            recording_id=recording_id,
            model_identifier=self.client.model_identifier,
            ollama_model_name=self.client.ollama_model_name,
            quantization=self.client.quantization,
            provider_version=self.client.get_provider_version(),
            total_runtime_sec=total_runtime,
            total_prompt_tokens=total_prompt_tok,
            total_completion_tokens=total_comp_tok,
            segment_count=len(segments),
            refined_words=all_refined,
            segment_telemetries=telemetries,
            summary_counts={
                "total_words": len(all_refined),
                "accepted_corrections": total_accepted,
                "rejected_proposals": total_rejected,
                "abstained_words": total_abstained,
                "unchanged_words": total_unchanged,
                "reasons_breakdown": dict(sorted(agg_reasons.items())),
            },
        )

    @staticmethod
    def verify_invariants(
        original_words: Sequence[ConvoFuseWord],
        refined_words: Sequence[ConvoFuseWord],
    ) -> None:
        """
        Enforces strict TPST post-conditions (Part 4.4):
        Word count, word_index, word text, start/end timestamps, baseline_speaker_id,
        overlap metadata, and UNK status must remain intact.
        """
        if len(original_words) != len(refined_words):
            raise RuntimeError(
                f"TPST invariant violation: word count changed ({len(original_words)} != {len(refined_words)})"
            )
        for orig, ref in zip(original_words, refined_words):
            if orig.word_index != ref.word_index:
                raise RuntimeError(
                    f"TPST invariant violation: word_index changed ({orig.word_index} != {ref.word_index})"
                )
            if orig.word != ref.word:
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: word text mutated ({orig.word!r} != {ref.word!r})"
                )
            if orig.start != ref.start or orig.end != ref.end:
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: timestamps mutated"
                )
            if orig.baseline_speaker_id != ref.baseline_speaker_id:
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: baseline_speaker_id mutated"
                )
            if orig.baseline_speaker_id == "UNK" and ref.refined_speaker_id != "UNK":
                raise RuntimeError(
                    f"TPST UNK invariant violation at index {orig.word_index}: UNK mutated to {ref.refined_speaker_id!r}"
                )
            if orig.overlap_metadata.to_dict() != ref.overlap_metadata.to_dict():
                raise RuntimeError(
                    f"TPST invariant violation at index {orig.word_index}: overlap_metadata mutated"
                )
