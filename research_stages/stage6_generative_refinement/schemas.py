"""
Stage 6 — Phase 2: Data Structures and Schemas
Defines immutable word-level items, context-window schemas, LLM request/response
contracts, TPST reconciliation statuses, and final output records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AttributionStatus(str, Enum):
    """Describes how an ASR word was initially associated with a diarization speaker."""
    SINGLE_SPEAKER_OVERLAP = "SINGLE_SPEAKER_OVERLAP"
    MULTI_SPEAKER_OVERLAP = "MULTI_SPEAKER_OVERLAP"
    MIDPOINT_TOLERANCE_FALLBACK = "MIDPOINT_TOLERANCE_FALLBACK"
    UNASSIGNED_UNK = "UNASSIGNED_UNK"
    MISSING_TIMESTAMPS = "MISSING_TIMESTAMPS"


class ConfidenceStatus(str, Enum):
    """Normalized confidence / uncertainty levels for speaker refinement."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"
    BASELINE_PRESERVED = "BASELINE_PRESERVED"


class ReconciliationReason(str, Enum):
    """Deterministic audit reason recorded for every word during TPST reconciliation."""
    UNCHANGED_ORIGINAL_CONFIRMED = "UNCHANGED_ORIGINAL_CONFIRMED"
    ACCEPTED_VALIDATED_CORRECTION = "ACCEPTED_VALIDATED_CORRECTION"
    ABSTAINED_BY_LLM = "ABSTAINED_BY_LLM"
    REJECTED_MALFORMED_JSON = "REJECTED_MALFORMED_JSON"
    REJECTED_SCHEMA_VIOLATION = "REJECTED_SCHEMA_VIOLATION"
    REJECTED_INVALID_SPEAKER_ID = "REJECTED_INVALID_SPEAKER_ID"
    REJECTED_MISSING_WORDS = "REJECTED_MISSING_WORDS"
    REJECTED_EXTRA_WORDS = "REJECTED_EXTRA_WORDS"
    REJECTED_REORDERED_OR_MUTATED_WORDS = "REJECTED_REORDERED_OR_MUTATED_WORDS"
    REJECTED_WORD_TEXT_MISMATCH = "REJECTED_WORD_TEXT_MISMATCH"
    REJECTED_LOW_CONFIDENCE = "REJECTED_LOW_CONFIDENCE"
    NO_LLM_PROPOSAL = "NO_LLM_PROPOSAL"


@dataclass(frozen=True)
class OrchestratedWord:
    """
    Immutable word record produced by Step A (Word-level orchestration).
    Preserves original ASR text, timestamps, confidence, and baseline speaker assignment.
    """
    word_index: int
    word: str
    start: Optional[float]
    end: Optional[float]
    asr_confidence: Optional[float]
    original_speaker_id: str
    attribution_status: str
    is_overlapping_speech: bool = False
    overlapping_speakers: Tuple[str, ...] = field(default_factory=tuple)
    overlap_durations: Dict[str, float] = field(default_factory=dict)
    nearest_segment_distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overlapping_speakers"] = list(self.overlapping_speakers)
        return d


@dataclass(frozen=True)
class ConversationalTurnSummary:
    """Compact turn representation for surrounding context in a window."""
    turn_index: int
    speaker_id: str
    start: Optional[float]
    end: Optional[float]
    text: str
    word_count: int
    has_overlap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextWindow:
    """
    Bounded conversational window constructed by Step B (Context-window builder).
    Contains the target core words, left/right context words, surrounding turn summaries,
    valid speaker IDs in the recording, and overlap metadata.
    """
    window_id: str
    recording_id: str
    condition_id: str
    window_index: int
    core_word_indices: Tuple[int, ...]
    context_word_indices: Tuple[int, ...]
    words: Tuple[OrchestratedWord, ...]
    surrounding_turns: Tuple[ConversationalTurnSummary, ...]
    available_speaker_ids: Tuple[str, ...]
    overlap_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "recording_id": self.recording_id,
            "condition_id": self.condition_id,
            "window_index": self.window_index,
            "core_word_indices": list(self.core_word_indices),
            "context_word_indices": list(self.context_word_indices),
            "words": [w.to_dict() for w in self.words],
            "surrounding_turns": [t.to_dict() for t in self.surrounding_turns],
            "available_speaker_ids": list(self.available_speaker_ids),
            "overlap_info": self.overlap_info,
        }


@dataclass(frozen=True)
class LLMRefinementRequest:
    """Provider-independent structured LLM request payload."""
    window_id: str
    recording_id: str
    condition_id: str
    available_speaker_ids: Tuple[str, ...]
    words_payload: Tuple[Dict[str, Any], ...]
    surrounding_turns_payload: Tuple[Dict[str, Any], ...]
    overlap_info: Dict[str, Any]
    system_prompt: str
    user_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "recording_id": self.recording_id,
            "condition_id": self.condition_id,
            "available_speaker_ids": list(self.available_speaker_ids),
            "words_payload": list(self.words_payload),
            "surrounding_turns_payload": list(self.surrounding_turns_payload),
            "overlap_info": self.overlap_info,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }


@dataclass(frozen=True)
class LLMWordProposal:
    """Single word decision parsed from an LLM response prior to TPST transfer."""
    word_index: int
    word: str
    proposed_speaker_id: Optional[str]
    abstain: bool = False
    confidence: str = ConfidenceStatus.MEDIUM.value
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefinedWordOutput:
    """
    Final word-level output record after Step D (Constrained Parser & TPST Reconciliation).
    Guarantees original word and timestamps are preserved unmodified alongside both
    baseline and refined speaker assignments.
    """
    word_index: int
    word: str
    start: Optional[float]
    end: Optional[float]
    asr_confidence: Optional[float]
    original_speaker_id: str
    refined_speaker_id: str
    correction_proposed: bool
    correction_accepted: bool
    abstained: bool
    proposed_speaker_id: Optional[str]
    reconciliation_reason: str
    reasoning_summary: str
    confidence_status: str
    context_window_id: str
    attribution_status: str
    is_overlapping_speech: bool = False
    overlapping_speakers: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overlapping_speakers"] = list(self.overlapping_speakers)
        return d
