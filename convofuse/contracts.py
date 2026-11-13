"""
ConvoFuse — Canonical Architecture and Data Contracts (Part 2).

Explicitly distinguishes three separate operations:
1. Task-specific dual-path audio processing (DFN 16kHz mono for TitaNet-Large 192-D;
   raw 16kHz mono for WeSpeaker ResNet293-LM 256-D and overlap detection; EBU R128
   normalized 16kHz mono for Faster-Whisper Large-v3 ASR and Wav2Vec2 CTC alignment).
2. Dual-encoder speaker affinity fusion (independent cosine affinities + off-diagonal
   Z-score normalization + multiscale equal/reliability-gated fusion + NME-SC spectral
   clustering without ground-truth speaker count).
3. Temporal fusion of aligned ASR words with diarization intervals (max_overlap_speaker
   with 0.5s midpoint tolerance + explicit UNK preservation + overlap metadata).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class ConvoFuseExecutionMode(str, Enum):
    """Distinguishes full raw-audio processing from frozen-input replay/integration mode."""
    FULL_PROCESSING = "FULL_PROCESSING"
    FROZEN_INPUT_REPLAY = "FROZEN_INPUT_REPLAY"


class ConvoFuseJobStage(str, Enum):
    """Canonical 15-state job lifecycle required by Part 2.4."""
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    AUDIO_EXTRACTION = "AUDIO_EXTRACTION"
    VAD = "VAD"
    ASR = "ASR"
    WORD_ALIGNMENT = "WORD_ALIGNMENT"
    SPEAKER_REPRESENTATION = "SPEAKER_REPRESENTATION"
    AFFINITY_FUSION = "AFFINITY_FUSION"
    CLUSTERING = "CLUSTERING"
    WORD_SPEAKER_ASSIGNMENT = "WORD_SPEAKER_ASSIGNMENT"
    GENERATIVE_REFINEMENT = "GENERATIVE_REFINEMENT"
    TRANSCRIPT_RECONSTRUCTION = "TRANSCRIPT_RECONSTRUCTION"
    EVALUATION = "EVALUATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RefinementDecision(str, Enum):
    """Four explicit word-level refinement outcomes required by Part 4.4."""
    UNCHANGED = "UNCHANGED"
    ACCEPTED_CORRECTION = "ACCEPTED_CORRECTION"
    REJECTED_PROPOSAL = "REJECTED_PROPOSAL"
    ABSTAINED = "ABSTAINED"


class RejectionOrAbstentionReason(str, Enum):
    """Explicit deterministic reasons recorded during TPST reconciliation (Part 4.4)."""
    UNCHANGED_BASELINE_CONFIRMED = "UNCHANGED_BASELINE_CONFIRMED"
    ACCEPTED_VALIDATED_TRANSFER = "ACCEPTED_VALIDATED_TRANSFER"
    ABSTAINED_BY_MODEL = "ABSTAINED_BY_MODEL"
    ABSTAINED_PRESERVE_UNK = "ABSTAINED_PRESERVE_UNK"
    ABSTAINED_LOCALITY_GATE = "ABSTAINED_LOCALITY_GATE"
    REJECTED_UNKNOWN_SPEAKER_MAPPING = "REJECTED_UNKNOWN_SPEAKER_MAPPING"
    REJECTED_OUT_OF_INVENTORY_TAG = "REJECTED_OUT_OF_INVENTORY_TAG"
    REJECTED_MISSING_TAG = "REJECTED_MISSING_TAG"
    REJECTED_MALFORMED_TAG = "REJECTED_MALFORMED_TAG"
    REJECTED_TRUNCATED_COMPLETION = "REJECTED_TRUNCATED_COMPLETION"
    REJECTED_INCOMPLETE_ALIGNMENT = "REJECTED_INCOMPLETE_ALIGNMENT"
    REJECTED_UNSUPPORTED_OVERLAP_CORRECTION = "REJECTED_UNSUPPORTED_OVERLAP_CORRECTION"
    REJECTED_WORD_OR_ORDER_MUTATION = "REJECTED_WORD_OR_ORDER_MUTATION"
    REJECTED_PROVIDER_ERROR = "REJECTED_PROVIDER_ERROR"


@dataclass(frozen=True)
class OverlapMetadata:
    """
    Overlap-aware event and posterior metadata for a single word (Part 1.3.F & Part 3.7).
    Preserves the distinction between detecting overlapping speech and attributing
    simultaneous speakers in a single-speaker-per-word schema.
    """
    is_overlapping_speech: bool = False
    overlapping_speakers: Tuple[str, ...] = field(default_factory=tuple)
    overlap_durations_sec: Dict[str, float] = field(default_factory=dict)
    frame_overlap_prob_mean: Optional[float] = None
    frame_overlap_prob_max: Optional[float] = None
    overlap_posterior_Dragging_source: Optional[str] = None
    simultaneous_attribution_supported: bool = False
    limitation_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_overlapping_speech": self.is_overlapping_speech,
            "overlapping_speakers": list(self.overlapping_speakers),
            "overlap_durations_sec": dict(self.overlap_durations_sec),
            "frame_overlap_prob_mean": self.frame_overlap_prob_mean,
            "frame_overlap_prob_max": self.frame_overlap_prob_max,
            "overlap_posterior_source": self.overlap_posterior_Dragging_source,
            "simultaneous_attribution_supported": self.simultaneous_attribution_supported,
            "limitation_note": self.limitation_note,
        }


@dataclass(frozen=True)
class WordProvenance:
    """Provenance record for a word across ASR, diarization, and Stage 6 refinement."""
    attribution_method: str
    attribution_status: str
    nearest_segment_distance_sec: Optional[float]
    raw_rttm_speaker_id: str
    diarization_backend: str
    diarization_condition: str
    context_segment_id: str
    refinement_model: str
    execution_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvoFuseWord:
    """
    Canonical word-level representation (Part 2.2).
    Original `word`, `start`, and `end` are strictly immutable during LLM refinement.
    """
    word_index: int
    word: str
    start: Optional[float]
    end: Optional[float]
    baseline_speaker_id: str
    refined_speaker_id: str
    asr_confidence: Optional[float] = None
    calibrated_confidence: Optional[float] = None
    alignment_score: Optional[float] = None
    alignment_status: str = "aligned"
    overlap_metadata: OverlapMetadata = field(default_factory=OverlapMetadata)
    provenance: Optional[WordProvenance] = None
    refinement_decision: str = RefinementDecision.UNCHANGED.value
    reconciliation_reason: str = RejectionOrAbstentionReason.UNCHANGED_BASELINE_CONFIRMED.value
    proposed_speaker_id: Optional[str] = None
    reasoning_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_index": self.word_index,
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "baseline_speaker_id": self.baseline_speaker_id,
            "refined_speaker_id": self.refined_speaker_id,
            "asr_confidence": self.asr_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "alignment_score": self.alignment_score,
            "alignment_status": self.alignment_status,
            "overlap_metadata": self.overlap_metadata.to_dict(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "refinement_decision": self.refinement_decision,
            "reconciliation_reason": self.reconciliation_reason,
            "proposed_speaker_id": self.proposed_speaker_id,
            "reasoning_note": self.reasoning_note,
        }


@dataclass(frozen=True)
class ConvoFuseTurn:
    """
    Chronological speaker turn reconstructed from refined word-level predictions (Part 5.2).
    """
    turn_id: str
    turn_index: int
    speaker_id: str
    start: float
    end: float
    duration: float
    text: str
    word_indices: Tuple[int, ...]
    word_count: int
    avg_confidence: float
    has_overlapping_speech: bool = False
    overlapping_speakers: Tuple[str, ...] = field(default_factory=tuple)
    accepted_refinement_count: int = 0
    rejected_refinement_count: int = 0
    abstained_refinement_count: int = 0
    unchanged_word_count: int = 0
    formatted_header: str = ""
    formatted_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "speaker_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "word_indices": list(self.word_indices),
            "word_count": self.word_count,
            "avg_confidence": self.avg_confidence,
            "has_overlapping_speech": self.has_overlapping_speech,
            "overlapping_speakers": list(self.overlapping_speakers),
            "accepted_refinement_count": self.accepted_refinement_count,
            "rejected_refinement_count": self.rejected_refinement_count,
            "abstained_refinement_count": self.abstained_refinement_count,
            "unchanged_word_count": self.unchanged_word_count,
            "formatted_header": self.formatted_header,
            "formatted_line": self.formatted_line,
        }


@dataclass(frozen=True)
class ConvoFuseSpeakerSummary:
    """
    Speaker index entry summarizing activity for a single speaker ID (Part 5.4).
    """
    speaker_id: str
    total_speaking_duration_sec: float
    turn_count: int
    word_count: int
    first_appearance_sec: Optional[float]
    last_appearance_sec: Optional[float]
    mean_word_confidence: float
    accepted_corrections_into_speaker: int = 0
    accepted_corrections_out_of_speaker: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessingStageRecord:
    """Audit record for a single pipeline lifecycle stage."""
    stage: str
    status: str  # "completed" | "replayed_from_frozen" | "skipped" | "failed" | "running"
    execution_mode: str
    progress: float
    started_at: str
    completed_at: str
    elapsed_sec: float
    detail: str
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConvoFuseTranscriptBundle:
    """
    Complete ConvoFuse output bundle containing all reconstructed views, manifests,
    and optional post-inference evaluation results.
    """
    recording_id: str
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int
    execution_mode: str
    diarization_condition: str
    diarization_backend: str
    refinement_model: str
    speaker_inventory: List[str]
    rttm_to_canonical_speaker_map: Dict[str, str]
    words: List[ConvoFuseWord]
    turns: List[ConvoFuseTurn]
    baseline_turns: List[ConvoFuseTurn]
    speaker_summaries: List[ConvoFuseSpeakerSummary]
    readable_transcript_txt: str
    baseline_vs_refined_comparison: Dict[str, Any]
    processing_manifest: Dict[str, Any]
    evaluation_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "filename": self.filename,
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "execution_mode": self.execution_mode,
            "diarization_condition": self.diarization_condition,
            "diarization_backend": self.diarization_backend,
            "refinement_model": self.refinement_model,
            "speaker_inventory": list(self.speaker_inventory),
            "rttm_to_canonical_speaker_map": dict(self.rttm_to_canonical_speaker_map),
            "turn_count": len(self.turns),
            "word_count": len(self.words),
            "speaker_summaries": [s.to_dict() for s in self.speaker_summaries],
            "turns": [t.to_dict() for t in self.turns],
            "words": [w.to_dict() for w in self.words],
            "baseline_vs_refined_comparison": self.baseline_vs_refined_comparison,
            "processing_manifest": self.processing_manifest,
            "evaluation_report": self.evaluation_report,
        }


class SpeakerIdentityMapper:
    """
    Deterministic speaker identity contract (Part 2.3):
    1. Maps RTTM speaker labels (e.g. `speaker_0`, `speaker_1`, `SPEAKER_00`) to stable
       canonical application speaker IDs (`SPEAKER_00`, `SPEAKER_01`, ...).
    2. Maps `"UNK"` and `"SPEAKER_UNASSIGNED"` strictly to `"UNK"`.
    3. Maps canonical speaker IDs to 1-indexed DiarizationLM tags (`<speaker:1>`, ..., `<speaker:K>`)
       while keeping `"UNK"` isolated so `"UNK"` is never silently mapped to a valid speaker.
    4. Validates proposed tags strictly against the recording's speaker inventory.
    """

    UNK_LABELS = {"", "UNK", "SPEAKER_UNASSIGNED", "NONE", "NULL"}

    def __init__(self, raw_rttm_speakers: Sequence[str]) -> None:
        unique_raw = sorted(
            {
                str(s).strip()
                for s in raw_rttm_speakers
                if s is not None and str(s).strip().upper() not in self.UNK_LABELS
            },
            key=self._natural_sort_key,
        )
        self.rttm_to_canonical: Dict[str, str] = {}
        self.canonical_to_rttm: Dict[str, str] = {}
        self.canonical_inventory: List[str] = []

        for idx, raw_spk in enumerate(unique_raw):
            canonical_id = f"SPEAKER_{idx:02d}"
            self.rttm_to_canonical[raw_spk] = canonical_id
            # Also map canonical_id to itself for idempotency
            self.rttm_to_canonical[canonical_id] = canonical_id
            self.canonical_to_rttm[canonical_id] = raw_spk
            self.canonical_inventory.append(canonical_id)

        # Deterministic 1-based tags for DiarizationLM (<speaker:1> .. <speaker:K>)
        self.canonical_to_dlm_tag: Dict[str, str] = {}
        self.dlm_tag_to_canonical: Dict[str, str] = {}
        for idx, canonical_id in enumerate(self.canonical_inventory, start=1):
            tag_num = str(idx)
            self.canonical_to_dlm_tag[canonical_id] = tag_num
            self.dlm_tag_to_canonical[tag_num] = canonical_id

    def set_dlm_appearance_order(self, appeared_canonical_speakers: Sequence[str]) -> None:
        """
        Orders DiarizationLM `<speaker:1>..<speaker:K>` tags deterministically by order of
        first appearance in the recording (with any remaining inventory speakers appended in
        canonical order), matching the DiarizationLM training convention while keeping
        `UNK` strictly excluded from `1..K`.
        """
        ordered: List[str] = []
        for spk in appeared_canonical_speakers:
            if spk in self.canonical_inventory and spk not in ordered:
                ordered.append(spk)
        for spk in self.canonical_inventory:
            if spk not in ordered:
                ordered.append(spk)
        self.canonical_to_dlm_tag.clear()
        self.dlm_tag_to_canonical.clear()
        for idx, canonical_id in enumerate(ordered, start=1):
            tag_num = str(idx)
            self.canonical_to_dlm_tag[canonical_id] = tag_num
            self.dlm_tag_to_canonical[tag_num] = canonical_id

    @staticmethod
    def _natural_sort_key(label: str) -> Tuple[int, str]:
        m = re.search(r"(\d+)$", label)
        if m:
            return (int(m.group(1)), label)
        return (999999, label)

    def to_canonical(self, raw_speaker: Optional[str]) -> str:
        """Convert raw RTTM or intermediate speaker label to canonical ID or 'UNK'."""
        if raw_speaker is None:
            return "UNK"
        s = str(raw_speaker).strip()
        if not s or s.upper() in self.UNK_LABELS:
            return "UNK"
        if s in self.rttm_to_canonical:
            return self.rttm_to_canonical[s]
        return "UNK"

    def to_rttm(self, canonical_speaker: Optional[str]) -> str:
        """Convert canonical speaker ID back to original RTTM speaker ID (or 'UNK')."""
        if canonical_speaker is None:
            return "UNK"
        s = str(canonical_speaker).strip()
        if not s or s.upper() in self.UNK_LABELS:
            return "UNK"
        return self.canonical_to_rttm.get(s, s)

    def is_valid_canonical_speaker(self, speaker_id: Optional[str]) -> bool:
        """Return True iff speaker_id is in the known non-UNK speaker inventory."""
        if not speaker_id or str(speaker_id).strip().upper() in self.UNK_LABELS:
            return False
        return str(speaker_id).strip() in self.canonical_to_dlm_tag

    def resolve_dlm_tag(self, tag_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Validate a DiarizationLM speaker tag string.
        Returns (canonical_speaker_id, error_reason).
        Never silently maps an invalid or out-of-inventory tag to a valid speaker.
        """
        if not tag_str or tag_str == "MISSING_TAG":
            return None, RejectionOrAbstentionReason.REJECTED_MISSING_TAG.value
        if tag_str.startswith("MALFORMED_TAG"):
            return None, RejectionOrAbstentionReason.REJECTED_MALFORMED_TAG.value
        if tag_str == "UNK" or tag_str == "0":
            return None, RejectionOrAbstentionReason.REJECTED_UNKNOWN_SPEAKER_MAPPING.value
        if not re.fullmatch(r"\d+", tag_str):
            return None, RejectionOrAbstentionReason.REJECTED_MALFORMED_TAG.value
        if tag_str not in self.dlm_tag_to_canonical:
            return None, RejectionOrAbstentionReason.REJECTED_OUT_OF_INVENTORY_TAG.value
        return self.dlm_tag_to_canonical[tag_str], None
