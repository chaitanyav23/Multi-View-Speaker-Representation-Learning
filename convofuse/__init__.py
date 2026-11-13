"""
ConvoFuse — Unified Conversational Intelligence Framework.

Evolves the Phase20 application with:
1. Audited task-specific dual-path audio processing (DFN TitaNet-Large 192-D + Raw Mono WeSpeaker ResNet293-LM 256-D).
2. Validated multiscale speaker affinity fusion and calibrated NME-SC spectral clustering.
3. Temporal word-to-speaker association with explicit overlap metadata.
4. Ground-truth-isolated DiarizationLM-Gemma generative speaker refinement with strict TPST reconciliation.
5. Complete chronological conversational transcript reconstruction, speaker summaries, and multi-format exports.
"""

from .contracts import (
    ConvoFuseExecutionMode,
    ConvoFuseJobStage,
    ConvoFuseSpeakerSummary,
    ConvoFuseTranscriptBundle,
    ConvoFuseTurn,
    ConvoFuseWord,
    OverlapMetadata,
    ProcessingStageRecord,
    RefinementDecision,
    RejectionOrAbstentionReason,
    SpeakerIdentityMapper,
    WordProvenance,
)
from .pipeline import ConvoFusePipeline

__all__ = [
    "ConvoFuseExecutionMode",
    "ConvoFuseJobStage",
    "ConvoFusePipeline",
    "ConvoFuseSpeakerSummary",
    "ConvoFuseTranscriptBundle",
    "ConvoFuseTurn",
    "ConvoFuseWord",
    "OverlapMetadata",
    "ProcessingStageRecord",
    "RefinementDecision",
    "RejectionOrAbstentionReason",
    "SpeakerIdentityMapper",
    "WordProvenance",
]
