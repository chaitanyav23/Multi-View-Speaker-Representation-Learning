"""Canonical Data Schemas for Speech Processing, Transcript Attribution, and Confidence Analytics."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ConfidenceFeatures(BaseModel):
    """Raw acoustic and model features utilized for confidence calibration."""
    acoustic_score: float = Field(..., description="Acoustic or token log-probability / likelihood")
    alignment_score: Optional[float] = Field(default=None, description="Wav2Vec2 CTC acoustic alignment score in [0.0, 1.0]")
    asr_confidence: Optional[float] = Field(default=None, description="Faster-Whisper sequence likelihood probability in [0.0, 1.0]")
    alignment_status: str = Field(default="aligned", description="aligned | unaligned | fallback")
    duration: float = Field(..., description="Word acoustic duration in seconds")
    speech_ratio: float = Field(1.0, description="Ratio of word duration to parent VAD chunk duration")
    overlap_detected: bool = Field(False, description="Flag indicating presence of overlapping speaker energy")
    energy_level: float = Field(1.0, description="Normalized RMS energy level of the word frame")


class AlignedWord(BaseModel):
    """A single recognized word with temporal bounds, speaker attribution, and confidence."""
    word: str = Field(..., description="Recognized word token")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    speaker: str = Field(default="SPEAKER_00", description="Diarized speaker label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score in [0.0, 1.0]")
    raw_confidence: float = Field(..., ge=0.0, le=1.0, description="Raw uncalibrated confidence score")
    is_calibrated: bool = Field(default=True, description="Whether calibration mapping was applied")
    alignment_status: str = Field(default="aligned", description="aligned | unaligned | fallback")
    alignment_score: Optional[float] = Field(default=None, description="Raw Wav2Vec2 CTC alignment score in [0.0, 1.0]")
    asr_confidence: Optional[float] = Field(default=None, description="Faster-Whisper sequence likelihood probability in [0.0, 1.0]")
    features: Optional[ConfidenceFeatures] = Field(default=None, description="Acoustic and model features")


class ConversationTurn(BaseModel):
    """Contiguous speech segment attributed to a single speaker."""
    turn_id: str = Field(..., description="Unique turn identifier")
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Combined text of all words in the turn")
    words: list[AlignedWord] = Field(default_factory=list, description="Contained word tokens")
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean confidence across words")


class CalibrationMetadata(BaseModel):
    """Provenance and metrics for the confidence calibration subsystem."""
    method: str = Field(default="isotonic_regression", description="Calibration technique used")
    version: str = Field(default="1.0.0", description="Calibration model version")
    is_calibrated: bool = Field(default=True, description="Whether scores are calibrated probabilities")
    brier_score: Optional[float] = Field(default=None, description="Empirical Brier Score")
    ece: Optional[float] = Field(default=None, description="Expected Calibration Error")


class TranscriptResult(BaseModel):
    """Complete canonical transcript, attribution, and confidence analysis result."""
    recording_id: str = Field(..., description="Unique recording identifier")
    filename: str = Field(..., description="Original filename")
    duration: float = Field(..., ge=0.0, description="Media duration in seconds")
    sample_rate: int = Field(default=16000, description="Sampling rate in Hz")
    channels: int = Field(default=1, description="Audio channel count")
    method: str = Field(default="Phase20-OracleMSDD-MidpointFusion", description="Pipeline method tag")
    calibration: CalibrationMetadata = Field(default_factory=CalibrationMetadata)
    speakers: list[str] = Field(default_factory=list, description="Unique detected speakers")
    segments: list[ConversationTurn] = Field(default_factory=list, description="Sequential speaker turns")
    words: list[AlignedWord] = Field(default_factory=list, description="All word tokens")
    summary: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


# API Data Transfer Objects (DTOs)

class MediaUploadResponse(BaseModel):
    recording_id: str
    filename: str
    size_bytes: int
    duration_seconds: float
    sha256: str
    status: str
    message: str


class TranscriptionJobRequest(BaseModel):
    recording_id: str
    use_mock: bool = Field(default=False, description="Deprecated; real Phase 20 pipeline is always executed")
    enable_denoising: bool = Field(default=False, description="Enable speech enhancement")
    language: str = Field(default="en", description="Spoken language code")
    initial_prompt: Optional[str] = Field(default=None, description="Domain vocabulary prompt")


class JobStatusResponse(BaseModel):
    job_id: str
    recording_id: str
    status: str = Field(..., description="queued | processing | completed | failed")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress ratio (0.0 to 1.0)")
    stage: str = Field(default="idle", description="Current pipeline stage")
    error: Optional[str] = None
    created_at: str
    updated_at: str


class UserFeedbackItem(BaseModel):
    feedback_id: Optional[str] = None
    recording_id: str
    turn_id: Optional[str] = None
    word_index: Optional[int] = None
    original_text: str
    corrected_text: str
    original_speaker: Optional[str] = None
    corrected_speaker: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str


# ---------------------------------------------------------------------------
# Stage 1.6.1: Typed Analytics & Plugin Framework Schemas
# ---------------------------------------------------------------------------

class AnalyticsTierStats(BaseModel):
    count: int = Field(..., ge=0, description="Word count in this confidence tier")
    ratio: float = Field(..., ge=0.0, le=1.0, description="Proportion of total words")


class AnalyticsTiers(BaseModel):
    low: AnalyticsTierStats = Field(..., description="Low confidence words (<0.30)")
    mid: AnalyticsTierStats = Field(..., description="Medium confidence words (0.30 - 0.60)")
    high: AnalyticsTierStats = Field(..., description="High confidence words (>=0.60)")


class AnalyticsResponse(BaseModel):
    """Complete confidence analytics and calibration report."""
    recording_id: str
    filename: str
    total_words: int
    mean_confidence: float
    mean_raw_confidence: float
    calibration: CalibrationMetadata
    brier_score: float
    ece: float
    tiers: AnalyticsTiers
    reliability_diagram: list[dict[str, Any]]
    speakers_count: int
    duration: float


class PluginManifest(BaseModel):
    """Metadata describing a trusted plugin extension."""
    plugin_id: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$", description="Immutable alphanumeric plugin ID")
    name: str = Field(..., min_length=1, description="Human-readable plugin name")
    version: str = Field("1.0.0", description="Semantic version string")
    description: str = Field(..., description="Functional summary of the plugin")
    author: str = Field("CS698B Student Team", description="Author or team attribution")
    capabilities: list[str] = Field(default_factory=list, description="Capabilities (e.g. ['summarization', 'topic_modeling'])")
    input_requirements: list[str] = Field(default_factory=lambda: ["transcript"], description="Required input context keys")
    output_types: list[str] = Field(default_factory=lambda: ["json"], description="Supported output data types")
    api_version: str = Field("1.0.0", description="Plugin host API compatibility version")
    enabled: bool = Field(True, description="Whether the plugin is enabled in the host UI")


class PluginContextResponse(BaseModel):
    """Consolidated, typed recording and transcript context for plugins."""
    recording_id: str
    filename: str
    duration_seconds: float
    status: str = Field(..., description="ready | processing | untranscribed | failed")
    has_transcript: bool
    transcript: Optional[TranscriptResult] = None
    speakers: list[str] = Field(default_factory=list)
    total_words: int = 0
    mean_confidence: Optional[float] = None
    turns: list[ConversationTurn] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PluginRunRequest(BaseModel):
    """Request payload to initiate a plugin analysis."""
    recording_id: str = Field(..., min_length=1, description="Target recording ID")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Plugin-specific validated hyperparameters")


class PluginAnalysisResult(BaseModel):
    """Persisted output of a plugin analysis run."""
    result_id: str = Field(..., description="Unique result execution UUID")
    plugin_id: str = Field(..., description="Executing plugin ID")
    recording_id: str = Field(..., description="Target recording ID")
    status: str = Field("completed", description="completed | failed")
    result_json: dict[str, Any] = Field(..., description="Structured output payload")
    plugin_version: str = Field("1.0.0", description="Plugin version at time of execution")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    execution_time_seconds: float = Field(0.0, ge=0.0, description="Duration of plugin computation")
    error: Optional[str] = Field(None, description="Sanitized error message if failed")
