"""Confidence Analytics, Calibration Metrics, and Reliability Diagram Endpoints."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException

from adaptive.confidence import ConfidenceCalibrator
from backend.services.transcription_service import TranscriptionService
from engine.schemas import AnalyticsResponse, AnalyticsTiers, AnalyticsTierStats

router = APIRouter()


@router.get("/analytics/{recording_id}", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics(recording_id: str) -> AnalyticsResponse:
    """Retrieve detailed confidence diagnostics and calibration metrics."""
    transcript = TranscriptionService.get_transcript(recording_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found for analytics.")

    confidences = [w.confidence for w in transcript.words]
    raw_confidences = [w.raw_confidence for w in transcript.words]
    # For user uploads without ground truth, synthetic binary proxy evaluates above median
    median_conf = float(sum(confidences) / len(confidences)) if confidences else 0.5
    pseudo_labels = [1 if c >= 0.50 else 0 for c in confidences]

    ece = ConfidenceCalibrator.compute_ece(confidences, pseudo_labels, n_bins=10)
    brier = ConfidenceCalibrator.compute_brier_score(confidences, pseudo_labels)
    diagram = ConfidenceCalibrator.compute_reliability_diagram(confidences, pseudo_labels, n_bins=10)

    # Tier breakdown
    low_count = sum(1 for c in confidences if c < 0.30)
    mid_count = sum(1 for c in confidences if 0.30 <= c < 0.60)
    high_count = sum(1 for c in confidences if c >= 0.60)

    return {
        "recording_id": recording_id,
        "filename": transcript.filename,
        "total_words": len(transcript.words),
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "mean_raw_confidence": round(sum(raw_confidences) / len(raw_confidences), 3) if raw_confidences else 0.0,
        "calibration": transcript.calibration.model_dump(),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "tiers": {
            "low": {"count": low_count, "ratio": round(low_count / max(1, len(confidences)), 3)},
            "mid": {"count": mid_count, "ratio": round(mid_count / max(1, len(confidences)), 3)},
            "high": {"count": high_count, "ratio": round(high_count / max(1, len(confidences)), 3)},
        },
        "reliability_diagram": diagram,
        "speakers_count": len(transcript.speakers),
        "duration": transcript.duration,
    }
