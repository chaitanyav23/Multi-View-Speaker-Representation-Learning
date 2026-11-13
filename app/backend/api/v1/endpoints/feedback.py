"""User Feedback, Correction Revisions, and Vocabulary Endpoints."""
from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, Query

from backend.services.feedback_service import FeedbackService
from engine.schemas import FeedbackResponse, UserFeedbackItem

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(feedback: UserFeedbackItem) -> FeedbackResponse:
    """Submit a manual correction or speaker label adjustment."""
    feedback_id = FeedbackService.record_feedback(feedback)
    return FeedbackResponse(
        status="recorded",
        message="Correction recorded and adaptive vocabulary updated.",
        feedback_id=feedback_id,
    )


@router.get("/feedback", tags=["Feedback"])
def list_feedback(recording_id: Optional[str] = Query(default=None)) -> list[dict[str, Any]]:
    """List historical corrections submitted by users."""
    return FeedbackService.list_feedback(recording_id)


@router.get("/feedback/vocabulary", tags=["Feedback"])
def get_verified_vocabulary() -> list[str]:
    """Retrieve accumulated domain terminology extracted from user feedback."""
    return FeedbackService.get_accumulated_vocabulary()
