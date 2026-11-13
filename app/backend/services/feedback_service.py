"""User Feedback, Manual Corrections, and Adaptive Vocabulary Service."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from adaptive.feedback_loop import FeedbackLoopManager
from backend.database.db import get_db_connection
from engine.schemas import UserFeedbackItem


class FeedbackService:
    """Handles submission and retrieval of user corrections and domain terminology."""

    _feedback_manager = FeedbackLoopManager()

    @classmethod
    def get_manager(cls) -> FeedbackLoopManager:
        return cls._feedback_manager

    @classmethod
    def record_feedback(cls, feedback: UserFeedbackItem) -> str:
        """Store correction and update adaptive vocabulary."""
        feedback_id = feedback.feedback_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Update adaptive feedback manager
        analysis = cls._feedback_manager.register_correction(
            original_text=feedback.original_text,
            corrected_text=feedback.corrected_text,
            original_speaker=feedback.original_speaker,
            corrected_speaker=feedback.corrected_speaker,
        )

        conn = get_db_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback (
                    feedback_id, recording_id, turn_id, word_index,
                    original_text, corrected_text, original_speaker, corrected_speaker,
                    notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    feedback.recording_id,
                    feedback.turn_id,
                    feedback.word_index,
                    feedback.original_text,
                    feedback.corrected_text,
                    feedback.original_speaker,
                    feedback.corrected_speaker,
                    feedback.notes or analysis.get("error_type"),
                    now_iso,
                ),
            )
        conn.close()
        return feedback_id

    @classmethod
    def list_feedback(cls, recording_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve recorded corrections."""
        conn = get_db_connection()
        if recording_id:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE recording_id = ? ORDER BY created_at DESC",
                (recording_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
        conn.close()

        return [dict(r) for r in rows]

    @classmethod
    def get_accumulated_vocabulary(cls) -> list[str]:
        return cls._feedback_manager.get_vocabulary()
