"""Speech Transcription Orchestration and Result Persistence Service."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from adaptive.confidence import ConfidenceCalibrator
from backend.config import settings
from backend.database.models import RecordingRepository, TranscriptRepository
from engine.base import BaseSpeechEngine
from engine.phase20_engine import Phase20SpeechEngine, PipelineDependencyError
from engine.schemas import TranscriptResult

from .audio_service import AudioService


class TranscriptionService:
    """Orchestrates audio preprocessing, speech intelligence inference, and result storage."""

    _engine: Optional[BaseSpeechEngine] = None
    _calibrator: Optional[ConfidenceCalibrator] = None

    @classmethod
    def get_engine(cls) -> BaseSpeechEngine:
        """Return the active Phase 20 speech processing engine."""
        if cls._engine is None:
            cls._engine = Phase20SpeechEngine(
                model_size=settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
                hf_token=settings.hf_token or None,
                calibrator=cls.get_calibrator(),
                confidence_threshold=settings.confidence_threshold,
            )
        return cls._engine

    @classmethod
    def set_engine(cls, engine: BaseSpeechEngine) -> None:
        cls._engine = engine

    @classmethod
    def get_calibrator(cls) -> ConfidenceCalibrator:
        if cls._calibrator is None:
            cls._calibrator = ConfidenceCalibrator(method=settings.calibration_method)
        return cls._calibrator

    @classmethod
    def execute_transcription(
        cls,
        recording_id: str,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        use_mock: bool = False,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptResult:
        """Run real Phase 20 pipeline, apply calibration, and persist canonical result."""
        record = RecordingRepository.get(recording_id)
        if not record:
            raise ValueError(f"Recording '{recording_id}' not found.")

        # Step 1: Preprocessing & format normalization
        if progress_cb:
            progress_cb(0.1, "Preprocessing audio")
        norm_audio_path = AudioService.prepare_16k_mono_audio(recording_id)

        # Step 2: Speech Engine Execution
        engine = cls.get_engine()
        options = {"initial_prompt": initial_prompt} if initial_prompt else {}
        
        result = engine.process(
            audio_path=str(norm_audio_path),
            recording_id=recording_id,
            filename=record.filename,
            duration=record.duration,
            progress_callback=progress_cb,
            options=options,
        )

        # Step 3: Confidence Calibration Post-Processing
        calibrator = cls.get_calibrator()
        for word in result.words:
            if not word.is_calibrated:
                word.confidence = calibrator.calibrate_score(word.raw_confidence)
                word.is_calibrated = True

        # Re-compute turn average confidences
        for turn in result.segments:
            if turn.words:
                turn.avg_confidence = round(sum(w.confidence for w in turn.words) / len(turn.words), 3)

        # Step 4: Storage Persistence
        now_iso = datetime.now(timezone.utc).isoformat()
        result_dict = result.model_dump()
        
        # Save to SQLite
        TranscriptRepository.save(recording_id, result_dict, now_iso)

        # Save to filesystem JSON
        json_path = settings.results_dir / f"{recording_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        return result

    @classmethod
    def get_transcript(cls, recording_id: str) -> Optional[TranscriptResult]:
        """Retrieve stored transcript result from database or file."""
        data = TranscriptRepository.get(recording_id)
        if not data:
            # Fallback to filesystem
            json_path = settings.results_dir / f"{recording_id}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        if not data:
            return None
        return TranscriptResult.model_validate(data)
