"""Abstract Base Interface for Speech Processing Engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from .schemas import TranscriptResult


class BaseSpeechEngine(ABC):
    """Abstract interface defining the contract for speech recognition and attribution engines."""

    @abstractmethod
    def process(
        self,
        audio_path: str,
        recording_id: str,
        filename: Optional[str] = None,
        duration: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> TranscriptResult:
        """Process an input audio file and generate a canonical TranscriptResult."""
        pass
