"""Adaptive Feedback Loop and Domain Vocabulary Extraction Subsystem."""
from __future__ import annotations

import re
from typing import Any, Optional


class FeedbackLoopManager:
    """Tracks manual user revisions, categorizes errors, and extracts domain vocabulary."""

    def __init__(self):
        self._verified_vocabulary: set[str] = set()

    def register_correction(
        self,
        original_text: str,
        corrected_text: str,
        original_speaker: Optional[str] = None,
        corrected_speaker: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze a correction pair, classify the error, and extract potential terminology."""
        is_speaker_change = (
            original_speaker is not None
            and corrected_speaker is not None
            and original_speaker != corrected_speaker
        )
        is_text_change = original_text.strip().lower() != corrected_text.strip().lower()

        error_type = "none"
        if is_speaker_change and is_text_change:
            error_type = "compound"
        elif is_speaker_change:
            error_type = "diarization"
        elif is_text_change:
            error_type = "acoustic_lexical"

        # Extract domain terminology (capitalized words, codes, abbreviations)
        tokens = re.findall(r"\b[A-Z0-9\-]{2,}\b|\b[A-Z][a-z]+\b", corrected_text)
        new_terms = []
        for token in tokens:
            cleaned = token.strip()
            if len(cleaned) > 2 and cleaned.lower() not in {"the", "and", "for", "with", "that"}:
                self._verified_vocabulary.add(cleaned)
                new_terms.append(cleaned)

        return {
            "error_type": error_type,
            "extracted_terms": new_terms,
            "total_verified_vocabulary": len(self._verified_vocabulary),
        }

    def get_whisper_prompt(self, base_prompt: Optional[str] = None) -> str:
        """Construct a contextual initial_prompt for Faster-Whisper to boost domain terminology."""
        terms = sorted(list(self._verified_vocabulary))[:30]
        prefix = f"Domain terms: {', '.join(terms)}. " if terms else ""
        if base_prompt:
            return f"{prefix}{base_prompt}".strip()
        return prefix.strip()

    def get_vocabulary(self) -> list[str]:
        """Return all accumulated verified domain terms."""
        return sorted(list(self._verified_vocabulary))
