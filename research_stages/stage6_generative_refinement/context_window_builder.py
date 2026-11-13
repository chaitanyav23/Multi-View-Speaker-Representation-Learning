"""
Stage 6 — Step B: Context-Window Builder
Constructs bounded, overlapping conversational windows for LLM diarization
refinement without sending entire long recordings in a single request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .schemas import ContextWindow, ConversationalTurnSummary, OrchestratedWord
from .word_orchestrator import build_conversational_turns


@dataclass(frozen=True)
class ContextWindowConfig:
    """Configuration for sliding conversational window construction."""
    window_size_words: int = 60
    context_overlap_words: int = 15
    surrounding_turns_radius: int = 4
    turn_grouping_max_gap_sec: float = 2.0

    def __post_init__(self) -> None:
        if self.window_size_words <= 0:
            raise ValueError(f"window_size_words must be > 0, got {self.window_size_words}")
        if self.context_overlap_words < 0:
            raise ValueError(f"context_overlap_words must be >= 0, got {self.context_overlap_words}")
        if self.surrounding_turns_radius < 0:
            raise ValueError(f"surrounding_turns_radius must be >= 0, got {self.surrounding_turns_radius}")


class ContextWindowBuilder:
    """
    Partitions an orchestrated word sequence into bounded core windows with
    configurable left/right word overlap and surrounding conversational turn context.
    """

    def __init__(self, config: ContextWindowConfig | None = None) -> None:
        self.config = config or ContextWindowConfig()

    def build_windows(
        self,
        words: Sequence[OrchestratedWord],
        available_speaker_ids: Sequence[str],
        recording_id: str = "unknown_recording",
        condition_id: str = "default",
    ) -> List[ContextWindow]:
        """
        Builds ordered ContextWindow instances covering all words in the transcript.
        Each word index belongs to the core_word_indices of exactly one window,
        while context_word_indices includes left and right overlap buffers.
        """
        n_words = len(words)
        if n_words == 0:
            return []

        valid_speakers = tuple(sorted({
            str(s).strip() for s in available_speaker_ids
            if s and str(s).strip() not in ("", "UNK")
        }))

        all_turns = build_conversational_turns(
            words, max_gap=self.config.turn_grouping_max_gap_sec
        )

        windows: List[ContextWindow] = []
        core_size = self.config.window_size_words
        ovl_size = self.config.context_overlap_words

        win_idx = 0
        for core_start in range(0, n_words, core_size):
            core_end = min(n_words, core_start + core_size)
            ctx_start = max(0, core_start - ovl_size)
            ctx_end = min(n_words, core_end + ovl_size)

            core_indices = tuple(range(core_start, core_end))
            context_indices = tuple(range(ctx_start, ctx_end))
            window_words = tuple(words[i] for i in context_indices)

            # Identify time span of the context window to select surrounding turns
            win_t_start = next((w.start for w in window_words if w.start is not None), None)
            win_t_end = next((w.end for w in reversed(window_words) if w.end is not None), None)

            surrounding_turns = self._select_surrounding_turns(
                all_turns=all_turns,
                win_t_start=win_t_start,
                win_t_end=win_t_end,
            )

            overlap_info = self._summarize_window_overlap(
                window_words=window_words,
                core_indices_set=set(core_indices),
            )

            window_id = f"{recording_id}__{condition_id}__win_{win_idx:03d}"
            windows.append(
                ContextWindow(
                    window_id=window_id,
                    recording_id=recording_id,
                    condition_id=condition_id,
                    window_index=win_idx,
                    core_word_indices=core_indices,
                    context_word_indices=context_indices,
                    words=window_words,
                    surrounding_turns=surrounding_turns,
                    available_speaker_ids=valid_speakers,
                    overlap_info=overlap_info,
                )
            )
            win_idx += 1

        return windows

    def _select_surrounding_turns(
        self,
        all_turns: Sequence[ConversationalTurnSummary],
        win_t_start: float | None,
        win_t_end: float | None,
    ) -> Tuple[ConversationalTurnSummary, ...]:
        if not all_turns:
            return ()
        if win_t_start is None or win_t_end is None:
            return tuple(all_turns[: max(1, 2 * self.config.surrounding_turns_radius)])

        overlapping_turn_indices: List[int] = []
        for idx, t in enumerate(all_turns):
            if t.start is None or t.end is None:
                continue
            if t.end >= win_t_start and t.start <= win_t_end:
                overlapping_turn_indices.append(idx)

        if not overlapping_turn_indices:
            return tuple(all_turns[: max(1, self.config.surrounding_turns_radius)])

        first_turn = max(0, min(overlapping_turn_indices) - self.config.surrounding_turns_radius)
        last_turn = min(
            len(all_turns),
            max(overlapping_turn_indices) + 1 + self.config.surrounding_turns_radius,
        )
        return tuple(all_turns[first_turn:last_turn])

    @staticmethod
    def _summarize_window_overlap(
        window_words: Sequence[OrchestratedWord],
        core_indices_set: set[int],
    ) -> Dict[str, Any]:
        overlapping_indices: List[int] = []
        unassigned_indices: List[int] = []
        fallback_indices: List[int] = []
        active_overlap_speakers: set[str] = set()

        for w in window_words:
            if w.is_overlapping_speech:
                overlapping_indices.append(w.word_index)
                active_overlap_speakers.update(w.overlapping_speakers)
            if w.original_speaker_id == "UNK":
                unassigned_indices.append(w.word_index)
            if w.attribution_status == "MIDPOINT_TOLERANCE_FALLBACK":
                fallback_indices.append(w.word_index)

        return {
            "has_overlapping_speech": len(overlapping_indices) > 0,
            "overlapping_word_indices": overlapping_indices,
            "overlapping_speakers": sorted(active_overlap_speakers),
            "unassigned_word_indices": unassigned_indices,
            "midpoint_fallback_word_indices": fallback_indices,
            "core_word_count": len(core_indices_set),
            "total_window_word_count": len(window_words),
        }
