"""Native Wav2Vec2 CTC Forced Aligner for Phase 20 Speech Engine.

Reproduces WhisperX v3.8.5 (commit 4a6477e) forced-alignment semantics:
- Uses torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H bundle definition.
- 29-label vocabulary with blank ID 0 ('-') and space delimiter ID 1 ('|').
- Dynamic wildcard column construction (excluding blank token, max non-blank per frame).
- Custom PyTorch CTC trellis dynamic programming and Viterbi backtracking.
- Tie-breaking: changed > stayed.
- Frame probability exponentiation to linear probabilities in [0.0, 1.0].
- Character run merging and word-level aggregation.
- Punctuation, digits, and OOV character retention with nearest-neighbor NaN interpolation.
- Phase 20-compatible output format: [{"word": str, "start": float, "end": float, "confidence": float}].
- Strict lazy model loading: checkpoint is NEVER loaded during initialization or test runs.

Audio handling note:
Accepts 16 kHz mono float32 waveforms in [-1.0, 1.0].
Current application preprocessing standardizes audio to -16 LUFS (EBU R128),
whereas historical Phase 20 standalone used -20 LUFS. Preprocessing configuration
remains unchanged at -16 LUFS in Stage 1.3.2-B.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


@dataclass
class TrellisPoint:
    """A single point in the CTC trellis alignment path."""
    token_index: int
    time_index: int
    score: float


@dataclass
class AlignedSegment:
    """Collapsed character run with temporal frame bounds and mean score."""
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def interpolate_nearest_1d(values: List[Optional[float]]) -> List[Optional[float]]:
    """Pure-Python 1D nearest-neighbor interpolation with forward/backward fill.

    Numerically matches WhisperX v3.8.5's pandas invocation:
    `Series(values).interpolate(method='nearest').ffill().bfill()`
    without incurring pandas/nltk runtime dependencies.
    """
    n = len(values)
    if n == 0:
        return []

    valid_indices = [i for i, v in enumerate(values) if v is not None]
    if len(valid_indices) == 0:
        return list(values)
    if len(valid_indices) == 1:
        single_val = values[valid_indices[0]]
        return [single_val] * n

    result = list(values)
    for i in range(n):
        if result[i] is None:
            # Find closest valid index; ties break toward earlier index (matching pandas)
            closest_idx = min(valid_indices, key=lambda idx: (abs(i - idx), idx))
            result[i] = values[closest_idx]
    return result


class Wav2Vec2ForcedAligner:
    """Native Wav2Vec2 CTC Forced Aligner reproducing WhisperX v3.8.5 semantics.

    Preserves exact Phase 20 word-level outputs, timestamp projections, and
    wildcard OOV handling.
    """

    def __init__(
        self,
        device: str = "cpu",
        lazy_load: bool = True,
    ) -> None:
        """Initialize aligner dictionary and configuration without loading weights.

        Args:
            device: Target execution device ('cuda', 'cuda:0', 'cpu').
            lazy_load: If True, model weights are not loaded until explicitly requested.
        """
        self.device = torch.device(device)
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sample_rate = self.bundle.sample_rate  # 16000

        # Exact 29 labels from Wav2Vec2 ASR Base 960h bundle:
        # ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U',
        #  'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        self.labels: Tuple[str, ...] = self.bundle.get_labels()
        self.blank_id: int = 0
        self.delimiter_id: int = 1

        # Case-insensitive mapping: lowercased char -> token index
        # Notice: '-' is at index 0 (blank_id), '|' is at index 1 (space delimiter)
        self.dictionary: dict[str, int] = {c.lower(): i for i, c in enumerate(self.labels)}

        # Internal model reference (strictly None during Stage 1.3.2-B.1)
        self._model: Optional[torch.nn.Module] = None

        if not lazy_load:
            self.load_model()

    def load_model(self) -> torch.nn.Module:
        """Lazily load the pretrained model from the local cache onto the target device."""
        if self._model is None:
            self._model = self.bundle.get_model().to(self.device)
            self._model.eval()
        return self._model

    def _ensure_model(self) -> torch.nn.Module:
        """Internal helper to ensure model is instantiated before forward pass."""
        if self._model is None:
            return self.load_model()
        return self._model

    def tokenize_text(
        self,
        text: str,
    ) -> Tuple[List[str], List[int], List[int], bool]:
        """Preprocess text into alignment tokens following WhisperX v3.8.5 semantics.

        Rules:
        1. Leading and trailing whitespace is stripped from alignment consideration.
        2. Internal whitespace is converted to delimiter '|' (index 1).
        3. All characters are lowercased.
        4. Characters in model_dictionary are mapped directly to their label index.
           Crucial: '-' is present in model_dictionary at index 0 (the blank token).
           It maps to token ID 0, NOT to wildcard.
        5. Any non-space character not in model_dictionary (digits, punctuation, symbols)
           is retained in clean_char and flagged for wildcard column routing.

        Returns:
            clean_char: List of lowercased/delimited characters to align.
            clean_cdx: Indices in original `text` corresponding to each clean_char.
            tokens_initial: Token IDs before wildcard assignment.
            has_wildcard: Boolean indicating if any character is outside the 29-label vocabulary.
        """
        if not text:
            return [], [], [], False

        num_leading = len(text) - len(text.lstrip())
        num_trailing = len(text) - len(text.rstrip())

        clean_char: List[str] = []
        clean_cdx: List[int] = []

        for cdx, char in enumerate(text):
            char_ = char.lower().replace(" ", "|")

            # Ignore leading and trailing whitespace
            if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                continue

            if char_ in self.dictionary:
                clean_char.append(char_)
                clean_cdx.append(cdx)
            elif char_ not in (" ", "|"):
                # Unknown char (digit, symbol, punctuation) - retained for wildcard alignment
                clean_char.append(char_)
                clean_cdx.append(cdx)

        if not clean_char:
            return [], [], [], False

        has_wildcard = any(c not in self.dictionary for c in clean_char)

        # Temporary placeholder: unknown chars mapped to -1 until wildcard column is constructed
        tokens_initial = [self.dictionary.get(c, -1) for c in clean_char]

        return clean_char, clean_cdx, tokens_initial, has_wildcard

    def construct_wildcard_emission(
        self,
        emission: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Extend log-softmax emissions with a synthetic wildcard column.

        Reproduces WhisperX v3.8.5 (PR #1386):
        - Excludes the blank column (index 0).
        - Computes the maximum log-probability among all non-blank tokens per frame.
        - Concatenates the column vector to the right of the emission tensor.
        - Returns extended emissions of shape (num_frames, vocab_size + 1) and wildcard_id.
        """
        non_blank_mask = torch.ones(emission.size(1), dtype=torch.bool, device=emission.device)
        non_blank_mask[self.blank_id] = False

        wildcard_col = emission[:, non_blank_mask].max(dim=1).values
        extended_emission = torch.cat([emission, wildcard_col.unsqueeze(1)], dim=1)
        wildcard_id = extended_emission.size(1) - 1

        return extended_emission, wildcard_id

    def compute_trellis(
        self,
        emission: torch.Tensor,
        tokens: List[int],
    ) -> torch.Tensor:
        """Compute the CTC trellis dynamic programming matrix.

        Matches the official PyTorch tutorial / WhisperX implementation:
        Trellis shape: (num_frames + 1, num_tokens + 1).
        Extra dimensions represent <SoS> start-of-sentence and time zero.
        """
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.empty((num_frame + 1, num_tokens + 1), dtype=emission.dtype, device=emission.device)
        trellis[0, 0] = 0.0
        trellis[1:, 0] = torch.cumsum(emission[:, self.blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token (blank transition)
                trellis[t, 1:] + emission[t, self.blank_id],
                # Score for advancing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def backtrack(
        self,
        trellis: torch.Tensor,
        emission: torch.Tensor,
        tokens: List[int],
    ) -> Optional[List[TrellisPoint]]:
        """Trace the optimal Viterbi alignment path through the trellis.

        Preserves WhisperX v3.8.5 tie-breaking:
        `changed > stayed` (advances token only if strictly greater, else stays).
        Frame probabilities are exponentiated from log-probability to linear scale in [0.0, 1.0].
        """
        j = trellis.size(1) - 1
        t_start = int(torch.argmax(trellis[:, j]).item())

        path: List[TrellisPoint] = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, self.blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # Frame-wise linear probability in [0.0, 1.0]
            token_idx = tokens[j - 1] if changed > stayed else self.blank_id
            prob = float(torch.exp(emission[t - 1, token_idx]).item())
            path.append(TrellisPoint(token_index=j - 1, time_index=t - 1, score=prob))

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            # Backtracking failed to reach token 0
            return None

        return path[::-1]

    def merge_repeats(
        self,
        path: List[TrellisPoint],
        clean_char: List[str],
    ) -> List[AlignedSegment]:
        """Collapse consecutive frames aligned to the same token into character spans."""
        i1, i2 = 0, 0
        segments: List[AlignedSegment] = []
        n = len(path)

        while i1 < n:
            while i2 < n and path[i1].token_index == path[i2].token_index:
                i2 += 1
            span_score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            token_idx = path[i1].token_index
            segments.append(
                AlignedSegment(
                    label=clean_char[token_idx],
                    start=path[i1].time_index,
                    end=path[i2 - 1].time_index + 1,
                    score=span_score,
                )
            )
            i1 = i2

        return segments

    def align_from_emissions(
        self,
        emission: torch.Tensor,
        text: str,
        start_time: float,
        end_time: float,
    ) -> List[dict[str, Any]]:
        """Perform forced alignment using precomputed log-softmax emissions.

        This method executes the entire CTC dynamic programming, wildcard handling,
        timestamp projection, and word aggregation logic without requiring a model forward pass.
        Enables deterministic testing against reference emissions.

        Args:
            emission: Tensor of shape (num_frames, 29) containing log-softmax probabilities.
            text: Transcription text string for the segment.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.

        Returns:
            List of Phase 20-compatible word dictionaries:
            [{"word": str, "start": float, "end": float, "confidence": float}]
        """
        clean_text = text.strip()
        if not clean_text or emission.size(0) == 0:
            return []

        clean_char, clean_cdx, tokens_initial, has_wildcard = self.tokenize_text(text)
        if not clean_char:
            return []

        # Prepare emission matrix (optionally extended with wildcard column)
        if has_wildcard:
            extended_emission, wildcard_id = self.construct_wildcard_emission(emission)
            tokens = [self.dictionary.get(c, wildcard_id) for c in clean_char]
            active_emission = extended_emission
        else:
            tokens = [self.dictionary[c] for c in clean_char]
            active_emission = emission

        trellis = self.compute_trellis(active_emission, tokens)
        path = self.backtrack(trellis, active_emission, tokens)

        duration = max(0.001, end_time - start_time)
        num_frames = active_emission.size(0)

        # Fallback if backtracking failed
        if path is None:
            # Segment could not be aligned; return unaligned fallback representation
            tokens_in_text = [w for w in clean_text.split() if w]
            return [
                {
                    "word": w,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "confidence": 0.0,
                }
                for w in tokens_in_text
            ]

        char_segments = self.merge_repeats(path, clean_char)

        # Linear frame ratio: duration / num_frames
        ratio = duration / num_frames

        # Map character timestamps back to original text indices
        char_data: List[dict[str, Any]] = []
        word_idx = 0

        for cdx, char in enumerate(text):
            c_start: Optional[float] = None
            c_end: Optional[float] = None
            c_score: Optional[float] = None

            if cdx in clean_cdx:
                clean_pos = clean_cdx.index(cdx)
                seg = char_segments[clean_pos]
                c_start = round(seg.start * ratio + start_time, 3)
                c_end = round(seg.end * ratio + start_time, 3)
                c_score = round(seg.score, 3)

            char_data.append({
                "char": char,
                "start": c_start,
                "end": c_end,
                "score": c_score,
                "word_idx": word_idx,
            })

            # Words are demarcated by spaces in the original text
            if cdx == len(text) - 1 or text[cdx + 1] == " ":
                word_idx += 1

        # Group characters into word tokens
        unique_word_indices = sorted(list({c["word_idx"] for c in char_data}))
        raw_words: List[dict[str, Any]] = []

        for w_idx in unique_word_indices:
            w_chars = [c for c in char_data if c["word_idx"] == w_idx]
            w_text = "".join(c["char"] for c in w_chars).strip()
            if not w_text:
                continue

            # Non-space characters define the bounds and score
            non_space = [c for c in w_chars if c["char"] != " "]
            valid_starts = [c["start"] for c in non_space if c["start"] is not None]
            valid_ends = [c["end"] for c in non_space if c["end"] is not None]
            valid_scores = [c["score"] for c in non_space if c["score"] is not None]

            w_start = min(valid_starts) if valid_starts else None
            w_end = max(valid_ends) if valid_ends else None
            w_score = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None

            raw_words.append({
                "word": w_text,
                "start": w_start,
                "end": w_end,
                "score": w_score,
            })

        # Interpolate missing boundaries for words with no alignable characters
        if raw_words:
            starts = [w["start"] for w in raw_words]
            ends = [w["end"] for w in raw_words]

            if any(s is None for s in starts) and any(s is not None for s in starts):
                interp_starts = interpolate_nearest_1d(starts)
                interp_ends = interpolate_nearest_1d(ends)

                for idx, w in enumerate(raw_words):
                    if w["start"] is None and interp_starts[idx] is not None:
                        w["start"] = interp_starts[idx]
                    if w["end"] is None and interp_ends[idx] is not None:
                        w["end"] = interp_ends[idx]

        # Construct Phase 20-compatible output format
        output_words: List[dict[str, Any]] = []
        for w in raw_words:
            # Fallback bounds if still None
            final_start = w["start"] if w["start"] is not None else round(start_time, 3)
            final_end = w["end"] if w["end"] is not None else round(end_time, 3)
            final_conf = w["score"] if w["score"] is not None else 0.0

            output_words.append({
                "word": w["word"],
                "start": round(final_start, 3),
                "end": round(final_end, 3),
                "confidence": round(final_conf, 3),
            })

        return output_words

    def align_segment(
        self,
        waveform: Union[torch.Tensor, np.ndarray],
        text: str,
        start_time: float,
        end_time: float,
    ) -> List[dict[str, Any]]:
        """Acoustic forced alignment of an audio segment against transcription text.

        Accepts 16 kHz mono float32 audio and performs Wav2Vec2 forward pass.
        Note: Padded to minimum 400 samples (25 ms) if chunk is shorter than Wav2Vec2 conv window.

        Args:
            waveform: 1D or 2D audio array/tensor of shape (N,) or (1, N).
            text: Transcription text string for this segment.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.

        Returns:
            List of Phase 20-compatible word dictionaries:
            [{"word": str, "start": float, "end": float, "confidence": float}]
        """
        model = self._ensure_model()

        if not torch.is_tensor(waveform):
            waveform_t = torch.from_numpy(waveform).to(dtype=torch.float32)
        else:
            waveform_t = waveform.to(dtype=torch.float32)

        if waveform_t.dim() == 1:
            waveform_t = waveform_t.unsqueeze(0)

        # Ensure minimum length of 400 samples
        if waveform_t.shape[-1] < 400:
            pad_len = 400 - waveform_t.shape[-1]
            lengths = torch.as_tensor([waveform_t.shape[-1]], device=self.device)
            waveform_t = torch.nn.functional.pad(waveform_t, (0, pad_len))
        else:
            lengths = None

        with torch.inference_mode():
            waveform_dev = waveform_t.to(self.device)
            emissions, _ = model(waveform_dev, lengths=lengths)
            log_probs = torch.log_softmax(emissions, dim=-1)

        # Unbatch to 2D CPU tensor
        emission_cpu = log_probs[0].detach().cpu()

        return self.align_from_emissions(
            emission=emission_cpu,
            text=text,
            start_time=start_time,
            end_time=end_time,
        )
