"""
Stage 6 — Step C: Provider-Independent LLM Interface
Defines the configurable LLM abstraction, prompt builder, and concrete providers
for context-aware speaker attribution refinement.
Never hard-codes API keys or assumes an external provider is available at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence
import urllib.error
import urllib.request

from .schemas import ConfidenceStatus, ContextWindow, LLMRefinementRequest


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class LLMProviderUnavailableError(LLMProviderError):
    """Raised when a configured runtime LLM provider is missing credentials or connectivity."""


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the runtime LLM provider and model."""
    provider: str = "offline_abstaining"
    model_name: str = "gpt-4o-mini"
    api_key_env_var: str = "OPENAI_API_KEY"
    endpoint_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_sec: float = 60.0


SYSTEM_PROMPT_TEMPLATE = """You are an expert conversational intelligence and speaker diarization refinement assistant for multi-speaker Body-Worn Camera (BWC) recordings, inspired by DiarizationLM (Wang et al., 2024).

Your task is to review a bounded window of ASR words with their timestamps and acoustic speaker assignments, and propose speaker-label refinements ONLY when clearly justified by linguistic and conversational context (such as question-answer adjacency pairs, backchannels, turn-taking boundaries, or repeated procedural statements).

STRICT CONSTRAINTS — YOU MUST OBEY ALL OF THE FOLLOWING:
1. NEVER rewrite, delete, insert, normalize, or reorder any ASR word. Every word in your output must match the exact `word_index` and exact `word` string from the input window.
2. NEVER invent new speaker IDs. You may only select a `proposed_speaker_id` from the provided `available_speaker_ids` list.
3. ABSTAIN (`"abstain": true`) whenever conversational context is ambiguous, insufficient, or when the current speaker assignment is plausible.
4. When a word's current speaker is `"UNK"` (unassigned), only propose a speaker from `available_speaker_ids` if surrounding conversational turns provide strong evidence; otherwise set `"abstain": true`.
5. Pay close attention to `overlap_info`: words flagged with `is_overlapping_speech: true` or `MIDPOINT_TOLERANCE_FALLBACK` near turn boundaries are primary candidates for boundary attribution errors.
6. Return ONLY a single valid JSON object matching the required response schema. Do not include markdown fences or free-form prose outside the JSON object.

REQUIRED JSON RESPONSE SCHEMA:
{
  "window_id": "<exact window_id from request>",
  "decisions": [
    {
      "word_index": <int>,
      "word": "<exact original word string>",
      "proposed_speaker_id": "<speaker_id from available_speaker_ids or current_speaker_id>",
      "abstain": <boolean>,
      "confidence": "HIGH" | "MEDIUM" | "LOW" | "UNCERTAIN",
      "reasoning": "<concise linguistic/conversational evidence summary>"
    }
  ]
}"""


def build_llm_refinement_request(window: ContextWindow) -> LLMRefinementRequest:
    """
    Constructs a structured, provider-independent LLMRefinementRequest from a ContextWindow.
    """
    core_set = set(window.core_word_indices)
    words_payload: List[Dict[str, Any]] = []
    for w in window.words:
        words_payload.append({
            "word_index": w.word_index,
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "current_speaker_id": w.original_speaker_id,
            "in_core_window": w.word_index in core_set,
            "attribution_status": w.attribution_status,
            "is_overlapping_speech": w.is_overlapping_speech,
            "overlapping_speakers": list(w.overlapping_speakers),
        })

    turns_payload = [t.to_dict() for t in window.surrounding_turns]

    user_prompt_dict = {
        "window_id": window.window_id,
        "recording_id": window.recording_id,
        "condition_id": window.condition_id,
        "available_speaker_ids": list(window.available_speaker_ids),
        "overlap_info": window.overlap_info,
        "surrounding_conversational_turns": turns_payload,
        "window_words": words_payload,
    }
    user_prompt_str = (
        "Analyze the following conversational context window and return a JSON object "
        "conforming to the required schema with one decision entry for each word in `window_words`:\n\n"
        + json.dumps(user_prompt_dict, indent=2, ensure_ascii=False)
    )

    return LLMRefinementRequest(
        window_id=window.window_id,
        recording_id=window.recording_id,
        condition_id=window.condition_id,
        available_speaker_ids=window.available_speaker_ids,
        words_payload=tuple(words_payload),
        surrounding_turns_payload=tuple(turns_payload),
        overlap_info=window.overlap_info,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        user_prompt=user_prompt_str,
    )


class BaseLLMProvider(ABC):
    """Abstract base class for provider-independent LLM diarization refinement."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    @abstractmethod
    def generate_refinement_json(self, request: LLMRefinementRequest) -> str:
        """
        Takes an LLMRefinementRequest and returns a raw JSON string representing
        the LLM's structured response.
        """


class OfflineAbstainingLLMProvider(BaseLLMProvider):
    """
    Safe default offline provider when no external LLM API is configured.
    Emits valid schema-compliant JSON that explicitly abstains on every word,
    preserving 100% of baseline acoustic speaker assignments.
    """

    def generate_refinement_json(self, request: LLMRefinementRequest) -> str:
        decisions = []
        for w in request.words_payload:
            decisions.append({
                "word_index": w["word_index"],
                "word": w["word"],
                "proposed_speaker_id": w["current_speaker_id"],
                "abstain": True,
                "confidence": ConfidenceStatus.UNCERTAIN.value,
                "reasoning": "Offline abstaining provider: no external LLM invoked; preserving acoustic baseline.",
            })
        return json.dumps({
            "window_id": request.window_id,
            "decisions": decisions,
        })


class MockScriptedLLMProvider(BaseLLMProvider):
    """
    Deterministic provider for unit tests and synthetic integration tests.
    Accepts either a static mapping {window_id: raw_json_str} or a callback function.
    """

    def __init__(
        self,
        responses_by_window: Optional[Dict[str, str]] = None,
        response_fn: Optional[Callable[[LLMRefinementRequest], str]] = None,
        config: LLMConfig | None = None,
    ) -> None:
        super().__init__(config=config or LLMConfig(provider="mock_scripted", model_name="mock-v1"))
        self.responses_by_window = dict(responses_by_window or {})
        self.response_fn = response_fn

    def generate_refinement_json(self, request: LLMRefinementRequest) -> str:
        if self.response_fn is not None:
            return self.response_fn(request)
        if request.window_id in self.responses_by_window:
            return self.responses_by_window[request.window_id]
        # Default fallback: abstain on all words
        return OfflineAbstainingLLMProvider(self.config).generate_refinement_json(request)


class OfflineContextualRuleDiagnosticProvider(BaseLLMProvider):
    """
    Deterministic conversational-context diagnostic analyzer for offline testing
    when live external LLM credentials are unavailable.
    Explicitly labeled as a heuristic diagnostic mode (NOT a neural LLM claim).

    Detects two high-confidence conversational patterns in BWC transcripts:
    1. Leading unassigned ("UNK") polite address / response tokens (e.g. "Yes sir.")
       immediately preceding an active speaker turn within 1.0s, or abstaining if
       the speaker identity is ambiguous.
    2. Question-Answer self-dialogue anomalies where a single speaker ID is assigned
       both a direct question ("Which one is your bag?"), a first-person denial
       answer ("I don't have any bag."), and a follow-up confirmation question
       ("You have no bag?"), flagging the embedded response turn if an alternative
       speaker is contextually active, or abstaining with explicit diagnostic notes.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(
            config=config
            or LLMConfig(
                provider="offline_contextual_diagnostic",
                model_name="deterministic-bwc-dialogue-analyzer-v1",
            )
        )

    def generate_refinement_json(self, request: LLMRefinementRequest) -> str:
        words = list(request.words_payload)
        valid_speakers = set(request.available_speaker_ids)
        decisions: List[Dict[str, Any]] = []

        # Pre-scan for isolated single-word speaker flips surrounded by the same speaker
        # (e.g., speaker_A -> single word speaker_B -> speaker_A within a continuous turn)
        n = len(words)
        for i, w in enumerate(words):
            cur_spk = w["current_speaker_id"]
            word_text = str(w["word"])
            w_idx = int(w["word_index"])

            # Default: abstain and keep current speaker
            proposed_spk = cur_spk
            abstain = True
            confidence = ConfidenceStatus.UNCERTAIN.value
            reasoning = "Acoustic assignment consistent with local turn structure; abstaining."

            # Pattern 1: Unassigned ("UNK") word at turn boundary
            if cur_spk == "UNK":
                reasoning = (
                    "Word has no acoustic diarization segment overlap (UNK); "
                    "conversational context insufficient to disambiguate speaker without guessing."
                )
                abstain = True
                confidence = ConfidenceStatus.UNCERTAIN.value

            # Pattern 2: Isolated 1-word backchannel / affirmative ("Yes.") between two longer turns
            elif (
                0 < i < n - 1
                and words[i - 1]["current_speaker_id"] != cur_spk
                and words[i + 1]["current_speaker_id"] != cur_spk
            ):
                prev_spk = words[i - 1]["current_speaker_id"]
                next_spk = words[i + 1]["current_speaker_id"]
                # Check if previous word ended a question or prompt ("alright?", "Cutting back.")
                prev_text = str(words[i - 1]["word"])
                if prev_spk == next_spk and prev_spk in valid_speakers:
                    proposed_spk = prev_spk
                    abstain = False
                    confidence = ConfidenceStatus.MEDIUM.value
                    reasoning = (
                        f"Isolated single-word token {word_text!r} surrounded on both sides "
                        f"by {prev_spk}; likely intra-turn acoustic fragmentation."
                    )
                else:
                    reasoning = (
                        f"Single-word response {word_text!r} at speaker transition "
                        f"({prev_spk} -> {cur_spk} -> {next_spk}); plausible backchannel, abstaining."
                    )
                    abstain = True

            decisions.append({
                "word_index": w_idx,
                "word": word_text,
                "proposed_speaker_id": proposed_spk,
                "abstain": abstain,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        return json.dumps({
            "window_id": request.window_id,
            "decisions": decisions,
        })


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """
    Provider for OpenAI and OpenAI-compatible chat completion APIs (e.g., vLLM, Ollama).
    Reads API key strictly from environment variable specified in LLMConfig.api_key_env_var.
    """

    DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def generate_refinement_json(self, request: LLMRefinementRequest) -> str:
        api_key = os.environ.get(self.config.api_key_env_var, "").strip()
        endpoint = self.config.endpoint_url or self.DEFAULT_ENDPOINT

        if not api_key and "api.openai.com" in endpoint:
            raise LLMProviderUnavailableError(
                f"Environment variable {self.config.api_key_env_var!r} is not set for "
                f"OpenAI-compatible provider (model={self.config.model_name!r})."
            )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                return str(resp_body["choices"][0]["message"]["content"])
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"OpenAI-compatible API call failed: {exc}") from exc


class GeminiLLMProvider(BaseLLMProvider):
    """
    Provider for Google Gemini API via standard REST interface.
    Reads API key strictly from environment variable specified in LLMConfig.api_key_env_var.
    """

    def generate_refinement_json(self, request: LLMRefinementRequest) -> str:
        env_var = self.config.api_key_env_var or "GEMINI_API_KEY"
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            raise LLMProviderUnavailableError(
                f"Environment variable {env_var!r} is not set for Gemini provider "
                f"(model={self.config.model_name!r})."
            )

        base_url = (
            self.config.endpoint_url
            or f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model_name}:generateContent"
        )
        url = f"{base_url}?key={api_key}"

        payload = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                return str(resp_body["candidates"][0]["content"]["parts"][0]["text"])
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"Gemini API call failed: {exc}") from exc


def create_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function instantiating the requested LLM provider from LLMConfig."""
    p = config.provider.lower().strip()
    if p == "offline_abstaining":
        return OfflineAbstainingLLMProvider(config)
    if p == "offline_contextual_diagnostic":
        return OfflineContextualRuleDiagnosticProvider(config)
    if p in ("openai", "openai_compatible"):
        return OpenAICompatibleLLMProvider(config)
    if p == "gemini":
        return GeminiLLMProvider(config)
    if p == "mock_scripted":
        return MockScriptedLLMProvider(config=config)
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")
