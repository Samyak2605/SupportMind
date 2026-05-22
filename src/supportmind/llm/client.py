from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from google import genai
from google.genai import types as genai_types
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from supportmind.config import EnvironmentSettings, SupportMindConfig
from supportmind.llm.cache import LLMCache

logger = logging.getLogger(__name__)


@dataclass
class FailoverEvent:
    timestamp: datetime
    from_model: str
    to_model: str
    reason: str


@dataclass
class LLMClient:
    """Single entry point for every LLM call in the agent layer.

    Primary: Groq (config.agent.brain_model). Fallback: Gemini
    (config.agent.fallback_model), used only if Groq raises after its own
    retries are exhausted. Every fallback is logged and recorded in
    `failover_events` so it can be surfaced in eval reports.
    """

    config: SupportMindConfig
    settings: EnvironmentSettings
    cache: LLMCache = field(init=False)
    failover_events: list[FailoverEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your environment or .env file.")
        self._groq = Groq(api_key=self.settings.groq_api_key, base_url=self.settings.groq_base_url)
        self._gemini = genai.Client(api_key=self.settings.gemini_api_key) if self.settings.gemini_api_key else None
        self.cache = LLMCache(self.config.paths.llm_cache)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(2), reraise=True)
    def _call_groq(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        response = self._groq.chat.completions.create(
            model=self.config.agent.brain_model,
            temperature=0.1,
            max_completion_tokens=1024,
            response_format={"type": "json_object"} if json_mode else None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.config.generation.request_timeout_seconds,
        )
        return response.choices[0].message.content or ""

    def _call_gemini(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        if self._gemini is None:
            raise RuntimeError("Gemini fallback unavailable: GEMINI_API_KEY is not configured")
        response = self._gemini.models.generate_content(
            model=self.config.agent.fallback_model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=system_prompt,
                response_mime_type="application/json" if json_mode else None,
            ),
        )
        return response.text or ""

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False, use_cache: bool = True) -> str:
        cache_key = LLMCache.make_key(self.config.agent.brain_model, system_prompt, user_prompt, json_mode)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            result = self._call_groq(system_prompt, user_prompt, json_mode)
        except Exception as exc:
            logger.warning("Groq brain call failed (%s); falling back to Gemini", exc)
            self.failover_events.append(
                FailoverEvent(
                    timestamp=datetime.now(timezone.utc),
                    from_model=self.config.agent.brain_model,
                    to_model=self.config.agent.fallback_model,
                    reason=str(exc),
                )
            )
            result = self._call_gemini(system_prompt, user_prompt, json_mode)

        if use_cache:
            self.cache.set(cache_key, result)
        return result
