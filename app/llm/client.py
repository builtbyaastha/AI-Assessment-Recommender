"""LLM client wrapper. Agents only ever call complete_json - keeps the
provider swap-able (one env var) and makes agents testable without a real
API key via MockLLMClient below.

Default is Gemini 2.5 Flash per the original stack decision - cheap, fast,
decent free tier, handles structured JSON output fine.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMClient(ABC):
    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        raise NotImplementedError


def _extract_json(text: str) -> dict:
    # models sometimes wrap output in ```json fences despite being told not to
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise LLMError(f"could not parse JSON from model output: {text[:200]!r}")


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai

        self.genai = genai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set")
        genai.configure(api_key=key)
        self.model_name = model

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        model = self.genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        last_error = None
        for attempt in range(2):
            try:
                response = await model.generate_content_async(user_prompt)
                return _extract_json(response.text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("gemini call failed (attempt %d): %s", attempt + 1, exc)
        raise LLMError(str(last_error))


class MockLLMClient(LLMClient):
    """Rule-based stand-in for local dev / tests, no API key needed. Not
    trying to match Gemini's extraction quality, just enough to exercise
    the pipeline shape end to end."""

    OFF_TOPIC_MARKERS = ["legal advice", "should i fire", "salary negotiation", "visa sponsorship", "immigration law"]
    INJECTION_MARKERS = ["ignore previous instructions", "ignore all previous", "system prompt", "you are now", "disregard your instructions"]

    ROLE_KEYWORDS = ["java developer", "python developer", "data analyst", "software engineer", "sales", "manager", "customer service", "administrative assistant", "qa", "tester"]
    SKILL_KEYWORDS = ["java", "python", "sql", "javascript", "aws", "excel", "communication", "leadership", "stakeholder"]

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        text_lower = user_prompt.lower()

        if any(marker in text_lower for marker in self.INJECTION_MARKERS):
            return {"intent": "injection", "has_enough_context": False, "reasoning": "injection markers detected"}

        if any(marker in text_lower for marker in self.OFF_TOPIC_MARKERS):
            return {"intent": "off_topic", "has_enough_context": False, "reasoning": "off-topic markers detected"}

        role = next((r for r in self.ROLE_KEYWORDS if r in text_lower), None)
        skills = [s for s in self.SKILL_KEYWORDS if s in text_lower]

        has_enough = role is not None and (len(skills) > 0 or "developer" in text_lower or "manager" in text_lower)
        intent = "recommend" if has_enough else "clarify_needed"
        if "compare" in text_lower or "difference between" in text_lower:
            intent = "compare"

        return {
            "role": role,
            "technical_skills": [s for s in skills if s not in ("communication", "leadership", "stakeholder")],
            "soft_skills": [s for s in skills if s in ("communication", "leadership", "stakeholder")],
            "intent": intent,
            "has_enough_context": has_enough,
            "missing_info": [] if has_enough else ["seniority level", "key skills"],
            "reasoning": "mock rule-based extraction",
        }


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "gemini":
        try:
            return GeminiClient()
        except LLMError:
            logger.warning("Gemini unavailable, falling back to mock client - set GEMINI_API_KEY to use Gemini")
            return MockLLMClient()
    raise LLMError(f"unknown LLM_PROVIDER: {provider}")
