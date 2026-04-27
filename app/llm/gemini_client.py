"""Gemini client wrapper. Provides text, structured, and streaming generation."""
from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    settings = get_settings()
    settings.require_keys()
    return genai.Client(api_key=settings.google_api_key)


def _retry():
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )


def _build_config(
    *,
    system: str | None,
    temperature: float,
    thinking_budget: int | None,
    response_schema: type | None = None,
) -> types.GenerateContentConfig:
    kwargs: dict = {
        "system_instruction": system,
        "temperature": temperature,
    }
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


async def generate_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    thinking_budget: int | None = None,
) -> str:
    """One-shot text generation."""
    settings = get_settings()
    client = get_client()
    cfg = _build_config(system=system, temperature=temperature, thinking_budget=thinking_budget)
    async for attempt in _retry():
        with attempt:
            resp = await client.aio.models.generate_content(
                model=model or settings.gemini_model_fast,
                contents=prompt,
                config=cfg,
            )
            return (resp.text or "").strip()
    return ""  # unreachable


async def generate_structured(
    prompt: str,
    schema: Type[T],
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    thinking_budget: int | None = None,
) -> T:
    """Structured JSON generation; returns a parsed Pydantic model."""
    settings = get_settings()
    client = get_client()
    cfg = _build_config(
        system=system, temperature=temperature,
        thinking_budget=thinking_budget, response_schema=schema,
    )
    async for attempt in _retry():
        with attempt:
            resp = await client.aio.models.generate_content(
                model=model or settings.gemini_model_fast,
                contents=prompt,
                config=cfg,
            )
            parsed = getattr(resp, "parsed", None)
            if parsed is not None:
                return parsed  # type: ignore[return-value]
            # Fallback: parse manually if SDK didn't auto-parse
            return schema.model_validate_json(resp.text or "{}")
    raise RuntimeError("structured generation failed after retries")


async def generate_stream(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    thinking_budget: int | None = None,
) -> AsyncIterator[str]:
    """Streamed text generation. Yields incremental chunks."""
    settings = get_settings()
    client = get_client()
    cfg = _build_config(system=system, temperature=temperature, thinking_budget=thinking_budget)
    stream = await client.aio.models.generate_content_stream(
        model=model or settings.gemini_model_fast,
        contents=prompt,
        config=cfg,
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text
