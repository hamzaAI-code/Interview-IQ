"""Generate the next interview question (streaming + non-streaming)."""
from __future__ import annotations

from typing import AsyncIterator

from app.config import get_settings
from app.core.context_manager import build_question_vars
from app.core.state import InterviewState, TopicState
from app.llm.gemini_client import generate_stream, generate_text
from app.prompts.question import QUESTION_SYSTEM, QUESTION_USER_TEMPLATE


async def generate_question(
    state: InterviewState,
    topic: TopicState,
    mode: str = "opening",
    followup_hint: str = "",
) -> str:
    settings = get_settings()
    user = QUESTION_USER_TEMPLATE.format(**build_question_vars(state, topic, mode, followup_hint))
    return await generate_text(
        prompt=user,
        system=QUESTION_SYSTEM,
        model=settings.gemini_model_fast,
        temperature=0.6,
    )


async def stream_question(
    state: InterviewState,
    topic: TopicState,
    mode: str = "opening",
    followup_hint: str = "",
) -> AsyncIterator[str]:
    settings = get_settings()
    user = QUESTION_USER_TEMPLATE.format(**build_question_vars(state, topic, mode, followup_hint))
    async for chunk in generate_stream(
        prompt=user,
        system=QUESTION_SYSTEM,
        model=settings.gemini_model_fast,
        temperature=0.6,
    ):
        yield chunk
