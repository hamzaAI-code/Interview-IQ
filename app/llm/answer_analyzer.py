"""Classify + analyze a candidate's response in a single Gemini call.

A single structured-output call handles ALL of:
  1. Classify response_type (answer / dodge / clarification_request / meta_question / off_topic).
  2. Score correctness, depth, relevance, coverage_delta — only meaningful when answer/dodge.
  3. Detect contradictions vs the resume claim.
  4. Generate a natural clarification_text the agent can say back when the response wasn't an answer.

This avoids a second round-trip and keeps perceived latency to ~analyzer time.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.gemini_client import generate_structured
from app.prompts.analysis import ANALYSIS_SYSTEM, ANALYSIS_USER_TEMPLATE


ResponseType = Literal[
    "answer",
    "dodge",
    "clarification_request",
    "meta_question",
    "off_topic",
    "terminate",
]


class AnalysisResult(BaseModel):
    response_type: ResponseType = Field(
        default="answer",
        description=(
            "Classification of the candidate's message. "
            "'answer' = they tried; "
            "'dodge' = explicit refusal/IDK; "
            "'clarification_request' = they asked us to rephrase/explain; "
            "'meta_question' = they asked us a question about ourselves/the process; "
            "'off_topic' = social/unrelated chatter."
        ),
    )
    correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    depth: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_delta: float = Field(default=0.0, ge=0.0, le=1.0,
                                  description="how much new signal this answer added")
    contradiction: bool = False
    contradiction_evidence: str = Field(
        default="",
        description="If contradiction=true, a one-line quote/summary formatted as: 'Resume says X, but candidate answered Y'.",
    )
    evasive: bool = False
    follow_up_hint: str = ""
    clarification_text: str = Field(
        default="",
        description=(
            "Natural 1-3 sentence reply to use VERBATIM when response_type is "
            "clarification_request / meta_question / off_topic. Should briefly "
            "address the candidate's input then re-ask the original question. "
            "Empty string for response_type = answer or dodge."
        ),
    )
    reasoning: str = ""

    def composite_score(self) -> float:
        return round((self.correctness * 0.5 + self.depth * 0.3 + self.relevance * 0.2), 3)

    @property
    def is_non_answer(self) -> bool:
        return self.response_type in {"clarification_request", "meta_question", "off_topic"}


async def analyze_answer(
    topic_name: str,
    candidate_claims: str,
    question: str,
    answer: str,
    *,
    recent_turns: str = "",
    attempts: int = 0,
    clarifications_used: int = 0,
) -> AnalysisResult:
    settings = get_settings()
    user = ANALYSIS_USER_TEMPLATE.format(
        topic_name=topic_name,
        candidate_claims=candidate_claims or "(none)",
        question=question,
        answer=answer,
        recent_turns=recent_turns or "(this is the first turn on this topic)",
        attempts=attempts,
        clarifications_used=clarifications_used,
    )
    return await generate_structured(
        prompt=user,
        schema=AnalysisResult,
        system=ANALYSIS_SYSTEM,
        model=settings.gemini_model_fast,
        temperature=0.1,
        thinking_budget=0,  # mechanical classification + scoring; thinking is wasted latency
    )
