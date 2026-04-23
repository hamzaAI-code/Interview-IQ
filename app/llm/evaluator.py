"""Final evaluation report for a completed interview."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.state import InterviewState
from app.llm.gemini_client import generate_structured
from app.prompts.evaluation import EVALUATION_SYSTEM, EVALUATION_USER_TEMPLATE


class TopicEvaluation(BaseModel):
    topic: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: str


class EvaluationReport(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    hire_recommendation: Literal["strong_hire", "hire", "lean_hire", "no_hire"] = "lean_hire"
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    per_topic: list[TopicEvaluation] = Field(default_factory=list)
    summary: str = ""


def _topic_table(state: InterviewState) -> str:
    rows = []
    for t in state.topics:
        rows.append(
            f"- {t.name} | imp={t.importance:.1f} | cov={t.coverage:.2f} | "
            f"depth={t.depth:.2f} | attempts={t.attempts} | "
            f"contradictions={len(t.contradictions)} | qa_count={len(t.qa)} | "
            f"avg_score={t.avg_score():.2f}"
        )
    return "\n".join(rows)


def _transcript(state: InterviewState, max_chars: int = 6000) -> str:
    parts = []
    for t in state.topics:
        if not t.qa:
            continue
        parts.append(f"### {t.name}")
        for q, a, s in t.qa:
            parts.append(f"Q: {q}\nA: {a}\n(score={s:.2f})")
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


async def final_evaluation(state: InterviewState) -> EvaluationReport:
    settings = get_settings()
    user = EVALUATION_USER_TEMPLATE.format(
        jd_title=state.jd_title or "(unspecified)",
        threshold=state.threshold,
        topic_table=_topic_table(state),
        transcript=_transcript(state),
    )
    return await generate_structured(
        prompt=user,
        schema=EvaluationReport,
        system=EVALUATION_SYSTEM,
        model=settings.gemini_model_pro,
        temperature=0.2,
    )
