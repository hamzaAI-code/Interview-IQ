"""Analyze a candidate's answer along multiple axes."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.gemini_client import generate_structured
from app.prompts.analysis import ANALYSIS_SYSTEM, ANALYSIS_USER_TEMPLATE


class AnalysisResult(BaseModel):
    correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    depth: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_delta: float = Field(default=0.0, ge=0.0, le=1.0,
                                  description="how much new signal this answer added")
    contradiction: bool = False
    contradiction_evidence: str = Field(
        default="",
        description="If contradiction=true, a one-line quote/summary of what the candidate said that contradicts a specific resume claim, formatted as: 'Resume says X, but candidate answered Y'.",
    )
    evasive: bool = False
    follow_up_hint: str = ""
    reasoning: str = ""

    def composite_score(self) -> float:
        return round((self.correctness * 0.5 + self.depth * 0.3 + self.relevance * 0.2), 3)


async def analyze_answer(
    topic_name: str,
    candidate_claims: str,
    question: str,
    answer: str,
) -> AnalysisResult:
    settings = get_settings()
    user = ANALYSIS_USER_TEMPLATE.format(
        topic_name=topic_name,
        candidate_claims=candidate_claims or "(none)",
        question=question,
        answer=answer,
    )
    return await generate_structured(
        prompt=user,
        schema=AnalysisResult,
        system=ANALYSIS_SYSTEM,
        model=settings.gemini_model_lite,
        temperature=0.1,
        thinking_budget=0,   # mechanical scoring — disable Gemini internal reasoning
    )
