"""Pydantic request/response models for the HTTP API.

Strict validation via Field(min_length, ge, le, ...). Internal types like
TopicState / InterviewState / TurnOutcome are flattened into HTTP-friendly
shapes so consumers don't depend on internal classes.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ----------------------------- Requests -----------------------------


class BuildSessionRequest(BaseModel):
    """Build a new session graph from already-parsed resume + JD text."""
    resume_text: str = Field(
        min_length=20,
        max_length=200_000,
        description="Plain text of the candidate's resume (PDF/DOCX must be parsed by the client).",
    )
    jd_text: str = Field(
        min_length=20,
        max_length=50_000,
        description="Plain text of the job description.",
    )
    session_id: str | None = Field(
        default=None,
        min_length=4,
        max_length=64,
        description=(
            "Optional caller-supplied session id. When set, the new graph is "
            "keyed by this id (used for cross-service correlation, e.g. when "
            "tekprep owns the session id and Interview-IQ runs the technical "
            "stage). When unset, the server generates one."
        ),
    )


class SubmitAnswerRequest(BaseModel):
    """Submit one user message in response to a question."""
    question: str = Field(min_length=1, description="The question that was asked (echoed back for binding).")
    answer: str = Field(min_length=1, max_length=20_000, description="The candidate's response text.")


# ----------------------------- Response components -----------------------------


TopicStatus = Literal["pending", "active", "done", "skipped"]
TopicSource = Literal["jd", "resume"]
ResponseTypeOut = Literal[
    "answer", "dodge", "clarification_request", "meta_question", "off_topic", "terminate"
]
HireRecommendation = Literal["strong_hire", "hire", "lean_hire", "no_hire"]


class TopicSummary(BaseModel):
    """Compact view of a single topic for client display."""
    name: str
    importance: float
    must_have: bool
    source: TopicSource
    coverage: float = Field(ge=0.0, le=1.0)
    depth: float = Field(ge=0.0, le=1.0)
    attempts: int = Field(ge=0)
    clarification_count: int = Field(ge=0)
    status: TopicStatus
    avg_score: float = Field(ge=0.0, le=1.0)


class SessionStateResponse(BaseModel):
    """The state any client needs to render an in-progress interview."""
    session_id: str
    jd_title: str
    finished: bool
    current_topic: str | None
    threshold: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    topics: list[TopicSummary]


class SessionSummary(BaseModel):
    """One row in the GET /sessions list."""
    id: str
    started_at: str | None = None
    candidate_name: str
    jd_title: str
    topic_count: int = Field(ge=0)


class SessionsListResponse(BaseModel):
    sessions: list[SessionSummary]


class QuestionResponse(BaseModel):
    """Result of generating the next question for a session."""
    question: str
    topic_name: str


class AnalysisSummary(BaseModel):
    """Slice of AnalysisResult exposed to clients."""
    response_type: ResponseTypeOut
    correctness: float = Field(ge=0.0, le=1.0)
    depth: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    coverage_delta: float = Field(ge=0.0, le=1.0)
    contradiction: bool
    contradiction_evidence: str = ""
    evasive: bool
    follow_up_hint: str = ""
    composite_score: float = Field(ge=0.0, le=1.0)


class TurnOutcomeResponse(BaseModel):
    """Result of POST /sessions/{id}/answers."""
    response_type: ResponseTypeOut
    attempt_consumed: bool
    advanced: bool
    terminated: bool
    keep_current_question: bool
    clarification_text: str = ""
    analysis: AnalysisSummary | None = None
    next_topic: str | None = None
    finished: bool


class TopicEvaluationResponse(BaseModel):
    topic: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: str


class EvaluationReportResponse(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    hire_recommendation: HireRecommendation
    strengths: list[str]
    weaknesses: list[str]
    per_topic: list[TopicEvaluationResponse]
    summary: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
