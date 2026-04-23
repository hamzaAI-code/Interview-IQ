"""Interview state primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TopicStatus = Literal["pending", "active", "done", "skipped"]


@dataclass
class TopicState:
    name: str
    importance: float
    must_have: bool = False
    candidate_claims: str = ""           # short summary from resume graph (for UI)
    # Detailed claim evidence (used by question prompt to ground questions in resume):
    years: float = 0.0
    proficiency: str = ""
    evidence_text: str = ""
    projects: list[dict] = field(default_factory=list)      # [{name, summary}, ...]
    experiences: list[dict] = field(default_factory=list)   # [{role, company, summary}, ...]
    has_evidence: bool = False
    coverage: float = 0.0                # 0..1, advances when new signal added
    depth: float = 0.0                   # 0..1, max depth observed
    attempts: int = 0
    contradictions: list[str] = field(default_factory=list)
    qa: list[tuple[str, str, float]] = field(default_factory=list)  # (q, a, score)
    status: TopicStatus = "pending"

    def avg_score(self) -> float:
        if not self.qa:
            return 0.0
        return sum(s for _, _, s in self.qa) / len(self.qa)


@dataclass
class InterviewState:
    session_id: str
    topics: list[TopicState] = field(default_factory=list)
    current_idx: int | None = None
    rolling_summary: str = ""
    threshold: float = 0.70
    max_followups: int = 3
    finished: bool = False
    jd_title: str = ""
    # Speculative prefetch: opening question for the NEXT topic, pre-generated
    # while the analyzer ran. Consumed only if we actually advance to that topic.
    prefetched_question: str = ""
    prefetched_for_idx: int | None = None

    @property
    def current(self) -> TopicState | None:
        if self.current_idx is None:
            return None
        return self.topics[self.current_idx]

    def overall_score(self) -> float:
        scored = [t for t in self.topics if t.qa]
        if not scored:
            return 0.0
        weighted = sum(t.avg_score() * t.importance for t in scored)
        weights = sum(t.importance for t in scored)
        return weighted / weights if weights else 0.0
