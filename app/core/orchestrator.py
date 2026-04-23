"""Interview orchestrator: state machine driving topic selection and follow-ups."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

from app.config import get_settings
from app.core.context_manager import compress_completed_topic
from app.core.state import InterviewState, TopicState
from app.graph.builder import ingest_both
from app.graph.matcher import derive_topics, record_qa
from app.graph.schema import ensure_schema, wipe_session
from app.llm.answer_analyzer import AnalysisResult, analyze_answer
from app.llm.evaluator import EvaluationReport, final_evaluation
from app.llm.extractors import extract_both
from app.llm.question_generator import generate_question, stream_question
from app.utils.logger import get_logger

log = get_logger(__name__)


class Orchestrator:
    """Single-session interview controller. Hold one per Streamlit user session."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.state: InterviewState | None = None
        self._schema_ready = False

    # ---------- Lifecycle ----------

    async def start(self, resume_text: str, jd_text: str) -> InterviewState:
        if not self._schema_ready:
            await ensure_schema()
            self._schema_ready = True

        session_id = uuid4().hex[:12]
        log.info("starting session=%s", session_id)

        # Parallel extraction → parallel ingestion
        rg, jg = await extract_both(resume_text, jd_text)
        await ingest_both(session_id, rg, jg)

        # Derive ranked topics
        topic_rows = await derive_topics(session_id, max_topics=self.settings.max_topics)

        topics = [
            TopicState(
                name=row["name"],
                importance=float(row["importance"]),
                must_have=bool(row["must_have"]),
                candidate_claims=row["candidate_claims"],
                years=float(row.get("years") or 0.0),
                proficiency=row.get("proficiency") or "",
                evidence_text=row.get("evidence_text") or "",
                projects=list(row.get("projects_list") or []),
                experiences=list(row.get("experiences_list") or []),
                has_evidence=bool(row.get("has_evidence", False)),
            )
            for row in topic_rows
        ]

        self.state = InterviewState(
            session_id=session_id,
            topics=topics,
            current_idx=0 if topics else None,
            jd_title=jg.role_title,
            threshold=self.settings.coverage_threshold,
            max_followups=self.settings.max_followups,
            finished=not topics,
        )
        if topics:
            topics[0].status = "active"
        log.info("session=%s topics=%d", session_id, len(topics))
        return self.state

    async def cleanup(self) -> None:
        if self.state:
            await wipe_session(self.state.session_id)

    # ---------- Question / answer cycle ----------

    def _question_mode(self) -> tuple[str, str]:
        """Determine mode + follow-up hint for the current topic."""
        topic = self.state.current  # type: ignore[union-attr]
        if topic is None:
            return ("opening", "")
        if not topic.qa:
            return ("opening", "")
        # Follow-up: pursue contradictions first, then depth
        hint = ""
        if topic.contradictions:
            hint = topic.contradictions[-1]
        return ("followup", hint)

    async def next_question(self) -> str:
        assert self.state and self.state.current is not None, "interview not active"
        mode, hint = self._question_mode()
        return await generate_question(self.state, self.state.current, mode, hint)

    def stream_next_question(self) -> AsyncIterator[str]:
        """Returns an async iterator the UI can drain."""
        assert self.state and self.state.current is not None, "interview not active"
        mode, hint = self._question_mode()
        return stream_question(self.state, self.state.current, mode, hint)

    async def submit_answer(self, question: str, answer: str) -> AnalysisResult:
        assert self.state and self.state.current is not None, "no active topic"
        topic = self.state.current
        topic.attempts += 1

        analysis = await analyze_answer(
            topic_name=topic.name,
            candidate_claims=topic.candidate_claims,
            question=question,
            answer=answer,
        )
        score = analysis.composite_score()
        topic.qa.append((question, answer, score))
        topic.coverage = min(1.0, topic.coverage + max(0.0, analysis.coverage_delta))
        topic.depth = max(topic.depth, analysis.depth)
        if analysis.contradiction:
            parts = [p for p in (analysis.contradiction_evidence, analysis.follow_up_hint) if p]
            if parts:
                topic.contradictions.append(" | ".join(parts))

        # Persist Q&A on the topic in graph
        await record_qa(
            self.state.session_id,
            topic.name,
            question,
            answer,
            score,
            analysis.contradiction,
            datetime.now(timezone.utc).isoformat(),
        )

        await self._maybe_advance(analysis)
        return analysis

    async def _maybe_advance(self, analysis: AnalysisResult) -> None:
        topic = self.state.current  # type: ignore[union-attr]
        assert topic is not None

        # Always probe at least one follow-up if a contradiction is detected
        if analysis.contradiction and topic.attempts < self.state.max_followups:
            return

        coverage_met = topic.coverage >= self.state.threshold
        attempts_capped = topic.attempts >= self.state.max_followups
        if not (coverage_met or attempts_capped):
            return

        topic.status = "done"
        try:
            await compress_completed_topic(self.state, topic)
        except Exception as e:
            log.warning("topic compression failed: %s", e)

        # Advance pointer
        next_idx = (self.state.current_idx or 0) + 1
        if next_idx >= len(self.state.topics):
            self.state.current_idx = None
            self.state.finished = True
            log.info("interview finished session=%s", self.state.session_id)
            return
        self.state.current_idx = next_idx
        self.state.topics[next_idx].status = "active"

    # ---------- Final report ----------

    async def finalize(self) -> EvaluationReport:
        assert self.state, "no active session"
        return await final_evaluation(self.state)
