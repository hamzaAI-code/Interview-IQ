"""Interview orchestrator: state machine driving topic selection and follow-ups."""
from __future__ import annotations

import asyncio
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


# Short-answer patterns that bypass the LLM analyzer.
_EVASIVE_PHRASES = {
    "i don't know", "i dont know", "no idea", "not sure", "pass", "skip",
    "no", "idk", "dunno", "can't say", "cannot say", "next",
}


def _cheap_analysis(answer: str) -> AnalysisResult | None:
    """Return a synthetic AnalysisResult for obvious non-answers, else None."""
    stripped = answer.strip()
    lowered = stripped.lower().rstrip(".!?")
    if len(stripped) < 25 or lowered in _EVASIVE_PHRASES:
        return AnalysisResult(
            correctness=0.0,
            depth=0.0,
            relevance=0.0,
            coverage_delta=0.0,
            contradiction=False,
            evasive=True,
            follow_up_hint="candidate did not engage — next question should probe a concrete sub-area of the topic or confirm the gap.",
        )
    return None


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

    def _pop_prefetched(self) -> str | None:
        """Return prefetched question if it matches the current topic index, else None."""
        if (
            self.state.prefetched_question
            and self.state.prefetched_for_idx is not None
            and self.state.prefetched_for_idx == self.state.current_idx
        ):
            q = self.state.prefetched_question
            self.state.prefetched_question = ""
            self.state.prefetched_for_idx = None
            return q
        # Otherwise discard any stale prefetch
        if self.state.prefetched_question:
            self.state.prefetched_question = ""
            self.state.prefetched_for_idx = None
        return None

    async def next_question(self) -> str:
        assert self.state and self.state.current is not None, "interview not active"
        cached = self._pop_prefetched()
        if cached is not None:
            return cached
        mode, hint = self._question_mode()
        return await generate_question(self.state, self.state.current, mode, hint)

    def stream_next_question(self) -> AsyncIterator[str]:
        """Returns an async iterator the UI can drain.

        If a speculative prefetch matches the current topic, yields it in one chunk
        (the user sees the full question instantly instead of a token-by-token stream).
        """
        assert self.state and self.state.current is not None, "interview not active"
        cached = self._pop_prefetched()
        if cached is not None:
            async def _from_cache() -> AsyncIterator[str]:
                yield cached
            return _from_cache()
        mode, hint = self._question_mode()
        return stream_question(self.state, self.state.current, mode, hint)

    def _should_advance(self, analysis: AnalysisResult, topic: TopicState) -> bool:
        """Pure predicate: would we advance past this topic given this analysis? No side effects."""
        # A contradiction always buys at least one more follow-up (up to cap).
        if analysis.contradiction and topic.attempts < self.state.max_followups:
            return False
        coverage_met = topic.coverage >= self.state.threshold
        attempts_capped = topic.attempts >= self.state.max_followups
        return coverage_met or attempts_capped

    async def submit_answer(self, question: str, answer: str) -> AnalysisResult:
        assert self.state and self.state.current is not None, "no active topic"
        topic = self.state.current
        topic.attempts += 1

        # (#5) Short-answer bypass — don't spend an LLM round-trip on obvious non-answers.
        cheap = _cheap_analysis(answer)

        # Speculative prefetch (#3/#8): while the analyzer runs, pre-generate the
        # opening question for the NEXT topic. If we ultimately advance, the user
        # sees the next question with zero extra latency. If we stay on this topic
        # (follow-up needed), we cancel and discard the speculative task.
        next_idx = (self.state.current_idx or 0) + 1
        speculative: asyncio.Task | None = None
        if next_idx < len(self.state.topics):
            next_topic = self.state.topics[next_idx]
            speculative = asyncio.create_task(
                generate_question(self.state, next_topic, mode="opening")
            )

        if cheap is not None:
            analysis = cheap
        else:
            try:
                analysis = await analyze_answer(
                    topic_name=topic.name,
                    candidate_claims=topic.candidate_claims,
                    question=question,
                    answer=answer,
                )
            except Exception:
                if speculative:
                    speculative.cancel()
                raise

        # Update in-memory state from analysis result
        score = analysis.composite_score()
        topic.qa.append((question, answer, score))
        topic.coverage = min(1.0, topic.coverage + max(0.0, analysis.coverage_delta))
        topic.depth = max(topic.depth, analysis.depth)
        if analysis.contradiction:
            parts = [p for p in (analysis.contradiction_evidence, analysis.follow_up_hint) if p]
            if parts:
                topic.contradictions.append(" | ".join(parts))

        # (#4) Fire-and-forget the audit-trail write so we don't block the critical path.
        asyncio.create_task(self._safe_record_qa(topic.name, question, answer, score, analysis.contradiction))

        advancing = self._should_advance(analysis, topic)

        # Consume or discard the speculative prefetch.
        if speculative is not None:
            if advancing:
                try:
                    self.state.prefetched_question = await speculative
                    self.state.prefetched_for_idx = next_idx
                except Exception as e:
                    log.warning("speculative prefetch failed: %s", e)
                    self.state.prefetched_question = ""
                    self.state.prefetched_for_idx = None
            else:
                speculative.cancel()

        if advancing:
            await self._execute_advance()

        return analysis

    async def _safe_record_qa(
        self, topic_name: str, question: str, answer: str, score: float, contradiction: bool
    ) -> None:
        try:
            await record_qa(
                self.state.session_id,  # type: ignore[union-attr]
                topic_name,
                question,
                answer,
                score,
                contradiction,
                datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:  # audit-trail write; don't surface to UI
            log.warning("record_qa failed (non-fatal): %s", e)

    async def _execute_advance(self) -> None:
        topic = self.state.current  # type: ignore[union-attr]
        assert topic is not None
        topic.status = "done"
        try:
            await compress_completed_topic(self.state, topic)
        except Exception as e:
            log.warning("topic compression failed: %s", e)

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
