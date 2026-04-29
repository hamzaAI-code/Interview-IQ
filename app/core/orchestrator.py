"""Interview orchestrator: state machine driving topic selection and follow-ups.

Knows nothing about how the InterviewState was built — see app/services/session_builder.py
for the parse/extract/ingest/topic pipeline. This module only runs the conversation.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from app.config import get_settings
from app.core.context_manager import compress_completed_topic, recent_turns
from app.core.state import InterviewState, TopicState
from app.graph.matcher import record_qa
from app.llm.answer_analyzer import AnalysisResult, analyze_answer
from app.llm.clarifier import generate_meta_response
from app.llm.evaluator import EvaluationReport, final_evaluation
from app.llm.question_generator import generate_question, stream_question
from app.utils.logger import get_logger

log = get_logger(__name__)


# Phrases that are obviously dodges — handled instantly without an LLM round-trip.
_DODGE_PHRASES = {
    "i don't know", "i dont know", "no idea", "not sure", "pass", "skip",
    "no", "idk", "dunno", "can't say", "cannot say", "next",
}


def _cheap_dodge(answer: str) -> "AnalysisResult | None":
    """Detect obvious dodges with zero LLM cost. None = let the LLM classify."""
    stripped = answer.strip()
    lowered = stripped.lower().rstrip(".!?")
    if len(stripped) < 8 or lowered in _DODGE_PHRASES:
        return AnalysisResult(
            response_type="dodge",
            evasive=True,
            follow_up_hint="candidate did not engage — next question should probe a concrete sub-area or confirm the gap.",
        )
    return None


@dataclass
class TurnOutcome:
    """What happened on a single answer submission. Returned by submit_answer."""
    analysis: AnalysisResult
    response_type: str = "answer"
    attempt_consumed: bool = True
    clarification_text: str = ""
    keep_current_question: bool = False  # True for non-answer turns
    advanced: bool = False               # True if topic changed after this turn
    terminated: bool = False             # True if candidate asked to end the interview


class Orchestrator:
    """Single-session interview controller. Hold one per Streamlit user session."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.state: InterviewState | None = None

    # ---------- Lifecycle ----------

    def attach(self, state: InterviewState) -> None:
        """Bind a pre-built InterviewState. Built by services.session_builder.build_session."""
        self.state = state
        log.info("orchestrator attached to session=%s topics=%d",
                 state.session_id, len(state.topics))

    # ---------- Question / answer cycle ----------

    def _question_mode(self) -> tuple[str, str]:
        """Return (mode, followup_hint) for the current topic.

        - "opening"    : very first question of the interview (first topic, no Q&A yet).
        - "transition" : first question on a topic that ISN'T the first topic — produces a bridge.
        - "followup"   : already asked at least one question on this topic.
        """
        topic = self.state.current  # type: ignore[union-attr]
        if topic is None:
            return ("opening", "")
        if not topic.qa:
            mode = "transition" if (self.state.current_idx or 0) > 0 else "opening"
            return (mode, "")
        # Follow-up: pursue contradictions first, then depth
        hint = ""
        if topic.contradictions:
            hint = topic.contradictions[-1]
        return ("followup", hint)

    def _pop_prefetched(self) -> str | None:
        """Return the prefetched question if it matches the current topic; else None."""
        if (
            self.state.prefetched_question
            and self.state.prefetched_for_idx is not None
            and self.state.prefetched_for_idx == self.state.current_idx
        ):
            q = self.state.prefetched_question
            self.state.prefetched_question = ""
            self.state.prefetched_for_idx = None
            return q
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
        assert self.state and self.state.current is not None, "interview not active"
        cached = self._pop_prefetched()
        if cached is not None:
            async def _from_cache() -> AsyncIterator[str]:
                yield cached
            return _from_cache()
        mode, hint = self._question_mode()
        return stream_question(self.state, self.state.current, mode, hint)

    def _should_advance(self, analysis: AnalysisResult, topic: TopicState) -> bool:
        """Pure predicate: would we advance past this topic given this analysis?"""
        if analysis.contradiction and topic.attempts < self.state.max_followups:
            return False
        coverage_met = topic.coverage >= self.state.threshold
        attempts_capped = topic.attempts >= self.state.max_followups
        return coverage_met or attempts_capped

    async def submit_answer(self, question: str, answer: str) -> TurnOutcome:
        """Process one user message.

        Branches on response_type:
          - terminate              → finish session, return brief acknowledgment.
          - meta_question          → graph-on-demand profile fetch + LLM-generated reply,
                                     don't consume attempt, keep current question.
          - clarification_request
          - off_topic              → use analyzer's clarification_text, don't consume attempt.
          - answer / dodge         → score, update state, possibly advance.

        Speculative next-topic prefetch and Neo4j audit write run in parallel with the analyzer.
        Anti-gaming: after `max_clarifications` non-answers on a topic, further non-answers
        are promoted to dodges so the candidate can't game the budget.
        """
        assert self.state and self.state.current is not None, "no active topic"
        topic = self.state.current

        # Speculate the NEXT topic's bridging opening question while we wait.
        # Cancelled if we don't actually advance.
        next_idx = (self.state.current_idx or 0) + 1
        speculative: asyncio.Task | None = None
        if next_idx < len(self.state.topics):
            next_topic = self.state.topics[next_idx]
            speculative = asyncio.create_task(
                generate_question(self.state, next_topic, mode="transition")
            )

        # Cheap dodge bypass.
        cheap = _cheap_dodge(answer)
        if cheap is not None:
            analysis = cheap
        else:
            try:
                analysis = await analyze_answer(
                    topic_name=topic.name,
                    candidate_claims=topic.candidate_claims,
                    question=question,
                    answer=answer,
                    recent_turns=recent_turns(topic),
                    attempts=topic.attempts,
                    clarifications_used=topic.clarification_count,
                )
            except Exception:
                if speculative:
                    speculative.cancel()
                raise

        rtype = analysis.response_type

        # ── 1. Terminate intent — honor immediately ────────────────────────
        if rtype == "terminate":
            if speculative:
                speculative.cancel()
            self.state.finished = True
            self.state.current_idx = None
            wrap = (analysis.clarification_text or "").strip() or (
                "Got it — wrapping up the interview here. I'll put together your evaluation now."
            )
            log.info("session=%s terminated by candidate", self.state.session_id)
            return TurnOutcome(
                analysis=analysis,
                response_type="terminate",
                attempt_consumed=False,
                clarification_text=wrap,
                keep_current_question=False,
                advanced=False,
                terminated=True,
            )

        # ── 2. Non-answer branch ───────────────────────────────────────────
        if rtype in ("clarification_request", "meta_question", "off_topic"):
            topic.clarification_count += 1

            # Anti-gaming: budget exhausted → promote to dodge, fall through.
            if topic.clarification_count > self.settings.max_clarifications:
                log.info(
                    "topic=%s clarification_count=%d exceeded budget — promoting to dodge",
                    topic.name, topic.clarification_count,
                )
                analysis.response_type = "dodge"
                analysis.evasive = True
                analysis.follow_up_hint = (
                    analysis.follow_up_hint
                    or "candidate has used clarifications repeatedly; next question should pivot to a concrete sub-area."
                )
                analysis.clarification_text = ""
                rtype = "dodge"
                # fall through to answer/dodge branch
            else:
                # Free pass.
                if speculative:
                    speculative.cancel()

                if rtype == "meta_question":
                    # Graph-on-demand: build a content-rich response anchored in
                    # full resume + JD + the current topic's specific projects/experiences.
                    current_topic_dict = {
                        "name": topic.name,
                        "candidate_claims": topic.candidate_claims,
                        "projects": topic.projects,
                        "experiences": topic.experiences,
                    }
                    try:
                        clarif = await generate_meta_response(
                            profile=self.state.candidate_profile or {},
                            jd_profile=self.state.jd_profile or {},
                            current_topic=current_topic_dict,
                            original_question=question,
                            user_msg=answer,
                            rolling_summary=self.state.rolling_summary or "",
                        )
                    except Exception as e:
                        log.warning("meta response generation failed: %s", e)
                        clarif = ""
                else:
                    clarif = (analysis.clarification_text or "").strip()

                if not clarif:
                    clarif = f"Let's stay on this. {question}"

                return TurnOutcome(
                    analysis=analysis,
                    response_type=rtype,
                    attempt_consumed=False,
                    clarification_text=clarif,
                    keep_current_question=True,
                    advanced=False,
                )

        # ── 3. Answer / dodge branch ───────────────────────────────────────
        topic.attempts += 1
        score = analysis.composite_score()
        topic.qa.append((question, answer, score))
        topic.coverage = min(1.0, topic.coverage + max(0.0, analysis.coverage_delta))
        topic.depth = max(topic.depth, analysis.depth)
        if analysis.contradiction:
            parts = [p for p in (analysis.contradiction_evidence, analysis.follow_up_hint) if p]
            if parts:
                topic.contradictions.append(" | ".join(parts))

        # Fire-and-forget audit write.
        asyncio.create_task(self._safe_record_qa(
            topic.name, question, answer, score, analysis.contradiction,
        ))

        advancing = self._should_advance(analysis, topic)

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

        return TurnOutcome(
            analysis=analysis,
            response_type=rtype,
            attempt_consumed=True,
            clarification_text="",
            keep_current_question=False,
            advanced=advancing,
        )

    async def _safe_record_qa(
        self, topic_name: str, question: str, answer: str, score: float, contradiction: bool,
    ) -> None:
        try:
            await record_qa(
                self.state.session_id,  # type: ignore[union-attr]
                topic_name, question, answer, score, contradiction,
                datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
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
