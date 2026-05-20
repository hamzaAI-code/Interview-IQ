"""Routes for the interview conversation loop."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.manager import SessionManager, get_session_manager
from app.api.schemas import (
    AnalysisSummary,
    EvaluationReportResponse,
    QuestionResponse,
    SubmitAnswerRequest,
    TopicEvaluationResponse,
    TurnOutcomeResponse,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["interview"])


async def _require_active(session_id: str, mgr: SessionManager):
    orch = await mgr.get(session_id)
    if orch is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Session '{session_id}' is not attached. POST /sessions/{session_id}/load first.",
        )
    if orch.state is None or orch.state.finished:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Interview is already finished — call POST /sessions/{id}/finalize for the report.",
        )
    if orch.state.current is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No current topic — interview state is inconsistent.",
        )
    return orch


@router.post(
    "/{session_id}/questions/next",
    response_model=QuestionResponse,
    summary="Generate (or return prefetched) next question for the current topic",
)
async def next_question(
    session_id: str,
    mgr: SessionManager = Depends(get_session_manager),
) -> QuestionResponse:
    orch = await _require_active(session_id, mgr)
    text = await orch.next_question()
    return QuestionResponse(
        question=text,
        topic_name=orch.state.current.name,  # type: ignore[union-attr]
    )


@router.post(
    "/{session_id}/questions/next/stream",
    summary="Stream the next question token-by-token as Server-Sent Events",
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_next_question(
    session_id: str,
    mgr: SessionManager = Depends(get_session_manager),
):
    orch = await _require_active(session_id, mgr)
    topic_name = orch.state.current.name  # type: ignore[union-attr]

    async def event_stream():
        # Initial topic event so the client knows what's being asked about.
        yield f"event: topic\ndata: {json.dumps({'topic': topic_name})}\n\n"
        async for chunk in orch.stream_next_question():
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/{session_id}/answers",
    response_model=TurnOutcomeResponse,
    summary="Submit an answer; returns classification + analysis + state delta",
)
async def submit_answer(
    session_id: str,
    req: SubmitAnswerRequest,
    mgr: SessionManager = Depends(get_session_manager),
) -> TurnOutcomeResponse:
    orch = await mgr.get(session_id)
    if orch is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Session '{session_id}' is not attached.",
        )
    if orch.state is None or orch.state.finished:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Interview is already finished — POST /sessions/{id}/finalize for the report.",
        )
    if orch.state.current is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No current topic — interview state is inconsistent.",
        )
    outcome = await orch.submit_answer(req.question, req.answer)

    analysis_summary = None
    if outcome.analysis is not None:
        a = outcome.analysis
        analysis_summary = AnalysisSummary(
            response_type=a.response_type,  # type: ignore[arg-type]
            correctness=a.correctness,
            depth=a.depth,
            relevance=a.relevance,
            coverage_delta=a.coverage_delta,
            contradiction=a.contradiction,
            contradiction_evidence=a.contradiction_evidence,
            evasive=a.evasive,
            follow_up_hint=a.follow_up_hint,
            composite_score=a.composite_score(),
        )

    return TurnOutcomeResponse(
        response_type=outcome.response_type,  # type: ignore[arg-type]
        attempt_consumed=outcome.attempt_consumed,
        advanced=outcome.advanced,
        terminated=outcome.terminated,
        keep_current_question=outcome.keep_current_question,
        clarification_text=outcome.clarification_text,
        analysis=analysis_summary,
        next_topic=(orch.state.current.name if (orch.state and orch.state.current) else None),
        finished=bool(orch.state and orch.state.finished),
    )


@router.post(
    "/{session_id}/finalize",
    response_model=EvaluationReportResponse,
    summary="Generate the final structured evaluation report",
)
async def finalize(
    session_id: str,
    mgr: SessionManager = Depends(get_session_manager),
) -> EvaluationReportResponse:
    orch = await mgr.get(session_id)
    if orch is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Session '{session_id}' is not attached.",
        )
    report = await orch.finalize()
    return EvaluationReportResponse(
        overall_score=report.overall_score,
        hire_recommendation=report.hire_recommendation,  # type: ignore[arg-type]
        strengths=report.strengths,
        weaknesses=report.weaknesses,
        per_topic=[
            TopicEvaluationResponse(topic=t.topic, score=t.score, verdict=t.verdict)
            for t in report.per_topic
        ],
        summary=report.summary,
    )
