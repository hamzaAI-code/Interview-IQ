"""Routes for session lifecycle: build, list, load, get, delete."""
from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.api.manager import SessionManager, get_session_manager
from app.api.schemas import (
    BuildSessionRequest,
    SessionsListResponse,
    SessionStateResponse,
    SessionSummary,
    StructuredBuildRequest,
    TopicSummary,
)
from app.core.orchestrator import Orchestrator
from app.services.external_adapter import (
    tekprep_jd_to_graph,
    tekprep_resume_to_graphs,
)
from app.services.jd_parser import parse_jd
from app.services.resume_parser import parse_resume_bytes
from app.services.session_builder import (
    build_session,
    build_session_from_graphs,
    list_sessions,
    load_session,
    teardown_session,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class BuildAcceptedResponse(BaseModel):
    """202 reply for fire-and-forget builds. Caller already knows the
    session_id (it supplied it); we just confirm acceptance."""
    session_id: str
    status: str = "accepted"


async def _build_and_attach(
    mgr: SessionManager,
    *,
    resume_text: str,
    jd_text: str,
    session_id: str,
) -> None:
    """Background-mode build — runs after the HTTP response has returned.
    Builds the InterviewState and attaches it to the session manager so
    subsequent /answers / /questions/next/stream calls find it. Logs and
    swallows exceptions so a failed build doesn't blow up the worker."""
    try:
        state = await build_session(resume_text, jd_text, session_id=session_id)
        orch = Orchestrator()
        orch.attach(state)
        await mgr.attach(orch)
    except Exception as exc:
        log.exception("background build failed for session=%s: %s", session_id, exc)


async def _build_structured_and_attach(
    mgr: SessionManager,
    *,
    req: StructuredBuildRequest,
    session_id: str,
) -> None:
    """Same as _build_and_attach but for the pre-parsed (tekprep) input
    shape. Skips Gemini extraction entirely."""
    try:
        core, depth = tekprep_resume_to_graphs(req.resume)
        jd_graph = tekprep_jd_to_graph(req.jd, resume=req.resume)
        state = await build_session_from_graphs(
            core, depth, jd_graph, session_id=session_id,
        )
        orch = Orchestrator()
        orch.attach(state)
        await mgr.attach(orch)
    except Exception as exc:
        log.exception(
            "background structured build failed for session=%s: %s",
            session_id, exc,
        )


def _state_to_response(orch: Orchestrator) -> SessionStateResponse:
    state = orch.state
    assert state is not None
    return SessionStateResponse(
        session_id=state.session_id,
        jd_title=state.jd_title or "Role",
        finished=state.finished,
        current_topic=(state.current.name if state.current else None),
        threshold=state.threshold,
        overall_score=state.overall_score(),
        topics=[
            TopicSummary(
                name=t.name,
                importance=t.importance,
                must_have=t.must_have,
                source=t.source,  # type: ignore[arg-type]
                coverage=t.coverage,
                depth=t.depth,
                attempts=t.attempts,
                clarification_count=t.clarification_count,
                status=t.status,
                avg_score=t.avg_score(),
            )
            for t in state.topics
        ],
    )


@router.post(
    "/build",
    status_code=status.HTTP_201_CREATED,
    summary="Build a new session graph from resume + JD text",
    description=(
        "Synchronous build by default — returns the SessionStateResponse once "
        "the graph is ready. Pass `?background=true` to fire-and-forget: the "
        "build runs after the response is sent (HTTP 202 with the session_id), "
        "and subsequent `/answers` calls block until the build finishes. Used "
        "by tekprep so the voice intro can run in parallel with graph build."
    ),
    response_model=None,
)
async def build(
    req: BuildSessionRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        default=False,
        description="When true, build runs in the background after HTTP 202 returns.",
    ),
    mgr: SessionManager = Depends(get_session_manager),
):
    if background:
        # Caller MUST supply session_id in background mode — they need the id
        # before the build finishes so they can call subsequent endpoints.
        session_id = req.session_id or uuid4().hex[:12]
        background_tasks.add_task(
            _build_and_attach,
            mgr,
            resume_text=req.resume_text,
            jd_text=req.jd_text,
            session_id=session_id,
        )
        return BuildAcceptedResponse(session_id=session_id)

    state = await build_session(req.resume_text, req.jd_text, session_id=req.session_id)
    orch = Orchestrator()
    orch.attach(state)
    await mgr.attach(orch)
    return _state_to_response(orch)


@router.post(
    "/build/upload",
    response_model=SessionStateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Build a new session by uploading a resume file (PDF/DOCX/TXT) + JD text",
)
async def build_upload(
    resume: UploadFile = File(..., description="PDF / DOCX / TXT resume file"),
    jd_text: str = Form(..., min_length=20, max_length=50_000),
    mgr: SessionManager = Depends(get_session_manager),
) -> SessionStateResponse:
    if not resume.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing resume filename")
    suffix = (resume.filename.rsplit(".", 1)[-1] or "").lower()
    if suffix not in {"pdf", "docx", "doc", "txt", "md"}:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Unsupported resume type: .{suffix}")
    raw = await resume.read()
    try:
        resume_text = parse_resume_bytes(raw, resume.filename)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not parse resume: {e}")
    state = await build_session(resume_text, parse_jd(jd_text))
    orch = Orchestrator()
    orch.attach(state)
    await mgr.attach(orch)
    return _state_to_response(orch)


@router.post(
    "/build/structured",
    status_code=status.HTTP_201_CREATED,
    summary="Build a new session graph from already-parsed resume + JD payload",
    description=(
        "Accepts the tekprep-shaped ResumeSchema + JDSchema instead of raw "
        "text. Skips Gemini extraction entirely — the payload is adapted to "
        "Interview-IQ's internal graph shape and ingested directly. Pass "
        "`?background=true` for fire-and-forget (HTTP 202) — same semantics "
        "as POST /sessions/build."
    ),
    response_model=None,
)
async def build_structured(
    req: StructuredBuildRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        default=False,
        description="When true, build runs in the background after HTTP 202 returns.",
    ),
    mgr: SessionManager = Depends(get_session_manager),
):
    if background:
        session_id = req.session_id or uuid4().hex[:12]
        background_tasks.add_task(
            _build_structured_and_attach,
            mgr,
            req=req,
            session_id=session_id,
        )
        return BuildAcceptedResponse(session_id=session_id)

    core, depth = tekprep_resume_to_graphs(req.resume)
    jd_graph = tekprep_jd_to_graph(req.jd)
    state = await build_session_from_graphs(
        core, depth, jd_graph, session_id=req.session_id,
    )
    orch = Orchestrator()
    orch.attach(state)
    await mgr.attach(orch)
    return _state_to_response(orch)


@router.get(
    "",
    response_model=SessionsListResponse,
    summary="List all sessions in Neo4j (most recent first)",
)
async def list_all() -> SessionsListResponse:
    rows = await list_sessions()
    return SessionsListResponse(
        sessions=[
            SessionSummary(
                id=r.get("id") or "",
                started_at=r.get("started_at"),
                candidate_name=r.get("candidate_name") or "Candidate",
                jd_title=r.get("jd_title") or "Role",
                topic_count=int(r.get("topic_count") or 0),
            )
            for r in rows
        ]
    )


@router.post(
    "/{session_id}/load",
    response_model=SessionStateResponse,
    summary="Attach an existing graph session to an in-memory Orchestrator",
)
async def load(
    session_id: str,
    mgr: SessionManager = Depends(get_session_manager),
) -> SessionStateResponse:
    try:
        state = await load_session(session_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    orch = Orchestrator()
    orch.attach(state)
    await mgr.attach(orch)
    return _state_to_response(orch)


@router.get(
    "/{session_id}",
    response_model=SessionStateResponse,
    summary="Read the current InterviewState (must be loaded into memory first)",
)
async def get_state(
    session_id: str,
    mgr: SessionManager = Depends(get_session_manager),
) -> SessionStateResponse:
    orch = await mgr.get(session_id)
    if orch is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Session '{session_id}' is not attached. Call POST /sessions/{session_id}/load first "
            "(or POST /sessions/build for a new one).",
        )
    return _state_to_response(orch)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Tear down a session: wipe its graph subgraph and drop its in-memory orchestrator",
)
async def delete(
    session_id: str,
    mgr: SessionManager = Depends(get_session_manager),
) -> None:
    await teardown_session(session_id)
    await mgr.remove(session_id)
    return None
