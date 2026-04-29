"""Session bootstrap: turn raw resume + JD text into a ready-to-use InterviewState.

This is the ONLY place that touches the graph-build pipeline (extraction,
ingestion, topic ranking, profile fetches). The orchestrator consumes the
finished state and runs the interview — it knows nothing about how the graph
was constructed.

Responsibilities:
- Idempotent schema DDL (cached across sessions in this process).
- Parallel resume-core / resume-depth / JD extraction (Gemini, Flash, thinking off).
- Per-side batched ingestion into Neo4j with depth waiting on core.
- Topic derivation + candidate profile + JD profile fetched in parallel.
- Returns a constructed InterviewState; pointer set to topic 0.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

from app.config import get_settings
from app.core.state import InterviewState, TopicState
from app.graph.builder import (
    create_session,
    ingest_jd,
    ingest_resume_core,
    ingest_resume_depth,
)
from app.graph.matcher import derive_topics
from app.graph.neo4j_client import run
from app.graph.profile import get_candidate_profile, get_jd_profile
from app.graph.schema import ensure_schema, wipe_session
from app.llm.extractors import (
    extract_jd,
    extract_resume_core,
    extract_resume_depth,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


# Module-level schema-ready cache. The DDL is idempotent but we still avoid the
# round-trip on every session. asyncio.Lock guards the very first call so
# concurrent setups (rare in our Streamlit app, but possible) can't race.
_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ensure_schema_once() -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        await ensure_schema()
        _schema_ready = True


async def build_session(resume_text: str, jd_text: str) -> InterviewState:
    """Build a complete InterviewState from raw resume + JD text."""
    settings = get_settings()
    session_id = uuid4().hex[:12]
    log.info("building session=%s", session_id)

    schema_task = asyncio.create_task(_ensure_schema_once())

    # All three extractions fire concurrently.
    core_extract = asyncio.create_task(extract_resume_core(resume_text))
    depth_extract = asyncio.create_task(extract_resume_depth(resume_text))
    jd_extract = asyncio.create_task(extract_jd(jd_text))

    # Schema must land before we start writing.
    await schema_task

    # Session root + per-side ingestion pipelines.
    await create_session(session_id)

    async def _core_pipeline():
        core = await core_extract
        await ingest_resume_core(session_id, core)
        return core

    async def _jd_pipeline():
        jd = await jd_extract
        await ingest_jd(session_id, jd)
        return jd

    core_task = asyncio.create_task(_core_pipeline())
    jd_task = asyncio.create_task(_jd_pipeline())

    async def _depth_pipeline():
        depth = await depth_extract
        await core_task  # depth ingestion needs core skill nodes to exist
        await ingest_resume_depth(session_id, depth)
        return depth

    depth_task = asyncio.create_task(_depth_pipeline())

    _, jg, _ = await asyncio.gather(core_task, jd_task, depth_task)

    # Three reads on the freshly-built session subgraph in one round trip.
    topic_rows, profile, jd_profile = await asyncio.gather(
        derive_topics(session_id, max_topics=settings.max_topics),
        get_candidate_profile(session_id),
        get_jd_profile(session_id),
    )

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

    state = InterviewState(
        session_id=session_id,
        topics=topics,
        current_idx=0 if topics else None,
        jd_title=jg.role_title,
        threshold=settings.coverage_threshold,
        max_followups=settings.max_followups,
        finished=not topics,
        candidate_profile=profile or {},
        jd_profile=jd_profile or {},
    )
    if topics:
        topics[0].status = "active"

    log.info(
        "session=%s built topics=%d profile_skills=%d jd_reqs=%d",
        session_id, len(topics),
        len((profile or {}).get("skills") or []),
        len((jd_profile or {}).get("requirements") or []),
    )
    return state


async def list_sessions() -> list[dict]:
    """List sessions already in the graph (most recent first).

    Each row: {id, started_at, candidate_name, jd_title, topic_count}.
    """
    rows = await run(
        """
        MATCH (s:Session)
        OPTIONAL MATCH (s)-[:HAS_CANDIDATE]->(c:Candidate)
        OPTIONAL MATCH (s)-[:HAS_JD]->(j:JobDescription)
        OPTIONAL MATCH (s)-[:HAS_TOPIC]->(t:Topic)
        RETURN s.id            AS id,
               s.started_at    AS started_at,
               coalesce(c.name, 'Candidate') AS candidate_name,
               coalesce(j.title, 'Role')     AS jd_title,
               count(DISTINCT t)             AS topic_count
        ORDER BY s.started_at DESC
        """
    )
    return rows or []


async def load_session(session_id: str) -> InterviewState:
    """Build a fresh InterviewState from an EXISTING graph (no extraction / ingestion).

    Use this to start an interview against a session graph that's already in Neo4j
    (e.g. one built by a prior `build_session` call, or hand-curated).

    The returned state is fresh — no prior Q&A is rehydrated; topic counters
    (attempts, coverage, contradictions, qa) all start at zero. The graph's
    audit trail (`:ASKED` edges from any prior runs) is left intact in Neo4j.
    """
    settings = get_settings()
    if not session_id:
        raise ValueError("session_id is required")

    # Verify the session exists before doing anything else.
    found = await run(
        "MATCH (s:Session {id:$sid}) RETURN s.id AS id LIMIT 1",
        sid=session_id,
    )
    if not found:
        raise ValueError(f"Session '{session_id}' not found in the graph")

    # All three reads in parallel — same pattern as build_session.
    topic_rows, profile, jd_profile = await asyncio.gather(
        derive_topics(session_id, max_topics=settings.max_topics),
        get_candidate_profile(session_id),
        get_jd_profile(session_id),
    )

    if not topic_rows:
        raise ValueError(
            f"Session '{session_id}' has no derivable topics — "
            "does it have a Candidate, JobDescription, and skill links?"
        )

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

    state = InterviewState(
        session_id=session_id,
        topics=topics,
        current_idx=0 if topics else None,
        jd_title=(jd_profile or {}).get("title") or "Role",
        threshold=settings.coverage_threshold,
        max_followups=settings.max_followups,
        finished=not topics,
        candidate_profile=profile or {},
        jd_profile=jd_profile or {},
    )
    if topics:
        topics[0].status = "active"

    log.info("session=%s loaded from existing graph (%d topics)", session_id, len(topics))
    return state


async def teardown_session(session_id: str) -> None:
    """Wipe a session's subgraph from Neo4j. Safe to call even if id doesn't exist."""
    if not session_id:
        return
    try:
        await wipe_session(session_id)
        log.info("session=%s torn down", session_id)
    except Exception as e:
        log.warning("teardown failed for session=%s: %s", session_id, e)
