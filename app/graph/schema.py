"""Idempotent Neo4j schema setup (constraints + indexes)."""
from __future__ import annotations

from app.graph.neo4j_client import run_write
from app.utils.logger import get_logger

log = get_logger(__name__)

DDL = [
    "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT tech_name IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
    "CREATE INDEX topic_session IF NOT EXISTS FOR (t:Topic) ON (t.session_id, t.name)",
    "CREATE INDEX candidate_session IF NOT EXISTS FOR (c:Candidate) ON (c.session_id)",
    "CREATE INDEX jd_session IF NOT EXISTS FOR (j:JobDescription) ON (j.session_id)",
]


async def ensure_schema() -> None:
    log.info("ensuring Neo4j schema (%d statements)", len(DDL))
    for stmt in DDL:
        await run_write(stmt)


async def wipe_session(session_id: str) -> None:
    """Delete all session-scoped nodes for a clean re-run."""
    await run_write(
        """
        MATCH (n)
        WHERE n.session_id = $sid
        DETACH DELETE n
        """,
        sid=session_id,
    )
