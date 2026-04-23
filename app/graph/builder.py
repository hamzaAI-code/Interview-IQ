"""Ingest extracted Resume / JD graphs into Neo4j (batched MERGE)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.graph.neo4j_client import run_write
from app.llm.extractors import JDGraph, ResumeGraph
from app.utils.logger import get_logger

log = get_logger(__name__)


def _norm(name: str) -> str:
    return (name or "").strip().lower()


async def _create_session(session_id: str) -> None:
    await run_write(
        """
        MERGE (s:Session {id: $sid})
        ON CREATE SET s.started_at = $ts, s.session_id = $sid
        """,
        sid=session_id,
        ts=datetime.now(timezone.utc).isoformat(),
    )


async def ingest_resume(session_id: str, rg: ResumeGraph) -> None:
    """Build the candidate side of the graph in one batched call set."""
    skills = [
        {"name": _norm(s.name), "category": s.category, "years": s.years,
         "proficiency": s.proficiency, "evidence": s.evidence}
        for s in rg.skills if s.name
    ]
    techs = [{"name": _norm(t.name), "evidence": t.evidence}
             for t in rg.technologies if t.name]
    concepts = [{"name": _norm(c.name), "evidence": c.evidence}
                for c in rg.concepts if c.name]
    experiences = [
        {"role": e.role, "company": e.company, "years": e.years,
         "summary": e.summary, "used": [_norm(u) for u in e.used]}
        for e in rg.experiences
    ]
    projects = [
        {"name": p.name, "summary": p.summary,
         "technologies": [_norm(t) for t in p.technologies]}
        for p in rg.projects if p.name
    ]

    await run_write(
        """
        MERGE (s:Session {id: $sid})
        MERGE (c:Candidate {session_id: $sid})
          ON CREATE SET c.name = $name
        MERGE (s)-[:HAS_CANDIDATE]->(c)

        // skills
        WITH c
        UNWIND $skills AS sk
        MERGE (sn:Skill {name: sk.name})
          ON CREATE SET sn.category = sk.category
        MERGE (c)-[r:HAS_SKILL]->(sn)
          SET r.years = sk.years,
              r.proficiency = sk.proficiency,
              r.evidence = sk.evidence
        """,
        sid=session_id, name=rg.candidate_name or "Candidate",
        skills=skills,
    )

    if techs:
        await run_write(
            """
            MATCH (c:Candidate {session_id: $sid})
            UNWIND $techs AS t
            MERGE (tn:Technology {name: t.name})
            MERGE (c)-[r:USES_TECH]->(tn)
              SET r.evidence = t.evidence
            """,
            sid=session_id, techs=techs,
        )

    if concepts:
        await run_write(
            """
            MATCH (c:Candidate {session_id: $sid})
            UNWIND $concepts AS k
            MERGE (cn:Concept {name: k.name})
            MERGE (c)-[r:KNOWS]->(cn)
              SET r.evidence = k.evidence
            """,
            sid=session_id, concepts=concepts,
        )

    if experiences:
        await run_write(
            """
            MATCH (c:Candidate {session_id: $sid})
            UNWIND $exps AS e
            CREATE (x:Experience {
              session_id: $sid, role: e.role, company: e.company,
              years: e.years, summary: e.summary
            })
            MERGE (c)-[:WORKED_AS]->(x)
            WITH x, e
            UNWIND e.used AS used_name
            MATCH (n) WHERE (n:Skill OR n:Technology) AND n.name = used_name
            MERGE (x)-[:USED]->(n)
            """,
            sid=session_id, exps=experiences,
        )

    if projects:
        await run_write(
            """
            MATCH (c:Candidate {session_id: $sid})
            UNWIND $projects AS p
            CREATE (pr:Project {
              session_id: $sid, name: p.name, summary: p.summary
            })
            MERGE (c)-[:BUILT]->(pr)
            WITH pr, p
            UNWIND p.technologies AS tname
            MATCH (n) WHERE (n:Technology OR n:Skill) AND n.name = tname
            MERGE (pr)-[:USES]->(n)
            """,
            sid=session_id, projects=projects,
        )

    log.info("resume ingested for session=%s", session_id)


async def ingest_jd(session_id: str, jd: JDGraph) -> None:
    reqs = [
        {"name": _norm(r.name), "kind": r.kind, "priority": int(r.priority),
         "must_have": bool(r.must_have)}
        for r in jd.requirements if r.name
    ]
    lines = [{"text": ln} for ln in jd.requirement_lines if ln]

    await run_write(
        """
        MERGE (s:Session {id: $sid})
        MERGE (j:JobDescription {session_id: $sid})
          ON CREATE SET j.title = $title
        MERGE (s)-[:HAS_JD]->(j)

        WITH j
        UNWIND $reqs AS rq
        CALL {
          WITH rq
          FOREACH (_ IN CASE WHEN rq.kind = 'skill' THEN [1] ELSE [] END |
            MERGE (n:Skill {name: rq.name})
          )
          FOREACH (_ IN CASE WHEN rq.kind = 'technology' THEN [1] ELSE [] END |
            MERGE (n:Technology {name: rq.name})
          )
          FOREACH (_ IN CASE WHEN rq.kind = 'concept' THEN [1] ELSE [] END |
            MERGE (n:Concept {name: rq.name})
          )
        }
        WITH j, rq
        MATCH (n) WHERE n.name = rq.name AND
          ((rq.kind = 'skill' AND n:Skill) OR
           (rq.kind = 'technology' AND n:Technology) OR
           (rq.kind = 'concept' AND n:Concept))
        MERGE (j)-[r:REQUIRES]->(n)
          SET r.priority = rq.priority, r.must_have = rq.must_have
        """,
        sid=session_id, title=jd.role_title or "Role", reqs=reqs,
    )

    if lines:
        await run_write(
            """
            MATCH (j:JobDescription {session_id: $sid})
            UNWIND $lines AS l
            CREATE (rq:Requirement {session_id: $sid, text: l.text})
            MERGE (j)-[:CONTAINS]->(rq)
            """,
            sid=session_id, lines=lines,
        )

    log.info("JD ingested for session=%s", session_id)


async def ingest_both(session_id: str, rg: ResumeGraph, jd: JDGraph) -> None:
    """Ingest in parallel after creating the session root."""
    await _create_session(session_id)
    await asyncio.gather(
        ingest_resume(session_id, rg),
        ingest_jd(session_id, jd),
    )
