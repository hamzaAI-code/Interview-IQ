"""Ingest extracted Resume / JD graphs into Neo4j.

Each side is a single batched Cypher statement (one round-trip) that uses CALL
subqueries so independent UNWINDs don't cross-product each other. Depth ingestion
pre-merges any names referenced by experiences/projects that weren't in core,
so no information is lost.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.graph.neo4j_client import run_write
from app.llm.extractors import JDGraph, ResumeCoreGraph, ResumeDepthGraph
from app.utils.logger import get_logger

log = get_logger(__name__)


def _norm(name: str) -> str:
    return (name or "").strip().lower()


async def create_session(session_id: str) -> None:
    await run_write(
        """
        MERGE (s:Session {id: $sid})
        ON CREATE SET s.started_at = $ts, s.session_id = $sid
        """,
        sid=session_id,
        ts=datetime.now(timezone.utc).isoformat(),
    )


async def ingest_resume_core(session_id: str, core: ResumeCoreGraph) -> None:
    """Candidate + skills + technologies + concepts in ONE transaction."""
    skills = [
        {"name": _norm(s.name), "category": s.category, "years": s.years,
         "proficiency": s.proficiency, "evidence": s.evidence}
        for s in core.skills if s.name
    ]
    techs = [{"name": _norm(t.name), "evidence": t.evidence}
             for t in core.technologies if t.name]
    concepts = [{"name": _norm(c.name), "evidence": c.evidence}
                for c in core.concepts if c.name]

    await run_write(
        """
        MERGE (s:Session {id: $sid})
        MERGE (c:Candidate {session_id: $sid})
          ON CREATE SET c.name = $name
          ON MATCH  SET c.name = CASE WHEN c.name IS NULL OR c.name = 'Candidate' OR c.name = ''
                                      THEN $name ELSE c.name END
        MERGE (s)-[:HAS_CANDIDATE]->(c)

        WITH c

        CALL {
          WITH c
          UNWIND $skills AS sk
          MERGE (sn:Skill {name: sk.name})
            ON CREATE SET sn.category = sk.category
          MERGE (c)-[r:HAS_SKILL]->(sn)
            SET r.years = sk.years,
                r.proficiency = sk.proficiency,
                r.evidence = sk.evidence
        }

        CALL {
          WITH c
          UNWIND $techs AS t
          MERGE (tn:Technology {name: t.name})
          MERGE (c)-[r:USES_TECH]->(tn)
            SET r.evidence = t.evidence
        }

        CALL {
          WITH c
          UNWIND $concepts AS k
          MERGE (cn:Concept {name: k.name})
          MERGE (c)-[r:KNOWS]->(cn)
            SET r.evidence = k.evidence
        }
        """,
        sid=session_id,
        name=core.candidate_name or "Candidate",
        skills=skills,
        techs=techs,
        concepts=concepts,
    )
    log.info(
        "core ingested session=%s skills=%d techs=%d concepts=%d",
        session_id, len(skills), len(techs), len(concepts),
    )


async def ingest_resume_depth(session_id: str, depth: ResumeDepthGraph) -> None:
    """Experiences + projects in ONE transaction. Pre-merges any names they
    reference so no USED/USES edges are silently dropped. Also writes
    Capability nodes + DEMONSTRATES edges so the matcher can find projects
    that did the *kind of work* a JD topic asks about, even when literal
    tech names don't overlap.
    """
    exps = [
        {
            "role": e.role,
            "company": e.company,
            "years": e.years,
            "summary": e.summary,
            "used": [_norm(u) for u in e.used if u],
            "capabilities": [_norm(c) for c in e.capabilities if c],
        }
        for e in depth.experiences if e.role
    ]
    projs = [
        {
            "name": p.name,
            "summary": p.summary,
            "technologies": [_norm(t) for t in p.technologies if t],
            "capabilities": [_norm(c) for c in p.capabilities if c],
        }
        for p in depth.projects if p.name
    ]

    # Every literal-tech name referenced across experiences + projects.
    # Pre-merged as :Skill if no node with that name exists.
    ref_names = sorted({
        n for e in exps for n in e["used"] if n
    } | {
        n for p in projs for n in p["technologies"] if n
    })

    await run_write(
        """
        MATCH (c:Candidate {session_id: $sid})

        // Pre-merge any referenced literal name that doesn't already exist.
        CALL {
          UNWIND $ref_names AS refname
          OPTIONAL MATCH (existing)
            WHERE existing.name = refname
              AND (existing:Skill OR existing:Technology OR existing:Concept)
          FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
            MERGE (:Skill {name: refname})
          )
        }

        // Experiences + USED + DEMONSTRATES edges.
        CALL {
          WITH c
          UNWIND $exps AS e
          CREATE (x:Experience {
            session_id: $sid, role: e.role, company: e.company,
            years: e.years, summary: e.summary
          })
          MERGE (c)-[:WORKED_AS]->(x)

          // USED — literal tech link
          WITH x, e
          CALL {
            WITH x, e
            UNWIND e.used AS used_name
            OPTIONAL MATCH (n)
              WHERE n.name = used_name
                AND (n:Skill OR n:Technology OR n:Concept)
            WITH x, collect(n)[0] AS target
            WHERE target IS NOT NULL
            MERGE (x)-[:USED]->(target)
          }

          // DEMONSTRATES — broader capability tags
          WITH x, e
          CALL {
            WITH x, e
            UNWIND coalesce(e.capabilities, []) AS cap_name
            MERGE (cap:Capability {name: cap_name})
            MERGE (x)-[:DEMONSTRATES]->(cap)
          }
        }

        // Projects + USES + DEMONSTRATES edges.
        CALL {
          WITH c
          UNWIND $projs AS p
          CREATE (pr:Project {
            session_id: $sid, name: p.name, summary: p.summary
          })
          MERGE (c)-[:BUILT]->(pr)

          WITH pr, p
          CALL {
            WITH pr, p
            UNWIND p.technologies AS tname
            OPTIONAL MATCH (n)
              WHERE n.name = tname
                AND (n:Skill OR n:Technology OR n:Concept)
            WITH pr, collect(n)[0] AS target
            WHERE target IS NOT NULL
            MERGE (pr)-[:USES]->(target)
          }

          WITH pr, p
          CALL {
            WITH pr, p
            UNWIND coalesce(p.capabilities, []) AS cap_name
            MERGE (cap:Capability {name: cap_name})
            MERGE (pr)-[:DEMONSTRATES]->(cap)
          }
        }
        """,
        sid=session_id,
        exps=exps,
        projs=projs,
        ref_names=ref_names,
    )

    cap_count = sum(len(e["capabilities"]) for e in exps) + sum(len(p["capabilities"]) for p in projs)
    log.info(
        "depth ingested session=%s experiences=%d projects=%d refs=%d capabilities=%d",
        session_id, len(exps), len(projs), len(ref_names), cap_count,
    )


async def ingest_jd(session_id: str, jd: JDGraph) -> None:
    """JobDescription + REQUIRES edges + raw Requirement lines in ONE transaction."""
    reqs = [
        {"name": _norm(r.name), "kind": r.kind, "priority": int(r.priority),
         "must_have": bool(r.must_have)}
        for r in jd.requirements if r.name
    ]
    lines = [ln for ln in jd.requirement_lines if ln]

    await run_write(
        """
        MERGE (s:Session {id: $sid})
        MERGE (j:JobDescription {session_id: $sid})
          ON CREATE SET j.title = $title
          ON MATCH  SET j.title = coalesce(j.title, $title)
        MERGE (s)-[:HAS_JD]->(j)

        WITH j

        CALL {
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
          MATCH (n)
            WHERE n.name = rq.name
              AND ((rq.kind = 'skill'      AND n:Skill)
                OR (rq.kind = 'technology' AND n:Technology)
                OR (rq.kind = 'concept'    AND n:Concept))
          MERGE (j)-[r:REQUIRES]->(n)
            SET r.priority = rq.priority, r.must_have = rq.must_have
        }

        CALL {
          WITH j
          UNWIND $lines AS ln
          CREATE (rq:Requirement {session_id: $sid, text: ln})
          MERGE (j)-[:CONTAINS]->(rq)
        }
        """,
        sid=session_id, title=jd.role_title or "Role", reqs=reqs, lines=lines,
    )
    log.info(
        "jd ingested session=%s requirements=%d lines=%d",
        session_id, len(reqs), len(lines),
    )
