"""Match resume vs JD; derive ranked :Topic nodes for the interview."""
from __future__ import annotations

from app.config import get_settings
from app.graph.neo4j_client import run, run_write
from app.utils.logger import get_logger

log = get_logger(__name__)


async def derive_topics(session_id: str, max_topics: int | None = None) -> list[dict]:
    """For every JD requirement, create a :Topic node with computed importance.

    Returns the ranked topics as plain dicts for the orchestrator.
    """
    settings = get_settings()
    cap = max_topics or settings.max_topics

    # Score formula:
    #   importance = priority*2 + (must_have?3:0) + evidence_strength - gap_penalty
    # evidence_strength: 0..3 from years + project count
    # gap_penalty: 2 if no HAS_SKILL/USES_TECH/KNOWS edge for that name
    rows = await run(
        """
        MATCH (j:JobDescription {session_id: $sid})-[req:REQUIRES]->(target)
        OPTIONAL MATCH (c:Candidate {session_id: $sid})
        OPTIONAL MATCH (c)-[hs:HAS_SKILL]->(target)
        OPTIONAL MATCH (c)-[ut:USES_TECH]->(target)
        OPTIONAL MATCH (c)-[kn:KNOWS]->(target)

        // aggregate projects that use this target
        OPTIONAL MATCH (c)-[:BUILT]->(p:Project)-[:USES]->(target)
        WITH j, req, target, c, hs, ut, kn,
             collect(DISTINCT CASE WHEN p IS NULL THEN null
                ELSE {name: p.name, summary: p.summary} END) AS projects_raw

        // aggregate experiences that used this target
        OPTIONAL MATCH (c)-[:WORKED_AS]->(x:Experience)-[:USED]->(target)
        WITH j, req, target, hs, ut, kn, projects_raw,
             collect(DISTINCT CASE WHEN x IS NULL THEN null
                ELSE {role: x.role, company: x.company, summary: x.summary} END) AS exps_raw

        WITH target, req, hs, ut, kn,
             [p IN projects_raw WHERE p IS NOT NULL] AS projects_list,
             [e IN exps_raw WHERE e IS NOT NULL] AS experiences_list,
             coalesce(hs.years, 0.0) AS years,
             coalesce(hs.proficiency, '') AS proficiency,
             coalesce(hs.evidence, '') AS evidence_text
        WITH target, req, years, proficiency, evidence_text, projects_list, experiences_list,
             (hs IS NOT NULL OR ut IS NOT NULL OR kn IS NOT NULL
              OR size(projects_list) > 0 OR size(experiences_list) > 0) AS has_evidence,
             size(projects_list) AS proj_count
        WITH target, req, years, proficiency, evidence_text,
             projects_list, experiences_list, has_evidence, proj_count,
             (CASE WHEN years > 4 THEN 2.0
                   WHEN years > 1 THEN 1.0
                   ELSE 0.0 END
              + CASE WHEN proj_count > 0 THEN 1.0 ELSE 0.0 END) AS evidence_strength
        WITH target, req, years, proficiency, evidence_text,
             projects_list, experiences_list, has_evidence, proj_count, evidence_strength,
             (req.priority * 2.0)
             + (CASE WHEN req.must_have THEN 3.0 ELSE 0.0 END)
             + evidence_strength
             - (CASE WHEN has_evidence THEN 0.0 ELSE 2.0 END) AS importance
        RETURN target.name AS name,
               labels(target)[0] AS kind,
               req.priority AS priority,
               req.must_have AS must_have,
               importance,
               has_evidence,
               years,
               proficiency,
               evidence_text,
               proj_count,
               projects_list,
               experiences_list
        ORDER BY importance DESC
        LIMIT $cap
        """,
        sid=session_id, cap=cap,
    )

    if not rows:
        log.warning("no topics derived for session=%s", session_id)
        return []

    # Persist as :Topic nodes for audit + later [:ASKED] writes
    topics_payload = [
        {
            "name": r["name"],
            "kind": r["kind"],
            "importance": float(r["importance"]),
            "must_have": bool(r["must_have"]),
            "priority": int(r["priority"]),
        }
        for r in rows
    ]
    await run_write(
        """
        MATCH (s:Session {id: $sid})
        UNWIND $topics AS t
        MERGE (tp:Topic {session_id: $sid, name: t.name})
          SET tp.importance = t.importance,
              tp.must_have = t.must_have,
              tp.priority = t.priority,
              tp.kind = t.kind
        MERGE (s)-[:HAS_TOPIC]->(tp)
        WITH tp, t
        MATCH (n) WHERE n.name = t.name AND
          ((t.kind = 'Skill' AND n:Skill) OR
           (t.kind = 'Technology' AND n:Technology) OR
           (t.kind = 'Concept' AND n:Concept))
        MERGE (tp)-[:COVERS]->(n)
        """,
        sid=session_id, topics=topics_payload,
    )

    # Build a candidate_claims string per topic from row data
    enriched = []
    for r in rows:
        if r["has_evidence"]:
            proj_names = [p.get("name") for p in (r.get("projects_list") or []) if p.get("name")]
            proj_str = f"; projects: {', '.join(proj_names[:3])}" if proj_names else ""
            claim = (
                f"~{r['years']:.0f}y, {r['proficiency']}, "
                f"{r['proj_count']} project(s); "
                f"{(r['evidence_text'] or '').strip()[:140]}"
                f"{proj_str}"
            ).strip(" ;,")
        else:
            claim = "no direct evidence in resume (gap area)"
        enriched.append({**r, "candidate_claims": claim})

    log.info("derived %d topics for session=%s", len(enriched), session_id)
    return enriched


async def record_qa(
    session_id: str,
    topic_name: str,
    question: str,
    answer: str,
    score: float,
    contradiction: bool,
    ts: str,
) -> None:
    await run_write(
        """
        MATCH (t:Topic {session_id: $sid, name: $tname})
        MATCH (s:Session {id: $sid})
        CREATE (t)-[:ASKED {
          question: $q, answer: $a, score: $score,
          contradiction: $contra, ts: $ts
        }]->(s)
        """,
        sid=session_id, tname=topic_name, q=question, a=answer,
        score=score, contra=contradiction, ts=ts,
    )
