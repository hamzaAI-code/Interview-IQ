"""On-demand candidate profile fetch from the session's subgraph.

Used to power meta-question handling ("what do you know about me?") without
keeping a snapshot inflated in every analyzer prompt. The graph is the canonical
source of candidate facts; this is the read accessor.
"""
from __future__ import annotations

from app.graph.neo4j_client import run
from app.utils.logger import get_logger

log = get_logger(__name__)


_PROFILE_CYPHER = """
MATCH (c:Candidate {session_id: $sid})
OPTIONAL MATCH (c)-[h:HAS_SKILL]->(s:Skill)
WITH c, collect(DISTINCT CASE WHEN s IS NULL THEN null ELSE {
        name: s.name,
        years: coalesce(h.years, 0.0),
        proficiency: coalesce(h.proficiency, ''),
        evidence: coalesce(h.evidence, '')
     } END) AS skills_raw
OPTIONAL MATCH (c)-[:BUILT]->(p:Project)
WITH c, skills_raw, collect(DISTINCT CASE WHEN p IS NULL THEN null ELSE {
        name: p.name,
        summary: coalesce(p.summary, '')
     } END) AS projects_raw
OPTIONAL MATCH (c)-[:WORKED_AS]->(x:Experience)
WITH c, skills_raw, projects_raw, collect(DISTINCT CASE WHEN x IS NULL THEN null ELSE {
        role: x.role,
        company: coalesce(x.company, ''),
        summary: coalesce(x.summary, ''),
        years: coalesce(x.years, 0.0)
     } END) AS experiences_raw
RETURN coalesce(c.name, 'Candidate') AS name,
       [sk IN skills_raw WHERE sk IS NOT NULL] AS skills,
       [pr IN projects_raw WHERE pr IS NOT NULL][..5] AS projects,
       [ex IN experiences_raw WHERE ex IS NOT NULL][..5] AS experiences
"""


async def get_candidate_profile(session_id: str) -> dict:
    """Fetch a compact profile for use in meta-question responses.

    Returns:
        dict with keys: name, skills (list of {name, years, proficiency, evidence}),
        projects (list of {name, summary}), experiences (list of {role, company, summary, years}).
        Empty dict if nothing found.
    """
    rows = await run(_PROFILE_CYPHER, sid=session_id)
    if not rows:
        log.warning("no candidate profile found for session=%s", session_id)
        return {}

    row = rows[0]
    skills = sorted(
        row.get("skills") or [],
        key=lambda s: float(s.get("years") or 0.0),
        reverse=True,
    )[:8]
    return {
        "name": row.get("name") or "Candidate",
        "skills": skills,
        "projects": row.get("projects") or [],
        "experiences": row.get("experiences") or [],
    }


_JD_PROFILE_CYPHER = """
MATCH (j:JobDescription {session_id: $sid})
OPTIONAL MATCH (j)-[r:REQUIRES]->(req)
WITH j, collect(DISTINCT CASE WHEN req IS NULL THEN null ELSE {
        name: req.name,
        kind: labels(req)[0],
        priority: coalesce(r.priority, 3),
        must_have: coalesce(r.must_have, false)
     } END) AS reqs_raw
OPTIONAL MATCH (j)-[:CONTAINS]->(rl:Requirement)
WITH j, reqs_raw, collect(DISTINCT rl.text) AS lines_raw
RETURN coalesce(j.title, 'Role') AS title,
       [r IN reqs_raw WHERE r IS NOT NULL] AS requirements,
       [l IN lines_raw WHERE l IS NOT NULL][..10] AS requirement_lines
"""


async def get_jd_profile(session_id: str) -> dict:
    """Fetch a compact JD snapshot for use in meta-question responses."""
    rows = await run(_JD_PROFILE_CYPHER, sid=session_id)
    if not rows:
        log.warning("no JD profile found for session=%s", session_id)
        return {}
    row = rows[0]
    # Sort requirements: must_have first, then by priority desc, cap at 15.
    reqs = sorted(
        row.get("requirements") or [],
        key=lambda r: (
            0 if r.get("must_have") else 1,
            -int(r.get("priority") or 3),
        ),
    )[:15]
    return {
        "title": row.get("title") or "Role",
        "requirements": reqs,
        "requirement_lines": row.get("requirement_lines") or [],
    }


def format_jd_profile(jd: dict, max_chars: int = 800) -> str:
    """Render the JD profile as compact prose for an LLM prompt."""
    if not jd:
        return "(no JD profile available)"
    lines: list[str] = [f"Title: {jd.get('title') or 'Role'}"]

    reqs = jd.get("requirements") or []
    must = [r for r in reqs if r.get("must_have")]
    nice = [r for r in reqs if not r.get("must_have")]
    if must:
        lines.append("Must-have requirements:")
        for r in must[:8]:
            lines.append(f"  - {r.get('name')} ({r.get('kind','Skill').lower()}, p{r.get('priority')})")
    if nice:
        lines.append("Nice-to-have / other:")
        for r in nice[:8]:
            lines.append(f"  - {r.get('name')} ({r.get('kind','Skill').lower()}, p{r.get('priority')})")

    rl = jd.get("requirement_lines") or []
    if rl:
        lines.append("Raw requirement lines:")
        for ln in rl[:5]:
            lines.append(f"  - {ln[:140]}")

    text = "\n".join(lines)
    return text[:max_chars]


def format_current_topic(topic: dict, max_chars: int = 600) -> str:
    """Compact rendering of the topic the question was anchored to.

    Always renders the projects + experiences section labels, even when empty,
    with an explicit "none — skill listed only" marker. The LLM needs to see
    the absence as a fact (not as missing data) to avoid hallucinating ties
    to projects in the broader profile that didn't actually use this topic.
    """
    if not topic:
        return "(no current topic)"
    lines = [f"Topic: {topic.get('name')}"]
    claims = (topic.get("candidate_claims") or "").strip()
    if claims:
        lines.append(f"Candidate claims on this topic: {claims[:240]}")

    projects = [p for p in (topic.get("projects") or []) if (p.get("name") or "").strip()]
    lines.append("Projects on resume that used this topic:")
    if projects:
        for p in projects[:4]:
            name = (p.get("name") or "").strip()
            summary = (p.get("summary") or "").strip()
            lines.append(f"  - {name}: {summary[:140]}" if summary else f"  - {name}")
    else:
        lines.append("  (none — this topic appears in the candidate's skills list only, NOT tied to any specific project on their resume)")

    experiences = [e for e in (topic.get("experiences") or []) if (e.get("role") or "").strip()]
    lines.append("Experiences on resume that used this topic:")
    if experiences:
        for e in experiences[:4]:
            role = (e.get("role") or "").strip()
            company = (e.get("company") or "").strip()
            summary = (e.get("summary") or "").strip()
            head = f"{role}" + (f" @ {company}" if company else "")
            lines.append(f"  - {head}: {summary[:140]}" if summary else f"  - {head}")
    else:
        lines.append("  (none — this topic is NOT tied to any specific role/experience on their resume)")

    text = "\n".join(lines)
    return text[:max_chars]


def format_profile(profile: dict, max_chars: int = 1200) -> str:
    """Render the profile as compact prose for an LLM prompt."""
    if not profile:
        return "(no resume profile available)"

    lines: list[str] = []
    name = profile.get("name") or "Candidate"
    lines.append(f"Name: {name}")

    skills = profile.get("skills") or []
    if skills:
        skill_lines = []
        for s in skills:
            yrs = float(s.get("years") or 0.0)
            prof = s.get("proficiency") or ""
            ev = (s.get("evidence") or "").strip()
            head = f"{s.get('name')}"
            tail = []
            if yrs:
                tail.append(f"{yrs:.1f}y")
            if prof:
                tail.append(prof)
            if ev:
                tail.append(ev[:80])
            skill_lines.append(f"  - {head} ({', '.join(tail)})" if tail else f"  - {head}")
        lines.append("Skills:")
        lines.extend(skill_lines)

    projects = profile.get("projects") or []
    if projects:
        lines.append("Projects:")
        for p in projects[:4]:
            summary = (p.get("summary") or "").strip()
            lines.append(f"  - {p.get('name')}: {summary[:140]}" if summary else f"  - {p.get('name')}")

    experiences = profile.get("experiences") or []
    if experiences:
        lines.append("Experience:")
        for e in experiences[:4]:
            company = e.get("company") or ""
            head = f"{e.get('role')}" + (f" @ {company}" if company else "")
            summary = (e.get("summary") or "").strip()
            lines.append(f"  - {head}: {summary[:140]}" if summary else f"  - {head}")

    text = "\n".join(lines)
    return text[:max_chars]
