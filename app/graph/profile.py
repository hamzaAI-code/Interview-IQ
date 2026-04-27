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
