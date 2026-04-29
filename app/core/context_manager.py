"""Build LLM-facing context windows from interview state. Keeps tokens small."""
from __future__ import annotations

from app.core.state import InterviewState, TopicState
from app.graph.profile import format_profile
from app.llm.gemini_client import generate_text


def recent_turns(topic: TopicState, n: int = 2) -> str:
    if not topic.qa:
        return "(none yet)"
    tail = topic.qa[-n:]
    return "\n".join(f"Q: {q}\nA: {a}" for q, a, _ in tail)


_EMPTY_PROJECTS = (
    "(none — this topic does NOT appear in any project on the candidate's resume; "
    "do NOT pretend a project used it)"
)
_EMPTY_EXPERIENCES = (
    "(none — this topic does NOT appear in any experience/role on the candidate's resume; "
    "do NOT pretend a role used it)"
)


def _fmt_projects(topic: TopicState) -> str:
    if not topic.projects:
        return _EMPTY_PROJECTS
    lines = []
    for p in topic.projects[:5]:
        name = (p.get("name") or "").strip()
        summary = (p.get("summary") or "").strip()
        if not name:
            continue
        lines.append(f"- {name}: {summary[:200]}" if summary else f"- {name}")
    return "\n".join(lines) if lines else _EMPTY_PROJECTS


def _fmt_experiences(topic: TopicState) -> str:
    if not topic.experiences:
        return _EMPTY_EXPERIENCES
    lines = []
    for e in topic.experiences[:5]:
        role = (e.get("role") or "").strip()
        company = (e.get("company") or "").strip()
        summary = (e.get("summary") or "").strip()
        if not role:
            continue
        head = f"{role}" + (f" @ {company}" if company else "")
        lines.append(f"- {head}: {summary[:200]}" if summary else f"- {head}")
    return "\n".join(lines) if lines else _EMPTY_EXPERIENCES


def build_question_vars(state: InterviewState, topic: TopicState,
                        mode: str, followup_hint: str = "") -> dict:
    # CONCRETE evidence = projects/experiences specifically tied to this topic.
    # A bare :HAS_SKILL relationship doesn't count — the candidate listed the skill
    # but didn't tie it to a project or role, so we have nothing concrete to anchor on.
    has_concrete_evidence = bool(
        (topic.projects and any((p.get("name") or "").strip() for p in topic.projects))
        or (topic.experiences and any((e.get("role") or "").strip() for e in topic.experiences))
    )
    has_skill_claim = bool(
        topic.has_evidence
        or (topic.years and topic.years > 0)
        or topic.proficiency
        or topic.evidence_text
    )

    # Three states the prompt branches on:
    #   - has_concrete_evidence  → MUST anchor in a specific project/experience
    #   - has_skill_claim_only   → respect years/proficiency, but DO NOT invent projects
    #   - is_gap_topic           → foundational question or genuine bridge
    if has_concrete_evidence:
        broader_block = "(not needed — this topic has concrete project/experience evidence above; anchor the question there)"
    else:
        broader_block = format_profile(state.candidate_profile or {}, max_chars=900)

    return {
        "topic_name": topic.name,
        "jd_weight": int(topic.importance),
        "must_have": topic.must_have,
        "is_gap_topic": not (has_concrete_evidence or has_skill_claim),
        "has_concrete_evidence": has_concrete_evidence,
        "has_skill_claim_only": (has_skill_claim and not has_concrete_evidence),
        "years": f"{topic.years:.1f}" if topic.years else "(not stated)",
        "proficiency": topic.proficiency or "(not stated)",
        "evidence_text": (topic.evidence_text or "(none)")[:240],
        "projects_block": _fmt_projects(topic),
        "experiences_block": _fmt_experiences(topic),
        "candidate_profile_block": broader_block,
        "rolling_summary": state.rolling_summary or "(empty)",
        "recent_turns": recent_turns(topic),
        "mode": mode,
        "followup_hint": f"Follow-up hint: {followup_hint}" if followup_hint else "",
    }


async def compress_completed_topic(state: InterviewState, topic: TopicState) -> None:
    """Summarise a finished topic and append to rolling_summary; clear topic.qa context."""
    if not topic.qa:
        return
    qa_block = "\n".join(
        f"Q: {q}\nA: {a}\nScore: {s:.2f}" for q, a, s in topic.qa
    )
    prompt = (
        f"Compress the interview on topic '{topic.name}' "
        f"(importance {topic.importance:.1f}, attempts {topic.attempts}, "
        f"avg score {topic.avg_score():.2f}) into ONE 2-3 sentence paragraph "
        f"capturing what the candidate demonstrated and any gaps/contradictions. "
        f"No preamble, just the paragraph.\n\n{qa_block}"
    )
    summary = await generate_text(prompt, temperature=0.2)
    state.rolling_summary = (
        (state.rolling_summary + "\n\n" if state.rolling_summary else "")
        + f"[{topic.name}] {summary}"
    )
    # Cap rolling summary length
    if len(state.rolling_summary) > 4000:
        state.rolling_summary = state.rolling_summary[-4000:]
