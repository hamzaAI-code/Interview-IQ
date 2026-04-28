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


def _fmt_projects(topic: TopicState) -> str:
    if not topic.projects:
        return "(none listed on resume for this topic)"
    lines = []
    for p in topic.projects[:5]:
        name = (p.get("name") or "").strip()
        summary = (p.get("summary") or "").strip()
        if not name:
            continue
        lines.append(f"- {name}: {summary[:200]}" if summary else f"- {name}")
    return "\n".join(lines) if lines else "(none listed on resume for this topic)"


def _fmt_experiences(topic: TopicState) -> str:
    if not topic.experiences:
        return "(none listed on resume for this topic)"
    lines = []
    for e in topic.experiences[:5]:
        role = (e.get("role") or "").strip()
        company = (e.get("company") or "").strip()
        summary = (e.get("summary") or "").strip()
        if not role:
            continue
        head = f"{role}" + (f" @ {company}" if company else "")
        lines.append(f"- {head}: {summary[:200]}" if summary else f"- {head}")
    return "\n".join(lines) if lines else "(none listed on resume for this topic)"


def build_question_vars(state: InterviewState, topic: TopicState,
                        mode: str, followup_hint: str = "") -> dict:
    has_topic_evidence = bool(
        topic.has_evidence
        or (topic.projects and any((p.get("name") or "").strip() for p in topic.projects))
        or (topic.experiences and any((e.get("role") or "").strip() for e in topic.experiences))
    )
    # Only show the broader resume profile when the current topic is a gap.
    # Saves ~250 tokens on every non-gap question (the common case) while still
    # giving the model material to bridge from for gap topics.
    if has_topic_evidence:
        broader_block = "(not needed — this topic has direct evidence above; anchor the question there)"
    else:
        broader_block = format_profile(state.candidate_profile or {}, max_chars=900)

    return {
        "topic_name": topic.name,
        "jd_weight": int(topic.importance),
        "must_have": topic.must_have,
        "is_gap_topic": not has_topic_evidence,
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
