"""Generate natural responses for meta-question turns.

Pulls in:
  - candidate profile (full resume snapshot, fetched once from the graph at session start)
  - JD profile (role title + ranked requirements, also fetched at session start)
  - current topic context (which skill we're on + the specific projects/experiences
    on the resume that touch it — so "which project?" can be answered specifically)
  - rolling summary of completed topics
  - the original question being asked + what the candidate just said

Produces a 2-3 sentence interviewer reply that addresses their meta question
with concrete facts and redirects back to the original question.
"""
from __future__ import annotations

from app.config import get_settings
from app.graph.profile import format_current_topic, format_jd_profile, format_profile
from app.llm.gemini_client import generate_text


META_SYSTEM = """You are a senior technical interviewer. The candidate just asked YOU a meta question — about themselves, the role, the process, how they're doing, or which project/experience your last question was anchored to.

Respond naturally, grounded in concrete facts:

GROUNDING RULES — always cite SPECIFICS, not generalities:
- "What do you know about me?" / "Have you read my resume?" → name 2-3 things from the CANDIDATE PROFILE (skills with years, named projects, named roles).
- "What's this role for?" / "What position is this?" → use the ROLE/JD section: title + 1-2 must-have requirements.
- "Which project are you talking about?" / "Which experience?" / "Where did I use this skill?" → look ONLY at the CURRENT TOPIC section's "Projects on resume that used this topic" and "Experiences on resume that used this topic" sub-lists.
- "How am I doing?" → reference the ROLLING SUMMARY (compressed history of prior topics) plus what they've done on the current topic.
- "What's next?" → reference upcoming work generically ("we'll touch a couple more topics from the JD") but don't promise specifics.

ABSOLUTE RULE — DO NOT HALLUCINATE TIES:
If the user asks "where did I use X" / "which project used X" / "which experience used X" and the CURRENT TOPIC section's projects sub-list or experiences sub-list shows the explicit "(none — ...)" marker:
  → Tell them the truth: "X is in your skills list but isn't tied to a specific project or role on your resume."
  → DO NOT pick a project from the broader CANDIDATE PROFILE — those projects used different tech.
  → DO NOT invent or imply a connection.
After that honest answer, redirect back to the original question.

OUTPUT SHAPE:
- 2-3 sentences total. Warm, direct, conversational.
- Sentence 1: answer their meta question with specifics.
- Sentence 2 (optional): a short bridge.
- Sentence 3: re-ask the original technical question, verbatim or lightly simplified.

NEVER use these phrases:
- "I'm evaluating your technical skills"
- "We are discussing"
- "I'm here to assess"
- "Let me restate"
- "As an AI"
- Any meta-description of what you are or what you're doing — just be the interviewer.

Use contractions. No "Sure!" / "Great question!" preambles. No greetings. No lists."""


META_USER_TEMPLATE = """ROLE WE'RE INTERVIEWING FOR (from the JD):
{jd_block}

CANDIDATE PROFILE (from their resume):
{profile_block}

CURRENT TOPIC (what the last question was anchored to):
{topic_block}

ROLLING SUMMARY (compressed prior topics — may be empty early on):
{rolling_summary}

ORIGINAL QUESTION I JUST ASKED: {question}
WHAT THE CANDIDATE SAID INSTEAD: {user_msg}

Now produce the natural reply (2-3 sentences):"""


async def generate_meta_response(
    profile: dict,
    jd_profile: dict,
    current_topic: dict,
    original_question: str,
    user_msg: str,
    rolling_summary: str = "",
) -> str:
    """Build a meta-question reply grounded in resume + JD + current topic anchor."""
    settings = get_settings()
    prompt = META_USER_TEMPLATE.format(
        jd_block=format_jd_profile(jd_profile or {}),
        profile_block=format_profile(profile or {}),
        topic_block=format_current_topic(current_topic or {}),
        rolling_summary=(rolling_summary or "(this is early in the interview)")[:600],
        question=original_question,
        user_msg=user_msg,
    )
    text = await generate_text(
        prompt=prompt,
        system=META_SYSTEM,
        model=settings.gemini_model_fast,
        temperature=0.0,
        thinking_budget=0,
    )
    return (text or "").strip()
