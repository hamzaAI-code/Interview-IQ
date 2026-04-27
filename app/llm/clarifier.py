"""Generate natural responses for non-answer turns where the candidate
asked us a meta-question about themselves. Pulls candidate facts from the
graph (passed in as a profile dict) and produces a 2-3 sentence reply that
answers their question with specifics, then redirects to the original question.
"""
from __future__ import annotations

from app.config import get_settings
from app.graph.profile import format_profile
from app.llm.gemini_client import generate_text


META_SYSTEM = """You are a senior technical interviewer. The candidate just asked YOU a meta question about themselves, the role, the process, or how they're doing. Respond naturally:

- Briefly answer their question with SPECIFIC facts from the candidate profile below (1 sentence, name 2-3 concrete things — years, skills, projects, roles).
- Then smoothly redirect back to the original technical question. Re-ask it (verbatim or lightly simplified).
- 2-3 sentences total. Warm, direct, conversational. Like a real interviewer who's also a person.

Hard bans (never use these phrases):
- "I'm evaluating your technical skills"
- "We are discussing"
- "I'm here to assess"
- "Let me restate"
- "As an AI"
- Any explanation of what you are or what you're doing — just be the interviewer.

Use contractions. Don't be preachy. No greetings, no "Sure!" or "Great question!" preambles."""


META_USER_TEMPLATE = """CANDIDATE PROFILE (from their resume — what we actually know):
{profile}

Where we are in the interview:
{rolling_summary}

Original question I was asking: {question}
What the candidate just said: {user_msg}

Now produce the natural reply (2-3 sentences):"""


async def generate_meta_response(
    profile: dict,
    original_question: str,
    user_msg: str,
    rolling_summary: str = "",
) -> str:
    settings = get_settings()
    prompt = META_USER_TEMPLATE.format(
        profile=format_profile(profile or {}),
        rolling_summary=(rolling_summary or "(this is early in the interview)")[:600],
        question=original_question,
        user_msg=user_msg,
    )
    text = await generate_text(
        prompt=prompt,
        system=META_SYSTEM,
        model=settings.gemini_model_fast,
        temperature=0.5,
        thinking_budget=0,
    )
    return (text or "").strip()
