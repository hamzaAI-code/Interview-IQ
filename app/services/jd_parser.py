"""Light cleanup of a pasted job description."""
from __future__ import annotations


def parse_jd(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
