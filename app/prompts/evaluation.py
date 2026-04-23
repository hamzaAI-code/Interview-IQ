"""Prompts for the final evaluation report."""

EVALUATION_SYSTEM = """You are a senior technical hiring manager. Produce a structured technical evaluation of a candidate based on the interview transcript.

Be specific, evidence-based, and concise. Cite topics by name. No hedging boilerplate.

Return STRICT JSON matching the schema."""


EVALUATION_USER_TEMPLATE = """JD title: {jd_title}
Coverage threshold used: {threshold}
Per-topic results (topic, importance, coverage, depth, attempts, contradictions, qa_count, score):
{topic_table}

Full Q&A transcript (compressed):
{transcript}

Now produce the evaluation."""
