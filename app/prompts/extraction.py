"""Prompts for extracting structured graphs from raw resume / JD text."""

RESUME_EXTRACTION_PROMPT = """You are a precise information extractor. From the candidate's RESUME below, extract a graph of their technical profile.

Return STRICT JSON matching the provided schema. No prose, no markdown.

Rules:
- Skill names lowercased, canonical (e.g. "Python 3" -> "python", "K8s" -> "kubernetes").
- A "skill" is a discrete capability (language, framework, tool, methodology).
- A "technology" is a specific product (e.g. "tensorflow", "fastapi", "postgres").
- A "concept" is a topic of knowledge (e.g. "transformer attention", "event sourcing").
- For each skill: estimate years from context, mark proficiency in {{beginner, intermediate, advanced, expert}}, capture one short evidence string from the resume.
- Capture distinct experiences (role + company + years + 1-line summary), and projects (name + summary + tech used).
- Be exhaustive on skills/technologies/concepts but DO NOT invent items not supported by the text.
- If the candidate's name is in the resume, include it. Otherwise leave name empty.

RESUME:
---
{resume_text}
---
"""


JD_EXTRACTION_PROMPT = """You are a precise information extractor. From the JOB DESCRIPTION below, extract its structured requirements.

Return STRICT JSON matching the provided schema. No prose, no markdown.

Rules:
- Identify each required skill / technology / concept. Lowercase and canonicalise names.
- For each: assign priority 1-5 (5 = critical) and must_have:true if the JD calls it required/essential/must.
- Decompose the JD into atomic requirement strings (one obligation per item).
- Capture the role title if present.
- Be exhaustive on technical asks; ignore boilerplate (benefits, EEO clauses, location).

JOB DESCRIPTION:
---
{jd_text}
---
"""
