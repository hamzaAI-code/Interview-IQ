"""Prompts for extracting structured graphs from raw resume / JD text.

Resume extraction is split into two parallel Gemini calls so the heavy
"experiences + projects" pass doesn't block topic derivation that only needs
the skills inventory.
"""

# Core pass: everything needed to derive topics (JD overlap, ranking).
RESUME_CORE_PROMPT = """You are a precise information extractor. From the RESUME below, extract the candidate's TECHNICAL INVENTORY.

Return STRICT JSON matching the provided schema. No prose, no markdown.

Rules:
- Skill names lowercased, canonical (e.g. "Python 3" -> "python", "K8s" -> "kubernetes").
- A "skill" is a discrete capability (language, framework, tool, methodology).
- A "technology" is a specific product (e.g. "tensorflow", "fastapi", "postgres").
- A "concept" is a topic of knowledge (e.g. "transformer attention", "event sourcing").
- For each skill: estimate years from context, mark proficiency in {{beginner, intermediate, advanced, expert}}, and capture one short evidence string pulled from the resume.
- BE EXHAUSTIVE — include every skill, technology, and concept the candidate mentions anywhere in the resume (summary, experiences, projects, skills section, certifications, education). Do NOT drop items to keep the list short.
- Prefer canonical lowercase form; deduplicate by canonical name.
- If the candidate's name is in the resume, include it; otherwise leave name empty.

RESUME:
---
{resume_text}
---
"""


# Depth pass: experiences + projects with their used tech. Runs in parallel with the core pass.
RESUME_DEPTH_PROMPT = """You are a precise information extractor. From the RESUME below, extract the candidate's EXPERIENCES and PROJECTS.

Return STRICT JSON matching the provided schema. No prose, no markdown.

Rules:
- Capture every distinct experience (role + company + years + 1-line summary). Be exhaustive — don't skip older roles.
- Capture every distinct project (name + summary + technologies used). Be exhaustive.
- For each experience, list the technologies/skills used in that role (lowercased, canonical names — same canonicalisation as the skills section).
- For each project, list the technologies used (lowercased, canonical).
- Names inside `used` / `technologies` MUST use the exact same lowercase canonical form as the candidate's skills list would use, so they can be linked.
- BE EXHAUSTIVE — don't drop technologies from an experience/project to keep the list short.

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
