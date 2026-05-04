"""Map tekprep's parsed resume / JD schemas to Interview-IQ's internal graph shapes.

Lossy on purpose — tekprep's payload doesn't carry skill/tech/concept `kind`
classification, per-skill years/proficiency, project capabilities, or JD
must_have flags. We fill safe defaults so the downstream Cypher (ingestion +
topic derivation) keeps working unchanged.

Anything tekprep DOES carry (names, project stacks, role titles, candidate
name, requirement weights) is propagated straight through.
"""
from __future__ import annotations

import math
import re

from app.api.external_schemas import TekprepJDSchema, TekprepResumeSchema
from app.llm.extractors import (
    ConceptEntry,
    ExperienceEntry,
    JDGraph,
    JDRequirement,
    ProjectEntry,
    ResumeCoreGraph,
    ResumeDepthGraph,
    SkillEntry,
    TechEntry,
)


_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9][a-z0-9+#./-]*")


def _canon(name: str) -> str:
    """Match the lowercased-canonical form Interview-IQ's extractor uses."""
    return _WS.sub(" ", (name or "").strip().lower())


# A small static fallback vocabulary — common tech terms that may appear in
# JD requirement text but NOT in a candidate's resume (so we can still
# canonicalise even when the resume vocab doesn't help). Kept short on
# purpose: this is a safety net, not a comprehensive ontology.
_FALLBACK_TECH_TERMS: tuple[str, ...] = (
    "python", "javascript", "typescript", "java", "go", "golang", "rust",
    "c++", "c#", "ruby", "php", "kotlin", "swift", "scala", "sql",
    "fastapi", "flask", "django", "express", "node", "node.js", "nodejs",
    "react", "next", "next.js", "nextjs", "vue", "angular", "svelte",
    "redux", "graphql", "rest", "grpc",
    "postgres", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "snowflake", "bigquery", "kafka", "rabbitmq",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "helm",
    "aws", "gcp", "azure", "ec2", "s3", "lambda", "cloudfunctions",
    "tensorflow", "pytorch", "scikit-learn", "sklearn", "pandas", "numpy",
    "spark", "hadoop", "airflow", "dbt",
    "git", "github", "gitlab", "ci/cd", "jenkins", "circleci",
    "linux", "bash", "macos", "windows",
    "rag", "llm", "langchain", "langgraph", "vector", "embedding", "embeddings",
    "openai", "anthropic", "gemini", "claude", "huggingface",
)


def _build_skill_vocab(resume_terms: list[str]) -> list[str]:
    """Combine the resume's own canonical terms with the static fallback.

    Resume terms come first (longer matches preferred via sorting below)
    so candidate-specific phrasings beat generic dictionary entries.
    """
    seen: set[str] = set()
    vocab: list[str] = []
    for raw in (*resume_terms, *_FALLBACK_TECH_TERMS):
        c = _canon(raw)
        if not c or c in seen:
            continue
        seen.add(c)
        vocab.append(c)
    # Longest first so "next.js" beats "next" when both are present.
    vocab.sort(key=len, reverse=True)
    return vocab


def _extract_skill_name(text: str, vocab: list[str]) -> str:
    """Pull a canonical skill name out of free-form JD requirement text.

    Strategy:
      1. Walk the vocab (longest-first) and return the first canonical
         term that appears as a substring of the lowercased text.
      2. If nothing matches, fall back to the lowercased full text — same
         as the previous behaviour. The downstream `:Skill` node will be
         orphaned (no resume overlap), which is fine: derive_topics keeps
         it with a low score thanks to the gap_penalty, and Interview-IQ
         won't rank it ahead of real overlaps.
    """
    if not text:
        return ""
    lowered = _canon(text)
    if not lowered:
        return ""
    # Tokenize to word-ish boundaries so "javascript" doesn't accidentally
    # match inside "macroscript". Build a set of the tokens for O(1) check.
    tokens = set(_WORD.findall(lowered))
    for term in vocab:
        # Multi-word vocab (e.g. "node.js", "ci/cd"): substring match is
        # fine because punctuation is preserved by _WORD's char class.
        if " " in term or "/" in term or "+" in term or "#" in term or "." in term or "-" in term:
            if term in lowered:
                return term
            continue
        if term in tokens:
            return term
    return lowered


def _summary(parts: list[str]) -> str:
    """Join a few responsibility / achievement bullets into one short summary."""
    blob = " ".join(p.strip() for p in parts if p and p.strip())
    return blob[:600]  # bounded so we don't bloat the ingest payload


def _years_from_tenure(tenure: str | None) -> float:
    """Best-effort tenure → years. Tekprep stores tenure as free-form text
    ("2020 - Present", "2 years", "Jan 2021 - Mar 2023"). We don't try hard
    here — Interview-IQ's downstream logic treats `years` as a soft signal and
    falls back gracefully to 0.0 / 'intermediate'."""
    if not tenure:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:year|yr)", tenure.lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _priority_from_weight(weight: float) -> int:
    """Map tekprep's 0.0-1.0 weight to Interview-IQ's 1-5 priority."""
    if weight is None:
        return 3
    p = int(math.ceil(max(0.0, min(1.0, float(weight))) * 5))
    return max(1, min(5, p or 1))


def tekprep_resume_to_graphs(
    resume: TekprepResumeSchema,
) -> tuple[ResumeCoreGraph, ResumeDepthGraph]:
    """Split tekprep's flat resume into Interview-IQ's core + depth graphs.

    `tech_stack` becomes the SkillEntry inventory (kind="other" by default,
    intermediate proficiency, no years — these are fields tekprep doesn't
    capture). technologies / concepts are left empty; topic derivation will
    still land them via the requirement_lines + skill matching path.
    """
    seen: set[str] = set()
    skills: list[SkillEntry] = []
    for raw in resume.tech_stack:
        name = _canon(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        skills.append(SkillEntry(name=name))

    # Project stacks may mention items not present in the top-level tech_stack.
    # Pull those into skills too so Interview-IQ's matcher can link them.
    for proj in resume.projects:
        for raw in proj.stack:
            name = _canon(raw)
            if not name or name in seen:
                continue
            seen.add(name)
            skills.append(SkillEntry(name=name))

    core = ResumeCoreGraph(
        candidate_name=resume.personal_info.name or "",
        skills=skills,
        technologies=[],  # tekprep doesn't distinguish tech vs skill
        concepts=[],      # tekprep doesn't capture concept-level items
    )

    experiences: list[ExperienceEntry] = []
    for exp in resume.experience:
        experiences.append(
            ExperienceEntry(
                role=exp.role or "",
                company=exp.company or "",
                years=_years_from_tenure(exp.tenure),
                summary=_summary([*exp.responsibilities, *exp.achievements]),
                used=[],          # not derivable from tekprep's shape
                capabilities=[],  # not derivable from tekprep's shape
            )
        )

    projects: list[ProjectEntry] = []
    for proj in resume.projects:
        bits: list[str] = []
        if proj.description:
            bits.append(proj.description)
        if proj.task:
            bits.append(proj.task)
        bits.extend(proj.achievements)
        projects.append(
            ProjectEntry(
                name=proj.name or "",
                summary=_summary(bits),
                technologies=[_canon(s) for s in proj.stack if _canon(s)],
                capabilities=[],
            )
        )

    depth = ResumeDepthGraph(experiences=experiences, projects=projects)
    return core, depth


def tekprep_jd_to_graph(
    jd: TekprepJDSchema,
    *,
    resume: TekprepResumeSchema | None = None,
) -> JDGraph:
    """Map tekprep's JD schema to Interview-IQ's JDGraph.

    Each tekprep `requirement.text` is reduced to a canonical skill name
    (e.g. "5+ years of Python experience" → "python") via _extract_skill_name
    using the resume's tech_stack + a static fallback vocab. This is what
    makes JD↔resume overlap actually happen in Neo4j — without it, every JD
    requirement becomes its own orphan node and topic derivation produces
    junk topics with no evidence.

    `requirement_lines` is built from BOTH responsibilities and requirements
    so the Neo4j :Requirement audit nodes still capture full JD content.
    """
    # Build the canonicalisation vocabulary once per call. Resume tech_stack
    # + project stacks first, then the static fallback for tech that wasn't
    # on the resume but is mentioned in the JD.
    resume_terms: list[str] = []
    if resume is not None:
        resume_terms.extend(resume.tech_stack)
        for proj in resume.projects:
            resume_terms.extend(proj.stack)
    vocab = _build_skill_vocab(resume_terms)

    seen_names: set[str] = set()
    requirements: list[JDRequirement] = []
    for r in jd.requirements:
        name = _extract_skill_name(r.text, vocab)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        requirements.append(
            JDRequirement(
                name=name,
                kind="skill",
                priority=_priority_from_weight(r.weight),
                must_have=(r.weight or 0.0) >= 0.7,
            )
        )

    lines: list[str] = []
    for resp in jd.responsibilities:
        text = (resp.text or "").strip()
        if text:
            lines.append(text[:200])
    for r in jd.requirements:
        text = (r.text or "").strip()
        if text:
            lines.append(text[:200])
    if jd.required_experience:
        lines.append(jd.required_experience.strip()[:200])
    if jd.required_education:
        lines.append(jd.required_education.strip()[:200])

    return JDGraph(
        role_title=(jd.role or "").strip()[:80] or "Role",
        requirements=requirements,
        requirement_lines=lines[:25],
    )
