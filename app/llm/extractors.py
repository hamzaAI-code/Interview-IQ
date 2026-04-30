"""Resume + JD extraction.

Resume is extracted as TWO parallel Gemini calls (core inventory vs. experiences/projects)
so the slower depth pass overlaps the JD extraction and does not block topic derivation
on the critical path.

All three extractions use Flash + thinking_budget=0 — a pattern-matching task, not a
reasoning task, so internal thinking is wasted latency.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.gemini_client import generate_structured
from app.prompts.extraction import (
    JD_EXTRACTION_PROMPT,
    RESUME_CORE_PROMPT,
    RESUME_DEPTH_PROMPT,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------- Resume schemas (split) ----------

class SkillEntry(BaseModel):
    name: str = Field(description="Lowercased canonical skill name")
    category: Literal["language", "framework", "tool", "methodology", "other"] = "other"
    years: float = 0.0
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"] = "intermediate"
    evidence: str = ""


class TechEntry(BaseModel):
    name: str
    evidence: str = ""


class ConceptEntry(BaseModel):
    name: str
    evidence: str = ""


class ExperienceEntry(BaseModel):
    role: str
    company: str = ""
    years: float = 0.0
    summary: str = ""
    used: list[str] = Field(default_factory=list, description="literal skill or tech names used")
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Broader concept-level capabilities demonstrated in this role — what the work "
            "actually involved beyond literal tech (e.g. 'model deployment', "
            "'real-time inference', 'data pipeline orchestration', 'incident response')."
        ),
    )


class ProjectEntry(BaseModel):
    name: str
    summary: str = ""
    technologies: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Broader concept-level capabilities demonstrated in this project — what the work "
            "actually involved beyond literal tech (e.g. 'retrieval augmented generation', "
            "'vector search', 'distributed training', 'feature engineering')."
        ),
    )


class ResumeCoreGraph(BaseModel):
    """Fast pass: candidate identity + technical inventory."""
    candidate_name: str = ""
    skills: list[SkillEntry] = Field(default_factory=list)
    technologies: list[TechEntry] = Field(default_factory=list)
    concepts: list[ConceptEntry] = Field(default_factory=list)


class ResumeDepthGraph(BaseModel):
    """Depth pass: experiences + projects (for question-grounding)."""
    experiences: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)


# ---------- JD schema ----------

class JDRequirement(BaseModel):
    name: str = Field(description="Lowercased canonical name of the required skill/tech/concept")
    kind: Literal["skill", "technology", "concept"] = "skill"
    priority: int = Field(default=3, ge=1, le=5)
    must_have: bool = False


class JDGraph(BaseModel):
    role_title: str = ""
    requirements: list[JDRequirement] = Field(default_factory=list)
    requirement_lines: list[str] = Field(default_factory=list)


# ---------- Public API ----------

_RESUME_CLIP = 30000
_JD_CLIP = 20000


async def extract_resume_core(text: str) -> ResumeCoreGraph:
    settings = get_settings()
    prompt = RESUME_CORE_PROMPT.format(resume_text=text[:_RESUME_CLIP])
    return await generate_structured(
        prompt=prompt,
        schema=ResumeCoreGraph,
        model=settings.gemini_model_fast,
        temperature=0.1,
        thinking_budget=0,
    )


async def extract_resume_depth(text: str) -> ResumeDepthGraph:
    settings = get_settings()
    prompt = RESUME_DEPTH_PROMPT.format(resume_text=text[:_RESUME_CLIP])
    return await generate_structured(
        prompt=prompt,
        schema=ResumeDepthGraph,
        model=settings.gemini_model_fast,
        temperature=0.1,
        thinking_budget=0,
    )


async def extract_jd(text: str) -> JDGraph:
    settings = get_settings()
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=text[:_JD_CLIP])
    return await generate_structured(
        prompt=prompt,
        schema=JDGraph,
        model=settings.gemini_model_fast,
        temperature=0.1,
        thinking_budget=0,
    )
