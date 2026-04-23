"""Resume + JD extraction into Pydantic-validated graphs."""
from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.gemini_client import generate_structured
from app.prompts.extraction import JD_EXTRACTION_PROMPT, RESUME_EXTRACTION_PROMPT
from app.utils.logger import get_logger

log = get_logger(__name__)


# ---------- Resume schema ----------

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
    used: list[str] = Field(default_factory=list, description="skill or tech names used")


class ProjectEntry(BaseModel):
    name: str
    summary: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeGraph(BaseModel):
    candidate_name: str = ""
    skills: list[SkillEntry] = Field(default_factory=list)
    technologies: list[TechEntry] = Field(default_factory=list)
    concepts: list[ConceptEntry] = Field(default_factory=list)
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

async def extract_resume(text: str) -> ResumeGraph:
    settings = get_settings()
    prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=text[:30000])
    return await generate_structured(
        prompt=prompt,
        schema=ResumeGraph,
        model=settings.gemini_model_pro,
        temperature=0.1,
    )


async def extract_jd(text: str) -> JDGraph:
    settings = get_settings()
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=text[:20000])
    return await generate_structured(
        prompt=prompt,
        schema=JDGraph,
        model=settings.gemini_model_pro,
        temperature=0.1,
    )


async def extract_both(resume_text: str, jd_text: str) -> tuple[ResumeGraph, JDGraph]:
    """Run resume + JD extraction concurrently."""
    log.info("extracting resume + JD in parallel")
    resume, jd = await asyncio.gather(
        extract_resume(resume_text),
        extract_jd(jd_text),
    )
    log.info(
        "extraction done: %d skills, %d techs, %d concepts | %d JD requirements",
        len(resume.skills), len(resume.technologies), len(resume.concepts),
        len(jd.requirements),
    )
    return resume, jd
