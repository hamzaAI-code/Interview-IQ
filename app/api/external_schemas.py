"""Tekprep-compatible resume / JD schemas accepted by /sessions/build/structured.

These mirror tekprep/src/models/{resume,jd}_schema.py 1:1 so tekprep can post
its already-parsed payload as-is, without re-stringifying or re-extracting on
this side. The shapes are kept identical (field names + types) on purpose —
do not "improve" them here without also updating tekprep.

Internal Interview-IQ logic still operates on the richer JDGraph /
ResumeCoreGraph / ResumeDepthGraph shapes from app/llm/extractors.py. See
app/services/external_adapter.py for the mapping.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------- Resume ----------

class TekprepPersonalInfo(BaseModel):
    name: str = Field(..., description="Full name of the candidate")
    email: Optional[str] = None
    phone: Optional[str] = None
    links: Dict[str, str] = Field(default_factory=dict)


class TekprepExperience(BaseModel):
    company: str
    role: str
    tenure: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)


class TekprepProject(BaseModel):
    name: str
    description: Optional[str] = None
    stack: List[str] = Field(default_factory=list)
    task: Optional[str] = None
    achievements: List[str] = Field(default_factory=list)


class TekprepEducation(BaseModel):
    institution: str
    degree: str
    timeline: Optional[str] = None


class TekprepResumeSchema(BaseModel):
    personal_info: TekprepPersonalInfo
    tech_stack: List[str] = Field(default_factory=list)
    experience: List[TekprepExperience] = Field(default_factory=list)
    projects: List[TekprepProject] = Field(default_factory=list)
    education: List[TekprepEducation] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


# ---------- JD ----------

class TekprepResponsibility(BaseModel):
    text: str
    weight: float = Field(..., ge=0.0, le=1.0)


class TekprepRequirement(BaseModel):
    text: str
    weight: float = Field(..., ge=0.0, le=1.0)


class TekprepJDSchema(BaseModel):
    role: str
    responsibilities: List[TekprepResponsibility] = Field(default_factory=list)
    requirements: List[TekprepRequirement] = Field(default_factory=list)
    required_experience: str = ""
    required_education: str = ""
