"""Environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model_fast: str
    gemini_model_pro: str

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    coverage_threshold: float
    max_followups: int
    max_topics: int
    max_clarifications: int

    log_level: str

    def require_keys(self) -> None:
        if not self.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in environment / .env")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model_fast=os.getenv("GEMINI_MODEL_FAST", "gemini-2.5-flash"),
        gemini_model_pro=os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-pro"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "interview_agent_pw"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        coverage_threshold=float(os.getenv("COVERAGE_THRESHOLD", "0.70")),
        max_followups=int(os.getenv("MAX_FOLLOWUPS", "3")),
        max_topics=int(os.getenv("MAX_TOPICS", "8")),
        max_clarifications=int(os.getenv("MAX_CLARIFICATIONS", "2")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
