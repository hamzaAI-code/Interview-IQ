"""FastAPI application — HTTP entry point alongside Streamlit.

Run with:
    uvicorn app.api.main:app --reload --port 8000

Streamlit (port 8501) and this API (port 8000) share Neo4j as the persistent
session store. Each process keeps its own in-memory orchestrator dict, so a
session built via Streamlit needs to be `POST /sessions/{id}/load`'d into the
API process before its interview-flow endpoints will work — and vice versa.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `app.*` importable when launched from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import interview, sessions
from app.api.schemas import HealthResponse


app = FastAPI(
    title="Technical Interview Agent API",
    version="1.0.0",
    description=(
        "HTTP layer over the interview orchestrator. Wraps "
        "`services.session_builder` (graph build + load + teardown) and "
        "`core.orchestrator` (interview flow) with Pydantic-validated routes."
    ),
)

# Permissive CORS for local dev — tighten origins for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(interview.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse()
