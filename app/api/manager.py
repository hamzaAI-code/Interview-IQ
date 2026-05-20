"""Process-local registry of attached Orchestrators, keyed by session_id.

The HTTP API needs to handle multiple concurrent sessions. The orchestrator is
stateful (qa, coverage, contradictions, prefetched_question, etc. live on the
in-memory InterviewState), so we keep one Orchestrator per active session in a
dict guarded by an asyncio.Lock.

Note: this is in-process state only. If you scale to multiple workers, you'll
need to back this with Redis or similar — but that requires serialising
InterviewState too. For a single-worker dev setup it's fine.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from app.core.orchestrator import Orchestrator
from app.utils.logger import get_logger

log = get_logger(__name__)


class SessionManager:
    def __init__(self) -> None:
        self._orchs: dict[str, Orchestrator] = {}
        self._lock = asyncio.Lock()

    async def attach(self, orch: Orchestrator) -> None:
        assert orch.state is not None, "orchestrator must already be attached to a state"
        sid = orch.state.session_id
        async with self._lock:
            self._orchs[sid] = orch
        log.info("session=%s registered with API session manager", sid)

    async def get(self, session_id: str) -> Orchestrator | None:
        async with self._lock:
            return self._orchs.get(session_id)

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._orchs.pop(session_id, None)
        log.info("session=%s removed from API session manager", session_id)

    async def known_ids(self) -> list[str]:
        async with self._lock:
            return list(self._orchs.keys())


@lru_cache(maxsize=1)
def get_session_manager() -> SessionManager:
    return SessionManager()
