"""Persistent background event loop for bridging sync (Streamlit) ↔ async code.

Why: Neo4j async driver and the Gemini async client bind to whichever loop they were
first awaited on. Streamlit reruns the script each interaction; if we kept calling
asyncio.run() we'd build/tear down loops constantly and the cached driver/client
would break. Instead we run a single daemon-thread loop for the whole app lifetime
and submit coroutines via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from functools import lru_cache
from queue import Queue
from typing import AsyncIterator, Awaitable, Callable, Iterator, TypeVar

T = TypeVar("T")


class AsyncRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._serve, name="interview-async-loop", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Awaitable[T]) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Awaitable[T]) -> T:
        return self.submit(coro).result()

    def stream(self, agen_factory: Callable[[], AsyncIterator[str]]) -> Iterator[str]:
        """Bridge an async iterator to a sync iterator (drains via a queue)."""
        q: Queue = Queue(maxsize=64)

        async def _drain() -> None:
            try:
                async for chunk in agen_factory():
                    q.put(("chunk", chunk))
                q.put(("done", None))
            except Exception as e:  # surface to caller
                q.put(("error", e))

        self.submit(_drain())
        while True:
            tag, val = q.get()
            if tag == "done":
                return
            if tag == "error":
                raise val  # type: ignore[misc]
            yield val  # type: ignore[misc]


@lru_cache(maxsize=1)
def get_runner() -> AsyncRunner:
    return AsyncRunner()
