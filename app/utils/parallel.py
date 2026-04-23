"""Async helpers for parallelism and timeouts."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Iterable, TypeVar

T = TypeVar("T")


async def gather_with_timeout(
    awaitables: Iterable[Awaitable[T]],
    timeout: float,
) -> list[T]:
    """Run awaitables concurrently with a shared timeout. Raises TimeoutError on overrun."""
    return await asyncio.wait_for(asyncio.gather(*awaitables), timeout=timeout)


def fire_and_forget(coro: Awaitable[None]) -> asyncio.Task:
    """Schedule a coroutine without awaiting it. Caller must keep a reference."""
    return asyncio.create_task(coro)


def run_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from sync code (e.g. Streamlit handlers)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        pass
    return asyncio.run(coro)
