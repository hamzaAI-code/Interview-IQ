"""Async Neo4j driver wrapper."""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_driver() -> AsyncDriver:
    s = get_settings()
    return AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


@asynccontextmanager
async def session():
    s = get_settings()
    driver = get_driver()
    async with driver.session(database=s.neo4j_database) as sess:
        yield sess


async def run(cypher: str, **params: Any) -> list[dict]:
    """Run a single Cypher query and return all records as dicts."""
    async with session() as sess:
        result = await sess.run(cypher, **params)
        records = [r.data() async for r in result]
        return records


async def run_write(cypher: str, **params: Any) -> None:
    """Run a write query (no return needed)."""
    async with session() as sess:
        await sess.run(cypher, **params)


async def close_driver() -> None:
    try:
        driver = get_driver()
        await driver.close()
    except Exception as e:  # pragma: no cover
        log.warning("driver close failed: %s", e)
