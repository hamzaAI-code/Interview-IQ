"""Pre-configured stdlib logger."""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=1)
def _configure() -> None:
    settings = get_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
