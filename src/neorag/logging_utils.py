from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("neorag")
    if logger.handlers:
        return logger
    handler = RichHandler(rich_tracebacks=True, markup=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = setup_logging()
