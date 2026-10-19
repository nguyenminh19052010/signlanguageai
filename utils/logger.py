
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    name: str = "signlang",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    numeric = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(numeric)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
