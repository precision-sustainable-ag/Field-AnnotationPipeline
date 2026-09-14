from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", log_dir: str | Path | None = None) -> Path | None:
    """Console logging always; also logs to a timestamped file under log_dir
    when given, so a long/unattended run (e.g. --limit 1000) can be checked
    afterward instead of only living in whatever terminal was watching it.
    Returns the log file path, or None if log_dir wasn't given.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_path = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"{timestamp}.log"
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=getattr(logging, level.upper()), format=LOG_FORMAT, handlers=handlers, force=True)
    return log_path


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
