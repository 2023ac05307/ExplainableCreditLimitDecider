from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class LogConfig:
    """
    Production logging config.

    - level: INFO by default
    - json: structured logs for cloud ingestion
    - name: logger namespace
    """
    level: str = "INFO"
    json: bool = False
    name: str = "app"
    propagate: bool = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # attach extras if any were passed
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in {
                "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName", "processName", "process",
                "name"
            }:
                continue
            payload.setdefault("extra", {})[k] = v
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(cfg: Optional[LogConfig] = None) -> None:
    """
    Configure root logger once. Safe to call multiple times.
    Respects env:
      - LOG_LEVEL
      - LOG_JSON (true/false)
    """
    cfg = cfg or LogConfig()
    env_level = os.getenv("LOG_LEVEL")
    env_json = os.getenv("LOG_JSON")

    level = (env_level or cfg.level).upper()
    json_mode = cfg.json
    if env_json is not None:
        json_mode = env_json.strip().lower() in {"1", "true", "yes", "y"}

    root = logging.getLogger()
    root.setLevel(level)

    # avoid duplicate handlers if called again
    if getattr(root, "_configured_by_common", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_mode:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    root.addHandler(handler)
    root.propagate = cfg.propagate
    setattr(root, "_configured_by_common", True)


def get_logger(name: str = "app") -> logging.Logger:
    """
    Get a namespaced logger. Ensure configure_logging() was called at entrypoint.
    """
    return logging.getLogger(name)
