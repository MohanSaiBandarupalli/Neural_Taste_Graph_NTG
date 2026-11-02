"""Structured logging with secret / PII scrubbing."""
from __future__ import annotations

import json
import logging
import re
import sys
import time

_SCRUB = re.compile(r"(?i)(password|secret|token|api[_-]?key|authorization)(\"?\s*[:=]\s*\"?)([^\s\",}]+)")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = _SCRUB.sub(r"\1\2***", record.getMessage())
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
        }
        for key in ("request_id", "path", "status", "latency_ms", "tenant"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def get_logger(name: str = "ntg") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def scrub(text: str) -> str:
    return _SCRUB.sub(r"\1\2***", text)
