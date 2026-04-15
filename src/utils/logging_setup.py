"""Structured logging setup with per-run correlation IDs.

Provides :func:`setup_logging` and :func:`get_run_id`. Every log record
carries a ``run_id`` attribute (injected via a :class:`logging.Filter`)
so downstream aggregators can correlate lines belonging to the same
training / evaluation run.

Two output formats are supported:

* ``json_format=False`` (default) — a human-readable line
  ``"<ts> [<name>] <LEVEL> (run=<run_id>): <message>"``.
* ``json_format=True`` — one JSON object per line with keys
  ``timestamp``, ``level``, ``name``, ``run_id``, ``message``.

The function is idempotent: repeated calls reset the root logger's
handlers before installing new ones so tests and re-entrant callers do
not accumulate duplicates.

Environment overrides:
    WAFER_RUN_ID     — use as run_id when ``run_id`` argument is None.
    WAFER_LOG_LEVEL  — overrides ``level`` argument when set.
    WAFER_LOG_FORMAT — ``json`` or ``human`` (default human).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_HUMAN_FORMAT = "%(asctime)s [%(name)s] %(levelname)s (run=%(run_id)s): %(message)s"

# Module-level state so get_run_id() / filter can share a single id.
_CURRENT_RUN_ID: Optional[str] = None


class _RunIdFilter(logging.Filter):
    """Inject the active run_id onto every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not hasattr(record, "run_id") or not getattr(record, "run_id", None):
            record.run_id = _CURRENT_RUN_ID or "-"
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "run_id": getattr(record, "run_id", _CURRENT_RUN_ID or "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _generate_run_id() -> str:
    return uuid.uuid4().hex[:12]


def setup_logging(
    level: str = "INFO",
    *,
    run_id: Optional[str] = None,
    json_format: bool = False,
    stream=sys.stdout,
) -> str:
    """Configure the root logger and return the resolved ``run_id``.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``). Env var ``WAFER_LOG_LEVEL``
        overrides when set.
    run_id:
        Explicit run identifier. If ``None`` uses ``WAFER_RUN_ID`` from
        the environment, or generates a short uuid4 hex.
    json_format:
        Emit JSON lines instead of human-readable lines. Env var
        ``WAFER_LOG_FORMAT=json`` forces JSON when the argument is False.
    stream:
        Output stream for the attached :class:`logging.StreamHandler`.

    Returns
    -------
    str
        The resolved ``run_id`` (also stored module-locally for
        :func:`get_run_id`).
    """
    global _CURRENT_RUN_ID

    env_level = os.getenv("WAFER_LOG_LEVEL")
    resolved_level = (env_level or level or "INFO").upper()

    env_format = os.getenv("WAFER_LOG_FORMAT", "").strip().lower()
    use_json = json_format or env_format == "json"

    resolved_run_id = run_id or os.getenv("WAFER_RUN_ID") or _generate_run_id()
    _CURRENT_RUN_ID = resolved_run_id

    root = logging.getLogger()
    # Idempotent: drop existing handlers/filters we may have installed.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for flt in list(root.filters):
        if isinstance(flt, _RunIdFilter):
            root.removeFilter(flt)

    handler = logging.StreamHandler(stream)
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT))
    handler.addFilter(_RunIdFilter())

    root.addHandler(handler)
    root.addFilter(_RunIdFilter())
    root.setLevel(getattr(logging, resolved_level, logging.INFO))

    return resolved_run_id


def get_run_id() -> str:
    """Return the current run_id, generating one if none has been set."""
    global _CURRENT_RUN_ID
    if not _CURRENT_RUN_ID:
        _CURRENT_RUN_ID = os.getenv("WAFER_RUN_ID") or _generate_run_id()
    return _CURRENT_RUN_ID
