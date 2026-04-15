"""Tests for src.utils.logging_setup."""

from __future__ import annotations

import io
import json
import logging

import pytest

from src.utils import logging_setup
from src.utils.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Ensure each test starts with a fresh root logger / run_id."""
    logging_setup._CURRENT_RUN_ID = None
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for flt in list(root.filters):
        root.removeFilter(flt)
    yield
    logging_setup._CURRENT_RUN_ID = None


def test_setup_logging_returns_run_id_of_sufficient_length():
    run_id = setup_logging("INFO")
    assert isinstance(run_id, str)
    assert len(run_id) >= 8


def test_setup_logging_json_format_emits_parseable_json():
    buffer = io.StringIO()
    run_id = setup_logging("INFO", json_format=True, stream=buffer)

    logging.getLogger("test.jsonfmt").info("hello-json")

    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"
    payload = json.loads(lines[-1])
    assert payload["message"] == "hello-json"
    assert payload["level"] == "INFO"
    assert payload["run_id"] == run_id
    assert payload["name"] == "test.jsonfmt"
    assert "timestamp" in payload


def test_setup_logging_accepts_explicit_run_id():
    run_id = setup_logging("INFO", run_id="fixed-id")
    assert run_id == "fixed-id"
    assert logging_setup.get_run_id() == "fixed-id"


def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    first_count = len(logging.getLogger().handlers)
    setup_logging("INFO")
    second_count = len(logging.getLogger().handlers)
    assert first_count == second_count == 1
