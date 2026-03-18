"""Tests for src/utils/cost_tracker.py — CostTracker singleton."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.utils.cost_tracker as ct_mod
from src.utils.cost_tracker import CostTracker, reset_instance


# ---------------------------------------------------------------------------
# Singleton reset fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cost_tracker() -> None:
    """Reset the CostTracker singleton before and after each test."""
    reset_instance()
    yield
    reset_instance()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_call_correct_cost_sonnet() -> None:
    """record_call() for claude-sonnet-4-6 with 1000 prompt + 500 completion tokens."""
    ct = CostTracker()
    cost = ct.record_call("claude-sonnet-4-6", 1000, 500)

    # input: 1000 / 1_000_000 * 3.0 = 0.003
    # output: 500 / 1_000_000 * 15.0 = 0.0075
    # total: 0.0105
    assert abs(cost - 0.0105) < 1e-9, f"Expected ~0.0105, got {cost}"


def test_get_session_cost_sums_multiple_calls() -> None:
    """get_session_cost() should sum all recorded call costs."""
    ct = CostTracker()
    c1 = ct.record_call("claude-sonnet-4-6", 1000, 500)
    c2 = ct.record_call("gpt-4o", 2000, 1000)
    c3 = ct.record_call("claude-haiku-4-5-20251001", 500, 200)

    expected = c1 + c2 + c3
    actual = ct.get_session_cost()
    assert abs(actual - expected) < 1e-9


def test_save_daily_load_daily_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_daily() / load_daily() round-trip should preserve total_usd."""
    monkeypatch.setattr(ct_mod, "_PROJECT_ROOT", tmp_path)

    ct = CostTracker()
    ct.record_call("claude-sonnet-4-6", 1000, 500)

    session_cost = ct.get_session_cost()
    ct.save_daily("2025-06-01")

    loaded = ct.load_daily("2025-06-01")
    assert abs(loaded["total_usd"] - session_cost) < 1e-9


def test_unknown_model_defaults_to_zero() -> None:
    """Unknown model should record 0.0 cost and not raise."""
    ct = CostTracker()
    cost = ct.record_call("gpt-99-unknown", 1000, 500)
    assert cost == 0.0


def test_reset_session_clears_accumulator() -> None:
    """reset_session() should zero out the session cost."""
    ct = CostTracker()
    ct.record_call("claude-sonnet-4-6", 1000, 500)
    assert ct.get_session_cost() > 0.0

    ct.reset_session()
    assert ct.get_session_cost() == 0.0


def test_load_daily_returns_empty_dict_for_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_daily() should return {} when the file does not exist."""
    monkeypatch.setattr(ct_mod, "_PROJECT_ROOT", tmp_path)

    ct = CostTracker()
    result = ct.load_daily("2025-01-01")
    assert result == {}
