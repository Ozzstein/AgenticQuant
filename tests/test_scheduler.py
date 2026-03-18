"""Tests for src/scheduler.py — idempotency, lock, dry-run, step execution."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.scheduler as sched
from src.scheduler import (
    DAILY_STEPS,
    _acquire_lock,
    _load_state,
    _run_step,
    _save_state,
    _step_done_today,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Redirect _STATE_FILE and _LOCK_FILE to tmp_path."""
    state_file = tmp_path / "scheduler_state.json"
    lock_file = tmp_path / ".scheduler.lock"
    monkeypatch.setattr(sched, "_STATE_FILE", state_file)
    monkeypatch.setattr(sched, "_LOCK_FILE", lock_file)
    return state_file, lock_file


def _noop() -> None:
    """No-operation step replacement."""
    return None


def _patch_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace all step functions in the registry with no-ops."""
    for name in list(sched._STEP_FUNCTIONS.keys()):
        monkeypatch.setitem(sched._STEP_FUNCTIONS, name, _noop)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_state_file_created_on_first_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """State file must be created after a simulate_day + dry_run run."""
    state_file, _ = _patch_paths(monkeypatch, tmp_path)

    run_pipeline(dry_run=True, simulate_day=True)

    assert state_file.exists(), "State file should be created after run."


def test_idempotency_same_step_not_run_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_step_done_today returns True for a step already marked 'ok'."""
    state_file, _ = _patch_paths(monkeypatch, tmp_path)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    step_name = DAILY_STEPS[0][0]
    state = {today: {step_name: {"status": "ok", "ts": "2025-01-01T06:00:00"}}}
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))

    loaded = _load_state()
    assert _step_done_today(loaded, today, step_name) is True


def test_lock_file_prevents_double_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_acquire_lock() must return False when a fresh lock file already exists."""
    _, lock_file = _patch_paths(monkeypatch, tmp_path)

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.touch()
    # Ensure the mtime is within the 2-hour window (fresh lock)
    now = time.time()
    import os
    os.utime(lock_file, (now, now))

    result = _acquire_lock()
    assert result is False, "Lock should not be acquired when a fresh lock file exists."


def test_dry_run_logs_all_9_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 9 step names must be recorded in the state file after dry-run simulate-day.

    loguru logs to the real stderr fd and bypasses pytest capsys capture, so we
    verify state-file contents instead, which is the canonical record of what was
    executed.
    """
    state_file, _ = _patch_paths(monkeypatch, tmp_path)

    run_pipeline(dry_run=True, simulate_day=True)

    state = json.loads(state_file.read_text())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    recorded_steps = set(state.get(today, {}).keys())

    for step_name, _, _ in DAILY_STEPS:
        assert step_name in recorded_steps, f"Step '{step_name}' not found in state file."


def test_step_failure_logs_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If one step raises, the remaining steps still run."""
    _patch_paths(monkeypatch, tmp_path)
    _patch_all_steps(monkeypatch)

    first_step = DAILY_STEPS[0][0]
    call_log: list[str] = []

    def _failing_step() -> None:
        raise RuntimeError("Simulated step failure")

    def _tracking_step(name: str):
        def _inner() -> None:
            call_log.append(name)
        return _inner

    monkeypatch.setitem(sched._STEP_FUNCTIONS, first_step, _failing_step)
    for step_name, _, _ in DAILY_STEPS[1:]:
        monkeypatch.setitem(sched._STEP_FUNCTIONS, step_name, _tracking_step(step_name))

    # Patch AlertManager at its source to avoid import side-effects
    with patch("src.monitoring.monitoring.AlertManager", MagicMock()):
        run_pipeline(dry_run=False, simulate_day=True)

    # All steps after the first should still have run
    for step_name, _, _ in DAILY_STEPS[1:]:
        assert step_name in call_log, f"Step '{step_name}' should have run after failure."


def test_simulate_day_executes_all_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State file must contain all 9 step entries after simulate_day + dry_run."""
    state_file, _ = _patch_paths(monkeypatch, tmp_path)

    run_pipeline(dry_run=True, simulate_day=True)

    state = json.loads(state_file.read_text())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_state = state.get(today, {})
    expected_steps = {s for s, _, _ in DAILY_STEPS}
    assert expected_steps == set(today_state.keys()), "All 9 steps should be in state file."


def test_state_file_idempotency_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running twice should not duplicate step entries in state file."""
    state_file, _ = _patch_paths(monkeypatch, tmp_path)

    run_pipeline(dry_run=True, simulate_day=True)
    state_after_first = json.loads(state_file.read_text())

    run_pipeline(dry_run=True, simulate_day=True)
    state_after_second = json.loads(state_file.read_text())

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # The keys for today should be identical (no duplicates)
    assert set(state_after_first.get(today, {}).keys()) == set(
        state_after_second.get(today, {}).keys()
    )


def test_stale_lock_file_is_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock file older than 2 hours must be cleared and lock acquired."""
    _, lock_file = _patch_paths(monkeypatch, tmp_path)

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.touch()

    # Set mtime to 3 hours ago
    stale_mtime = time.time() - 3 * 3600
    import os
    os.utime(lock_file, (stale_mtime, stale_mtime))

    result = _acquire_lock()
    assert result is True, "Stale lock file should be cleared and lock acquired."

    # Cleanup
    lock_file.unlink(missing_ok=True)
