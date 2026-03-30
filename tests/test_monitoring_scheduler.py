"""Unit tests for PipelineScheduler intraday scheduling methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.scheduler import PipelineScheduler
from src.utils.config import AppConfig, IntradayConfig


@pytest.fixture
def mock_scheduler():
    """Return a PipelineScheduler with a mocked APScheduler."""
    with patch("src.monitoring.scheduler.BackgroundScheduler") as MockSched:
        mock_sched_instance = MagicMock()
        mock_sched_instance.running = True
        MockSched.return_value = mock_sched_instance
        sched = PipelineScheduler(config=AppConfig())
        yield sched, mock_sched_instance


def test_start_intraday_adds_interval_job(mock_scheduler):
    """start_intraday() adds an interval job with the correct minutes parameter."""
    sched, mock_sched = mock_scheduler
    mock_job = MagicMock()
    mock_job.id = "intraday_runner"
    mock_sched.add_job.return_value = mock_job

    runner = MagicMock()
    sched.start_intraday(runner, 60)

    mock_sched.add_job.assert_called_once()
    call_kwargs = mock_sched.add_job.call_args
    assert call_kwargs.kwargs["trigger"] == "interval"
    assert call_kwargs.kwargs["minutes"] == 60
    assert sched._intraday_job_id == "intraday_runner"


def test_start_intraday_no_scheduler_is_noop():
    """start_intraday() is a no-op when APScheduler is unavailable."""
    sched = PipelineScheduler.__new__(PipelineScheduler)
    sched._config = AppConfig()
    sched.config = sched._config
    sched._scheduler = None
    sched._intraday_job_id = None

    runner = MagicMock()
    # Should not raise
    sched.start_intraday(runner)
    assert sched._intraday_job_id is None


def test_stop_intraday_removes_job(mock_scheduler):
    """stop_intraday() calls remove_job and clears _intraday_job_id."""
    sched, mock_sched = mock_scheduler
    mock_job = MagicMock()
    mock_job.id = "intraday_runner"
    mock_sched.add_job.return_value = mock_job

    runner = MagicMock()
    sched.start_intraday(runner, 60)
    assert sched._intraday_job_id == "intraday_runner"

    sched.stop_intraday()

    mock_sched.remove_job.assert_called_once_with("intraday_runner")
    assert sched._intraday_job_id is None


def test_stop_calls_stop_intraday(mock_scheduler):
    """stop() removes the intraday job AND shuts down the scheduler."""
    sched, mock_sched = mock_scheduler
    mock_job = MagicMock()
    mock_job.id = "intraday_runner"
    mock_sched.add_job.return_value = mock_job

    runner = MagicMock()
    sched.start_intraday(runner, 60)

    sched.stop()

    mock_sched.remove_job.assert_called_once_with("intraday_runner")
    mock_sched.shutdown.assert_called_once()
    assert sched._intraday_job_id is None


def test_interval_minutes_from_config(mock_scheduler):
    """start_intraday() uses config.intraday.interval_minutes when no override given."""
    sched, mock_sched = mock_scheduler
    sched._config.intraday = IntradayConfig(interval_minutes=120)

    mock_job = MagicMock()
    mock_job.id = "intraday_runner"
    mock_sched.add_job.return_value = mock_job

    runner = MagicMock()
    sched.start_intraday(runner)

    call_kwargs = mock_sched.add_job.call_args
    assert call_kwargs.kwargs["minutes"] == 120


def test_interval_minutes_override(mock_scheduler):
    """start_intraday() uses the explicit override, ignoring config default."""
    sched, mock_sched = mock_scheduler
    sched._config.intraday = IntradayConfig(interval_minutes=240)

    mock_job = MagicMock()
    mock_job.id = "intraday_runner"
    mock_sched.add_job.return_value = mock_job

    runner = MagicMock()
    sched.start_intraday(runner, interval_minutes=30)

    call_kwargs = mock_sched.add_job.call_args
    assert call_kwargs.kwargs["minutes"] == 30


def test_stop_intraday_no_job_is_noop(mock_scheduler):
    """stop_intraday() without prior start_intraday does not raise and leaves state clean."""
    sched, mock_sched = mock_scheduler

    sched.stop_intraday()  # should not raise

    mock_sched.remove_job.assert_not_called()
    assert sched._intraday_job_id is None


def test_intraday_job_id_stored_after_start(mock_scheduler):
    """_intraday_job_id is populated after a successful start_intraday()."""
    sched, mock_sched = mock_scheduler
    mock_job = MagicMock()
    mock_job.id = "intraday_runner"
    mock_sched.add_job.return_value = mock_job

    runner = MagicMock()
    sched.start_intraday(runner, 15)

    assert sched._intraday_job_id is not None
    assert sched._intraday_job_id == "intraday_runner"
