"""Tests for src/monitoring/monitoring.py — AlertManager with rate limiting."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.monitoring import AlertManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alert_manager(mock_send: MagicMock) -> AlertManager:
    """Return an AlertManager whose underlying TelegramNotifier.send_message is mocked."""
    with patch("src.monitoring.monitoring.TelegramNotifier") as mock_cls:
        instance = mock_cls.return_value
        instance.send_message = mock_send
        instance.send_daily_report = mock_send
        instance.send_trade_alert = mock_send
        instance.send_drawdown_warning = mock_send
        instance.send_agent_disagreement = mock_send
        instance.send_system_health = mock_send
        manager = AlertManager()
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_send_drawdown_warning_calls_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_drawdown_warning() must invoke the underlying notifier."""
    send_mock = MagicMock(return_value=True)

    with patch("src.monitoring.monitoring.TelegramNotifier") as mock_cls:
        notifier_instance = mock_cls.return_value
        notifier_instance.send_drawdown_warning = send_mock
        manager = AlertManager()

    manager.send_drawdown_warning(-5.0, 95000.0)

    send_mock.assert_called_once()


def test_agent_disagreement_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_agent_disagreement() must invoke the underlying notifier."""
    send_mock = MagicMock(return_value=True)

    with patch("src.monitoring.monitoring.TelegramNotifier") as mock_cls:
        notifier_instance = mock_cls.return_value
        notifier_instance.send_agent_disagreement = send_mock
        manager = AlertManager()

    reports = {"fundamental": "BUY", "technical": "SELL", "sentiment": "HOLD"}
    manager.send_agent_disagreement("AAPL", reports)

    send_mock.assert_called_once()


def test_alert_manager_send_test_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_test() should not raise even with a mocked notifier."""
    send_mock = MagicMock(return_value=True)

    with patch("src.monitoring.monitoring.TelegramNotifier") as mock_cls:
        notifier_instance = mock_cls.return_value
        notifier_instance.send_message = send_mock
        manager = AlertManager()

    # Should not raise
    manager.send_test()
    send_mock.assert_called_once()


def test_rate_limiting_skips_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_drawdown_warning called twice within 1 hour should only dispatch once."""
    send_mock = MagicMock(return_value=True)

    with patch("src.monitoring.monitoring.TelegramNotifier") as mock_cls:
        notifier_instance = mock_cls.return_value
        notifier_instance.send_drawdown_warning = send_mock
        manager = AlertManager()

    # First call — should send
    manager.send_drawdown_warning(-5.0, 95000.0)
    # Simulate that _last_sent was just set (already done by the method)
    # Second call within 1 hour — should be suppressed
    manager.send_drawdown_warning(-5.0, 95000.0)

    assert send_mock.call_count == 1, "Second alert within rate limit window should be suppressed."


def test_smtp_stub_no_raise_when_not_configured() -> None:
    """_send_smtp() should silently skip (not raise) when SMTP is unconfigured."""
    with patch("src.monitoring.monitoring.TelegramNotifier"):
        manager = AlertManager()

    # Default config has no SMTP credentials — should not raise
    manager._send_smtp("Test Subject", "Test body")


def test_rate_limit_allows_after_1h(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_drawdown_warning should fire again after the rate limit window expires."""
    send_mock = MagicMock(return_value=True)

    with patch("src.monitoring.monitoring.TelegramNotifier") as mock_cls:
        notifier_instance = mock_cls.return_value
        notifier_instance.send_drawdown_warning = send_mock
        manager = AlertManager()

    # Pre-populate _last_sent with a timestamp from 2 hours ago
    manager._last_sent["drawdown_warning"] = datetime.now() - timedelta(hours=2)

    # Should fire because the last send was > 1 hour ago
    manager.send_drawdown_warning(-10.0, 90000.0)

    send_mock.assert_called_once()
