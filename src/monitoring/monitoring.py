"""AlertManager: orchestrates Telegram and SMTP notifications with rate limiting."""

from __future__ import annotations

import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

from loguru import logger

from src.monitoring.telegram_bot import TelegramNotifier


class AlertManager:
    """Orchestrates Telegram and SMTP notifications with per-alert-type rate limiting.

    Attributes:
        _notifier: Underlying TelegramNotifier instance.
        _last_sent: Maps alert_type string → last datetime it was dispatched.
        _rate_limit: Minimum interval between repeated alerts of the same type.
    """

    _RATE_LIMIT = timedelta(hours=1)

    def __init__(self) -> None:
        self._notifier = TelegramNotifier()
        self._last_sent: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_rate_limited(self, alert_type: str) -> bool:
        """Return True when the same alert type was sent less than 1 hour ago.

        Args:
            alert_type: Identifier string for the alert category.

        Returns:
            True if the alert should be suppressed, False otherwise.
        """
        last = self._last_sent.get(alert_type)
        if last is None:
            return False
        return datetime.now() - last < self._RATE_LIMIT

    def _mark_sent(self, alert_type: str) -> None:
        """Record the current time as the last send time for the given alert type.

        Args:
            alert_type: Identifier string for the alert category.
        """
        self._last_sent[alert_type] = datetime.now()

    def _send_smtp(self, subject: str, body: str) -> None:
        """Send an email via SMTP if fully configured, otherwise log and skip.

        Args:
            subject: Email subject line.
            body: Plain-text email body.
        """
        from src.utils.config import get_config

        cfg = get_config().monitoring
        if not all([cfg.smtp_host, cfg.smtp_user, cfg.smtp_password, cfg.smtp_to]):
            logger.debug("AlertManager SMTP: not configured — skipping email.")
            return

        try:
            msg = MIMEText(body, "plain")
            msg["Subject"] = subject
            msg["From"] = cfg.smtp_user  # type: ignore[assignment]
            msg["To"] = cfg.smtp_to  # type: ignore[assignment]

            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:  # type: ignore[arg-type]
                server.starttls()
                server.login(cfg.smtp_user, cfg.smtp_password)  # type: ignore[arg-type]
                server.sendmail(cfg.smtp_user, cfg.smtp_to, msg.as_string())  # type: ignore[arg-type]

            logger.info("AlertManager SMTP: email sent to {}", cfg.smtp_to)
        except Exception as exc:
            logger.warning("AlertManager SMTP send failed: {}", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_daily_summary(self, portfolio_dict: dict, metrics_dict: dict) -> None:
        """Send a daily portfolio summary via Telegram and SMTP.

        Args:
            portfolio_dict: Portfolio state mapping (e.g. positions, cash, NAV).
            metrics_dict: Performance metrics (e.g. Sharpe, drawdown, P&L).
        """
        alert_type = "daily_summary"
        if self._is_rate_limited(alert_type):
            logger.debug("AlertManager: '{}' rate-limited — skipping.", alert_type)
            return

        nav = portfolio_dict.get("nav", portfolio_dict.get("total_value", 0.0))
        pnl = metrics_dict.get("total_pnl", metrics_dict.get("pnl", 0.0))
        sharpe = metrics_dict.get("sharpe_ratio", metrics_dict.get("sharpe", "N/A"))
        body = (
            f"Daily Portfolio Summary\n"
            f"NAV: ${nav:,.2f}\n"
            f"P&L: ${pnl:,.2f}\n"
            f"Sharpe: {sharpe}\n"
        )
        self._notifier.send_daily_report(body)
        self._send_smtp("QuantAgentLab — Daily Summary", body)
        self._mark_sent(alert_type)

    def send_trade_alert(
        self, ticker: str, side: str, qty: int, price: float, reasoning: str
    ) -> None:
        """Send a trade execution alert via Telegram and SMTP.

        Args:
            ticker: Stock/crypto ticker symbol.
            side: Order side — "BUY" or "SELL".
            qty: Number of shares/units traded.
            price: Fill price per share/unit.
            reasoning: Short explanation of the trade rationale.
        """
        alert_type = "trade_alert"
        if self._is_rate_limited(alert_type):
            logger.debug("AlertManager: '{}' rate-limited — skipping.", alert_type)
            return

        self._notifier.send_trade_alert(ticker=ticker, action=side, price=price, quantity=qty)
        body = (
            f"Trade Alert: {side} {qty} {ticker} @ ${price:.2f}\n"
            f"Reasoning: {reasoning}"
        )
        self._send_smtp(f"QuantAgentLab — Trade: {side} {ticker}", body)
        self._mark_sent(alert_type)

    def send_drawdown_warning(self, drawdown_pct: float, nav: float) -> None:
        """Send a drawdown warning. Fires at -5%, -10%, -15% thresholds.

        Args:
            drawdown_pct: Current drawdown as a negative percentage (e.g. -10.5).
            nav: Current net asset value of the portfolio.
        """
        alert_type = "drawdown_warning"
        if self._is_rate_limited(alert_type):
            logger.debug("AlertManager: '{}' rate-limited — skipping.", alert_type)
            return

        self._notifier.send_drawdown_warning(drawdown_pct=drawdown_pct, nav=nav)
        body = f"Drawdown Warning\nDrawdown: {drawdown_pct:.2f}%\nNAV: ${nav:,.2f}"
        self._send_smtp("QuantAgentLab — Drawdown Warning", body)
        self._mark_sent(alert_type)

    def send_agent_disagreement(self, ticker: str, reports: dict[str, str]) -> None:
        """Send alert when max spread between agent decisions exceeds 60 points.

        Args:
            ticker: The ticker symbol being analysed.
            reports: Mapping of agent name → report/decision string.
        """
        alert_type = "agent_disagreement"
        if self._is_rate_limited(alert_type):
            logger.debug("AlertManager: '{}' rate-limited — skipping.", alert_type)
            return

        self._notifier.send_agent_disagreement(ticker=ticker, reports=reports)
        summary = "\n".join(f"  {a}: {r}" for a, r in reports.items())
        body = f"Agent Disagreement on {ticker}:\n{summary}"
        self._send_smtp(f"QuantAgentLab — Agent Disagreement: {ticker}", body)
        self._mark_sent(alert_type)

    def send_system_health(self, step: str, error: str) -> None:
        """Send pipeline step failure notification via Telegram and SMTP.

        Args:
            step: Name of the pipeline step that failed.
            error: Error message or description of the failure.
        """
        alert_type = "system_health"
        if self._is_rate_limited(alert_type):
            logger.debug("AlertManager: '{}' rate-limited — skipping.", alert_type)
            return

        self._notifier.send_system_health(step=step, error=error)
        body = f"Pipeline Failure\nStep: {step}\nError: {error}"
        self._send_smtp("QuantAgentLab — Pipeline Failure", body)
        self._mark_sent(alert_type)

    def send_test(self) -> None:
        """Send a test message to both Telegram and SMTP channels (bypasses rate limiting).

        Useful for verifying that notification channels are correctly configured.
        """
        msg = "QuantAgentLab test message"
        logger.info("AlertManager: sending test notification.")
        self._notifier.send_message(msg)
        self._send_smtp("QuantAgentLab — Test", msg)
