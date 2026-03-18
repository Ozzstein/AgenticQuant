"""Telegram notification bot for trading alerts and daily reports."""

from __future__ import annotations

from loguru import logger


class TelegramNotifier:
    """Sends notifications via Telegram Bot API.

    Gracefully stubs all operations when token is not configured.

    Attributes:
        token: Telegram bot token.
        chat_id: Target chat ID.
        _available: True when python-telegram-bot is installed AND token is set.
    """

    def __init__(self, token: str = "", chat_id: str = "", config=None) -> None:
        from src.utils.config import get_config

        cfg = config or get_config()
        self.token = token or cfg.monitoring.telegram_token
        self.chat_id = chat_id or cfg.monitoring.telegram_chat_id
        self._available = False

        if not self.token:
            logger.info("TelegramNotifier: no token configured — stub mode.")
            return

        try:
            import telegram  # noqa: F401

            self._available = True
            logger.info("TelegramNotifier: telegram package available.")
        except ImportError:
            logger.warning("python-telegram-bot not installed — stub mode.")

    def send_message(self, text: str) -> bool:
        """Send a text message.

        Args:
            text: Message text (max 4096 chars).

        Returns:
            True if sent, False if stubbed/failed.
        """
        if not self._available:
            logger.debug("TelegramNotifier stub: would send: {}", text[:100])
            return False
        try:
            import asyncio

            import telegram

            bot = telegram.Bot(token=self.token)
            asyncio.run(bot.send_message(chat_id=self.chat_id, text=text[:4096]))
            logger.info("Telegram message sent.")
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: {}", exc)
            return False

    def send_trade_alert(self, ticker: str, action: str, price: float, quantity: float) -> bool:
        """Send a trade execution alert.

        Args:
            ticker: Stock ticker.
            action: BUY or SELL.
            price: Fill price.
            quantity: Shares traded.

        Returns:
            True if sent, False if stubbed/failed.
        """
        msg = f"🔔 TRADE: {action} {quantity:.0f} {ticker} @ ${price:.2f}"
        return self.send_message(msg)

    def send_daily_report(self, portfolio_summary: str) -> bool:
        """Send the daily portfolio report.

        Args:
            portfolio_summary: Formatted portfolio summary string.

        Returns:
            True if sent, False if stubbed/failed.
        """
        msg = f"📊 Daily Report\n{portfolio_summary}"
        return self.send_message(msg)

    def send_alert(self, alert_text: str) -> bool:
        """Send an alert message.

        Args:
            alert_text: Alert description.

        Returns:
            True if sent, False if stubbed/failed.
        """
        msg = f"⚠️ ALERT: {alert_text}"
        return self.send_message(msg)
