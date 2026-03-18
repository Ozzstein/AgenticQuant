"""Portfolio health monitoring and reporting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger


class PortfolioMonitor:
    """Monitors portfolio health and generates reports.

    Health score is 0-100 based on:
    - Drawdown component (0-40): 40 if drawdown < 5%, scaled down to 0 at -15%
    - Concentration component (0-30): 30 if max position < 10%, scaled
    - Cash component (0-20): 20 if cash > 5% of NAV
    - Trade activity component (0-10): 10 if recent trades exist

    Attributes:
        portfolio_path: Path to paper_portfolio.json
        config: AppConfig instance
    """

    def __init__(self, portfolio_path: Path | None = None, config=None) -> None:
        from src.utils.config import get_config

        self.config = config or get_config()
        self.portfolio_path = portfolio_path or Path(self.config.output_dir) / "paper_portfolio.json"

    def load_portfolio(self) -> dict:
        """Load portfolio JSON from disk.

        Returns:
            Portfolio dict or empty dict if file missing.
        """
        if not self.portfolio_path.exists():
            logger.warning("Portfolio file not found at {}", self.portfolio_path)
            return {}
        try:
            with self.portfolio_path.open() as fh:
                data = json.load(fh)
            logger.info("Portfolio loaded from {}", self.portfolio_path)
            return data
        except Exception as exc:
            logger.error("Failed to load portfolio from {}: {}", self.portfolio_path, exc)
            return {}

    def compute_health_score(self, portfolio: dict) -> int:
        """Compute portfolio health score 0-100.

        Args:
            portfolio: Portfolio dict with positions, cash, nav keys.

        Returns:
            Integer health score 0-100.
        """
        score = 0

        nav = portfolio.get("nav", 0.0)
        cash = portfolio.get("cash", 0.0)
        positions = portfolio.get("positions", {})

        # ------------------------------------------------------------------
        # 1. Drawdown component (0-40)
        # ------------------------------------------------------------------
        # Derive a rough drawdown from the portfolio's high watermark if available.
        # If no peak_nav is stored, use the current NAV as peak (no drawdown).
        peak_nav = portfolio.get("peak_nav", nav)
        if peak_nav and peak_nav > 0 and nav > 0:
            drawdown_pct = (nav - peak_nav) / peak_nav * 100  # negative if drawdown
        else:
            drawdown_pct = 0.0

        # 40 points if drawdown >= -5%, linearly scales to 0 at -15%
        drawdown_floor = -15.0
        drawdown_good = -5.0
        if drawdown_pct >= drawdown_good:
            drawdown_component = 40
        elif drawdown_pct <= drawdown_floor:
            drawdown_component = 0
        else:
            # Linear interpolation between drawdown_floor (0 pts) and drawdown_good (40 pts)
            ratio = (drawdown_pct - drawdown_floor) / (drawdown_good - drawdown_floor)
            drawdown_component = int(40 * ratio)

        score += drawdown_component

        # ------------------------------------------------------------------
        # 2. Concentration component (0-30)
        # ------------------------------------------------------------------
        max_weight_pct = 0.0
        if positions and nav > 0:
            for pos in positions.values():
                if isinstance(pos, dict):
                    weight = pos.get("weight_pct", 0.0)
                    if weight > max_weight_pct:
                        max_weight_pct = weight

        # 30 points if max position < 10%, linearly scales to 0 at 30%
        concentration_good = 10.0
        concentration_bad = 30.0
        if max_weight_pct <= concentration_good:
            concentration_component = 30
        elif max_weight_pct >= concentration_bad:
            concentration_component = 0
        else:
            ratio = 1.0 - (max_weight_pct - concentration_good) / (
                concentration_bad - concentration_good
            )
            concentration_component = int(30 * ratio)

        score += concentration_component

        # ------------------------------------------------------------------
        # 3. Cash component (0-20)
        # ------------------------------------------------------------------
        if nav > 0:
            cash_pct = cash / nav * 100
        else:
            cash_pct = 100.0

        min_cash_pct = 5.0
        cash_component = 20 if cash_pct >= min_cash_pct else 0
        score += cash_component

        # ------------------------------------------------------------------
        # 4. Trade activity component (0-10)
        # ------------------------------------------------------------------
        # Reward the portfolio for having some recent trades logged
        trade_log = portfolio.get("trade_log", [])
        trade_component = 10 if trade_log else 0
        score += trade_component

        logger.debug(
            "Health score: drawdown={} concentration={} cash={} trades={} total={}",
            drawdown_component,
            concentration_component,
            cash_component,
            trade_component,
            score,
        )
        return max(0, min(100, score))

    def generate_report(self) -> str:
        """Generate a rich-formatted health report string.

        Returns:
            Multi-line string with portfolio summary and health score.
        """
        portfolio = self.load_portfolio()
        if not portfolio:
            return (
                "=== Portfolio Health Report ===\n"
                "No portfolio data found.\n"
                f"Expected path: {self.portfolio_path}\n"
                "Health Score: N/A\n"
            )

        health = self.compute_health_score(portfolio)
        nav = portfolio.get("nav", 0.0)
        cash = portfolio.get("cash", 0.0)
        positions = portfolio.get("positions", {})
        timestamp = portfolio.get("timestamp", datetime.now().isoformat())

        # Health label
        if health >= 80:
            health_label = "EXCELLENT"
        elif health >= 60:
            health_label = "GOOD"
        elif health >= 40:
            health_label = "FAIR"
        else:
            health_label = "POOR"

        lines = [
            "=== Portfolio Health Report ===",
            f"Timestamp : {timestamp}",
            f"NAV       : ${nav:,.2f}",
            f"Cash      : ${cash:,.2f}  ({cash / nav * 100:.1f}% of NAV)" if nav > 0 else f"Cash : ${cash:,.2f}",
            f"Positions : {len(positions)}",
            "",
            "--- Positions ---",
        ]

        for ticker, pos in positions.items():
            if isinstance(pos, dict):
                qty = pos.get("quantity", 0)
                price = pos.get("current_price", 0.0)
                weight = pos.get("weight_pct", 0.0)
                pnl = pos.get("unrealized_pnl", 0.0)
                lines.append(
                    f"  {ticker:<8} qty={qty:<8.0f} price=${price:<10.2f} "
                    f"weight={weight:.1f}%  PnL=${pnl:,.2f}"
                )

        lines += [
            "",
            f"Health Score: {health}/100  [{health_label}]",
            "=== End Report ===",
        ]

        report = "\n".join(lines)
        logger.info("Portfolio health report generated (score={})", health)
        return report

    def check_alerts(self) -> list[str]:
        """Check for alert conditions.

        Returns:
            List of alert strings. Empty list if healthy.
        """
        portfolio = self.load_portfolio()
        if not portfolio:
            return ["Portfolio data unavailable — cannot check alerts."]

        alerts: list[str] = []
        nav = portfolio.get("nav", 0.0)
        cash = portfolio.get("cash", 0.0)
        positions = portfolio.get("positions", {})

        # Drawdown alert
        peak_nav = portfolio.get("peak_nav", nav)
        if peak_nav and peak_nav > 0 and nav > 0:
            drawdown_pct = (nav - peak_nav) / peak_nav * 100
            alert_threshold = self.config.monitoring.drawdown_alert_pct
            if drawdown_pct < -alert_threshold:
                alerts.append(f"Drawdown exceeds {alert_threshold:.0f}%: current={drawdown_pct:.1f}%")
            if drawdown_pct < -self.config.risk.max_drawdown_halt:
                alerts.append(
                    f"Drawdown circuit breaker triggered: {drawdown_pct:.1f}% "
                    f"< -{self.config.risk.max_drawdown_halt:.0f}%"
                )

        # Concentration alert
        if positions and nav > 0:
            for ticker, pos in positions.items():
                if isinstance(pos, dict):
                    weight = pos.get("weight_pct", 0.0)
                    if weight > 20.0:
                        alerts.append(
                            f"Position concentration > 20%: {ticker} = {weight:.1f}%"
                        )
                    if weight > self.config.risk.max_position_weight:
                        alerts.append(
                            f"Position exceeds max weight limit: "
                            f"{ticker} = {weight:.1f}% > {self.config.risk.max_position_weight:.0f}%"
                        )

        # Cash buffer alert
        if nav > 0:
            cash_pct = cash / nav * 100
            if cash_pct < self.config.risk.min_cash_pct:
                alerts.append(
                    f"Cash buffer below minimum: {cash_pct:.1f}% < {self.config.risk.min_cash_pct:.0f}%"
                )

        if alerts:
            logger.warning("Portfolio alerts: {}", alerts)
        else:
            logger.info("Portfolio health check: no alerts.")

        return alerts
