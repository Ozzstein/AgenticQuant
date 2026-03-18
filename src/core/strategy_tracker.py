"""Strategy performance tracker for per-strategy live P&L and drift detection."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.config import AppConfig, get_config
from src.utils.schemas import StrategyPerformance


class StrategyTracker:
    """Tracks per-strategy live performance and detects drift from backtest.

    Maintains rolling Sharpe ratios, drawdown metrics, and win rates for each
    strategy. Alerts when live performance diverges significantly from the
    backtest expectation.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the tracker, loading any previously persisted performance data.

        Args:
            config: Application configuration. Falls back to get_config() if None.
        """
        self.config: AppConfig = config if config is not None else get_config()
        self.strategy_config = self.config.strategy
        self.perf_dir = Path("data/strategy_performance")
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.performances: dict[str, StrategyPerformance] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all persisted strategy performance files from disk.

        Reads every *.json file in self.perf_dir and hydrates StrategyPerformance
        objects into self.performances. Missing or empty directories are handled
        gracefully.
        """
        try:
            for json_file in self.perf_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    strategy_name = json_file.stem
                    perf = StrategyPerformance(**data)
                    self.performances[strategy_name] = perf
                    logger.debug("Loaded performance data for strategy '{}'", strategy_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to load performance file {}: {}", json_file, exc
                    )
        except FileNotFoundError:
            logger.debug("Performance directory not found; starting fresh.")

    def save(self) -> None:
        """Persist all strategy performance objects to disk as JSON.

        Each strategy is stored in a separate file named ``{strategy_name}.json``
        under self.perf_dir. Datetime fields are serialised as ISO-8601 strings
        via Pydantic's ``model_dump(mode='json')``.
        """
        for name, perf in self.performances.items():
            file_path = self.perf_dir / f"{name}.json"
            try:
                file_path.write_text(json.dumps(perf.model_dump(mode="json"), indent=2))
                logger.debug("Saved performance data for strategy '{}'", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to save performance for '{}': {}", name, exc)

    # ------------------------------------------------------------------
    # P&L Attribution
    # ------------------------------------------------------------------

    def attribute_pnl(self, strategy_name: str, daily_pnl: float) -> None:
        """Record a single day's P&L for a strategy and refresh all metrics.

        Args:
            strategy_name: Unique strategy identifier.
            daily_pnl: Dollar (or %) P&L for the day to attribute.
        """
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        perf = self.performances[strategy_name]
        perf.daily_pnl.append(daily_pnl)
        perf.total_pnl += daily_pnl
        perf.last_updated = datetime.now()

        self.update_metrics(strategy_name)
        self.save()

    # ------------------------------------------------------------------
    # Metric Computation
    # ------------------------------------------------------------------

    def update_metrics(self, strategy_name: str) -> None:
        """Recompute all rolling metrics for a strategy.

        Calculates rolling Sharpe ratios (30/60/90-day), maximum and current
        drawdown, and win rate, then checks for strategy drift.

        Args:
            strategy_name: Unique strategy identifier.
        """
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        perf = self.performances[strategy_name]
        pnl = perf.daily_pnl

        # Rolling Sharpe ratios
        perf.rolling_sharpe_30d = self._rolling_sharpe(pnl, 30)
        perf.rolling_sharpe_60d = self._rolling_sharpe(pnl, 60)
        perf.rolling_sharpe_90d = self._rolling_sharpe(pnl, 90)

        # Drawdown metrics
        if len(pnl) > 0:
            cumulative_pnl = np.cumsum(pnl)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            perf.max_drawdown = float(np.min(drawdown))
            perf.current_drawdown = float(drawdown[-1])
        else:
            perf.max_drawdown = 0.0
            perf.current_drawdown = 0.0

        # Win rate
        total_days = len(pnl)
        if total_days > 0:
            wins = sum(1 for d in pnl if d > 0)
            perf.win_rate = wins / total_days
        else:
            perf.win_rate = 0.0

        # Drift detection
        self.check_drift(strategy_name)

    def _rolling_sharpe(self, pnl_list: list[float], window: int) -> float:
        """Compute annualised Sharpe ratio over the last *window* daily P&L values.

        Args:
            pnl_list: Full history of daily P&L values.
            window: Number of trailing days to include.

        Returns:
            Annualised Sharpe ratio, or 0.0 when there is insufficient data or
            the standard deviation is zero.
        """
        if window == 0 or len(pnl_list) < window:
            return 0.0

        recent = pnl_list[-window:]
        mean = float(np.mean(recent))
        std = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0

        if std == 0.0:
            return 0.0

        return mean * math.sqrt(252) / std

    # ------------------------------------------------------------------
    # Drift Detection
    # ------------------------------------------------------------------

    def check_drift(self, strategy_name: str) -> bool:
        """Check whether a strategy's live Sharpe has drifted below its backtest value.

        Drift is flagged when the rolling 60-day live Sharpe falls below
        ``drift_threshold * backtest_sharpe``.  If either value is unavailable
        (zero), the check is skipped and False is returned.

        Args:
            strategy_name: Unique strategy identifier.

        Returns:
            True if drift is detected, False otherwise.
        """
        if strategy_name not in self.performances:
            return False

        perf = self.performances[strategy_name]
        live_sharpe = perf.rolling_sharpe_60d
        backtest_sharpe = perf.backtest_sharpe

        if live_sharpe == 0.0 or backtest_sharpe == 0.0:
            return False

        drift_threshold = self.strategy_config.drift_threshold
        detected = live_sharpe < drift_threshold * backtest_sharpe
        perf.drift_detected = detected

        if detected:
            logger.warning(
                "Strategy {} drift detected: live={:.2f} vs backtest={:.2f}",
                strategy_name,
                live_sharpe,
                backtest_sharpe,
            )

        return detected

    # ------------------------------------------------------------------
    # Backtest Anchor
    # ------------------------------------------------------------------

    def set_backtest_sharpe(self, strategy_name: str, backtest_sharpe: float) -> None:
        """Store the historical backtest Sharpe ratio for a strategy.

        This acts as the reference against which live drift is measured.

        Args:
            strategy_name: Unique strategy identifier.
            backtest_sharpe: Annualised out-of-sample Sharpe from walk-forward backtest.
        """
        if strategy_name not in self.performances:
            self.performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        self.performances[strategy_name].backtest_sharpe = backtest_sharpe

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_performances(self) -> dict[str, StrategyPerformance]:
        """Return a copy of all tracked strategy performance objects.

        Returns:
            Shallow copy mapping strategy names to their StrategyPerformance.
        """
        return dict(self.performances)

    def report(self) -> dict:
        """Generate a summary report suitable for CLI output or a dashboard.

        Returns:
            Dict with ``total_strategies`` count and per-strategy key metrics.
        """
        return {
            "total_strategies": len(self.performances),
            "strategies": {
                name: {
                    "rolling_sharpe_60d": perf.rolling_sharpe_60d,
                    "max_drawdown": perf.max_drawdown,
                    "win_rate": perf.win_rate,
                    "total_pnl": perf.total_pnl,
                    "drift_detected": perf.drift_detected,
                }
                for name, perf in self.performances.items()
            },
        }
