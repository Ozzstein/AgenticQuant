"""Strategy selection layer for the daily trading pipeline.

Selects or blends trading strategies based on macro regime, historical
performance, bandit exploration, or manual override.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.utils.config import AppConfig, get_config
from src.utils.schemas import (
    MacroRegime,
    SelectionMethod,
    Strategy,
    StrategyAllocation,
    StrategyPerformance,
)


class StrategySelector:
    """Selects or blends strategies based on regime, performance, or manual override."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialise the selector and load the strategy library.

        Args:
            config: Optional AppConfig instance. Falls back to get_config() if None.
        """
        self.config: AppConfig = config if config is not None else get_config()
        self.strategy_config = self.config.strategy
        self.library_dir: Path = Path("data/strategy_library")
        self.strategies: list[Strategy] = []
        self.load_library()

    # ------------------------------------------------------------------
    # Library loading
    # ------------------------------------------------------------------

    def load_library(self) -> list[Strategy]:
        """Read and validate all strategy JSON files from the library directory.

        Returns:
            List of validated Strategy objects loaded from disk.
        """
        if not self.library_dir.exists():
            logger.warning(
                "Strategy library directory not found: {}. Returning empty list.",
                self.library_dir,
            )
            self.strategies = []
            return []

        loaded: list[Strategy] = []
        for json_path in sorted(self.library_dir.glob("*.json")):
            try:
                with open(json_path) as fh:
                    data = json.load(fh)
                strategy = Strategy(**data)
                loaded.append(strategy)
            except FileNotFoundError:
                logger.warning("Strategy file disappeared during load: {}", json_path)
            except Exception as e:
                logger.warning("Failed to load strategy from {}: {}", json_path, e)

        self.strategies = loaded
        logger.info("Loaded {} strategies from {}", len(self.strategies), self.library_dir)
        return self.strategies

    # ------------------------------------------------------------------
    # Selection methods
    # ------------------------------------------------------------------

    def select_regime_based(
        self,
        regime: str,
        performances: dict[str, StrategyPerformance],
    ) -> StrategyAllocation:
        """Select the best strategy for the current macro regime.

        Filters strategies whose ``regime_applicability`` includes *regime*.
        Among those, picks the one with the highest ``rolling_sharpe_60d`` when
        performance data is available; otherwise falls back to the first
        validated strategy (or simply the first strategy in the library).

        Args:
            regime: Current macro regime string (e.g. ``"risk_on"``).
            performances: Mapping of strategy name → live StrategyPerformance.

        Returns:
            StrategyAllocation with 100 % weight on the selected strategy.
        """
        try:
            macro = MacroRegime(regime)
        except (ValueError, KeyError):
            logger.warning(
                "Unknown macro regime '{}'; defaulting to NEUTRAL.", regime
            )
            macro = MacroRegime.NEUTRAL

        matching = [s for s in self.strategies if macro in s.regime_applicability]

        if not matching:
            logger.warning(
                "No strategies matched regime '{}'. Falling back to default.", regime
            )
            selected = self._fallback_strategy()
        elif performances:
            # Pick highest rolling_sharpe_60d among matching strategies.
            def _sharpe(s: Strategy) -> float:
                perf = performances.get(s.name)
                return perf.rolling_sharpe_60d if perf is not None else float("-inf")

            selected = max(matching, key=_sharpe)
        else:
            # No performance data — prefer validated, then first available.
            validated = [s for s in matching if s.validated]
            selected = validated[0] if validated else matching[0]

        logger.info(
            "Regime-based selection: regime={} → strategy={}",
            regime,
            selected.name,
        )
        return StrategyAllocation(
            allocations={selected.name: 1.0},
            active_strategy=selected.name,
            selection_method=SelectionMethod.REGIME_BASED,
            regime=macro,
            timestamp=datetime.now(),
        )

    def select_performance_weighted(
        self,
        performances: dict[str, StrategyPerformance],
    ) -> StrategyAllocation:
        """Allocate capital proportionally to each strategy's rolling Sharpe.

        Strategies with ``rolling_sharpe_60d <= 0`` or ``drift_detected=True``
        receive zero allocation.  If no strategy has valid performance data,
        all strategies are weighted equally.

        Args:
            performances: Mapping of strategy name → live StrategyPerformance.

        Returns:
            StrategyAllocation with normalised weights across strategies.
        """
        if not self.strategies:
            logger.warning("No strategies available for performance-weighted selection.")
            return StrategyAllocation(
                selection_method=SelectionMethod.PERFORMANCE_WEIGHTED,
                timestamp=datetime.now(),
            )

        sharpes: dict[str, float] = {}
        for s in self.strategies:
            perf = performances.get(s.name)
            if perf is None:
                continue
            if perf.drift_detected or perf.rolling_sharpe_60d <= 0:
                continue
            sharpes[s.name] = perf.rolling_sharpe_60d

        if not sharpes:
            # Equal weight fallback.
            logger.info(
                "No valid performance data for weighted selection; using equal weight."
            )
            weight = 1.0 / len(self.strategies)
            allocations = {s.name: weight for s in self.strategies}
            active = max(allocations, key=lambda k: allocations[k])
            return StrategyAllocation(
                allocations=allocations,
                active_strategy=active,
                selection_method=SelectionMethod.PERFORMANCE_WEIGHTED,
                timestamp=datetime.now(),
            )

        total_sharpe = sum(sharpes.values())
        allocations = {name: sharpe / total_sharpe for name, sharpe in sharpes.items()}
        active = max(allocations, key=lambda k: allocations[k])

        logger.info(
            "Performance-weighted selection: {} strategies, active={}",
            len(allocations),
            active,
        )
        return StrategyAllocation(
            allocations=allocations,
            active_strategy=active,
            selection_method=SelectionMethod.PERFORMANCE_WEIGHTED,
            timestamp=datetime.now(),
        )

    def select_bandit(
        self,
        performances: dict[str, StrategyPerformance],
    ) -> StrategyAllocation:
        """Select a strategy using Thompson sampling (Beta-distributed bandits).

        For each strategy models success as Beta(α, β) where:
        - α = max(1, win_days + 1)
        - β = max(1, loss_days + 1)

        Draws one sample per strategy; the highest sample wins all capital
        (winner-takes-all).

        Args:
            performances: Mapping of strategy name → live StrategyPerformance.

        Returns:
            StrategyAllocation with 100 % weight on the Thompson-sampled winner.
        """
        if not self.strategies:
            logger.warning("No strategies available for bandit selection.")
            return StrategyAllocation(
                selection_method=SelectionMethod.BANDIT,
                timestamp=datetime.now(),
            )

        samples: dict[str, float] = {}
        for s in self.strategies:
            perf = performances.get(s.name)
            if perf is not None and perf.daily_pnl:
                wins = sum(1 for pnl in perf.daily_pnl if pnl > 0)
                losses = len(perf.daily_pnl) - wins
            else:
                wins = 0
                losses = 0

            alpha = max(1, wins + 1)
            beta = max(1, losses + 1)
            samples[s.name] = float(np.random.beta(alpha, beta))

        winner = max(samples, key=lambda k: samples[k])
        logger.info(
            "Bandit (Thompson) selection: winner={} (θ={:.4f})",
            winner,
            samples[winner],
        )
        return StrategyAllocation(
            allocations={winner: 1.0},
            active_strategy=winner,
            selection_method=SelectionMethod.BANDIT,
            timestamp=datetime.now(),
        )

    def select_manual(self, strategy_name: str) -> StrategyAllocation:
        """Return an allocation forcing 100 % weight on the named strategy.

        If *strategy_name* is not found in the library, logs a warning and
        falls back to the first available strategy.

        Args:
            strategy_name: Exact name of the strategy to force.

        Returns:
            StrategyAllocation with 100 % weight on the resolved strategy.
        """
        match = next((s for s in self.strategies if s.name == strategy_name), None)
        if match is None:
            logger.warning(
                "Strategy '{}' not found in library. Falling back to first available.",
                strategy_name,
            )
            fallback = self._fallback_strategy()
            name = fallback.name
        else:
            name = match.name

        logger.info("Manual selection: strategy={}", name)
        return StrategyAllocation(
            allocations={name: 1.0},
            active_strategy=name,
            selection_method=SelectionMethod.MANUAL,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def select(
        self,
        regime: str,
        performances: dict[str, StrategyPerformance] | None = None,
        method: str | None = None,
    ) -> StrategyAllocation:
        """Select or blend strategies based on the configured selection method.

        This is the primary entry point used by the daily pipeline.

        Args:
            regime: Current macro regime string (e.g. ``"risk_on"``).
            performances: Optional mapping of strategy name → StrategyPerformance.
                Defaults to an empty dict when *None*.
            method: Selection method override.  When *None*, falls back to
                ``strategy_config.selection_method``.

        Returns:
            StrategyAllocation describing capital distribution.
        """
        if performances is None:
            performances = {}

        # Manual override takes highest precedence.
        if self.strategy_config.forced_strategy:
            return self.select_manual(self.strategy_config.forced_strategy)

        resolved_method = method if method is not None else self.strategy_config.selection_method

        match resolved_method:
            case "regime_based":
                return self.select_regime_based(regime, performances)
            case "performance_weighted":
                return self.select_performance_weighted(performances)
            case "bandit":
                return self.select_bandit(performances)
            case "manual":
                name = (
                    self.strategy_config.forced_strategy
                    or self.strategy_config.default_strategy
                )
                return self.select_manual(name)
            case _:
                logger.warning(
                    "Unknown selection method '{}'; defaulting to regime_based.",
                    resolved_method,
                )
                return self.select_regime_based(regime, performances)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_strategy(self) -> Strategy:
        """Return the configured default strategy or the first available one.

        Returns:
            A Strategy object to use as fallback.
        """
        if self.strategies:
            # Try configured default name first.
            default_name = self.strategy_config.default_strategy
            for s in self.strategies:
                if s.name == default_name:
                    return s
            # Otherwise return first loaded strategy.
            return self.strategies[0]

        # No strategies loaded at all — return a minimal built-in sentinel.
        logger.error(
            "No strategies available in library. Returning bare default strategy."
        )
        return Strategy(
            name="default",
            description="Auto-generated default strategy (no library found)",
            regime_applicability=list(MacroRegime),
        )


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.schemas import StrategyPerformance

    selector = StrategySelector()
    logger.info("Loaded {} strategies", len(selector.strategies))

    alloc = selector.select("risk_on")
    logger.info("Selected: {} via {}", alloc.active_strategy, alloc.selection_method)

    # Test with fake performance data.
    perf = {
        s.name: StrategyPerformance(
            strategy_name=s.name,
            rolling_sharpe_60d=round(random.uniform(0.2, 1.5), 3),
            daily_pnl=[random.gauss(50, 200) for _ in range(30)],
        )
        for s in selector.strategies
    }
    alloc_perf = selector.select("neutral", performances=perf, method="performance_weighted")
    logger.info("Perf-weighted: {} ({})", alloc_perf.active_strategy, alloc_perf.allocations)

    alloc_bandit = selector.select("neutral", performances=perf, method="bandit")
    logger.info("Bandit: {}", alloc_bandit.active_strategy)

    logger.info("OK")
