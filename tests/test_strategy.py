"""Tests for the Strategy layer: schemas, selector, tracker, and pipeline integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.config import AppConfig, get_config
from src.utils.schemas import (
    MacroRegime,
    SelectionMethod,
    SizingMethod,
    Strategy,
    StrategyAllocation,
    StrategyPerformance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(name: str, regimes: list[str] | None = None) -> dict:
    """Return a minimal valid strategy dict for use in tests."""
    return {
        "name": name,
        "description": f"Test {name}",
        "factor_set": ["Alpha158"],
        "model": "LightGBM",
        "universe": "SP500",
        "entry_rules": {"rank_threshold": 20},
        "exit_rules": {"holding_period_days": 5},
        "sizing_method": "equal_weight",
        "rebalance_frequency": "weekly",
        "regime_applicability": regimes or ["risk_on", "neutral", "risk_off", "crisis"],
        "risk_overrides": {},
        "backtest_sharpe": 0.75,
        "backtest_max_drawdown": -0.15,
        "validated": True,
        "source": "test",
    }


def _write_strategy(tmp_path: Path, name: str, regimes: list[str] | None = None) -> Path:
    """Write a strategy JSON file to tmp_path and return the path."""
    data = _make_strategy(name, regimes)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(data))
    return p


def _make_selector_with_dir(tmp_path: Path):
    """Return a StrategySelector whose library_dir points to tmp_path."""
    from src.core.strategy_selector import StrategySelector

    selector = StrategySelector.__new__(StrategySelector)
    selector.config = get_config()
    selector.strategy_config = selector.config.strategy
    selector.library_dir = tmp_path
    selector.strategies = []
    selector.load_library()
    return selector


def _make_tracker_with_dir(tmp_path: Path):
    """Return a StrategyTracker whose perf_dir points to tmp_path."""
    from src.core.strategy_tracker import StrategyTracker

    tracker = StrategyTracker.__new__(StrategyTracker)
    tracker.config = get_config()
    tracker.strategy_config = tracker.config.strategy
    tracker.perf_dir = tmp_path
    tracker.performances = {}
    tracker._load()
    return tracker


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestStrategySchemas:
    def test_strategy_default_fields(self):
        """Strategy() with minimal input should set sensible defaults."""
        s = Strategy(name="test")
        assert s.name == "test"
        assert s.factor_set == ["Alpha158"]
        assert s.model == "LightGBM"
        assert s.universe == "SP500"
        assert s.sizing_method == SizingMethod.EQUAL_WEIGHT
        assert s.validated is False
        assert s.source == "builtin"

    def test_strategy_regime_applicability_default(self):
        """Default regime_applicability should contain all 4 MacroRegime values."""
        s = Strategy(name="test")
        all_regimes = list(MacroRegime)
        assert len(all_regimes) == 4
        assert set(s.regime_applicability) == set(all_regimes)

    def test_strategy_allocation_schema(self):
        """StrategyAllocation should store allocations and active_strategy correctly."""
        alloc = StrategyAllocation(
            allocations={"momentum_topk": 0.6, "mean_reversion": 0.4},
            active_strategy="momentum_topk",
            selection_method=SelectionMethod.PERFORMANCE_WEIGHTED,
            regime=MacroRegime.RISK_ON,
        )
        assert alloc.allocations["momentum_topk"] == pytest.approx(0.6)
        assert alloc.active_strategy == "momentum_topk"
        assert alloc.selection_method == SelectionMethod.PERFORMANCE_WEIGHTED
        assert alloc.regime == MacroRegime.RISK_ON

    def test_strategy_performance_schema(self):
        """StrategyPerformance should create with defaults and accept updates."""
        perf = StrategyPerformance(strategy_name="test_strat")
        assert perf.strategy_name == "test_strat"
        assert perf.rolling_sharpe_30d == 0.0
        assert perf.rolling_sharpe_60d == 0.0
        assert perf.rolling_sharpe_90d == 0.0
        assert perf.win_rate == 0.0
        assert perf.drift_detected is False
        assert perf.daily_pnl == []

    def test_sizing_method_enum(self):
        """SizingMethod should have exactly 5 variants."""
        values = {m.value for m in SizingMethod}
        assert values == {
            "equal_weight",
            "inverse_volatility",
            "kelly",
            "risk_parity",
            "signal_proportional",
        }
        assert len(values) == 5

    def test_macro_regime_enum(self):
        """MacroRegime should have exactly 4 variants."""
        values = {m.value for m in MacroRegime}
        assert values == {"risk_on", "neutral", "risk_off", "crisis"}
        assert len(values) == 4


# ---------------------------------------------------------------------------
# 2. Strategy library load
# ---------------------------------------------------------------------------


_LIBRARY_DIR = Path("data/strategy_library")
_BASELINE_NAMES = {
    "momentum_topk",
    "value_quality",
    "mean_reversion",
    "crypto_trend",
    "defensive",
    "agent_consensus",
}


class TestStrategyLibrary:
    def test_load_all_6_baselines(self):
        """StrategySelector should load at least the 6 baseline strategies."""
        from src.core.strategy_selector import StrategySelector

        selector = StrategySelector()
        # Library may contain extra strategies from RD-Agent; ensure baselines present.
        loaded_names = {s.name for s in selector.strategies}
        assert _BASELINE_NAMES.issubset(loaded_names), (
            f"Missing baselines: {_BASELINE_NAMES - loaded_names}"
        )

    def test_baseline_strategy_names(self):
        """All 6 expected baseline strategy names should be present in the library."""
        from src.core.strategy_selector import StrategySelector

        selector = StrategySelector()
        loaded_names = {s.name for s in selector.strategies}
        for name in _BASELINE_NAMES:
            assert name in loaded_names, f"Expected strategy '{name}' not found"

    def test_strategy_library_json_validates(self):
        """Each strategy JSON in data/strategy_library/ should parse to a valid Strategy."""
        assert _LIBRARY_DIR.exists(), f"Strategy library dir not found: {_LIBRARY_DIR}"
        json_files = list(_LIBRARY_DIR.glob("*.json"))
        assert json_files, "No JSON files found in strategy library"

        for json_path in json_files:
            with open(json_path) as fh:
                data = json.load(fh)
            strategy = Strategy(**data)
            # name field must match filename stem
            assert strategy.name == json_path.stem, (
                f"Strategy name '{strategy.name}' does not match filename '{json_path.stem}'"
            )


# ---------------------------------------------------------------------------
# 3. StrategySelector tests
# ---------------------------------------------------------------------------


class TestStrategySelector:
    def test_load_library_from_dir(self, tmp_path):
        """Selector should load strategies from a custom directory."""
        _write_strategy(tmp_path, "strat_a")
        _write_strategy(tmp_path, "strat_b")

        selector = _make_selector_with_dir(tmp_path)
        assert len(selector.strategies) == 2
        names = {s.name for s in selector.strategies}
        assert names == {"strat_a", "strat_b"}

    def test_load_library_missing_dir(self, tmp_path):
        """Selector should return empty list and not raise on missing dir."""
        missing_dir = tmp_path / "does_not_exist"
        selector = _make_selector_with_dir(missing_dir)
        assert selector.strategies == []

    def test_select_regime_based_filters_correctly(self, tmp_path):
        """Regime-based selection should return only strategies matching the regime."""
        _write_strategy(tmp_path, "risk_on_only", regimes=["risk_on"])
        _write_strategy(tmp_path, "all_regimes")

        selector = _make_selector_with_dir(tmp_path)
        alloc = selector.select_regime_based("risk_on", {})
        # Both strategies match risk_on; the active_strategy must be one of them.
        assert alloc.active_strategy in {"risk_on_only", "all_regimes"}

        # Select for crisis: only all_regimes matches.
        alloc_crisis = selector.select_regime_based("crisis", {})
        assert alloc_crisis.active_strategy == "all_regimes"

    def test_select_regime_based_uses_best_sharpe(self, tmp_path):
        """When performances are provided, the higher-Sharpe strategy is selected."""
        _write_strategy(tmp_path, "low_sharpe_strat", regimes=["risk_on"])
        _write_strategy(tmp_path, "high_sharpe_strat", regimes=["risk_on"])

        selector = _make_selector_with_dir(tmp_path)
        performances = {
            "low_sharpe_strat": StrategyPerformance(
                strategy_name="low_sharpe_strat", rolling_sharpe_60d=0.3
            ),
            "high_sharpe_strat": StrategyPerformance(
                strategy_name="high_sharpe_strat", rolling_sharpe_60d=1.2
            ),
        }
        alloc = selector.select_regime_based("risk_on", performances)
        assert alloc.active_strategy == "high_sharpe_strat"

    def test_select_performance_weighted_normalizes(self, tmp_path):
        """Performance-weighted allocations should sum to approximately 1.0."""
        _write_strategy(tmp_path, "strat_a")
        _write_strategy(tmp_path, "strat_b")

        selector = _make_selector_with_dir(tmp_path)
        performances = {
            "strat_a": StrategyPerformance(strategy_name="strat_a", rolling_sharpe_60d=1.0),
            "strat_b": StrategyPerformance(strategy_name="strat_b", rolling_sharpe_60d=2.0),
        }
        alloc = selector.select_performance_weighted(performances)
        total = sum(alloc.allocations.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_select_performance_weighted_skips_drift(self, tmp_path):
        """Strategies with drift_detected=True should receive zero allocation."""
        _write_strategy(tmp_path, "drifted_strat")
        _write_strategy(tmp_path, "healthy_strat")

        selector = _make_selector_with_dir(tmp_path)
        performances = {
            "drifted_strat": StrategyPerformance(
                strategy_name="drifted_strat",
                rolling_sharpe_60d=0.8,
                drift_detected=True,
            ),
            "healthy_strat": StrategyPerformance(
                strategy_name="healthy_strat",
                rolling_sharpe_60d=0.8,
                drift_detected=False,
            ),
        }
        alloc = selector.select_performance_weighted(performances)
        # Drifted strategy should not appear in allocations (or have zero weight).
        drift_weight = alloc.allocations.get("drifted_strat", 0.0)
        assert drift_weight == pytest.approx(0.0, abs=1e-9)

    def test_select_bandit_returns_valid_strategy(self, tmp_path):
        """Bandit selection should return a StrategyAllocation with a single 1.0 weight."""
        _write_strategy(tmp_path, "strat_a")
        _write_strategy(tmp_path, "strat_b")

        selector = _make_selector_with_dir(tmp_path)
        alloc = selector.select_bandit({})
        assert alloc.selection_method == SelectionMethod.BANDIT
        assert alloc.active_strategy in {"strat_a", "strat_b"}
        assert alloc.allocations[alloc.active_strategy] == pytest.approx(1.0)
        assert len(alloc.allocations) == 1

    def test_select_manual_finds_by_name(self, tmp_path):
        """Manual selection with a known name should return that strategy at 1.0."""
        _write_strategy(tmp_path, "target_strat")
        _write_strategy(tmp_path, "other_strat")

        selector = _make_selector_with_dir(tmp_path)
        alloc = selector.select_manual("target_strat")
        assert alloc.active_strategy == "target_strat"
        assert alloc.allocations["target_strat"] == pytest.approx(1.0)
        assert alloc.selection_method == SelectionMethod.MANUAL

    def test_select_manual_fallback_on_missing(self, tmp_path):
        """Manual selection with an unknown name should fall back without raising."""
        _write_strategy(tmp_path, "only_strat")

        selector = _make_selector_with_dir(tmp_path)
        alloc = selector.select_manual("nonexistent_strategy")
        # Should fall back to the only available strategy.
        assert alloc.active_strategy == "only_strat"
        assert alloc.allocations.get("only_strat", 0.0) == pytest.approx(1.0)

    def test_select_dispatcher_forced_strategy(self, tmp_path):
        """When forced_strategy is set in config, select() should honour it."""
        _write_strategy(tmp_path, "forced_one")
        _write_strategy(tmp_path, "other_one")

        selector = _make_selector_with_dir(tmp_path)
        # Override via strategy_config
        selector.strategy_config.forced_strategy = "forced_one"
        alloc = selector.select("risk_on")
        assert alloc.active_strategy == "forced_one"
        assert alloc.selection_method == SelectionMethod.MANUAL


# ---------------------------------------------------------------------------
# 4. StrategyTracker tests
# ---------------------------------------------------------------------------


class TestStrategyTracker:
    def test_attribute_pnl_appends(self, tmp_path):
        """Attributing P&L should grow the daily_pnl list."""
        tracker = _make_tracker_with_dir(tmp_path)
        tracker.attribute_pnl("test_strat", 100.0)
        tracker.attribute_pnl("test_strat", -50.0)
        tracker.attribute_pnl("test_strat", 75.0)

        perf = tracker.performances["test_strat"]
        assert len(perf.daily_pnl) == 3
        assert perf.daily_pnl == [100.0, -50.0, 75.0]

    def test_update_metrics_rolling_sharpe(self, tmp_path):
        """After 65 daily P&L entries, rolling Sharpe values should be non-zero."""
        tracker = _make_tracker_with_dir(tmp_path)
        import numpy as np
        rng = np.random.default_rng(42)
        # Use clearly positive mean so Sharpe > 0
        pnls = rng.normal(50, 100, 65).tolist()

        for pnl in pnls:
            tracker.performances.setdefault(
                "sharpe_strat",
                StrategyPerformance(strategy_name="sharpe_strat"),
            )
            tracker.performances["sharpe_strat"].daily_pnl.append(pnl)
            tracker.performances["sharpe_strat"].total_pnl += pnl

        tracker.update_metrics("sharpe_strat")
        perf = tracker.performances["sharpe_strat"]
        assert isinstance(perf.rolling_sharpe_30d, float)
        assert isinstance(perf.rolling_sharpe_60d, float)
        assert perf.rolling_sharpe_30d != 0.0
        assert perf.rolling_sharpe_60d != 0.0

    def test_update_metrics_win_rate(self, tmp_path):
        """Win rate should be approximately 0.5 for balanced wins and losses."""
        tracker = _make_tracker_with_dir(tmp_path)
        # 10 positive + 10 negative days
        pnls = [100.0] * 10 + [-100.0] * 10

        for pnl in pnls:
            tracker.performances.setdefault(
                "win_strat",
                StrategyPerformance(strategy_name="win_strat"),
            )
            tracker.performances["win_strat"].daily_pnl.append(pnl)
            tracker.performances["win_strat"].total_pnl += pnl

        tracker.update_metrics("win_strat")
        perf = tracker.performances["win_strat"]
        assert perf.win_rate == pytest.approx(0.5, abs=0.01)

    def test_check_drift_not_triggered_below_threshold(self, tmp_path):
        """Drift should NOT be flagged when live Sharpe >= threshold * backtest Sharpe."""
        tracker = _make_tracker_with_dir(tmp_path)
        tracker.performances["no_drift"] = StrategyPerformance(
            strategy_name="no_drift",
            rolling_sharpe_60d=0.8,
            backtest_sharpe=1.0,
        )
        # threshold = 0.5 (default); 0.8 >= 0.5 * 1.0 → no drift
        result = tracker.check_drift("no_drift")
        assert result is False
        assert tracker.performances["no_drift"].drift_detected is False

    def test_check_drift_triggered_above_threshold(self, tmp_path):
        """Drift SHOULD be flagged when live Sharpe < threshold * backtest Sharpe."""
        tracker = _make_tracker_with_dir(tmp_path)
        tracker.performances["drifted"] = StrategyPerformance(
            strategy_name="drifted",
            rolling_sharpe_60d=0.3,
            backtest_sharpe=1.0,
        )
        # threshold = 0.5; 0.3 < 0.5 * 1.0 → drift
        result = tracker.check_drift("drifted")
        assert result is True
        assert tracker.performances["drifted"].drift_detected is True

    def test_save_and_reload(self, tmp_path):
        """Attributing P&L, saving, then reloading should restore performance data."""
        tracker1 = _make_tracker_with_dir(tmp_path)
        tracker1.attribute_pnl("persist_strat", 200.0)
        tracker1.attribute_pnl("persist_strat", 150.0)
        # save() is already called inside attribute_pnl, but call explicitly to be sure.
        tracker1.save()

        # Create a new tracker from the same directory.
        tracker2 = _make_tracker_with_dir(tmp_path)
        assert "persist_strat" in tracker2.performances
        perf = tracker2.performances["persist_strat"]
        assert len(perf.daily_pnl) == 2
        assert perf.total_pnl == pytest.approx(350.0)

    def test_report_structure(self, tmp_path):
        """report() should return a dict with 'total_strategies' and 'strategies' keys."""
        tracker = _make_tracker_with_dir(tmp_path)
        tracker.attribute_pnl("rpt_strat", 50.0)

        report = tracker.report()
        assert "total_strategies" in report
        assert "strategies" in report
        assert report["total_strategies"] == 1
        assert "rpt_strat" in report["strategies"]
        strat_info = report["strategies"]["rpt_strat"]
        assert "rolling_sharpe_60d" in strat_info
        assert "win_rate" in strat_info
        assert "drift_detected" in strat_info


# ---------------------------------------------------------------------------
# 5. Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineStrategyIntegration:
    def test_run_strategy_list_imports(self):
        """scripts/run_strategy.py should import the 'app' typer object without errors."""
        from scripts.run_strategy import app  # noqa: F401

        assert app is not None

    def test_strategy_selector_with_real_library(self):
        """StrategySelector using the real library should return a valid StrategyAllocation."""
        from src.core.strategy_selector import StrategySelector

        selector = StrategySelector()
        alloc = selector.select("risk_on")
        assert isinstance(alloc, StrategyAllocation)
        assert alloc.active_strategy != ""
        assert len(alloc.allocations) >= 1
        total = sum(alloc.allocations.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_strategy_tracker_attribute_save_load(self, tmp_path):
        """Full round-trip: attribute P&L → save → reload → verify consistency."""
        from src.core.strategy_tracker import StrategyTracker

        # Step 1: create tracker and attribute several days of P&L.
        tracker_a = _make_tracker_with_dir(tmp_path)
        for pnl in [100.0, -20.0, 50.0, 80.0, -10.0]:
            tracker_a.attribute_pnl("roundtrip_strat", pnl)

        expected_total = sum([100.0, -20.0, 50.0, 80.0, -10.0])
        assert tracker_a.performances["roundtrip_strat"].total_pnl == pytest.approx(
            expected_total
        )

        # Step 2: reload from the same directory.
        tracker_b = _make_tracker_with_dir(tmp_path)
        assert "roundtrip_strat" in tracker_b.performances
        reloaded = tracker_b.performances["roundtrip_strat"]
        assert reloaded.total_pnl == pytest.approx(expected_total)
        assert len(reloaded.daily_pnl) == 5
