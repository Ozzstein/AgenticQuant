"""Matched actual-ledger benchmark cases; fixture prices are synthetic."""

from decimal import Decimal as D

import numpy as np
import pytest

from src.core.backtester import PortfolioReplay
from src.core.benchmark import BenchmarkRunner, verify_comparison
from src.utils.exceptions import BacktestError
from tests.test_core_backtest import B, replay_inputs


def varied_inputs():
    args = replay_inputs()
    args["bars"] = [
        b.model_copy(
            update={
                "open": D(100 if b.event_id == "B-1" else 103),
                "high": D(105),
                "low": D(97),
                "close": D(103 if b.event_id == "B-1" else 97),
            }
        )
        if b.instrument == B and b.event_id != "B-0"
        else b
        for b in args["bars"]
    ]
    return args


def test_1000_random_portfolios_use_real_costed_nav_and_disclose_missing_benchmarks(tmp_path):
    args = varied_inputs()
    base = PortfolioReplay(tmp_path / "base.db").run(**args)
    result = BenchmarkRunner().compare_replay(
        base, inputs=args, output_dir=tmp_path / "comparison", n_simulations=1000
    )
    assert len(result.random_sharpes) == len(result.random_nav) == 1000
    assert all(len(nav) == len(base.timestamps) for nav in result.random_nav)
    assert np.std([s for s in result.random_sharpes if s is not None]) > 0
    assert result.monte_carlo_percentile is not None
    assert {"SPY", "BTC", "60/40"} <= set(result.unavailable)
    assert "equal_weight" in result.benchmark_metrics
    assert (tmp_path / "comparison" / "benchmarks.json").exists()
    assert len(result.random_artifacts) == 1000
    # Verification traverses all 1000 actual ledgers before rejecting missing named coverage.
    with pytest.raises(BacktestError, match="missing durable named benchmark SPY"):
        verify_comparison(base, result)


def test_small_universe_does_not_fabricate_random_percentile(tmp_path):
    args = varied_inputs()
    args["topk"] = 2
    base = PortfolioReplay(tmp_path / "base.db").run(**args)
    result = BenchmarkRunner().compare_replay(
        base, inputs=args, output_dir=tmp_path / "comparison", n_simulations=10
    )
    assert result.monte_carlo_percentile is None
    assert result.random_sharpes == []
    assert "degenerate" in result.unavailable["random_selection"]


def test_cost_scenarios_replay_fills_once_and_match_dates(tmp_path):
    args = varied_inputs()
    result = BenchmarkRunner().cost_sensitivity(inputs=args, output_dir=tmp_path / "costs")
    assert len(result) == 3
    assert result[0].timestamps == result[1].timestamps == result[2].timestamps
    assert result[0].metrics.fees == 0
    assert result[1].metrics.fees > 0
    for candidate in result:
        assert candidate.metrics.fees == pytest.approx(
            sum(float(t["fee"]) for t in candidate.trades)
        )
        assert candidate.reconciliation["reconciled"]
    assert result[1].metrics.fees < result[2].metrics.fees


def test_mismatched_benchmark_input_fails(tmp_path):
    args = varied_inputs()
    base = PortfolioReplay(tmp_path / "base.db").run(**args)
    args["signals"] = [s.model_copy(update={"score": 100}) for s in args["signals"]]
    with pytest.raises(BacktestError, match="differ"):
        BenchmarkRunner().compare_replay(
            base, inputs=args, output_dir=tmp_path / "comparison", n_simulations=5
        )


def named_inputs():
    from src.domain.common import Instrument
    from src.domain.portfolio import InstrumentRisk

    args = varied_inputs()
    for symbol, asset in [("SPY", "etf"), ("AGG", "etf"), ("BTC/USD", "crypto")]:
        instrument = Instrument(
            instrument_id=symbol, symbol=symbol, asset_class=asset, quote_currency="USD"
        )
        for source in list(args["bars"]):
            if source.instrument.instrument_id == "A":
                args["bars"].append(
                    source.model_copy(
                        update={
                            "instrument": instrument,
                            "event_id": symbol + source.event_id,
                            "calendar": "24/7" if asset == "crypto" else "XNYS",
                        }
                    )
                )
        args["metadata"].append(
            InstrumentRisk(
                instrument=instrument,
                sector=symbol,
                adv_base=100000000,
                as_of=args["account"].opened_at,
                lot_size=".000001",
            )
        )
    args["universes"] = [
        args["universes"][0].model_copy(
            update={"instrument_ids": ("A", "B", "SPY", "AGG", "BTC/USD")}
        )
    ]
    return args


def test_named_benchmarks_use_supplied_equity_crypto_prices_and_6040_weights(tmp_path):
    args = named_inputs()
    base = PortfolioReplay(tmp_path / "base.db").run(**args)
    result = BenchmarkRunner().compare_replay(
        base, inputs=args, output_dir=tmp_path / "comparison", n_simulations=5
    )
    verify_comparison(base, result, minimum_simulations=5)
    assert all(name in result.benchmark_metrics for name in ("SPY", "BTC", "60/40", "equal_weight"))
    assert result.benchmark_metrics["60/40"]["weights"] == {"SPY": "0.6", "AGG": "0.4"}
    assert result.benchmark_metrics["SPY"]["total_return"] == pytest.approx(
        base.metrics.total_return
    )
