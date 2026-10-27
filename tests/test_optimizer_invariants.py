"""Analytic mathematics and complete constraints, using synthetic test fixtures."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.core.backtester import PortfolioReplay
from src.domain.portfolio import FillPolicy, PortfolioSnapshot, RiskPolicy
from src.execution.portfolio_optimizer import AllocationConstraints, PortfolioOptimizer
from tests.test_core_backtest import START, A, B, replay_inputs

METHODS = ["risk_parity", "mean_variance", "black_litterman", "hrp", "max_diversification"]


def loose(keys=("A", "B")):
    return AllocationConstraints(
        dict.fromkeys(keys, 1.0), {k: k for k in keys}, sector_cap=1.0, min_cash=0.0
    )


def allocate(method, cov, limits):
    optimizer = PortfolioOptimizer()
    if method in ("mean_variance", "black_litterman"):
        args = [pd.Series(0.1, index=cov.columns), cov]
        if method == "black_litterman":
            args.append({cov.columns[0]: 0.2})
    else:
        args = [cov]
    return getattr(optimizer, method)(*args, limits=limits)


def test_analytic_two_asset_solutions_and_covariance_scale_invariance():
    optimizer = PortfolioOptimizer()
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])
    for scale in (1.0, 1e-9, 1e9):
        erc = optimizer.risk_parity(cov * scale, limits=loose())
        assert erc.status == "optimized"
        assert list(erc.weights.values()) == pytest.approx([2 / 3, 1 / 3], abs=1e-5)
        assert list(erc.risk_contribution.values()) == pytest.approx([50, 50], abs=0.002)
    hrp = optimizer.hrp(cov, limits=loose())
    assert list(hrp.weights.values()) == pytest.approx([0.8, 0.2], abs=1e-5)
    div = optimizer.max_diversification(cov, limits=loose())
    assert list(div.weights.values()) == pytest.approx([2 / 3, 1 / 3], abs=1e-5)
    assert div.status == hrp.status == "optimized"


def test_black_litterman_uses_supplied_prior_and_independent_view_confidence():
    cov = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])
    optimizer = PortfolioOptimizer()
    prior = pd.Series({"A": 0.1, "B": 0.02})
    result = optimizer.black_litterman(
        prior, cov, {"A": 0.2}, confidence={"A": 0.8}, limits=loose()
    )
    assert result.diagnostics["posterior"] == pytest.approx({"A": 0.18, "B": 0.02})
    assert optimizer.black_litterman(prior, cov, {}, limits=loose()).diagnostics[
        "posterior"
    ] == pytest.approx(prior.to_dict())
    with pytest.raises(ValueError, match="known instruments"):
        optimizer.black_litterman(prior, cov, {"missing": 0.1})
    with pytest.raises(ValueError, match="confidence"):
        optimizer.black_litterman(prior, cov, {"A": 0.1}, confidence={"A": 1.0})


@pytest.mark.parametrize("method", METHODS)
def test_all_five_obey_joint_constraints_including_cost_denominator_and_turnover(method):
    keys = ["A", "B", "C", "BTC", "ETH"]
    cov = pd.DataFrame(np.diag([0.1, 0.2, 0.3, 0.5, 0.6]), index=keys, columns=keys)
    limits = AllocationConstraints(
        dict(zip(keys, [0.10, 0.10, 0.10, 0.05, 0.05], strict=True)),
        dict(zip(keys, ["tech", "tech", "energy", "crypto", "crypto"], strict=True)),
        frozenset(["BTC", "ETH"]),
        sector_cap=0.12,
        crypto_cap=0.07,
        crypto_min=0.02,
        previous_weights=dict.fromkeys(keys, 0.0),
        max_turnover=0.25,
        buy_cost_rates=dict.fromkeys(keys, 0.02),
        trade_caps={"A": 0.03},
    )
    result = allocate(method, cov, limits)
    assert result.status == "optimized", result.diagnostics
    assert limits.violations(result.weights) == []
    assert result.weights["A"] <= 0.03 + 1e-9
    assert result.weights["A"] + result.weights["B"] <= 0.12
    assert sum(result.weights[k] for k in ["BTC", "ETH"]) <= 0.07
    assert sum(result.weights.values()) <= 0.25 + 1e-9
    # No post-solver renormalization to full investment.
    assert result.weights["B"] < 0.1


def test_feasible_fallback_is_identified_and_rechecked(monkeypatch):
    import src.execution.portfolio_optimizer as module

    def failed(*args, **kwargs):
        raise RuntimeError("forced optimizer outage")

    monkeypatch.setattr(module, "minimize", failed)
    cov = pd.DataFrame(np.diag([0.1, 0.2]), index=["A", "B"], columns=["A", "B"])
    for method in METHODS:
        result = allocate(method, cov, loose())
        assert result.status == "fallback_feasible"
        assert result.constraints.violations(result.weights) == []
        assert "forced optimizer outage" in result.diagnostics["solver_failure"]
    impossible = replace(loose(), crypto_min=0.1)
    with pytest.raises(ValueError, match="infeasible"):
        allocate("risk_parity", cov, impossible)


def test_shrinkage_from_observed_returns_and_input_rejections():
    observed = pd.DataFrame(
        [[0.1, 0.2], [-0.2, 0.1], [0.05, -0.1], [0.01, 0.03]], columns=["A", "B"]
    )
    cov = PortfolioOptimizer.estimate_covariance(observed)
    assert cov.to_numpy() == pytest.approx(LedoitWolf().fit(observed).covariance_)
    result = PortfolioOptimizer().optimize(
        "mean_variance", observed, limits=loose(), expected_returns=pd.Series({"A": 0.1, "B": -0.1})
    )
    assert result.diagnostics["covariance_estimator"] == "LedoitWolf"
    assert result.weights["B"] < 1e-7
    with pytest.raises(ValueError, match="complete"):
        PortfolioOptimizer().mean_variance(pd.Series({"A": 0.1}), cov)
    with pytest.raises(ValueError, match="complete"):
        PortfolioOptimizer.estimate_covariance(observed.mask(observed == 0.1))
    with pytest.raises(ValueError, match="semidefinite"):
        PortfolioOptimizer().hrp(
            pd.DataFrame([[1, 2], [2, 1]], index=["A", "B"], columns=["A", "B"])
        )


def test_cash_halt_and_post_rounding_turnover_revalidation():
    cov = pd.DataFrame(np.diag([0.1, 0.2]), index=["A", "B"], columns=["A", "B"])
    limits = replace(
        loose(), previous_weights={"A": 0.4, "B": 0.3}, max_turnover=0.1, buy_halt=True
    )
    result = allocate("mean_variance", cov, limits)
    assert result.weights["A"] <= 0.4 and result.weights["B"] <= 0.3
    assert limits.violations(result.weights) == []
    metadata = {
        m.instrument.instrument_id: m.model_copy(update={"lot_size": D(100)})
        for m in replay_inputs()["metadata"]
    }
    result.constraints = replace(limits, max_turnover=0.0)
    result.weights = {"A": 0.4, "B": 0.3}
    with pytest.raises(ValueError, match="post-rounding"):
        result.to_target(
            instruments={"A": A, "B": B},
            target_id="bad-round",
            decision_at=START,
            strategy_id="test",
            artifact_version="test",
            data_snapshot_id="fixture",
            universe_version="test-universe",
            nav=D(10000),
            prices={"A": D(100), "B": D(100)},
            metadata=metadata,
        )


@pytest.mark.parametrize("method", METHODS)
def test_real_optimizer_to_typed_target_to_durable_replay(method, tmp_path):
    inputs = replay_inputs()
    inputs.pop("signals")
    metadata = {m.instrument.instrument_id: m for m in inputs["metadata"]}
    snapshot = PortfolioSnapshot(
        account_id=inputs["account"].account_id,
        as_of=START + timedelta(hours=6),
        cash=10000,
        reserved_cash=0,
        nav=10000,
        high_water_mark=10000,
        realized_pnl=0,
        fees=0,
        turnover_today=0,
    )
    policy = RiskPolicy(equity_weight=".4", sector_weight=".4", max_daily_turnover=".8")
    costs = FillPolicy(equity_cost_bps=10)
    limits = AllocationConstraints.for_portfolio(snapshot, metadata, policy, costs, pending=[])
    observed = pd.DataFrame(
        [[0.1, 0.2], [-0.2, 0.1], [0.05, -0.1], [0.01, 0.03]], columns=["A", "B"]
    )
    result = PortfolioOptimizer().optimize(
        method,
        observed,
        limits=limits,
        expected_returns=pd.Series({"A": 0.1, "B": 0.05}),
        views={"A": 0.15},
    )
    target = result.to_target(
        instruments={"A": A, "B": B},
        target_id=method,
        decision_at=snapshot.as_of,
        strategy_id=method,
        artifact_version="test",
        data_snapshot_id="fixture",
        universe_version="test-universe",
        nav=snapshot.nav,
        prices={"A": D(100), "B": D(100)},
        metadata=metadata,
    )
    replay = PortfolioReplay(tmp_path / f"{method}.sqlite3").run(
        **{**inputs, "targets": [target], "risk_policy": policy, "fill_policy": costs}
    )
    assert replay.reconciliation["reconciled"]
    assert replay.metrics.total_trades == 2
    assert replay.metrics.fees > 0
    assert replay.equity_curve[-1] < replay.equity_curve[0]
    assert replay.metrics.max_drawdown < 0
