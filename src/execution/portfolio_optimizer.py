"""Five constrained allocators over observed returns and explicit covariance.

Weights are fractions of pre-trade NAV. Trading costs reduce the denominator of
all concentration and cash limits. Covariance and expected returns must use the
same sampling convention; this module never invents returns or fills missing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.spatial.distance import squareform

from src.domain.common import Instrument
from src.domain.portfolio import (
    FillPolicy,
    InstrumentRisk,
    PortfolioSnapshot,
    PortfolioTarget,
    RiskPolicy,
    TargetWeight,
)
from src.utils.config import AppConfig, get_config


@dataclass(frozen=True)
class AllocationConstraints:
    """Complete, explicit allocation limits. Turnover is gross one-way notional.

    Current weights cover the entire account, including instruments to exit.
    Open orders must be reconciled/cancelled before constructing a new allocation;
    the execution router separately checks all pending orders at submission.
    """

    asset_caps: dict[str, float]
    sectors: dict[str, str]
    crypto_assets: frozenset[str] = frozenset()
    sector_cap: float = 0.30
    crypto_cap: float = 0.20
    crypto_min: float = 0
    min_cash: float = 0.05
    previous_weights: dict[str, float] | None = None
    max_turnover: float | None = None
    trade_caps: dict[str, float] = field(default_factory=dict)
    buy_cost_rates: dict[str, float] = field(default_factory=dict)
    sell_cost_rates: dict[str, float] = field(default_factory=dict)
    buy_halt: bool = False

    @classmethod
    def for_portfolio(
        cls,
        snapshot: PortfolioSnapshot,
        metadata: dict[str, InstrumentRisk],
        policy: RiskPolicy,
        costs: FillPolicy,
        *,
        pending: list[dict],
        crypto_min: float = 0,
    ) -> AllocationConstraints:
        if pending or snapshot.reserved_cash:
            raise ValueError(
                "reconcile/cancel open orders before allocating; pending risk stays binding"
            )
        if snapshot.nav <= 0 or costs.minimum_commission or costs.commission_per_unit:
            raise ValueError(
                "positive NAV and proportional costs required for allocation; fixed fees need explicit routing"
            )
        if not {p.instrument.instrument_id for p in snapshot.positions} <= set(metadata):
            raise ValueError("allocation must include every existing holding")
        for key, meta in metadata.items():
            if (
                key != meta.instrument.instrument_id
                or not 0
                <= (snapshot.as_of - meta.as_of).total_seconds()
                <= policy.max_liquidity_age_seconds
            ):
                raise ValueError("instrument identity or available liquidity metadata invalid")
        for holding in snapshot.positions:
            if (
                holding.instrument != metadata[holding.instrument.instrument_id].instrument
                or not 0
                <= (snapshot.as_of - holding.marked_at).total_seconds()
                <= policy.max_price_age_seconds
            ):
                raise ValueError("holding identity or available mark invalid")
        previous = {key: 0.0 for key in metadata}
        previous.update(
            {
                p.instrument.instrument_id: float(p.market_value / snapshot.nav)
                for p in snapshot.positions
            }
        )
        crypto = frozenset(k for k, m in metadata.items() if m.instrument.asset_class == "crypto")
        rates = {
            k: float((costs.crypto_cost_bps if k in crypto else costs.equity_cost_bps) / 10000)
            for k in metadata
        }
        # Strict ADV boundary leaves numerical margin; actual lots are checked again.
        trade_caps = {
            k: float(m.adv_base * policy.max_adv_participation / snapshot.nav) * (1 - 1e-8)
            if m.adv_base >= policy.min_adv
            else 0.0
            for k, m in metadata.items()
        }
        return cls(
            asset_caps={
                k: float(policy.crypto_weight if k in crypto else policy.equity_weight)
                for k in metadata
            },
            sectors={k: m.sector for k, m in metadata.items()},
            crypto_assets=crypto,
            sector_cap=float(policy.sector_weight),
            crypto_cap=float(policy.crypto_total),
            crypto_min=crypto_min,
            min_cash=float(policy.min_cash),
            previous_weights=previous,
            max_turnover=max(
                0.0, float(policy.max_daily_turnover - snapshot.turnover_today / snapshot.nav)
            ),
            trade_caps=trade_caps,
            buy_cost_rates=rates,
            sell_cost_rates=rates,
            buy_halt=snapshot.nav <= snapshot.high_water_mark * (1 - policy.max_drawdown),
        )

    def validate(self, keys: list[str]) -> None:
        ids = set(keys)
        if (
            not keys
            or len(ids) != len(keys)
            or set(self.asset_caps) != ids
            or set(self.sectors) != ids
        ):
            raise ValueError("complete unique asset caps and sectors required")
        if not self.crypto_assets <= ids or any(not s for s in self.sectors.values()):
            raise ValueError("unknown crypto assets or missing sectors")
        for mapping in (self.trade_caps, self.buy_cost_rates, self.sell_cost_rates):
            if not set(mapping) <= ids or any(
                not np.isfinite(v) or v < 0 for v in mapping.values()
            ):
                raise ValueError("unknown or invalid trade/cost constraints")
        values = [
            *self.asset_caps.values(),
            self.sector_cap,
            self.crypto_cap,
            self.crypto_min,
            self.min_cash,
        ]
        if (
            any(not np.isfinite(v) or not 0 <= v <= 1 for v in values)
            or self.crypto_min > self.crypto_cap
        ):
            raise ValueError("allocation fractions must be finite and within [0,1]")
        if self.max_turnover is not None and (
            not np.isfinite(self.max_turnover) or self.max_turnover < 0
        ):
            raise ValueError("turnover must be finite and nonnegative")
        if self.previous_weights is None:
            if self.max_turnover is not None or self.trade_caps or self.buy_halt:
                raise ValueError("trade constraints require current weights")
        elif (
            set(self.previous_weights) != ids
            or any(not np.isfinite(v) or v < 0 for v in self.previous_weights.values())
            or sum(self.previous_weights.values()) > 1 + 1e-10
        ):
            raise ValueError("complete unlevered current weights required")

    def violations(self, weights: dict[str, float], tolerance: float = 1e-9) -> list[str]:
        keys = list(self.asset_caps)
        self.validate(keys)
        if set(weights) != set(keys) or any(not np.isfinite(v) for v in weights.values()):
            return ["nonfinite/incomplete weights"]
        old = self.previous_weights or dict.fromkeys(keys, 0.0)
        change = {k: weights[k] - old[k] for k in keys}
        fee = sum(
            max(0.0, change[k]) * self.buy_cost_rates.get(k, 0.0)
            + max(0.0, -change[k]) * self.sell_cost_rates.get(k, 0.0)
            for k in keys
        )
        nav = 1 - fee
        failures = []
        if min(weights.values()) < -tolerance or nav <= 0:
            failures.append("long_only/solvency")
        if any(weights[k] > self.asset_caps[k] * nav + tolerance for k in keys):
            failures.append("asset_cap")
        if sum(weights.values()) > (1 - self.min_cash) * nav + tolerance:
            failures.append("cash_reserve")
        for sector in set(self.sectors.values()):
            if (
                sum(weights[k] for k in keys if self.sectors[k] == sector)
                > self.sector_cap * nav + tolerance
            ):
                failures.append("sector_cap")
        crypto = sum(weights[k] for k in self.crypto_assets)
        if not self.crypto_min * nav - tolerance <= crypto <= self.crypto_cap * nav + tolerance:
            failures.append("crypto_band")
        if (
            self.max_turnover is not None
            and sum(abs(v) for v in change.values()) > self.max_turnover + tolerance
        ):
            failures.append("turnover")
        if any(abs(change[k]) > cap + tolerance for k, cap in self.trade_caps.items()):
            failures.append("participation")
        if self.buy_halt and any(v > tolerance for v in change.values()):
            failures.append("buy_halt")
        return failures


@dataclass
class OptimizationResult:
    weights: dict[str, float]
    risk_contribution: dict[str, float | None]
    method: str = ""
    status: str = "optimized"
    diagnostics: dict = field(default_factory=dict)
    constraints: AllocationConstraints | None = field(default=None, repr=False)

    def to_target(
        self,
        *,
        instruments: dict[str, Instrument],
        target_id: str,
        decision_at,
        strategy_id: str,
        artifact_version: str,
        data_snapshot_id: str,
        universe_version: str,
        nav: Decimal | None = None,
        prices: dict[str, Decimal] | None = None,
        metadata: dict[str, InstrumentRisk] | None = None,
    ) -> PortfolioTarget:
        """Explicit Decimal boundary; optional lot rounding is revalidated, never normalized."""
        if set(instruments) != set(self.weights):
            raise ValueError("exact instrument identities required")
        weights = {k: Decimal(str(max(0.0, v))) for k, v in self.weights.items()}
        if nav is not None or prices is not None or metadata is not None:
            if (
                nav is None
                or nav <= 0
                or prices is None
                or metadata is None
                or set(prices) != set(weights)
                or set(metadata) != set(weights)
            ):
                raise ValueError(
                    "rounding requires NAV, base-currency prices and complete lot metadata"
                )
            for key, weight in weights.items():
                if prices[key] <= 0 or metadata[key].instrument != instruments[key]:
                    raise ValueError("invalid rounding price or identity")
                lot = metadata[key].lot_size
                qty = (weight * nav / prices[key] / lot).to_integral_value(
                    rounding=ROUND_DOWN
                ) * lot
                weights[key] = qty * prices[key] / nav
        if self.constraints is None or (
            failures := self.constraints.violations({k: float(v) for k, v in weights.items()})
        ):
            raise ValueError(
                f"target failed post-rounding constraints: {failures if self.constraints else 'missing limits'}"
            )
        return PortfolioTarget(
            target_id=target_id,
            decision_at=decision_at,
            strategy_id=strategy_id,
            artifact_version=artifact_version,
            data_snapshot_id=data_snapshot_id,
            universe_version=universe_version,
            weights=tuple(
                TargetWeight(instrument=instruments[k], weight=w) for k, w in weights.items()
            ),
        )


class PortfolioOptimizer:
    def __init__(self, config: AppConfig | None = None):
        self._config = config or get_config()

    @staticmethod
    def estimate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage on complete observed rows, with no forward filling."""
        from sklearn.covariance import LedoitWolf

        if (
            returns.empty
            or len(returns) < 3
            or not returns.columns.is_unique
            or not returns.index.is_unique
            or not returns.index.is_monotonic_increasing
            or not np.isfinite(returns.to_numpy(dtype=float)).all()
        ):
            raise ValueError("at least three chronological complete observed return rows required")
        fit = LedoitWolf().fit(returns.to_numpy(dtype=float))
        return pd.DataFrame(fit.covariance_, index=returns.columns, columns=returns.columns)

    def optimize(
        self,
        method: str,
        historical_returns: pd.DataFrame,
        *,
        limits: AllocationConstraints,
        expected_returns: pd.Series | None = None,
        views: dict[str, float] | None = None,
        **options,
    ) -> OptimizationResult:
        """Production entry point; all five methods use the same shrunk observed covariance."""
        if method not in {
            "risk_parity",
            "mean_variance",
            "black_litterman",
            "hrp",
            "max_diversification",
        }:
            raise ValueError(f"unsupported optimizer {method}")
        cov = self.estimate_covariance(historical_returns)
        fn = getattr(self, method)
        if method in {"mean_variance", "black_litterman"}:
            if expected_returns is None:
                raise ValueError("explicit expected returns/prior required")
            args = [expected_returns, cov]
            if method == "black_litterman":
                if views is None:
                    raise ValueError("explicit views (possibly empty) required")
                args.append(views)
        else:
            args = [cov]
        result = fn(*args, limits=limits, **options)
        result.diagnostics["covariance_estimator"] = "LedoitWolf"
        result.diagnostics["observations"] = len(historical_returns)
        return result

    @staticmethod
    def _covariance(cov: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        keys = list(cov.columns)
        array = cov.to_numpy(dtype=float)
        if (
            not keys
            or not cov.columns.is_unique
            or list(cov.index) != keys
            or array.shape != (len(keys), len(keys))
            or not np.isfinite(array).all()
        ):
            raise ValueError("finite square aligned covariance required")
        scale = float(np.max(np.abs(array)))
        if (
            scale <= 0
            or np.any(np.diag(array) <= 0)
            or not np.allclose(array, array.T, rtol=1e-10, atol=scale * 1e-12)
            or np.linalg.eigvalsh(array).min() < -scale * 1e-10
        ):
            raise ValueError(
                "symmetric positive semidefinite covariance with positive asset variances required"
            )
        return keys, (array + array.T) / 2

    @staticmethod
    def _returns(returns: pd.Series, keys: list[str]) -> np.ndarray:
        if (
            not returns.index.is_unique
            or set(returns.index) != set(keys)
            or not np.isfinite(returns.to_numpy(dtype=float)).all()
        ):
            raise ValueError("complete aligned finite expected returns required")
        return returns.reindex(keys).to_numpy(dtype=float)

    def _limits(self, keys, limits):
        if limits is None:
            # Compatibility covariance-only API has no instrument classifications.
            # Production callers must use optimize(..., limits=typed_limits).
            limits = AllocationConstraints(
                asset_caps=dict.fromkeys(keys, self._config.risk.max_position_weight / 100),
                sectors={k: k for k in keys},
                min_cash=self._config.risk.min_cash_pct / 100,
            )
        limits.validate(keys)
        return limits

    @staticmethod
    def _polytope(keys, limits):
        """Linear epigraph variables [weights, buys, sells] preserve every hard bound."""
        n = len(keys)
        old = np.array([(limits.previous_weights or {}).get(k, 0.0) for k in keys])
        fee = np.r_[
            np.zeros(n),
            [limits.buy_cost_rates.get(k, 0.0) for k in keys],
            [limits.sell_cost_rates.get(k, 0.0) for k in keys],
        ]
        rows, rhs = [], []

        def upper(mask, cap):
            row = cap * fee.copy()
            row[:n] += mask
            rows.append(row)
            rhs.append(cap)

        upper(np.ones(n), 1 - limits.min_cash)
        for i, key in enumerate(keys):
            upper(np.eye(n)[i], limits.asset_caps[key])
        for sector in sorted(set(limits.sectors.values())):
            upper(
                np.array([limits.sectors[k] == sector for k in keys], dtype=float),
                limits.sector_cap,
            )
        crypto_mask = np.array([k in limits.crypto_assets for k in keys], dtype=float)
        upper(crypto_mask, limits.crypto_cap)
        if limits.crypto_min:
            rows.append(-np.r_[crypto_mask, np.zeros(2 * n)] - limits.crypto_min * fee)
            rhs.append(-limits.crypto_min)
        if limits.max_turnover is not None:
            rows.append(np.r_[np.zeros(n), np.ones(2 * n)])
            rhs.append(limits.max_turnover)
        for i, key in enumerate(keys):
            if key in limits.trade_caps:
                row = np.zeros(3 * n)
                row[n + i] = row[2 * n + i] = 1
                rows.append(row)
                rhs.append(limits.trade_caps[key])
        # w - buys + sells = previous; buy/sell simultaneity only consumes limits.
        equality = np.c_[np.eye(n), -np.eye(n), np.eye(n)]
        bounds = [(0.0, float(old[i]) if limits.buy_halt else 1.0) for i in range(n)] + [
            (0.0, 1.0)
        ] * (2 * n)
        return np.array(rows), np.array(rhs), equality, old, bounds

    def _solve(self, method, cov, objective, *, limits=None, full_budget=True, diagnostics=None):
        keys, matrix = self._covariance(cov)
        limits = self._limits(keys, limits)
        n = len(keys)
        a, b, eq, old, bounds = self._polytope(keys, limits)
        seed = linprog(
            np.r_[-np.ones(n), np.full(2 * n, 1e-10)],
            A_ub=a,
            b_ub=b,
            A_eq=eq,
            b_eq=old,
            bounds=bounds,
            method="highs",
        )
        if not seed.success:
            raise ValueError(f"infeasible allocation constraints: {seed.message}")
        budget = float(seed.x[:n].sum())
        constraint_list = [LinearConstraint(a, -np.inf, b), LinearConstraint(eq, old, old)]
        if full_budget:
            constraint_list.append(
                LinearConstraint(np.r_[np.ones(n), np.zeros(2 * n)], budget, budget)
            )
        diag = dict(diagnostics or {})
        diag.update(
            investment_budget=budget, allocation_convention="pre_trade_nav_with_fee_adjusted_limits"
        )
        try:
            # ERC's boundary gradient can vanish. Project a diversified start
            # onto the exact feasible set before solving the risk objective.
            start = minimize(
                lambda x: float(np.square(x[:n] - budget / n).sum()) + 1e-8 * x[n:].sum(),
                seed.x,
                method="SLSQP",
                bounds=Bounds(*np.array(bounds).T),
                constraints=constraint_list,
                options={"maxiter": 1000, "ftol": 1e-13},
            )
            x0 = start.x if start.success else seed.x
            solution = minimize(
                lambda x: objective(x[:n], matrix, budget) + 1e-10 * x[n:].sum(),
                x0,
                method="SLSQP",
                bounds=Bounds(*np.array(bounds).T),
                constraints=constraint_list,
                options={"maxiter": 2000, "ftol": 1e-12},
            )
            values = dict(zip(keys, solution.x[:n], strict=True))
            if not solution.success or limits.violations(values):
                raise RuntimeError(str(solution.message))
            raw, status = solution.x[:n], "optimized"
        except Exception as exc:  # solver failure is explicit and revalidated
            raw, status = seed.x[:n], "fallback_feasible"
            diag["solver_failure"] = str(exc)
        weights = {k: float(max(0.0, w)) for k, w in zip(keys, raw, strict=True)}
        if failures := limits.violations(weights):
            raise ValueError(f"optimizer/fallback violated constraints: {failures}")
        vector = np.array(list(weights.values()))
        variance = float(vector @ matrix @ vector)
        contributions = vector * (matrix @ vector) / variance * 100 if variance > 0 else [None] * n
        return OptimizationResult(
            weights, dict(zip(keys, contributions, strict=True)), method, status, diag, limits
        )

    def risk_parity(self, cov: pd.DataFrame, *, limits: AllocationConstraints | None = None):
        def objective(w, matrix, budget):
            variance = w @ matrix @ w
            if budget <= 0 or variance <= 0:
                return 0.0
            rc = w * (matrix @ w) / variance
            return float(np.square(rc - 1 / len(w)).sum())

        return self._solve("risk_parity", cov, objective, limits=limits)

    def mean_variance(
        self,
        returns: pd.Series,
        cov: pd.DataFrame,
        *,
        limits: AllocationConstraints | None = None,
        risk_aversion: float = 1,
    ):
        keys, matrix = self._covariance(cov)
        mu = self._returns(returns, keys)
        if not np.isfinite(risk_aversion) or risk_aversion <= 0:
            raise ValueError("positive finite risk aversion required")
        scale = max(float(np.abs(mu).max()), risk_aversion * float(np.abs(matrix).max()), 1e-12)
        return self._solve(
            "mean_variance",
            cov,
            lambda w, c, _: (risk_aversion * (w @ c @ w) - mu @ w) / scale,
            limits=limits,
            full_budget=False,
        )

    def black_litterman(
        self,
        returns: pd.Series,
        cov: pd.DataFrame,
        views: dict[str, float],
        *,
        limits: AllocationConstraints | None = None,
        confidence: dict[str, float] | None = None,
        tau: float = 0.025,
        risk_aversion: float = 1,
    ):
        keys, matrix = self._covariance(cov)
        prior = self._returns(returns, keys)
        if (
            not set(views) <= set(keys)
            or any(not np.isfinite(v) for v in views.values())
            or not np.isfinite(tau)
            or tau <= 0
        ):
            raise ValueError("finite views on known instruments and positive tau required")
        confidence = dict.fromkeys(views, 0.5) if confidence is None else confidence
        if set(confidence) != set(views) or any(
            not np.isfinite(v) or not 0 < v < 1 for v in confidence.values()
        ):
            raise ValueError("each view requires confidence strictly between zero and one")
        posterior = prior.copy()
        if views:
            p = np.array([np.eye(len(keys))[keys.index(k)] for k in views])
            uncertainty = tau * matrix
            omega = np.diag(
                [
                    uncertainty[keys.index(k), keys.index(k)] * (1 - confidence[k]) / confidence[k]
                    for k in views
                ]
            )
            posterior += (
                uncertainty
                @ p.T
                @ np.linalg.solve(
                    p @ uncertainty @ p.T + omega, np.array(list(views.values())) - p @ prior
                )
            )
        result = self.mean_variance(
            pd.Series(posterior, index=keys), cov, limits=limits, risk_aversion=risk_aversion
        )
        result.method = "black_litterman"
        result.diagnostics.update(
            prior=dict(zip(keys, prior, strict=True)),
            posterior=dict(zip(keys, posterior, strict=True)),
            confidence=confidence,
            tau=tau,
        )
        return result

    def hrp(self, cov: pd.DataFrame, *, limits: AllocationConstraints | None = None):
        keys, matrix = self._covariance(cov)
        std = np.sqrt(np.diag(matrix))
        corr = np.clip(matrix / np.outer(std, std), -1, 1)
        distance = np.sqrt(np.maximum(0.0, (1 - corr) / 2))
        np.fill_diagonal(distance, 0)
        ordered = (
            list(
                leaves_list(
                    linkage(
                        squareform(distance, checks=False), method="single", optimal_ordering=True
                    )
                )
            )
            if len(keys) > 1
            else [0]
        )
        raw = np.ones(len(keys))
        clusters = [ordered]

        def cluster_variance(indices):
            sub = matrix[np.ix_(indices, indices)]
            inverse = 1 / np.diag(sub)
            w = inverse / inverse.sum()
            return float(w @ sub @ w)

        while clusters:
            splits = [c for c in clusters if len(c) > 1]
            clusters = []
            for c in splits:
                left, right = c[: len(c) // 2], c[len(c) // 2 :]
                vl, vr = cluster_variance(left), cluster_variance(right)
                alpha = vr / (vl + vr) if vl + vr > 0 else 0.5
                raw[left] *= alpha
                raw[right] *= 1 - alpha
                clusters.extend([left, right])
        return self._solve(
            "hrp",
            cov,
            lambda w, _c, budget: float(np.square(w - raw * budget).sum()),
            limits=limits,
            diagnostics={
                "unconstrained_weights": dict(zip(keys, raw, strict=True)),
                "linkage": "single",
                "leaf_order": [keys[i] for i in ordered],
            },
        )

    def max_diversification(
        self, cov: pd.DataFrame, *, limits: AllocationConstraints | None = None
    ):
        def objective(w, matrix, budget):
            variance = float(w @ matrix @ w)
            if budget <= 0:
                return 0.0
            if variance <= 0:
                return 1e12
            return -float(w @ np.sqrt(np.diag(matrix)) / np.sqrt(variance))

        return self._solve("max_diversification", cov, objective, limits=limits)
