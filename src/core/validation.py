"""One evidence-based validation service; missing critical evidence blocks approval."""

from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
import pandas as pd

from src.core.backtester import ReplayResult, canonical_hash, performance_metrics
from src.execution.ledger import PortfolioLedger
from src.utils.schemas import CheckResult, CheckVerdict, ValidationReport


class ValidationSuite:
    def __init__(self, config=None):
        self.config = config

    def validate(
        self, result, *, ledger_path=None, cost_results=None, benchmark=None
    ) -> ValidationReport:
        checks = []

        def check(name, passed, explanation, warning=False):
            checks.append(
                CheckResult(
                    name=name,
                    verdict=CheckVerdict.PASS
                    if passed
                    else CheckVerdict.WARNING
                    if warning
                    else CheckVerdict.FAIL,
                    explanation=explanation,
                )
            )

        if not isinstance(result, ReplayResult):
            check(
                "engine_version",
                False,
                "Legacy performance is invalid/unverified; rerun through chronological-ledger-v1.",
            )
            return ValidationReport(
                checks=checks,
                overall="NOT RELIABLE",
                summary="Legacy result cannot support promotion.",
            )
        check(
            "complete_run",
            result.status == "complete",
            f"Run status: {result.status}. Incomplete/failed folds cannot pass.",
        )

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        snapshots = result.snapshots
        try:
            timestamps = [s.as_of for s in snapshots]
            require(
                len(timestamps) >= 2 and timestamps == sorted(set(timestamps)),
                "invalid observation dates",
            )
            require(
                [s.as_of.isoformat() for s in snapshots] == result.timestamps, "timestamp mismatch"
            )
            require([float(s.nav) for s in snapshots] == result.equity_curve, "NAV curve mismatch")
            for snapshot in snapshots:
                require(
                    snapshot.cash + sum((p.market_value for p in snapshot.positions), Decimal(0))
                    == snapshot.nav,
                    "cash + holdings != NAV",
                )
                require(snapshot.reserved_cash <= snapshot.cash, "over-reserved cash")
            computed = performance_metrics(result.equity_curve, result.manifest["samples_per_year"])
            require(
                np.allclose(
                    result.daily_returns,
                    np.asarray(result.equity_curve[1:]) / result.equity_curve[:-1] - 1,
                    rtol=0,
                    atol=1e-12,
                ),
                "return mismatch",
            )
            for name, expected in computed.model_dump().items():
                if name in ("total_trades", "fees", "turnover"):
                    continue
                actual = getattr(result.metrics, name)
                require(
                    actual is None
                    if expected is None
                    else actual is not None
                    and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                    f"metric mismatch: {name}",
                )
            require(result.metrics.total_trades == len(result.trades), "trade count mismatch")
            fees = sum((Decimal(trade["fee"]) for trade in result.trades), Decimal(0))
            require(
                fees == snapshots[-1].fees
                and math.isclose(float(fees), result.metrics.fees, rel_tol=1e-12, abs_tol=1e-12),
                "fee mismatch",
            )
            turnover = 0.0
            for trade in result.trades:
                before = [s for s in snapshots if s.as_of < pd.Timestamp(trade["executed_at"])]
                denominator = before[-1].nav if before else snapshots[0].nav
                turnover += float(
                    Decimal(trade["quantity"])
                    * Decimal(trade["price"])
                    * Decimal(trade["fx_rate"])
                    / denominator
                )
            require(
                math.isclose(turnover, result.metrics.turnover, rel_tol=1e-12, abs_tol=1e-12),
                "turnover mismatch",
            )
            check(
                "nav_arithmetic",
                True,
                "Cash, holdings, dates, returns and every NAV/fill-derived metric reconcile, including initial capital and undefined values.",
            )
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            check("nav_arithmetic", False, f"Reported arithmetic does not reconcile: {exc}")
        if ledger_path is None:
            check(
                "durable_accounting",
                False,
                "Persisted ledger required for independent journal/state reconciliation.",
            )
        else:
            ledger = PortfolioLedger(ledger_path)
            try:
                ledger.reconcile(result.account_id)
                require(
                    ledger.history(result.account_id) == snapshots, "persisted snapshots differ"
                )
                row = ledger.db.execute(
                    "SELECT payload FROM runs WHERE account=? AND id=?",
                    (result.account_id, result.run_id),
                ).fetchone()
                require(row is not None, "run is absent from ledger")
                persisted = json.loads(row[0])
                require(persisted.get("manifest") == result.manifest, "persisted manifest differs")
                require(
                    persisted.get("manifest_hash") == canonical_hash(result.manifest),
                    "canonical manifest hash differs",
                )
                require(persisted.get("status") == result.status, "persisted run status differs")
                require(
                    persisted.get("result") == result.model_dump(mode="json"),
                    "report differs from persisted run result",
                )
                expected_trades = []
                for fill in ledger.fills(result.account_id):
                    order = ledger.order(result.account_id, fill.intent_id)["intent"]
                    economic = ledger.db.execute(
                        "SELECT cash_delta,realized,notional FROM fills WHERE account=? AND id=?",
                        (result.account_id, fill.fill_id),
                    ).fetchone()
                    expected_trades.append(
                        {
                            **fill.model_dump(mode="json"),
                            "cash_delta": economic["cash_delta"],
                            "realized_pnl": economic["realized"],
                            "notional": economic["notional"],
                            "instrument_id": order.instrument.instrument_id,
                            "side": order.side,
                            "strategy_id": order.strategy_id,
                            "decision_at": order.decision_at.isoformat(),
                        }
                    )
                require(
                    expected_trades == result.trades,
                    "persisted full fill/order/economic content differs",
                )
                check(
                    "durable_accounting",
                    True,
                    "Independent journal/state reconciliation, exact immutable manifest/hash binding, complete fill/order content and persisted report agree.",
                )
            except (ValueError, KeyError, TypeError) as exc:
                check("durable_accounting", False, f"Durable ledger mismatch: {exc}")
            finally:
                ledger.close()
        timing = bool(result.trades) and len({t["fill_id"] for t in result.trades}) == len(
            result.trades
        )
        for t in result.trades:
            timing &= pd.Timestamp(t["executed_at"]) > pd.Timestamp(t["decision_at"])
            timing &= (
                all(Decimal(t[k]) > 0 for k in ("quantity", "price", "fx_rate"))
                and Decimal(t["fee"]) >= 0
            )
        check(
            "fill_realism",
            timing,
            "Unique positive fills must follow decisions; simulator uses only observed open prices and bounded opening liquidity.",
        )
        provenance = result.manifest.get("provenance", {})
        needed = (
            "universe_history_id",
            "availability_audit_id",
            "calendar_artifact_id",
            "corporate_actions_id",
        )
        missing = [key for key in needed if not provenance.get(key)]
        check(
            "historical_provenance",
            not missing,
            "Missing immutable evidence references: " + ", ".join(missing)
            if missing
            else "Historical universe, availability audit, calendar and corporate-action artifacts referenced; source entitlement/content audit remains external.",
        )
        folds = result.walk_forward_results
        split_valid = bool(folds)
        for i, fold in enumerate(folds):
            split_valid &= (
                pd.Timestamp(fold.label_end_max)
                < pd.Timestamp(fold.train_end)
                <= pd.Timestamp(fold.validation_start)
                < pd.Timestamp(fold.validation_end)
                <= pd.Timestamp(fold.test_start)
            )
            split_valid &= pd.Timestamp(fold.validation_label_end_max) < pd.Timestamp(
                fold.validation_end
            )
            split_valid &= fold.train_rows > 0 and fold.validation_rows > 0 and fold.test_rows > 0
            if i:
                split_valid &= pd.Timestamp(folds[i - 1].test_end) <= pd.Timestamp(fold.test_start)
        check(
            "purged_oos",
            split_valid,
            "Requires fresh-model fold artifacts with training/validation separation, label-end purging and nonoverlapping OOS windows.",
        )
        # Inspect actual holdings. Price-driven drift is reported, not rewritten away.
        caps = result.manifest["risk_policy"]
        breaches = [
            (s.as_of, p.instrument.instrument_id)
            for s in snapshots
            for p in s.positions
            if s.nav > 0
            and p.market_value / s.nav
            > Decimal(
                caps["crypto_weight"]
                if p.instrument.asset_class == "crypto"
                else caps["equity_weight"]
            )
            + Decimal("0.000001")
        ]
        check(
            "concentration",
            not breaches,
            f"Observed position-cap breaches (including market drift): {len(breaches)}.",
            warning=True,
        )
        costs = cost_results or []
        matched = len(costs) >= 3
        for c in costs:
            matched &= c.status == "complete" and c.timestamps == result.timestamps
            matched &= all(
                c.manifest.get(k) == result.manifest.get(k)
                for k in (
                    "bars_hash",
                    "signals_hash",
                    "targets_hash",
                    "metadata_hash",
                    "universe_hash",
                    "actions_hash",
                    "risk_policy",
                )
            )
        eq_levels = {Decimal(c.manifest["fill_policy"]["equity_cost_bps"]) for c in costs}
        crypto_levels = {Decimal(c.manifest["fill_policy"]["crypto_cost_bps"]) for c in costs}
        matched &= {Decimal(0), Decimal(10), Decimal(20)} <= eq_levels and {
            Decimal(0),
            Decimal(20),
            Decimal(40),
        } <= crypto_levels
        check(
            "cost_sensitivity",
            matched,
            "Requires matched actual fill reruns at 0/10/20 bps equity and 0/20/40 bps crypto; no return-subtraction shortcut.",
        )
        values = [v for v in result.cross_sectional_ic.values() if v is not None]
        ic = float(np.mean(values)) if values else None
        ic_std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        icir = ic / ic_std if ic_std and ic_std > 0 else None
        check(
            "factor_ic",
            ic is not None and ic > 0.03 and icir is not None and icir > 0.5,
            f"Date-local IC={ic}, ICIR={icir}; required IC>0.03 and ICIR>0.5 (unannualized).",
        )
        corr = provenance.get("max_existing_factor_correlation")
        check(
            "factor_correlation",
            isinstance(corr, (float, int)) and math.isfinite(corr) and abs(corr) < 0.7,
            "Independent existing-factor correlation evidence must be below 0.7; absent is not a pass.",
        )
        trial = provenance.get("trial_registry", {})
        count = trial.get("trials", 0)
        registry = (
            isinstance(count, int)
            and count >= 1
            and bool(trial.get("artifact_id"))
            and bool(trial.get("untouched_holdout_id"))
        )
        check(
            "trial_holdout_registry",
            registry,
            "Requires complete trial count, immutable trial registry and an untouched holdout identity.",
        )
        returns = np.asarray(result.daily_returns)
        significant = False
        lower = None
        if len(returns) >= 60 and registry:
            rng = np.random.default_rng(42)
            block = max(2, round(len(returns) ** (1 / 3)))
            samples = []
            for _ in range(2000):
                starts = rng.integers(
                    0, len(returns) - block + 1, size=math.ceil(len(returns) / block)
                )
                sample = np.concatenate([returns[s : s + block] for s in starts])[: len(returns)]
                samples.append(float(sample.mean()))
            # Family-wise correction is conservative; it does not prove a trial
            # registry is complete. Registry integrity is a separate critical check.
            alpha = 0.05 / count
            lower = float(np.quantile(samples, alpha)) if alpha >= 1 / 2000 else None
            significant = lower is not None and lower > 0
        check(
            "statistical_significance",
            significant,
            f"Moving-block bootstrap lower mean-return bound after trial correction: {lower}; requires >=60 observations and resolvable positive bound.",
        )
        bench_ok = (
            benchmark is not None
            and benchmark.simulations >= 1000
            and benchmark.monte_carlo_percentile is not None
        )
        if bench_ok:
            bench_ok &= (
                benchmark.match.get("timestamps") == result.timestamps
                and benchmark.match.get("signals_hash") == result.manifest["signals_hash"]
            )
            bench_ok &= all(
                k in benchmark.benchmark_metrics for k in ("SPY", "BTC", "60/40", "equal_weight")
            )
        check(
            "matched_benchmarks",
            bench_ok,
            "Requires SPY/BTC/60-40/equal-weight and 1000 nondegenerate random selections under declared matched replay assumptions.",
        )
        check(
            "regime_dependence",
            bool(provenance.get("regime_evaluation_artifact")),
            "A causal regime-stratified evaluation artifact is required; fold count or IS Sharpe is not evidence.",
        )
        reliable = all(c.verdict == CheckVerdict.PASS for c in checks)
        return ValidationReport(
            checks=checks,
            overall="RELIABLE" if reliable else "NOT RELIABLE",
            summary=f"{sum(c.verdict == CheckVerdict.PASS for c in checks)}/{len(checks)} checks passed. Missing/failed critical evidence prevents promotion.",
        )
