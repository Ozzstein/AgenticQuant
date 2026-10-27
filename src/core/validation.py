"""One evidence-based validation service; missing critical evidence blocks approval."""

from __future__ import annotations

import json
import math
import sqlite3
from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd

from src.core.backtester import (
    ReplayResult,
    canonical_hash,
    load_replay_artifact,
    performance_metrics,
    recorded_income_events,
    recorded_replay_inputs,
    require_matching_inputs,
    verify_decision_trace,
)
from src.domain.common import Sha256, VersionedContract
from src.domain.portfolio import DividendPayment, Fill, exact_decimal_sum
from src.execution.fill_arithmetic import FillArithmetic
from src.execution.ledger import PortfolioLedger
from src.utils.exceptions import BacktestError
from src.utils.schemas import CheckResult, CheckVerdict, ValidationReport


def _reported_fill(trade):
    return Fill.model_validate(
        {key: value for key, value in trade.items() if key in Fill.model_fields}
    )


def _fill_fees(previous, trade):
    fill = _reported_fill(trade)
    result = FillArithmetic(fill).add(previous, fill.fee)
    for fee in fill.inventory_fees:
        result = exact_decimal_sum([result, fee.base_value])
    return result


class AdviceValidationProof(VersionedContract):
    """Trusted verifier output, never constructed from manifest claims alone.

    The injected application verifier must reexport/recompute authoritative
    histories and compare actual replay evidence before returning this contract.
    It is an in-process trust boundary, not a user-supplied approval artifact.
    """

    implementation: Literal["authoritative-advice-v1"] = "authoritative-advice-v1"
    manifest_hash: Sha256
    decision_records_hash: Sha256
    actual_decision_hashes: tuple[Sha256, ...]
    # Sorted unique (decision hash, receipt hash, original review snapshot hash).
    receipt_bindings: tuple[tuple[Sha256, Sha256, Sha256], ...]


class ValidationSuite:
    def __init__(self, config=None):
        self.config = config

    def validate(
        self,
        result,
        *,
        ledger_path=None,
        cost_results=None,
        cost_artifacts=None,
        benchmark=None,
        advice_verifier=None,
        _accounting_only=False,
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
                nav = snapshot.cash + sum((p.market_value for p in snapshot.positions), Decimal(0))
                if snapshot.receivables:
                    nav = exact_decimal_sum([nav, snapshot.receivable_balance])
                require(
                    nav == snapshot.nav,
                    "cash + holdings + nonspendable receivables != NAV",
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
            fees = Decimal(0)
            for trade in result.trades:
                fees = _fill_fees(fees, trade)
            taxes = Decimal(0)
            if result.economic_events:
                # Match the canonical chronological Decimal accumulation order,
                # including deductions that are not a trading fill.
                fees = Decimal(0)
                entries = [(t["executed_at"], 1, t) for t in result.trades]
                entries += [
                    (e["at"], 0, e)
                    for e in result.economic_events
                    if e["kind"] == "dividend_payment"
                ]
                for _, kind, entry in sorted(entries, key=lambda e: (e[0], e[1])):
                    if kind:
                        fees = _fill_fees(fees, entry)
                    else:
                        payment = DividendPayment.model_validate(entry["payload"])
                        fees = exact_decimal_sum([fees, payment.fee])
                        taxes = exact_decimal_sum([taxes, payment.withholding_tax])
            require(taxes == snapshots[-1].withholding_taxes, "withholding tax mismatch")
            require(
                fees == snapshots[-1].fees
                and math.isclose(float(fees), result.metrics.fees, rel_tol=1e-12, abs_tol=1e-12),
                "fee mismatch",
            )
            turnover = 0.0
            for trade in result.trades:
                before = [s for s in snapshots if s.as_of < pd.Timestamp(trade["executed_at"])]
                denominator = before[-1].nav if before else snapshots[0].nav
                turnover += float(_reported_fill(trade).base_notional / denominator)
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
            ledger = None
            try:
                ledger = PortfolioLedger(ledger_path, read_only=True)
                ledger.reconcile(result.account_id)
                inputs = recorded_replay_inputs(ledger, result)
                require_matching_inputs(result, inputs)
                verify_decision_trace(ledger, result, inputs)
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
                require(
                    recorded_income_events(ledger, result.account_id) == result.economic_events,
                    "persisted entitlement/payment journal content differs",
                )
                check(
                    "durable_accounting",
                    True,
                    "Independent journal/state reconciliation, exact immutable manifest/hash binding, complete fill/order content and persisted report agree.",
                )
            except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, BacktestError) as exc:
                check("durable_accounting", False, f"Durable ledger mismatch: {exc}")
            finally:
                if ledger is not None:
                    ledger.close()
        if _accounting_only:
            reliable = all(c.verdict == CheckVerdict.PASS for c in checks)
            return ValidationReport(
                checks=checks,
                overall="RELIABLE" if reliable else "NOT RELIABLE",
                summary="; ".join(c.explanation for c in checks if c.verdict != CheckVerdict.PASS),
            )
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
        advice_requested = provenance.get("advice_policy") is not None
        if advice_requested:
            verified, reason = False, "An injected authoritative history verifier is required."
            try:
                require(
                    provenance["advice_policy"] == "historical-advice-current-ledger-sizing-1",
                    "unsupported advice policy",
                )
                require(
                    not result.walk_forward_results
                    and not result.cross_sectional_ic
                    and not provenance.get("folds")
                    and not provenance.get("oos_signals_artifact"),
                    "model-free advice cannot claim numerical folds, IC or model OOS signals",
                )
                require(
                    ledger_path is not None
                    and any(
                        c.name == "durable_accounting" and c.verdict == CheckVerdict.PASS
                        for c in checks
                    ),
                    "authoritative advice requires independently verified durable replay economics",
                )
                require(callable(advice_verifier), reason)
                proof = advice_verifier(result, ledger_path)
                require(
                    isinstance(proof, AdviceValidationProof),
                    "verifier must return typed AdviceValidationProof",
                )
                # Revalidate rather than trust unchecked model_construct/model_copy.
                proof = AdviceValidationProof.model_validate(proof.model_dump())
                require(
                    proof.manifest_hash == canonical_hash(result.manifest),
                    "advice proof manifest mismatch",
                )
                require(
                    result.decisions
                    and proof.decision_records_hash == result.manifest.get("decision_records_hash"),
                    "advice proof requires exact nonempty replay decision trace",
                )
                hashes = proof.actual_decision_hashes
                require(
                    hashes
                    and tuple(sorted(set(hashes))) == hashes
                    and list(hashes) == provenance.get("actual_decision_hashes"),
                    "advice proof actual decision hashes differ",
                )
                bindings = proof.receipt_bindings
                require(
                    bindings
                    and tuple(sorted(set(bindings))) == bindings
                    and tuple(row[0] for row in bindings) == hashes,
                    "each unique decision requires one exact sorted receipt/snapshot binding",
                )
                require(
                    {row[0]: row[2] for row in bindings}
                    == provenance.get("original_review_snapshots"),
                    "advice proof original review snapshots differ",
                )
                verified, reason = (
                    True,
                    (
                        "Injected authoritative verifier recomputed histories and replay; exact manifest, "
                        "decision trace and sorted receipt/original-snapshot identities agree."
                    ),
                )
            except Exception as exc:
                # External authority failures are failed evidence, never an approval fallback.
                reason = f"Authoritative advice proof failed: {type(exc).__name__}: {exc}"
            check("authoritative_advice_history", verified, reason)
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
        if provenance.get("oos_signals_artifact"):
            check(
                "temporal_training",
                provenance.get("temporal_training") == "contextual-v1"
                and all(
                    isinstance(provenance.get(key), str)
                    and len(provenance[key]) == 64
                    and all(c in "0123456789abcdef" for c in provenance[key])
                    for key in ("feature_spec_hash", "data_snapshot_hash")
                ),
                "Certified walk-forward requires explicit feature/snapshot hashes and contextual training; generic/unversioned diagnostic fits cannot qualify. Actual artifact provenance is independently verified by the research evaluator.",
            )
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
        if not advice_requested:
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
        cost_error = "persisted result/ledger bindings are required"
        costs = []
        matched = False
        try:
            if not cost_artifacts:
                raise ValueError(cost_error)
            costs = [load_replay_artifact(reference) for reference in cost_artifacts]
            if cost_results is not None and {(c.run_id, c.account_id): c for c in costs} != {
                (c.run_id, c.account_id): c for c in cost_results
            }:
                raise ValueError("cost summaries differ from verified replay artifacts")
            from src.core.benchmark import require_matched_replay

            levels = set()
            for candidate in costs:
                require_matched_replay(result, candidate, cost_variant=True)
                levels.add(
                    (
                        Decimal(candidate.manifest["fill_policy"]["equity_cost_bps"]),
                        Decimal(candidate.manifest["fill_policy"]["crypto_cost_bps"]),
                    )
                )
            required_levels = {
                (Decimal(0), Decimal(0)),
                (Decimal(10), Decimal(20)),
                (Decimal(20), Decimal(40)),
            }
            matched = (
                len({(c.account_id, c.run_id) for c in costs}) == len(costs)
                and required_levels <= levels
            )
            cost_error = "all persisted cost runs verified; required paired cost levels " + (
                "present" if matched else "absent/duplicated"
            )
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, BacktestError) as exc:
            cost_error = str(exc)
        check(
            "cost_sensitivity",
            matched,
            "Requires independently reconciled cost replay result and ledger bytes at paired 0/0,10/20,20/40 bps. "
            + cost_error,
        )
        values = [v for v in result.cross_sectional_ic.values() if v is not None]
        ic = float(np.mean(values)) if values else None
        ic_std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        icir = ic / ic_std if ic_std and ic_std > 0 else None
        if not advice_requested:
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
        bench_ok, bench_error = False, "persisted benchmark evidence absent"
        try:
            if benchmark is None:
                raise ValueError(bench_error)
            from src.core.benchmark import verify_comparison

            verify_comparison(result, benchmark, minimum_simulations=1000)
            bench_ok, bench_error = (
                True,
                "All named/random replay bytes, manifests, input events, accounting and summaries verified.",
            )
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, BacktestError) as exc:
            bench_error = str(exc)
        check(
            "matched_benchmarks",
            bench_ok,
            "Requires SPY/BTC/60-40/equal-weight and 1000 nondegenerate seeded selection replays. "
            + bench_error,
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
