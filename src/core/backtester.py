"""Authoritative chronological portfolio replay; no label-derived NAV paths.

Run ranked predictions or explicit optimizer targets through the exact paper
execution/ledger implementation. Backtest namespaces cannot access live venues.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from src.domain.common import RunContext, Sha256, UtcTimestamp, VersionedContract, _utc
from src.domain.portfolio import (
    AccountSpec,
    CorporateAction,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    OrderIntent,
    PortfolioSnapshot,
    PortfolioTarget,
    RankedSignal,
    RiskPolicy,
    TargetWeight,
    UniverseEligibility,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.simulator import PaperExecution
from src.utils.exceptions import BacktestError

D = Decimal

DecisionProvider = Callable[[datetime, PortfolioSnapshot], PortfolioTarget | None]


class ReplayDecision(VersionedContract):
    """Actual callback input/output, including deliberate abstentions."""

    at: UtcTimestamp
    recipe_hash: Sha256
    snapshot: PortfolioSnapshot
    snapshot_hash: Sha256
    target: PortfolioTarget | None

    @model_validator(mode="after")
    def coherent(self):
        if self.snapshot.as_of != self.at or self.snapshot.content_hash() != self.snapshot_hash:
            raise ValueError("decision snapshot clock/hash mismatch")
        if self.target is not None and self.target.decision_at != self.at:
            raise ValueError("generated target must use the actual decision clock")
        return self


class ReplayMetrics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    total_return: float
    annual_return: float | None
    sharpe_ratio: float | None
    max_drawdown: float
    volatility: float | None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    win_rate: float | None = None
    total_trades: int = 0
    turnover: float = 0.0  # one-way traded base notional / pre-trade NAV, summed
    fees: float = 0.0


class FoldEvidence(BaseModel):
    fold_id: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    train_rows: int
    validation_rows: int
    test_rows: int
    purged_rows: int
    embargo_days: int
    model_version: str
    train_hash: str
    validation_hash: str
    predictions_hash: str
    label_end_max: str
    validation_label_end_max: str
    oos_sharpe: float | None = None
    ic: float | None = None
    feature_importance: dict[str, float] = Field(default_factory=dict)


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["2.0"] = "2.0"
    engine: Literal["chronological-ledger-v1"] = "chronological-ledger-v1"
    status: Literal["complete", "incomplete", "failed"]
    run_id: str
    account_id: str
    manifest: dict
    metrics: ReplayMetrics
    daily_returns: list[float]
    timestamps: list[str]  # includes initial cash observation, as does equity_curve
    equity_curve: list[float]
    snapshots: list[PortfolioSnapshot]
    trades: list[dict]
    decisions: list[dict]
    walk_forward_results: list[FoldEvidence] = Field(default_factory=list)
    cross_sectional_ic: dict[str, float | None] = Field(default_factory=dict)
    reconciliation: dict
    diagnostics: list[str] = Field(default_factory=list)


class ReplayArtifact(BaseModel):
    """External immutable binding; kept outside ReplayResult to avoid self-hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    result_path: str
    result_sha256: Sha256
    ledger_path: str
    ledger_sha256: Sha256
    run_id: str
    account_id: str
    manifest_hash: Sha256


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_text() != content:
                    raise BacktestError(f"immutable artifact content changed: {path}")
        finally:
            temporary.unlink(missing_ok=True)


def persist_replay_artifact(result: ReplayResult, ledger_path, result_path) -> ReplayArtifact:
    ledger_path, result_path = Path(ledger_path).resolve(), Path(result_path).resolve()
    wal = Path(str(ledger_path) + "-wal")
    if not ledger_path.is_file() or (wal.exists() and wal.stat().st_size):
        raise BacktestError("artifact requires a closed, checkpointed persistent ledger")
    write_immutable(
        result_path, json.dumps(result.model_dump(mode="json"), sort_keys=True, indent=2)
    )
    return ReplayArtifact(
        result_path=str(result_path),
        result_sha256=file_hash(result_path),
        ledger_path=str(ledger_path),
        ledger_sha256=file_hash(ledger_path),
        run_id=result.run_id,
        account_id=result.account_id,
        manifest_hash=canonical_hash(result.manifest),
    )


def load_replay_artifact(reference: ReplayArtifact, *, verify_accounting=True) -> ReplayResult:
    """Load bytes, verify hashes and independently reconcile persisted economics."""
    reference = ReplayArtifact.model_validate(reference)
    if (
        file_hash(reference.result_path) != reference.result_sha256
        or file_hash(reference.ledger_path) != reference.ledger_sha256
    ):
        raise BacktestError("replay artifact bytes differ from their binding")
    wal = Path(reference.ledger_path + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise BacktestError("replay artifact has uncheckpointed writes")
    result = ReplayResult.model_validate_json(Path(reference.result_path).read_text())
    if (result.run_id, result.account_id, canonical_hash(result.manifest)) != (
        reference.run_id,
        reference.account_id,
        reference.manifest_hash,
    ):
        raise BacktestError("replay artifact identity/manifest mismatch")
    if verify_accounting:
        from src.core.validation import ValidationSuite

        report = ValidationSuite().validate(
            result, ledger_path=reference.ledger_path, _accounting_only=True
        )
        if report.overall != "RELIABLE":
            raise BacktestError("artifact accounting verification failed: " + report.summary)
    if file_hash(reference.ledger_path) != reference.ledger_sha256:
        raise BacktestError("replay ledger changed during verification")
    return result


def recorded_replay_inputs(ledger: PortfolioLedger, result: ReplayResult) -> dict:
    """Recover actual input events, never treating manifest hash strings as data."""
    rows = list(
        ledger.db.execute(
            "SELECT id,payload FROM events WHERE account=? ORDER BY rowid", (result.account_id,)
        )
    )

    def records(prefix, contract):
        return [
            contract.model_validate_json(row["payload"])
            for row in rows
            if row["id"].startswith(prefix)
        ]

    manifest = result.manifest
    inputs = {
        "context": RunContext.model_validate(manifest["context"]),
        "account": ledger.spec(result.account_id),
        "bars": records("open:", ExecutionBar),
        "signals": records("input:signal:", RankedSignal),
        "targets": records("input:target:", PortfolioTarget),
        "metadata": records("input:metadata:", InstrumentRisk),
        "universes": records("input:universe:", UniverseEligibility),
        "actions": records("input:action:", CorporateAction),
        "fill_policy": FillPolicy.model_validate(manifest["fill_policy"]),
        "risk_policy": RiskPolicy.model_validate(manifest["risk_policy"]),
        "topk": manifest["topk"],
        "samples_per_year": manifest["samples_per_year"],
        "valuation_calendar": manifest["valuation_calendar"],
        "valuation_times": [datetime.fromisoformat(t) for t in manifest["valuation_times"]],
        "provenance": manifest.get("provenance", {}),
    }
    if "decision_recipe_hash" in manifest:
        inputs.update(
            decision_recipe_hash=manifest["decision_recipe_hash"],
            decision_times=[datetime.fromisoformat(t) for t in manifest["decision_times"]],
        )
    return inputs


def replay_request_manifest(manifest: dict) -> dict:
    """Frozen recipe before outputs exist; completion cannot change its inputs."""
    request = dict(manifest)
    if "decision_recipe_hash" in request:
        request.pop("decision_records_hash", None)
        request["targets_hash"] = canonical_hash([])
    return request


def recorded_decisions(ledger: PortfolioLedger, result: ReplayResult) -> list[ReplayDecision]:
    records = [
        ReplayDecision.model_validate_json(row[0])
        for row in ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id LIKE 'provider-decision:%' ORDER BY id",
            (result.account_id,),
        )
    ]
    if "decision_recipe_hash" not in result.manifest:
        if records:
            raise BacktestError("callback evidence on a precomputed replay")
        return []
    if (
        [r.at.isoformat() for r in records] != result.manifest["decision_times"]
        or any(r.recipe_hash != result.manifest["decision_recipe_hash"] for r in records)
        or canonical_hash([r.model_dump(mode="json") for r in records])
        != result.manifest.get("decision_records_hash")
    ):
        raise BacktestError("decision trace differs from frozen schedule/recipe/output binding")
    request = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id='replay-request'", (result.account_id,)
    ).fetchone()
    if request is None or json.loads(request[0]) != replay_request_manifest(result.manifest):
        raise BacktestError("generated run differs from its immutable original request")
    return records


def verify_decision_trace(ledger: PortfolioLedger, result: ReplayResult, inputs: dict) -> None:
    records = recorded_decisions(ledger, result)
    if "decision_recipe_hash" not in result.manifest:
        return
    if [r.target for r in records if r.target is not None] != inputs["targets"]:
        raise BacktestError("generated target events differ from recorded callback outputs")
    iterator = iter(records)

    def checked_provider(at, snapshot):
        record = next(iterator)
        if record.at != at or record.snapshot != snapshot:
            raise BacktestError("callback snapshot differs from actual causal replay state")
        return record.target

    reproduced = PortfolioReplay(":memory:").run(
        **{**inputs, "targets": None, "decision_provider": checked_provider}
    )
    if reproduced.model_dump(mode="json") != result.model_dump(mode="json"):
        raise BacktestError("recorded decision trace does not reproduce exact replay economics")


def reconstruct_decision_inputs(
    result: ReplayResult,
    original_replay_kwargs: dict,
    *,
    ledger_path: str | Path,
    decision_provider: DecisionProvider,
) -> dict:
    """Rebind the verified recipe for new cost/holdings-dependent evaluations.

    The caller supplies the actual recipe implementation; code/model artifact
    authorization remains the research registry's responsibility. Reusing old
    targets at different costs would not evaluate the holdings-aware strategy.
    """
    if "decision_recipe_hash" not in result.manifest or not callable(decision_provider):
        raise BacktestError("a generated replay and actual decision provider are required")
    ledger = PortfolioLedger(ledger_path, read_only=True)
    try:
        inputs = recorded_replay_inputs(ledger, result)
        require_matching_inputs(result, inputs)
        verify_decision_trace(ledger, result, inputs)
    finally:
        ledger.close()
    supplied = {**original_replay_kwargs, "decision_provider": decision_provider}
    require_matching_inputs(result, supplied)
    return {**supplied, "signals": None, "targets": None}


def require_matching_inputs(result: ReplayResult, inputs: dict) -> None:
    """Bind input content and conventions, including clocks, actions and costs."""
    expected = {
        "bars_hash": canonical_hash(
            [
                b.model_dump(mode="json")
                for b in sorted(
                    inputs["bars"], key=lambda b: (b.opened_at, b.instrument.instrument_id)
                )
            ]
        ),
        "signals_hash": canonical_hash(
            [
                s.model_dump(mode="json")
                for s in sorted(inputs.get("signals") or [], key=lambda s: s.signal_id)
            ]
        ),
        "targets_hash": canonical_hash(
            [t.model_dump(mode="json") for t in inputs.get("targets") or []]
        ),
        "metadata_hash": canonical_hash([m.model_dump(mode="json") for m in inputs["metadata"]]),
        "universe_hash": canonical_hash([u.model_dump(mode="json") for u in inputs["universes"]]),
        "actions_hash": canonical_hash(
            [a.model_dump(mode="json") for a in inputs.get("actions") or []]
        ),
        "account": inputs["account"].model_dump(mode="json"),
        "context": inputs["context"].model_dump(mode="json"),
        "fill_policy": (inputs.get("fill_policy") or FillPolicy()).model_dump(mode="json"),
        "risk_policy": (inputs.get("risk_policy") or RiskPolicy()).model_dump(mode="json"),
        "topk": inputs.get("topk", 30),
        "samples_per_year": inputs.get("samples_per_year", 252),
        "valuation_calendar": inputs.get("valuation_calendar", "declared-equity-sessions"),
    }
    times = inputs.get("valuation_times")
    if times is None:
        times = sorted({b.closed_at for b in inputs["bars"]})
    expected["valuation_times"] = [t.isoformat() for t in times]
    if "decision_recipe_hash" in result.manifest:
        expected["decision_recipe_hash"] = inputs.get("decision_recipe_hash")
        expected["decision_times"] = [t.isoformat() for t in inputs.get("decision_times") or []]
        # A live recipe's outputs depend on actual holdings. Its input contract
        # freezes the recipe/schedule; durable trace verification binds outputs.
        if callable(inputs.get("decision_provider")):
            expected.pop("targets_hash")
    changed = [key for key, value in expected.items() if result.manifest.get(key) != value]
    if changed:
        raise BacktestError("replay inputs differ from frozen manifest: " + ", ".join(changed))


def performance_metrics(nav: list[float], samples_per_year: int) -> ReplayMetrics:
    if len(nav) < 2 or samples_per_year <= 0 or not np.isfinite(nav).all() or min(nav) <= 0:
        raise BacktestError(
            "at least two positive finite NAV observations and declared annualization required"
        )
    values = np.asarray(nav, dtype=float)
    returns = values[1:] / values[:-1] - 1
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else None
    vol = std * math.sqrt(samples_per_year) if std is not None else None
    total = float(values[-1] / values[0] - 1)
    exponent = math.log(values[-1] / values[0]) * samples_per_year / len(returns)
    annual = math.expm1(exponent) if abs(exponent) < 700 else None
    drawdown = float(np.min(values / np.maximum.accumulate(values) - 1))
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    return ReplayMetrics(
        total_return=total,
        annual_return=annual,
        sharpe_ratio=float(returns.mean() / std * math.sqrt(samples_per_year))
        if std and std > 0
        else None,
        max_drawdown=drawdown,
        volatility=vol,
        sortino_ratio=float(returns.mean() / downside * math.sqrt(samples_per_year))
        if downside > 0
        else None,
        calmar_ratio=annual / abs(drawdown) if annual is not None and drawdown < 0 else None,
        win_rate=float(np.mean(returns > 0)),
    )


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class PortfolioReplay:
    def __init__(self, ledger_path: str | Path):
        self.ledger_path = ledger_path

    def run(
        self,
        *,
        context: RunContext,
        account: AccountSpec,
        bars: list[ExecutionBar],
        metadata: list[InstrumentRisk],
        universes: list[UniverseEligibility],
        signals: list[RankedSignal] | None = None,
        targets: list[PortfolioTarget] | None = None,
        actions: list[CorporateAction] | None = None,
        topk: int = 30,
        fill_policy: FillPolicy | None = None,
        risk_policy: RiskPolicy | None = None,
        samples_per_year: int = 252,
        valuation_calendar: str = "declared-equity-sessions",
        valuation_times: list[datetime] | None = None,
        provenance: dict | None = None,
        decision_provider: DecisionProvider | None = None,
        decision_times: list[datetime] | None = None,
        decision_recipe_hash: str | None = None,
    ) -> ReplayResult:
        if (
            context.mode != "backtest"
            or account.mode != "backtest"
            or context.account_id != account.account_id
        ):
            raise BacktestError("replay requires an isolated backtest account matching RunContext")
        if (
            not bars
            or not metadata
            or not universes
            or sum((bool(signals), bool(targets), decision_provider is not None)) != 1
        ):
            raise BacktestError(
                "actual bars, risk metadata, documented universe eligibility and exactly one signal/target/provider stream are required"
            )
        generated = decision_provider is not None
        if generated:
            if not callable(decision_provider) or not decision_times:
                raise BacktestError("decision provider requires an explicit nonempty UTC schedule")
            decision_recipe_hash = TypeAdapter(Sha256).validate_python(decision_recipe_hash)
            decision_times = [_utc(t) for t in decision_times]
            if decision_times != sorted(set(decision_times)) or any(
                t < account.opened_at or t > context.as_of for t in decision_times
            ):
                raise BacktestError(
                    "decision schedule must be unique, increasing and within run bounds"
                )
        elif decision_times is not None or decision_recipe_hash is not None:
            raise BacktestError("decision schedule/recipe requires the actual decision provider")
        if topk < 1 or samples_per_year <= 0:
            raise BacktestError("topk and annualization must be positive")
        signals, targets, actions = signals or [], targets or [], actions or []
        policy, risk = fill_policy or FillPolicy(), risk_policy or RiskPolicy()
        bars = sorted(bars, key=lambda b: (b.opened_at, b.instrument.instrument_id))
        if len({b.event_id for b in bars}) != len(bars):
            raise BacktestError("duplicate market event IDs")
        per_instrument = defaultdict(list)
        for b in bars:
            per_instrument[b.instrument.instrument_id].append(b)
        for events in per_instrument.values():
            if any(a.closed_at > b.opened_at for a, b in zip(events, events[1:])):
                raise BacktestError("overlapping instrument bars")
        times = valuation_times or sorted({b.closed_at for b in bars})
        if times != sorted(set(times)) or any(
            t <= account.opened_at or t > context.as_of for t in times
        ):
            raise BacktestError(
                "valuation schedule must be unique, increasing and inside run bounds"
            )
        if len({t.date() for t in times}) != len(times):
            raise BacktestError(
                "daily metrics require an explicit one-observation-per-day valuation schedule"
            )
        if account.opened_at > bars[0].opened_at or max(b.closed_at for b in bars) > context.as_of:
            raise BacktestError("market data outside account/run clock")
        identities = {}
        for b in bars:
            old = identities.setdefault(b.instrument.instrument_id, b.instrument)
            if old != b.instrument:
                raise BacktestError("conflicting instrument identities")
        universe_keys = [(u.universe_version, u.effective_at, u.available_at) for u in universes]
        if len(set(universe_keys)) != len(universe_keys):
            raise BacktestError("duplicate/ambiguous universe eligibility snapshot")
        signal_groups = defaultdict(list)
        for signal in signals:
            signal_groups[signal.decision_at].append(signal)
        if len({s.signal_id for s in signals}) != len(signals):
            raise BacktestError("duplicate signal IDs")
        target_groups = {t.decision_at: t for t in targets}
        if len(target_groups) != len(targets):
            raise BacktestError(
                "multiple targets at one decision instant must be netted before replay"
            )

        def eligible(at, version):
            candidates = [
                u
                for u in universes
                if u.universe_version == version
                and u.effective_at <= at
                and u.available_at <= at
                and u.data_snapshot_id in context.data_snapshot_ids
            ]
            if not candidates:
                raise BacktestError(
                    "no documented universe membership available at decision cutoff"
                )
            return set(
                max(candidates, key=lambda u: (u.effective_at, u.available_at)).instrument_ids
            )

        for at, group in signal_groups.items():
            if len({s.instrument.instrument_id for s in group}) != len(group):
                raise BacktestError("duplicate instrument predictions at a decision instant")
            if len({s.universe_version for s in group}) != 1:
                raise BacktestError("rankings at one date require a coherent universe version")
            if len({s.strategy_id for s in group}) != 1:
                raise BacktestError(
                    "net multiple strategy sleeves into an explicit PortfolioTarget"
                )
            for s in group:
                if s.instrument.instrument_id not in eligible(at, s.universe_version):
                    raise BacktestError("prediction instrument was outside point-in-time universe")
                if (
                    s.data_snapshot_id not in context.data_snapshot_ids
                    or identities.get(s.instrument.instrument_id) != s.instrument
                ):
                    raise BacktestError("signal instrument/data snapshot provenance mismatch")
                if not any(
                    b.opened_at > at and b.tradable
                    for b in per_instrument[s.instrument.instrument_id]
                ):
                    raise BacktestError("signal has no subsequent eligible observed open")

        def require_target(t):
            if t.data_snapshot_id not in context.data_snapshot_ids:
                raise BacktestError("target data snapshot provenance mismatch")
            available_members = eligible(t.decision_at, t.universe_version)
            for w in t.weights:
                if w.weight > 0 and w.instrument.instrument_id not in available_members:
                    raise BacktestError("target instrument outside documented eligible universe")
                if identities.get(w.instrument.instrument_id) != w.instrument:
                    raise BacktestError("target instrument is absent from execution data")

        for t in targets:
            require_target(t)
        manifest = {
            "context": context.model_dump(mode="json"),
            "account": account.model_dump(mode="json"),
            "fill_policy": policy.model_dump(mode="json"),
            "risk_policy": risk.model_dump(mode="json"),
            "topk": topk,
            "samples_per_year": samples_per_year,
            "valuation_calendar": valuation_calendar,
            "valuation_times": [t.isoformat() for t in times],
            "cash_return": "zero",
            "cost_convention": "one-way charged once per fill",
            "bars_hash": canonical_hash([b.model_dump(mode="json") for b in bars]),
            "signals_hash": canonical_hash(
                [s.model_dump(mode="json") for s in sorted(signals, key=lambda s: s.signal_id)]
            ),
            "targets_hash": canonical_hash([t.model_dump(mode="json") for t in targets]),
            "metadata_hash": canonical_hash([m.model_dump(mode="json") for m in metadata]),
            "universe_hash": canonical_hash([u.model_dump(mode="json") for u in universes]),
            "actions_hash": canonical_hash([a.model_dump(mode="json") for a in actions]),
            "provenance": provenance or {},
        }
        if generated:
            manifest.update(
                decision_recipe_hash=decision_recipe_hash,
                decision_times=[at.isoformat() for at in decision_times],
            )
        manifest_hash = canonical_hash(manifest)
        request_manifest, request_hash = dict(manifest), manifest_hash
        # Finished evidence is immutable, including SQLite file bytes. An
        # idempotent rerun must not rewrite run rows or invalidate artifact hashes.
        if self.ledger_path != ":memory:" and Path(self.ledger_path).is_file():
            existing = PortfolioLedger(self.ledger_path, read_only=True)
            try:
                previous = existing.db.execute(
                    "SELECT payload FROM runs WHERE account=? AND id=?",
                    (account.account_id, context.run_id),
                ).fetchone()
                if previous:
                    payload = json.loads(previous[0])
                    if canonical_hash(replay_request_manifest(payload["manifest"])) != request_hash:
                        raise BacktestError("run inputs changed; use a new isolated account/run")
                    if payload.get("status") in ("complete", "incomplete") and payload.get(
                        "result"
                    ):
                        saved = ReplayResult.model_validate(payload["result"])
                        existing.reconcile(account.account_id)
                        require_matching_inputs(saved, recorded_replay_inputs(existing, saved))
                        from src.core.validation import ValidationSuite

                        report = ValidationSuite().validate(
                            saved, ledger_path=self.ledger_path, _accounting_only=True
                        )
                        if any(
                            c.name != "complete_run" and c.verdict.value != "PASS"
                            for c in report.checks
                        ):
                            raise BacktestError(
                                "stored replay failed accounting verification: " + report.summary
                            )
                        return saved
            finally:
                existing.close()
        ledger = PortfolioLedger(self.ledger_path)
        try:
            ledger.create_account(account)
            prior = ledger.db.execute(
                "SELECT payload FROM runs WHERE account=? AND id=?",
                (account.account_id, context.run_id),
            ).fetchone()
            if (
                prior
                and canonical_hash(replay_request_manifest(json.loads(prior[0])["manifest"]))
                != request_hash
            ):
                raise BacktestError("run inputs changed; use a new isolated account/run")
            other = ledger.db.execute(
                "SELECT id FROM runs WHERE account=? AND id<>?",
                (account.account_id, context.run_id),
            ).fetchone()
            if other:
                raise BacktestError("historical accounts are exclusive to one reproducible run")
            # Generated outputs cannot be committed before they exist. Freeze
            # the full request separately, then write the final run once. This
            # preserves save_run's immutable-manifest rule without exceptions.
            if not generated:
                ledger.save_run(
                    account.account_id,
                    context.run_id,
                    {"status": "running", "manifest_hash": manifest_hash, "manifest": manifest},
                )
            with ledger.atomic():
                ledger.record_event(
                    account.account_id,
                    "replay-request",
                    json.dumps(request_manifest, sort_keys=True),
                )
                for kind, records in (
                    ("signal", signals),
                    ("target", targets),
                    ("metadata", metadata),
                    ("universe", universes),
                    ("action", actions),
                ):
                    for record in records:
                        ledger.record_event(
                            account.account_id,
                            f"input:{kind}:" + record.content_hash(),
                            record.model_dump_json(),
                        )
            engine = PaperExecution(ledger, account.account_id, policy, risk)
            if not ledger.history(account.account_id):
                ledger.snapshot(account.account_id, account.opened_at, risk.max_price_age_seconds)
            timeline = defaultdict(lambda: defaultdict(list))
            for b in bars:
                timeline[b.opened_at]["opens"].append(b)
                timeline[b.closed_at]["closes"].append(b)
            for a in actions:
                timeline[a.effective_at]["actions"].append(a)
            for t in times:
                timeline[t]["valuation"] = True
            for at, group in signal_groups.items():
                timeline[at]["signals"] = group
            for at, target in target_groups.items():
                timeline[at]["target"] = target
            for at in decision_times or []:
                timeline[at]["provider"] = True
            for at, event in sorted(timeline.items()):
                if at < account.opened_at or at > context.as_of:
                    raise BacktestError("event outside account/run clock")
                with ledger.atomic():
                    if not ledger.record_event(
                        account.account_id, "step:" + at.isoformat(), manifest_hash
                    ):
                        continue
                    meta = self._metadata_as_of(metadata, at)
                    # End of preceding interval precedes next interval's open at the
                    # same boundary. Signals at that boundary cannot fill that open.
                    for b in event["closes"]:
                        engine.close_bar(b)
                    for a in event["actions"]:
                        ledger.corporate_action(account.account_id, a)
                    # Mark all simultaneous opens before evaluating portfolio risk.
                    from src.domain.portfolio import PriceMark

                    if event["opens"]:
                        ledger.mark(
                            account.account_id,
                            [
                                PriceMark(
                                    instrument=b.instrument,
                                    price=b.open,
                                    fx_rate=b.fx_rate,
                                    as_of=at,
                                )
                                for b in event["opens"]
                            ],
                        )
                    for b in event["opens"]:
                        engine.open_bar(b, meta)
                    if event["provider"]:
                        snapshot = ledger.snapshot(
                            account.account_id, at, risk.max_price_age_seconds, persist=False
                        )
                        target = decision_provider(at, snapshot)
                        if target is not None:
                            # model_copy can bypass Pydantic validation; validate
                            # the boundary rather than trusting the callback.
                            target = PortfolioTarget.model_validate(target.model_dump())
                            require_target(target)
                        record = ReplayDecision(
                            at=at,
                            recipe_hash=decision_recipe_hash,
                            snapshot=snapshot,
                            snapshot_hash=snapshot.content_hash(),
                            target=target,
                        )
                        ledger.record_event(
                            account.account_id,
                            "provider-decision:" + at.isoformat(),
                            record.model_dump_json(),
                        )
                        if target is not None:
                            ledger.record_event(
                                account.account_id,
                                "input:target:" + target.content_hash(),
                                target.model_dump_json(),
                            )
                            event["target"] = target
                    if event["signals"] or event["target"]:
                        target = event["target"] or self._rank_target(event["signals"], topk, risk)
                        self._rebalance(engine, target, meta, identities, per_instrument)
                    if event["valuation"]:
                        ledger.snapshot(account.account_id, at, risk.max_price_age_seconds)
            # The evaluation ends with marked holdings; unfilled remainders expire,
            # without fabricated liquidation proceeds or resetting fold cash.
            with ledger.atomic():
                for pending in ledger.pending(account.account_id):
                    ledger.cancel(
                        account.account_id,
                        pending["intent"].intent_id,
                        context.as_of,
                        "evaluation cutoff",
                    )
            reconciliation = ledger.reconcile(account.account_id)
            snapshots = ledger.history(account.account_id)
            nav = [float(s.nav) for s in snapshots]
            metrics = performance_metrics(nav, samples_per_year)
            fills = ledger.fills(account.account_id)
            trades = []
            turnover = 0.0
            for f in fills:
                order = ledger.order(account.account_id, f.intent_id)["intent"]
                previous = [s for s in snapshots if s.as_of < f.executed_at]
                denominator = previous[-1].nav if previous else account.initial_cash
                turnover += float(f.quantity * f.price * f.fx_rate / denominator)
                economic = ledger.db.execute(
                    "SELECT cash_delta,realized,notional FROM fills WHERE account=? AND id=?",
                    (account.account_id, f.fill_id),
                ).fetchone()
                trades.append(
                    {
                        "cash_delta": economic["cash_delta"],
                        "realized_pnl": economic["realized"],
                        "notional": economic["notional"],
                        **f.model_dump(mode="json"),
                        "instrument_id": order.instrument.instrument_id,
                        "side": order.side,
                        "strategy_id": order.strategy_id,
                        "decision_at": order.decision_at.isoformat(),
                    }
                )
            metrics = metrics.model_copy(
                update={
                    "total_trades": len(fills),
                    "fees": float(snapshots[-1].fees),
                    "turnover": turnover,
                }
            )
            decisions = [
                json.loads(r[0])
                for r in ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id LIKE 'decision:%' ORDER BY rowid",
                    (account.account_id,),
                )
            ]
            if generated:
                records = [
                    ReplayDecision.model_validate_json(row[0])
                    for row in ledger.db.execute(
                        "SELECT payload FROM events WHERE account=? AND id LIKE 'provider-decision:%' ORDER BY id",
                        (account.account_id,),
                    )
                ]
                manifest = {
                    **request_manifest,
                    "targets_hash": canonical_hash(
                        [r.target.model_dump(mode="json") for r in records if r.target is not None]
                    ),
                    "decision_records_hash": canonical_hash(
                        [r.model_dump(mode="json") for r in records]
                    ),
                }
                manifest_hash = canonical_hash(manifest)
            result = ReplayResult(
                status="complete",
                run_id=context.run_id,
                account_id=account.account_id,
                manifest=manifest,
                metrics=metrics,
                daily_returns=[b / a - 1 for a, b in zip(nav, nav[1:])],
                timestamps=[s.as_of.isoformat() for s in snapshots],
                equity_curve=nav,
                snapshots=snapshots,
                trades=trades,
                decisions=decisions,
                reconciliation=reconciliation,
                diagnostics=[
                    f"{sum(o['outcome'].startswith('rejected') for d in decisions for o in d['orders'])} order intents rejected by deterministic risk; inspect decisions"
                ]
                if any(o["outcome"].startswith("rejected") for d in decisions for o in d["orders"])
                else [],
            )
            ledger.save_run(
                account.account_id,
                context.run_id,
                {
                    "status": "complete",
                    "manifest_hash": manifest_hash,
                    "manifest": manifest,
                    "result": result.model_dump(mode="json"),
                },
            )
            return result
        except Exception as exc:
            if (
                generated
                and ledger.db.execute(
                    "SELECT id FROM accounts WHERE id=?", (account.account_id,)
                ).fetchone()
            ):
                failure = {"request_hash": request_hash, "error": str(exc), "status": "failed"}
                with ledger.atomic():
                    ledger.record_event(
                        account.account_id,
                        "run-failure:" + canonical_hash(failure),
                        json.dumps(failure, sort_keys=True),
                    )
            if (
                "manifest_hash" in locals()
                and ledger.db.execute(
                    "SELECT id FROM accounts WHERE id=?", (account.account_id,)
                ).fetchone()
            ):
                # A conflicting retry must not overwrite the original run evidence.
                current = ledger.db.execute(
                    "SELECT payload FROM runs WHERE account=? AND id=?",
                    (account.account_id, context.run_id),
                ).fetchone()
                if current and json.loads(current[0]).get("manifest_hash") == manifest_hash:
                    ledger.save_run(
                        account.account_id,
                        context.run_id,
                        {
                            "status": "failed",
                            "manifest_hash": manifest_hash,
                            "manifest": manifest,
                            "error": str(exc),
                        },
                    )
            if isinstance(exc, BacktestError):
                raise
            raise BacktestError(f"chronological replay failed: {exc}") from exc
        finally:
            ledger.close()

    @staticmethod
    def _metadata_as_of(metadata, at):
        result = {}
        for m in sorted(metadata, key=lambda m: m.as_of):
            if m.as_of <= at:
                result[m.instrument.instrument_id] = m
        return result

    @staticmethod
    def _rank_target(signals, topk, risk):
        ranked = sorted(signals, key=lambda s: (-s.score, s.instrument.instrument_id))[:topk]
        if not ranked:
            raise BacktestError("empty date-local ranking")
        weights = tuple(
            TargetWeight(
                instrument=s.instrument,
                weight=min(
                    (1 - risk.min_cash) / len(ranked),
                    risk.crypto_weight
                    if s.instrument.asset_class == "crypto"
                    else risk.equity_weight,
                ),
            )
            for s in ranked
        )
        return PortfolioTarget(
            universe_version=ranked[0].universe_version,
            target_id="rank:" + canonical_hash([s.content_hash() for s in ranked]),
            decision_at=ranked[0].decision_at,
            strategy_id=ranked[0].strategy_id,
            artifact_version=canonical_hash([s.model_version for s in ranked]),
            data_snapshot_id=ranked[0].data_snapshot_id,
            weights=weights,
        )

    @staticmethod
    def _rebalance(engine, target, metadata, identities, bars):
        ledger, account = engine.ledger, engine.account_id
        at = target.decision_at
        # A new complete target supersedes all old unfilled rebalance intents.
        for pending in ledger.pending(account):
            ledger.cancel(account, pending["intent"].intent_id, at, "superseded target")
        snap = ledger.snapshot(account, at, engine.risk_policy.max_price_age_seconds, persist=False)
        held = {p.instrument.instrument_id: p.quantity for p in snap.positions}
        weights = {w.instrument.instrument_id: w.weight for w in target.weights}
        candidates, diagnostics = [], []
        for key in sorted(set(held) | set(weights)):
            mark_row = ledger.db.execute(
                "SELECT payload FROM marks WHERE account=? AND instrument=?", (account, key)
            ).fetchone()
            if mark_row is None or key not in metadata:
                raise BacktestError(f"target lacks observed price or risk metadata: {key}")
            from src.domain.portfolio import PriceMark

            mark = PriceMark.model_validate_json(mark_row[0])
            if (
                not 0
                <= (at - mark.as_of).total_seconds()
                <= engine.risk_policy.max_price_age_seconds
            ):
                raise BacktestError(f"target uses stale/future price: {key}")
            meta = metadata[key]
            unit = mark.price * mark.fx_rate
            desired = (snap.nav * weights.get(key, D(0)) / unit / meta.lot_size).to_integral_value(
                rounding=ROUND_DOWN
            ) * meta.lot_size
            delta = desired - held.get(key, D(0))
            if delta == 0:
                continue
            if not any(b.opened_at > at and b.tradable for b in bars[key]):
                raise BacktestError(f"target has no subsequent executable event: {key}")
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            # Leave room for explicitly modeled fees/slippage at allocation boundary.
            price = mark.price * (
                1 + engine.policy.slippage_bps / 10000 * (1 if side == "buy" else -1)
            )
            if side == "buy":
                bps = (
                    engine.policy.crypto_cost_bps
                    if identities[key].asset_class == "crypto"
                    else engine.policy.equity_cost_bps
                )
                budget = snap.nav * weights.get(key, D(0))
                qty = max(
                    D(0),
                    (
                        (budget - engine.policy.minimum_commission)
                        / (
                            price * mark.fx_rate * (1 + bps / 10000)
                            + engine.policy.commission_per_unit
                        )
                        - held.get(key, D(0))
                    )
                    / meta.lot_size,
                )
                qty = qty.to_integral_value(rounding=ROUND_DOWN) * meta.lot_size
            if qty <= 0:
                continue
            intent = OrderIntent(
                intent_id=target.target_id + ":" + key,
                account_id=account,
                run_id=target.target_id,
                instrument=identities[key],
                side=side,
                quantity=qty,
                decision_at=at,
                eligible_at=at + timedelta(microseconds=1),
                reference_price=price,
                fx_rate=mark.fx_rate,
                estimated_fee=engine.policy.fee(identities[key], qty, price, mark.fx_rate),
                strategy_id=target.strategy_id,
            )
            candidates.append(intent)
        for order in sorted(
            candidates, key=lambda o: (o.side != "sell", o.instrument.instrument_id)
        ):
            try:
                engine.submit(order, metadata)
                outcome = "submitted"
            except LedgerError as exc:
                outcome = "rejected: " + str(exc)
            diagnostics.append({"intent": order.model_dump(mode="json"), "outcome": outcome})
        ledger.record_event(
            account,
            "decision:" + target.target_id,
            json.dumps(
                {"target": target.model_dump(mode="json"), "orders": diagnostics}, sort_keys=True
            ),
        )


def split_folds(
    dates: pd.DatetimeIndex,
    train_start: str,
    walk_forward_months: int = 6,
    embargo_days: int = 5,
    validation_months: int = 1,
    initial_train_months: int = 12,
) -> list[dict]:
    """Calendar boundaries; row-level label purging is performed by the runner."""
    if (
        walk_forward_months <= 0
        or validation_months <= 0
        or initial_train_months <= 0
        or embargo_days < 0
    ):
        raise BacktestError("invalid walk-forward split configuration")
    unique = dates.sort_values().unique()
    if unique.empty:
        return []
    start = pd.Timestamp(train_start)
    if unique.tz is not None and start.tzinfo is None:
        start = start.tz_localize(unique.tz)
    start = max(start, unique[0])
    current = (
        start
        + pd.DateOffset(months=initial_train_months + validation_months)
        + pd.Timedelta(days=embargo_days)
    )
    folds = []
    while current <= unique[-1]:
        end = current + pd.DateOffset(months=walk_forward_months)
        validation_end = current - pd.Timedelta(days=embargo_days)
        validation_start = validation_end - pd.DateOffset(months=validation_months)
        train_end = validation_start - pd.Timedelta(days=embargo_days)
        folds.append(
            {
                "train_start": start.isoformat(),
                "train_end": train_end.isoformat(),
                "validation_start": validation_start.isoformat(),
                "validation_end": validation_end.isoformat(),
                "test_start": current.isoformat(),
                "test_end": end.isoformat(),
                "complete_window": end <= unique[-1] + pd.Timedelta(days=1),
            }
        )
        current = end
    return folds


class WalkForwardBacktester:
    """Compatibility entry point delegating to the one chronological runner.

    Old label-only calls fail: forward labels cannot manufacture executable
    price bars or NAV. Pass bars, metadata, context, account and ledger_path.
    """

    def __init__(self, config=None):
        self.config = config

    def run(
        self,
        features,
        labels,
        model=None,
        topk=30,
        transaction_cost_bps=10.0,
        walk_forward_months=6,
        embargo_days=5,
        **kwargs,
    ):
        from src.core.walk_forward import WalkForwardRunner

        if model is not None and "model_factory" not in kwargs:
            raise BacktestError("supply model_factory for a fresh unfitted model per fold")
        return WalkForwardRunner(self.config).run(
            features,
            labels,
            topk=topk,
            cost_bps=transaction_cost_bps,
            walk_months=walk_forward_months,
            embargo_days=embargo_days,
            **kwargs,
        )

    @staticmethod
    def _extract_dates(df):
        return pd.DatetimeIndex(
            df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index
        )

    @staticmethod
    def _top_k_period_returns(y, preds, topk, cost_bps=0):
        """Date-local label diagnostic only, never used as portfolio performance."""
        if cost_bps:
            raise BacktestError("costs require actual turnover from executable price replay")
        if (
            not isinstance(y.index, pd.MultiIndex)
            or len(preds) != len(y)
            or y.index.has_duplicates
            or topk <= 0
        ):
            raise BacktestError("aligned unique (date,instrument) labels and predictions required")
        frame = pd.DataFrame({"label": y, "prediction": preds}).sort_index()
        if not np.isfinite(frame.to_numpy()).all():
            raise BacktestError("finite labels/predictions required")
        return [
            float(g.nlargest(topk, "prediction")["label"].mean())
            for _, g in frame.groupby(level=0, sort=True)
        ]

    @staticmethod
    def _compute_metrics(daily_returns):
        values = np.asarray(daily_returns, dtype=float)
        return performance_metrics([1.0, *np.cumprod(1 + values)], 252)


class BacktestValidator:
    """Legacy return-shape adapter over the authoritative ValidationSuite."""

    def __init__(self, config=None):
        self.config = config

    def validate(self, result, **kwargs):
        from src.core.validation import ValidationSuite
        from src.utils.schemas import ValidationCheck, ValidationResult, ValidationVerdict

        report = ValidationSuite(self.config).validate(result, **kwargs)
        return ValidationResult(
            checks=[
                ValidationCheck(
                    name=c.name,
                    passed=c.verdict.value == "PASS",
                    details=c.explanation,
                    severity="critical" if c.verdict.value == "FAIL" else "warning",
                )
                for c in report.checks
            ],
            verdict=ValidationVerdict.APPROVED
            if report.overall == "RELIABLE"
            else ValidationVerdict.REJECTED,
            summary=report.summary,
        )
