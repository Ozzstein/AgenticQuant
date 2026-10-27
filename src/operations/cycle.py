"""Causal, durable composition of the paper desk and the canonical accounting engine.

This service never fetches ambient data, invents model outputs, sends notifications,
or reads research holdouts. Dispatch and fills retain distinct causal clocks. Calls
are isolated in a nonrecoverable stage; local work has explicit replay receipts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.evidence.service import ROLE_KINDS, EvidenceService
from src.data.features.contextual import materialize_evidence_fields
from src.data.features.dataset import FeatureDataset, feature_dataset
from src.domain.portfolio import (
    ActionCoverage,
    CashDividend,
    CorporateAction,
    DividendDistribution,
    DividendPayment,
    ExecutionBar,
    InstrumentRisk,
    OpeningEvent,
    PaperDividendPolicy,
    PortfolioSnapshot,
    PortfolioTarget,
    PriceMark,
    SleeveSnapshot,
    UniverseEligibility,
)
from src.execution.routing import target_from_blended_frame
from src.execution.sleeves import SleeveBook, exact_sum, partition
from src.execution.venue import PaperVenue, VenueExecution, VenueRules
from src.operations.store import RecoveryRequired, RunManifest, canonical, digest
from src.research.evidence import (
    BoundArtifact,
    SourceBinding,
    expected_artifacts,
    json_pointer,
    verify_market_snapshot,
)
from src.research.regime_experiments import (
    FrozenEvidence,
    materialize_macro_dataset,
    verify_frozen_evidence,
)
from src.strategies.allocation import blend_sleeves, select_allocations
from src.strategies.runner import evaluate_strategy, require_strategy_eligible

from src.agents.consensus import consensus_recommendation
from src.agents.contracts import PortfolioReviewContext, hmm_context_from_result
from src.agents.graph import analyze_instrument
from src.agents.history import DeskHistoryStore
from src.core.macro_regime import CausalHMMRegimeDetector, CausalRegimeResult
from src.core.model_zoo import ModelWrapper
from src.core.walk_forward import frame_hash
from src.decision.evidence import source_from_passage
from src.decision.service import DecisionService
from src.decision.store import DecisionStore
from src.domain.common import RunContext
from src.domain.decisions import AgentDecision
from src.execution.portfolio_optimizer import AllocationConstraints, PortfolioOptimizer
from src.operations.contracts import (
    INCOME_FIELDS,
    CycleResult,
    ObservedOpenReceipt,
    PaperCycleInputs,
    PaperCyclePlan,
    PaperCycleTemplate,
)
from src.utils.config import AppConfig, QlibConfig

D = Decimal
PASSIVE = frozenset({"HOLD", "ABSTAIN", "VETO"})


def cycle_config_hash(config) -> str:
    """Public settings identity. SecretStr values are masked, never archived."""
    return digest(canonical(config.model_dump(mode="json")))


def training_frame_hash(dataset: FeatureDataset, fitted_at) -> str:
    rows = dataset.eligible_training(fitted_at)
    finite = np.isfinite(rows.features.to_numpy(dtype=float)).all(axis=1)
    return digest(
        canonical(
            {
                name: frame_hash(value.loc[finite])
                for name, value in (
                    ("features", rows.features),
                    ("labels", rows.labels),
                    ("feature_available_at", rows.feature_available_at),
                    ("label_available_at", rows.label_available_at),
                    ("label_end_at", rows.label_end_at),
                )
            }
        )
    )


def contextual_fields(spec):
    """External numeric inputs required by exact bounded feature expressions."""
    required = set().union(*(set(f.required_fields) for f in spec.features))
    for feature in spec.features:
        tree = ast.parse(
            re.sub(r"\$([A-Za-z][A-Za-z0-9_]*)", r"\1", feature.expression), mode="eval"
        )
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        required |= (
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            - calls
            - {"np", "numpy", "df"}
        )
        required |= {
            str(n.slice.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
        }
    required -= {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "market_type",
        "available_at",
    }
    return required


def _frame(frame):
    # Missing scores are explicitly null, never a neutral score or a zero.
    return {
        "split": {
            "index": frame.index.tolist(),
            "columns": frame.columns.tolist(),
            "data": [
                [None if pd.isna(value) else value for value in row]
                for row in frame.itertuples(index=False, name=None)
            ],
        },
        "attrs": frame.attrs,
    }


def _unframe(value):
    split = value["split"]
    frame = pd.DataFrame(split["data"], index=split["index"], columns=split["columns"])
    frame.index.name = "instrument"
    frame.attrs = value["attrs"]
    return frame


def _sleeves(values):
    return {name: SleeveSnapshot.model_validate(value) for name, value in values.items()}


class _FeatureReads:
    """One materialization's read cache over already verified immutable records."""

    def __init__(self, store, records):
        self.store, self.records, self.queries, self.values = store, records, {}, {}

    def query(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self.queries:
            self.queries[key] = self.store.query(**kwargs)
        return self.queries[key]

    def normalized(self, identity, *, as_of):
        self.records[identity].require_available(as_of)
        if identity not in self.values:
            self.values[identity] = self.store.normalized(identity, as_of=as_of)
        return self.values[identity]


class PaperCycleService:
    def __init__(
        self,
        config,
        *,
        operations_store,
        evidence_store,
        registry,
        ledger,
        attribution_store,
        artifact_dir,
        decision_service=None,
        decision_policy=None,
        llm_factory=None,
        resolver=None,
    ):
        self.config, self.operations, self.evidence, self.registry = (
            config,
            operations_store,
            evidence_store,
            registry,
        )
        self.ledger, self.attribution = ledger, attribution_store
        self.artifacts = Path(artifact_dir).resolve()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.decisions = decision_service or DecisionService(
            DecisionStore(Path(config.output_dir) / "decisions.sqlite")
        )
        self.policy, self.llm_factory, self.resolver = decision_policy, llm_factory, resolver
        self.history = DeskHistoryStore(Path(config.output_dir) / "desk-history.sqlite3")

    def _source_values(self, sources, as_of):
        if not sources:
            raise ValueError("explicit immutable source bindings required")
        values = []
        for source in sources:
            if source.available_at > as_of:
                raise ValueError("economic source binding is after the operating cutoff")
            record = self.evidence.get(source.evidence_id, as_of=source.available_at).record
            if digest(self.evidence.raw(source.evidence_id)) != record.raw_hash:
                raise ValueError("economic source raw bytes differ")
            selected = json_pointer(
                self.evidence.normalized(source.evidence_id, as_of=source.available_at),
                source.pointer,
            )
            values.extend(selected if isinstance(selected, list) else [selected])
        return values

    def _verify_inputs(self, inputs, references, rule_sources, income_authorities=()):
        from src.operations.evidence import verify_paper_provenance

        if inputs.income_policy is None:
            raise ValueError("fresh paper cycles require an explicit sourced income policy")
        prior = self._prior_account_anchor(inputs.account.account_id, inputs.context.run_id)
        started = next(iter(prior.values())).portfolio.as_of if prior else inputs.account.opened_at
        if started >= inputs.context.as_of:
            raise ValueError("income interval must follow the actual committed account anchor")
        return verify_paper_provenance(
            self.registry,
            self.evidence,
            inputs,
            references,
            receipts=inputs.receipts,
            rule_sources=rule_sources,
            income_policy=inputs.income_policy,
            required_started_at=started,
            required_ended_at=inputs.context.as_of,
            income_authorities=income_authorities,
        )

    def prepare(
        self,
        template: PaperCycleTemplate,
        *,
        refresh_run_id,
        run_id,
        scheduled_for,
        economic_sources,
        macro_dataset,
        contextual_sources: tuple[FrozenEvidence, ...] = (),
        income_authorities: tuple[SourceBinding, ...] = (),
    ) -> PaperCyclePlan:
        """Build a current plan from an authoritative refresh and archived arrays.

        No provider fetch, guessed opening volume, copied old input context or
        supplied receipt-shaped object is accepted as a successful refresh.
        """
        template = PaperCycleTemplate.model_validate(template.model_dump())
        if template.income_policy is None:
            raise ValueError("fresh paper preparation requires an explicit sourced income policy")
        status = self.operations.status(refresh_run_id)
        manifest = self.operations.manifest(refresh_run_id)
        if (
            status["status"] != "completed"
            or manifest.kind != "data"
            or manifest.context.mode != "paper"
        ):
            raise ValueError("a completed authoritative paper refresh is required")
        receipt = self.operations.artifact(status["stages"]["report"]["output_hash"])
        captured = RunContext.model_validate(receipt["context"])
        completed = datetime.fromisoformat(receipt["completed_at"])
        if (
            captured.as_of != completed
            or receipt["plan_hash"] != manifest.input_hash
            or receipt["requested_context"] != manifest.context.model_dump(mode="json")
            or captured.account_id != template.account.account_id
        ):
            raise ValueError("refresh identity or measured completion clock differs")
        for fetch in receipt["fetches"].values():
            verify_frozen_evidence(
                self.evidence,
                tuple(FrozenEvidence.model_validate(s) for s in fetch["sources"]),
                as_of=completed,
            )
        sources = dict(economic_sources)
        names = {
            "universe_history": "universe_history_id",
            "availability_audit": "availability_audit_id",
            "calendar": "calendar_artifact_id",
            "corporate_actions": "corporate_actions_id",
            "execution_bars": "execution_bars_id",
            "opening_liquidity": "opening_liquidity_id",
            "instrument_metadata": "instrument_metadata_id",
        }
        if "opening_events" in sources:
            names.pop("execution_bars")
            names["opening_events"] = "opening_events_id"
        names.update({kind: kind + "_id" for kind in INCOME_FIELDS})
        if len(sources) != len(economic_sources) or set(sources) != set(names) | {"venue_rules"}:
            raise ValueError("exact eight base and five income source kinds required")
        values = {
            kind: self._source_values(bindings, completed) for kind, bindings in sources.items()
        }
        context = RunContext(
            run_id=run_id,
            mode="paper",
            as_of=completed,
            config_hash=cycle_config_hash(self.config),
            code_version=template.code_version,
            account_id=template.account.account_id,
        )
        market = EvidenceService(self.evidence).market_snapshot(
            template.instruments, context, start=template.market_start, price_basis="raw"
        )
        universes = tuple(UniverseEligibility.model_validate(v) for v in values["universe_history"])
        context = context.model_copy(
            update={
                "data_snapshot_ids": tuple(
                    sorted({market.content_hash(), *(u.data_snapshot_id for u in universes)})
                )
            }
        )
        open_receipts = []
        for source in sources["opening_liquidity"]:
            envelope = self.evidence.get(source.evidence_id, as_of=completed)
            for row in self._source_values((source,), completed):
                open_receipts.append(
                    ObservedOpenReceipt(
                        event_id=row["event_id"],
                        evidence_id=source.evidence_id,
                        evidence_hash=envelope.content_hash(),
                        observed_at=envelope.record.ingested_at,
                        available_at=envelope.record.available_at,
                    )
                )
        inputs = PaperCycleInputs(
            context=context,
            snapshot=market,
            account=template.account,
            bars=tuple(ExecutionBar.model_validate(v) for v in values.get("execution_bars", ())),
            opening_events=tuple(
                OpeningEvent.model_validate(v) for v in values.get("opening_events", ())
            ),
            metadata=tuple(InstrumentRisk.model_validate(v) for v in values["instrument_metadata"]),
            universes=universes,
            universe_version=template.universe_version,
            actions=tuple(CorporateAction.model_validate(v) for v in values["corporate_actions"]),
            fill_policy=template.fill_policy,
            risk_policy=template.risk_policy,
            rules=tuple(VenueRules.model_validate(v) for v in values["venue_rules"]),
            initial_capital=template.initial_capital,
            receipts=tuple(open_receipts),
            income_policy=template.income_policy,
            cash_dividends=tuple(CashDividend.model_validate(v) for v in values["cash_dividends"]),
            dividend_payments=tuple(
                DividendPayment.model_validate(v) for v in values["dividend_payments"]
            ),
            dividend_distributions=tuple(
                DividendDistribution.model_validate(v) for v in values["dividend_distributions"]
            ),
            action_coverage=tuple(
                ActionCoverage.model_validate(v) for v in values["action_coverage"]
            ),
            paper_dividend_policies=tuple(
                PaperDividendPolicy.model_validate(v) for v in values["paper_dividend_policies"]
            ),
        )
        refs = []
        for kind, expected in expected_artifacts(inputs).items():
            if values[kind] != expected:
                raise ValueError(
                    "fresh " + kind + " does not match actual selected market/economic inputs"
                )
            bound = BoundArtifact(
                kind=kind,
                inputs_hash=inputs.content_hash(),
                value_json=canonical(expected).decode(),
                sources=sources[kind],
            )
            refs.append((names[kind], self.registry.archive_contract(bound)))
        with self.operations.exclusive("account:" + inputs.account.account_id):
            self._verify_inputs(inputs, dict(refs), sources["venue_rules"], income_authorities)
        numeric = tuple(FrozenEvidence.model_validate(v) for v in contextual_sources)
        if any(contextual_fields(spec) for spec in template.strategies) and not numeric:
            raise ValueError("explicit contextual source whitelist required; no ambient discovery")
        verify_frozen_evidence(self.evidence, numeric, as_of=completed)
        documents = set()
        for instrument in template.instruments:
            for kind in sorted(set().union(*ROLE_KINDS.values())):
                query = self.evidence.query(
                    as_of=completed,
                    kind=kind,
                    instrument_id=instrument.instrument_id,
                    current_revisions_only=False,
                )
                documents.update(e.record.evidence_id for e in query.records if e.document_hash)
        plan = PaperCyclePlan(
            inputs=inputs,
            scheduled_for=scheduled_for,
            provenance=tuple(refs),
            rule_sources=sources["venue_rules"],
            income_authorities=income_authorities,
            strategies=template.strategies,
            models=template.models,
            macro_recipe=template.macro_recipe,
            macro_dataset=macro_dataset,
            numeric_bindings=template.numeric_bindings,
            numeric_sources=numeric,
            allocation=template.allocation,
            desk_evidence_ids=tuple(sorted(documents)),
            max_source_chars=template.max_source_chars,
        )
        # Source construction is independently executable before any provider call.
        materialize_macro_dataset(
            plan.macro_dataset,
            self.evidence,
            availability_policy=plan.macro_recipe.availability_policy,
        )
        self.operations.archive(
            {
                "template_hash": template.content_hash(),
                "refresh_manifest_hash": manifest.content_hash(),
                "refresh_report_hash": status["stages"]["report"]["output_hash"],
                "plan": plan.model_dump(mode="json"),
            }
        )
        return plan

    def materialize_features(
        self, plan: PaperCyclePlan, *, training=False
    ) -> dict[str, FeatureDataset]:
        """Public deterministic preparation API for exact recipe/frame approvals."""
        read_caches = {}
        result = {}
        recipes = {r.strategy: r for r in plan.models}
        for spec in plan.strategies:
            if training and spec.name not in recipes:
                continue
            source = recipes[spec.name].training_snapshot if training else plan.inputs.snapshot
            verify_market_snapshot(self.evidence, source)
            bars = tuple(b for b in source.bars if b.instrument.asset_class in spec.asset_classes)
            ids = {b.evidence_id for b in bars}
            snapshot = source.model_copy(
                update={
                    "bars": bars,
                    "evidence": tuple(e for e in source.evidence if e.evidence_id in ids),
                }
            )
            index = pd.MultiIndex.from_tuples(
                sorted((b.closed_at, b.instrument.instrument_id) for b in bars),
                names=["datetime", "instrument"],
            )
            required = contextual_fields(spec)
            pinned_sources = (
                recipes[spec.name].training_numeric_sources if training else plan.numeric_sources
            )
            source_context = (
                source.context.model_copy(update={"as_of": recipes[spec.name].fitted_at})
                if training
                else plan.inputs.context
            )
            if required and not pinned_sources:
                raise ValueError("explicit frozen training/current numeric sources required")
            cache_key = (
                source_context.as_of,
                tuple((s.evidence_id, s.envelope_hash) for s in pinned_sources),
            )
            if cache_key not in read_caches:
                read_caches[cache_key] = _FeatureReads(
                    self.evidence,
                    verify_frozen_evidence(
                        self.evidence, pinned_sources, as_of=source_context.as_of
                    ),
                )
            reads = read_caches[cache_key]
            bindings = tuple(
                b
                for b in plan.numeric_bindings
                if b.name in required
                and b.target_instrument_id in index.get_level_values("instrument")
            )
            fields = (
                materialize_evidence_fields(
                    reads,
                    index,
                    source_context,
                    bindings,
                    allowed_evidence_ids=frozenset(s.evidence_id for s in pinned_sources),
                    availability_policy=(
                        recipes[spec.name].training_availability_policy
                        if training
                        else "captured-vintage-at-fit-1"
                    ),
                )
                if required
                else None
            )
            dataset = feature_dataset(
                snapshot,
                spec.features,
                horizon=recipes[spec.name].label_horizon if spec.name in recipes else 1,
                evidence_fields=fields,
            )
            if training:
                recipe = recipes[spec.name]
                selected_universe = recipe.training_universe
                if self._source_values(recipe.training_universe_sources, recipe.fitted_at) != [
                    selected_universe.model_dump(mode="json")
                ]:
                    raise ValueError("frozen training membership differs from immutable sources")
                if any(
                    source.available_at > selected_universe.available_at
                    for source in recipe.training_universe_sources
                ):
                    raise ValueError("training membership source precedes no declared availability")
            else:
                candidates = [
                    u
                    for u in plan.inputs.universes
                    if u.universe_version == plan.inputs.universe_version
                    and u.available_at <= plan.inputs.context.as_of
                    and u.effective_at <= plan.inputs.context.as_of
                ]
                if not candidates:
                    raise ValueError("actual current eligible universe missing")
                selected_universe = max(candidates, key=lambda u: (u.effective_at, u.available_at))
            # Current training cohort is explicitly fixed at fit time. It does
            # not claim these instruments comprised a historical research index.
            mask = np.asarray(
                [key in selected_universe.instrument_ids for _, key in dataset.features.index]
            )
            result[spec.name] = FeatureDataset(
                dataset.features.loc[mask],
                dataset.labels.loc[mask],
                dataset.feature_available_at.loc[mask],
                dataset.label_available_at.loc[mask],
                dataset.snapshot_hash,
                dataset.feature_spec_hash,
                dataset.label_end_at.loc[mask] if dataset.label_end_at is not None else None,
            )
        return result

    def _approved(self, recipe, context, slot):
        identity = recipe.content_hash()
        if identity in self.config.operations.approved_model_hashes:
            return
        activation = (
            self.resolver(slot, payload_hash=identity, as_of=context.as_of)
            if self.resolver
            else None
        )
        if (
            activation is None
            or activation.payload_hash != identity
            or activation.activated_at > context.as_of
        ):
            raise ValueError("exact model recipe has no explicit approval or registry activation")

    def _metadata(self, plan, at):
        result = {}
        for item in sorted(plan.inputs.metadata, key=lambda m: m.as_of):
            if item.as_of <= at:
                result[item.instrument.instrument_id] = item
        if not result or any(
            (at - m.as_of).total_seconds() > plan.inputs.risk_policy.max_liquidity_age_seconds
            for m in result.values()
        ):
            raise ValueError("missing or stale observed liquidity metadata")
        return result

    def _returns(self, plan):
        """Estimate now from actual lagged intervals, never backdate capture."""
        snapshot, context, policy = plan.inputs.snapshot, plan.inputs.context, plan.allocation
        records = {e.evidence_id: e for e in snapshot.evidence}
        by_instrument = {}
        for bar in snapshot.bars:
            records[bar.evidence_id].require_available(context.as_of)
            if bar.closed_at > context.as_of:
                raise ValueError("covariance endpoint is future to estimation cutoff")
            by_instrument.setdefault(bar.instrument.instrument_id, []).append(bar)
        # Use actual observed equity closes for a mixed book. Pure crypto uses
        # actual crypto closes. Weekend returns are never copied into extra rows.
        equities = [b.closed_at for b in snapshot.bars if b.instrument.asset_class == "equity"]
        anchors = sorted(set(equities or [b.closed_at for b in snapshot.bars]))
        intervals = {}
        for key, bars in by_instrument.items():
            bars = sorted(bars, key=lambda b: b.closed_at)
            if len({b.closed_at for b in bars}) != len(bars):
                raise ValueError("ambiguous covariance price endpoint")
            intervals[key] = list(zip(bars, bars[1:]))
        selected, rows, used = [], [], {key: set() for key in intervals}
        for anchor in anchors:
            sample, values = {}, {}
            for key, pairs in intervals.items():
                eligible = [(start, end) for start, end in pairs if end.closed_at <= anchor]
                if not eligible:
                    break
                start, end = eligible[-1]
                if (
                    anchor - start.closed_at
                ).total_seconds() > policy.covariance_max_endpoint_age_seconds:
                    break
                a, b = records[start.evidence_id], records[end.evidence_id]
                value = end.close / start.close - 1
                if not value.is_finite():
                    raise ValueError("nonfinite actual covariance return")
                values[key] = float(value)
                sample[key] = {
                    "started_at": start.closed_at.isoformat(),
                    "ended_at": end.closed_at.isoformat(),
                    "start_price": str(start.close),
                    "end_price": str(end.close),
                    "start_evidence_id": start.evidence_id,
                    "end_evidence_id": end.evidence_id,
                    "start_envelope_hash": self.evidence.get(
                        start.evidence_id, as_of=context.as_of
                    ).content_hash(),
                    "end_envelope_hash": self.evidence.get(
                        end.evidence_id, as_of=context.as_of
                    ).content_hash(),
                    "start_available_at": a.available_at.isoformat(),
                    "end_available_at": b.available_at.isoformat(),
                    "return": str(value),
                }
            if len(sample) != len(intervals):
                continue
            identities = {
                key: (v["start_evidence_id"], v["end_evidence_id"]) for key, v in sample.items()
            }
            if any(identity in used[key] for key, identity in identities.items()):
                continue
            for key, identity in identities.items():
                used[key].add(identity)
            rows.append((anchor, values))
            selected.append({"anchor": anchor.isoformat(), "intervals": sample})
        rows, selected = (
            rows[-policy.covariance_observations :],
            selected[-policy.covariance_observations :],
        )
        if len(rows) < 20:
            raise ValueError("insufficient distinct actual paired covariance intervals")
        frame = pd.DataFrame([v for _, v in rows], index=pd.DatetimeIndex([at for at, _ in rows]))
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("covariance requires finite actual paired returns")
        receipt = {
            "policy": policy.covariance_sampling_policy,
            "estimation_cutoff": context.as_of.isoformat(),
            "anchor_calendar": "observed_equity_closes" if equities else "observed_crypto_closes",
            "max_endpoint_age_seconds": policy.covariance_max_endpoint_age_seconds,
            "market_snapshot_hash": snapshot.content_hash(),
            "samples": selected,
        }
        frame.attrs["sampling_receipt"] = receipt
        frame.attrs["sampling_hash"] = digest(canonical(receipt))
        return frame

    def _record_period(self, book, opening, at, available_at):
        closing = book.snapshot(at)
        if opening and next(iter(opening.values())).portfolio.as_of < at:
            periods = book.performance_period(opening, at, available_at=available_at)
            for name, period in periods.items():
                if period["opening_nav"] > 0:
                    self.attribution.record_nav(name, **period)
                elif closing[name].portfolio.nav > 0:
                    self._funded_anchor(name, closing[name], available_at)
        return {name: value.model_dump(mode="json") for name, value in closing.items()}

    def _funded_anchor(self, name, sleeve, available_at):
        self.attribution.start_funded_epoch(
            name,
            at=sleeve.portfolio.as_of,
            available_at=available_at,
            opening_nav=sleeve.portfolio.nav,
            source_event_id="funded:" + sleeve.portfolio.account_id + ":" + sleeve.source_hash,
            source_event_hash=sleeve.content_hash(),
        )

    def _prior_anchor(self, plan):
        return self._prior_account_anchor(
            plan.inputs.account.account_id, plan.inputs.context.run_id
        )

    def _prior_account_anchor(self, account_id, run_id):
        candidates = []
        for run in self.operations.account_runs(account_id):
            if (
                run["kind"] != "paper_cycle"
                or run["account_id"] != account_id
                or run["run_id"] == run_id
            ):
                continue
            if run["status"] != "completed":
                if run["stages"].get("account", {}).get("state") not in {
                    "completed",
                    "running",
                    "uncertain",
                }:
                    continue
                raise RecoveryRequired(
                    "another account cycle requires explicit completion/reconciliation"
                )
            stage = run["stages"].get("attribution", {})
            if stage.get("output_hash"):
                receipt = self.operations.artifact(stage["output_hash"])
                values = _sleeves(receipt["anchor"])
                candidates.append((next(iter(values.values())).portfolio.as_of, values))
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        return None

    def _require_resolved_execution(self, account, execution):
        for pending in self.ledger.pending(account):
            if execution.state(pending["intent"].intent_id)["state"] in {
                "ambiguous",
                "submitting",
                "cancel_pending",
                "reconciliation_required",
            }:
                raise RecoveryRequired("unresolved venue outcome blocks observed opening")

    def _settle_observed_open(self, plan, venue, execution, event):
        account = plan.inputs.account.account_id
        self._require_resolved_execution(account, execution)
        receipt = next(r for r in plan.inputs.receipts if r.event_id == event.event_id)
        from src.operations.evidence import prepare_opening_source

        identity = prepare_opening_source(
            self.evidence, venue, event, receipt, as_of=plan.inputs.context.as_of
        )
        accepted = venue.accepted_intent_ids_before(event.opened_at)
        metadata = {
            p["intent"].intent_id: execution.admission_metadata(p["intent"].intent_id)
            for p in self.ledger.pending(account)
            if p["intent"].intent_id in accepted
            and p["intent"].instrument == event.instrument
            and p["intent"].eligible_at <= event.opened_at
        }
        settled = venue.observed_open_receipt(event.event_id)
        if settled is not None:
            # Filled orders may no longer be pending. Recover the
            # original immutable admission map, never today's risk
            # metadata or an empty replacement for a consumed order.
            metadata = {
                intent_id: execution.admission_metadata(intent_id)
                for intent_id, _ in settled.admission_metadata_hashes
            }
        return venue.settle_open(
            event,
            metadata,
            available_at=receipt.available_at,
            observed_at=receipt.observed_at,
            as_of=plan.inputs.context.as_of,
            evidence_id=receipt.evidence_id,
            evidence_hash=receipt.evidence_hash,
            source_identity=identity,
        )

    def _admit_income(self, plan, book, anchor):
        """Admit original terms; ownership/cash remain canonical ledger operations."""
        with self.ledger.atomic():
            account = plan.inputs.account.account_id
            started = next(iter(anchor.values())).portfolio.as_of
            supplied = {t.content_hash() for t in plan.inputs.cash_dividends}
            outstanding = {
                r.entitlement.action_id for r in self.ledger.dividend_receivables(account)
            }
            accrued = {
                e.action_id
                for e in self.ledger.dividend_entitlements(account, as_of=plan.inputs.context.as_of)
            }
            # The canonical reader verifies the full stored revision/admission
            # chain. A source refresh cannot silently drop a
            # previously announced entitlement or an unsettled receivable.
            for terms in self.ledger.cash_dividends(account, as_of=plan.inputs.context.as_of):
                if (terms.action_id not in accrued or terms.action_id in outstanding) and (
                    terms.content_hash() not in supplied
                ):
                    raise ValueError(
                        "original admitted dividend terms must remain explicitly sourced"
                    )
            for terms in sorted(plan.inputs.cash_dividends, key=lambda t: t.available_at):
                self.ledger.register_cash_dividend(
                    account, terms, as_of=max(started, terms.available_at)
                )
            for coverage in plan.inputs.action_coverage:
                self.ledger.record_action_coverage(
                    account, coverage, as_of=plan.inputs.context.as_of
                )
            return {
                "policy": plan.inputs.income_policy,
                "required_started_at": started.isoformat(),
                "required_ended_at": plan.inputs.context.as_of.isoformat(),
                "terms_hashes": [t.content_hash() for t in plan.inputs.cash_dividends],
                "coverage_hashes": [c.content_hash() for c in plan.inputs.action_coverage],
                "receivables": [
                    r.model_dump(mode="json") for r in self.ledger.dividend_receivables(account)
                ],
            }

    @staticmethod
    def _ordered_observations(events):
        priority = {
            "dividend_entitlement": 0,
            "dividend_payment": 1,
            "dividend_distribution": 1,
            "action": 2,
            "close": 3,
            "market_close": 3,
            "open": 4,
            "opening_event": 4,
        }
        return sorted(events, key=lambda pair: priority[pair[0]])

    @staticmethod
    def _observe_income(plan, book, kind, event, at):
        if kind == "dividend_entitlement":
            return book.accrue_dividend(event.action_id, at=at)
        if kind == "dividend_payment":
            return book.apply_dividend_payment(event, as_of=at)
        if kind == "dividend_distribution":
            policy = next(
                p
                for p in plan.inputs.paper_dividend_policies
                if p.distribution_hash == event.content_hash()
            )
            return book.apply_paper_dividend_distribution(event, policy=policy, as_of=at)
        raise ValueError("unknown canonical income event")

    def _observe(self, plan, book, venue, execution, opening, events, at):
        account = plan.inputs.account.account_id
        for kind, event in self._ordered_observations(events):
            if kind.startswith("dividend_"):
                self._observe_income(plan, book, kind, event, at)
            elif kind == "action":
                self.ledger.corporate_action(account, event)
            elif kind in {"open", "opening_event"}:
                if plan.inputs.receipts:
                    self._settle_observed_open(plan, venue, execution, event)
                else:
                    self._require_resolved_execution(account, execution)
                    venue.advance_open(event, self._metadata(plan, at))
            elif kind == "market_close":
                event.require_execution_price()
                if event.instrument.quote_currency != plan.inputs.account.base_currency:
                    raise ValueError(
                        "actual source-bound close FX is required for nonbase valuation"
                    )
                with self.ledger.atomic():
                    if self.ledger.record_event(
                        account, self._market_close_id(event), event.model_dump_json()
                    ):
                        self.ledger.mark(
                            account,
                            [
                                PriceMark(
                                    instrument=event.instrument,
                                    price=event.close,
                                    fx_rate=D(1),
                                    as_of=event.closed_at,
                                )
                            ],
                        )
            else:
                venue.advance_close(event)
        # Opening-only receipts do not carry a completed ExecutionBar. Expire
        # orders through the canonical venue after the independently observed
        # close marks, at this actual economic timestamp.
        venue.expire_orders(as_of=at)
        # Book sync uses actual fills and actions, never target weights as holdings.
        anchor = self._record_period(book, opening, at, plan.inputs.context.as_of)
        return {"anchor": anchor, "observed_at": at.isoformat()}

    @staticmethod
    def _market_close_id(bar):
        return "market-close:" + bar.instrument.instrument_id + ":" + bar.closed_at.isoformat()

    def _dispatch(self, execution, now):
        if now.tzinfo is None:
            raise ValueError("dispatch requires actual aware observation clock")
        states = []
        for pending in self.ledger.pending(execution.account_id):
            intent = pending["intent"]
            state = execution.state(intent.intent_id)
            if state["state"] in {
                "ambiguous",
                "submitting",
                "cancel_pending",
                "reconciliation_required",
            }:
                raise RecoveryRequired("ambiguous dispatch requires authoritative reconciliation")
            if state["state"] == "prepared" and intent.eligible_at <= now:
                states.append(execution.dispatch(intent.intent_id, as_of=now))
        if any(
            s["state"] in {"ambiguous", "submitting", "cancel_pending", "reconciliation_required"}
            for s in states
        ):
            raise RecoveryRequired("dispatch outcome is uncertain")
        return states

    def dispatch_pending(self, plan, *, now=None):
        """Scheduler hook before future opens; never submit against a past print."""
        plan = PaperCyclePlan.model_validate(plan.model_dump())
        with self.operations.exclusive("account:" + plan.inputs.account.account_id):
            manifest = self.operations.manifest(plan.inputs.context.run_id)
            if (
                manifest.kind != "paper_cycle"
                or manifest.input_hash != plan.content_hash()
                or self.operations.status(plan.inputs.context.run_id)["status"] != "completed"
                or cycle_config_hash(self.config) != plan.inputs.context.config_hash
            ):
                raise ValueError("dispatch requires the exact completed operating plan/config")
            rules = {r.instrument.instrument_id: r for r in plan.inputs.rules}
            venue = PaperVenue(
                self.ledger,
                plan.inputs.account.account_id,
                rules,
                fill_policy=plan.inputs.fill_policy,
                risk_policy=plan.inputs.risk_policy,
            )
            execution = VenueExecution(
                self.ledger,
                plan.inputs.account.account_id,
                {"paper": venue},
                {key: "paper" for key in rules},
                owner="direct",
                risk_policy=plan.inputs.risk_policy,
                fill_policy=plan.inputs.fill_policy,
            )
            return self._dispatch(execution, now or datetime.now(timezone.utc))

    def _model_config(self, recipe):
        # Ambient YAML/model_params do not enter an approved recipe.
        return AppConfig(_env_file=None).model_copy(
            update={
                "qlib": QlibConfig(
                    model_name=recipe.model_name, model_params=dict(recipe.parameters)
                )
            }
        )

    def _trusted_save(self, identity, path, recipe, kind, run_id):
        files = {
            suffix: hashlib.sha256(Path(str(path) + suffix).read_bytes()).hexdigest()
            for suffix in ("", ".json", *([".sha256"] if kind == "macro" else []))
        }
        stage_name = "macro" if kind == "macro" else "models"
        stage = self.operations.status(run_id)["stages"][stage_name]
        if stage["state"] != "running":
            raise ValueError("fresh model requires its durably admitted creation stage")
        completed = datetime.now(timezone.utc).isoformat()
        receipt = {
            "recipe": recipe.model_dump(mode="json"),
            "recipe_hash": identity,
            "kind": kind,
            "files": files,
            "training_cutoff": recipe.fitted_at.isoformat(),
            "creation": {
                "run_id": run_id,
                "stage": stage_name,
                "attempt": stage["attempt"],
                "stage_started_at": stage["started_at"],
                "artifact_completed_at": completed,
                "available_at": completed,
            },
        }
        key = self.operations.archive(receipt)
        pointer = self.artifacts / (identity + ".receipt")
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(key)
        temporary.replace(pointer)
        return receipt

    def _trusted_read(self, recipe, kind):
        identity = recipe.content_hash()
        path = self.artifacts / (identity + ".joblib")
        pointer = self.artifacts / (identity + ".receipt")
        if not pointer.exists():
            return path, None
        receipt = self.operations.artifact(pointer.read_text())
        if (
            receipt["recipe_hash"] != identity
            or receipt["recipe"] != recipe.model_dump(mode="json")
            or receipt["kind"] != kind
            or set(receipt["files"])
            != ({"", ".json", ".sha256"} if kind == "macro" else {"", ".json"})
        ):
            raise ValueError("trusted model receipt identity differs")
        for suffix, value in receipt["files"].items():
            if (
                suffix not in {"", ".json", ".sha256"}
                or hashlib.sha256(Path(str(path) + suffix).read_bytes()).hexdigest() != value
            ):
                raise ValueError("trusted model bytes/metadata differ before deserialization")
        return path, receipt

    def _fit_models(self, plan, datasets=None):
        result = {}
        for recipe in plan.models:
            with self.operations.exclusive("model:" + recipe.content_hash()):
                self._approved(recipe, plan.inputs.context, "model:" + recipe.strategy)
                path, receipt = self._trusted_read(recipe, "numeric")
                if receipt is None:
                    if datasets is None:
                        datasets = self.materialize_features(plan, training=True)
                    dataset = datasets[recipe.strategy]
                    if (
                        dataset.snapshot_hash != recipe.data_snapshot_hash
                        or training_frame_hash(dataset, recipe.fitted_at)
                        != recipe.training_frame_hash
                    ):
                        raise ValueError("approved feature/training data identity differs")
                    train = dataset.eligible_training(recipe.fitted_at)
                    valid = np.isfinite(train.features.to_numpy(dtype=float)).all(axis=1)
                    if valid.sum() < recipe.minimum_rows:
                        raise ValueError("insufficient complete train-only observations")
                    model = ModelWrapper(recipe.model_name, self._model_config(recipe))
                    model.train(
                        train.features.loc[valid],
                        train.labels.loc[valid],
                        context=plan.inputs.context.model_copy(update={"as_of": recipe.fitted_at}),
                        feature_available_at=train.feature_available_at.loc[valid],
                        label_available_at=train.label_available_at.loc[valid],
                        feature_spec_hash=recipe.feature_spec_hash,
                        data_snapshot_hash=recipe.data_snapshot_hash,
                        label_end_at=train.label_end_at.loc[valid],
                        availability_policy=recipe.training_availability_policy,
                    )
                    model.save(str(path))
                    receipt = self._trusted_save(
                        recipe.content_hash(), path, recipe, "numeric", plan.inputs.context.run_id
                    )
                result[recipe.strategy] = receipt
        return result

    def _macro(self, plan):
        with self.operations.exclusive("model:" + plan.macro_recipe.content_hash()):
            return self._load_or_fit_macro(plan)

    def _load_or_fit_macro(self, plan):
        recipe = plan.macro_recipe
        self._approved(recipe, plan.inputs.context, "regime:macro")
        data = materialize_macro_dataset(
            plan.macro_dataset,
            self.evidence,
            availability_policy=plan.macro_recipe.availability_policy,
        )
        path, receipt = self._trusted_read(recipe, "macro")
        detector = CausalHMMRegimeDetector(
            self.artifacts
            / (
                digest(
                    canonical(
                        {"recipe": recipe.content_hash(), "account": plan.inputs.account.account_id}
                    )
                )
                + ".state.sqlite3"
            ),
            **recipe.variant.detector_kwargs(),
        )
        if receipt is None:
            training = materialize_macro_dataset(
                recipe.training_dataset,
                self.evidence,
                availability_policy=recipe.availability_policy,
            )
            mask = (training.signals.index <= recipe.fitted_at) & (
                training.available_at <= recipe.fitted_at
            )
            raw = training.signals.loc[mask]
            evidence = tuple(
                e
                for e in training.evidence
                if e.available_at <= recipe.fitted_at
                and (e.requested_as_of is None or e.requested_as_of <= recipe.fitted_at)
            )
            detector.fit(
                raw,
                plan.inputs.context.model_copy(update={"as_of": recipe.fitted_at}),
                available_at=training.available_at.loc[mask],
                source_ids=training.source_ids.loc[mask],
                evidence=evidence,
                availability_policy=recipe.availability_policy,
            )
            detector.save(path)
            self._trusted_save(
                recipe.content_hash(), path, recipe, "macro", plan.inputs.context.run_id
            )
        else:
            detector.load(path)
        return detector, data

    def _model_availability(self, plan):
        """Bind physical creation and completed consumption stages separately.

        The original creation receipt survives reuse and recovery; a failed
        original stage never becomes a fabricated completed attempt. Actual
        completion of the present model/macro stages gates this desk run.
        """
        status = self.operations.status(plan.inputs.context.run_id)
        values, available = {}, [plan.inputs.context.as_of, datetime.now(timezone.utc)]
        for recipe, kind in [*((r, "numeric") for r in plan.models), (plan.macro_recipe, "macro")]:
            _, receipt = self._trusted_read(recipe, kind)
            if receipt is None:
                raise ValueError("model availability lacks an immutable creation receipt")
            creation = receipt["creation"]
            stage_name = "macro" if kind == "macro" else "models"
            stage = status["stages"][stage_name]
            if stage["state"] != "completed" or not stage["finished_at"]:
                raise ValueError("model consumption stage is not completed")
            with self.operations.connection() as db:
                row = db.execute(
                    "SELECT * FROM stage_attempts WHERE run_id=? AND stage=? AND attempt=?",
                    (creation["run_id"], creation["stage"], creation["attempt"]),
                ).fetchone()
            if row is None or row["started_at"] != creation["stage_started_at"]:
                raise ValueError("model creation stage authority differs")
            created = datetime.fromisoformat(creation["available_at"])
            finished = datetime.fromisoformat(stage["finished_at"])
            if (
                receipt["training_cutoff"] != recipe.fitted_at.isoformat()
                or creation["stage"] != stage_name
                or created.tzinfo is None
                or not datetime.fromisoformat(row["started_at"]) <= created <= finished
                or creation["artifact_completed_at"] != creation["available_at"]
            ):
                raise ValueError("model physical creation chronology differs")
            available.extend([created, finished])
            values[recipe.content_hash()] = {
                "training_cutoff": receipt["training_cutoff"],
                "creation_receipt_hash": digest(canonical(receipt)),
                "creation": creation,
                "creation_stage_state": row["state"],
                "creation_stage_completed_at": row["finished_at"]
                if row["state"] == "completed"
                else None,
                "consumption_stage": {
                    k: stage[k]
                    for k in (
                        "run_id",
                        "stage",
                        "attempt",
                        "started_at",
                        "finished_at",
                        "output_hash",
                    )
                },
                "available_at": max(created, finished).isoformat(),
            }
        context = plan.inputs.context.model_copy(update={"as_of": max(available)})
        if (
            context.as_of - plan.inputs.context.as_of
        ).total_seconds() > self.config.operations.max_cycle_age_seconds:
            raise ValueError("model completion exceeded operating source freshness deadline")
        return {"models": values, "desk_context": context.model_dump(mode="json")}

    @staticmethod
    def _predict_regime(detector, data, context, max_observation_age_seconds):
        mask = (data.signals.index <= context.as_of) & (data.available_at <= context.as_of)
        result = detector.predict(
            data.signals.loc[mask],
            context,
            available_at=data.available_at.loc[mask],
            source_ids=data.source_ids.loc[mask],
            evidence=tuple(e for e in data.evidence if e.available_at <= context.as_of),
            availability_policy=data.availability_policy,
            max_observation_age_seconds=max_observation_age_seconds,
        )
        if result.status != "available":
            raise ValueError("authoritative HMM unavailable; no neutral fallback")
        return result

    def _desk_sources(self, plan, instrument):
        allowed = set(plan.desk_evidence_ids)
        coverage, ids, kinds = {}, set(), {}
        for role, role_kinds in ROLE_KINDS.items():
            coverage[role] = {}
            for kind in role_kinds:
                query = self.evidence.query(
                    as_of=plan.inputs.context.as_of,
                    kind=kind,
                    instrument_id=instrument.instrument_id,
                    current_revisions_only=False,
                )
                eligible = tuple(e for e in query.records if e.record.evidence_id in allowed)
                coverage[role][kind] = {
                    "status": query.status if eligible else "missing",
                    "evidence_ids": [e.record.evidence_id for e in eligible],
                    "excluded": list(query.excluded),
                }
                for item in eligible:
                    if item.document_hash:
                        identity = item.record.evidence_id
                        ids.add(identity)
                        kinds[identity] = (
                            "transcript"
                            if kind == "transcript"
                            else "news"
                            if kind in {"news", "analyst_estimates", "crypto_global"}
                            else "technical"
                            if kind in {"technical", "funding", "open_interest", "liquidity"}
                            else "fundamental"
                        )
        passages = self.evidence.retrieve(
            sorted(ids),
            as_of=plan.inputs.context.as_of,
            max_chars=plan.max_source_chars,
            max_passages=32,
        )
        return tuple(
            source_from_passage(
                self.evidence,
                p,
                as_of=plan.inputs.context.as_of,
                kind=kinds[p.evidence.evidence_id],
            )
            for p in passages
        ), coverage

    def _desk(self, plan, regime, snapshot, returns, models, datasets, now, *, context=None):
        context = context or plan.inputs.context
        correlation = returns.corr()
        if not np.isfinite(correlation.to_numpy()).all():
            raise ValueError("actual correlation unavailable")
        portfolio = PortfolioReviewContext(
            as_of=snapshot.as_of,
            snapshot_id=snapshot.content_hash(),
            portfolio_json=snapshot.model_dump_json(),
            correlation_json=correlation.to_json(),
        )
        result = {}
        identities = {b.instrument.instrument_id: b.instrument for b in plan.inputs.snapshot.bars}
        for key, instrument in sorted(identities.items()):
            sources, coverage = self._desk_sources(plan, instrument)
            quantitative = {}
            for name, model in models.items():
                dataset = datasets[name]
                rows = dataset.features.loc[
                    dataset.features.index.get_level_values("instrument") == key
                ].tail(1)
                ready = dataset.feature_available_at.reindex(rows.index)
                eligible = (
                    not rows.empty
                    and np.isfinite(rows.to_numpy()).all()
                    and ready.notna().all()
                    and (ready <= context.as_of).all()
                )
                quantitative[name] = {
                    "status": "available" if eligible else "unavailable",
                    "model_artifact_hash": model.artifact.content_hash(),
                    "feature_spec_hash": model.artifact.feature_spec_hash,
                    "training_data_hash": model.artifact.training_data_hash,
                    "fitted_at": model.artifact.fitted_at.isoformat(),
                    "training_cutoff": model.artifact.fitted_at.isoformat(),
                }
                if eligible:
                    quantitative[name].update(
                        prediction=float(
                            model.predict(
                                rows,
                                context=context,
                                feature_available_at=ready,
                                availability_policy="captured-vintage-at-fit-1",
                            ).iloc[0]
                        ),
                        signal_at=rows.index[0][0].isoformat(),
                        features={k: float(v) for k, v in rows.iloc[0].items()},
                    )
            advice = analyze_instrument(
                instrument,
                context,
                sources=sources,
                hmm_context=hmm_context_from_result(regime),
                portfolio_context=portfolio,
                qlib_context=json.dumps(quantitative, sort_keys=True, allow_nan=False),
                config=self.config,
                decision_service=self.decisions,
                policy=self.policy,
                llm_factory=self.llm_factory,
            )
            # A typed failure remains visible. Missing authoritative history is
            # not upgraded into a usable recommendation by a receipt-shaped blob.
            receipt = self.history.export(
                advice.content_hash(),
                as_of=max(now, datetime.now(timezone.utc)),
                evidence_store=self.evidence,
                decision_store=self.decisions.store,
                budget_store_path=self.config.agents.budget_store_path,
            )
            archive_hash = self.operations.archive(receipt.model_dump(mode="json"))
            result[key] = {
                "decision": advice.model_dump(mode="json"),
                "coverage": coverage,
                "history_hash": archive_hash,
                "available_at": receipt.available_at.isoformat(),
            }
        return {"decisions": result}

    def _allocation(self, plan, book, context, regime, snapshot):
        policy = plan.allocation
        performance = self.attribution.metrics(
            as_of=context.as_of,
            annualization=policy.performance_annualization,
            window=policy.performance_window,
        )
        # A newly funded epoch has no completed daily observations yet. It
        # cannot supply a performance row to a selector requiring return lineage.
        # Regime/manual/exploration policies remain explicit; exploitation fails
        # naturally when no sufficiently observed rows remain.
        performance = performance.loc[performance.observations > 0]
        desired = select_allocations(
            tuple(s for s in plan.strategies if s.name not in policy.retiring_sleeves),
            context,
            regime,
            mode=policy.selector,
            performance=None if performance.empty else performance,
            manual=policy.manual_strategy,
            exploration=policy.exploration,
            seed=policy.seed,
            resolver=self.resolver,
            approved_baseline_hashes=frozenset(self.config.operations.approved_baseline_hashes),
        )
        capital = partition(
            snapshot.nav, {name: D(str(desired.get(name, 0))) for name in snapshot.sleeves}
        )
        capital = {name: capital.get(name, D(0)) for name in snapshot.sleeves}
        donors, receivers = {}, {}
        for name, sleeve in snapshot.sleeves.items():
            nav = sleeve.portfolio.nav
            if nav > capital[name]:
                donors[name] = min(
                    exact_sum([nav, capital[name].copy_negate()]),
                    exact_sum(
                        [sleeve.portfolio.cash, sleeve.portfolio.reserved_cash.copy_negate()]
                    ),
                )
            elif nav < capital[name]:
                receivers[name] = exact_sum([capital[name], nav.copy_negate()])
        transfers = []
        transfer_hash = digest(
            canonical(
                {
                    "plan_hash": plan.content_hash(),
                    "desired": desired,
                    "snapshot": snapshot.content_hash(),
                }
            )
        )
        # Commit the complete deterministic local funding operation atomically.
        # Existing invested capital cannot be transferred as though it were cash.
        with self.ledger.atomic():
            for donor in sorted(donors):
                for receiver in sorted(receivers):
                    amount = min(donors[donor], receivers[receiver])
                    if amount <= 0:
                        continue
                    event_id = f"cycle:{context.run_id}:capital:{donor}:{receiver}"
                    book.transfer_cash(
                        event_id, donor, receiver, amount, context.as_of, transfer_hash
                    )
                    donors[donor] = exact_sum([donors[donor], amount.copy_negate()])
                    receivers[receiver] = exact_sum([receivers[receiver], amount.copy_negate()])
                    transfers.append(
                        {"event_id": event_id, "from": donor, "to": receiver, "amount": str(amount)}
                    )
        funded = book.snapshot(context.as_of)
        return {
            "desired": desired,
            "transfers": transfers,
            "funded": {name: float(s.portfolio.nav / snapshot.nav) for name, s in funded.items()},
            "sleeves": {name: s.model_dump(mode="json") for name, s in funded.items()},
        }

    def _strategies(self, plan, context, regime, snapshot, datasets, models, desk, allocation):
        identities = {b.instrument.instrument_id: b.instrument for b in plan.inputs.snapshot.bars}
        universe = max(
            (
                u
                for u in plan.inputs.universes
                if u.available_at <= context.as_of
                and u.effective_at <= context.as_of
                and u.universe_version == plan.inputs.universe_version
            ),
            key=lambda u: (u.effective_at, u.available_at),
        )
        decisions = {
            key: AgentDecision.model_validate(value["decision"])
            for key, value in desk["decisions"].items()
        }
        consensus = {
            key: consensus_recommendation(value)
            for key, value in decisions.items()
            if key in universe.instrument_ids
        }
        outputs = {}
        for spec in plan.strategies:
            eligible_dates = datasets[spec.name].features.index.get_level_values("datetime")
            eligible_dates = eligible_dates[eligible_dates <= context.as_of]
            if eligible_dates.empty:
                raise ValueError("strategy has no actual observed feature rows")
            signal_at = eligible_dates.max()
            sleeve = snapshot.sleeves[spec.name]
            holdings = pd.Series(
                {
                    p.instrument.instrument_id: float(p.market_value / sleeve.portfolio.nav)
                    for p in sleeve.portfolio.positions
                },
                dtype=float,
            )
            ages = {}
            for held in sleeve.portfolio.positions:
                if held.opened_at is None:
                    raise ValueError("actual holding age is unavailable")
                sessions = {
                    b.session
                    for b in (
                        *plan.inputs.snapshot.bars,
                        *plan.inputs.opening_events,
                        *plan.inputs.bars,
                    )
                    if b.instrument == held.instrument
                    and held.opened_at <= b.opened_at <= signal_at
                }
                if not sessions:
                    raise ValueError("observed session coverage does not establish holding age")
                ages[held.instrument.instrument_id] = len(sessions)
            output = evaluate_strategy(
                spec,
                datasets[spec.name],
                context,
                universe=universe,
                instruments=tuple(identities.values()),
                holdings=holdings,
                signal_at=signal_at,
                regime=regime,
                model=models.get(spec.name),
                holding_sessions=pd.Series(ages, dtype=float),
                last_rebalanced_at=sleeve.last_rebalanced_at,
                consensus=consensus,
                resolver=self.resolver,
                approved_baseline_hashes=frozenset(self.config.operations.approved_baseline_hashes),
                market_snapshot_hash=plan.inputs.snapshot.content_hash(),
                feature_sampling_policy="latest-available-per-instrument-1",
            )
            for key in output.index:
                decision = decisions.get(key)
                current = holdings.get(key, 0.0)
                self._desk_boundary(output, key, decision, current)
                if (
                    spec.name in plan.allocation.retiring_sleeves
                    and output.loc[key, "action"] not in PASSIVE
                ):
                    output.loc[key, ["action", "target_weight", "reason"]] = [
                        "EXIT",
                        0.0,
                        "explicit_sleeve_retirement",
                    ]
                elif (
                    allocation["desired"].get(spec.name, 0) == 0
                    and output.loc[key, "action"] == "BUY"
                ):
                    output.loc[key, ["action", "target_weight", "reason"]] = [
                        "HOLD",
                        current,
                        "unselected_sleeve_preserves_invested_ownership",
                    ]
            outputs[spec.name] = output
        return {name: _frame(value) for name, value in outputs.items()}

    @staticmethod
    def _desk_boundary(output, key, decision, current):
        action = decision.action.value if decision else "ABSTAIN"
        if action in PASSIVE:
            output.loc[key, ["action", "target_weight", "reason"]] = [
                action,
                current,
                "canonical_desk_preserves_ownership",
            ]
        elif action == "EXIT":
            output.loc[key, ["action", "target_weight", "reason"]] = [
                "EXIT",
                0.0,
                "explicit_desk_exit",
            ]
        elif action == "REDUCE":
            output.loc[key, ["action", "target_weight", "reason"]] = [
                "REDUCE",
                min(current, float(output.loc[key, "target_weight"])),
                "desk_reduction_cap",
            ]
        elif decision.new_exposure_blocked and output.loc[key, "target_weight"] > current:
            output.loc[key, ["action", "target_weight", "reason"]] = [
                "HOLD",
                current,
                "desk_blocks_new_exposure",
            ]

    def _optimize(self, plan, context, snapshot, output_receipts, allocation, returns):
        outputs = {name: _unframe(value) for name, value in output_receipts.items()}
        current = pd.Series(
            {
                p.instrument.instrument_id: float(p.market_value / snapshot.nav)
                for p in snapshot.positions
            },
            dtype=float,
        )
        contributions = {
            name: pd.Series(
                {
                    p.instrument.instrument_id: float(p.market_value / snapshot.nav)
                    for p in s.portfolio.positions
                },
                dtype=float,
            )
            for name, s in snapshot.sleeves.items()
        }
        blended = blend_sleeves(
            outputs,
            allocation["funded"],
            context,
            current_weights=current,
            current_contributions=contributions,
        )
        keys = sorted(set(returns.columns) | set(blended.index))
        desired = blended.target_weight.reindex(keys, fill_value=0.0)
        floors = dict.fromkeys(keys, 0.0)
        fixed = {}
        for name, frame in outputs.items():
            fixed[name] = {}
            for key in frame.index:
                if frame.loc[key, "action"] in PASSIVE:
                    amount = float(contributions[name].get(key, 0))
                    fixed[name][key] = amount
                    floors[key] += amount
        metadata = self._metadata(plan, context.as_of)
        if set(keys) - set(metadata):
            raise ValueError("risk metadata does not cover all actual covariance/target assets")
        metadata = {key: metadata[key] for key in keys}
        limits = AllocationConstraints.for_portfolio(
            snapshot,
            metadata,
            plan.inputs.risk_policy,
            plan.inputs.fill_policy,
            pending=self.ledger.pending(snapshot.account_id),
            asset_mins=floors,
        )
        limits = replace(
            limits,
            asset_caps={key: min(limits.asset_caps[key], float(desired[key])) for key in keys},
        )
        observed = returns.reindex(columns=keys)
        optimizer = PortfolioOptimizer(self.config)
        result = optimizer.optimize(
            plan.allocation.optimizer,
            observed,
            limits=limits,
            expected_returns=observed.mean(),
            views={},
        )
        # Preserve passive actual ownership exactly; distribute the remaining
        # constrained amount only among originally authorized active targets.
        allocated = {name: {} for name in outputs}
        for key in keys:
            weight = float(result.weights.get(key, 0))
            floor = floors[key]
            if weight + 1e-9 < floor:
                raise ValueError("optimizer violated passive ownership lower bound")
            active = {
                name: float(blended.attrs["contributions"][name].get(key, 0))
                for name in outputs
                if key not in fixed[name]
            }
            total = sum(active.values())
            remainder = max(0.0, weight - floor)
            if remainder > total + 1e-9:
                raise ValueError("optimizer introduced exposure outside declared sleeve targets")
            for name, frame in outputs.items():
                if key not in frame.index:
                    continue
                value = fixed[name].get(
                    key, remainder * active.get(name, 0) / total if total else 0.0
                )
                allocated[name][key] = value
                if key not in fixed[name]:
                    frame.loc[key, "target_weight"] = (
                        value / allocation["funded"][name] if allocation["funded"][name] else 0.0
                    )
        # Re-run the canonical blending invariants on the optimizer-adjusted rows.
        blended = blend_sleeves(
            outputs,
            allocation["funded"],
            context,
            current_weights=current,
            current_contributions=contributions,
        )
        return {
            "frame": _frame(blended),
            "outputs": {name: _frame(frame) for name, frame in outputs.items()},
            "optimizer": {
                "method": plan.allocation.optimizer,
                "diagnostics": result.diagnostics,
                "weights": {str(k): float(v) for k, v in result.weights.items()},
                "asset_mins": floors,
                "covariance_sampling_hash": returns.attrs["sampling_hash"],
                "covariance_sampling": returns.attrs["sampling_receipt"],
            },
        }

    def run(self, plan: PaperCyclePlan, *, now=None) -> CycleResult:
        plan = PaperCyclePlan.model_validate(plan.model_dump())
        if cycle_config_hash(self.config) != plan.inputs.context.config_hash:
            raise ValueError("runtime public configuration differs from frozen plan")
        start = now or datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        if start.tzinfo is None:
            raise ValueError("explicit timezone-aware operating clock required")
        events = {}
        for bar in plan.inputs.bars:
            events.setdefault(bar.opened_at, []).append(("open", bar))
            events.setdefault(bar.closed_at, []).append(("close", bar))
        for event in plan.inputs.opening_events:
            events.setdefault(event.opened_at, []).append(("opening_event", event))
        if plan.inputs.opening_events:
            for bar in plan.inputs.snapshot.bars:
                if plan.inputs.account.opened_at <= bar.closed_at:
                    events.setdefault(bar.closed_at, []).append(("market_close", bar))
        for action in plan.inputs.actions:
            events.setdefault(action.effective_at, []).append(("action", action))
        latest_terms = {}
        for terms in sorted(plan.inputs.cash_dividends, key=lambda t: t.available_at):
            latest_terms[terms.action_id] = terms
        for terms in latest_terms.values():
            if terms.entitlement_at <= plan.inputs.context.as_of:
                events.setdefault(terms.entitlement_at, []).append(("dividend_entitlement", terms))
        for payment in plan.inputs.dividend_payments:
            events.setdefault(payment.available_at, []).append(("dividend_payment", payment))
        for distribution in plan.inputs.dividend_distributions:
            events.setdefault(distribution.available_at, []).append(
                ("dividend_distribution", distribution)
            )
        event_names = {at: "observe:" + at.isoformat() for at in sorted(events)}
        stages = (
            "verified_inputs",
            "account",
            *(("income_admission",) if plan.inputs.income_policy is not None else ()),
            *event_names.values(),
            "reconcile",
            "features",
            "models",
            "macro",
            "model_availability",
            "agent_desk",
            "decision_clock",
            "sizing_snapshot",
            "allocation",
            "strategies",
            "optimization",
            "target",
            "prepare",
            "attribution",
            "dispatch",
            "completion",
        )
        manifest = RunManifest(
            context=plan.inputs.context,
            kind="paper_cycle",
            input_hash=plan.content_hash(),
            scheduled_for=plan.scheduled_for,
            stages=stages,
        )
        context, target, snapshot, prepared = None, None, None, []
        with self.operations.exclusive("account:" + plan.inputs.account.account_id):
            for previous in self.operations.account_runs(plan.inputs.account.account_id):
                if previous["run_id"] == plan.inputs.context.run_id:
                    continue
                prior_manifest = self.operations.manifest(previous["run_id"])
                if prior_manifest.scheduled_for == plan.scheduled_for and previous["stages"].get(
                    "account", {}
                ).get("state") in {"completed", "running", "uncertain"}:
                    raise ValueError("account schedule already has an admitted economic cycle")
            manifest_hash = self.operations.admit(manifest, plan.model_dump(mode="json"))

            def stage(name, operation, recoverable=True):
                return self.operations.stage(
                    plan.inputs.context.run_id,
                    name,
                    {"plan_hash": plan.content_hash()},
                    operation,
                    recoverable=recoverable,
                )

            prior = self.operations.status(plan.inputs.context.run_id)
            if prior["status"] == "completed":
                receipt = self.operations.artifact(prior["stages"]["completion"]["output_hash"])
                return CycleResult.model_validate(receipt)
            try:
                # Recovery cannot inherit an earlier freshness approval. This
                # check precedes every resumed mutation, even when the pure
                # verified_inputs stage already has a durable cached result.
                age = (start - plan.inputs.context.as_of).total_seconds()
                if prior["stages"]["verified_inputs"].get("state") == "completed" and (
                    age < 0 or age > self.config.operations.max_cycle_age_seconds
                ):
                    raise ValueError("future or stale operating input cutoff")
                if (
                    prior["stages"]["verified_inputs"].get("state") == "completed"
                    and plan.inputs.income_policy is None
                ):
                    raise ValueError("fresh paper cycles require an explicit sourced income policy")

                def verify():
                    if cycle_config_hash(self.config) != plan.inputs.context.config_hash:
                        raise ValueError(
                            "runtime public configuration hash differs from frozen input"
                        )
                    age = (start - plan.inputs.context.as_of).total_seconds()
                    if age < 0 or age > self.config.operations.max_cycle_age_seconds:
                        raise ValueError("future or stale operating input cutoff")
                    source_proofs = self._verify_inputs(
                        plan.inputs,
                        dict(plan.provenance),
                        plan.rule_sources,
                        plan.income_authorities,
                    )
                    for spec in plan.strategies:
                        require_strategy_eligible(
                            spec,
                            plan.inputs.context,
                            resolver=self.resolver,
                            approved_baseline_hashes=frozenset(
                                self.config.operations.approved_baseline_hashes
                            ),
                        )
                    for recipe in (*plan.models, plan.macro_recipe):
                        self._approved(
                            recipe,
                            plan.inputs.context,
                            "regime:macro"
                            if recipe is plan.macro_recipe
                            else "model:" + recipe.strategy,
                        )
                    return {
                        "inputs_hash": plan.inputs.content_hash(),
                        "verified_at": start.isoformat(),
                        "source_proofs": source_proofs,
                    }

                stage("verified_inputs", verify)

                def account():
                    prior = self._prior_anchor(plan)
                    self.ledger.create_account(plan.inputs.account)
                    book = SleeveBook(self.ledger, plan.inputs.account.account_id)
                    book.initialize(dict(plan.inputs.initial_capital))
                    anchor = prior or book.snapshot(plan.inputs.account.opened_at)
                    if prior is None:
                        for name, sleeve in anchor.items():
                            if sleeve.portfolio.nav > 0:
                                self._funded_anchor(name, sleeve, plan.inputs.context.as_of)
                    return {
                        "anchor": {name: s.model_dump(mode="json") for name, s in anchor.items()}
                    }

                anchor = _sleeves(stage("account", account)["anchor"])
                book = SleeveBook(self.ledger, plan.inputs.account.account_id)
                if plan.inputs.income_policy is not None:
                    stage("income_admission", lambda: self._admit_income(plan, book, anchor))
                rules = {r.instrument.instrument_id: r for r in plan.inputs.rules}
                venue = PaperVenue(
                    self.ledger,
                    book.account_id,
                    rules,
                    fill_policy=plan.inputs.fill_policy,
                    risk_policy=plan.inputs.risk_policy,
                )
                execution = VenueExecution(
                    self.ledger,
                    book.account_id,
                    {"paper": venue},
                    {key: "paper" for key in rules},
                    owner="direct",
                    risk_policy=plan.inputs.risk_policy,
                    fill_policy=plan.inputs.fill_policy,
                )
                for at, name in event_names.items():
                    ordered = self._ordered_observations(events[at])
                    if at < next(iter(anchor.values())).portfolio.as_of:
                        # Overlapping historical bundles must exactly match the
                        # previously consumed economic events; no new past fills.
                        def historical(ordered=ordered):
                            for kind, event in ordered:
                                if kind.startswith("dividend_"):
                                    # Canonical APIs verify the exact persisted
                                    # entitlement/payment before idempotent return.
                                    # A missing old event hits the published-NAV
                                    # fence; no historical cash is synthesized.
                                    self._observe_income(plan, book, kind, event, at)
                                    continue
                                identity = (
                                    "action:" + event.action_id
                                    if kind == "action"
                                    else self._market_close_id(event)
                                    if kind == "market_close"
                                    else "opening-event:" + event.event_id
                                    if kind == "opening_event"
                                    else kind + ":" + event.event_id
                                )
                                row = self.ledger.db.execute(
                                    "SELECT payload FROM events WHERE account=? AND id=?",
                                    (book.account_id, identity),
                                ).fetchone()
                                same = row is not None and row[0] == event.model_dump_json()
                                if row is not None and kind == "market_close":
                                    # A later identical capture can corroborate a
                                    # price without rewriting the original NAV's
                                    # immutable source receipt. Revisions reject.
                                    prior_bar = type(event).model_validate_json(row[0])
                                    same = prior_bar.model_dump(
                                        exclude={"evidence_id"}
                                    ) == event.model_dump(exclude={"evidence_id"})
                                if (
                                    not same
                                    and row is None
                                    and kind in {"open", "opening_event"}
                                    and plan.inputs.receipts
                                ):
                                    # Only a source-proved alias of consumed native
                                    # liquidity can cross an already committed NAV.
                                    # Canonical settlement records the alias, never
                                    # a second fill/mark or another NAV period.
                                    with self.ledger.atomic():
                                        fills = self._settle_observed_open(
                                            plan, venue, execution, event
                                        )
                                        if fills:
                                            raise ValueError(
                                                "historical opening alias cannot introduce a fill"
                                            )
                                    same = True
                                if not same:
                                    raise ValueError(
                                        "historical economic event is missing or changed"
                                    )
                            return {
                                "anchor": {k: v.model_dump(mode="json") for k, v in anchor.items()}
                            }

                        receipt = stage(name, historical)
                    else:
                        receipt = stage(
                            name,
                            lambda: self._observe(
                                plan, book, venue, execution, anchor, ordered, at
                            ),
                        )
                    anchor = _sleeves(receipt["anchor"])

                def reconcile():
                    execution.reconcile_all(as_of=plan.inputs.context.as_of)
                    for pending in self.ledger.pending(book.account_id):
                        intent = pending["intent"]
                        if execution.state(intent.intent_id)["state"] in {
                            "ambiguous",
                            "submitting",
                            "cancel_pending",
                            "reconciliation_required",
                        }:
                            raise RecoveryRequired("unresolved venue outcome blocks reallocation")
                        execution.cancel(intent.intent_id, as_of=plan.inputs.context.as_of)
                    execution.reconcile_all(as_of=plan.inputs.context.as_of)
                    if self.ledger.pending(book.account_id):
                        raise RecoveryRequired("pending orders remain reserved after cancellation")
                    return {
                        "pending": 0,
                        "snapshot": self.ledger.snapshot(
                            book.account_id,
                            plan.inputs.context.as_of,
                            plan.inputs.risk_policy.max_price_age_seconds,
                        ).model_dump(mode="json"),
                    }

                reconciled = stage("reconcile", reconcile)
                datasets = self.materialize_features(plan)
                stage(
                    "features",
                    lambda: {
                        name: {
                            "snapshot_hash": d.snapshot_hash,
                            "feature_spec_hash": d.feature_spec_hash,
                            "frame_hash": frame_hash(d.features),
                        }
                        for name, d in datasets.items()
                    },
                )
                receipts = stage(
                    "models",
                    lambda: self._fit_models(plan),
                )
                models = {}
                for recipe in plan.models:
                    path, receipt = self._trusted_read(recipe, "numeric")
                    if receipt != receipts[recipe.strategy]:
                        raise ValueError("model stage differs from trusted local receipt")
                    models[recipe.strategy] = ModelWrapper(
                        recipe.model_name, self._model_config(recipe)
                    ).load(str(path))
                live_macro = {}

                def macro_stage():
                    detector, macro = self._macro(plan)
                    live_macro.update(detector=detector, data=macro)
                    return self._predict_regime(
                        detector,
                        macro,
                        plan.inputs.context,
                        plan.macro_dataset.max_source_age_seconds,
                    ).model_dump(mode="json")

                regime = CausalRegimeResult.model_validate(stage("macro", macro_stage))
                if live_macro:
                    detector, macro = live_macro["detector"], live_macro["data"]
                else:
                    if self._trusted_read(plan.macro_recipe, "macro")[1] is None:
                        raise ValueError("completed macro stage lacks its trusted fitted artifact")
                    detector, macro = self._macro(plan)
                returns = self._returns(plan)
                snapshot = PortfolioSnapshot.model_validate(reconciled["snapshot"])
                availability = stage("model_availability", lambda: self._model_availability(plan))
                desk_context = RunContext.model_validate(availability["desk_context"])
                desk = stage(
                    "agent_desk",
                    lambda: self._desk(
                        plan,
                        self._predict_regime(
                            detector, macro, desk_context, plan.macro_dataset.max_source_age_seconds
                        ),
                        snapshot,
                        returns,
                        models,
                        datasets,
                        start,
                        context=desk_context,
                    ),
                    recoverable=False,
                )

                def decision_clock():
                    at = max(
                        start + timedelta(seconds=time.monotonic() - started_monotonic),
                        datetime.now(timezone.utc),
                        *(
                            datetime.fromisoformat(v["available_at"])
                            for v in desk["decisions"].values()
                        ),
                    )
                    if (
                        at - plan.inputs.context.as_of
                    ).total_seconds() > self.config.operations.max_cycle_age_seconds:
                        raise ValueError(
                            "completed desk exceeded operating source freshness deadline"
                        )
                    return plan.inputs.context.model_copy(
                        update={
                            "as_of": at,
                            "account_id": snapshot.account_id,
                            "data_snapshot_ids": tuple(
                                sorted(
                                    set(plan.inputs.context.data_snapshot_ids)
                                    | {d.snapshot_hash for d in datasets.values()}
                                )
                            ),
                        }
                    ).model_dump(mode="json")

                context = RunContext.model_validate(stage("decision_clock", decision_clock))
                regime = self._predict_regime(
                    detector, macro, context, plan.macro_dataset.max_source_age_seconds
                )
                snapshot = PortfolioSnapshot.model_validate(
                    stage(
                        "sizing_snapshot",
                        lambda: (
                            self.ledger.snapshot(
                                book.account_id,
                                context.as_of,
                                plan.inputs.risk_policy.max_price_age_seconds,
                                persist=False,
                            )
                            .model_copy(update={"sleeves": book.snapshot(context.as_of)})
                            .model_dump(mode="json")
                        ),
                    )
                )
                allocation = stage(
                    "allocation",
                    lambda: self._allocation(plan, book, context, regime, snapshot),
                    recoverable=False,
                )
                snapshot = snapshot.model_copy(update={"sleeves": _sleeves(allocation["sleeves"])})
                outputs = stage(
                    "strategies",
                    lambda: self._strategies(
                        plan, context, regime, snapshot, datasets, models, desk, allocation
                    ),
                )
                optimized = stage(
                    "optimization",
                    lambda: self._optimize(plan, context, snapshot, outputs, allocation, returns),
                )
                instruments = tuple(
                    {
                        b.instrument.instrument_id: b.instrument for b in plan.inputs.snapshot.bars
                    }.values()
                )
                target = PortfolioTarget.model_validate(
                    stage(
                        "target",
                        lambda: target_from_blended_frame(
                            _unframe(optimized["frame"]),
                            context,
                            instruments,
                            target_id="cycle:" + context.run_id,
                            current_snapshot=snapshot,
                            outputs={
                                name: _unframe(value)
                                for name, value in optimized["outputs"].items()
                            },
                        ).model_dump(mode="json"),
                    )
                )
                prepared = stage(
                    "prepare",
                    lambda: execution.prepare_target(
                        target, self._metadata(plan, context.as_of), sleeve_book=book
                    ),
                )
                stage(
                    "attribution",
                    lambda: {
                        "anchor": self._record_period(book, anchor, context.as_of, context.as_of),
                        "snapshot_hash": self.ledger.snapshot(
                            book.account_id,
                            context.as_of,
                            plan.inputs.risk_policy.max_price_age_seconds,
                        ).content_hash(),
                    },
                )
                stage("dispatch", lambda: self._dispatch(execution, datetime.now(timezone.utc)))
                status, reason = (
                    "completed",
                    "paper target durably prepared; eligible future observed opens execute it",
                )
            except Exception as exc:
                status = "recovery_required" if isinstance(exc, RecoveryRequired) else "blocked"
                reason = f"{type(exc).__name__}: {exc}"
                self.operations.alert(
                    run_id=plan.inputs.context.run_id,
                    severity="critical",
                    code="paper_cycle_" + status,
                    payload={"error_type": type(exc).__name__, "plan_hash": plan.content_hash()},
                )
            state = self.operations.status(plan.inputs.context.run_id)
            result = CycleResult(
                run_id=plan.inputs.context.run_id,
                manifest_hash=manifest_hash,
                plan_hash=plan.content_hash(),
                status=status,
                reason=reason,
                decision_context=context,
                target_hash=target.content_hash() if target else None,
                prepared_intent_ids=tuple(item["intent"] for item in prepared),
                snapshot_hash=snapshot.content_hash() if snapshot else None,
                stage_outputs=tuple(
                    (name, row["output_hash"])
                    for name, row in state["stages"].items()
                    if row.get("output_hash")
                ),
            )
            if status == "completed":
                stage("completion", lambda: result.model_dump(mode="json"))
            return result
