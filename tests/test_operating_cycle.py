"""Offline full-cycle tests: real source stores, learned models, HMM and ledger.

All synthetic economic/source/provider fixtures live here, never in runtime.
"""

from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.data.calendars import sessions
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.data.features.contextual import NumericEvidenceBinding
from src.domain.market import MarketBar, MarketSnapshot
from src.domain.portfolio import (
    AccountSpec,
    FillPolicy,
    InstrumentRisk,
    OpeningEvent,
    RiskPolicy,
    UniverseEligibility,
)
from src.execution.ledger import PortfolioLedger
from src.execution.venue import VenueRules
from src.operations.store import OperationsStore
from src.research.evidence import BoundArtifact, SourceBinding, expected_artifacts
from src.research.regime_experiments import (
    FrozenEvidence,
    HMMVariantSpec,
    MacroDatasetSpec,
    MacroRawObservation,
)
from src.research.registry import ResearchRegistry
from src.strategies.specification import baseline_specs

from src.core.strategy_tracker import AttributedPerformanceStore
from src.domain.common import EvidenceRecord, Instrument, RunContext
from src.operations.contracts import (
    AllocationPolicy,
    MacroRecipe,
    ModelRecipe,
    ObservedOpenReceipt,
    PaperCycleInputs,
    PaperCyclePlan,
)
from src.operations.cycle import PaperCycleService, cycle_config_hash, training_frame_hash
from src.utils.config import AppConfig, ModelPricing
from tests.test_agent_desk_invariants import GenerativeFixture, jev_service

D = Decimal


def put(store, key, payload, at, *, kind="fixture", instrument_ids=(), text=None):
    raw = canonical_bytes(payload)
    record = EvidenceRecord(
        evidence_id=key,
        provider="synthetic-test-only",
        provider_version="1",
        source_uri="test://" + key,
        raw_hash=digest(raw),
        ingested_at=at,
        available_at=at,
        observed_at=at,
        requested_as_of=at,
        availability_basis="documented",
        quality="usable",
        instrument_ids=instrument_ids,
    )
    return store.put(record, raw, kind=kind, logical_key=key, normalized=payload, document=text)


def prefix(snapshot, at):
    bars = tuple(b for b in snapshot.bars if b.closed_at <= at)
    ids = {b.evidence_id for b in bars}
    return MarketSnapshot(
        context=snapshot.context.model_copy(update={"as_of": at}),
        bars=bars,
        evidence=tuple(e for e in snapshot.evidence if e.evidence_id in ids),
    )


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("cycle-synthetic-archive")
    store = EvidenceStore(path / "evidence.sqlite")
    schedule = list(sessions("2025-06-02", "2027-06-01", calendar="XNYS").items())[:365]
    dates = pd.DatetimeIndex([bounds[1] for _, bounds in schedule])
    rng = np.random.default_rng(152)
    instruments = tuple(
        Instrument(
            instrument_id=f"equity:{i}",
            symbol=f"TEST{i}",
            asset_class="equity",
            quote_currency="USD",
            venue="test",
        )
        for i in range(3)
    ) + (
        Instrument(
            instrument_id="crypto:btc",
            symbol="BTCUSD",
            asset_class="crypto",
            quote_currency="USD",
            venue="test",
        ),
    )
    bars, records, numeric = [], [], []
    for j, instrument in enumerate(instruments):
        actual_schedule = (
            schedule
            if j < 3
            else [
                (day, bounds)
                for day, bounds in sessions(
                    dates[0].date().isoformat(), dates[-1].date().isoformat(), calendar="24/7"
                ).items()
                if bounds[1] <= dates[-1]
            ]
        )
        prices = (100 + j * 20) * np.cumprod(1 + rng.normal(0.001, 0.01, len(actual_schedule)))
        for i, (session, (opened, closed)) in enumerate(actual_schedule):
            price = D(str(prices[i]))
            key = f"bar-{j}-{i}"
            bar = MarketBar(
                instrument=instrument,
                opened_at=opened,
                closed_at=closed,
                open=price,
                high=price * D("1.01"),
                low=price * D(".99"),
                close=price,
                volume=100000 + i * 1000,
                vwap=price,
                evidence_id=key,
                price_basis="raw",
                market_type="spot" if j == 3 else "equity",
                calendar="24/7" if j == 3 else "XNYS",
                session=session,
            )
            envelope = put(
                store,
                key,
                bar.model_dump(mode="json", exclude={"evidence_id"}),
                closed,
                kind="bar",
                instrument_ids=(instrument.instrument_id,),
            )
            bars.append(bar)
            records.append(envelope.record)
            values = {
                "pe": 15 + i / 1000,
                "roic": 0.2 + 0.01 * np.sin(i),
                "free_cash_flow": 10000000 + i,
                "benchmark_close": 100 * np.exp(0.001 * i + 0.01 * np.sin(i)),
                "funding_rate": 0.001 * np.sin(i),
                "open_interest": 10000 + 50 * i + 20 * np.cos(i),
                "btc_dominance": 0.5 + 0.03 * np.cos(i / 4),
                "market_type": "perpetual" if j == 3 else "equity",
            }
            source = put(
                store,
                f"numeric-{j}-{i}",
                values,
                closed,
                kind="cycle_numeric",
                instrument_ids=(instrument.instrument_id,),
            )
            numeric.append(
                FrozenEvidence(
                    evidence_id=source.record.evidence_id, envelope_hash=source.content_hash()
                )
            )
    context = RunContext(
        run_id="synthetic-archive",
        mode="research",
        as_of=dates[-1],
        config_hash="0" * 64,
        code_version="test",
    )
    snapshot = MarketSnapshot(
        context=context,
        bars=tuple(sorted(bars, key=lambda b: (b.closed_at, b.instrument.instrument_id))),
        evidence=tuple(records),
    )
    bindings = []
    for instrument in instruments:
        names = (
            ("funding_rate", "open_interest", "btc_dominance", "benchmark_close")
            if instrument.asset_class == "crypto"
            else ("pe", "roic", "free_cash_flow", "benchmark_close")
        )
        for name in names:
            bindings.append(
                NumericEvidenceBinding(
                    name=name,
                    target_instrument_id=instrument.instrument_id,
                    source_instrument_id=instrument.instrument_id,
                    source_market="perpetual"
                    if name in ("funding_rate", "open_interest")
                    else "instrument",
                    kind="cycle_numeric",
                    value_path=(name,),
                    max_age_seconds=86400,
                )
            )
    equity_snapshot = snapshot.model_copy(
        update={
            "bars": tuple(b for b in snapshot.bars if b.instrument.asset_class == "equity"),
            "evidence": tuple(
                e for e in snapshot.evidence if e.instrument_ids[0].startswith("equity:")
            ),
        }
    )
    member = UniverseEligibility(
        universe_version="fixture",
        data_snapshot_id=equity_snapshot.content_hash(),
        effective_at=dates[0] - timedelta(days=1),
        available_at=dates[0] - timedelta(days=1),
        instrument_ids=tuple(i.instrument_id for i in instruments[:3]),
    )
    member_source = put(store, "membership", [member.model_dump(mode="json")], member.available_at)
    macro_sources = [
        FrozenEvidence(
            evidence_id=e.evidence_id,
            envelope_hash=store.get(e.evidence_id, as_of=dates[-1]).content_hash(),
        )
        for e in equity_snapshot.evidence
    ]
    macro_sources.append(
        FrozenEvidence(evidence_id="membership", envelope_hash=member_source.content_hash())
    )
    rng = np.random.default_rng(34)
    raw = pd.DataFrame(
        {
            "vix": 20 * np.exp(np.cumsum(rng.normal(0, 0.03, len(dates)))),
            "dgs10": 3 + rng.normal(0, 0.3, len(dates)),
            "dgs2": 2 + rng.normal(0, 0.3, len(dates)),
            **{
                key: 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(dates))))
                for key in ("sp500", "dxy", "hyg", "tlt")
            },
        },
        index=dates,
    )
    observations = []
    for number, (at, row) in enumerate(raw.iterrows()):
        key = f"macro-{number}"
        source = put(store, key, row.to_dict(), at)
        macro_sources.append(FrozenEvidence(evidence_id=key, envelope_hash=source.content_hash()))
        observations.append(
            MacroRawObservation(
                availability_policy="captured-vintage-at-fit-1",
                at=at,
                values=tuple(
                    (name, SourceBinding(evidence_id=key, pointer="/" + name, available_at=at))
                    for name in row.index
                ),
            )
        )

    def macro_at(at):
        return MacroDatasetSpec(
            availability_policy="captured-vintage-at-fit-1",
            breadth_snapshot=prefix(equity_snapshot, at),
            universe_version="fixture",
            universes=(member,),
            universe_sources=(
                SourceBinding(evidence_id="membership", available_at=member.available_at),
            ),
            raw=tuple(o for o in observations if o.at <= at),
            sources=tuple(
                s
                for s in macro_sources
                if store.get(s.evidence_id, as_of=dates[-1]).record.available_at <= at
            ),
        )

    docs = []
    for instrument in instruments:
        for kind in (
            "income_statement",
            "balance_sheet",
            "cash_flow",
            "ratios",
            "filing",
            "token_context",
            "news",
            "analyst_estimates",
            "transcript",
            "crypto_global",
            "technical",
            "funding",
            "open_interest",
            "earnings",
            "liquidity",
        ):
            key = f"doc-{instrument.instrument_id}-{kind}"
            put(
                store,
                key,
                {"text": "Demand fell 10%."},
                dates[350],
                kind=kind,
                instrument_ids=(instrument.instrument_id,),
                text="Demand fell 10%.",
            )
            docs.append(key)
    return SimpleNamespace(
        store=store,
        snapshot=snapshot,
        dates=dates,
        bindings=tuple(bindings),
        numeric=tuple(numeric),
        macro_at=macro_at,
        instruments=instruments,
        docs=tuple(docs),
    )


def prove(plan, store, registry, *, observed=False):
    opening = (
        SourceBinding(
            evidence_id=plan.inputs.receipts[0].evidence_id,
            available_at=plan.inputs.receipts[0].available_at,
            pointer="/opening_liquidity" if plan.inputs.opening_events else "",
        )
        if plan.inputs.receipts
        else None
    )
    if observed:
        from src.operations.contracts import ObservedOpenReceipt

        key = "received-opening-" + plan.inputs.content_hash()
        at = plan.inputs.context.as_of
        payload = expected_artifacts(plan.inputs)
        envelope = put(
            store,
            key,
            {
                "opening_liquidity": payload["opening_liquidity"],
                "opening_events": payload["opening_events"],
            },
            at,
        )
        receipts = tuple(
            ObservedOpenReceipt(
                event_id=b.event_id,
                evidence_id=key,
                evidence_hash=envelope.content_hash(),
                observed_at=at,
                available_at=at,
            )
            for b in (plan.inputs.opening_events or plan.inputs.bars)
        )
        plan = plan.model_copy(
            update={"inputs": plan.inputs.model_copy(update={"receipts": receipts})}
        )
        opening = SourceBinding(evidence_id=key, available_at=at, pointer="/opening_liquidity")
    names = {
        "universe_history": "universe_history_id",
        "availability_audit": "availability_audit_id",
        "calendar": "calendar_artifact_id",
        "corporate_actions": "corporate_actions_id",
        "execution_bars": "execution_bars_id",
        "opening_events": "opening_events_id",
        "opening_liquidity": "opening_liquidity_id",
        "instrument_metadata": "instrument_metadata_id",
    }
    refs = []
    for kind, value in expected_artifacts(plan.inputs).items():
        at = (
            min(u.available_at for u in plan.inputs.universes)
            if kind in ("universe_history", "calendar", "opening_liquidity", "instrument_metadata")
            else plan.inputs.context.as_of
        )
        key = "proof-" + plan.inputs.content_hash() + "-" + kind
        if kind in {"opening_liquidity", "opening_events"} and opening is not None:
            source = opening.model_copy(update={"pointer": "/" + kind})
        else:
            put(store, key, value, at)
            source = SourceBinding(evidence_id=key, available_at=at)
        bound = BoundArtifact(
            kind=kind,
            inputs_hash=plan.inputs.content_hash(),
            value_json=canonical_bytes(value).decode(),
            sources=(source,),
        )
        refs.append((names[kind], registry.archive_contract(bound)))
    rules_key = "rules-" + plan.inputs.content_hash()
    put(
        store,
        rules_key,
        [r.model_dump(mode="json") for r in plan.inputs.rules],
        plan.inputs.context.as_of,
    )
    return plan.model_copy(
        update={
            "provenance": tuple(refs),
            "rule_sources": (
                SourceBinding(evidence_id=rules_key, available_at=plan.inputs.context.as_of),
            ),
        }
    )


@pytest.fixture(scope="module")
def runtime(archive, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cycle-runtime")
    cfg = AppConfig(
        _env_file=None, output_dir=str(tmp_path / "output"), log_dir=str(tmp_path / "logs")
    )
    cfg.memory.enabled = False
    price = ModelPricing(
        provider=cfg.agents.llm_provider,
        model="fixture-chat",
        price_version="synthetic-test-rate",
        source_uri="test://pricing",
        effective_at=archive.dates[0],
        input_usd_per_million=D(".01"),
        output_usd_per_million=D(".02"),
    )
    cfg.agents = cfg.agents.model_copy(
        update={
            "llm_model": "fixture-chat",
            "deep_think_model": "fixture-chat",
            "pricing": [price],
            "budget_store_path": str(tmp_path / "budget.sqlite"),
            "max_tokens_per_analysis": 1000000,
            "max_tokens_per_day": 10000000,
            "max_requests_per_day": 2000,
        }
    )
    registry = ResearchRegistry(tmp_path / "registry.sqlite")
    ops = OperationsStore(tmp_path / "operations.sqlite")
    ledger = PortfolioLedger(tmp_path / "ledger.sqlite")
    attribution = AttributedPerformanceStore(tmp_path / "attribution.sqlite")
    calls = []

    def factory(role, **_):
        return GenerativeFixture(role, calls)

    # All actual six baseline schemas/models/features; provider boundary only mocked.
    service = PaperCycleService(
        cfg,
        operations_store=ops,
        evidence_store=archive.store,
        registry=registry,
        ledger=ledger,
        attribution_store=attribution,
        artifact_dir=tmp_path / "models",
        decision_service=jev_service(tmp_path),
        llm_factory=factory,
    )
    specs = tuple(baseline_specs().values())
    train = prefix(archive.snapshot, archive.dates[348])
    training_universe = UniverseEligibility(
        universe_version="fixture",
        data_snapshot_id=train.content_hash(),
        effective_at=train.context.as_of,
        available_at=train.context.as_of,
        instrument_ids=tuple(i.instrument_id for i in archive.instruments),
    )
    put(
        archive.store,
        "frozen-training-membership",
        [training_universe.model_dump(mode="json")],
        train.context.as_of,
    )
    parameters = {
        "Linear": (),
        "LightGBM": (("n_estimators", 8), ("n_jobs", 1), ("min_child_samples", 5)),
        "CatBoost": (("iterations", 8), ("thread_count", 1)),
        "XGBoost": (("n_estimators", 8), ("n_jobs", 1)),
    }
    recipes = tuple(
        ModelRecipe(
            strategy=s.name,
            model_name=s.model_name,
            parameters=parameters[s.model_name],
            feature_spec_hash=s.feature_spec_hash,
            data_snapshot_hash="0" * 64,
            training_frame_hash="0" * 64,
            training_snapshot=train,
            training_universe=training_universe,
            training_universe_sources=(
                SourceBinding(
                    evidence_id="frozen-training-membership", available_at=train.context.as_of
                ),
            ),
            fitted_at=train.context.as_of,
            label_horizon=1,
            code_version="test",
        )
        for s in specs
        if s.model_name
    )
    macro_train = archive.macro_at(archive.dates[301])
    macro_recipe = MacroRecipe(
        dataset_hash=macro_train.content_hash(),
        training_dataset=macro_train,
        variant=HMMVariantSpec(
            normalization_window=20, n_init=1, n_iter=10, confidence_threshold=0.25
        ),
        fitted_at=archive.dates[301],
        code_version="test",
    )
    account = AccountSpec(
        account_id="paper:operating-test",
        mode="paper",
        base_currency="USD",
        initial_cash=100000,
        opened_at=next(
            b.opened_at
            for b in archive.snapshot.bars
            if b.instrument.asset_class == "equity" and b.closed_at == archive.dates[356]
        )
        - timedelta(seconds=1),
    )

    def make_plan(
        number=357, *, run_id="cycle-1", optimizer="risk_parity", selector="regime_matched"
    ):
        at = archive.dates[number] + timedelta(minutes=15)
        market = prefix(archive.snapshot, archive.dates[number])
        context = RunContext(
            run_id=run_id,
            mode="paper",
            as_of=at,
            config_hash=cycle_config_hash(cfg),
            code_version="test",
            account_id=account.account_id,
            data_snapshot_ids=(market.content_hash(),),
        )
        universe = UniverseEligibility(
            universe_version="fixture",
            data_snapshot_id=market.content_hash(),
            effective_at=archive.dates[0] - timedelta(days=1),
            available_at=archive.dates[0] - timedelta(days=1),
            instrument_ids=tuple(i.instrument_id for i in archive.instruments),
        )
        selected = tuple(b for b in archive.snapshot.bars if account.opened_at <= b.opened_at <= at)
        opening_events = tuple(
            OpeningEvent(
                event_id="opening:" + b.evidence_id,
                instrument=b.instrument,
                opened_at=b.opened_at,
                price=b.open,
                opening_volume=100000,
                calendar=b.calendar,
                session=b.session,
                calendar_version="synthetic-test-only",
                session_open=b.opened_at,
                session_close=b.closed_at,
            )
            for b in selected
        )
        opening_key = f"actual-opening-receipt:{run_id}:{number}"
        liquidity = [
            {
                k: e.model_dump(mode="json")[k]
                for k in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")
            }
            for e in opening_events
        ]
        envelope = put(
            archive.store,
            opening_key,
            {
                "opening_liquidity": liquidity,
                "opening_events": [e.model_dump(mode="json") for e in opening_events],
            },
            at,
        )
        receipts = tuple(
            ObservedOpenReceipt(
                event_id=e.event_id,
                evidence_id=opening_key,
                evidence_hash=envelope.content_hash(),
                observed_at=at,
                available_at=at,
            )
            for e in opening_events
        )
        metadata = tuple(
            InstrumentRisk(
                instrument=b.instrument,
                sector=b.instrument.instrument_id,
                adv_base=100000000,
                as_of=b.opened_at,
                lot_size=".001",
            )
            for b in selected
        )
        rules = tuple(
            VenueRules(
                venue="paper",
                instrument=i,
                symbol=i.symbol,
                available_at=at,
                quantity_step=".001",
                price_tick=".01",
                minimum_quantity=D(0),
                minimum_notional=D(0),
                market_type="spot" if i.asset_class == "crypto" else "equity",
                active=True,
            )
            for i in archive.instruments
        )
        inputs = PaperCycleInputs(
            context=context,
            snapshot=market,
            account=account,
            opening_events=opening_events,
            receipts=receipts,
            metadata=metadata,
            universes=(universe,),
            universe_version="fixture",
            fill_policy=FillPolicy(),
            risk_policy=RiskPolicy(),
            rules=rules,
            initial_capital=tuple(
                (s.name, D(100000) if s.name == "agent_consensus" else D(0)) for s in specs
            ),
        )
        return PaperCyclePlan(
            inputs=inputs,
            scheduled_for=at,
            provenance=(),
            strategies=specs,
            numeric_bindings=archive.bindings,
            numeric_sources=tuple(
                s
                for s in archive.numeric
                if archive.store.get(s.evidence_id, as_of=archive.dates[-1]).record.available_at
                <= at
            ),
            macro_dataset=archive.macro_at(archive.dates[number]),
            macro_recipe=macro_recipe,
            models=recipes,
            allocation=AllocationPolicy(selector=selector, optimizer=optimizer),
            desk_evidence_ids=archive.docs,
        )

    draft = make_plan()
    datasets = service.materialize_features(draft, training=True)
    recipes = tuple(
        r.model_copy(
            update={
                "data_snapshot_hash": datasets[r.strategy].snapshot_hash,
                "training_frame_hash": training_frame_hash(datasets[r.strategy], r.fitted_at),
            }
        )
        for r in recipes
    )
    cfg.operations = cfg.operations.model_copy(
        update={
            "approved_baseline_hashes": [s.content_hash() for s in specs],
            "approved_model_hashes": [r.content_hash() for r in (*recipes, macro_recipe)],
        }
    )

    def plan(**kwargs):
        return prove(make_plan(**kwargs), archive.store, registry)

    yield SimpleNamespace(
        service=service,
        plan=plan,
        config=cfg,
        ops=ops,
        ledger=ledger,
        calls=calls,
        archive=archive,
        path=tmp_path,
    )
    ledger.close()


def test_actual_six_sleeve_cycle_and_completed_replay(runtime):
    plan = runtime.plan()
    result = runtime.service.run(plan, now=plan.inputs.context.as_of)
    assert result.status == "completed", result.reason
    assert len(runtime.calls) == 3 * 17 + 11
    state = runtime.ops.status(result.run_id)
    strategies = runtime.ops.artifact(state["stages"]["strategies"]["output_hash"])
    assert set(strategies) == set(baseline_specs())
    readiness = stage_value(runtime, "model_availability")
    desk = stage_value(runtime, "agent_desk")
    assert len(readiness["models"]) == len(plan.models) + 1
    for receipt in readiness["models"].values():
        creation = receipt["creation"]
        assert creation["run_id"] == plan.inputs.context.run_id
        assert creation["stage_started_at"] <= creation["artifact_completed_at"]
        assert creation["available_at"] <= receipt["creation_stage_completed_at"]
        assert receipt["creation_stage_completed_at"] <= receipt["available_at"]
        assert receipt["creation_receipt_hash"]
        assert all(
            pd.Timestamp(v["decision"]["context"]["as_of"]) >= pd.Timestamp(receipt["available_at"])
            for v in desk["decisions"].values()
        )
        assert result.decision_context.as_of >= pd.Timestamp(receipt["available_at"])
    assert runtime.ledger.fills(plan.inputs.account.account_id) == []
    assert runtime.service.run(plan, now=plan.inputs.context.as_of) == result
    assert len(runtime.calls) == 3 * 17 + 11


def stage_value(runtime, name, run_id="cycle-1"):
    state = runtime.ops.status(run_id)
    return runtime.ops.artifact(state["stages"][name]["output_hash"])


@pytest.fixture(scope="module")
def completed(runtime):
    plan = runtime.plan()
    result = runtime.service.run(plan, now=plan.inputs.context.as_of)
    assert result.status == "completed", result.reason
    return plan, result


@pytest.mark.parametrize(
    "method", ["risk_parity", "mean_variance", "black_litterman", "hrp", "max_diversification"]
)
def test_all_five_optimizers_consume_actual_sleeve_outputs(runtime, completed, tmp_path, method):
    import copy

    from src.domain.portfolio import PortfolioSnapshot, SleeveSnapshot

    plan, _ = completed
    context = RunContext.model_validate(stage_value(runtime, "decision_clock"))
    snapshot = PortfolioSnapshot.model_validate(stage_value(runtime, "sizing_snapshot"))
    allocation = stage_value(runtime, "allocation")
    snapshot = snapshot.model_copy(
        update={
            "sleeves": {
                n: SleeveSnapshot.model_validate(v) for n, v in allocation["sleeves"].items()
            }
        }
    )
    # Independent canonical empty-order ledger for this pure allocation check.
    ledger = PortfolioLedger(tmp_path / "optimizer.sqlite")
    ledger.create_account(plan.inputs.account)
    service = copy.copy(runtime.service)
    service.ledger = ledger
    try:
        changed = plan.model_copy(
            update={"allocation": plan.allocation.model_copy(update={"optimizer": method})}
        )
        result = service._optimize(
            changed,
            context,
            snapshot,
            stage_value(runtime, "strategies"),
            allocation,
            service._returns(plan),
        )
        assert result["optimizer"]["method"] == method
        assert all(np.isfinite(v) and v >= 0 for v in result["optimizer"]["weights"].values())
        assert result["optimizer"]["diagnostics"]["covariance_estimator"] == "LedoitWolf"
        assert set(result["outputs"]) == set(baseline_specs())
    finally:
        ledger.close()


def test_next_observed_open_fills_only_preaccepted_orders_and_reuses_models(
    runtime, completed, monkeypatch
):
    plan, first = completed
    assert first.prepared_intent_ids
    before = {p.name: p.read_bytes() for p in (runtime.path / "models").glob("*.joblib")}
    runtime.service.dispatch_pending(plan, now=first.decision_context.as_of + timedelta(seconds=1))
    next_plan = prove(
        runtime.plan(number=358, run_id="cycle-2"),
        runtime.archive.store,
        runtime.service.registry,
        observed=True,
    )
    from src.execution.venue import PaperVenue

    original = PaperVenue.settle_open
    interrupted = []

    def after_committed_fill(venue, bar, metadata, **kwargs):
        result = original(venue, bar, metadata, **kwargs)
        if result and not interrupted:
            interrupted.append(bar.event_id)
            raise KeyboardInterrupt("termination after committed opening fill before stage receipt")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PaperVenue, "settle_open", after_committed_fill)
        with pytest.raises(KeyboardInterrupt):
            runtime.service.run(next_plan, now=next_plan.inputs.context.as_of)
    assert interrupted
    assert runtime.ledger.fills(plan.inputs.account.account_id)
    second = runtime.service.run(next_plan, now=next_plan.inputs.context.as_of)
    assert second.status == "completed", second.reason
    fills = runtime.ledger.fills(plan.inputs.account.account_id)
    assert fills
    assert all(f.executed_at > first.decision_context.as_of for f in fills)
    assert all(f.intent_id in first.prepared_intent_ids for f in fills)
    assert before == {p.name: p.read_bytes() for p in (runtime.path / "models").glob("*.joblib")}
    first_ready = stage_value(runtime, "model_availability")["models"]
    second_ready = stage_value(runtime, "model_availability", "cycle-2")["models"]
    for key, receipt in second_ready.items():
        assert receipt["creation"] == first_ready[key]["creation"]
        assert receipt["creation_receipt_hash"] == first_ready[key]["creation_receipt_hash"]
        assert receipt["training_cutoff"] == first_ready[key]["training_cutoff"]
        assert receipt["consumption_stage"]["run_id"] == "cycle-2"
    # Completed replay cannot consume the same observed print twice.
    assert runtime.service.run(next_plan, now=next_plan.inputs.context.as_of) == second
    assert runtime.ledger.fills(plan.inputs.account.account_id) == fills
    assert runtime.service.run(plan, now=next_plan.inputs.context.as_of) == first


def test_unapproved_recipe_tampered_training_and_missing_sources_fail_before_provider(
    runtime, completed
):
    plan, _ = completed
    calls = len(runtime.calls)
    changed = plan.models[0].model_copy(update={"parameters": (("n_estimators", 9),)})
    with pytest.raises(ValueError, match="approval"):
        runtime.service._approved(changed, plan.inputs.context, "model:" + changed.strategy)
    with pytest.raises(ValueError, match="source bindings"):
        runtime.service._source_values((), plan.inputs.context.as_of)
    with pytest.raises(ValueError, match="after the operating cutoff"):
        runtime.service._source_values(
            (
                plan.rule_sources[0].model_copy(
                    update={"available_at": plan.inputs.context.as_of + timedelta(seconds=1)}
                ),
            ),
            plan.inputs.context.as_of,
        )
    assert len(runtime.calls) == calls


def test_trusted_model_tamper_rejects_before_loading(runtime, completed):
    plan, _ = completed
    path, receipt = runtime.service._trusted_read(plan.models[0], "numeric")
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"corrupt")
        with pytest.raises(ValueError, match="before deserialization"):
            runtime.service._trusted_read(plan.models[0], "numeric")
    finally:
        path.write_bytes(original)
    assert runtime.service._trusted_read(plan.models[0], "numeric")[1] == receipt


def test_account_lock_and_ambiguous_provider_stage_are_not_repeated(runtime, completed, tmp_path):
    from src.operations.store import OperationBusy, RecoveryRequired, RunManifest

    plan, _ = completed
    with runtime.ops.exclusive("account:" + plan.inputs.account.account_id):
        with pytest.raises(OperationBusy):
            runtime.service.run(plan, now=plan.inputs.context.as_of)
    ops = OperationsStore(tmp_path / "ambiguous.sqlite")
    ctx = plan.inputs.context.model_copy(update={"run_id": "ambiguous-provider"})
    value = {"source_hash": plan.inputs.content_hash()}
    manifest = RunManifest(
        context=ctx,
        kind="paper_cycle",
        input_hash=digest(canonical_bytes(value)),
        scheduled_for=ctx.as_of,
        stages=("agent_desk",),
    )
    ops.admit(manifest, value)
    calls = []

    def interrupted_provider():
        calls.append(1)
        raise KeyboardInterrupt("simulated process termination after provider admission")

    with pytest.raises(KeyboardInterrupt):
        ops.stage(ctx.run_id, "agent_desk", value, interrupted_provider, recoverable=False)
    with pytest.raises(RecoveryRequired):
        ops.stage(ctx.run_id, "agent_desk", value, interrupted_provider, recoverable=False)
    assert len(calls) == 1


def test_receipt_clocks_and_exact_event_coverage_are_validated(runtime, completed):
    from src.operations.contracts import ObservedOpenReceipt

    plan, _ = completed
    with pytest.raises(ValueError, match="precedes actual receipt"):
        ObservedOpenReceipt(
            event_id="x",
            evidence_id="x",
            evidence_hash="0" * 64,
            observed_at=plan.inputs.context.as_of,
            available_at=plan.inputs.context.as_of - timedelta(seconds=1),
        )
    receipt = ObservedOpenReceipt(
        event_id="missing",
        evidence_id="x",
        evidence_hash="0" * 64,
        observed_at=plan.inputs.context.as_of,
        available_at=plan.inputs.context.as_of,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        PaperCycleInputs.model_validate({**plan.inputs.model_dump(), "receipts": (receipt,)})


def test_prepare_uses_completed_refresh_and_exact_late_receipt_sources(runtime, monkeypatch):
    from datetime import datetime

    from src.data.evidence.service import EvidenceService
    from src.operations.refresh import FetchTask, RefreshPlan, RefreshService
    from src.research.evidence import verify_provenance

    from src.operations import refresh
    from src.operations.contracts import PaperCycleTemplate

    plan = runtime.plan()
    service, store = runtime.service, runtime.archive.store
    at = plan.inputs.context.as_of + timedelta(seconds=10)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at

    monkeypatch.setattr(refresh, "datetime", Clock)
    requested = plan.inputs.context.model_copy(update={"run_id": "refresh-builder", "as_of": at})

    def fetched(task, context, evidence):
        payload = {"receipt": "synthetic boundary response"}
        raw = canonical_bytes(payload)
        record = EvidenceRecord(
            evidence_id="refresh-builder-received",
            provider="fmp",
            provider_version="fixture",
            source_uri="test://current-refresh",
            raw_hash=digest(raw),
            ingested_at=at,
            available_at=at,
            requested_as_of=context.as_of,
            quality="usable",
            availability_basis="documented",
        )
        evidence.put(record, raw, kind="fixture", logical_key="refresh-builder", normalized=payload)
        return {"status": "available", "evidence_ids": [record.evidence_id]}

    refresh_plan = RefreshPlan(
        context=requested,
        scheduled_for=at,
        tasks=(FetchTask(task_id="current-source", provider="fmp", operation="fixture"),),
    )
    refreshed = RefreshService(runtime.ops, store, fetcher=fetched).run(refresh_plan)
    assert refreshed["status"] == "completed"
    context = requested.model_copy(update={"run_id": "prepared-fresh", "data_snapshot_ids": ()})
    template = PaperCycleTemplate(
        account=plan.inputs.account,
        instruments=runtime.archive.instruments,
        market_start=runtime.archive.dates[0],
        code_version="test",
        universe_version=plan.inputs.universe_version,
        fill_policy=plan.inputs.fill_policy,
        risk_policy=plan.inputs.risk_policy,
        initial_capital=plan.inputs.initial_capital,
        strategies=plan.strategies,
        models=plan.models,
        macro_recipe=plan.macro_recipe,
        numeric_bindings=plan.numeric_bindings,
        allocation=plan.allocation,
    )
    market = EvidenceService(store).market_snapshot(
        template.instruments,
        context,
        start=template.market_start,
        price_basis="raw",
    )
    universes = tuple(
        u.model_copy(update={"data_snapshot_id": market.content_hash()})
        for u in plan.inputs.universes
    )
    context = context.model_copy(update={"data_snapshot_ids": (market.content_hash(),)})
    inputs = plan.inputs.model_copy(
        update={"context": context, "snapshot": market, "universes": universes}
    )
    sources = []
    economic_values = expected_artifacts(inputs)
    put(
        store,
        "builder-opening-source",
        {k: economic_values[k] for k in ("opening_events", "opening_liquidity")},
        at,
    )
    for kind, rows in {
        **expected_artifacts(inputs),
        "venue_rules": [r.model_dump(mode="json") for r in inputs.rules],
    }.items():
        available = (
            min(u.available_at for u in universes)
            if kind in {"universe_history", "calendar", "instrument_metadata"}
            else plan.inputs.context.as_of
            if kind == "venue_rules"
            else at
        )
        key = "builder-source-" + kind
        if kind in {"opening_events", "opening_liquidity"}:
            sources.append(
                (
                    kind,
                    (
                        SourceBinding(
                            evidence_id="builder-opening-source",
                            available_at=at,
                            pointer="/" + kind,
                        ),
                    ),
                )
            )
        else:
            put(store, key, rows, available)
            sources.append((kind, (SourceBinding(evidence_id=key, available_at=available),)))
    prepared = service.prepare(
        template,
        refresh_run_id=requested.run_id,
        run_id=context.run_id,
        scheduled_for=at,
        economic_sources=tuple(sources),
        macro_dataset=plan.macro_dataset,
    )
    assert prepared.inputs.context.as_of == at
    assert prepared.inputs.context.run_id == context.run_id
    assert prepared.models == plan.models  # Training identity is never restamped.
    assert prepared.inputs.snapshot.content_hash() == market.content_hash()
    assert prepared.inputs.snapshot.content_hash() != plan.inputs.snapshot.content_hash()
    assert all(r.observed_at == at == r.available_at for r in prepared.inputs.receipts)
    assert service._verify_inputs(prepared.inputs, dict(prepared.provenance), prepared.rule_sources)
    with pytest.raises(ValueError, match="opening|available|liquidity"):
        verify_provenance(service.registry, store, prepared.inputs, dict(prepared.provenance))
    damaged = prepared.inputs.receipts[0].model_copy(update={"evidence_hash": "0" * 64})
    changed = prepared.inputs.model_copy(
        update={"receipts": (damaged, *prepared.inputs.receipts[1:])}
    )
    with pytest.raises(ValueError, match="receipt|identity|input"):
        service._verify_inputs(changed, dict(prepared.provenance), prepared.rule_sources)
    with pytest.raises(ValueError, match="eight"):
        service.prepare(
            template,
            refresh_run_id=requested.run_id,
            run_id="missing-array",
            scheduled_for=at,
            economic_sources=tuple(sources[:-1]),
            macro_dataset=plan.macro_dataset,
        )


@pytest.fixture(scope="module")
def funded_performance(runtime, completed, tmp_path_factory):
    import copy

    from src.domain.portfolio import Fill, OrderIntent, PriceMark
    from src.execution.sleeves import SleeveBook

    from src.core.macro_regime import CausalRegimeResult

    plan, _ = completed
    path = tmp_path_factory.mktemp("real-funded-performance")
    ledger = PortfolioLedger(path / "ledger.sqlite")
    service = copy.copy(runtime.service)
    service.ledger = ledger
    service.attribution = AttributedPerformanceStore(path / "performance.sqlite")
    regime = CausalRegimeResult.model_validate(stage_value(runtime, "macro"))
    chosen = next(s.name for s in plan.strategies if regime.regime in s.regimes)
    dates = runtime.archive.dates[331:358]
    account = plan.inputs.account.model_copy(
        update={
            "account_id": "paper:selector-attribution",
            "opened_at": dates[0] - timedelta(hours=1),
        }
    )
    ledger.create_account(account)
    book = SleeveBook(ledger, account.account_id).initialize(
        {s.name: account.initial_cash if s.name == chosen else D(0) for s in plan.strategies}
    )
    instrument = runtime.archive.instruments[0]
    intent = OrderIntent(
        intent_id="actual-selector-holding",
        account_id=account.account_id,
        run_id="funded",
        instrument=instrument,
        side="buy",
        quantity=100,
        reference_price=100,
        decision_at=account.opened_at,
        eligible_at=account.opened_at + timedelta(seconds=1),
        strategy_id=chosen,
    )
    book.register_intent(intent, {chosen: D(100)}, source_hash="f" * 64, quantity_quantum=D(1))
    ledger.submit(intent)
    ledger.apply_fill(
        Fill(
            fill_id="selector-fill",
            intent_id=intent.intent_id,
            account_id=account.account_id,
            quantity=100,
            price=100,
            fee=0,
            executed_at=intent.eligible_at,
            event_id="selector-observed-print",
        )
    )
    ledger.mark(account.account_id, [PriceMark(instrument=instrument, price=100, as_of=dates[0])])
    anchor = book.snapshot(dates[0])
    service._funded_anchor(chosen, anchor[chosen], dates[0])
    price = D(100)
    for i, at in enumerate(dates[1:]):
        price = (price * (D("1.003") if i % 2 else D("1.001"))).quantize(D(".01"))
        ledger.mark(account.account_id, [PriceMark(instrument=instrument, price=price, as_of=at)])
        payload = service._record_period(book, anchor, at, at + timedelta(minutes=15))
        from src.domain.portfolio import SleeveSnapshot

        anchor = {n: SleeveSnapshot.model_validate(v) for n, v in payload.items()}
    context = plan.inputs.context.model_copy(
        update={"account_id": account.account_id, "run_id": "selectors"}
    )
    regime = regime.model_copy(update={"context": context})
    snapshot = ledger.snapshot(
        account.account_id, context.as_of, 86400 * 4, persist=False
    ).model_copy(update={"sleeves": book.snapshot(context.as_of)})
    yield service, book, context, regime, snapshot, chosen
    ledger.close()


@pytest.mark.parametrize("mode", ["regime_matched", "performance_weighted", "bandit", "manual"])
def test_all_selectors_use_actual_completed_daily_ledger_returns(
    runtime, completed, funded_performance, mode
):
    plan, _ = completed
    service, book, context, regime, snapshot, chosen = funded_performance
    metrics = service.attribution.metrics(as_of=context.as_of, annualization=252)
    assert metrics.loc[chosen, "observations"] == 26
    assert metrics.loc[chosen, "sharpe"] > 0
    assert metrics.loc[chosen, "source_event_hashes"]
    changed = plan.model_copy(
        update={
            "allocation": plan.allocation.model_copy(
                update={
                    "selector": mode,
                    "manual_strategy": chosen,
                    "exploration": 0,
                }
            )
        }
    )
    result = service._allocation(changed, book, context, regime, snapshot)
    assert result["desired"] == {chosen: 1.0}
    assert result["transfers"] == []
    assert D(result["sleeves"][chosen]["portfolio"]["positions"][0]["quantity"]) == 100


def test_unobserved_performance_does_not_enable_weighting_or_bandit_exploitation(
    runtime, completed, tmp_path
):
    import copy

    from src.domain.portfolio import PortfolioSnapshot
    from src.execution.sleeves import SleeveBook

    from src.core.macro_regime import CausalRegimeResult

    plan, _ = completed
    service = copy.copy(runtime.service)
    service.attribution = AttributedPerformanceStore(tmp_path / "empty-performance.sqlite")
    context = RunContext.model_validate(stage_value(runtime, "decision_clock"))
    regime = CausalRegimeResult.model_validate(stage_value(runtime, "macro")).model_copy(
        update={"context": context}
    )
    snapshot = PortfolioSnapshot.model_validate(stage_value(runtime, "sizing_snapshot"))
    book = SleeveBook(runtime.ledger, plan.inputs.account.account_id)
    for mode in ("performance_weighted", "bandit"):
        changed = plan.model_copy(
            update={
                "allocation": plan.allocation.model_copy(
                    update={"selector": mode, "exploration": 0}
                )
            }
        )
        with pytest.raises(ValueError, match="unavailable|observations|observed"):
            service._allocation(changed, book, context, regime, snapshot)


@pytest.fixture
def mixed_covariance(tmp_path):
    from src.operations.contracts import AllocationPolicy

    store = EvidenceStore(tmp_path / "mixed-clocks.sqlite")
    equity = Instrument(
        instrument_id="mixed:equity", symbol="EQ", asset_class="equity", quote_currency="USD"
    )
    crypto = Instrument(
        instrument_id="mixed:crypto", symbol="COIN", asset_class="crypto", quote_currency="USD"
    )
    cutoff = pd.Timestamp("2026-03-20T21:15:00Z").to_pydatetime()
    bars, evidence = [], []
    for instrument, calendar in ((equity, "XNYS"), (crypto, "24/7")):
        schedule = sessions("2026-02-01", "2026-03-20", calendar=calendar)
        for i, (session, (opened, closed)) in enumerate(schedule.items()):
            if closed > cutoff:
                continue
            price = D(100) + D(i) / 10
            identity = f"covariance:{instrument.instrument_id}:{session}"
            bar = MarketBar(
                instrument=instrument,
                opened_at=opened,
                closed_at=closed,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10000,
                evidence_id=identity,
                price_basis="raw",
                market_type="spot" if calendar == "24/7" else "equity",
                calendar=calendar,
                session=session,
            )
            # A current capture can estimate risk now. This is not a claim that
            # the source was available at a historical decision anchor.
            envelope = put(
                store,
                identity,
                bar.model_dump(mode="json", exclude={"evidence_id"}),
                cutoff,
                kind="bar",
                instrument_ids=(instrument.instrument_id,),
            )
            bars.append(bar)
            evidence.append(envelope.record)
    context = RunContext(
        run_id="mixed-covariance",
        mode="paper",
        account_id="paper:mixed",
        as_of=cutoff,
        config_hash="a" * 64,
        code_version="fixture",
    )
    market = MarketSnapshot(context=context, bars=tuple(bars), evidence=tuple(evidence))
    plan = SimpleNamespace(
        inputs=SimpleNamespace(snapshot=market, context=context),
        allocation=AllocationPolicy(selector="regime_matched", optimizer="hrp"),
    )
    service = object.__new__(PaperCycleService)
    service.evidence = store
    return service, plan


def test_covariance_pairs_actual_cross_calendar_intervals_without_weekend_copies(mixed_covariance):
    service, plan = mixed_covariance
    values = service._returns(plan)
    receipt = values.attrs["sampling_receipt"]
    assert len(values) >= 20
    assert receipt["policy"] == "retrospective-lagged-intervals-1"
    assert receipt["estimation_cutoff"] == plan.inputs.context.as_of.isoformat()
    assert values.attrs["sampling_hash"] == digest(canonical_bytes(receipt))
    observed_hours = set()
    seen = {key: set() for key in values.columns}
    monday = False
    for sample in receipt["samples"]:
        anchor = pd.Timestamp(sample["anchor"])
        assert anchor.weekday() < 5
        observed_hours.add(anchor.hour)
        for key, interval in sample["intervals"].items():
            start, end = pd.Timestamp(interval["started_at"]), pd.Timestamp(interval["ended_at"])
            assert start < end <= anchor
            assert (anchor - start).total_seconds() <= receipt["max_endpoint_age_seconds"]
            pair = interval["start_evidence_id"], interval["end_evidence_id"]
            assert pair not in seen[key]
            seen[key].add(pair)
            assert pd.Timestamp(interval["end_available_at"]) == plan.inputs.context.as_of
            if key == "mixed:crypto":
                assert end.hour == 0
                if anchor.weekday() == 0:
                    assert start.weekday() == 6
                    monday = True
            else:
                assert end == anchor
    assert observed_hours == {20, 21}  # DST changes actual equity close only.
    assert monday
    assert len(values) < sum(
        b.instrument.asset_class == "crypto" for b in plan.inputs.snapshot.bars
    )


def test_covariance_rejects_insufficient_or_stale_pair_coverage(mixed_covariance):
    service, plan = mixed_covariance
    plan.allocation = plan.allocation.model_copy(
        update={"covariance_max_endpoint_age_seconds": 3600}
    )
    with pytest.raises(ValueError, match="insufficient distinct actual paired"):
        service._returns(plan)


def test_duplicate_schedule_cannot_repeat_economic_cycle(runtime, completed):
    plan, _ = completed
    changed = plan.model_copy(
        update={
            "inputs": plan.inputs.model_copy(
                update={
                    "context": plan.inputs.context.model_copy(
                        update={"run_id": "same-schedule-new-name"}
                    )
                }
            )
        }
    )
    before = len(runtime.calls)
    with pytest.raises(ValueError, match="schedule already"):
        runtime.service.run(changed, now=plan.inputs.context.as_of)
    assert len(runtime.calls) == before


def test_exact_long_decimal_attribution_does_not_assign_negative_nav_to_empty_sleeve(tmp_path):
    from src.domain.portfolio import Fill, OrderIntent, PriceMark
    from src.execution.sleeves import SleeveBook, exact_sum

    at = pd.Timestamp("2026-10-08T20:00:00Z").to_pydatetime()
    instrument = Instrument(
        instrument_id="precise:equity", symbol="PREC", asset_class="equity", quote_currency="USD"
    )
    ledger = PortfolioLedger(tmp_path / "precise.sqlite")
    account = AccountSpec(
        account_id="paper:precise",
        mode="paper",
        base_currency="USD",
        initial_cash=100000,
        opened_at=at - timedelta(hours=1),
    )
    ledger.create_account(account)
    names = tuple(baseline_specs())
    funded = names[0]
    book = SleeveBook(ledger, account.account_id).initialize(
        {name: D(100000) if name == funded else D(0) for name in names}
    )
    order = OrderIntent(
        intent_id="precise-order",
        account_id=account.account_id,
        run_id="precise",
        instrument=instrument,
        side="buy",
        quantity=100,
        reference_price=100,
        decision_at=account.opened_at,
        eligible_at=account.opened_at + timedelta(seconds=1),
        strategy_id=funded,
    )
    book.register_intent(order, {funded: D(100)}, source_hash="d" * 64, quantity_quantum=D(1))
    ledger.submit(order)
    ledger.apply_fill(
        Fill(
            fill_id="precise-fill",
            intent_id=order.intent_id,
            account_id=account.account_id,
            quantity=100,
            price=100,
            fee=0,
            executed_at=order.eligible_at,
            event_id="actual-print",
        )
    )
    ledger.mark(
        account.account_id,
        [PriceMark(instrument=instrument, as_of=at, price=D("101.7124509288088173404513081"))],
    )
    try:
        projection = book.snapshot(at)
        actual = ledger.snapshot(account.account_id, at, 86400)
        assert exact_sum(s.portfolio.nav for s in projection.values()) == actual.nav
        assert all(s.portfolio.nav == 0 for name, s in projection.items() if name != funded)
        assert projection[funded].portfolio.positions[0].quantity == 100
    finally:
        ledger.close()


def test_open_only_close_mark_expires_remaining_order_without_completed_execution_bar(tmp_path):
    from src.domain.portfolio import Fill, OrderIntent, PriceMark
    from src.execution.sleeves import SleeveBook
    from src.execution.venue import PaperVenue

    at = pd.Timestamp("2026-10-08T20:00:00Z").to_pydatetime()
    instrument = Instrument(
        instrument_id="close-only", symbol="CLS", asset_class="equity", quote_currency="USD"
    )
    account = AccountSpec(
        account_id="paper:close-only",
        mode="paper",
        base_currency="USD",
        initial_cash=10000,
        opened_at=at - timedelta(hours=7),
    )
    ledger = PortfolioLedger(tmp_path / "close-only.sqlite")
    ledger.create_account(account)
    book = SleeveBook(ledger, account.account_id).initialize({"owned": D(10000)})
    order = OrderIntent(
        intent_id="partially-filled",
        account_id=account.account_id,
        run_id="close-only",
        instrument=instrument,
        side="buy",
        quantity=2,
        reference_price=100,
        decision_at=account.opened_at,
        eligible_at=account.opened_at + timedelta(seconds=1),
        strategy_id="owned",
        tif="DAY",
        expires_at=at,
    )
    book.register_intent(order, {"owned": D(2)}, source_hash="a" * 64, quantity_quantum=D(1))
    ledger.submit(order)
    ledger.apply_fill(
        Fill(
            fill_id="partial",
            intent_id=order.intent_id,
            account_id=account.account_id,
            quantity=1,
            price=100,
            fee=0,
            executed_at=order.eligible_at,
            event_id="actual-print",
        )
    )
    service = object.__new__(PaperCycleService)
    service.ledger = ledger
    service.attribution = AttributedPerformanceStore(tmp_path / "close-attribution.sqlite")
    ledger.mark(
        account.account_id, [PriceMark(instrument=instrument, price=100, as_of=order.eligible_at)]
    )
    opening = book.snapshot(order.eligible_at)
    service._funded_anchor("owned", opening["owned"], order.eligible_at)
    close = MarketBar(
        instrument=instrument,
        opened_at=at - timedelta(hours=6, minutes=30),
        closed_at=at,
        open=100,
        high=100,
        low=90,
        close=90,
        volume=1000,
        evidence_id="actual-close",
        price_basis="raw",
        market_type="equity",
        calendar="XNYS",
        session="2026-10-08",
    )
    plan = SimpleNamespace(
        inputs=SimpleNamespace(
            account=account, context=SimpleNamespace(as_of=at + timedelta(minutes=15))
        )
    )
    venue = PaperVenue(ledger, account.account_id, {})
    try:
        result = service._observe(plan, book, venue, None, opening, [("market_close", close)], at)
        assert ledger.order(account.account_id, order.intent_id)["status"] == "cancelled"
        assert ledger.quantity(account.account_id, instrument.instrument_id) == 1
        assert D(result["anchor"]["owned"]["portfolio"]["nav"]) == 9990
        assert len(ledger.fills(account.account_id)) == 1
        assert not ledger.db.execute(
            "SELECT 1 FROM events WHERE id LIKE 'close:%' OR id LIKE 'open:%'"
        ).fetchone()
        assert ledger.reconcile(account.account_id)["reconciled"]
    finally:
        ledger.close()


@pytest.mark.parametrize("action", ["HOLD", "ABSTAIN", "VETO", "REDUCE", "EXIT", None])
def test_desk_boundary_preserves_passive_quantity_and_explicit_exit_or_reduce(action):
    from src.domain.decisions import AgentDecision

    instrument = Instrument(
        instrument_id="boundary", symbol="BOUND", asset_class="equity", quote_currency="USD"
    )
    context = RunContext(
        run_id="boundary",
        mode="paper",
        as_of=pd.Timestamp("2026-10-08T20:15:00Z").to_pydatetime(),
        config_hash="a" * 64,
        code_version="test",
        account_id="paper:boundary",
    )
    decision = (
        AgentDecision(
            instrument=instrument,
            context=context,
            action=action,
            reasoning="synthetic boundary fixture",
            new_exposure_blocked=action in {"ABSTAIN", "VETO"},
        )
        if action
        else None
    )
    for current in (0.0, 0.1):
        for proposed in (0.0, 0.05, 0.2):
            frame = pd.DataFrame(
                {
                    "action": ["BUY"],
                    "target_weight": [proposed],
                    "reason": ["quantitative selection"],
                },
                index=[instrument.instrument_id],
            )
            PaperCycleService._desk_boundary(frame, instrument.instrument_id, decision, current)
            row = frame.iloc[0]
            assert row.action == (action or "ABSTAIN")
            if action == "EXIT":
                assert row.target_weight == 0
            elif action == "REDUCE":
                assert row.target_weight == min(current, proposed)
            else:
                assert row.target_weight == current
