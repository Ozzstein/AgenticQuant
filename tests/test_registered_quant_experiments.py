"""Actual HMM/model/replay integration; synthetic archives exist only in tests."""
from datetime import timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from src.domain.portfolio import AccountSpec, ExecutionBar, InstrumentRisk, UniverseEligibility
from src.execution.ledger import PortfolioLedger
from src.research.datasets import ResearchInputs
from src.research.evidence import BoundArtifact, SourceBinding, expected_artifacts
from src.research.registry import Candidate, PromotionPolicy, RegistryError, ResearchRegistry, Trial

from src.core.backtester import (
    PortfolioReplay,
    reconstruct_decision_inputs,
    recorded_decisions,
    split_folds,
)
from src.data.calendars import sessions
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import EvidenceRecord
from src.domain.market import FeatureSpec, MarketSnapshot
from src.research.regime_experiments import (
    FrozenEvidence,
    HMMVariantSpec,
    MacroDatasetSpec,
    MacroRawObservation,
    build_regime_artifact,
    materialize_macro_dataset,
    recompute_regime_artifact,
)
from src.research.strategy_experiments import (
    StrategyExperimentPlan,
    execute_registered_strategy,
    materialize_strategy_plan,
)
from src.strategies.runner import StrategyInputError, require_strategy_eligible
from src.strategies.specification import baseline_specs
from tests.data_quant_fixtures import archived_snapshot


def archive(store, key, payload, at, *, observed=None, kind='fixture'):
    raw = canonical_bytes(payload)
    record = EvidenceRecord(evidence_id=key,provider='synthetic-test-only',provider_version='1',source_uri='test://'+key,
        raw_hash=digest(raw),ingested_at=at+timedelta(days=1000),available_at=at,observed_at=observed or at,
        requested_as_of=at,availability_basis='documented',quality='usable')
    return store.put(record,raw,kind=kind,logical_key=key,normalized=payload)


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    path = tmp_path_factory.mktemp('quant-registered')
    store,registry = EvidenceStore(path/'sources.sqlite'),ResearchRegistry(path/'registry.sqlite')
    original = archived_snapshot(periods=350)
    actual_sessions = list(sessions('2020-01-02','2022-01-01',calendar='XNYS').items())[:350]
    old_dates = sorted({b.closed_at for b in original.bars})
    mapped = dict(zip(old_dates,actual_sessions,strict=True))
    context = original.context.model_copy(update={'as_of':actual_sessions[-1][1][1]})
    snapshot = MarketSnapshot(context=context,
        bars=tuple(b.model_copy(update={'volume':b.volume*100,'session':mapped[b.closed_at][0],
            'opened_at':mapped[b.closed_at][1][0],'closed_at':mapped[b.closed_at][1][1]}) for b in original.bars),
        evidence=tuple(r.model_copy(update={'available_at':mapped[r.available_at][1][1],
            'observed_at':mapped[r.observed_at][1][1],'requested_as_of':actual_sessions[0][1][0]}) for r in original.evidence))
    records = {e.evidence_id:e for e in snapshot.evidence}
    frozen = []
    for bar in snapshot.bars:
        envelope = store.put(records[bar.evidence_id],bar.evidence_id.encode(),kind='bar',logical_key=bar.evidence_id,
                             normalized=bar.model_dump(mode='json'))
        frozen.append(FrozenEvidence(evidence_id=bar.evidence_id,envelope_hash=envelope.content_hash()))
    dates = pd.DatetimeIndex(sorted({b.closed_at for b in snapshot.bars}))
    selected = snapshot.bars[-80*3:]
    selected_ids = {b.evidence_id for b in selected}
    market = MarketSnapshot(context=snapshot.context,bars=selected,evidence=tuple(r for r in snapshot.evidence if r.evidence_id in selected_ids))
    account = AccountSpec(account_id='backtest:quant-trial',mode='backtest',base_currency='USD',initial_cash=100000,
                          opened_at=selected[0].opened_at-timedelta(days=1))
    context = market.context.model_copy(update={'mode':'backtest','account_id':account.account_id,
                                               'data_snapshot_ids':(market.content_hash(),)})
    universe = UniverseEligibility(universe_version='test',data_snapshot_id=market.content_hash(),effective_at=account.opened_at,
        available_at=account.opened_at,instrument_ids=tuple(sorted({b.instrument.instrument_id for b in selected})))
    bars = tuple(ExecutionBar(event_id=b.evidence_id,instrument=b.instrument,
        **{k:getattr(b,k) for k in ('opened_at','closed_at','open','high','low','close','volume','calendar','session')},
        opening_volume=10000,calendar_version='synthetic-test-only',session_open=b.opened_at,session_close=b.closed_at) for b in selected)
    metadata = tuple(InstrumentRisk(instrument=b.instrument,sector=b.instrument.instrument_id,adv_base=100000000,
        as_of=b.opened_at,lot_size='.001') for b in selected)
    inputs = ResearchInputs(snapshot=market,context=context,account=account,bars=bars,metadata=metadata,universes=(universe,),
        universe_version='test',valuation_calendar='XNYS',walk_months=1,initial_train_months=1,validation_months=1,
        embargo_days=0,synthetic=True)
    history = universe.model_copy(update={'effective_at':dates[0]-timedelta(days=1),'available_at':dates[0]-timedelta(days=1)})
    envelope = archive(store,'macro-membership',[history.model_dump(mode='json')],history.available_at)
    frozen.append(FrozenEvidence(evidence_id='macro-membership',envelope_hash=envelope.content_hash()))
    member_source = SourceBinding(evidence_id='macro-membership',available_at=history.available_at)
    rng = np.random.default_rng(34)
    raw = pd.DataFrame({
        'vix':20*np.exp(np.cumsum(rng.normal(0,.03,len(dates)))),
        'dgs10':3+rng.normal(0,.3,len(dates)), 'dgs2':2+rng.normal(0,.3,len(dates)),
        **{key:100*np.exp(np.cumsum(rng.normal(.001,.01,len(dates)))) for key in ('sp500','dxy','hyg','tlt')},
    },index=dates)
    observations = []
    for number,(at,row) in enumerate(raw.iterrows()):
        key = f'macro-{number}'
        envelope = archive(store,key,row.to_dict(),at)
        frozen.append(FrozenEvidence(evidence_id=key,envelope_hash=envelope.content_hash()))
        observations.append(MacroRawObservation(at=at,values=tuple((name,SourceBinding(evidence_id=key,pointer='/'+name,available_at=at)) for name in row.index)))
    macro = MacroDatasetSpec(breadth_snapshot=snapshot,universe_version='test',universes=(history,),
        universe_sources=(member_source,),raw=tuple(observations),sources=tuple(frozen))
    refs = []
    names = {'universe_history':'universe_history_id','availability_audit':'availability_audit_id','calendar':'calendar_artifact_id',
             'corporate_actions':'corporate_actions_id','execution_bars':'execution_bars_id','opening_liquidity':'opening_liquidity_id',
             'instrument_metadata':'instrument_metadata_id'}
    for kind,value in expected_artifacts(inputs).items():
        at = account.opened_at if kind in ('universe_history','calendar','opening_liquidity','instrument_metadata') else context.as_of
        key = 'proof-'+kind
        archive(store,key,value,at)
        bound = BoundArtifact(kind=kind,inputs_hash=inputs.content_hash(),value_json=canonical_bytes(value).decode(),
                              sources=(SourceBinding(evidence_id=key,available_at=at),))
        refs.append((names[kind],registry.archive_contract(bound)))
    spec = baseline_specs()['momentum_topk'].model_copy(update={'name':'trial-momentum','origin':'research_candidate',
        'model_name':'Linear','cadence':'daily','regimes':('risk_on','neutral','risk_off','crisis'),
        'features':(FeatureSpec(name='adv_20',expression='Mean(close*volume,2)'),FeatureSpec(name='composite_momentum',expression='close/Ref(close,2)-1')),
        'parameters':(('minimum_prediction',-10.),('minimum_momentum',-10.))})
    data = inputs.materialize(spec.features)
    folds = split_folds(data.features.index.get_level_values(0),data.features.index.get_level_values(0).min().isoformat(),1,0,1,1)
    first = pd.Timestamp(folds[0]['test_start'])
    times = tuple(dates[dates >= first].to_pydatetime())
    variant = HMMVariantSpec(normalization_window=20,n_init=1,n_iter=10,confidence_threshold=.25)
    artifact = build_regime_artifact(registry,store,inputs,dataset=macro,variant=variant,fitted_at=dates[301],
                                     inference_times=times,work_dir=path/'hmm-build')
    plan = StrategyExperimentPlan(inputs=inputs,strategy=spec,regime_artifact_hash=registry.archive_contract(artifact),
                                   provenance=tuple(refs),reference_features=(FeatureSpec(name='reference',expression='close/Ref(close,3)-1'),))
    candidate = Candidate(candidate_id='strategy-one',kind='strategy',origin='custom',payload_hash=spec.content_hash(),
        code_hash=registry.archive(b'test-code'),environment_hash=registry.archive(b'test-environment'),
        source_evidence_hashes=(registry.archive_contract(market),),created_at=context.as_of,synthetic=True)
    registry.quarantine(candidate,canonical_bytes(plan.candidate_payload()))
    trial = Trial(trial_id='strategy-one',candidate_id=candidate.candidate_id,family_id='strategy',
        dataset_hash=market.content_hash(),split_hash=registry.archive(canonical_bytes(inputs.split_manifest())),
        inputs_hash=registry.archive_contract(plan),policy_hash=registry.archive_contract(PromotionPolicy()),
        registered_at=context.as_of,purpose='development',evaluation_start=first,evaluation_end=pd.Timestamp(folds[-1]['test_end']))
    registry.register_trial(trial)
    return path,store,registry,plan,artifact,trial


def test_source_rebuilt_macro_and_regime_proof(experiment,tmp_path):
    _,store,registry,plan,artifact,_ = experiment
    data = materialize_macro_dataset(artifact.dataset,store)
    assert len(data.signals)==151
    actual = recompute_regime_artifact(registry,store,artifact,plan.inputs,work_dir=tmp_path/'verified')
    assert actual == artifact.observations
    assert all(o.trained_through <= artifact.fitted_at < o.context.as_of for o in actual)
    forged = artifact.model_copy(update={'model_hash':'f'*64})
    with pytest.raises(RegistryError,match='do not reproduce'):
        recompute_regime_artifact(registry,store,forged,plan.inputs,work_dir=tmp_path/'forged')
    changed = artifact.dataset.model_copy(update={'universes':(artifact.dataset.universes[0].model_copy(update={'instrument_ids':('missing',)}),)})
    with pytest.raises(RegistryError,match='membership differs'):
        materialize_macro_dataset(changed,store)


def test_registered_real_model_and_holdings_callback(experiment,tmp_path):
    _,store,registry,plan,_,trial = experiment
    executed = execute_registered_strategy(registry,trial.trial_id,store=store,ledger_path=tmp_path/'ledger.sqlite',artifact_dir=tmp_path/'models')
    result = executed.result
    assert result.trades and result.walk_forward_results
    assert result.manifest['provenance']['trial_registry']['trials']==registry.trial_count_at_admission(trial.trial_id)
    assert all(pd.Timestamp(t['executed_at']) > pd.Timestamp(t['decision_at']) for t in result.trades)
    ledger = PortfolioLedger(executed.ledger_path)
    try:
        decisions = recorded_decisions(ledger,result)
    finally:
        ledger.close()
    from src.core.validation import ValidationSuite
    report = ValidationSuite().validate(result,ledger_path=executed.ledger_path)
    durable = next(c for c in report.checks if c.name == 'durable_accounting')
    assert durable.verdict.value == 'PASS', durable.explanation
    assert decisions and any(d.snapshot.positions for d in decisions)
    assert all(p.opened_at is not None for d in decisions for p in d.snapshot.positions)
    kwargs = reconstruct_decision_inputs(result,executed.replay_kwargs,ledger_path=executed.ledger_path,decision_provider=executed.decision_provider)
    # Actual cost rerun invokes the same pure recipe against its own changed holdings/NAV.
    kwargs['fill_policy'] = plan.inputs.fill_policy.model_copy(update={'equity_cost_bps':Decimal('50')})
    changed = PortfolioReplay(tmp_path/'cost.sqlite').run(**kwargs)
    assert changed.equity_curve != result.equity_curve
    assert changed.manifest['decision_recipe_hash'] == result.manifest['decision_recipe_hash']
    import json

    from src.core.backtester import ReplayArtifact, load_replay_artifact
    from src.core.benchmark import BenchmarkRunner
    cost_results = BenchmarkRunner().cost_sensitivity(inputs=kwargs,output_dir=tmp_path/'cost-scenarios')
    assert len(cost_results) == 3 and all(r.trades for r in cost_results)
    for cost in cost_results:
        assert cost.account_id.startswith('backtest:cost-')
        assert cost.manifest['decision_recipe_hash'] == result.manifest['decision_recipe_hash']
    references = json.loads((tmp_path/'cost-scenarios/cost-artifacts.json').read_text())
    for reference in references:
        persisted = load_replay_artifact(ReplayArtifact.model_validate(reference))
        assert persisted.trades
    assert registry.active('strategy:'+plan.strategy.name) is None
    with pytest.raises(RegistryError,match='already admitted'):
        execute_registered_strategy(registry,trial.trial_id,store=store,ledger_path=tmp_path/'ledger.sqlite',artifact_dir=tmp_path/'models')


def test_trial_authorization_is_not_activation(experiment):
    _,_,registry,plan,_,trial = experiment
    context = plan.inputs.context.model_copy(update={'as_of':trial.evaluation_start})
    assert require_strategy_eligible(plan.strategy,context,research_registry=registry,trial_id=trial.trial_id).startswith('registered_trial:')
    with pytest.raises(StrategyInputError,match='isolated backtest'):
        require_strategy_eligible(plan.strategy,context.model_copy(update={'mode':'paper'}),research_registry=registry,trial_id=trial.trial_id)
    with pytest.raises(StrategyInputError,match='outside'):
        require_strategy_eligible(plan.strategy,context.model_copy(update={'as_of':trial.evaluation_end}),research_registry=registry,trial_id=trial.trial_id)
    with pytest.raises(StrategyInputError,match='requires explicit'):
        require_strategy_eligible(plan.strategy,context)


def test_frozen_numeric_envelopes_are_verified(experiment):
    _,store,_,plan,artifact,_ = experiment
    from src.data.features.contextual import NumericEvidenceBinding
    binding = NumericEvidenceBinding(name='macro',target_instrument_id='equity:0',source_instrument_id=None,
        source_market='global',kind='fixture',value_path=('vix',),filters=(),max_age_seconds=86400)
    source = artifact.dataset.sources[-1].model_copy(update={'envelope_hash':'a'*64})
    altered = plan.model_copy(update={'numeric_bindings':(binding,),'numeric_sources':(source,)})
    with pytest.raises(RegistryError,match='envelope differs'):
        materialize_strategy_plan(altered,store)


def test_paired_regime_uses_one_admission_and_actual_replays(experiment,tmp_path):
    from src.research.regime_experiments import RegimeExperimentPlan, execute_registered_regime
    _,store,registry,strategy_plan,artifact,_ = experiment
    baseline_variant = artifact.variant.model_copy(update={'min_regime_days':6})
    baseline_artifact = build_regime_artifact(registry,store,strategy_plan.inputs,dataset=artifact.dataset,
        variant=baseline_variant,fitted_at=artifact.fitted_at,inference_times=artifact.inference_times,
        work_dir=tmp_path/'baseline-proof')
    plan = RegimeExperimentPlan(**{**strategy_plan.model_dump(), 'strategy':baseline_specs()['mean_reversion'],
        'variant':artifact.variant,'baseline_variant':baseline_variant,
        'baseline_regime_artifact_hash':registry.archive_contract(baseline_artifact)})
    assert plan.candidate_payload() == artifact.variant.model_dump(mode='json')
    key = 'regime-one'
    candidate = Candidate(candidate_id=key,kind='regime',origin='custom',payload_hash=artifact.variant.content_hash(),
        code_hash=registry.archive(b'test-code'),environment_hash=registry.archive(b'test-environment'),
        source_evidence_hashes=(plan.inputs.snapshot.content_hash(),),created_at=plan.inputs.context.as_of,synthetic=True)
    registry.quarantine(candidate,canonical_bytes(plan.candidate_payload()))
    data = materialize_strategy_plan(plan,store)
    dates = data.features.index.get_level_values(0)
    folds = split_folds(dates,dates.min().isoformat(),1,0,1,1)
    trial = Trial(trial_id=key,candidate_id=key,family_id='regime',dataset_hash=plan.inputs.snapshot.content_hash(),
        split_hash=registry.archive(canonical_bytes(plan.inputs.split_manifest())),inputs_hash=registry.archive_contract(plan),
        policy_hash=registry.archive_contract(PromotionPolicy()),registered_at=plan.inputs.context.as_of,purpose='development',
        evaluation_start=pd.Timestamp(folds[0]['test_start']),evaluation_end=pd.Timestamp(folds[-1]['test_end']))
    registry.register_trial(trial)
    count = registry.trial_count_at_admission(key)
    executed = execute_registered_regime(registry,key,store=store,ledger_path=tmp_path/'candidate-ledger.sqlite',artifact_dir=tmp_path/'models')
    baseline = dict(executed.comparisons)['baseline']
    assert executed.inputs_hash == baseline.inputs_hash == trial.inputs_hash
    assert executed.result.manifest['bars_hash'] == baseline.result.manifest['bars_hash']
    assert executed.result.timestamps == baseline.result.timestamps
    assert executed.result.manifest['decision_recipe_hash'] != baseline.result.manifest['decision_recipe_hash']
    assert [f.model_version for f in executed.result.walk_forward_results] == [f.model_version for f in baseline.result.walk_forward_results]
    assert baseline.ledger_path != executed.ledger_path
    assert baseline.result.manifest['provenance']['trial_registry']['trials'] == count
    assert registry.active('regime') is None


def test_macro_omitted_session_fails_explicitly(experiment):
    _,store,_,_,artifact,_ = experiment
    missing = artifact.dataset.model_copy(update={'raw':artifact.dataset.raw[:220]+artifact.dataset.raw[221:]})
    with pytest.raises(RegistryError,match='missing or nonexchange sessions'):
        materialize_macro_dataset(missing,store)



def test_regime_later_decision_clocks_reuse_exact_causal_session(experiment,tmp_path):
    _,store,registry,plan,artifact,_ = experiment
    at = artifact.inference_times[0]
    times = (at,at+timedelta(minutes=15),at+timedelta(minutes=30))
    delayed = build_regime_artifact(registry,store,plan.inputs,dataset=artifact.dataset,variant=artifact.variant,
        fitted_at=artifact.fitted_at,inference_times=times,work_dir=tmp_path/'late')
    assert delayed.model_hash==artifact.model_hash and delayed.model_payload_hash==artifact.model_payload_hash
    assert [o.context.as_of for o in delayed.observations]==list(times)
    states = [o.model_dump(exclude={'context'}) for o in delayed.observations]
    assert states[0]==states[1]==states[2]==artifact.observations[0].model_dump(exclude={'context'})
    assert recompute_regime_artifact(registry,store,delayed,plan.inputs,work_dir=tmp_path/'verify')==delayed.observations
    strict = artifact.dataset.model_copy(update={'max_source_age_seconds':60})
    with pytest.raises(RegistryError,match='macro session is stale'):
        build_regime_artifact(registry,store,plan.inputs,dataset=strict,variant=artifact.variant,
            fitted_at=artifact.fitted_at,inference_times=(times[1],),work_dir=tmp_path/'stale')


def test_captured_macro_keeps_actual_vintages_and_historical_membership(experiment):
    _,store,_,_,artifact,_=experiment
    original=artifact.dataset
    captured_at=original.breadth_snapshot.context.as_of+timedelta(days=1)
    snapshot_context=original.breadth_snapshot.context.model_copy(update={'as_of':captured_at+timedelta(minutes=1),'mode':'paper'})
    keys={source.evidence_id:source.evidence_id+'-current-capture' for source in original.sources}
    records,envelopes={},[]
    history=original.universes[0].model_copy(update={'available_at':captured_at})
    for source in original.sources:
        old=store.get(source.evidence_id)
        raw=store.raw(source.evidence_id)
        normalized=store.normalized(source.evidence_id,as_of=original.breadth_snapshot.context.as_of)
        if source.evidence_id==original.universe_sources[0].evidence_id:
            normalized=[history.model_dump(mode='json')]
            raw=canonical_bytes(normalized)
        record=old.record.model_copy(update={'evidence_id':keys[source.evidence_id],
            'raw_hash':digest(raw),'ingested_at':captured_at,'available_at':captured_at,'requested_as_of':captured_at})
        envelope=store.put(record,raw,kind=old.kind,logical_key=keys[source.evidence_id],normalized=normalized)
        records[source.evidence_id]=record
        envelopes.append(FrozenEvidence(evidence_id=record.evidence_id,envelope_hash=envelope.content_hash()))
    snapshot=MarketSnapshot(context=snapshot_context,
        bars=tuple(b.model_copy(update={'evidence_id':keys[b.evidence_id]}) for b in original.breadth_snapshot.bars),
        evidence=tuple(records[e.evidence_id] for e in original.breadth_snapshot.evidence))
    observations=tuple(MacroRawObservation(at=o.at,availability_policy='captured-vintage-at-fit-1',
        values=tuple((name,source.model_copy(update={'evidence_id':keys[source.evidence_id],
            'available_at':captured_at})) for name,source in o.values)) for o in original.raw)
    dataset=MacroDatasetSpec(breadth_snapshot=snapshot,universe_version=original.universe_version,
        universes=(history,),universe_sources=tuple(s.model_copy(update={'evidence_id':keys[s.evidence_id],
            'available_at':captured_at}) for s in original.universe_sources),raw=observations,
        sources=tuple(envelopes),availability_policy='captured-vintage-at-fit-1')
    with pytest.raises(RegistryError,match='availability policies differ'):
        materialize_macro_dataset(dataset,store)
    current=materialize_macro_dataset(dataset,store,availability_policy='captured-vintage-at-fit-1')
    baseline=materialize_macro_dataset(original,store)
    pd.testing.assert_frame_equal(current.signals,baseline.signals)
    assert (current.available_at==captured_at).all()
    assert (current.available_at>current.signals.index).all()
    assert current.availability_policy=='captured-vintage-at-fit-1'
    assert dataset.raw[0].at==original.raw[0].at
    # A current membership observation cannot become historical breadth, even
    # under current-vintage training. Its effective date remains authoritative.
    current_history=history.model_copy(update={'effective_at':captured_at})
    payload=[current_history.model_dump(mode='json')]
    envelope=archive(store,'current-members-only',payload,captured_at)
    changed=dataset.model_copy(update={'universes':(current_history,),
        'universe_sources':(SourceBinding(evidence_id='current-members-only',available_at=captured_at),),
        'sources':(*dataset.sources,FrozenEvidence(evidence_id='current-members-only',envelope_hash=envelope.content_hash()))})
    with pytest.raises(RegistryError,match='historical breadth membership missing'):
        materialize_macro_dataset(changed,store,availability_policy='captured-vintage-at-fit-1')
