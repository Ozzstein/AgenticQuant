"""Test-only synthetic markets execute exact features and actual saved estimators."""
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.macro_regime import CausalRegimeResult
from src.core.model_zoo import ModelWrapper
from src.core.strategy_tracker import AttributedPerformanceStore
from src.data.evidence.store import digest
from src.data.features.contextual import EvidenceFields
from src.data.features.dataset import feature_dataset
from src.strategies.allocation import blend_sleeves, select_allocations
from src.strategies.runner import StrategyInputError, evaluate_strategy
from src.strategies.specification import baseline_specs
from tests.data_quant_fixtures import archived_snapshot


def market_inputs(spec):
    crypto = spec.family == "crypto_trend"
    snapshot = archived_snapshot(crypto=crypto, periods=330)
    transformed = []
    dates = sorted({bar.closed_at for bar in snapshot.bars})
    for bar in snapshot.bars:
        t = dates.index(bar.closed_at)
        j = int(bar.instrument.instrument_id.split(":")[-1])
        price = (100+j*20)*np.exp(.003*t+.002*np.sin(t/3+j))
        if j == 0 and t >= 320:
            price *= .97**(t-319)
        price = Decimal(str(price))
        transformed.append(bar.model_copy(update=dict(open=price, high=price*Decimal('1.01'), low=price*Decimal('.99'), close=price, vwap=price, volume=Decimal(1_000_000+t*1000))))
    snapshot = snapshot.model_copy(update={"bars": tuple(transformed)})
    index = pd.MultiIndex.from_tuples([(bar.closed_at, bar.instrument.instrument_id) for bar in transformed], names=["datetime", "instrument"])
    t = np.repeat(np.arange(330), 3)
    fields = pd.DataFrame(dict(pe=12+t*.001, roic=.2+t*.00001, free_cash_flow=1e8+t,
                               benchmark_close=100*np.exp(.002*t+.03*np.sin(t/3)),
                               funding_rate=.001*np.sin(t), open_interest=1e6*np.exp(.001*t+.01*np.cos(t)),
                               btc_dominance=.5+.01*np.sin(t/4)), index=index)
    evidence = EvidenceFields(fields, pd.Series(index.get_level_values('datetime'), index=index),
                              pd.Series([(bar.evidence_id,) for bar in transformed], index=index), (), digest(b"test-contextual-fields"))
    dataset = feature_dataset(snapshot, spec.features, evidence_fields=evidence)
    return snapshot, dataset


def regime_for(context, label='risk_on'):
    return CausalRegimeResult(context=context, status="available", regime=label, confidence=.85,
                             model_hash='1'*64, input_hash='2'*64, evidence_ids=('macro-test',),
                             trained_through=context.as_of-pd.Timedelta(days=90),
                             state_probabilities=tuple((name, .85 if name == label else .05) for name in ('risk_on','neutral','risk_off','crisis')))


def args_for(snapshot, dataset):
    context = snapshot.context
    identities = tuple({bar.instrument.instrument_id: bar.instrument for bar in snapshot.bars}.values())
    # Structural fixture for accounting's canonical UniverseEligibility contract.
    universe = SimpleNamespace(universe_version='historical-test', data_snapshot_id=dataset.snapshot_hash,
                               effective_at=context.as_of, available_at=context.as_of,
                               instrument_ids=tuple(item.instrument_id for item in identities))
    return dict(universe=universe, instruments=identities, holdings=pd.Series(dtype=float),
                signal_at=context.as_of, regime=regime_for(context), holding_sessions=pd.Series(dtype=float))


@pytest.mark.parametrize('name', ['momentum_topk','value_quality','mean_reversion','crypto_trend','defensive'])
def test_all_quant_strategies_use_actual_features_models_and_caps(name, tmp_path):
    spec = baseline_specs()[name]
    snapshot, dataset = market_inputs(spec)
    dates = dataset.features.index.get_level_values('datetime')
    fit_at = sorted(dates.unique())[310]
    training = dataset.eligible_training(fit_at)
    complete = training.features.notna().all(axis=1)
    model = ModelWrapper(spec.model_name)
    if spec.model_name == 'Linear':
        model._params = {'alpha': 1e12}  # stable intercept with actual positive training returns
    elif spec.model_name == 'CatBoost':
        model._params.update(iterations=5, depth=2, thread_count=1)
    else:
        model._params.update(n_estimators=5, max_depth=2, n_jobs=1)
    model.train(training.features.loc[complete], training.labels.loc[complete],
                context=snapshot.context.model_copy(update={'as_of':fit_at.to_pydatetime()}),
                feature_available_at=training.feature_available_at.loc[complete],
                label_available_at=training.label_available_at.loc[complete],
                feature_spec_hash=dataset.feature_spec_hash, data_snapshot_hash=dataset.snapshot_hash)
    model.save(str(tmp_path / f'{name}.joblib'))
    args = args_for(snapshot, dataset)
    args['regime'] = regime_for(snapshot.context, spec.regimes[0])
    result = evaluate_strategy(spec, dataset, snapshot.context, model=model, **args)
    assert result.target_weight.sum() > 0, result
    assert result.target_weight.max() <= spec.position_cap
    assert result.target_weight.sum() <= spec.capital_fraction
    assert result.attrs['model_hash'] == model.artifact.content_hash()
    assert (result.action == 'BUY').any()
    # Repeated same-session rebalance retains actual holdings without predicting.
    args.update(holdings=result.target_weight, last_rebalanced_at=snapshot.context.as_of)
    held = evaluate_strategy(spec, dataset, snapshot.context, **args)
    pd.testing.assert_series_equal(held.target_weight, result.target_weight)
    assert not held.attrs['cadence_due']


def test_consensus_preserves_hold_abstain_veto_and_explicit_exits():
    spec = baseline_specs()['agent_consensus']
    snapshot, dataset = market_inputs(spec)
    args = args_for(snapshot, dataset)
    ids = list(args['universe'].instrument_ids)
    args['holdings'] = pd.Series(.04, index=ids)
    votes = {key:SimpleNamespace(action=action, agreeing_roles=('fundamental','sentiment','technical','risk_manager'),
                                derived_votes=(('strategist',('fundamental','technical')),), reason='test consensus')
             for key, action in zip(ids, ('HOLD','ABSTAIN','VETO'))}
    result = evaluate_strategy(spec, dataset, snapshot.context, consensus=votes, **args)
    assert result.target_weight.tolist() == [.04]*3
    votes[ids[0]].action, votes[ids[1]].action = 'REDUCE', 'EXIT'
    result = evaluate_strategy(spec, dataset, snapshot.context, consensus=votes, **args)
    assert result.target_weight.tolist() == [.02, 0, .04]
    assert result.attrs['consensus_votes'][ids[0]]['derived_votes']


def test_universe_removal_missing_coverage_and_activation_fail_closed():
    spec = baseline_specs()['agent_consensus']
    snapshot, dataset = market_inputs(spec)
    args = args_for(snapshot, dataset)
    ids = list(args['universe'].instrument_ids)
    args['holdings'] = pd.Series(.04, index=ids)
    args['universe'].instrument_ids = tuple(ids[1:])
    dataset.features.loc[(snapshot.context.as_of, ids[1]), 'adv_20'] = np.nan
    result = evaluate_strategy(spec, dataset, snapshot.context, **args)
    assert result.loc[ids[0], 'action'] == 'EXIT'
    assert result.loc[ids[0], 'target_weight'] == 0
    assert result.loc[ids[1], 'action'] == 'ABSTAIN'
    assert result.loc[ids[1], 'target_weight'] == .04
    paper = snapshot.context.model_copy(update={'mode':'paper'})
    with pytest.raises(StrategyInputError, match='approval'):
        evaluate_strategy(spec, dataset, paper, **args)
    candidate = spec.model_copy(update={'origin':'research_candidate'})
    with pytest.raises(StrategyInputError, match='not active'):
        evaluate_strategy(candidate, dataset, snapshot.context, resolver=lambda *a, **k:None, **args)


def test_attribution_returns_cutoffs_and_allocation_modes(tmp_path):
    specs = baseline_specs()
    snapshot, _ = market_inputs(specs['momentum_topk'])
    context = snapshot.context
    store = AttributedPerformanceStore(tmp_path / 'nav.sqlite')
    # Actual ledger-like Decimal reconciliation, synthetic only in this test.
    for name, scale in [('momentum_topk',1), ('crypto_trend',.5)]:
        nav = Decimal(100)
        for i in range(30):
            end = pd.Timestamp(context.as_of).normalize()-pd.Timedelta(days=30-i)
            closed = nav * Decimal(str(1+scale*(.003 if i%3 else -.001)))
            store.record_nav(name, started_at=end-pd.Timedelta(days=1), ended_at=end, available_at=end,
                             opening_nav=nav, closing_nav=closed, end_flow=Decimal(0), source_event_id=f'{name}:{i}', source_event_hash='a'*64)
            nav = closed
    performance = store.metrics(as_of=context.as_of, annualization=365)
    available_specs = tuple(specs.values())
    regime = regime_for(context)
    assert select_allocations(available_specs, context, regime, mode='manual', manual='crypto_trend') == {'crypto_trend':1}
    assert select_allocations(available_specs, context, regime, mode='regime_matched', performance=performance)
    blend = select_allocations(available_specs, context, regime, mode='performance_weighted', performance=performance)
    assert sum(blend.values()) == pytest.approx(1) and len(blend) == 2
    first = select_allocations(available_specs, context, regime, mode='bandit', performance=performance, seed=4, exploration=1)
    assert first == select_allocations(available_specs, context, regime, mode='bandit', performance=performance, seed=4, exploration=1)
    early = store.metrics(as_of=context.as_of-pd.Timedelta(days=20), annualization=365)
    assert early.observations.max() < performance.observations.min()
    with pytest.raises(StrategyInputError, match='unavailable'):
        select_allocations(available_specs, context, regime, mode='performance_weighted')


def test_ledger_flow_drawdown_and_duplicate_invariants(tmp_path):
    store = AttributedPerformanceStore(tmp_path / 'nav.sqlite')
    end = pd.Timestamp('2020-01-02', tz='UTC').to_pydatetime()
    values = dict(strategy='loss', started_at=end-pd.Timedelta(days=1), ended_at=end, available_at=end,
                  opening_nav=Decimal(100), closing_nav=Decimal(180), end_flow=Decimal(100), source_event_id='event',source_event_hash='b'*64)
    store.record_nav(**values)
    store.record_nav(**values)
    metrics = store.metrics(as_of=end, annualization=365)
    assert metrics.loc['loss','max_drawdown'] == pytest.approx(-.2)
    assert metrics.loc['loss','total_return'] == pytest.approx(-.2)
    assert metrics.loc['loss','observations'] == 1
    assert np.isnan(metrics.loc['loss','sharpe'])
    with pytest.raises(ValueError, match='rewritten'):
        store.record_nav(**{**values,'closing_nav':Decimal(181)})


def test_netted_sleeves_veto_and_omitted_holdings():
    spec = baseline_specs()['agent_consensus']
    snapshot, dataset = market_inputs(spec)
    args = args_for(snapshot, dataset)
    ids = list(args['universe'].instrument_ids)
    votes = {key:SimpleNamespace(action='BUY',agreeing_roles=('fundamental','sentiment','technical','risk_manager'),derived_votes=(),reason='buy') for key in ids}
    output = evaluate_strategy(spec, dataset, snapshot.context, consensus=votes, **args)
    veto = output.copy(deep=True)
    veto.attrs = {**output.attrs,'strategy':'veto_sleeve'}
    veto.loc[ids[0],'action'] = 'VETO'
    current = pd.Series({ids[0]:.01})
    merged = blend_sleeves({'agent_consensus':output,'veto_sleeve':veto},{'agent_consensus':.5,'veto_sleeve':.5},snapshot.context,current_weights=current,current_contributions={'agent_consensus':current*.5,'veto_sleeve':current*.5})
    assert merged.loc[ids[0],'target_weight'] == .01
    assert sum(merged.attrs['contributions'][s][ids[0]] for s in merged.attrs['contributions']) == pytest.approx(.01)
    with pytest.raises(StrategyInputError, match='omitted'):
        blend_sleeves({'agent_consensus':output},{'agent_consensus':1},snapshot.context,current_weights=pd.Series({'missing':.01}))


def test_weekly_monthly_cadence_and_future_universe():
    from src.strategies.runner import rebalance_due
    specs = baseline_specs()
    assert not rebalance_due(specs['momentum_topk'], pd.Timestamp('2024-01-04',tz='UTC'), pd.Timestamp('2024-01-02',tz='UTC'))
    assert rebalance_due(specs['momentum_topk'], pd.Timestamp('2024-01-08',tz='UTC'), pd.Timestamp('2024-01-02',tz='UTC'))
    assert not rebalance_due(specs['value_quality'], pd.Timestamp('2024-01-31',tz='UTC'), pd.Timestamp('2024-01-02',tz='UTC'))
    assert rebalance_due(specs['value_quality'], pd.Timestamp('2024-02-01',tz='UTC'), pd.Timestamp('2024-01-02',tz='UTC'))
    snapshot, dataset = market_inputs(specs['agent_consensus'])
    args = args_for(snapshot, dataset)
    args['universe'].available_at += pd.Timedelta(days=1)
    with pytest.raises(StrategyInputError, match='universe'):
        evaluate_strategy(specs['agent_consensus'], dataset, snapshot.context, **args)


def test_registry_exact_hash_and_executable_parameter_change():
    spec = baseline_specs()['agent_consensus']
    snapshot, dataset = market_inputs(spec)
    candidate = spec.model_copy(update={'origin':'research_candidate','parameters':(('reduction_fraction',.25),)})
    args = args_for(snapshot, dataset)
    key = args['universe'].instrument_ids[0]
    args['holdings'] = pd.Series({key:.04})
    vote = SimpleNamespace(action='REDUCE', agreeing_roles=('fundamental','sentiment','technical','risk_manager'),derived_votes=(), reason='reduce')
    calls = []
    def resolve(slot, *, payload_hash, as_of):
        calls.append((slot,payload_hash,as_of))
        return SimpleNamespace(payload_hash=payload_hash, activated_at=as_of, candidate_hash='a'*64)
    result = evaluate_strategy(candidate, dataset, snapshot.context, consensus={key:vote}, resolver=resolve, **args)
    assert result.loc[key, 'target_weight'] == pytest.approx(.03)
    assert calls == [('strategy:agent_consensus',candidate.content_hash(),snapshot.context.as_of)]
    # A different activated payload never enables this specification.
    with pytest.raises(StrategyInputError, match='not active'):
        evaluate_strategy(candidate, dataset, snapshot.context,
                          resolver=lambda *a, **k:SimpleNamespace(payload_hash='0'*64,activated_at=snapshot.context.as_of), **args)


def test_sampling_gaps_rejected_before_annualized_sharpe(tmp_path):
    store = AttributedPerformanceStore(tmp_path/'gap.sqlite')
    start = pd.Timestamp('2020-01-01',tz='UTC')
    store.record_nav('gap',started_at=start,ended_at=start+pd.Timedelta(days=1),available_at=start+pd.Timedelta(days=1),
        opening_nav=Decimal(100),closing_nav=Decimal(101),end_flow=Decimal(0),source_event_id='first',source_event_hash='a'*64)
    with pytest.raises(ValueError,match='without gaps'):
        store.record_nav('gap',started_at=start+pd.Timedelta(days=2),ended_at=start+pd.Timedelta(days=3),
            available_at=start+pd.Timedelta(days=3),opening_nav=Decimal(101),closing_nav=Decimal(102),
            end_flow=Decimal(0),source_event_id='missing-day',source_event_hash='b'*64)


def test_blend_preserves_passive_holdings_and_rejects_mixed_provenance():
    spec = baseline_specs()['agent_consensus']
    snapshot, dataset = market_inputs(spec)
    args = args_for(snapshot, dataset)
    key = args['universe'].instrument_ids[0]
    args['holdings'] = pd.Series({key:.04})
    output = evaluate_strategy(spec,dataset,snapshot.context,**args)  # unavailable consensus => ABSTAIN
    merged = blend_sleeves({'agent_consensus':output},{'agent_consensus':.1},snapshot.context,current_weights=args['holdings'])
    assert merged.loc[key,'target_weight'] == .04  # changing allocation does not liquidate ABSTAIN
    assert merged.attrs['sleeve_provenance']['agent_consensus'] == output.attrs
    assert merged.attrs['data_snapshot_hash'] == dataset.snapshot_hash
    other = output.copy(deep=True)
    other.attrs = {**output.attrs,'strategy':'other','universe_version':'different'}
    with pytest.raises(StrategyInputError,match='mismatched'):
        blend_sleeves({'agent_consensus':output,'other':other},{'agent_consensus':.5,'other':.5},snapshot.context,
                      current_weights=args['holdings'],current_contributions={'agent_consensus':args['holdings'],'other':pd.Series(dtype=float)})


def test_blend_common_market_preserves_distinct_actual_feature_datasets():
    spec = baseline_specs()['agent_consensus']
    snapshot, contextual = market_inputs(spec)
    plain = feature_dataset(snapshot, spec.features)
    common = snapshot.content_hash()
    assert contextual.snapshot_hash != plain.snapshot_hash
    context = snapshot.context.model_copy(update={'data_snapshot_ids':(common,contextual.snapshot_hash,plain.snapshot_hash)})
    outputs = {}
    for name,data in [('contextual',contextual),('plain',plain)]:
        args = args_for(snapshot,data)
        args['regime'] = regime_for(context)
        frame = evaluate_strategy(spec,data,context,market_snapshot_hash=common,**args)
        frame.attrs['strategy'] = name
        outputs[name] = frame
    result = blend_sleeves(outputs,{'contextual':.5,'plain':.5},context,current_weights=pd.Series(dtype=float))
    assert result.attrs['market_snapshot_hash'] == result.attrs['data_snapshot_hash'] == common
    assert result.attrs['data_snapshot_hashes'] == {'contextual':contextual.snapshot_hash,'plain':plain.snapshot_hash}
    assert result.attrs['sleeve_provenance']['contextual']['data_snapshot_hash'] == contextual.snapshot_hash
    assert result.attrs['sleeve_provenance']['plain']['data_snapshot_hash'] == plain.snapshot_hash
    args = args_for(snapshot,contextual)
    with pytest.raises(StrategyInputError,match='approved in RunContext'):
        evaluate_strategy(spec,contextual,snapshot.context,market_snapshot_hash=common,**args)
    outputs['plain'].attrs['market_snapshot_hash'] = 'f'*64
    with pytest.raises(StrategyInputError,match='approved in RunContext'):
        blend_sleeves(outputs,{'contextual':.5,'plain':.5},context,current_weights=pd.Series(dtype=float))


def test_mixed_universe_sparse_contributions_route_and_preserve_actual_holdings():
    from src.domain.portfolio import Holding, PortfolioSnapshot, SleeveSnapshot
    from src.execution.routing import target_from_blended_frame

    equity = archived_snapshot(periods=1).bars[0].instrument
    crypto = archived_snapshot(crypto=True, periods=1).bars[0].instrument
    context = archived_snapshot(periods=1).context.model_copy(update={
        'account_id':'paper:mixed', 'data_snapshot_ids':('a'*64, 'b'*64, 'c'*64)})
    held = Holding(instrument=equity, quantity='1.23456789', cost_basis=10,
                   market_value=20, price=Decimal(20)/Decimal('1.23456789'), fx_rate=1,
                   marked_at=context.as_of)
    common = dict(account_id=context.account_id, as_of=context.as_of, reserved_cash=0,
                  realized_pnl=0, fees=0, turnover_today=0)
    eq_portfolio = PortfolioSnapshot(**common, cash=30, nav=50, high_water_mark=50, positions=(held,))
    cr_portfolio = PortfolioSnapshot(**common, cash=50, nav=50, high_water_mark=50)
    actual = PortfolioSnapshot(**common, cash=80, nav=100, high_water_mark=100, positions=(held,),
        sleeves={name:SleeveSnapshot(portfolio=portfolio,net_capital_flow=50,last_rebalanced_at=None,
                                   source_hash='d'*64,source_event_ids=(name,))
                 for name,portfolio in [('equity',eq_portfolio),('crypto',cr_portfolio)]})
    outputs = {}
    for name,instrument,action,weight,source in [('equity',equity,'HOLD',.4,'b'*64),
                                                ('crypto',crypto,'BUY',.6,'c'*64)]:
        frame = pd.DataFrame(dict(target_weight=[weight], action=[action], reason=['test'], score=[1.]),
                             index=[instrument.instrument_id])
        frame.attrs.update(strategy=name, context=context.model_dump(mode='json'),
                           data_snapshot_hash=source, market_snapshot_hash='a'*64, universe_version='current-test',
                           signal_at=context.as_of.isoformat(), strategy_hash='e'*64, feature_spec_hash='f'*64,
                           model_hash=None, model_training_data_hash=None)
        outputs[name] = frame
    result = blend_sleeves(outputs, {'equity':.5,'crypto':.5}, context,
        current_weights=pd.Series({equity.instrument_id:.2}),
        current_contributions={'equity':pd.Series({equity.instrument_id:.2}),'crypto':pd.Series(dtype=float)})
    assert result.attrs['contributions'] == {'equity':{equity.instrument_id:.2},'crypto':{crypto.instrument_id:.3}}
    target = target_from_blended_frame(result, context, (equity,crypto), target_id='mixed',
                                      current_snapshot=actual, outputs=outputs)
    assert target.sleeve_preserve_quantities['equity'][equity.instrument_id] == held.quantity
    assert target.sleeve_contributions == {'equity':{equity.instrument_id:Decimal('.2')},
                                         'crypto':{crypto.instrument_id:Decimal('.3')}}
    assert result.target_weight.sum() == pytest.approx(.5)
