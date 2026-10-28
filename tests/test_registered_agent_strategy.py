"""Synthetic provider/clock fixtures exercising actual generation and ledger authorities."""
import importlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from src.agents.contracts import PortfolioReviewContext, hmm_context_from_result
from src.agents.graph import analyze_instrument
from src.agents.history import DeskHistoryStore
from src.core.backtester import recorded_decisions
from src.core.benchmark import BenchmarkRunner
from src.data.evidence.store import canonical_bytes, digest
from src.decision.service import DecisionService, OperatingLimits
from src.decision.store import DecisionStore
from src.decision.transport import JevHTTPTransport
from src.domain.common import EvidenceRecord
from src.domain.decisions import CitedClaim, ConfidenceSemantics, DecisionSource, SpecialistReport
from src.domain.portfolio import AccountSpec
from src.execution.ledger import PortfolioLedger
from src.execution.sleeves import SleeveBook
from src.research.registry import Candidate, PromotionPolicy, RegistryError, Trial
from src.research.strategy_experiments import (
    HISTORICAL_ADVICE_POLICY,
    AgentHistoryAuthorities,
    StrategyExperimentPlan,
    _verify_agent_decisions,
    archive_agent_decision,
    execute_registered_strategy,
    recompute_agent_decisions,
)
from src.strategies.specification import baseline_specs
from src.utils.config import AppConfig
from tests.test_registered_quant_experiments import experiment  # noqa: F401


class ReportProvider:
    """Test-only provider with actual schema/usage settlement through the full graph."""
    def __init__(self, role, action):
        self.role, self.action = role, action

    def with_structured_output(self, schema, include_raw=False):
        assert schema is SpecialistReport and include_raw
        return self

    def invoke(self, messages):
        source = json.loads(messages[-1].content)['sources'][0]
        report = SpecialistReport(role=self.role, action='HOLD' if self.role == 'risk_manager' else self.action,
            status='ok', reasoning='Synthetic test provider: source demand and risk reviewed.',
            confidence=ConfidenceSemantics(analyst_conviction=.6,evidence_completeness=.5),
            claims=(CitedClaim(claim='Demand declined',evidence_id=source['evidence_id'],
                              passage_hash=source['passage_hash'],quote='Demand fell 10%.'),))
        return {'parsed':report,'parsing_error':None,'raw':SimpleNamespace(
            usage_metadata={'input_tokens':100,'output_tokens':100},response_metadata={'model':'fixture-chat'})}


def semantic_provider(req):
    payload = json.loads(req.content)
    answers = {}
    for key, question in payload['questions'].items():
        choice = {'claim_support':'supported','debate_discrepancy':'consistent','relevance':'direct','conflict':'conflict'}.get(key,next(iter(question['criteria'])))
        answers[key] = {'type':'choice','choice':choice,'confidence':1.,
                        'probabilities':{k:float(k==choice) for k in question['criteria']}}
    return httpx.Response(200,json={'model':payload['model'],'answers':answers,'usage':{'input_tokens':100,'output_tokens':100}})


@pytest.fixture(scope='module')
def agent_experiment(experiment,tmp_path_factory):  # noqa: F811
    import pytest_socket
    pytest_socket.enable_socket()  # Replace the earlier guard; do not nest guards.
    pytest_socket.disable_socket(allow_unix_socket=True)  # asyncio self-pipe only; TCP stays blocked.
    _,store,registry,numeric,artifact,_ = experiment
    path = tmp_path_factory.mktemp('agent-registered')
    observations = artifact.observations
    # Historical observations really precede generation; test clocks never enter runtime APIs.
    original = next(o for o in observations[:-4] if o.status == 'available' and o.regime != 'crisis')
    dates = tuple(o.context.as_of for o in observations if o.context.as_of > original.context.as_of)[:4]
    assert len(dates)==4
    original_path = path/'original-paper.sqlite'
    ledger = PortfolioLedger(original_path)
    ledger.create_account(AccountSpec(account_id='paper:observed-advice',mode='paper',base_currency='USD',initial_cash=123456,
                                     opened_at=original.context.as_of-timedelta(days=1)))
    book = SleeveBook(ledger,'paper:observed-advice').initialize({'observed-alpha':Decimal(100000),'observed-beta':Decimal(23456)})
    snapshot = ledger.snapshot('paper:observed-advice',original.context.as_of,86400)
    snapshot = snapshot.model_copy(update={'sleeves':book.snapshot(original.context.as_of,86400)})
    ledger.close()
    portfolio = PortfolioReviewContext(as_of=snapshot.as_of,snapshot_id=snapshot.content_hash(),
        portfolio_json=snapshot.model_dump_json(),correlation_json='{"positions":[]}')
    cfg = AppConfig(_env_file=None,output_dir=str(path/'output'),log_dir=str(path/'logs'))
    cfg.agents = cfg.agents.model_copy(update={'llm_model':'fixture-chat','deep_think_model':'fixture-chat',
        'budget_store_path':str(path/'budget.sqlite'),'max_tokens_per_analysis':2000000,'max_tokens_per_day':20000000,
        'max_requests_per_day':2000,'pricing':[SimpleNamespace(provider=cfg.agents.llm_provider,model='fixture-chat',
            price_version='synthetic-test-only',source_uri='test://rate',effective_at=original.context.as_of,
            input_usd_per_million=Decimal('.01'),output_usd_per_million=Decimal('.02'))]})
    cfg.memory.enabled = False
    service = DecisionService(DecisionStore(path/'jev.sqlite'),transport=JevHTTPTransport('fixture',transport=httpx.MockTransport(semantic_provider)),
        limits=OperatingLimits(min_interval_seconds=0,run_spend_usd=1,run_tokens=10000000,day_tokens=100000000,day_requests=10000))
    authorities = AgentHistoryAuthorities(DeskHistoryStore(Path(cfg.output_dir)/'desk-history.sqlite3'),service.store,
                                         cfg.agents.budget_store_path,{'paper:observed-advice':original_path})
    identities = {b.instrument.instrument_id:b.instrument for b in numeric.inputs.bars}
    references = []
    for step,(at,action) in enumerate(zip(dates,('BUY','HOLD','EXIT'),strict=False)):
        clock = original.context.as_of + timedelta(seconds=step+1)
        class FixtureClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.astimezone(tz or timezone.utc)
        with pytest.MonkeyPatch.context() as patch:
            for module in ('src.agents.budget','src.agents.receipts','src.agents.history','src.decision.service','src.decision.store'):
                patch.setattr(importlib.import_module(module),'datetime',FixtureClock)
            for key,instrument in identities.items():
                context = numeric.inputs.context.model_copy(update={'account_id':snapshot.account_id,'mode':'paper',
                    'run_id':f'observed-{step}-{key}','as_of':original.context.as_of})
                text = 'Management said: Demand fell 10%. Guidance was lowered.'
                sources = []
                for kind in ('fundamental','technical','news','transcript','macro'):
                    evidence_id = f'advice-{key}-{kind}'
                    record = EvidenceRecord(evidence_id=evidence_id,provider='synthetic-test-only',provider_version='1',
                        source_uri='test://'+evidence_id,raw_hash=digest(text.encode()),ingested_at=context.as_of,
                        observed_at=context.as_of,available_at=context.as_of,requested_as_of=context.as_of,
                        availability_basis='documented',quality='usable',instrument_ids=(key,))
                    store.put(record,text.encode(),kind=kind,logical_key=evidence_id,document=text)
                    sources.append(DecisionSource.from_document(record,text,start=0,end=len(text),kind=kind))
                decision = analyze_instrument(instrument,context,sources=tuple(sources),hmm_context=hmm_context_from_result(original),
                    portfolio_context=portfolio,config=cfg,decision_service=service,llm_factory=lambda role:ReportProvider(role,action))
                assert decision.action.value == action, decision.reasoning
                reference = archive_agent_decision(registry,store,decision_hash=decision.content_hash(),signal_at=at,authorities=authorities)
                references.append(reference)
    spec = baseline_specs()['agent_consensus'].model_copy(update={'name':'trial-observed-advice','origin':'research_candidate',
        'max_signal_age_seconds':7*86400,'top_k':1})
    plan = StrategyExperimentPlan(inputs=numeric.inputs,strategy=spec,agent_decisions=tuple(references),
        regime_artifact_hash=numeric.regime_artifact_hash,provenance=numeric.provenance)
    candidate = Candidate(candidate_id='observed-advice',kind='strategy',origin='custom',payload_hash=spec.content_hash(),
        code_hash=registry.archive(b'test-code'),environment_hash=registry.archive(b'test-env'),
        source_evidence_hashes=tuple(r.receipt_hash for r in references),created_at=plan.inputs.context.as_of,synthetic=True)
    registry.quarantine(candidate,canonical_bytes(plan.candidate_payload()))
    trial = Trial(trial_id='observed-advice',candidate_id=candidate.candidate_id,family_id='strategy',dataset_hash=plan.inputs.snapshot.content_hash(),
        split_hash=registry.archive(canonical_bytes(plan.inputs.split_manifest())),inputs_hash=registry.archive_contract(plan),
        policy_hash=registry.archive_contract(PromotionPolicy()),registered_at=plan.inputs.context.as_of,purpose='development',
        evaluation_start=dates[0],evaluation_end=dates[3])
    registry.register_trial(trial)
    return store,registry,plan,artifact,trial,authorities,snapshot


def test_archived_advice_real_replay_and_cost_holdings(agent_experiment,tmp_path):
    store,registry,plan,_,trial,authorities,original = agent_experiment
    executed = execute_registered_strategy(registry,trial.trial_id,store=store,ledger_path=tmp_path/'replay.sqlite',
                                           artifact_dir=tmp_path/'artifacts',authorities=authorities)
    result = executed.result
    assert result.trades and not result.walk_forward_results and not result.cross_sectional_ic
    assert len(executed.actual_decision_hashes)==9
    assert not list(executed.artifact_dir.glob('fold-*'))
    assert len(list((executed.artifact_dir/'receipts').glob('*.json')))==9
    assert result.manifest['provenance']['advice_policy']==HISTORICAL_ADVICE_POLICY
    ledger = PortfolioLedger(executed.ledger_path,read_only=True)
    try:
        trace = recorded_decisions(ledger,result)
        ledger.reconcile(result.account_id)
    finally:
        ledger.close()
    assert len(trace)==3 and trace[0].target is not None and trace[1].target is None and trace[2].target is not None
    assert trace[1].snapshot.positions and trace[1].snapshot.cash != original.cash
    opportunities = json.loads((executed.artifact_dir/'opportunities.json').read_text())
    assert [r['eligible'][0]['consensus_action'] for r in opportunities['rows']]==['BUY','HOLD','EXIT']
    assert all(r['model_payload_hash'] is None and all(c['prediction'] is None for c in r['eligible']) for r in opportunities['rows'])
    costs = BenchmarkRunner().cost_sensitivity(inputs=executed.replay_kwargs,output_dir=tmp_path/'costs')
    assert len(costs)==3 and all(c.trades for c in costs)
    assert all(c.account_id.startswith('backtest:cost-') for c in costs)
    assert len({tuple(c.equity_curve) for c in costs}) > 1
    from src.core.benchmark import verify_comparison
    comparison = BenchmarkRunner().compare_replay(result,inputs=executed.replay_kwargs,
        ledger_path=executed.ledger_path,output_dir=tmp_path/'comparisons',n_simulations=3,seed=7)
    assert len(comparison.random_artifacts)==3 and comparison.strategy_artifact is not None
    assert comparison.opportunity_artifact and not comparison.signal_artifact
    from src.utils.exceptions import BacktestError
    with pytest.raises(BacktestError,match="missing durable random replay artifacts"):
        verify_comparison(result,comparison)  # Three diagnostic draws cannot meet the 1000-run gate.
    assert "SPY" in comparison.unavailable
    with pytest.raises(BacktestError,match="missing durable callback named benchmark SPY"):
        verify_comparison(result,comparison,minimum_simulations=3)  # Synthetic archive lacks SPY/BTC coverage.
    assert registry.active('strategy:'+plan.strategy.name) is None


def test_agent_authority_cutoffs_and_contexts(agent_experiment,tmp_path):
    store,registry,plan,artifact,_,authorities,_ = agent_experiment
    verified = recompute_agent_decisions(registry,store,plan,authorities=authorities,work_dir=tmp_path/'recompute')
    assert len(verified)==9
    missing = replace(authorities,portfolio_ledgers={})
    with pytest.raises(RegistryError,match='portfolio authority is missing'):
        _verify_agent_decisions(registry,store,plan,authorities=missing,observations=artifact.observations)
    with pytest.raises(RegistryError,match='HMM context'):
        _verify_agent_decisions(registry,store,plan,authorities=authorities,observations=())
    reference = plan.agent_decisions[0]
    receipt = verified[(pd.Timestamp(reference.signal_at),reference.instrument_id)]
    with pytest.raises(ValueError,match='after signal_at'):
        archive_agent_decision(registry,store,decision_hash=reference.decision_hash,signal_at=receipt.decision.context.as_of,authorities=authorities)
    forged = registry.archive_contract(receipt.model_copy(update={'inputs_json':'{}'}))
    altered = plan.model_copy(update={'agent_decisions':(reference.model_copy(update={'receipt_hash':forged}),)})
    with pytest.raises(RegistryError,match='differs from authoritative history'):
        _verify_agent_decisions(registry,store,altered,authorities=authorities,observations=artifact.observations)
    stale = plan.model_copy(update={'strategy':plan.strategy.model_copy(update={'max_signal_age_seconds':1})})
    with pytest.raises(RegistryError,match='future or stale'):
        _verify_agent_decisions(registry,store,stale,authorities=authorities,observations=artifact.observations)
