"""Late execution receipts are paper evidence, never historical availability."""
from datetime import timedelta
from decimal import Decimal

import pytest

from src.data.evidence.store import EvidenceStore
from src.domain.common import EvidenceRecord, RunContext, VersionedContract
from src.domain.market import MarketSnapshot
from src.domain.portfolio import (
    AccountSpec,
    CorporateAction,
    ExecutionBar,
    InstrumentRisk,
    UniverseEligibility,
)
from src.execution.venue import VenueRules
from src.operations.evidence import verify_paper_provenance
from src.research.evidence import (
    BoundArtifact,
    SourceBinding,
    expected_artifacts,
    verify_provenance,
)
from src.research.registry import RegistryError, ResearchRegistry, canonical, content_hash
from tests.data_quant_fixtures import archived_snapshot


class ReceiptFixture(VersionedContract):
    event_id: str
    evidence_id: str
    evidence_hash: str
    observed_at: object
    available_at: object


class PaperFixture(VersionedContract):
    context: RunContext
    snapshot: MarketSnapshot
    account: AccountSpec
    bars: tuple[ExecutionBar,...]
    metadata: tuple[InstrumentRisk,...]
    universes: tuple[UniverseEligibility,...]
    actions: tuple[CorporateAction,...]=()
    rules: tuple[VenueRules,...]
    receipts: tuple[ReceiptFixture,...]=()


@pytest.fixture
def paper_evidence(tmp_path):
    store=EvidenceStore(tmp_path/'source.db')
    registry=ResearchRegistry(tmp_path/'registry.db')
    snapshot=archived_snapshot(crypto=True,periods=3)
    market=snapshot.bars[-1]
    for b in snapshot.bars:
        record=next(r for r in snapshot.evidence if r.evidence_id==b.evidence_id)
        store.put(record,record.evidence_id.encode(),kind='bar',logical_key=record.evidence_id,
                  normalized=b.model_dump(mode='json',exclude={'evidence_id'}))
    opened=market.opened_at
    cutoff=market.closed_at+timedelta(seconds=10)
    received=market.closed_at+timedelta(seconds=1)
    account=AccountSpec(account_id='paper:proof',mode='paper',initial_cash=10000,base_currency='USD',
                        opened_at=opened-timedelta(days=10))
    ctx=snapshot.context.model_copy(update={'mode':'paper','account_id':account.account_id,'as_of':cutoff,
                                          'data_snapshot_ids':(snapshot.content_hash(),)})
    bar=ExecutionBar(event_id='print-1',instrument=market.instrument,opened_at=opened,closed_at=market.closed_at,
        open=market.open,high=market.high,low=market.low,close=market.close,volume=market.volume,opening_volume=1,
        calendar='24/7',calendar_version='fixture',session=market.session,session_open=opened,session_close=market.closed_at)
    rules=(VenueRules(venue='paper',instrument=market.instrument,symbol=market.instrument.symbol,
        available_at=opened,quantity_step='.001',price_tick='.01',market_type='spot',active=True),)
    inputs=PaperFixture(context=ctx,snapshot=snapshot,account=account,bars=(bar,),rules=rules,
        metadata=(InstrumentRisk(instrument=market.instrument,sector='crypto',adv_base=10000000,as_of=opened),),
        universes=(UniverseEligibility(universe_version='fixture',data_snapshot_id=snapshot.content_hash(),
                    instrument_ids=(market.instrument.instrument_id,),effective_at=opened,available_at=opened),))
    names={'universe_history':'universe_history_id','availability_audit':'availability_audit_id',
           'calendar':'calendar_artifact_id','corporate_actions':'corporate_actions_id','execution_bars':'execution_bars_id',
           'opening_liquidity':'opening_liquidity_id','instrument_metadata':'instrument_metadata_id'}
    sources={}
    for kind,value in {**expected_artifacts(inputs),'venue_rules':[r.model_dump(mode='json') for r in rules]}.items():
        at=received if kind=='opening_liquidity' else opened-timedelta(seconds=1)
        raw=canonical(value)
        record=EvidenceRecord(evidence_id=kind,provider='synthetic-fixture',provider_version='1',source_uri='test://'+kind,
            raw_hash=content_hash(raw),ingested_at=at,available_at=at,requested_as_of=at,quality='usable',availability_basis='documented')
        envelope=store.put(record,raw,kind=kind,logical_key=kind,normalized=value)
        sources[kind]=(SourceBinding(evidence_id=kind,available_at=at),)
        if kind=='opening_liquidity':
            receipt=ReceiptFixture(event_id=bar.event_id,evidence_id=kind,evidence_hash=envelope.content_hash(),
                                   observed_at=at,available_at=at)
    inputs=inputs.model_copy(update={'receipts':(receipt,)})
    def bind(value):
        return {names[kind]:registry.archive_contract(BoundArtifact(kind=kind,inputs_hash=value.content_hash(),
                    value_json=canonical(payload).decode(),sources=sources[kind]))
                for kind,payload in expected_artifacts(value).items()}
    return registry,store,inputs,bind,sources


def test_actual_late_paper_receipt_passes_but_same_sources_fail_historical_gate(paper_evidence):
    registry,store,inputs,bind,sources=paper_evidence
    refs=bind(inputs)
    assert verify_paper_provenance(registry,store,inputs,refs,receipts=inputs.receipts,
                                  rule_sources=sources['venue_rules'])==refs
    with pytest.raises(RegistryError,match='unavailable at the observed open'):
        verify_provenance(registry,store,inputs,refs)


@pytest.mark.parametrize('field',['evidence_hash','observed_at','available_at','event_id'])
def test_paper_receipt_cannot_substitute_source_or_any_clock(paper_evidence,field):
    registry,store,inputs,bind,sources=paper_evidence
    original=inputs.receipts[0]
    value='f'*64 if field=='evidence_hash' else 'unrelated' if field=='event_id' else getattr(original,field)+timedelta(seconds=1)
    changed=inputs.model_copy(update={'receipts':(original.model_copy(update={field:value}),)})
    with pytest.raises(RegistryError):
        verify_paper_provenance(registry,store,changed,bind(changed),receipts=changed.receipts,
                                rule_sources=sources['venue_rules'])


def test_rule_and_opening_quantity_changes_require_exact_source_contents(paper_evidence):
    registry,store,inputs,bind,sources=paper_evidence
    changed=inputs.model_copy(update={'bars':(inputs.bars[0].model_copy(update={'opening_volume':Decimal(2)}),)})
    with pytest.raises(RegistryError,match='opening print contents differ'):
        verify_paper_provenance(registry,store,changed,bind(changed),receipts=changed.receipts,
                                rule_sources=sources['venue_rules'])
    with pytest.raises(RegistryError,match='venue rules'):
        verify_paper_provenance(registry,store,inputs,bind(inputs),receipts=inputs.receipts)


def test_late_paper_receipt_is_not_available_to_earlier_cycle(paper_evidence):
    registry,store,inputs,bind,sources=paper_evidence
    early=inputs.model_copy(update={'context':inputs.context.model_copy(update={'as_of':inputs.bars[0].closed_at})})
    with pytest.raises(RegistryError,match='chronology'):
        verify_paper_provenance(registry,store,early,bind(early),receipts=early.receipts,
                                rule_sources=sources['venue_rules'])
