"""Actual native wire proof migrates consumed prints without new liquidity."""
from datetime import timedelta
from decimal import Decimal

import pytest

from src.data.evidence.store import canonical_bytes, digest
from src.domain.portfolio import (
    AccountSpec,
    ExecutionBar,
    FillPolicy,
    InstrumentRisk,
    OpeningEvent,
    OrderIntent,
    RiskPolicy,
)
from src.execution.ledger import PortfolioLedger
from src.execution.venue import PaperVenue, VenueExecution, VenueRules
from src.operations.contracts import ObservedOpenReceipt
from src.operations.evidence import prepare_opening_source
from src.research.registry import RegistryError
from tests.test_opening_identity import _archive_variant, _equity_print
from tests.test_operating_capture import equity_capture as equity_capture
from tests.test_operating_capture import read


def setup(f):
    proof = _equity_print(f)
    payload = read(f.store,proof,f.clock)
    event = OpeningEvent.model_validate(payload['opening_events'][0])
    old_id = digest(canonical_bytes([event.instrument.model_dump(mode='json'),payload['convention'],payload['prints'][0]['trade_id']]))
    old = ExecutionBar(**event.model_dump(exclude={'price','event_id'}),event_id=old_id,
        closed_at=event.opened_at+timedelta(microseconds=1),open=event.price,high=event.price,
        low=event.price,close=event.price,volume=event.opening_volume)
    value={k:v for k,v in payload.items() if k!='opening_events'}
    value.update(execution_bars=[old.model_dump(mode='json')],
                 prints=[{**payload['prints'][0],'event_id':old_id}],
                 opening_liquidity=[{**payload['opening_liquidity'][0],'event_id':old_id}])
    legacy=_archive_variant(f,proof,value,version='operating-wire-1')
    metadata_values=read(f.store,f.reference,f.clock)
    metadata={m['instrument']['instrument_id']:InstrumentRisk.model_validate(m) for m in metadata_values['instrument_metadata']}
    rules={r['instrument']['instrument_id']:VenueRules.model_validate(r) for r in metadata_values['venue_rules']}
    ledger=PortfolioLedger(f.store.path.parent/'paper-migration.db')
    account=AccountSpec(account_id='paper:migration',mode='paper',base_currency='USD',initial_cash=100000,
                        opened_at=f.reference.record.available_at)
    ledger.create_account(account)
    costs,risk=FillPolicy(opening_participation=1),RiskPolicy(min_adv=100000)
    venue=PaperVenue(ledger,account.account_id,rules,fill_policy=costs,risk_policy=risk)
    execution=VenueExecution(ledger,account.account_id,{'paper':venue},{f.instrument.instrument_id:'paper'},
                             owner='direct',fill_policy=costs,risk_policy=risk)
    admitted=account.opened_at+timedelta(minutes=1)
    order=OrderIntent(intent_id='migration-order',account_id=account.account_id,run_id='test',instrument=f.instrument,
        side='buy',quantity=8,decision_at=admitted,eligible_at=admitted+timedelta(microseconds=1),reference_price=101,
        estimated_fee='0.808',strategy_id='test',tif='DAY',expires_at=event.session_close)
    execution.prepare(order,metadata,as_of=admitted)
    execution.dispatch(order.intent_id,as_of=order.eligible_at)
    retained={order.intent_id:execution.admission_metadata(order.intent_id)}
    def receipt(envelope,event_id):
        return ObservedOpenReceipt(event_id=event_id,evidence_id=envelope.record.evidence_id,
            evidence_hash=envelope.content_hash(),observed_at=envelope.record.ingested_at,available_at=envelope.record.available_at)
    legacy_receipt=receipt(legacy,old_id)
    fills=venue.settle_open(old,retained,as_of=f.clock(),**legacy_receipt.model_dump(exclude={'schema_version','event_id'}))
    assert sum(v.quantity for v in fills)==Decimal(7)
    return ledger,venue,execution,order,event,receipt(proof,event.event_id),legacy_receipt,retained


def test_source_migration_alias_after_nav_keeps_original_partial_fill_and_restarts(equity_capture):
    f=equity_capture
    ledger,venue,execution,order,event,receipt,old,retained=setup(f)
    before=ledger.snapshot(order.account_id,f.clock(),max_age_seconds=86400)
    native=prepare_opening_source(f.store,venue,event,receipt,as_of=f.clock())
    assert not venue.unbound_opening_receipts(f.instrument)
    assert venue.settle_open(event,retained,as_of=f.clock(),source_identity=native,
        **receipt.model_dump(exclude={'schema_version','event_id'})) == []
    assert ledger.snapshot(order.account_id,f.clock(),max_age_seconds=86400)==before
    assert ledger.quantity(order.account_id,f.instrument.instrument_id)==7
    assert ledger.order(order.account_id,order.intent_id)['remaining']==1
    assert venue.opening_source_binding(event.event_id).consumed_event_id==old.event_id
    ledger.close()
    reopened=PortfolioLedger(ledger.path)
    venue=PaperVenue(reopened,order.account_id,{f.instrument.instrument_id:venue.rules(f.instrument,f.clock())},fill_policy=venue.engine.policy,risk_policy=venue.engine.risk_policy)
    assert prepare_opening_source(f.store,venue,event,receipt,as_of=f.clock())==native
    assert venue.settle_open(event,retained,as_of=f.clock(),source_identity=native,
        **receipt.model_dump(exclude={'schema_version','event_id'})) == []
    assert len(reopened.fills(order.account_id))==1 and reopened.reconcile(order.account_id)['reconciled']
    reopened.close()


def test_migration_authenticates_all_economics_before_committing_any_mapping(equity_capture):
    f=equity_capture
    ledger,venue,_,order,event,receipt,old,_=setup(f)
    with pytest.raises(RegistryError,match='economics'):
        prepare_opening_source(f.store,venue,event.model_copy(update={'price':event.price+1}),receipt,as_of=f.clock())
    assert venue.opening_source_binding(old.event_id) is None
    with pytest.raises(RegistryError,match='original archive'):
        prepare_opening_source(f.store,venue,event,receipt.model_copy(update={'evidence_hash':'0'*64}),as_of=f.clock())
    assert venue.opening_source_binding(old.event_id) is None
    assert len(ledger.fills(order.account_id))==1
    ledger.close()
