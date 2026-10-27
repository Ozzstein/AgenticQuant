"""Uses the actual agent-lane contracts when integrated; no duplicate schemas."""
from datetime import datetime, timezone

import pytest

from src.data.evidence.store import EvidenceStore, digest
from src.data.features.text import store_text_feature
from src.domain.common import EvidenceRecord, Instrument, RunContext


def test_text_generation_clock_and_exact_source_integrity(tmp_path):
    decisions = pytest.importorskip('src.domain.decisions')
    features = pytest.importorskip('src.decision.features')
    source_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    generated = datetime(2025, 1, 1, tzinfo=timezone.utc)
    context = RunContext(run_id='text',mode='research',as_of=source_time,config_hash='0'*64,code_version='test')
    instrument = Instrument(instrument_id='equity:test',symbol='TEST',asset_class='equity',quote_currency='USD')
    text = 'Complete transcript: demand is slowing. Ω retained exactly.'
    raw = text.encode()
    record = EvidenceRecord(evidence_id='transcript',provider='test',provider_version='1',source_uri='test://full',raw_hash=digest(raw),
                            available_at=source_time,ingested_at=generated,requested_as_of=source_time,
                            availability_basis='documented',quality='usable',instrument_ids=(instrument.instrument_id,))
    store = EvidenceStore(tmp_path/'evidence')
    store.put(record,raw,kind='transcript',logical_key='call',document=text)
    source = decisions.DecisionSource.from_document(record,text,start=0,end=len(text),kind='transcript')
    feature = features.TextFeature(name='demand',transformation_version='score-v1',value=.8,status='ok',
                                   evidence_available_at=source_time,generated_at=generated,observation_ids=('actual-test-observation',),
                                   question_hash='1'*64,model='jev-test',policy_hash='2'*64,source_hashes=(source.content_hash(),))
    envelope = store_text_feature(store,feature,instrument,context,sources=(source,),ingested_at=generated)
    assert envelope.record.available_at == generated
    assert not store.query(as_of=source_time,kind='text_feature').records
    assert store.query(as_of=generated,kind='text_feature').records[0] == envelope
    assert store.normalized(envelope.record.evidence_id,as_of=generated)['value'] == .8
    missing = feature.model_copy(update={'value':None,'status':'insufficient_evidence'})
    absent = store_text_feature(store,missing,instrument,context,sources=(source,),ingested_at=generated)
    assert absent.record.quality == 'missing'
    assert store.normalized(envelope.record.evidence_id,as_of=generated)['source_evidence_ids'] == ['transcript']
    forged = source.model_copy(update={'text':'x'*len(text)})
    altered = feature.model_copy(update={'source_hashes':(forged.content_hash(),)})
    with pytest.raises(ValueError,match='span'):
        store_text_feature(store,altered,instrument,context,sources=(forged,),ingested_at=generated)
