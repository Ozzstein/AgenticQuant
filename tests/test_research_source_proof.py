"""Archived values, rather than plausible hashes, bind executable research data."""
from decimal import Decimal

import pytest

from src.data.evidence.store import EvidenceStore
from src.research.evidence import expected_artifacts, verify_market_snapshot
from src.research.registry import RegistryError
from tests.data_quant_fixtures import archived_snapshot
from tests.test_research_inputs import research_inputs


def test_normalized_bar_substitution_with_same_source_ids_is_rejected(tmp_path):
    snapshot = archived_snapshot(crypto=True,periods=3)
    store = EvidenceStore(tmp_path/'evidence.db')
    bars = {b.evidence_id:b for b in snapshot.bars}
    for record in snapshot.evidence:
        store.put(record,record.evidence_id.encode(),kind='bar',logical_key=record.evidence_id,
                  normalized=bars[record.evidence_id].model_dump(mode='json',exclude={'evidence_id'}))
    verify_market_snapshot(store,snapshot)
    first = snapshot.bars[0]
    forged = first.model_copy(update={'volume':first.volume+Decimal(1)})
    changed = snapshot.model_copy(update={'bars':(forged,*snapshot.bars[1:])})
    with pytest.raises(RegistryError,match='normalized source'):
        verify_market_snapshot(store,changed)


def test_opening_liquidity_fx_and_metadata_are_independent_required_source_proofs():
    inputs = research_inputs()
    expected = expected_artifacts(inputs)
    assert len(expected['execution_bars']) == len(inputs.bars)
    assert expected['opening_liquidity'][0]['opening_volume'] == str(inputs.bars[0].opening_volume)
    assert 'volume' not in expected['opening_liquidity'][0]
    assert 'fx_rate' in expected['opening_liquidity'][0]
    assert expected['instrument_metadata'] == [v.model_dump(mode='json') for v in inputs.metadata]
