from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.data.features.contextual import NumericEvidenceBinding, materialize_evidence_fields
from src.domain.common import EvidenceRecord, RunContext


def date(day):
    return datetime(2020, 1, day, tzinfo=timezone.utc)


def put(store, name, value, available, *, instrument="equity:a", kind="ratios", supersedes=None):
    payload = {"pe": value, "value": value, "market_type": "perpetual" if kind == "funding" else "equity", "date": "2020-01-01"}
    raw = canonical_bytes(payload)
    record = EvidenceRecord(evidence_id=name, provider="test-archive", provider_version="1", source_uri="test://record", raw_hash=digest(raw),
                            available_at=available, ingested_at=date(20), observed_at=date(1), requested_as_of=date(1),
                            availability_basis="documented", quality="usable", instrument_ids=(instrument,))
    return store.put(record, raw, kind=kind, logical_key=f"{instrument}:{kind}:2020-01-01", normalized=payload, supersedes_id=supersedes)


def test_revisions_historical_backfill_conflicts_and_gaps(tmp_path):
    store = EvidenceStore(tmp_path / "evidence")
    put(store, "first", 10, date(2))
    put(store, "revision", 20, date(4), supersedes="first")
    index = pd.MultiIndex.from_product([pd.date_range(date(1), date(5)), ["equity:a"]], names=["datetime", "instrument"])
    context = RunContext(run_id="test", mode="research", as_of=date(5), config_hash="0"*64, code_version="test")
    binding = NumericEvidenceBinding(name="pe", target_instrument_id="equity:a", source_instrument_id="equity:a", kind="ratios", value_path=("pe",), max_age_seconds=86400*10)
    fields = materialize_evidence_fields(store, index, context, (binding,))
    assert pd.isna(fields.values.iloc[0, 0])
    assert fields.values.iloc[1:, 0].tolist() == [10, 10, 20, 20]
    assert fields.source_ids.iloc[1] == ("first",)
    assert fields.source_ids.iloc[-1] == ("revision",)
    put(store, "contradiction", 99, date(5))
    conflict = materialize_evidence_fields(store, index, context, (binding,))
    assert pd.isna(conflict.values.iloc[-1, 0]) and pd.isna(conflict.available_at.iloc[-1])
    assert conflict.coverage[-1][-1] == "conflicting"


def test_perpetual_feature_binding_does_not_change_spot_identity(tmp_path):
    store = EvidenceStore(tmp_path / "evidence")
    put(store, "funding", .001, date(2), instrument="perp:btc", kind="funding")
    context = RunContext(run_id="test", mode="research", as_of=date(3), config_hash="0"*64, code_version="test")
    index = pd.MultiIndex.from_tuples([(date(3), "spot:btc")], names=["datetime", "instrument"])
    binding = NumericEvidenceBinding(name="funding_rate", target_instrument_id="spot:btc", source_instrument_id="perp:btc", kind="funding", value_path=("value",), source_market="perpetual", max_age_seconds=86400*3)
    fields = materialize_evidence_fields(store, index, context, (binding,))
    assert fields.values.index[0][1] == "spot:btc"
    assert fields.values.iloc[0, 0] == .001
    assert fields.source_ids.iloc[0] == ("funding",)
    with pytest.raises(ValueError, match="within requested cutoff"):
        materialize_evidence_fields(store, index, context.model_copy(update={"as_of": date(2)}), (binding,))
