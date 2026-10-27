from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.data.evidence.identity import SecurityMaster
from src.data.evidence.store import EvidenceStore, digest
from src.domain.common import EvidenceRecord, Instrument, RunContext
from src.domain.market import MarketBar, MarketSnapshot

UTC = timezone.utc
T = datetime(2020, 1, 2, 21, tzinfo=UTC)


def record(key="one", raw=b"raw", **kwargs):
    values = dict(evidence_id=key, provider="test", provider_version="1", source_uri="https://example.org/fact",
                  raw_hash=digest(raw), ingested_at=T + timedelta(days=1000),
                  available_at=T, observed_at=T, requested_as_of=T,
                  availability_basis="documented", quality="usable", instrument_ids=("equity:1",))
    values.update(kwargs)
    return EvidenceRecord(**values)


@pytest.fixture
def store(tmp_path):
    return EvidenceStore(tmp_path / "evidence.sqlite")


def test_backfill_revision_cutoff_and_contradictions(store):
    store.put(record(), b"raw", kind="earnings", logical_key="2019Q4", normalized={"eps": 1})
    store.put(record("revision", b"revision", available_at=T + timedelta(days=30)), b"revision",
              kind="earnings", logical_key="2019Q4", normalized={"eps": 0.8},
              revised_at=T + timedelta(days=30), supersedes_id="one")
    old = store.query(as_of=T, kind="earnings")
    assert [e.record.evidence_id for e in old.records] == ["one"]
    assert old.excluded[0][0] == "revision"
    current = store.query(as_of=T + timedelta(days=40), current_revisions_only=True)
    assert [e.record.evidence_id for e in current.records] == ["revision"]
    assert len(store.query(as_of=T + timedelta(days=40)).records) == 2
    store.put(record("other", b"other", provider="other"), b"other", kind="earnings",
              logical_key="2019Q4", normalized={"eps": 3})
    contradictory = store.query(as_of=T)
    assert contradictory.status == "conflicting"
    assert len(contradictory.records) == 2
    assert store.raw("one") == b"raw"


def test_missing_unknown_and_future_are_distinct(store):
    assert store.query(as_of=T).status == "missing"
    store.put(record(available_at=None, availability_basis="unknown"), b"raw", kind="news", logical_key="1")
    result = store.query(as_of=T)
    assert result.status == "ineligible"
    assert "documented" in result.excluded[0][1]
    with pytest.raises(ValueError, match="documented"):
        store.get("one", as_of=T)


def test_full_document_bounded_exact_passages_and_immutability(store):
    content = ("ordinary unicode café 🧾. " * 900) + "GUIDANCE CUT — demand fell."
    store.put(record(), b"raw", kind="transcript", logical_key="Q4", document=content)
    assert store.document("one", as_of=T).text == content
    passages = store.retrieve(["one"], as_of=T, query="GUIDANCE", max_chars=500, passage_chars=300)
    assert sum(len(p.text) for p in passages) <= 500
    assert any("GUIDANCE" in p.text for p in passages)
    for passage in passages:
        assert content[passage.start:passage.end] == passage.text
        assert passage == store.passage("one", passage.start, passage.end, as_of=T)
    with pytest.raises(ValueError, match="immutable"):
        store.put(record(), b"raw", kind="transcript", logical_key="Q4", document="replacement")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.put(record(), b"tampered", kind="news", logical_key="x")
    with pytest.raises(ValueError, match="unavailable"):
        store.document("one", as_of=T - timedelta(seconds=1))
    assert EvidenceStore(store.path).document("one", as_of=T).text == content


def test_symbol_reuse_and_future_universe_rejected(store):
    base = dict(symbol="ABC", asset_class="equity", quote_currency="USD", venue="XNYS")
    for n, first, last in [(1, T - timedelta(days=100), T), (2, T, None)]:
        item = dict(instrument=Instrument(instrument_id=f"equity:{n}", **base).model_dump(mode="json"),
                    symbol="ABC", venue="XNYS", valid_from=first.isoformat(), valid_to=last.isoformat() if last else None)
        store.put(record(str(n), available_at=first, requested_as_of=first), b"raw", kind="symbol_alias",
                  logical_key=str(n), normalized=item)
    master = SecurityMaster(store)
    assert master.resolve("ABC", "XNYS", at=T - timedelta(days=1), as_of=T).instrument_id == "equity:1"
    assert master.resolve("ABC", "XNYS", at=T, as_of=T).instrument_id == "equity:2"
    with pytest.raises(ValueError, match="coverage"):
        master.members("SP500", at=T, as_of=T)


def test_market_snapshot_cutoffs_price_conventions_and_decimal():
    instrument = Instrument(instrument_id="equity:1", symbol="ABC", asset_class="equity", quote_currency="USD")
    bar = MarketBar(instrument=instrument, opened_at=T - timedelta(hours=6), closed_at=T,
                    open="10.01", high="11.02", low="10", close="10.04", volume="100", evidence_id="one",
                    price_basis="raw", market_type="equity", calendar="XNYS", session="2020-01-02")
    assert bar.close == Decimal("10.04")
    context = RunContext(run_id="test", mode="backtest", as_of=T, config_hash="0" * 64, code_version="test")
    snap = MarketSnapshot(context=context, bars=(bar,), evidence=(record(),))
    assert snap.content_hash()
    with pytest.raises(ValueError, match="before it closes"):
        MarketSnapshot(context=context, bars=(bar,), evidence=(record(available_at=T-timedelta(seconds=1)),))
    with pytest.raises(ValueError, match="duplicate"):
        MarketSnapshot(context=context, bars=(bar, bar), evidence=(record(),))
    with pytest.raises(ValueError, match="raw"):
        bar.model_copy(update={"price_basis": "total_return"}).require_execution_price()


def test_requested_cutoff_and_snapshot_integrity(store):
    with pytest.raises(ValueError, match="explicit requested"):
        store.put(record(requested_as_of=None), b"raw", kind="news", logical_key="x")
    key = store.save_snapshot({"run": "one", "ids": ["1", "2"]})
    assert store.snapshot(key)["ids"] == ["1", "2"]
