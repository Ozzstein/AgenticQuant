"""Native source identities are proved from archived synthetic provider wires."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from src.domain.portfolio import ExecutionBar

from src.data.adapters.base import ProviderError, store_observation
from src.data.evidence.store import canonical_bytes, digest
from src.data.opening_identity import verified_opening_source_identity
from tests.test_operating_capture import context, primary, read
from tests.test_operating_capture import equity_capture as equity_capture

UTC = timezone.utc


def _equity_print(f, *, feed="iex"):
    f.clock.at = max(f.clock(), datetime(2024, 3, 12, 13, 32, tzinfo=UTC))
    requested = context(datetime(2024, 3, 12, 13, 31, tzinfo=UTC))
    return primary(
        f.adapter.trades(
            f.instrument,
            requested,
            start=datetime(2024, 3, 12, 13, 30, tzinfo=UTC),
            end=requested.as_of,
            calendar_id=f.calendar.record.evidence_id,
            feed=feed,
            symbol_asof="2024-03-12",
        ),
        "operating_prints",
    )


def _archive_variant(f, proof, value, *, version="operating-wire-2"):
    """Tests may construct false normalized claims; runtime must reject them."""
    manifest = {**value["source_manifest"], "implementation": version}
    return store_observation(
        f.store,
        raw=canonical_bytes(manifest),
        normalized={**value, "source_manifest": manifest},
        provider="operating",
        version=version,
        source_uri="test://archived-print-variant",
        kind="operating_prints",
        logical_key=f.instrument.instrument_id,
        context=context(proof.record.requested_as_of),
        ingested_at=f.clock(),
        instrument=f.instrument,
    )


def test_v1_v2_and_feed_aliases_share_one_exact_native_identity(equity_capture):
    f = equity_capture
    proof = _equity_print(f)
    value = read(f.store, proof, f.clock)
    event = value["opening_events"][0]
    native = verified_opening_source_identity(
        f.store, proof.record.evidence_id, event["event_id"], as_of=f.clock()
    )
    assert native.provider == "alpaca"
    assert native.instrument == f.instrument
    assert native.trade_id == "V:9007199254740993123"
    assert native.wire_timestamp == "1710250200123456789"
    assert event["opened_at"] == "2024-03-12T13:30:00.123457Z"
    old_id = digest(canonical_bytes([event["instrument"], value["convention"], native.trade_id]))
    old_bar = ExecutionBar(
        **{k: v for k, v in event.items() if k not in ("price", "event_id")},
        event_id=old_id,
        closed_at=datetime.fromisoformat(event["opened_at"]) + timedelta(microseconds=1),
        open=event["price"],
        high=event["price"],
        low=event["price"],
        close=event["price"],
        volume=event["opening_volume"],
    )
    old_value = {k: v for k, v in value.items() if k != "opening_events"}
    old_value.update(
        execution_bars=[old_bar.model_dump(mode="json")],
        prints=[{**value["prints"][0], "event_id": old_id}],
        opening_liquidity=[{**value["opening_liquidity"][0], "event_id": old_id}],
    )
    old = _archive_variant(f, proof, old_value, version="operating-wire-1")
    assert old_id != event["event_id"]
    assert (
        verified_opening_source_identity(f.store, old.record.evidence_id, old_id, as_of=f.clock())
        == native
    )
    sip = _equity_print(f, feed="sip")
    sip_event = read(f.store, sip, f.clock)["opening_events"][0]
    assert sip_event["event_id"] not in (event["event_id"], old_id)
    assert (
        verified_opening_source_identity(
            f.store, sip.record.evidence_id, sip_event["event_id"], as_of=f.clock()
        )
        == native
    )
    assert f.store.get(old.record.evidence_id).content_hash() == old.content_hash()


@pytest.mark.parametrize(
    "change", ["price", "size", "trade_id", "wire_time", "projection", "duplicate", "instrument"]
)
def test_normalized_print_cannot_fabricate_native_identity(equity_capture, change):
    f = equity_capture
    proof = _equity_print(f)
    value = read(f.store, proof, f.clock)
    event, printed = value["opening_events"][0], value["prints"][0]
    if change == "price":
        event["price"] = "102"
    elif change == "size":
        event["opening_volume"] = "8"
    elif change == "trade_id":
        printed["trade_id"] = "V:9007199254740993124"
    elif change == "wire_time":
        printed["event_time_wire"] = "2024-03-12T13:30:00.123456788Z"
    elif change == "projection":
        event["opened_at"] = "2024-03-12T13:30:00.123456Z"
    elif change == "duplicate":
        value["prints"].append(dict(printed))
    else:
        event["instrument"]["symbol"] = "OTHER"
    event_id = digest(
        canonical_bytes(
            [
                event["instrument"],
                value["convention"],
                printed["trade_id"],
                pd.Timestamp(printed["event_time_wire"]).value,
            ]
        )
    )
    event["event_id"] = printed["event_id"] = event_id
    value["opening_liquidity"] = [
        {k: event[k] for k in ("event_id", "instrument", "opened_at", "opening_volume", "fx_rate")}
    ]
    changed = _archive_variant(f, proof, value)
    with pytest.raises(ProviderError):
        verified_opening_source_identity(
            f.store, changed.record.evidence_id, event_id, as_of=f.clock()
        )


def test_unknown_event_and_future_receipt_fail_closed(equity_capture):
    f = equity_capture
    proof = _equity_print(f)
    event = read(f.store, proof, f.clock)["opening_events"][0]
    with pytest.raises(ProviderError, match="unique"):
        verified_opening_source_identity(
            f.store, proof.record.evidence_id, "unrelated-event", as_of=f.clock()
        )
    with pytest.raises(ProviderError):
        verified_opening_source_identity(
            f.store,
            proof.record.evidence_id,
            event["event_id"],
            as_of=proof.record.ingested_at - timedelta(microseconds=1),
        )


def test_reused_native_id_with_distinct_wire_ns_does_not_merge_capacity(equity_capture):
    f = equity_capture
    first = _equity_print(f)
    old_event = read(f.store, first, f.clock)["opening_events"][0]
    f.state["trade_pages"] = [
        dict(
            trades={
                f.instrument.symbol: [
                    dict(
                        i=9007199254740993123,
                        x="V",
                        t="2024-03-12T13:30:00.123456788Z",
                        p="101.000000000000000001",
                        s="7",
                        c=["@"],
                    )
                ]
            },
            next_page_token=None,
        )
    ]
    second = _equity_print(f)
    new_event = read(f.store, second, f.clock)["opening_events"][0]
    old = verified_opening_source_identity(
        f.store, first.record.evidence_id, old_event["event_id"], as_of=f.clock()
    )
    new = verified_opening_source_identity(
        f.store, second.record.evidence_id, new_event["event_id"], as_of=f.clock()
    )
    assert old.trade_id == new.trade_id
    assert old_event["opened_at"] == new_event["opened_at"]
    assert old_event["event_id"] != new_event["event_id"]
    assert old.wire_timestamp != new.wire_timestamp
    assert old != new  # Projected price/time equality cannot establish an alias.
