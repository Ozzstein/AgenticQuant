"""Synthetic archival data through the actual daily export/verification boundary."""

import json
from datetime import timedelta

import numpy as np
import pytest

from src.data.calendars import sessions
from src.data.evidence.store import canonical_bytes, digest
from src.data.qlib_us import USQlibScenario, export_bundle, verify_bundle
from src.domain.market import MarketSnapshot
from tests.data_quant_fixtures import archived_snapshot


def changed(snapshot, bar, **updates):
    replacement = bar.model_copy(update=updates)
    evidence = tuple(
        r.model_copy(
            update={"available_at": replacement.closed_at, "observed_at": replacement.closed_at}
        )
        if r.evidence_id == bar.evidence_id
        else r
        for r in snapshot.evidence
    )
    return MarketSnapshot(
        context=snapshot.context,
        bars=tuple(replacement if b == bar else b for b in snapshot.bars),
        evidence=evidence,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "minute",
        "late_open",
        "late_close",
        "calendar",
        "holiday",
        "gap",
        "duplicate",
        "missing_vwap",
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_daily_inputs_leave_no_published_files(tmp_path, mutation, existing):
    snapshot = archived_snapshot(periods=5, use_exchange_calendar=True)
    bar = snapshot.bars[-1]
    if mutation == "minute":
        snapshot = changed(snapshot, bar, opened_at=bar.closed_at - timedelta(minutes=1))
    elif mutation == "late_open":
        snapshot = changed(snapshot, bar, opened_at=bar.opened_at + timedelta(seconds=1))
    elif mutation == "late_close":
        snapshot = snapshot.model_copy(
            update={
                "context": snapshot.context.model_copy(
                    update={"as_of": bar.closed_at + timedelta(seconds=1)}
                )
            }
        )
        snapshot = changed(snapshot, bar, closed_at=bar.closed_at + timedelta(seconds=1))
        snapshot = snapshot.model_copy(
            update={
                "context": snapshot.context.model_copy(
                    update={"as_of": bar.closed_at + timedelta(seconds=1)}
                )
            }
        )
    elif mutation == "calendar":
        snapshot = changed(snapshot, bar, calendar="24/7")
    elif mutation == "holiday":
        snapshot = changed(snapshot, bar, session="2020-01-01")
    elif mutation == "gap":
        snapshot = snapshot.model_copy(
            update={"bars": tuple(b for b in snapshot.bars if b != snapshot.bars[3])}
        )
    elif mutation == "duplicate":
        snapshot = snapshot.model_copy(update={"bars": (*snapshot.bars, bar)})
    else:
        snapshot = changed(snapshot, bar, vwap=None)
    target = tmp_path / "unpublished"
    if existing:
        target.mkdir()
    with pytest.raises(ValueError):
        export_bundle(snapshot, target)
    assert not target.exists() or list(target.iterdir()) == []


def test_same_session_intraday_order_cannot_choose_daily_close(tmp_path):
    source = archived_snapshot(periods=1, use_exchange_calendar=True)
    first = source.bars[0]
    early = first.model_copy(update={"closed_at": first.opened_at + timedelta(minutes=1)})
    late = first.model_copy(
        update={"opened_at": first.closed_at - timedelta(minutes=1), "evidence_id": "late"}
    )
    record = source.evidence[0]
    records = (
        record.model_copy(update={"available_at": early.closed_at, "observed_at": early.closed_at}),
        record.model_copy(update={"evidence_id": "late"}),
    )
    for i, bars in enumerate(((early, late), (late, early))):
        snapshot = MarketSnapshot(context=source.context, bars=bars, evidence=records)
        with pytest.raises(ValueError, match="XNYS|duplicate"):
            export_bundle(snapshot, tmp_path / str(i))
        assert not (tmp_path / str(i)).exists()


def test_valid_input_order_preserves_files_and_original_source_metadata(tmp_path):
    snapshot = archived_snapshot(periods=8, use_exchange_calendar=True)
    reverse = snapshot.model_copy(update={"bars": tuple(reversed(snapshot.bars))})
    first = export_bundle(snapshot, tmp_path / "first")
    second = export_bundle(reverse, tmp_path / "second")
    assert first["files"] == second["files"]
    assert first["instrument_map"] == second["instrument_map"]
    assert first["snapshot"] == snapshot.model_dump(mode="json")
    assert second["snapshot"] == reverse.model_dump(mode="json")
    for identity, symbol in first["instrument_map"].items():
        actual = np.fromfile(
            tmp_path / "first" / "features" / symbol / "close.day.bin", dtype="<f4"
        )
        expected = [float(b.close) for b in snapshot.bars if b.instrument.instrument_id == identity]
        np.testing.assert_array_equal(actual, np.array([0, *expected], dtype="<f4"))


def test_only_supplied_instrument_endpoints_bound_required_coverage(tmp_path):
    snapshot = archived_snapshot(periods=5, use_exchange_calendar=True)
    identity = snapshot.bars[0].instrument.instrument_id
    midpoint = sorted({b.session for b in snapshot.bars})[2]
    bars = tuple(
        b for b in snapshot.bars if b.instrument.instrument_id != identity or b.session == midpoint
    )
    snapshot = snapshot.model_copy(update={"bars": bars})
    manifest = export_bundle(snapshot, tmp_path / "bundle")
    symbol = manifest["instrument_map"][identity]
    actual = np.fromfile(tmp_path / "bundle" / "features" / symbol / "close.day.bin", dtype="<f4")
    assert len(actual) == 2 and actual[0] == 2
    assert (
        f"{symbol}\t{midpoint}\t{midpoint}"
        in (tmp_path / "bundle" / "instruments/all.txt").read_text()
    )


@pytest.mark.parametrize("start,end", [("2024-03-08", "2024-03-11"), ("2024-11-27", "2024-11-29")])
def test_actual_dst_and_early_close_boundaries_are_accepted(tmp_path, start, end):
    schedule = sessions(start, end, calendar="XNYS")
    source = archived_snapshot(periods=len(schedule), use_exchange_calendar=True)
    old = sorted({b.session for b in source.bars})
    mapping = dict(zip(old, schedule, strict=True))
    bars = tuple(
        b.model_copy(
            update={
                "session": mapping[b.session],
                "opened_at": schedule[mapping[b.session]][0],
                "closed_at": schedule[mapping[b.session]][1],
            }
        )
        for b in source.bars
    )
    closes = {b.evidence_id: b.closed_at for b in bars}
    first = next(iter(schedule.values()))[0]
    evidence = tuple(
        r.model_copy(
            update={
                "available_at": closes[r.evidence_id],
                "observed_at": closes[r.evidence_id],
                "requested_as_of": first,
            }
        )
        for r in source.evidence
    )
    snapshot = MarketSnapshot(
        context=source.context.model_copy(update={"as_of": max(closes.values())}),
        bars=bars,
        evidence=evidence,
    )
    assert export_bundle(snapshot, tmp_path / "bundle")["calendar"] == "XNYS"
    if start == "2024-03-08":
        assert schedule[start][0].hour == 14 and schedule[end][0].hour == 13
    else:
        assert schedule[end][1].hour == 18


@pytest.mark.parametrize("mutation", ["minute", "gap", "late_availability", "calendar"])
def test_published_malformed_snapshot_fails_without_rewriting_files(tmp_path, mutation):
    snapshot = archived_snapshot(periods=12, use_exchange_calendar=True)
    root = tmp_path / "published"
    manifest = export_bundle(snapshot, root)
    bar = snapshot.bars[3]
    if mutation == "minute":
        bad = changed(snapshot, bar, opened_at=bar.closed_at - timedelta(minutes=1))
    elif mutation == "gap":
        bad = snapshot.model_copy(update={"bars": tuple(b for b in snapshot.bars if b != bar)})
    elif mutation == "calendar":
        bad = changed(snapshot, bar, calendar="XNAS")
    else:
        bad = snapshot.model_copy(
            update={
                "evidence": tuple(
                    r.model_copy(update={"available_at": bar.closed_at + timedelta(seconds=1)})
                    if r.evidence_id == bar.evidence_id
                    else r
                    for r in snapshot.evidence
                )
            }
        )
    manifest.pop("bundle_hash")
    manifest["snapshot"] = bad.model_dump(mode="json")
    raw = canonical_bytes(manifest)
    (root / "manifest.json").write_bytes(raw)
    dates = sorted({b.session for b in snapshot.bars})
    names = tuple(manifest["instrument_map"].values())
    scenario = USQlibScenario(
        provider_uri=str(root),
        bundle_hash=digest(raw),
        universe=names,
        benchmark=names[0],
        price_basis="raw",
        train=(dates[0], dates[2]),
        valid=(dates[4], dates[6]),
        test=(dates[8], dates[-1]),
    )
    before = {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError):
        verify_bundle(scenario)
    after = {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }
    assert before == after
    assert json.loads((root / "manifest.json").read_bytes())["snapshot"] == bad.model_dump(
        mode="json"
    )


def test_rehashed_calendar_cannot_change_declared_snapshot_sessions(tmp_path):
    snapshot = archived_snapshot(periods=12, use_exchange_calendar=True)
    root = tmp_path / "published"
    manifest = export_bundle(snapshot, root)
    manifest.pop("bundle_hash")
    calendar = root / "calendars/day.txt"
    lines = calendar.read_text().splitlines()
    lines[1] = "2020-01-04"  # A Saturday, even with an internally consistent file hash.
    calendar.write_text("\n".join(lines) + "\n")
    manifest["files"]["calendars/day.txt"] = digest(calendar.read_bytes())
    raw = canonical_bytes(manifest)
    (root / "manifest.json").write_bytes(raw)
    dates = sorted({b.session for b in snapshot.bars})
    names = tuple(manifest["instrument_map"].values())
    scenario = USQlibScenario(
        provider_uri=str(root),
        bundle_hash=digest(raw),
        universe=names,
        benchmark=names[0],
        price_basis="raw",
        train=(dates[0], dates[2]),
        valid=(dates[4], dates[6]),
        test=(dates[8], dates[-1]),
    )
    before = calendar.read_bytes(), (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="calendar differs"):
        verify_bundle(scenario)
    assert before == (calendar.read_bytes(), (root / "manifest.json").read_bytes())
