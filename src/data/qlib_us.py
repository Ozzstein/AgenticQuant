"""Explicit, immutable US Qlib bundle export and Alpha158 research scenario.

No China bundle or current SP500 membership is substituted. The caller supplies
eligible historical instruments/bars; provenance and coverage stay in manifest.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.calendars import sessions
from src.data.evidence.store import canonical_bytes, digest
from src.domain.common import NonEmptyString, Sha256, VersionedContract
from src.domain.market import MarketSnapshot


class USQlibScenario(VersionedContract):
    provider_uri: NonEmptyString
    bundle_hash: Sha256
    universe: tuple[NonEmptyString, ...]
    benchmark: NonEmptyString
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]
    price_basis: NonEmptyString
    calendar: str = "XNYS"
    region: str = "us"
    label_horizon: int = 1
    equity_cost_bps: float = 10.0

    def validate_splits(self):
        train, valid, test = [tuple(pd.Timestamp(d) for d in interval) for interval in (self.train, self.valid, self.test)]
        if not train[0] <= train[1] < valid[0] <= valid[1] < test[0] <= test[1]:
            raise ValueError("Qlib train/valid/test segments must be strictly ordered and disjoint")
        if self.region != "us" or self.calendar != "XNYS" or self.label_horizon < 1:
            raise ValueError("explicit US scenario and positive label horizon required")


def _daily_inputs(snapshot: MarketSnapshot, *, require_vwap: bool):
    """Validate the declared daily grid before writing or admitting a bundle.

    Each instrument's supplied endpoints bound the checked grid. No listing,
    delisting or index-membership coverage is inferred outside those endpoints.
    """
    if snapshot.gaps:
        raise ValueError("cannot export a bundle with unresolved coverage gaps")
    if not snapshot.bars or any(b.market_type != "equity" for b in snapshot.bars):
        raise ValueError("US Qlib bundle requires equity/ETF bars")
    bases = {bar.price_basis for bar in snapshot.bars}
    if len(bases) != 1:
        raise ValueError("Qlib bundle requires one explicit price convention")
    if require_vwap and any(b.vwap is None for b in snapshot.bars):
        raise ValueError("Alpha158 requires actual VWAP; OHLC typical-price substitution is forbidden")
    first, last = min(b.session for b in snapshot.bars), max(b.session for b in snapshot.bars)
    schedule = sessions(first, last, calendar="XNYS")
    calendar = list(schedule)
    offsets = {day: i for i, day in enumerate(calendar)}
    records = {e.evidence_id: e for e in snapshot.evidence}
    grouped, seen = {}, set()
    for bar in snapshot.bars:
        key = (bar.instrument.instrument_id, bar.session)
        if key in seen:
            raise ValueError("duplicate daily instrument/session in Qlib snapshot")
        seen.add(key)
        if bar.calendar != "XNYS" or bar.session not in schedule:
            raise ValueError("Qlib daily bar requires an actual XNYS session/calendar")
        if (bar.opened_at, bar.closed_at) != schedule[bar.session]:
            raise ValueError("Qlib daily bar must span the actual XNYS session open/close")
        records[bar.evidence_id].require_available(bar.closed_at)
        grouped.setdefault(bar.instrument.instrument_id, []).append(bar)
    for identity, bars in grouped.items():
        start, stop = min(offsets[b.session] for b in bars), max(offsets[b.session] for b in bars)
        observed = {b.session for b in bars}
        missing = [day for day in calendar[start:stop+1] if day not in observed]
        if missing:
            raise ValueError(f"unresolved internal daily session gap for {identity}: {missing[0]}")
    return calendar, grouped, bases


def export_bundle(snapshot: MarketSnapshot, path: str | Path, *, require_vwap: bool = True) -> dict:
    calendar, grouped, bases = _daily_inputs(snapshot, require_vwap=require_vwap)
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise ValueError("bundle target must be empty; published bundles are immutable")
    root.mkdir(parents=True, exist_ok=True)
    offsets = {day: i for i, day in enumerate(calendar)}
    for directory in ("calendars", "instruments", "features"):
        (root / directory).mkdir(exist_ok=True)
    (root / "calendars/day.txt").write_text("\n".join(calendar) + "\n")
    instrument_map, instrument_lines = {}, []
    fields = ("open", "high", "low", "close", "volume", "vwap")
    for identity, bars in sorted(grouped.items()):
        symbol = "us_" + digest(identity.encode())[:20]
        instrument_map[identity] = symbol
        folder = root / "features" / symbol
        folder.mkdir()
        start_idx, stop_idx = min(offsets[b.session] for b in bars), max(offsets[b.session] for b in bars)
        observed = {bar.session: bar for bar in bars}
        for field in fields:
            values = [float(getattr(observed[day], field)) if day in observed and getattr(observed[day], field) is not None else np.nan
                      for day in calendar[start_idx:stop_idx+1]]
            np.array([start_idx, *values], dtype="<f4").tofile(folder / f"{field}.day.bin")
        # 1 is correct ONLY for the explicitly raw, unadjusted bundle convention;
        # it does not assert that corporate-action adjusted prices were supplied.
        if bases == {"raw"}:
            np.array([start_idx, *([1.] * (stop_idx-start_idx+1))], dtype="<f4").tofile(folder / "factor.day.bin")
        instrument_lines.append(f"{symbol}\t{calendar[start_idx]}\t{calendar[stop_idx]}")
    (root / "instruments/all.txt").write_text("\n".join(instrument_lines) + "\n")
    files = {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}
    manifest = {"schema_version": "1.0", "region": "us", "calendar": "XNYS", "price_basis": next(iter(bases)),
                "snapshot": snapshot.model_dump(mode="json"), "instrument_map": instrument_map, "files": files,
                "qlib_version": importlib.metadata.version("pyqlib"), "coverage": "caller-supplied-historical-universe"}
    raw = canonical_bytes(manifest)
    (root / "manifest.json").write_bytes(raw)
    return {"bundle_hash": digest(raw), **manifest}


def verify_bundle(scenario: USQlibScenario) -> dict:
    scenario.validate_splits()
    root = Path(scenario.provider_uri)
    raw = (root / "manifest.json").read_bytes()
    if digest(raw) != scenario.bundle_hash:
        raise ValueError("Qlib bundle manifest hash mismatch")
    manifest = json.loads(raw)
    snapshot = MarketSnapshot.model_validate(manifest["snapshot"])
    # Older published files may have valid hashes but an invalid daily grid.
    # Verification never repairs/re-exports them in place.
    expected_calendar, _, _ = _daily_inputs(snapshot, require_vwap=False)
    if (manifest["region"] != "us" or manifest.get("calendar") != "XNYS"
            or manifest["price_basis"] != scenario.price_basis
            or {b.price_basis for b in snapshot.bars} != {scenario.price_basis}):
        raise ValueError("Qlib bundle market/convention mismatch")
    for filename, expected in manifest["files"].items():
        file_path = root / filename
        if not file_path.resolve().is_relative_to(root.resolve()) or digest(file_path.read_bytes()) != expected:
            raise ValueError("Qlib bundle file integrity mismatch")
    ids = set(manifest["instrument_map"].values())
    if not set(scenario.universe) <= ids or scenario.benchmark not in ids:
        raise ValueError("Qlib scenario instrument/benchmark coverage missing")
    calendar_lines = (root / "calendars/day.txt").read_text().splitlines()
    if calendar_lines != expected_calendar:
        raise ValueError("Qlib calendar differs from actual snapshot XNYS sessions")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_lines))
    for interval in (scenario.train, scenario.valid, scenario.test):
        if pd.Timestamp(interval[0]) < calendar[0] or pd.Timestamp(interval[1]) > calendar[-1]:
            raise ValueError("scenario dates outside bundle coverage")
    # Purge by actual exchange-session count, not calendar days or row count.
    for left, right in ((scenario.train, scenario.valid), (scenario.valid, scenario.test)):
        if sum((calendar > pd.Timestamp(left[1])) & (calendar < pd.Timestamp(right[0]))) < scenario.label_horizon:
            raise ValueError("Qlib split lacks label-horizon purge")
    return manifest


def alpha158_dataset(scenario: USQlibScenario):
    verify_bundle(scenario)
    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    qlib.init(provider_uri=scenario.provider_uri, region="us", expression_cache=None, dataset_cache=None, kernels=1)
    handler = Alpha158(instruments=list(scenario.universe), start_time=scenario.train[0], end_time=scenario.test[1],
                       fit_start_time=scenario.train[0], fit_end_time=scenario.train[1],
                       infer_processors=[{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                                         {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
                       learn_processors=[{"class": "DropnaLabel"}],
                       label=([f"Ref($close, -{scenario.label_horizon})/$close-1"], ["LABEL0"]))
    return DatasetH(handler=handler, segments={"train": scenario.train, "valid": scenario.valid, "test": scenario.test})
