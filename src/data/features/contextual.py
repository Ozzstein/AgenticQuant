"""Point-in-time scalar features from immutable evidence, including derivatives.

Bindings explicitly relate a target spot/security identity to an evidence source.
They do not grant permission to trade the source derivative instrument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import RunContext, VersionedContract


class NumericEvidenceBinding(VersionedContract):
    name: str
    target_instrument_id: str
    source_instrument_id: str | None
    kind: str
    value_path: tuple[str | int, ...]
    observation_date_path: tuple[str | int, ...] = ()
    filters: tuple[tuple[str, str], ...] = ()
    source_market: Literal["instrument", "global", "perpetual"] = "instrument"
    max_age_seconds: int = Field(gt=0)


@dataclass(frozen=True)
class EvidenceFields:
    values: pd.DataFrame
    available_at: pd.Series
    source_ids: pd.Series
    coverage: tuple[tuple[str, str, str, str], ...]
    snapshot_hash: str


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def materialize_evidence_fields(store: EvidenceStore, index: pd.MultiIndex, context: RunContext,
                                bindings: tuple[NumericEvidenceBinding, ...], *,
                                allowed_evidence_ids: frozenset[str] | None = None) -> EvidenceFields:
    if index.names != ["datetime", "instrument"] or not index.is_unique:
        raise ValueError("unique datetime/instrument identity required")
    dates = index.get_level_values("datetime")
    if not isinstance(dates, pd.DatetimeIndex) or dates.tz is None or (dates > context.as_of).any():
        raise ValueError("field dates must be timezone-aware and within requested cutoff")
    if not bindings or len({(b.name, b.target_instrument_id) for b in bindings}) != len(bindings):
        raise ValueError("unique explicit target/name bindings required")
    values = pd.DataFrame(np.nan, index=index, columns=sorted({b.name for b in bindings}))
    availability = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    sources = pd.Series([() for _ in index], index=index, dtype=object)
    coverage, used = [], {}
    complete = pd.Series(True, index=index)
    for binding in bindings:
        if binding.source_market == "global" and binding.source_instrument_id is not None:
            raise ValueError("global binding cannot invent an instrument identity")
        if binding.source_market != "global" and not binding.source_instrument_id:
            raise ValueError("explicit source instrument required")
        for date, instrument_id in index:
            if instrument_id != binding.target_instrument_id:
                continue
            key = (date, instrument_id)
            query = store.query(as_of=date.to_pydatetime(), kind=binding.kind,
                                instrument_id=binding.source_instrument_id, current_revisions_only=False)
            pinned = [item for item in query.records if allowed_evidence_ids is None or item.record.evidence_id in allowed_evidence_ids]
            superseded = {item.supersedes_id for item in pinned if item.supersedes_id}
            candidates = []
            for envelope in pinned:
                record = envelope.record
                if record.evidence_id in superseded:
                    continue
                payload = store.normalized(record.evidence_id, as_of=date.to_pydatetime())
                if any(str(payload.get(field)) != value for field, value in binding.filters):
                    continue
                if binding.source_market == "global" and record.instrument_ids:
                    continue
                if binding.source_market == "perpetual" and payload.get("market_type") != "perpetual":
                    raise ValueError("derivative binding requires declared perpetual provenance")
                try:
                    observed = pd.Timestamp(_at(payload, binding.observation_date_path)) if binding.observation_date_path else pd.Timestamp(record.observed_at or record.available_at)
                    # A date-only fiscal/observation label is an event date, never
                    # a publication/availability assertion.
                    if observed.tzinfo is None:
                        observed = observed.tz_localize("UTC")
                    number = float(_at(payload, binding.value_path))
                except (KeyError, IndexError, TypeError, ValueError):
                    raise ValueError(f"malformed numeric evidence for {binding.name}: {record.evidence_id}") from None
                if not np.isfinite(number):
                    raise ValueError(f"nonfinite numeric evidence: {record.evidence_id}")
                if observed > date or (date-observed).total_seconds() > binding.max_age_seconds:
                    continue
                candidates.append((observed, number, record))
            if not candidates:
                complete.loc[key] = False
                coverage.append((date.isoformat(), instrument_id, binding.name, "missing_or_ineligible"))
                continue
            latest = max(c[0] for c in candidates)
            selected = [c for c in candidates if c[0] == latest]
            if len({c[1] for c in selected}) > 1:
                complete.loc[key] = False
                coverage.append((date.isoformat(), instrument_id, binding.name, "conflicting"))
                continue
            values.loc[key, binding.name] = selected[0][1]
            ready = max(max(c[2].available_at, c[2].requested_as_of or c[2].available_at) for c in selected)
            availability.loc[key] = max(availability.loc[key], ready) if pd.notna(availability.loc[key]) else ready
            selected_ids = tuple(c[2].evidence_id for c in selected)
            sources.at[key] = tuple(sorted(set(sources.at[key]+selected_ids)))
            used.update({c[2].evidence_id: c[2].content_hash() for c in selected})
    # Each identity must have every requested field explicitly bound; missing
    # scope does not become an imputable successful feature observation.
    complete &= values.notna().all(axis=1)
    availability = availability.where(complete)
    identity = digest(canonical_bytes({"bindings": [b.model_dump(mode="json") for b in bindings],
                                       "records": used, "values": values.to_json(orient="split", date_format="iso")}))
    return EvidenceFields(values, availability, sources, tuple(coverage), identity)
