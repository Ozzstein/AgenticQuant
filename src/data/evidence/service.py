"""Shared per-run market snapshots and source bundles for the analyst desk."""
from __future__ import annotations

from datetime import datetime

from src.data.evidence.store import EvidenceStore
from src.domain.common import Instrument, RunContext
from src.domain.market import CoverageGap, MarketBar, MarketSnapshot

ROLE_KINDS = {
    "fundamental": ("income_statement", "balance_sheet", "cash_flow", "ratios", "filing", "token_context"),
    "sentiment": ("news", "analyst_estimates", "transcript", "crypto_global"),
    "technical": ("technical", "funding", "open_interest"),
    "risk": ("earnings", "filing", "news", "liquidity"),
    "strategist": ("transcript", "news", "technical", "filing", "token_context"),
}


class EvidenceService:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def source_bundle(self, role: str, instrument: Instrument, context: RunContext, *,
                      query: str = "", max_chars: int = 6000) -> dict:
        if role not in ROLE_KINDS:
            raise ValueError("unknown analyst role")
        ids, coverage = [], {}
        for kind in ROLE_KINDS[role]:
            result = self.store.query(as_of=context.as_of, kind=kind,
                                      instrument_id=instrument.instrument_id, current_revisions_only=True)
            coverage[kind] = result
            ids.extend(e.record.evidence_id for e in result.records if e.document_hash)
        return {"context": context, "instrument": instrument, "coverage": coverage,
                "passages": self.store.retrieve(ids, as_of=context.as_of, query=query, max_chars=max_chars)}

    def market_snapshot(self, instruments: tuple[Instrument, ...], context: RunContext, *,
                        start: datetime, expected_closes: dict[str, tuple[datetime, ...]] | None = None,
                        price_basis: str = "raw", market_type: str | None = None) -> MarketSnapshot:
        bars, evidence, gaps = [], {}, []
        for instrument in instruments:
            result = self.store.query(as_of=context.as_of, kind="bar", instrument_id=instrument.instrument_id,
                                      current_revisions_only=True)
            grouped = {}
            for item in result.records:
                payload = self.store.normalized(item.record.evidence_id, as_of=context.as_of)
                bar = MarketBar(evidence_id=item.record.evidence_id, **payload)
                if bar.closed_at < start or bar.closed_at > context.as_of or bar.price_basis != price_basis:
                    continue
                if market_type and bar.market_type != market_type:
                    continue
                grouped.setdefault((bar.closed_at, bar.market_type), []).append((bar, item.record))
            observed = set()
            for (closed, _), versions in grouped.items():
                values = {b.model_dump_json(exclude={"evidence_id"}) for b, _ in versions}
                if len(values) > 1:
                    gaps.append(CoverageGap(instrument_id=instrument.instrument_id, capability="bar", reason="conflict",
                                            detail="multiple eligible source prices require explicit resolution", expected_at=closed))
                    continue
                bar, record = versions[-1]
                bars.append(bar)
                evidence[record.evidence_id] = record
                observed.add(closed)
            if not observed:
                gaps.append(CoverageGap(instrument_id=instrument.instrument_id, capability="bar", reason="missing",
                                        detail=f"eligible bars unavailable ({result.status}); exclusions={len(result.excluded)}"))
            for close in (expected_closes or {}).get(instrument.instrument_id, ()):
                if start <= close <= context.as_of and close not in observed:
                    gaps.append(CoverageGap(instrument_id=instrument.instrument_id, capability="bar", reason="missing",
                                            detail="expected session has no eligible observation", expected_at=close))
        return MarketSnapshot(context=context, bars=tuple(sorted(bars, key=lambda b: (b.closed_at, b.instrument.instrument_id))),
                              evidence=tuple(evidence.values()), gaps=tuple(gaps))
