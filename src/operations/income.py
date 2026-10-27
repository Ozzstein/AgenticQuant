"""Closed source verification for operating income, distinct from ledger arithmetic.

Native Alpaca process/creation-date queries currently prove explicit gaps only.
No normalized assertion, supplied callback, or hash-shaped authority can turn
those queries into complete economic coverage. A positive source needs its own
documented wire parser and independently established scope before registration.
"""

from __future__ import annotations

from src.domain.common import _utc
from src.domain.portfolio import ActionCoverage
from src.research.evidence import BoundArtifact, json_pointer
from src.research.registry import RegistryError, canonical, content_hash

INCOME_KINDS = (
    "cash_dividends",
    "dividend_payments",
    "dividend_distributions",
    "action_coverage",
    "paper_dividend_policies",
)
POLICIES = ("posted-account", "source-confirmed-paper-distribution-1")


def _parsed_capture(store, evidence_id, *, as_of):
    envelope = store.get(evidence_id, as_of=as_of)
    if (envelope.record.provider, envelope.record.provider_version, envelope.kind) not in (
        ("operating", "operating-wire-2", "operating_dividend_terms"),
        ("operating", "operating-wire-2", "operating_dividend_payments"),
    ):
        raise RegistryError("income source has no supported independent native parser")
    try:
        from src.data.adapters.dividends import verify_operating_dividend_capture
    except ImportError as exc:
        raise RegistryError("native income source parser is unavailable") from exc
    try:
        parsed = verify_operating_dividend_capture(store, evidence_id, as_of=as_of)
    except (ValueError, KeyError, TypeError) as exc:
        raise RegistryError("income native wire/source verification failed: " + str(exc)) from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(parsed.get(k), list) for k in INCOME_KINDS
    ):
        raise RegistryError("income parser returned an invalid canonical array contract")
    # This parser's documented native APIs cannot establish positive economics
    # or complete coverage. A future adapter must earn a separate closed branch.
    if any(parsed[k] for k in INCOME_KINDS if k != "action_coverage"):
        raise RegistryError("native Alpaca observations cannot certify positive income economics")
    if parsed.get("income_authorities") or any(
        ActionCoverage.model_validate(row).completeness == "complete"
        for row in parsed["action_coverage"]
    ):
        raise RegistryError("native Alpaca query scope cannot certify complete income coverage")
    return envelope, parsed


def require_coverage_union(
    rows, *, instrument_id, account_id, started_at, ended_at, as_of, scope, event_type
):
    """Require an uninterrupted union for one exact capability and identity.

    Call only after source parsing. This interval operation does not authenticate
    a row's claims; raw canonical ActionCoverage is never source authority.
    """
    start, end, cutoff = map(_utc, (started_at, ended_at, as_of))
    if end < start or end > cutoff:
        raise RegistryError("income coverage interval exceeds its actual cutoff")
    intervals = []
    for row in rows:
        if (
            instrument_id not in row.instrument_ids
            or event_type not in row.event_types
            or row.scope != scope
        ):
            continue
        if row.account_id not in (None, account_id) or (
            scope == "custodial-credit-ledger" and row.account_id != account_id
        ):
            continue
        if row.started_at > end or row.ended_at < start:
            continue
        # Conflicting uncertainty is not silently discarded in favour of a
        # convenient complete row. Source revision selection must precede this.
        if row.available_at > cutoff or row.completeness != "complete" or row.gaps:
            raise RegistryError("income source coverage is unavailable, partial, or unknown")
        row.require_complete(
            instrument_ids=(instrument_id,),
            started_at=row.started_at,
            ended_at=row.ended_at,
            as_of=cutoff,
            scope=scope,
            event_types=(event_type,),
            account_id=account_id,
        )
        intervals.append((row.started_at, row.ended_at))
    cursor, covered = start, False
    for left, right in sorted(intervals):
        if left > cursor:
            break
        cursor = max(cursor, right)
        covered = True
        if cursor >= end:
            return
    if not covered or cursor < end:
        raise RegistryError(
            f"missing complete {event_type} coverage for {instrument_id} over required account interval"
        )


def verify_income_provenance(
    registry,
    store,
    inputs,
    references,
    *,
    income_policy,
    required_started_at,
    required_ended_at,
    income_authorities=(),
):
    if income_policy not in POLICIES or income_policy != getattr(inputs, "income_policy", None):
        raise RegistryError("explicit matching paper income policy required")
    if required_started_at is None or required_ended_at is None:
        raise RegistryError("authoritative prior account anchor and income cutoff required")
    start, end = map(_utc, (required_started_at, required_ended_at))
    if start < inputs.account.opened_at or end != inputs.context.as_of or start > end:
        raise RegistryError("income window differs from account lifetime or actual cycle cutoff")
    checked, parsed = {}, {}
    for kind in INCOME_KINDS:
        name = kind + "_id"
        ref = references.get(name)
        if ref is None:
            raise RegistryError("missing source-bound " + kind)
        artifact = BoundArtifact.model_validate_json(registry.blob(ref))
        expected = [item.model_dump(mode="json") for item in getattr(inputs, kind)]
        if (
            artifact.kind != kind
            or artifact.inputs_hash != inputs.content_hash()
            or artifact.value() != expected
        ):
            raise RegistryError("income artifact differs from frozen economic input: " + kind)
        actual = []
        for source in artifact.sources:
            if source.available_at > end or source.pointer != "/" + kind:
                raise RegistryError(
                    "income source requires its exact canonical array and eligible clock"
                )
            if source.evidence_id not in parsed:
                parsed[source.evidence_id] = _parsed_capture(store, source.evidence_id, as_of=end)
            envelope, wire = parsed[source.evidence_id]
            if envelope.record.available_at != source.available_at:
                raise RegistryError("income source binding differs from actual archive eligibility")
            actual.extend(wire[kind])
        if actual != expected:
            raise RegistryError("parsed native income arrays do not substantiate frozen economics")
        checked[name] = ref
    # This allowlist currently has no positive authority roles. Preserve the
    # supplied documents as evidence without treating them as executable claims.
    for source in income_authorities:
        envelope = store.get(source.evidence_id, as_of=end)
        if source.available_at != envelope.record.available_at or source.available_at > end:
            raise RegistryError("income authority availability differs from source")
        if content_hash(store.raw(source.evidence_id)) != envelope.record.raw_hash:
            raise RegistryError("income authority raw bytes differ")
        json_pointer(store.normalized(source.evidence_id, as_of=end), source.pointer)
        raise RegistryError(
            "positive income authority format is not supported by an independent parser"
        )
    coverage = tuple(
        ActionCoverage.model_validate(row) for row in expected_array(inputs, "action_coverage")
    )
    instruments = {bar.instrument.instrument_id: bar.instrument for bar in inputs.snapshot.bars}
    instruments.update({item.instrument.instrument_id: item.instrument for item in inputs.metadata})
    if not instruments:
        raise RegistryError("income verification requires explicit account instruments")
    for instrument in instruments.values():
        if instrument.asset_class == "crypto":
            raise RegistryError(
                "crypto income/protocol coverage requires a supported custodial or protocol authority"
            )
        require_coverage_union(
            coverage,
            instrument_id=instrument.instrument_id,
            account_id=inputs.account.account_id,
            started_at=start,
            ended_at=end,
            as_of=end,
            scope="effective-event-stream",
            event_type="cash_dividend_terms",
        )
        posted = income_policy == "posted-account"
        require_coverage_union(
            coverage,
            instrument_id=instrument.instrument_id,
            account_id=inputs.account.account_id,
            started_at=start,
            ended_at=end,
            as_of=end,
            scope="custodial-credit-ledger" if posted else "effective-event-stream",
            event_type="cash_dividend_payments" if posted else "cash_dividend_distributions",
        )
    checked["income_authorities_hash"] = content_hash(
        canonical([s.model_dump(mode="json") for s in income_authorities])
    )
    return checked


def expected_array(inputs, kind):
    return [item.model_dump(mode="json") for item in getattr(inputs, kind)]
