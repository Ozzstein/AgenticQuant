"""Immutable identities, execution context, and historical evidence boundaries.

Legacy models remain in ``src.utils.schemas`` while callers migrate explicitly.
An evidence availability timestamp is distinct from its observation and ingestion
times: a historical backfill may be ingested years after publication.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _utc(value: datetime) -> datetime:
    """Require an explicit timezone and normalize an instant to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp requires an explicit timezone")
    return value.astimezone(timezone.utc)


UtcTimestamp = Annotated[datetime, AfterValidator(_utc)]


class VersionedContract(BaseModel):
    """Closed, immutable contract with a reproducible serialized identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"

    def content_hash(self) -> str:
        """Hash the canonical JSON representation, including schema version."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class Instrument(VersionedContract):
    """Explicit instrument identity; display tickers alone are not identities.

    ``instrument_id`` is assigned by the security master and must survive symbol
    changes. ``venue`` identifies the listing/execution venue where applicable;
    its presence does not grant permission to trade there.
    """

    instrument_id: NonEmptyString
    symbol: NonEmptyString
    asset_class: Literal["equity", "etf", "crypto"]
    quote_currency: NonEmptyString
    venue: NonEmptyString | None = None


class RunContext(VersionedContract):
    """Explicit run identity and clock boundary, without ambient time defaults."""

    run_id: NonEmptyString
    mode: Literal["research", "backtest", "paper", "live"]
    as_of: UtcTimestamp
    config_hash: Sha256
    code_version: NonEmptyString
    account_id: NonEmptyString | None = None
    data_snapshot_ids: tuple[NonEmptyString, ...] = ()


class EvidenceRecord(VersionedContract):
    """Source metadata used to enforce historical availability.

    Raw content lives in a separate immutable store identified by ``raw_hash``.
    Unknown or inferred publication times remain distinguishable from documented
    availability. An adapter must establish provenance before marking a record
    usable; this contract does not independently verify a provider's claims.
    """

    evidence_id: NonEmptyString
    provider: NonEmptyString
    provider_version: NonEmptyString
    source_uri: NonEmptyString
    raw_hash: Sha256
    ingested_at: UtcTimestamp
    available_at: UtcTimestamp | None = None
    observed_at: UtcTimestamp | None = None
    requested_as_of: UtcTimestamp | None = None
    availability_basis: Literal["documented", "inferred", "unknown"] = "unknown"
    quality: Literal["usable", "unknown", "missing", "stale", "invalid"] = "unknown"
    instrument_ids: tuple[NonEmptyString, ...] = ()
    tool_name: NonEmptyString | None = None
    tool_version: NonEmptyString | None = None
    currency: NonEmptyString | None = None
    units: NonEmptyString | None = None
    adjustment_policy: NonEmptyString | None = None
    revision_id: NonEmptyString | None = None

    def require_available(self, as_of: datetime) -> None:
        """Reject evidence that cannot establish eligibility at ``as_of``.

        Documented historical availability permits later ingestion/backfill.
        Inferred or unknown availability never silently qualifies as point-in-time
        evidence. A caller may inspect such records for diagnostics separately.

        Args:
            as_of: Timezone-aware decision cutoff.

        Raises:
            ValueError: If the cutoff is naive or evidence is ineligible.
        """
        cutoff = _utc(as_of)
        if self.quality != "usable":
            raise ValueError(f"evidence quality is {self.quality}, not usable")
        if self.availability_basis != "documented" or self.available_at is None:
            raise ValueError("documented evidence availability is required")
        if self.available_at > cutoff:
            raise ValueError("evidence was unavailable at the decision cutoff")
        if self.requested_as_of is not None and self.requested_as_of > cutoff:
            raise ValueError("evidence was requested with a later historical cutoff")
