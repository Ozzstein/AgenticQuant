"""Closed native fact/audit contracts. No economic settlement API exists here.

Instantiating these values never authorizes booking. Public operations resolve
stored IDs through the fixed read-only source exporters before returning proof.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.domain.common import Instrument, NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.domain.portfolio import Fill

NativeVenue = Literal["alpaca", "binance", "kraken"]
AdmissionReason = Literal[
    "outside_expiry",
    "before_eligibility",
    "limit_breach",
    "quantity_excess",
    "immutable_terms_changed",
]


def _sorted_unique(values, *, key=None):
    keys = tuple(key(value) if key else value for value in values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("sorted unique canonical bindings required")


class RetainedSourceRef(VersionedContract):
    store_kind: Literal["ledger-event", "venue-report", "evidence-envelope"]
    account_id: NonEmptyString
    record_id: NonEmptyString
    intent_id: NonEmptyString | None = None
    payload_sha256: Sha256

    @model_validator(mode="after")
    def exact_key(self):
        if (self.store_kind == "venue-report") != (self.intent_id is not None):
            raise ValueError("only a venue-report reference requires its original intent key")
        return self


class NativeExecutionKey(VersionedContract):
    venue: NativeVenue
    network: Literal["mainnet", "alpaca-paper", "binance-testnet"]
    broker_account_id: NonEmptyString
    instrument: Instrument
    venue_order_id: NonEmptyString
    native_execution_id: NonEmptyString

    @model_validator(mode="after")
    def exact_network(self):
        if (self.network == "alpaca-paper" and self.venue != "alpaca") or (
            self.network == "binance-testnet" and self.venue != "binance"
        ):
            raise ValueError("native network does not belong to the execution venue")
        return self


class NativeExecutionAuthority(VersionedContract):
    implementation: Literal["retained-native-execution-1"] = "retained-native-execution-1"
    key: NativeExecutionKey
    account_id: NonEmptyString
    account_spec_hash: Sha256
    owner: Literal["direct", "lean"]
    execution_book_hash: Sha256
    outbox_identity_hash: Sha256
    intent_id: NonEmptyString
    intent_hash: Sha256
    fill: Fill
    original_report: RetainedSourceRef
    first_execution_receipt: RetainedSourceRef
    first_observed_at: UtcTimestamp
    first_complete_at: UtcTimestamp
    native_timestamp: NonEmptyString
    native_timestamp_unit: Literal["iso8601-utc", "unix-milliseconds", "unix-seconds-decimal"]
    timestamp_policy: Literal["exact-microsecond-1"] = "exact-microsecond-1"
    sources: tuple[tuple[NonEmptyString, RetainedSourceRef], ...]
    source_manifest_hash: Sha256

    @model_validator(mode="after")
    def immutable_fact(self):
        _sorted_unique(self.sources, key=lambda item: item[0])
        if (
            self.fill.account_id != self.account_id
            or self.fill.intent_id != self.intent_id
            or self.fill.fill_id != self.key.venue + ":" + self.key.native_execution_id
            or not self.fill.executed_at <= self.first_observed_at <= self.first_complete_at
            or self.original_report.store_kind != "venue-report"
            or self.original_report.intent_id != self.intent_id
            or any(
                reference.account_id != self.account_id
                for reference in (
                    self.original_report,
                    self.first_execution_receipt,
                    *(reference for _, reference in self.sources),
                )
            )
        ):
            raise ValueError("native execution authority identity/receipt clocks differ")
        return self


class AdmissionViolation(VersionedContract):
    case_id: Sha256
    authority_hash: Sha256
    intent_hash: Sha256
    reason_codes: tuple[AdmissionReason, ...] = Field(min_length=1)
    executed_at: UtcTimestamp
    admitted_eligible_at: UtcTimestamp
    admitted_expires_at: UtcTimestamp | None
    first_observed_at: UtcTimestamp
    compliance: Literal["FAIL"] = "FAIL"

    @model_validator(mode="after")
    def actual_breach(self):
        _sorted_unique(self.reason_codes)
        if (
            self.first_observed_at < self.executed_at
            or "outside_expiry" in self.reason_codes
            and (self.admitted_expires_at is None or self.executed_at < self.admitted_expires_at)
            or "before_eligibility" in self.reason_codes
            and self.executed_at >= self.admitted_eligible_at
        ):
            raise ValueError("admission violation contradicts original execution/receipt clocks")
        return self


class ExecutionDiscrepancy(VersionedContract):
    case_id: Sha256
    account_id: NonEmptyString
    key: NativeExecutionKey
    authority_hash: Sha256
    original_report: RetainedSourceRef
    first_detected_at: UtcTimestamp
    first_observed_at: UtcTimestamp
    reason_codes: tuple[AdmissionReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def original_identity(self):
        _sorted_unique(self.reason_codes)
        if (
            self.original_report.account_id != self.account_id
            or self.first_detected_at < self.first_observed_at
        ):
            raise ValueError("discrepancy account/detection clock differs from source")
        return self


class PrefixBinding(VersionedContract):
    count: int = Field(ge=0)
    last_sequence: int = Field(ge=0)
    sha256: Sha256


class LedgerCut(VersionedContract):
    account_id: NonEmptyString
    account_spec_hash: Sha256
    resource_state_hash: Sha256
    order_state_hash: Sha256
    journal_prefix: PrefixBinding
    native_source_prefix: PrefixBinding
    sleeve_event_prefix: PrefixBinding | None
    sleeve_allocation_hash: Sha256 | None
    publication_hash: Sha256
    outbox_hash: Sha256


class SettlementAssessment(VersionedContract):
    """Metadata-only eligibility assessment; no reachable settlement transition."""

    implementation: Literal["post-expiry-native-fact-1"] = "post-expiry-native-fact-1"
    account_id: NonEmptyString
    requested_case_ids: tuple[Sha256, ...] = Field(min_length=1)
    required_case_ids: tuple[Sha256, ...]
    authority_hashes: tuple[tuple[Sha256, Sha256], ...]
    closure_hash: Sha256
    basis: LedgerCut
    disposition: Literal["representable", "blocked"]
    blocking_codes: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def closed_membership(self):
        for values in (self.requested_case_ids, self.required_case_ids, self.blocking_codes):
            _sorted_unique(values)
        _sorted_unique(self.authority_hashes, key=lambda item: item[0])
        if self.basis.account_id != self.account_id or (self.disposition == "representable") != (
            not self.blocking_codes
        ):
            raise ValueError("assessment identity/disposition contradicts its actual blockers")
        return self


class ViolationBinding(VersionedContract):
    case_id: Sha256
    authority_hash: Sha256
    violation_hash: Sha256
    # Phase one deliberately has no settlement receipts or economic exception.
    settlement_id: None = None


class ExecutionComplianceAudit(VersionedContract):
    implementation: Literal["execution-compliance-1"] = "execution-compliance-1"
    account_id: NonEmptyString
    account_spec_hash: Sha256
    execution_binding_hash: Sha256
    authority_prefix_hash: Sha256
    lineage: Literal["historical-replay", "local-paper", "lean-paper", "native", "unverified"]
    status: Literal["PASS", "FAIL"]
    violations: tuple[ViolationBinding, ...]
    unresolved_sources: tuple[RetainedSourceRef, ...]
    blocking_codes: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def derived_failure(self):
        _sorted_unique(self.violations, key=lambda item: item.case_id)
        _sorted_unique(self.blocking_codes)
        _sorted_unique(self.unresolved_sources, key=lambda item: item.content_hash())
        failed = bool(
            self.violations
            or self.unresolved_sources
            or self.blocking_codes
            or self.lineage == "unverified"
        )
        if (self.status == "FAIL") != failed or any(
            value.account_id != self.account_id for value in self.unresolved_sources
        ):
            raise ValueError("audit status/account contradicts retained violation/source evidence")
        return self
