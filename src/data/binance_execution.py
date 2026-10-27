"""Frozen EvidenceStore authority for the canonical native Binance factory."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from src.data.adapters.base import ProviderError
from src.data.binance_reference import parse_binance_notional_reference
from src.data.evidence.store import EvidenceEnvelope, EvidenceStore, canonical_bytes
from src.domain.common import Instrument, RunContext, VersionedContract, _utc

if TYPE_CHECKING:
    from src.domain.portfolio import AccountSpec
    from src.execution.spot_venue import NativeSpotMarket
    from src.execution.venue import VenueExecution, VenueNotionalReference, VenueRules


def _fail(message):
    raise ProviderError("binance-evidence", message)


class BinanceReferenceBinding(VersionedContract):
    """Exact source envelopes, never a caller-supplied normalized price/proof."""

    instrument: Instrument
    average_minutes: int = Field(strict=True, ge=0)
    network: Literal["mainnet", "testnet"]
    reference: EvidenceEnvelope
    average: EvidenceEnvelope | None = None
    last_trade: EvidenceEnvelope | None = None

    @model_validator(mode="after")
    def independent_sources(self):
        sources = self.sources()
        if (
            len({source.record.evidence_id for source in sources}) != len(sources)
            or (
                self.average is not None
                and (self.last_trade is not None or self.average_minutes == 0)
            )
            or (self.last_trade is not None and self.average_minutes != 0)
        ):
            raise ValueError("distinct native sources and exact fallback window required")
        return self

    def sources(self) -> tuple[EvidenceEnvelope, ...]:
        return tuple(
            value for value in (self.reference, self.average, self.last_trade) if value is not None
        )


def _verify_binding(store, binding, context):
    for source in binding.sources():
        if store.get(source.record.evidence_id, as_of=context.as_of) != source:
            _fail("frozen_reference_envelope_differs")
    value = parse_binance_notional_reference(
        store,
        binding.instrument,
        context,
        reference_evidence_id=binding.reference.record.evidence_id,
        average_minutes=binding.average_minutes,
        average_evidence_id=binding.average.record.evidence_id if binding.average else None,
        last_trade_evidence_id=binding.last_trade.record.evidence_id
        if binding.last_trade
        else None,
        network=binding.network,
    )
    if value.kind == "venue-reference" and len(binding.sources()) != 1:
        _fail("unused_native_fallback_sources")
    return value


def freeze_binance_reference(
    store: EvidenceStore,
    instrument: Instrument,
    context: RunContext,
    *,
    reference_evidence_id: str,
    average_minutes: int,
    network: Literal["mainnet", "testnet"],
    average_evidence_id: str | None = None,
    last_trade_evidence_id: str | None = None,
) -> BinanceReferenceBinding:
    """Freeze exact stored envelopes after actual parser certification."""
    binding = BinanceReferenceBinding(
        instrument=instrument,
        average_minutes=average_minutes,
        network=network,
        reference=store.get(reference_evidence_id, as_of=context.as_of),
        average=store.get(average_evidence_id, as_of=context.as_of)
        if average_evidence_id
        else None,
        last_trade=store.get(last_trade_evidence_id, as_of=context.as_of)
        if last_trade_evidence_id
        else None,
    )
    _verify_binding(store, binding, context)
    return binding


@dataclass(frozen=True)
class BinanceEvidenceReferenceResolver:
    """Pure frozen callable; new refreshes require an explicit new binding set.

    Each call rechecks the exact stored envelopes and original bytes. There is
    no source discovery, network operation, clock default or cached success.
    """

    store: EvidenceStore
    context: RunContext
    bindings: tuple[BinanceReferenceBinding, ...]

    def __post_init__(self):
        context = RunContext.model_validate(self.context.model_dump())
        bindings = tuple(
            BinanceReferenceBinding.model_validate(value.model_dump()) for value in self.bindings
        )
        network = {"paper": "testnet", "live": "mainnet"}.get(context.mode)
        keys = [(value.instrument.instrument_id, value.average_minutes) for value in bindings]
        identities = {}
        for value in bindings:
            previous = identities.setdefault(value.instrument.instrument_id, value.instrument)
            if previous != value.instrument:
                _fail("frozen_canonical_instrument_identity_differs")
        if (
            network is None
            or not context.account_id
            or not bindings
            or len(set(keys)) != len(keys)
            or any(value.network != network for value in bindings)
        ):
            _fail("unique_operational_account_network_bindings_required")
        object.__setattr__(self, "context", context)
        object.__setattr__(
            self,
            "bindings",
            tuple(
                sorted(
                    bindings,
                    key=lambda value: (value.instrument.instrument_id, value.average_minutes),
                )
            ),
        )
        for binding in bindings:
            _verify_binding(self.store, binding, context)

    def __call__(
        self, instrument: Instrument, cutoff: datetime, average_minutes: int
    ) -> VenueNotionalReference:
        from src.execution.ledger import LedgerError

        try:
            cutoff = _utc(cutoff)
            if (
                cutoff < self.context.as_of
                or type(average_minutes) is not int
                or average_minutes < 0
            ):
                _fail("reference_cutoff_or_window_precedes_frozen_admission")
            matches = [
                value
                for value in self.bindings
                if value.instrument == instrument and value.average_minutes == average_minutes
            ]
            if len(matches) != 1:
                _fail("exact_frozen_instrument_window_missing")
            return _verify_binding(
                self.store, matches[0], self.context.model_copy(update={"as_of": cutoff})
            )
        except (
            ProviderError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            OSError,
            sqlite3.Error,
        ) as exc:
            # Native preflight catches LedgerError before any submit attempt.
            # A data denial must not become an ambiguous transmitted order.
            code = exc.code if isinstance(exc, ProviderError) else "invalid_frozen_source"
            raise LedgerError("certified Binance reference denied: " + code) from None


def create_evidence_backed_binance_execution(
    *,
    evidence_store: EvidenceStore,
    context: RunContext,
    reference_bindings: tuple[BinanceReferenceBinding, ...],
    ledger_path: str | Path,
    account: AccountSpec,
    broker_account_id,
    api_key,
    api_secret,
    markets: tuple[NativeSpotMarket, ...],
    rules: dict[str, VenueRules],
    fee_valuation=None,
    fill_policy=None,
    risk_policy=None,
    allow_live=False,
    max_balance_age_seconds=30,
    client=None,
    clock=None,
    operation_timeout_seconds=30,
    max_requests_per_operation=128,
    maximum_response_bytes=8 * 1024 * 1024,
) -> VenueExecution:
    """Bind the actual execution factory to certified immutable source bytes.

    No network call occurs at construction. Caller owns all returned adapter
    transports and the ledger. Noncash-fee valuation remains execution-owned;
    this factory never invents it. The existing explicit live gate is retained.
    """
    from src.execution.broker import create_typed_binance_execution

    if (
        context.account_id != account.account_id
        or context.mode != account.mode
        or context.as_of < account.opened_at
    ):
        _fail("reference_context_differs_from_actual_account")
    resolver = BinanceEvidenceReferenceResolver(evidence_store, context, reference_bindings)
    expected = {value.instrument.instrument_id: value.instrument for value in markets}
    actual = {value.instrument.instrument_id: value.instrument for value in resolver.bindings}
    if expected != actual:
        _fail("reference_bindings_differ_from_exact_execution_instruments")
    execution = create_typed_binance_execution(
        ledger_path=ledger_path,
        account=account,
        broker_account_id=broker_account_id,
        api_key=api_key,
        api_secret=api_secret,
        markets=markets,
        rules=rules,
        notional_reference=resolver,
        fee_valuation=fee_valuation,
        fill_policy=fill_policy,
        risk_policy=risk_policy,
        allow_live=allow_live,
        max_balance_age_seconds=max_balance_age_seconds,
        client=client,
        clock=clock,
        operation_timeout_seconds=operation_timeout_seconds,
        max_requests_per_operation=max_requests_per_operation,
        maximum_response_bytes=maximum_response_bytes,
    )
    try:
        manifest = dict(
            implementation="binance-evidence-reference-binding-1",
            context=resolver.context.model_dump(mode="json"),
            bindings=[value.model_dump(mode="json") for value in resolver.bindings],
        )
        with execution.ledger.atomic():
            execution.ledger.record_event(
                account.account_id,
                "binance-reference-binding:" + context.content_hash(),
                canonical_bytes(manifest).decode("utf-8"),
            )
        return execution
    except Exception:
        for adapter in execution.adapters.values():
            adapter.close()
        execution.ledger.close()
        raise
