"""Read-only native income receipts with independently reconstructed coverage gaps.

Alpaca's process-date announcements and creation-time account activities do not
establish complete economic-event coverage, pre-ex ownership, or linked gross
payments/deductions. Preserve the actual records without inventing those proofs.
No declaration/payable date or average net amount becomes canonical cash income.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import model_validator

from src.data.adapters.base import ReadHTTP, store_observation
from src.data.adapters.operating import (
    VERSION,
    OperatingCapture,
    _fail,
    _operation,
    verify_operating_capture,
)
from src.data.evidence.store import EvidenceStore, canonical_bytes
from src.domain.common import (
    Instrument,
    NonEmptyString,
    RunContext,
    UtcTimestamp,
    VersionedContract,
)
from src.domain.portfolio import ActionCoverage

INCOME_KINDS = (
    "cash_dividends",
    "dividend_payments",
    "dividend_distributions",
    "action_coverage",
    "paper_dividend_policies",
)
DATA = "https://data.alpaca.markets"
PAPER = "https://paper-api.alpaca.markets"


class IncomeCaptureRequest(VersionedContract):
    operation: Literal["alpaca-process-date-terms-1", "alpaca-created-account-activities-1"]
    context: RunContext
    instrument: Instrument
    started_at: UtcTimestamp
    ended_at: UtcTimestamp
    account_id: NonEmptyString | None = None
    provider_account_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def scope(self):
        if self.context.mode not in ("paper", "research") or self.instrument.asset_class not in (
            "equity",
            "etf",
        ):
            raise ValueError("read-only equity/ETF income context required")
        if not self.started_at < self.ended_at <= self.context.as_of:
            raise ValueError("explicit nonfuture income observation window required")
        account = self.operation == "alpaca-created-account-activities-1"
        if account:
            if (
                not self.account_id
                or self.context.account_id != self.account_id
                or not self.provider_account_id
            ):
                raise ValueError("explicit matching canonical/provider account identities required")
            if str(UUID(self.provider_account_id)) != self.provider_account_id:
                raise ValueError("canonical native Alpaca account UUID required")
        elif self.account_id is not None or self.provider_account_id is not None:
            raise ValueError("instrument announcement scope cannot invent an account")
        return self


def _original_wire(
    capture, http, path, request, *, upstream, params=None, instrument=None, headers=None
):
    # OperatingCapture archives the complete raw bytes BEFORE parsing them.
    first, payload = capture.wire(
        http,
        path,
        request.context,
        upstream=upstream,
        params=params,
        headers=headers,
        instrument=instrument,
    )
    metadata = capture.store.normalized(first.record.evidence_id, as_of=capture.now())
    # This original-authority envelope still contains the full provider raw body,
    # not a derived economic array. Its /payload is exactly raw JSON, allowing
    # SourceBindings to identify native subtrees. Retain the first receipt too.
    source = store_observation(
        capture.store,
        raw=capture.store.raw(first.record.evidence_id),
        normalized={
            **metadata,
            "payload": payload,
            "capture_source": {
                "evidence_id": first.record.evidence_id,
                "envelope_hash": first.content_hash(),
            },
        },
        provider="operating",
        version=VERSION,
        source_uri=first.record.source_uri,
        kind="operating_raw",
        logical_key="income-original:" + first.record.evidence_id,
        context=request.context,
        ingested_at=capture.now(),
        instrument=instrument,
    )
    return source, payload


def _read_original(store, envelope, *, as_of):
    metadata = verify_operating_capture(store, envelope.record.evidence_id, as_of=as_of)
    if envelope.kind != "operating_raw" or envelope.record.provider_version != VERSION:
        _fail("income_original_wire_required")
    payload = json.loads(store.raw(envelope.record.evidence_id), parse_float=str)
    if "payload" not in metadata or canonical_bytes(metadata["payload"]) != canonical_bytes(
        payload
    ):
        _fail("income_native_payload_differs_from_original_raw")
    parent = store.get(metadata["capture_source"]["evidence_id"], as_of=as_of)
    original = verify_operating_capture(store, parent.record.evidence_id, as_of=as_of)
    if (
        parent.content_hash() != metadata["capture_source"]["envelope_hash"]
        or parent.kind != "operating_raw"
        or "capture_source" in original
        or parent.record.raw_hash != envelope.record.raw_hash
        or parent.record.source_uri != envelope.record.source_uri
        or parent.record.instrument_ids != envelope.record.instrument_ids
        or parent.record.requested_as_of != envelope.record.requested_as_of
        or parent.record.ingested_at > envelope.record.ingested_at
        or original != {k: v for k, v in metadata.items() if k not in ("payload", "capture_source")}
    ):
        _fail("income_original_receipt_binding_differs")
    return metadata, payload


def _expected_params(request):
    if request.operation == "alpaca-process-date-terms-1":
        return dict(
            symbols=request.instrument.symbol,
            start=request.started_at.date().isoformat(),
            end=request.ended_at.date().isoformat(),
            region="us",
            data_quality="all",
            sort="asc",
            limit=1000,
        )
    return dict(
        category="non_trade_activity",
        after=request.started_at.isoformat(),
        until=request.ended_at.isoformat(),
        direction="asc",
        page_size=100,
    )


def _normalize(store, sources, request, captured_at):
    terms = request.operation == "alpaca-process-date-terms-1"
    if not sources:
        _fail("income_actual_wire_sources_required")
    facts, gaps = [], set()
    pages = sources
    if not terms:
        account, pages = sources[0], sources[1:]
        meta, body = _read_original(store, account, as_of=captured_at)
        if (
            account.record.source_uri != PAPER + "/v2/account"
            or meta["upstream"] != "alpaca-paper-account-v2"
            or meta["request"] != {"path": "/v2/account", "params": {}}
            or account.record.instrument_ids
            or body.get("id") != request.provider_account_id
            or account.record.requested_as_of != request.context.as_of
        ):
            _fail("income_native_account_identity_differs")
        # Capturing an account UUID is not authority to map credits into any
        # arbitrary local book. Root must authenticate its existing venue bind.
        gaps.add("canonical-account-association-authority-required")
    if not pages:
        _fail("income_native_query_response_required")
    params = _expected_params(request)
    path = "/v1/corporate-actions" if terms else "/v2/account/activities"
    base, upstream = (
        (DATA, "alpaca-market-data") if terms else (PAPER, "alpaca-paper-activities-v2")
    )
    seen_ids, tokens = set(), set()
    for index, source in enumerate(pages):
        meta, body = _read_original(store, source, as_of=captured_at)
        expected_ids = (request.instrument.instrument_id,) if terms else ()
        if (
            source.record.source_uri != base + path
            or meta["upstream"] != upstream
            or meta["request"] != {"path": path, "params": params}
            or source.record.instrument_ids != expected_ids
            or source.record.requested_as_of != request.context.as_of
        ):
            _fail("income_native_query_identity_or_window_differs")
        if terms:
            groups = body.get("corporate_actions")
            if not isinstance(groups, dict):
                _fail("income_native_action_shape_differs")
            for kind, rows in groups.items():
                if not isinstance(rows, list):
                    _fail("income_native_action_array_required")
                for number, row in enumerate(rows):
                    if (
                        not isinstance(row, dict)
                        or row.get("symbol", request.instrument.symbol) != request.instrument.symbol
                    ):
                        _fail("income_native_action_instrument_differs")
                    native_id = row.get("id")
                    if native_id is not None and native_id in seen_ids:
                        _fail("duplicate_or_conflicting_income_action")
                    seen_ids.add(native_id)
                    token = kind.replace("~", "~0").replace("/", "~1")
                    facts.append(
                        dict(
                            category=kind,
                            source_id=source.record.evidence_id,
                            source_hash=source.content_hash(),
                            pointer=f"/payload/corporate_actions/{token}/{number}",
                            observation=row,
                        )
                    )
                    if kind != "cash_dividends":
                        gaps.add("unsupported-action-type:" + kind)
                    elif (
                        row.get("special") is not False
                        or row.get("foreign") is not False
                        or row.get("due_bill_on_date")
                        or row.get("due_bill_off_date")
                    ):
                        gaps.add("ordinary-domestic-no-due-bill-applicability-not-proved")
            token = body.get("next_page_token")
            done = not token
        else:
            if not isinstance(body, list):
                _fail("income_native_activity_array_required")
            for number, row in enumerate(body):
                if not isinstance(row, dict) or not row.get("id") or row["id"] in seen_ids:
                    _fail("duplicate_or_missing_income_activity_identity")
                seen_ids.add(row["id"])
                # Unrelated non-trade activities remain in the original raw
                # response; only exact selected-security income is exposed here.
                if (
                    str(row.get("activity_type", "")).startswith("DIV")
                    and row.get("symbol") == request.instrument.symbol
                ):
                    facts.append(
                        dict(
                            category=row["activity_type"],
                            source_id=source.record.evidence_id,
                            source_hash=source.content_hash(),
                            pointer=f"/payload/{number}",
                            observation=row,
                        )
                    )
                    if row["activity_type"] != "DIV":
                        gaps.add(
                            "unsupported-or-unassociated-income-component:" + row["activity_type"]
                        )
            done = len(body) < 100
            token = None if done else body[-1]["id"]
        if done != (index == len(pages) - 1):
            _fail("income_native_pagination_coverage_differs")
        if not done:
            if not isinstance(token, str) or not token or token in tokens:
                _fail("income_native_pagination_did_not_advance")
            tokens.add(token)
            params = {**params, "page_token": token}
    if terms:
        gaps.update(
            (
                "process-date-query-is-not-effective-event-coverage",
                "provider-announcement-availability-not-guaranteed",
                "pre-ex-entitlement-convention-authority-required",
                "publication-instant-not-provided",
            )
        )
    else:
        gaps.update(
            (
                "creation-time-query-is-not-settlement-coverage",
                "action-entitlement-association-authority-required",
                "actual-gross-fee-and-withholding-authorities-required-including-zeros",
                "per-unit-deduction-applicability-not-proved",
            )
        )
    coverage = ActionCoverage(
        coverage_id="income-query:" + pages[0].record.evidence_id,
        account_id=request.account_id,
        instrument_ids=(request.instrument.instrument_id,),
        venue="alpaca",
        scope="provider-process-date-query" if terms else "custodial-credit-ledger",
        started_at=request.started_at,
        ended_at=request.ended_at,
        available_at=captured_at,
        event_types=("cash_dividend_terms" if terms else "cash_dividend_payments",),
        completeness="unknown",
        gaps=tuple(sorted(gaps)),
        source_id=pages[0].record.evidence_id,
        source_hash=pages[0].content_hash(),
    )
    return dict(
        capture_request=request.model_dump(mode="json"),
        native_observations=facts,
        time_basis="provider-process-date" if terms else "provider-creation-time",
        income_authorities=[],
        cash_dividends=[],
        dividend_payments=[],
        dividend_distributions=[],
        action_coverage=[coverage.model_dump(mode="json")],
        paper_dividend_policies=[],
    )


class AlpacaDividendAdapter(OperatingCapture):
    """Closed paper/data GETs; native observations never authorize cash movement."""

    def __init__(
        self, store, api_key, secret_key, *, data_http=None, trading_http=None, clock=None
    ):
        super().__init__(store, clock=clock)
        if not api_key or not secret_key:
            _fail("alpaca_credentials_required")
        self.headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self.data = data_http or ReadHTTP("alpaca", DATA)
        self.trading = trading_http or ReadHTTP("alpaca", PAPER)

    def close(self):
        self.data.close()
        self.trading.close()

    def _capture(self, request, *, max_pages):
        if not 1 <= max_pages <= 1000:
            _fail("income_page_budget_invalid")
        terms = request.operation == "alpaca-process-date-terms-1"
        sources = []
        if not terms:
            account, _ = _original_wire(
                self,
                self.trading,
                "/v2/account",
                request,
                upstream="alpaca-paper-account-v2",
                headers=self.headers,
            )
            sources.append(account)
        params, seen = _expected_params(request), set()
        for _ in range(max_pages):
            wire, value = _original_wire(
                self,
                self.data if terms else self.trading,
                "/v1/corporate-actions" if terms else "/v2/account/activities",
                request,
                upstream="alpaca-market-data" if terms else "alpaca-paper-activities-v2",
                params=params,
                instrument=request.instrument if terms else None,
                headers=self.headers,
            )
            sources.append(wire)
            if terms:
                token = value.get("next_page_token")
                done = not token
            else:
                if not isinstance(value, list):
                    _fail("income_native_activity_array_required")
                done = len(value) < 100
                token = None if done else value[-1]["id"]
            if done:
                break
            if not isinstance(token, str) or token in seen:
                _fail("income_native_pagination_did_not_advance")
            seen.add(token)
            params = {**params, "page_token": token}
        else:
            _fail("income_pagination_budget_exhausted_wire_archived")
        proof = self.derived(
            request.context,
            sources,
            lambda at: _normalize(self.store, sources, request, at),
            kind="operating_dividend_terms" if terms else "operating_dividend_payments",
            logical_key=request.instrument.instrument_id,
            instrument=request.instrument,
            policy="native-income-observation-explicit-gaps-1",
        )
        return self.result(
            [*sources, proof],
            detail="native source capture succeeded; canonical income/complete coverage remain unproved",
        )

    @_operation
    def cash_dividend_sources(self, instrument, context, *, started_at, ended_at, max_pages=100):
        return self._capture(
            IncomeCaptureRequest(
                operation="alpaca-process-date-terms-1",
                context=context,
                instrument=instrument,
                started_at=started_at,
                ended_at=ended_at,
            ),
            max_pages=max_pages,
        )

    @_operation
    def account_dividend_sources(
        self,
        instrument,
        context,
        *,
        account_id,
        provider_account_id,
        started_at,
        ended_at,
        max_pages=100,
    ):
        return self._capture(
            IncomeCaptureRequest(
                operation="alpaca-created-account-activities-1",
                context=context,
                instrument=instrument,
                account_id=account_id,
                provider_account_id=provider_account_id,
                started_at=started_at,
                ended_at=ended_at,
            ),
            max_pages=max_pages,
        )


def verify_operating_dividend_capture(
    store: EvidenceStore, evidence_id: str, *, as_of: datetime
) -> dict:
    """Reparse original native bodies and reconstruct ALL normalized income data.

    This parser currently proves the observed native records and explicit gaps.
    It rejects fabricated canonical income, deduction policies or complete
    economic coverage that these provider APIs do not substantiate.
    """
    try:
        envelope = store.get(evidence_id, as_of=as_of)
        value = verify_operating_capture(store, evidence_id, as_of=as_of)
        request = IncomeCaptureRequest.model_validate(value["capture_request"])
        expected_kind = (
            "operating_dividend_terms"
            if request.operation == "alpaca-process-date-terms-1"
            else "operating_dividend_payments"
        )
        if (
            envelope.kind != expected_kind
            or envelope.record.provider_version != VERSION
            or envelope.record.requested_as_of != request.context.as_of
            or envelope.record.instrument_ids != (request.instrument.instrument_id,)
        ):
            _fail("income_capture_request_identity_differs")
        sources = tuple(
            store.get(s["evidence_id"], as_of=as_of) for s in value["source_manifest"]["sources"]
        )
        if len({s.record.evidence_id for s in sources}) != len(sources):
            _fail("income_duplicate_source_identity")
        actual = _normalize(store, sources, request, envelope.record.ingested_at)
        if canonical_bytes(
            {**actual, "source_manifest": value["source_manifest"]}
        ) != canonical_bytes(value):
            _fail("income_normalization_not_proved_by_original_wire")
        return {**actual, "source_manifest": value["source_manifest"]}
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError, AttributeError):
        _fail("income_original_source_or_scope_unproved")
