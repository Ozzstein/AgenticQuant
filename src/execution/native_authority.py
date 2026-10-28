"""Independent native execution reexport from immutable retained bytes only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal as D
from types import SimpleNamespace
from urllib.parse import quote

from src.domain.common import _utc
from src.domain.portfolio import Fill, OrderIntent, exact_decimal_sum
from src.execution.discrepancies import (
    NativeExecutionAuthority,
    NativeExecutionKey,
    RetainedSourceRef,
)
from src.execution.ledger import LedgerError
from src.execution.native_archive import NativeArchive, canonical, decode, digest, exact_time
from src.execution.native_order_terms import require_alpaca_order
from src.execution.venue import VenueExecution, VenueOrderReport, VenueRules


@dataclass(frozen=True)
class ReexportedReport:
    original_report: RetainedSourceRef
    report: VenueOrderReport
    authorities: tuple[NativeExecutionAuthority, ...]
    native_report: VenueOrderReport
    sources: tuple[tuple[str, RetainedSourceRef], ...]


def _order_binding(archive, reference):
    report = VenueOrderReport.model_validate(archive.verify_reference(reference))
    intent = archive.ledger.order(archive.account_id, reference.intent_id)["intent"]
    if intent.account_id != archive.account_id or report.observed_at > archive.as_of:
        raise LedgerError("native report original account/cutoff differs")
    journal = archive.ledger.db.execute(
        "SELECT payload FROM journal WHERE account=? AND id=? AND kind='order'",
        (archive.account_id, "order:" + intent.intent_id),
    ).fetchone()
    if journal is None or OrderIntent.model_validate_json(journal[0]) != intent:
        raise LedgerError("native intent differs from its original journal admission")
    if not archive.table_exists("execution_books") or not archive.table_exists("venue_outbox"):
        raise LedgerError("missing original native execution ownership")
    book = archive.ledger.db.execute(
        "SELECT * FROM execution_books WHERE account=?", (archive.account_id,)
    ).fetchone()
    outbox = archive.ledger.db.execute(
        "SELECT * FROM venue_outbox WHERE account=? AND intent=?",
        (archive.account_id, intent.intent_id),
    ).fetchone()
    if book is None or outbox is None:
        raise LedgerError("native report lacks original owned outbox")
    configuration = decode(book["configuration"])
    expected_client = VenueExecution.client_id(archive.account_id, intent.intent_id)
    if (
        book["owner"] not in ("direct", "lean")
        or outbox["venue"] != report.venue
        or outbox["client_id"] != expected_client
        or report.client_order_id != expected_client
        or configuration["routes"].get(intent.instrument.instrument_id) != report.venue
        or report.venue not in configuration["venues"]
        or outbox["venue_id"] not in (None, report.venue_order_id)
    ):
        raise LedgerError("native report differs from original execution owner/route/client")
    rules = VenueRules.model_validate_json(outbox["rules"])
    if rules.instrument != intent.instrument or rules.venue != report.venue:
        raise LedgerError("native original instrument/rules identity differs")
    metadata, _, _ = archive.event("venue-admission-metadata:" + intent.intent_id)
    from src.domain.portfolio import InstrumentRisk

    if intent.instrument.instrument_id not in metadata:
        raise LedgerError("native original admission metadata incomplete")
    risk = InstrumentRisk.model_validate(metadata[intent.instrument.instrument_id])
    if risk.instrument != intent.instrument or risk.as_of > intent.decision_at:
        raise LedgerError("native original admission instrument/clock differs")
    owner = book["owner"]
    claim, _, _ = archive.event(
        ("lean" if owner == "lean" else report.venue) + "-submit-attempt:" + intent.intent_id
    )
    at = _utc(datetime.fromisoformat(claim["at"]))
    if (
        claim["intent_hash"] != intent.content_hash()
        or claim["client_order_id"] != expected_client
        or not intent.eligible_at <= at <= report.observed_at
        or intent.expires_at
        and at >= intent.expires_at
    ):
        raise LedgerError("native original submission claim identity/admission clock differs")
    identity_hash = digest(
        {key: outbox[key] for key in ("account", "intent", "venue", "client_id", "rules")}
    )
    return report, intent, owner, digest(dict(book)), identity_hash, rules, claim


def _alpaca_binding(archive, intent):
    binding, _, _ = archive.event("remote-binding:alpaca")
    network = {
        "https://paper-api.alpaca.markets": "alpaca-paper",
        "https://api.alpaca.markets": "mainnet",
    }.get(binding.get("endpoint"))
    if (
        network is None
        or (network == "alpaca-paper") != (archive.spec.mode == "paper")
        or archive.spec.mode not in ("paper", "live")
        or archive.spec.base_currency != "USD"
        or binding.get("fee_convention") != "separately-posted-account-fees-1"
        or not binding.get("broker_account_id")
        or intent.fx_rate != 1
    ):
        raise LedgerError("native Alpaca account/network/cash binding differs")
    mapping, _, _ = archive.event("remote-instrument:alpaca:" + intent.instrument.instrument_id)
    if mapping["instrument"] != intent.instrument.model_dump(mode="json"):
        raise LedgerError("native Alpaca original instrument mapping differs")
    # Verify a retained actual account response, not the configured label alone.
    candidates = archive.ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id LIKE 'alpaca-capture:%' ORDER BY rowid DESC",
        (archive.account_id,),
    ).fetchall()
    for row in candidates:
        capture = decode(row["payload"])
        if _utc(datetime.fromisoformat(capture["observed_at"])) > intent.decision_at:
            continue
        capture, _, _ = archive.event(row["id"], content_addressed=True)
        raw, account, _, _ = archive.raw(capture["references"]["account"], "alpaca", network)
        if (
            raw["method"] != "GET"
            or raw["path"] != "/v2/account"
            or raw.get("params") is not None
            or raw.get("request") is not None
            or account != capture["account"]
            or account.get("id") != binding["broker_account_id"]
            or account.get("currency") != "USD"
            or _utc(datetime.fromisoformat(raw["observed_at"]))
            > _utc(datetime.fromisoformat(capture["observed_at"]))
        ):
            raise LedgerError("native Alpaca original account source identity differs")
        return network, binding["broker_account_id"], mapping["symbol"]
    raise LedgerError("missing original native Alpaca account source")


def _alpaca_first(archive, network, activity):
    probe = NativeArchive(archive.ledger, archive.account_id, as_of=archive.as_of)
    matches = []
    for row in archive.ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id LIKE 'alpaca-http:%' ORDER BY rowid",
        (archive.account_id,),
    ):
        header = decode(row["payload"])
        if (
            header.get("path") != "/v2/account/activities"
            or header.get("status") != 200
            or _utc(datetime.fromisoformat(header["observed_at"])) > archive.as_of
        ):
            continue
        raw, payload, _, _ = probe.raw(row["id"], "alpaca", network)
        params = raw.get("params") or {}
        if (
            raw["method"] != "GET"
            or raw.get("request") is not None
            or set(params)
            - {"activity_types", "direction", "page_size", "after", "order_id", "page_token"}
            or params.get("activity_types") != "FILL"
            or params.get("direction") != "asc"
            or params.get("page_size") != 100
            or params.get("after") != archive.spec.opened_at.isoformat()
        ):
            raise LedgerError("native activity source method differs")
        if not isinstance(payload, list):
            raise LedgerError("native activity source coverage differs")
        for candidate in payload:
            if candidate.get("activity_type") == "FILL" and candidate.get("id") == activity["id"]:
                if params.get("order_id") not in (None, activity["order_id"]) or canonical(
                    candidate
                ) != canonical(activity):
                    raise LedgerError("native execution has a later/earlier conflicting revision")
                matches.append((row["id"], _utc(datetime.fromisoformat(raw["observed_at"]))))
    if not matches:
        raise LedgerError("native execution first raw receipt unavailable")
    raw, _, reference, _ = archive.raw(matches[0][0], "alpaca", network)
    return reference, _utc(datetime.fromisoformat(raw["observed_at"]))


def _alpaca_report(archive, report, intent, native_client):
    network, account, symbol = _alpaca_binding(archive, intent)
    content, _, _ = archive.event(report.evidence_id, content_addressed=True)
    if (
        not report.evidence_id.startswith("alpaca-order:")
        or content.get("fee_convention") != "separately-posted-account-fees-1"
    ):
        raise LedgerError("native Alpaca report convention differs")
    order = content["order"]
    require_alpaca_order(intent, native_client, order, symbol)
    if order["id"] != report.venue_order_id:
        raise LedgerError("native Alpaca order ID differs")
    if intent.tif == "DAY":
        from src.execution.alpaca_calendar import read

        proof = read(SimpleNamespace(ledger=archive.ledger, account_id=archive.account_id), intent)
        archive.event("alpaca-day-calendar:" + intent.intent_id)
        archive.raw(proof["source"], "alpaca", network)
    references = content["references"]
    if (
        not isinstance(references, list)
        or not references
        or len(set(references)) != len(references)
    ):
        raise LedgerError("native Alpaca report source coverage differs")
    raw, actual_order, _, _ = archive.raw(references[0], "alpaca", network)
    direct = (
        raw["method"] == "GET"
        and raw["path"] == "/v2/orders/" + quote(str(order["id"]), safe="")
        and raw.get("params") is None
        and raw.get("request") is None
    )
    by_client = (
        raw["method"] == "GET"
        and raw["path"] == "/v2/orders:by_client_order_id"
        and raw.get("params") == {"client_order_id": native_client}
        and raw.get("request") is None
    )
    submission = {
        "symbol": symbol,
        "qty": str(intent.quantity),
        "side": intent.side,
        "type": intent.order_type,
        "time_in_force": intent.tif.lower(),
        "client_order_id": native_client,
        "extended_hours": False,
        "order_class": "simple",
    }
    for name in ("limit_price", "stop_price", "trail_price", "trail_percent"):
        if getattr(intent, name) is not None:
            submission[name] = str(getattr(intent, name))
    submitted = (
        raw["method"] == "POST"
        and raw["path"] == "/v2/orders"
        and raw.get("params") is None
        and raw.get("request") == submission
    )
    if not (direct or by_client or submitted) or canonical(actual_order) != canonical(order):
        raise LedgerError("native Alpaca order differs from original raw response")
    activities, seen, token = [], set(), None
    observed = _utc(datetime.fromisoformat(raw["observed_at"]))
    cumulative = D(order["filled_qty"])
    if bool(cumulative) != bool(references[1:]):
        raise LedgerError("native Alpaca execution page coverage differs")
    for number, source in enumerate(references[1:]):
        raw, page, _, _ = archive.raw(source, "alpaca", network)
        expected = {
            "activity_types": "FILL",
            "direction": "asc",
            "page_size": 100,
            "after": archive.spec.opened_at.isoformat(),
            "order_id": order["id"],
        }
        if token is not None:
            expected["page_token"] = token
        if (
            raw["method"] != "GET"
            or raw["path"] != "/v2/account/activities"
            or raw.get("params") != expected
            or raw.get("request") is not None
            or not isinstance(page, list)
            or len(page) > 100
        ):
            raise LedgerError("native Alpaca execution endpoint/pagination differs")
        at = _utc(datetime.fromisoformat(raw["observed_at"]))
        if at < observed or (number < len(references) - 2) != (len(page) == 100):
            raise LedgerError("native Alpaca incomplete or noncausal execution pages")
        observed = at
        for activity in page:
            if (
                activity.get("activity_type") != "FILL"
                or not activity.get("id")
                or activity["id"] in seen
                or activity.get("order_id") != order["id"]
                or activity.get("symbol") != symbol
                or activity.get("side") != intent.side
            ):
                raise LedgerError("native Alpaca execution identity/coverage differs")
            seen.add(activity["id"])
            activities.append(activity)
        token = page[-1]["id"] if page else None
    status = {
        "new": "new",
        "accepted": "new",
        "pending_new": "new",
        "partially_filled": "partial",
        "filled": "filled",
        "canceled": "cancelled",
        "expired": "expired",
        "rejected": "rejected",
    }.get(order.get("status"), "unknown")
    facts = []
    for activity in activities:
        first, first_at = _alpaca_first(archive, network, activity)
        native_at = exact_time(activity["transaction_time"], "iso8601-utc")
        retained, _, _ = archive.event(
            "alpaca-activity:" + digest(activity), content_addressed=True
        )
        if canonical(retained) != canonical(activity):
            raise LedgerError("original Alpaca activity receipt changed")
        fill = Fill(
            fill_id="alpaca:" + activity["id"],
            intent_id=intent.intent_id,
            account_id=archive.account_id,
            quantity=activity["qty"],
            price=activity["price"],
            fee=D(0),
            executed_at=native_at,
            event_id="alpaca-activity:" + digest(activity),
        )
        facts.append(
            (
                fill,
                NativeExecutionKey(
                    venue="alpaca",
                    network=network,
                    broker_account_id=account,
                    instrument=intent.instrument,
                    venue_order_id=order["id"],
                    native_execution_id=activity["id"],
                ),
                first,
                first_at,
                activity["transaction_time"],
                "iso8601-utc",
            )
        )
    fills = tuple(fact[0] for fact in facts)
    quantity = exact_decimal_sum(fill.quantity for fill in fills)
    expected = VenueOrderReport(
        venue="alpaca",
        client_order_id=native_client,
        venue_order_id=order["id"],
        observed_at=observed,
        status=status,
        cumulative_quantity=cumulative,
        fills=fills,
        executions_complete=quantity == cumulative and status != "unknown",
        evidence_id=report.evidence_id,
    )
    if expected != report or quantity > cumulative:
        raise LedgerError(
            "native Alpaca normalized report differs from exact original raw economics"
        )
    return facts


def reexport_report(ledger, original_report, *, as_of):
    """Read-only proof. Callers cannot replace any original source or economic field."""
    reference = RetainedSourceRef.model_validate(original_report.model_dump())
    if reference.store_kind != "venue-report":
        raise LedgerError("native reexport requires the exact original stored venue-report key")
    with ledger._reading():
        archive = NativeArchive(ledger, reference.account_id, as_of=as_of)
        report, intent, owner, book_hash, outbox_hash, _, claim = _order_binding(archive, reference)
        native = report
        if owner == "lean":
            from src.execution.native_lean_authority import unwrap

            native = unwrap(archive, report, intent, claim)
        if native.venue == "alpaca":
            facts = _alpaca_report(archive, native, intent, native.client_order_id)
        elif native.venue in ("binance", "kraken"):
            from src.execution.native_spot_authority import binance_report, kraken_report

            exporter = binance_report if native.venue == "binance" else kraken_report
            native_client = native.client_order_id
            if native.venue == "kraken" and owner == "lean":
                content, _, _ = archive.event(native.evidence_id, content_addressed=True)
                native_client = content["order"].get("cl_ord_id")
            facts = exporter(archive, native, intent, native_client)
        else:
            raise LedgerError("unsupported_native_reexport_profile")
        sources = archive.source_bindings()
        if not report.executions_complete:
            # The raw report can truthfully describe incomplete coverage. It is
            # not execution authority until a later independently verified full
            # prefix covers every one of its actual observed executions.
            return ReexportedReport(reference, report, (), native, sources)
        manifest_hash = digest([(name, source.model_dump(mode="json")) for name, source in sources])
        authorities = tuple(
            NativeExecutionAuthority(
                key=key,
                account_id=archive.account_id,
                account_spec_hash=archive.spec.content_hash(),
                owner=owner,
                execution_book_hash=book_hash,
                outbox_identity_hash=outbox_hash,
                intent_id=intent.intent_id,
                intent_hash=intent.content_hash(),
                fill=fill,
                original_report=reference,
                first_execution_receipt=first,
                first_observed_at=first_at,
                first_complete_at=report.observed_at,
                native_timestamp=wire,
                native_timestamp_unit=unit,
                sources=sources,
                source_manifest_hash=manifest_hash,
            )
            for fill, key, first, first_at, wire, unit in facts
        )
        return ReexportedReport(reference, report, authorities, native, sources)


def reexport_execution(ledger, account_id, fill_id, *, as_of):
    """Resolve a stored native ID to its first complete, independently proved prefix.

    This is factual authority only, including facts outside admitted order terms.
    Later retained reports/source revisions are checked before returning the first
    authority. Callers supply neither a Fill nor an authority object.
    """
    with ledger._reading():
        archive = NativeArchive(ledger, account_id, as_of=as_of)
        if not archive.table_exists("venue_reports"):
            raise LedgerError("original native report history unavailable")
        first = None
        for row in ledger.db.execute(
            "SELECT intent,evidence,payload FROM venue_reports WHERE account=? ORDER BY rowid",
            (account_id,),
        ):
            report = VenueOrderReport.model_validate_json(row["payload"])
            if report.observed_at > archive.as_of or not any(
                fill.fill_id == fill_id for fill in report.fills
            ):
                continue
            reference = archive.reference(
                "venue-report", row["evidence"], row["payload"], intent_id=row["intent"]
            )
            exported = reexport_report(ledger, reference, as_of=archive.as_of)
            for authority in exported.authorities:
                if authority.fill.fill_id != fill_id:
                    continue
                if first is None:
                    first = authority
                elif (
                    first.model_dump(
                        exclude={
                            "original_report",
                            "first_complete_at",
                            "sources",
                            "source_manifest_hash",
                        }
                    )
                    != authority.model_dump(
                        exclude={
                            "original_report",
                            "first_complete_at",
                            "sources",
                            "source_manifest_hash",
                        }
                    )
                    or authority.first_complete_at < first.first_complete_at
                ):
                    raise LedgerError(
                        "native execution first complete authority conflicts with retained history"
                    )
        if first is None:
            raise LedgerError("native execution has no complete retained source authority")
        return first
