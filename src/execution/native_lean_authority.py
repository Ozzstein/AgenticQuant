"""Original LEAN/native ownership reexport, without a runtime or transport."""

from datetime import datetime
from decimal import Decimal as D

from src.domain.common import _utc
from src.domain.portfolio import exact_decimal_sum
from src.execution.ledger import LedgerError
from src.execution.native_archive import canonical, digest, exact_time
from src.execution.venue import VenueOrderReport


def unwrap(archive, report, intent, claim):
    from src.execution.lean_venue import LeanDeployment, LeanVenue

    payload, _, _ = archive.event("remote-binding:lean")
    deployment = LeanDeployment.model_validate(payload)
    venue = {
        "AlpacaBrokerage": "alpaca",
        "BinanceBrokerage": "binance",
        "KrakenBrokerage": "kraken",
    }.get(deployment.brokerage)
    if (
        venue is None
        or deployment.mode != "live"
        or archive.spec.mode != "live"
        or deployment.account_id != archive.account_id
        or intent.instrument not in [item.instrument for item in deployment.instruments]
    ):
        raise LedgerError("native LEAN original deployment/account/instrument differs")
    proof, _, _ = archive.event(report.evidence_id, content_addressed=True)
    if (
        not report.evidence_id.startswith("lean-native-report:")
        or proof["client_order_id"] != report.client_order_id
    ):
        raise LedgerError("native LEAN report source/client differs")
    native = VenueOrderReport.model_validate(proof["native_report"])
    if native.venue != venue or native.observed_at > report.observed_at:
        raise LedgerError("native LEAN underlying report venue/clock differs")
    bound, _, _ = archive.event(proof["native_binding"])
    if proof["native_binding"] != "lean-native-order:" + digest([venue, native.venue_order_id]):
        raise LedgerError("native LEAN original broker identity differs")
    if not archive.table_exists("lean_ticket_bindings"):
        raise LedgerError("native LEAN original ticket authority unavailable")
    row = archive.ledger.db.execute(
        "SELECT * FROM lean_ticket_bindings WHERE account=? AND intent=?",
        (archive.account_id, intent.intent_id),
    ).fetchone()
    if row is None:
        raise LedgerError("native LEAN original ticket binding unavailable")
    expected = {
        "deployment_hash": deployment.content_hash(),
        "venue": venue,
        "broker_order_id": native.venue_order_id,
        "runtime_id": row["runtime"],
        "ticket_id": row["ticket"],
        "intent_hash": intent.content_hash(),
        "intent_id": intent.intent_id,
        "client_order_id": report.client_order_id,
    }
    if (
        bound != expected
        or report.venue_order_id != row["runtime"] + ":" + str(row["ticket"])
        or row["client"] != report.client_order_id
        or claim["runtime_id"] != row["runtime"]
    ):
        raise LedgerError("native LEAN original submission/ticket/broker ownership differs")
    native_client, _, _ = archive.event(proof["native_binding"] + ":client")
    if native_client != {"native_client": native.client_order_id}:
        raise LedgerError("native LEAN original broker client identity differs")
    binding, _, _ = archive.event("remote-binding:" + venue)
    if binding["broker_account_id"] != deployment.broker_account_id:
        raise LedgerError("native LEAN broker account differs from deployment")
    for runtime_id in {row["runtime"], proof["runtime_id"]}:
        runtime, _, _ = archive.event("lean-runtime:" + runtime_id)
        if (
            runtime["runtime_id"] != runtime_id
            or runtime["deployment_hash"] != deployment.content_hash()
            or runtime.get("live_mode") is not True
            or not runtime.get("engine_version")
        ):
            raise LedgerError("native LEAN retained runtime/deployment identity differs")
        if (
            venue in ("binance", "kraken")
            and runtime.get("native_credential_identity") != binding["credential_identity"]
        ):
            raise LedgerError("native LEAN retained credential identity differs")
    ticket, _, _ = archive.event(proof["ticket_source"], content_addressed=True)
    if (
        not proof["ticket_source"].startswith("lean-ticket:")
        or ticket["runtime_id"] != row["runtime"]
        or ticket["ticket_id"] != row["ticket"]
        or ticket["client_order_id"] != report.client_order_id
        or ticket["broker_ids"] != [native.venue_order_id]
    ):
        raise LedgerError("native LEAN original ticket source differs")
    LeanVenue._validate_ticket_intent(intent, ticket)
    recovered = proof.get("recovery_policy")
    if recovered is not None and recovered != "original-native-order-history-1":
        raise LedgerError("unsupported native LEAN recovery policy")
    complete = native.executions_complete
    if recovered is None:
        ids, net, gross = [], [], []
        for event in ticket["events"]:
            identity = event["event_id"]
            if (
                type(identity) is not int
                or identity < 1
                or identity in ids
                or event["ticket_id"] != row["ticket"]
                or event["instrument_id"] != intent.instrument.instrument_id
                or event.get("is_assignment") is not False
            ):
                raise LedgerError("native LEAN original event identity differs")
            ids.append(identity)
            event_record, _, _ = archive.event(
                "lean-order-event:" + digest([row["runtime"], row["ticket"], identity])
            )
            original = {
                "runtime_id": row["runtime"],
                "event": event,
                "client_order_id": report.client_order_id,
                "intent_hash": intent.content_hash(),
            }
            if canonical(event_record["native"]) != canonical(original):
                raise LedgerError("native LEAN original event revision differs")
            first_ticket, _, _ = archive.event(event_record["source"], content_addressed=True)
            if event not in first_ticket["events"] or first_ticket["runtime_id"] != row["runtime"]:
                raise LedgerError("native LEAN event absent from first ticket source")
            quantity, charge = D(event["fill_quantity"]), D(event["fee_amount"])
            net.append(quantity.copy_abs())
            if charge < 0 or not quantity and charge:
                raise LedgerError("unsupported native LEAN runtime fee event")
            if not quantity:
                continue
            occurred = exact_time(event["occurred_at"], "iso8601-utc")
            if (
                quantity * (1 if intent.side == "buy" else -1) <= 0
                or event["status"] not in ("Filled", "PartiallyFilled")
                or event["price_currency"] != archive.spec.base_currency
                or occurred > _utc(datetime.fromisoformat(event_record["observed_at"]))
                or occurred > _utc(datetime.fromisoformat(ticket["observed_at"]))
            ):
                raise LedgerError("native LEAN event side/currency/receipt clock differs")
            if event["fee_type"] == "QuantConnect.Orders.Fees.ModifiedFillQuantityOrderFee":
                base = next(
                    item.base_currency
                    for item in deployment.instruments
                    if item.instrument == intent.instrument
                )
                if intent.side != "buy" or base is None or event["fee_currency"] != base:
                    raise LedgerError("native LEAN modified gross quantity basis differs")
                quantity = exact_decimal_sum([quantity, charge])
            elif event["fee_type"] != "QuantConnect.Orders.Fees.OrderFee":
                raise LedgerError("unsupported native LEAN fee semantics")
            gross.append(quantity.copy_abs())
        filled = D(ticket["filled_quantity"])
        if (
            filled * (1 if intent.side == "buy" else -1) < 0
            or filled.copy_abs() > intent.quantity
            or exact_decimal_sum(gross) > intent.quantity
        ):
            raise LedgerError("native LEAN original cumulative quantity exceeds admission")
        complete &= (
            ids == list(range(1, len(ids) + 1))
            and exact_decimal_sum(net) == filled.copy_abs()
            and ticket["status"]
            in ("New", "Submitted", "PartiallyFilled", "Filled", "Canceled", "Invalid")
            and exact_decimal_sum(gross) == native.cumulative_quantity
        )
    at = (
        native.observed_at
        if recovered
        else max(native.observed_at, _utc(datetime.fromisoformat(ticket["observed_at"])))
    )
    if (
        report.observed_at != at
        or _utc(datetime.fromisoformat(proof["observed_at"])) != at
        or report.status != native.status
        or report.cumulative_quantity != native.cumulative_quantity
        or report.fills != native.fills
        or report.executions_complete != complete
    ):
        raise LedgerError("native LEAN report differs from original runtime/native closure")
    return native
