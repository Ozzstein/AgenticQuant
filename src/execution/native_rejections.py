"""Closed negative submission authority from original retained native responses."""

from datetime import datetime

from src.domain.common import _utc
from src.domain.portfolio import OrderIntent
from src.execution.execution_admission import _read_submission_sources
from src.execution.ledger import LedgerError
from src.execution.native_archive import decode, exact_time
from src.execution.native_submission_terms import definitive_rejection, submission_payload


def reexport_rejection(archive, intent_id):
    from src.execution.native_authority import _alpaca_binding
    from src.execution.native_spot_authority import binding
    from src.execution.venue import VenueExecution, VenueRules

    ledger, account = archive.ledger, archive.account_id
    intent = ledger.order(account, intent_id)["intent"]
    book = ledger.db.execute("SELECT * FROM execution_books WHERE account=?", (account,)).fetchone()
    row = ledger.db.execute(
        "SELECT * FROM venue_outbox WHERE account=? AND intent=?", (account, intent_id)
    ).fetchone()
    journal = ledger.db.execute(
        "SELECT payload FROM journal WHERE account=? AND id=? AND kind='order'",
        (account, "order:" + intent_id),
    ).fetchone()
    if (
        book is None
        or row is None
        or book["owner"] != "direct"
        or journal is None
        or OrderIntent.model_validate_json(journal[0]) != intent
    ):
        raise LedgerError("negative native response lacks original direct owner/admission")
    venue, client = row["venue"], VenueExecution.client_id(account, intent_id)
    configuration = decode(book["configuration"])
    rules = VenueRules.model_validate_json(row["rules"])
    if (
        row["client_id"] != client
        or configuration["routes"].get(intent.instrument.instrument_id) != venue
        or rules.venue != venue
        or rules.instrument != intent.instrument
    ):
        raise LedgerError("negative native response original route/client/rules differ")
    claim, _, _ = archive.event(venue + "-submit-attempt:" + intent_id)
    claim_at = _utc(datetime.fromisoformat(claim["at"]))
    if (
        claim["intent_hash"] != intent.content_hash()
        or claim["client_order_id"] != client
        or not intent.eligible_at <= claim_at <= archive.as_of
        or (intent.expires_at is not None and claim_at >= intent.expires_at)
    ):
        raise LedgerError("negative native response original claim differs")
    _read_submission_sources(archive, intent, claim)
    if venue == "alpaca":
        network, _, symbol = _alpaca_binding(archive, intent)
        expected = submission_payload(venue, intent, client, symbol=symbol)
    else:
        _, market, _, network = binding(archive, intent, venue)
        expected = submission_payload(
            venue, intent, client, symbol=market.symbol, native_id=market.native_id
        )
    candidates = []
    for source in ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id LIKE ? ORDER BY rowid",
        (account, venue + "-http:%"),
    ):
        value = decode(source["payload"])
        params = value.get("request") or value.get("parameters") or {}
        if value.get("method") == "POST" and any(
            params.get(name) == client
            for name in ("client_order_id", "newClientOrderId", "cl_ord_id")
        ):
            candidates.append(source["id"])
    if len(candidates) != 1:
        raise LedgerError("native submission has no single complete definite rejection source")
    header, payload, ref, _ = archive._response(
        candidates[0], venue, network, statuses=(200, 400, 401, 403, 422)
    )
    observed = _utc(datetime.fromisoformat(header["observed_at"]))
    if observed < claim_at or not definitive_rejection(venue, header["status"], payload):
        raise LedgerError("native transmission remains ambiguous or accepted")
    if venue == "alpaca":
        valid = (
            header["path"] == "/v2/orders"
            and header.get("params") is None
            and header.get("request") == expected
        )
    else:
        params = header["parameters"]
        if venue == "binance":
            endpoint = (
                "https://testnet.binance.vision"
                if network == "binance-testnet"
                else "https://api.binance.com"
            )
            valid = (
                header["endpoint"] == endpoint + "/api/v3/order"
                and params.get("recvWindow") == 5000
                and exact_time(params.get("timestamp"), "unix-milliseconds") <= observed
            )
            params = {
                key: value
                for key, value in params.items()
                if key not in ("timestamp", "recvWindow")
            }
        else:
            valid = header["endpoint"] == "https://api.kraken.com/0/private/AddOrder"
        valid &= header.get("api") == "private" and params == expected
    if not valid:
        raise LedgerError("negative response belongs to another native request")
    return ref
