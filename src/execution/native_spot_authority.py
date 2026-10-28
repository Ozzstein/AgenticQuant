"""Pure Binance/Kraken stored trade and debit/credit reconstruction."""

from datetime import datetime
from decimal import Decimal as D

from src.domain.common import _utc
from src.domain.portfolio import Fill, FillSettlement, InventoryFee, exact_decimal_sum
from src.execution.discrepancies import NativeExecutionKey
from src.execution.ledger import LedgerError
from src.execution.native_archive import NativeArchive, canonical, decode, digest, exact_time
from src.execution.native_order_terms import require_binance_order, require_kraken_order
from src.execution.spot_venue import NativeSpotMarket
from src.execution.venue import VenueOrderReport


def raw(archive, source, venue, network, path, params, *, method=None):
    receipt, value, reference, sequence = archive.raw(source, venue, network)
    if venue == "binance":
        expected_url = (
            (
                "https://testnet.binance.vision"
                if network == "binance-testnet"
                else "https://api.binance.com"
            )
            + "/api/v3/"
            + path
        )
        actual = {
            key: value
            for key, value in receipt["parameters"].items()
            if key not in ("timestamp", "recvWindow")
        }
        if receipt["parameters"].get("recvWindow") != 5000 or exact_time(
            receipt["parameters"].get("timestamp"), "unix-milliseconds"
        ) > _utc(datetime.fromisoformat(receipt["observed_at"])):
            raise LedgerError("native Binance original request clock/window differs")
    else:
        expected_url = "https://api.kraken.com/0/private/" + path
        actual = receipt["parameters"]
    if (
        receipt["endpoint"] != expected_url
        or receipt["method"] != (method or ("GET" if venue == "binance" else "POST"))
        or receipt["api"] != "private"
        or actual != params
    ):
        raise LedgerError("native spot original endpoint/request parameters differ")
    return receipt, value, reference, sequence


def binding(archive, intent, venue):
    value, _, _ = archive.event("remote-binding:" + venue)
    if (
        value["mode"] != archive.spec.mode
        or archive.spec.mode not in ("paper", "live")
        or value["base_currency"] != archive.spec.base_currency
        or intent.fx_rate != 1
    ):
        raise LedgerError("native spot original mode/cash binding differs")
    if venue == "kraken" and archive.spec.mode != "live":
        raise LedgerError("native Kraken paper authority unsupported")
    network = "binance-testnet" if archive.spec.mode == "paper" else "mainnet"
    markets = {key: NativeSpotMarket.model_validate(item) for key, item in value["markets"].items()}
    market = markets.get(intent.instrument.instrument_id)
    if (
        market is None
        or market.instrument != intent.instrument
        or market.instrument.quote_currency != archive.spec.base_currency
        or market.quote_asset != value["cash_asset"]
        or venue == "binance"
        and market.quote_asset != archive.spec.base_currency
    ):
        raise LedgerError("native spot exact instrument/cash mapping differs")
    rows = archive.ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id LIKE ? ORDER BY rowid DESC",
        (archive.account_id, venue + "-capture:%"),
    ).fetchall()
    for row in rows:
        capture = decode(row["payload"])
        if _utc(datetime.fromisoformat(capture["observed_at"])) > intent.decision_at:
            continue
        archive.event(row["id"], content_addressed=True)
        role, endpoint = (
            ("account", "account") if venue == "binance" else ("wallets", "ListWalletAccounts")
        )
        receipt, payload, _, _ = raw(
            archive, capture["sources"][role], venue, network, endpoint, {}
        )
        if canonical(payload) != canonical(capture["payload"][role]) or _utc(
            datetime.fromisoformat(receipt["observed_at"])
        ) > _utc(datetime.fromisoformat(capture["observed_at"])):
            raise LedgerError("native spot original account capture differs from raw bytes")
        if venue == "binance":
            from src.execution.binance_venue import native_integer

            if (
                str(native_integer(payload.get("uid"))) != value["broker_account_id"]
                or payload.get("accountType") != "SPOT"
            ):
                raise LedgerError("native Binance original account identity differs")
        else:
            from src.execution.kraken_venue import result

            wallets = result(payload)
            main = [entry for entry in wallets["accounts"] if entry["type"] == "main"]
            if (
                len(main) != 1
                or main[0]["account_id"] != value["broker_account_id"]
                or wallets["cursor"]["next"] is not None
            ):
                raise LedgerError("native Kraken original wallet identity/coverage differs")
        return value, market, markets, network
    raise LedgerError("missing original native spot account source")


def kraken_proof(archive, source, network):
    from src.execution.kraken_venue import result

    proof, _, _ = archive.event(source, content_addressed=True)
    if not source.startswith("kraken-trade-proof:"):
        raise LedgerError("original Kraken trade debit proof namespace differs")
    trade_receipt, trade_raw, _, _ = archive.raw(proof["trade_source"], "kraken", network)
    params = trade_receipt["parameters"]
    raw(archive, proof["trade_source"], "kraken", network, "QueryTrades", params)
    if (
        set(params) != {"txid", "ledgers"}
        or params["ledgers"] is not True
        or proof["trade_id"] not in params["txid"].split(",")
        or canonical(result(trade_raw).get(proof["trade_id"])) != canonical(proof["trade"])
    ):
        raise LedgerError("original Kraken trade source membership differs")
    ledgers, clocks = {}, [_utc(datetime.fromisoformat(trade_receipt["observed_at"]))]
    ids = proof["trade"].get("ledgers")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise LedgerError("original Kraken debit IDs unavailable/duplicate")
    if len(proof["ledger_sources"]) != (len(ids) + 19) // 20:
        raise LedgerError("original Kraken debit proof page coverage differs")
    for index, ledger_source in enumerate(proof["ledger_sources"]):
        requested = ids[index * 20 : (index + 1) * 20]
        receipt, payload, _, _ = raw(
            archive, ledger_source, "kraken", network, "QueryLedgers", {"id": ",".join(requested)}
        )
        values = result(payload)
        if set(values) != set(requested) or set(values) & set(ledgers):
            raise LedgerError("original Kraken debit membership differs")
        ledgers.update(values)
        clocks.append(_utc(datetime.fromisoformat(receipt["observed_at"])))
        for key, value in values.items():
            retained, _, _ = archive.event("kraken-ledger-entry:" + digest(key))
            original_receipt, original_raw, _, _ = archive.raw(
                retained["source"], "kraken", network
            )
            raw(
                archive,
                retained["source"],
                "kraken",
                network,
                "QueryLedgers",
                original_receipt["parameters"],
            )
            if (
                set(original_receipt["parameters"]) != {"id"}
                or key not in original_receipt["parameters"]["id"].split(",")
                or retained["id"] != key
                or canonical(retained["entry"]) != canonical(value)
                or canonical(result(original_raw).get(key)) != canonical(value)
            ):
                raise LedgerError("original Kraken debit entry revision conflict")
    if (
        canonical(ledgers) != canonical(proof["ledgers"])
        or max(clocks) != _utc(datetime.fromisoformat(proof["observed_at"]))
        or clocks != sorted(clocks)
    ):
        raise LedgerError("original Kraken debit proof content/clock differs")
    return proof


def original_fill(
    archive,
    intent,
    venue,
    network,
    market,
    native_id,
    trade,
    cost,
    quantity,
    price,
    charges,
):
    retained, _, _ = archive.event(venue + "-native-trade:" + digest(str(native_id)))
    original_source = retained["source_id"]
    if venue == "binance":
        receipt, payload, _, _ = archive.raw(original_source, venue, network)
        params = {
            k: v for k, v in receipt["parameters"].items() if k not in ("timestamp", "recvWindow")
        }
        raw(archive, original_source, venue, network, "myTrades", params)
        if (
            set(params) != {"symbol", "orderId", "fromId", "limit"}
            or params["symbol"] != market.symbol
            or params["orderId"] != trade["orderId"]
            or type(params["fromId"]) is not int
            or params["fromId"] > trade["id"]
            or params["fromId"] < 0
            or params["limit"] != 1000
        ):
            raise LedgerError("native Binance first trade query differs")
        if not isinstance(payload, list) or not any(
            canonical(item) == canonical(trade) for item in payload
        ):
            raise LedgerError("native Binance trade absent from first raw receipt")
        source_value = receipt
        source_at = _utc(datetime.fromisoformat(receipt["observed_at"]))
        at = exact_time(trade["time"], "unix-milliseconds")
    else:
        source_value = kraken_proof(archive, original_source, network)
        if (
            canonical(source_value["trade"]) != canonical(trade)
            or source_value["trade_id"] != native_id
        ):
            raise LedgerError("native Kraken trade first debit proof differs")
        source_at = _utc(datetime.fromisoformat(source_value["observed_at"]))
        at = exact_time(trade["time"], "unix-seconds-decimal")
    if digest(source_value) != retained["source_hash"]:
        raise LedgerError("native trade first source hash differs")
    fees, cash_fee = [], D(0)
    for asset, amount in charges:
        if amount < 0:
            raise LedgerError("unsupported_native_rebate")
        if not amount:
            continue
        if asset == market.quote_asset:
            cash_fee = exact_decimal_sum([cash_fee, amount])
        elif asset == market.base_asset:
            fees.append(
                InventoryFee(
                    instrument=market.instrument,
                    quantity=amount,
                    price=price,
                    valued_at=at,
                    available_at=source_at,
                    source_id=original_source,
                    source_hash=digest(source_value),
                    valuation_source_id=original_source,
                    valuation_source_hash=digest(source_value),
                )
            )
        else:
            raise LedgerError("unsupported_external_valuation_closure")
    fill = Fill(
        fill_id=venue + ":" + str(native_id),
        intent_id=intent.intent_id,
        account_id=archive.account_id,
        quantity=quantity,
        price=price,
        fee=cash_fee,
        executed_at=at,
        event_id=original_source,
        settlement=FillSettlement(
            gross_quote=cost, source_id=original_source, source_hash=digest(source_value)
        ),
        inventory_fees=tuple(fees),
    )
    if fill != Fill.model_validate(retained["fill"]):
        raise LedgerError(
            "retained native Fill differs from independently reconstructed raw economics"
        )
    economics = retained["economics"]
    expected_fees = (
        {"fee_asset": trade["commissionAsset"], "fee_quantity": str(D(trade["commission"]))}
        if venue == "binance"
        else {
            "fee_asset": market.quote_asset,
            "fee_quantity": "0",
            "asset_charges": [[asset, str(amount)] for asset, amount in sorted(charges)],
        }
    )
    if (
        canonical(economics["native"]) != canonical(trade)
        or economics["instrument"] != intent.instrument.model_dump(mode="json")
        or economics["side"] != intent.side
        or D(economics["quantity"]) != fill.quantity
        or D(economics["price"]) != fill.price
        or D(economics["cost"]) != cost
        or _utc(datetime.fromisoformat(economics["executed_at"])) != at
        or {
            k: economics[k]
            for k in ("fee_asset", "fee_quantity", "asset_charges")
            if k in economics
        }
        != expected_fees
    ):
        raise LedgerError("native first execution economics/identity differ")
    return fill


def first_trade(archive, venue, network, native_id, trade):
    from src.execution.kraken_venue import result

    probe = NativeArchive(archive.ledger, archive.account_id, as_of=archive.as_of)
    matches = []
    for row in archive.ledger.db.execute(
        "SELECT id,payload FROM events WHERE account=? AND id LIKE ? ORDER BY rowid",
        (archive.account_id, venue + "-http:%"),
    ):
        header = decode(row["payload"])
        path = "/api/v3/myTrades" if venue == "binance" else "/0/private/QueryTrades"
        if (
            not header.get("endpoint", "").endswith(path)
            or header.get("status") != 200
            or _utc(datetime.fromisoformat(header["observed_at"])) > archive.as_of
        ):
            continue
        receipt, payload, _, _ = probe.raw(row["id"], venue, network)
        params = {
            k: v for k, v in receipt["parameters"].items() if k not in ("timestamp", "recvWindow")
        }
        raw(
            probe,
            row["id"],
            venue,
            network,
            "myTrades" if venue == "binance" else "QueryTrades",
            params,
        )
        if venue == "binance":
            if (
                not isinstance(payload, list)
                or set(params) != {"symbol", "orderId", "fromId", "limit"}
                or type(params["fromId"]) is not int
                or params["fromId"] < 0
                or params["limit"] != 1000
            ):
                raise LedgerError("native Binance first execution request scope differs")
        elif set(params) != {"txid", "ledgers"} or params["ledgers"] is not True:
            raise LedgerError("native Kraken first execution request scope differs")
        candidates = (
            payload
            if venue == "binance"
            else [result(payload)[native_id]]
            if native_id in result(payload)
            else []
        )
        for candidate in candidates:
            same = (
                str(candidate.get("symbol")) + ":" + str(candidate.get("id")) == native_id
                if venue == "binance"
                else True
            )
            if same:
                if (
                    venue == "binance"
                    and (
                        params["symbol"] != trade["symbol"]
                        or params["orderId"] != trade["orderId"]
                        or params["fromId"] > trade["id"]
                    )
                ) or (venue == "kraken" and native_id not in params["txid"].split(",")):
                    raise LedgerError("native first receipt execution membership differs")
                if canonical(candidate) != canonical(trade):
                    raise LedgerError("native execution has a later/earlier conflicting revision")
                matches.append((row["id"], receipt))
    if not matches:
        raise LedgerError("native execution first raw receipt unavailable")
    receipt, _, reference, _ = archive.raw(matches[0][0], venue, network)
    return reference, _utc(datetime.fromisoformat(receipt["observed_at"]))


def binance_report(archive, report, intent, native_client):
    from src.execution.binance_venue import native_integer
    from src.execution.ccxt_transport import decimal

    bound, market, _, network = binding(archive, intent, "binance")
    content, _, _ = archive.event(report.evidence_id, content_addressed=True)
    if (
        not report.evidence_id.startswith("binance-order:")
        or content["intent_hash"] != intent.content_hash()
    ):
        raise LedgerError("native Binance original report/intent differs")
    order = content["order"]
    require_binance_order(intent, native_client, order, market)
    order_id = native_integer(order["orderId"])
    refs = content["references"]
    if not refs or len(refs) != len(set(refs)) or str(order_id) != report.venue_order_id:
        raise LedgerError("native Binance report order/source coverage differs")
    header, payload, _, _ = archive.raw(refs[0], "binance", network)
    params = {
        key: value
        for key, value in header["parameters"].items()
        if key not in ("timestamp", "recvWindow")
    }
    method = header["method"]
    if method == "GET":
        allowed = [{"symbol": market.symbol, "origClientOrderId": native_client}]
        for native_order_id in (order_id, str(order_id)):
            allowed.extend(
                [
                    {"symbol": market.symbol, "orderId": native_order_id},
                    {
                        "symbol": market.symbol,
                        "orderId": native_order_id,
                        "origClientOrderId": native_client,
                    },
                ]
            )
        if params not in allowed:
            raise LedgerError("native Binance original order query differs")
    elif method == "POST":
        from src.execution.native_submission_terms import submission_payload

        expected_params = submission_payload("binance", intent, native_client, symbol=market.symbol)
        if params != expected_params:
            raise LedgerError("native Binance original submission payload differs")
    else:
        raise LedgerError("native Binance original order source method differs")
    raw(archive, refs[0], "binance", network, "order", params, method=method)
    if canonical(payload) != canonical(order):
        raise LedgerError("native Binance original order raw content differs")
    observed = _utc(datetime.fromisoformat(header["observed_at"]))
    start, facts, seen = 0, [], set()
    cumulative = decimal(order["executedQty"])
    if bool(cumulative) != bool(refs[1:]):
        raise LedgerError("native Binance execution page coverage differs")
    for index, source in enumerate(refs[1:]):
        header, trades, _, _ = raw(
            archive,
            source,
            "binance",
            network,
            "myTrades",
            {"symbol": market.symbol, "orderId": order_id, "fromId": start, "limit": 1000},
        )
        at = _utc(datetime.fromisoformat(header["observed_at"]))
        if (
            not isinstance(trades, list)
            or len(trades) > 1000
            or at < observed
            or (index < len(refs) - 2) != (len(trades) == 1000)
        ):
            raise LedgerError("native Binance execution pagination/clock incomplete")
        observed, last = at, start - 1
        for trade in trades:
            identity = native_integer(trade["id"])
            if (
                identity in seen
                or identity <= last
                or identity < start
                or trade["orderId"] != order_id
                or trade["symbol"] != market.symbol
                or trade["isBuyer"] is not (intent.side == "buy")
            ):
                raise LedgerError("native Binance execution identity/side/order differs")
            seen.add(identity)
            last = identity
            native_id = market.symbol + ":" + str(identity)
            fill = original_fill(
                archive,
                intent,
                "binance",
                network,
                market,
                native_id,
                trade,
                decimal(trade["quoteQty"]),
                decimal(trade["qty"]),
                decimal(trade["price"]),
                [(trade["commissionAsset"], decimal(trade["commission"]))],
            )
            first, first_at = first_trade(archive, "binance", network, native_id, trade)
            key = NativeExecutionKey(
                venue="binance",
                network=network,
                broker_account_id=bound["broker_account_id"],
                instrument=intent.instrument,
                venue_order_id=str(order_id),
                native_execution_id=native_id,
            )
            facts.append((fill, key, first, first_at, str(trade["time"]), "unix-milliseconds"))
        start = last + 1
    fills = tuple(fact[0] for fact in facts)
    quantity = exact_decimal_sum(fill.quantity for fill in fills)
    cost = exact_decimal_sum(fill.settlement.gross_quote for fill in fills)
    status = {
        "NEW": "new",
        "PENDING_NEW": "new",
        "PARTIALLY_FILLED": "partial",
        "FILLED": "filled",
        "CANCELED": "cancelled",
        "EXPIRED": "expired",
        "EXPIRED_IN_MATCH": "expired",
        "REJECTED": "rejected",
    }.get(order.get("status"), "unknown")
    expected = VenueOrderReport(
        venue="binance",
        client_order_id=native_client,
        venue_order_id=str(order_id),
        observed_at=observed,
        status=status,
        cumulative_quantity=cumulative,
        fills=fills,
        executions_complete=quantity == cumulative and status != "unknown",
        evidence_id=report.evidence_id,
    )
    if (
        expected != report
        or quantity > cumulative
        or quantity == cumulative
        and cost != decimal(order["cummulativeQuoteQty"])
    ):
        raise LedgerError("native Binance report differs from complete raw economics")
    return facts


def kraken_order_sources(archive, refs, network, order_id, order, native_client):
    """Prove either the known ID query or both complete lost-ack cursor streams."""
    from src.execution.kraken_venue import result

    header, _, _, _ = archive.raw(refs[0], "kraken", network)
    if header["endpoint"].endswith("/QueryOrders"):
        params = header["parameters"]
        expected = {"txid": order_id, "trades": True, "consolidate_taker": False}
        if params not in (expected, {**expected, "cl_ord_id": native_client}):
            raise LedgerError("native Kraken original order query differs")
        header, payload, _, _ = raw(archive, refs[0], "kraken", network, "QueryOrders", params)
        if set(result(payload)) != {order_id} or canonical(result(payload)[order_id]) != canonical(
            order
        ):
            raise LedgerError("native Kraken original order raw content differs")
        return 1, _utc(datetime.fromisoformat(header["observed_at"]))
    cursor, matches, observed = 0, {}, archive.spec.opened_at
    if native_client is None:
        raise LedgerError("native Kraken recovery lacks original client identity")
    for endpoint, field in (("OpenOrders", "open"), ("ClosedOrders", "closed")):
        params = {
            "trades": True,
            "consolidate_taker": False,
            "with_cursor": True,
            "cl_ord_id": native_client,
        }
        seen = set()
        while True:
            if cursor >= len(refs):
                raise LedgerError("native Kraken original recovery cursor coverage incomplete")
            header, payload, _, _ = raw(archive, refs[cursor], "kraken", network, endpoint, params)
            cursor += 1
            at = _utc(datetime.fromisoformat(header["observed_at"]))
            page = result(payload)
            if (
                at < observed
                or not isinstance(page.get(field), dict)
                or not isinstance(page.get("cursor"), dict)
                or "next" not in page["cursor"]
            ):
                raise LedgerError("native Kraken original recovery cursor/clock differs")
            observed = at
            for key, value in page[field].items():
                if value.get("cl_ord_id") != native_client or (
                    key in matches and canonical(matches[key]) != canonical(value)
                ):
                    raise LedgerError("native Kraken original recovery changed identity/economics")
                matches[key] = value
            token = page["cursor"]["next"]
            if token is None:
                break
            if not isinstance(token, str) or not token or token in seen:
                raise LedgerError("native Kraken original recovery repeated/invalid cursor")
            seen.add(token)
            params = {**params, "cursor": token}
    if set(matches) != {order_id} or canonical(matches[order_id]) != canonical(order):
        raise LedgerError("native Kraken recovery does not prove one exact economic order")
    return cursor, observed


def kraken_report(archive, report, intent, native_client):
    from src.execution.ccxt_transport import decimal
    from src.execution.kraken_venue import KrakenVenue, result

    bound, market, _, network = binding(archive, intent, "kraken")
    content, _, _ = archive.event(report.evidence_id, content_addressed=True)
    if (
        not report.evidence_id.startswith("kraken-order:")
        or content["intent_hash"] != intent.content_hash()
    ):
        raise LedgerError("native Kraken original report/intent differs")
    order, order_id, refs = content["order"], content["order_id"], content["references"]
    require_kraken_order(intent, native_client, order, market)
    if not refs or len(refs) != len(set(refs)) or order_id != report.venue_order_id:
        raise LedgerError("native Kraken report order/source coverage differs")
    cursor, observed = kraken_order_sources(archive, refs, network, order_id, order, native_client)
    ids = order.get("trades", [])
    if (
        not isinstance(ids, list)
        or len(ids) != len(set(ids))
        or any(not isinstance(key, str) or not key for key in ids)
    ):
        raise LedgerError("native Kraken original trade IDs invalid")
    facts, fees, all_trades = [], [], True
    for start in range(0, len(ids), 20):
        requested = ids[start : start + 20]
        if cursor >= len(refs):
            raise LedgerError("native Kraken execution pages incomplete")
        trade_source = refs[cursor]
        cursor += 1
        header, payload, _, _ = raw(
            archive,
            trade_source,
            "kraken",
            network,
            "QueryTrades",
            {"txid": ",".join(requested), "ledgers": True},
        )
        if set(result(payload)) - set(requested):
            raise LedgerError("native Kraken execution page has unrequested identities")
        trade_at = _utc(datetime.fromisoformat(header["observed_at"]))
        if trade_at < observed:
            raise LedgerError("native Kraken execution page clock regressed")
        observed = trade_at
        all_trades &= set(result(payload)) == set(requested)
        for key in requested:
            if key not in result(payload):
                continue
            trade = result(payload)[key]
            if (
                trade["ordertxid"] != order_id
                or trade["pair"] != market.native_id
                or trade["type"] != intent.side
                or decimal(trade["margin"])
                or trade["ordertype"]
                not in (
                    ("market", order["descr"]["ordertype"])
                    if intent.order_type in ("stop", "trailing_stop")
                    else (order["descr"]["ordertype"],)
                )
            ):
                raise LedgerError("native Kraken execution original identity differs")
            if cursor >= len(refs):
                raise LedgerError("native Kraken debit source coverage incomplete")
            proof_source = refs[cursor]
            cursor += 1
            proof = kraken_proof(archive, proof_source, network)
            if (
                proof["trade_source"] != trade_source
                or proof["trade_id"] != key
                or canonical(proof["trade"]) != canonical(trade)
            ):
                raise LedgerError("native Kraken report/debit source membership differs")
            proof_at = _utc(datetime.fromisoformat(proof["observed_at"]))
            if proof_at < observed:
                raise LedgerError("native Kraken execution proof clock regressed")
            observed = proof_at
            charges = KrakenVenue._settlement(market, key, trade, proof["ledgers"])
            fill = original_fill(
                archive,
                intent,
                "kraken",
                network,
                market,
                key,
                trade,
                decimal(trade["cost"]),
                decimal(trade["vol"]),
                decimal(trade["price"]),
                charges,
            )
            first, first_at = first_trade(archive, "kraken", network, key, trade)
            identity = NativeExecutionKey(
                venue="kraken",
                network=network,
                broker_account_id=bound["broker_account_id"],
                instrument=intent.instrument,
                venue_order_id=order_id,
                native_execution_id=key,
            )
            facts.append(
                (fill, identity, first, first_at, str(trade["time"]), "unix-seconds-decimal")
            )
            fees.append(decimal(trade["fee"]))
    if cursor != len(refs):
        raise LedgerError("native Kraken report has unexpected source pages")
    fills = tuple(fact[0] for fact in facts)
    quantity = exact_decimal_sum(fill.quantity for fill in fills)
    cumulative = decimal(order["vol_exec"])
    complete = all_trades and quantity == cumulative
    status = {
        "pending": "new",
        "open": "partial" if cumulative else "new",
        "closed": "filled",
        "canceled": "cancelled",
        "expired": "expired",
    }.get(order["status"], "unknown")
    expected = VenueOrderReport(
        venue="kraken",
        client_order_id=report.client_order_id,
        venue_order_id=order_id,
        observed_at=observed,
        status=status,
        cumulative_quantity=cumulative,
        fills=fills,
        executions_complete=complete and status != "unknown",
        evidence_id=report.evidence_id,
    )
    if (
        expected != report
        or quantity > cumulative
        or complete
        and (
            exact_decimal_sum(fill.settlement.gross_quote for fill in fills)
            != decimal(order["cost"])
            or exact_decimal_sum(fees) != decimal(order["fee"])
        )
    ):
        raise LedgerError("native Kraken report differs from complete raw economics")
    return facts
