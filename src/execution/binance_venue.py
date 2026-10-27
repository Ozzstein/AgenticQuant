"""Actual Binance Spot account/order/trade protocol over durable CCXT signing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from src.domain.portfolio import exact_decimal_product, exact_decimal_sum
from src.execution.ccxt_transport import canonical, decimal, digest
from src.execution.ledger import LedgerError
from src.execution.spot_venue import CCXTSpotVenue, bounded
from src.execution.venue import (
    DefinitiveRejection,
    VenueBalances,
    VenueNotionalReference,
    VenueOrderReport,
)


def native_integer(value):
    if type(value) is not int or value < 0:
        raise LedgerError("native Binance identity/time requires a nonnegative JSON integer")
    return value


def native_time(value):
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=native_integer(value))


class BinanceVenue(CCXTSpotVenue):
    venue = "binance"

    def __init__(self, *args, notional_reference=None, **kwargs):
        # Resolver signature: (Instrument, cutoff, native_average_minutes) ->
        # VenueNotionalReference. Bind the data-owned authenticated store parser.
        self._notional_reference = notional_reference
        super().__init__(*args, **kwargs)

    def _request(
        self, path, *, api="private", method="GET", params=None, missing=False, submitting=False
    ):
        value, status, observed, evidence = self.transport.request(
            path, api=api, method=method, params=params
        )
        code = value.get("code") if isinstance(value, dict) else None
        if status == 200 and (code is None or code == 0):
            return value, observed, evidence
        if missing and code == -2013 and status in (400, 404):
            return None, observed, evidence
        # These explicit validation/auth rejection codes do not mean acceptance.
        # Matching-engine -2010 can include duplicate client identity; preserve
        # uncertainty and query it, never automatically retry an order.
        definite = {
            -1013,
            -1100,
            -1101,
            -1102,
            -1103,
            -1111,
            -1116,
            -1117,
            -1118,
            -1121,
            -1130,
            -2014,
            -2015,
        }
        if submitting and status == 400 and code in definite:
            raise DefinitiveRejection("Binance rejected native order; receipt " + evidence)
        raise LedgerError("Binance native request unresolved/rejected; receipt " + evidence)

    @bounded
    def capture(self):
        payload, sources = {}, {}
        for role, path, api, parameters in self._capture_requests():
            value, _, evidence = self._request(path, api=api, params=parameters)
            payload[role], sources[role] = value, evidence
        return self._save_capture(payload, sources)

    def _capture_requests(self):
        yield "account", "account", "private", {}
        yield "orders", "openOrders", "private", {}
        for key, market in sorted(self.markets.items()):
            yield "market:" + key, "exchangeInfo", "public", {"symbol": market.symbol}
            yield "filters:" + key, "myFilters", "private", {"symbol": market.symbol}

    def _require_capture_source(self, role, receipt):
        expected = {item[0]: item[1:] for item in self._capture_requests()}
        if role not in expected:
            raise LedgerError("unknown native Binance capture role")
        path, api, params = expected[role]
        actual = {
            key: value
            for key, value in receipt["parameters"].items()
            if key not in ("timestamp", "recvWindow")
        }
        if (
            receipt["api"] != api
            or receipt["method"] != "GET"
            or not receipt["endpoint"].endswith("/api/v3/" + path)
            or actual != params
        ):
            raise LedgerError("Binance capture endpoint/parameters differ from source scope")

    def _balances(self, capture, identity):
        payload = capture["payload"]
        if set(payload) != {item[0] for item in self._capture_requests()}:
            raise LedgerError("Binance exact market/account capture coverage required")
        state = payload["account"]
        if (
            str(native_integer(state.get("uid"))) != self.broker_account_id
            or state.get("accountType") != "SPOT"
            or state.get("canTrade") is not True
        ):
            raise LedgerError("native Binance account identity/spot permission differs")
        quantities, cash, free, seen = {}, D(0), D(0), set()
        native = {m.base_asset: key for key, m in self.markets.items()}
        for row in state["balances"]:
            asset = row["asset"]
            if asset in seen:
                raise LedgerError("duplicate native balance asset")
            seen.add(asset)
            available, locked = decimal(row["free"]), decimal(row["locked"])
            if min(available, locked) < 0:
                raise LedgerError("native spot borrowing/negative inventory is unsupported")
            total = exact_decimal_sum([available, locked])
            if asset == self.cash_asset:
                cash, free = total, available
            elif asset in native:
                quantities[native[asset]] = total
            elif total:
                raise LedgerError("unmapped native balance requires explicit audited migration")
        if self.cash_asset not in seen or not isinstance(payload["orders"], list):
            raise LedgerError("native cash/open-order coverage unavailable")
        for key in self.markets:
            self._symbol(capture, key)
        return VenueBalances(
            venue=self.venue,
            observed_at=capture["observed_at"],
            cash_base=cash,
            available_cash_base=free,
            quantities=quantities,
            open_client_order_ids=tuple(
                row.get("clientOrderId") or "unowned:" + str(row["orderId"])
                for row in payload["orders"]
            ),
            evidence_id=identity,
        )

    def _symbol(self, capture, key):
        market = self.markets[key]
        response = capture["payload"]["market:" + key]
        if not isinstance(response.get("symbols"), list) or len(response["symbols"]) != 1:
            raise LedgerError("native Binance symbol coverage differs")
        symbol = response["symbols"][0]
        if (
            symbol.get("symbol") != market.symbol
            or market.native_id != market.symbol
            or symbol.get("baseAsset") != market.base_asset
            or symbol.get("quoteAsset") != market.quote_asset
            or symbol.get("status") != "TRADING"
            or symbol.get("isSpotTradingAllowed") is not True
        ):
            raise LedgerError("native Binance symbol identity/status differs")
        filters = capture["payload"]["filters:" + key]
        if any(
            not isinstance(filters.get(kind), list)
            for kind in ("symbolFilters", "exchangeFilters", "assetFilters")
        ):
            raise LedgerError("native account-relevant filter coverage incomplete")
        return symbol

    def rules(self, instrument, as_of):
        self._market(instrument)
        capture, _ = self._capture(as_of)
        self._balances(capture, self.venue + "-capture:" + digest(capture))
        return self._rules[instrument.instrument_id]

    def _reference(self, instrument, as_of, minutes):
        if type(minutes) is not int or minutes < 0 or self._notional_reference is None:
            raise LedgerError(
                "native filter requires authenticated reference/absence/window evidence"
            )
        source = VenueNotionalReference.model_validate(
            self._notional_reference(instrument, as_of, minutes).model_dump()
        )
        if (
            source.available_at > as_of
            or not 0 <= (as_of - source.observed_at).total_seconds() <= 30
            or (source.kind != "venue-reference" and source.average_minutes != minutes)
        ):
            raise LedgerError("native reference age/availability/window differs")
        return source.price

    def _check_order(self, intent, as_of):
        if intent.order_type not in ("market", "limit", "stop"):
            raise LedgerError("native Binance trailing order support is not yet implemented")
        market = self._market(intent.instrument)
        capture, _ = self._capture(as_of)
        symbol = self._symbol(capture, intent.instrument.instrument_id)
        native_type = {"market": "MARKET", "limit": "LIMIT", "stop": "STOP_LOSS"}[intent.order_type]
        if (
            native_type not in symbol["orderTypes"]
            or intent.tif == "DAY"
            or (intent.order_type != "limit" and intent.tif != "GTC")
        ):
            raise LedgerError("native Binance order type/time-in-force unsupported")
        if intent.expires_at is not None:
            raise LedgerError(
                "Binance spot has no native expiry for this order; use a proved cancellation scheduler"
            )
        filters = capture["payload"]["filters:" + intent.instrument.instrument_id]
        orders = capture["payload"]["orders"]
        # Retain both exchangeInfo and account-relevant predicates. Redundant
        # predicates are harmless; different windows remain separate checks.
        all_filters = [
            *symbol["filters"],
            *capture["payload"]["market:" + intent.instrument.instrument_id].get(
                "exchangeFilters", []
            ),
            *filters["symbolFilters"],
            *filters["exchangeFilters"],
            *filters["assetFilters"],
        ]
        for native in all_filters:
            kind = native.get("filterType")
            price = intent.limit_price if intent.order_type == "limit" else intent.stop_price
            if kind in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                if kind == "MARKET_LOT_SIZE" and intent.order_type != "market":
                    continue
                minimum, maximum, step = map(
                    decimal, (native["minQty"], native["maxQty"], native["stepSize"])
                )
                if (
                    intent.quantity < minimum
                    or (maximum and intent.quantity > maximum)
                    or (step and intent.quantity % step)
                ):
                    raise LedgerError("native Binance lot quantity constraint failed")
            elif kind == "PRICE_FILTER":
                if price is not None:
                    minimum, maximum, tick = map(
                        decimal, (native["minPrice"], native["maxPrice"], native["tickSize"])
                    )
                    if (
                        (minimum and price < minimum)
                        or (maximum and price > maximum)
                        or (tick and price % tick)
                    ):
                        raise LedgerError("native Binance price constraint failed")
            elif kind in ("MIN_NOTIONAL", "NOTIONAL"):
                market_order = intent.order_type != "limit"
                min_active = (
                    not market_order
                    or native["applyToMarket" if kind == "MIN_NOTIONAL" else "applyMinToMarket"]
                    is True
                )
                max_active = kind == "NOTIONAL" and (
                    not market_order or native["applyMaxToMarket"] is True
                )
                if not min_active and not max_active:
                    continue
                reference = (
                    self._reference(intent.instrument, as_of, native["avgPriceMins"])
                    if market_order
                    else price
                )
                notional = exact_decimal_product(intent.quantity, reference)
                if (min_active and notional < decimal(native["minNotional"])) or (
                    max_active and notional > decimal(native["maxNotional"])
                ):
                    raise LedgerError("native Binance notional constraint failed")
            elif kind in ("PERCENT_PRICE", "PERCENT_PRICE_BY_SIDE"):
                if intent.order_type == "limit":
                    reference = self._reference(intent.instrument, as_of, native["avgPriceMins"])
                    prefix = (
                        "multiplier"
                        if kind == "PERCENT_PRICE"
                        else "bidMultiplier"
                        if intent.side == "buy"
                        else "askMultiplier"
                    )
                    if (
                        not exact_decimal_product(reference, decimal(native[prefix + "Down"]))
                        <= price
                        <= exact_decimal_product(reference, decimal(native[prefix + "Up"]))
                    ):
                        raise LedgerError("native Binance percent-price constraint failed")
            elif kind in (
                "MAX_NUM_ORDERS",
                "EXCHANGE_MAX_NUM_ORDERS",
                "MAX_NUM_ALGO_ORDERS",
                "EXCHANGE_MAX_NUM_ALGO_ORDERS",
            ):
                scoped = [
                    row
                    for row in orders
                    if kind.startswith("EXCHANGE_") or row["symbol"] == market.symbol
                ]
                if "ALGO" in kind:
                    if intent.order_type != "stop":
                        continue
                    scoped = [
                        row
                        for row in scoped
                        if row["type"]
                        in ("STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT")
                    ]
                count_field = "maxNumAlgoOrders" if "ALGO" in kind else "maxNumOrders"
                if len(scoped) >= native_integer(native[count_field]):
                    raise LedgerError("native Binance open-order count limit reached")
            elif kind == "MAX_POSITION":
                if intent.side == "buy":
                    held = self._balances(
                        capture, self.venue + "-capture:" + digest(capture)
                    ).quantities.get(intent.instrument.instrument_id, D(0))
                    pending = [
                        decimal(row["origQty"]) - decimal(row["executedQty"])
                        for row in orders
                        if row["symbol"] == market.symbol and row["side"] == "BUY"
                    ]
                    if exact_decimal_sum([held, intent.quantity, *pending]) > decimal(
                        native["maxPosition"]
                    ):
                        raise LedgerError("native Binance maximum position violated")
            elif kind == "MAX_ASSET":
                if native["asset"] == market.base_asset:
                    amount = intent.quantity
                elif native["asset"] == market.quote_asset:
                    if intent.order_type != "limit":
                        raise LedgerError(
                            "native market MAX_ASSET quote sizing requires explicit supported policy"
                        )
                    amount = exact_decimal_product(intent.quantity, price)
                else:
                    raise LedgerError("native MAX_ASSET scope differs from actual market")
                if amount > decimal(native["limit"]):
                    raise LedgerError("native Binance maximum asset quantity violated")
            elif kind in (
                "ICEBERG_PARTS",
                "MAX_NUM_ICEBERG_ORDERS",
                "EXCHANGE_MAX_NUM_ICEBERG_ORDERS",
                "TRAILING_DELTA",
                "MAX_NUM_ORDER_AMENDS",
                "MAX_NUM_ORDER_LISTS",
                "EXCHANGE_MAX_NUM_ORDER_LISTS",
            ):
                # This endpoint emits plain individual orders with no iceberg,
                # trailing delta, amendment or order-list fields.
                continue
            else:
                raise LedgerError("unrecognized native Binance filter: " + str(kind))

    @bounded
    def submit(self, intent, client_order_id, rules, as_of):
        self._claim(intent, client_order_id, rules, as_of)
        params = {
            "symbol": self._market(intent.instrument).symbol,
            "side": intent.side.upper(),
            "type": {"market": "MARKET", "limit": "LIMIT", "stop": "STOP_LOSS"}[intent.order_type],
            "quantity": str(intent.quantity),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        if intent.order_type == "limit":
            params.update(price=str(intent.limit_price), timeInForce=intent.tif)
        if intent.order_type == "stop":
            params["stopPrice"] = str(intent.stop_price)
        order, observed, reference = self._request(
            "order", method="POST", params=params, submitting=True
        )
        return self._report(intent, client_order_id, order, observed, [reference])

    @bounded
    def lookup(self, intent, client_order_id, venue_order_id, as_of):
        params = {
            "symbol": self._market(intent.instrument).symbol,
            "origClientOrderId": client_order_id,
        }
        if venue_order_id is not None:
            params["orderId"] = venue_order_id
        order, observed, reference = self._request("order", params=params, missing=True)
        if order is None:
            return None
        if venue_order_id is not None and str(order["orderId"]) != venue_order_id:
            raise LedgerError("native order lookup returned another venue identity")
        return self._report(intent, client_order_id, order, observed, [reference])

    @bounded
    def cancel(self, intent, client_order_id, venue_order_id, as_of):
        self._claim(
            intent, client_order_id, None, as_of, cancel=True, venue_order_id=venue_order_id
        )
        self._request(
            "order",
            method="DELETE",
            params={
                "symbol": self._market(intent.instrument).symbol,
                "orderId": venue_order_id,
                "origClientOrderId": client_order_id,
            },
        )
        report = self.lookup(intent, client_order_id, venue_order_id, as_of)
        if report is None:
            raise LedgerError("native cancellation remains ambiguous; retain reservations")
        return report

    @bounded
    def read_native_order(self, intent, venue_order_id):
        """Read exact native history; its external owner proves the ID binding."""
        order, observed, reference = self._request(
            "order",
            params={"symbol": self._market(intent.instrument).symbol, "orderId": venue_order_id},
            missing=True,
        )
        if order is None:
            return None
        if str(order.get("orderId")) != venue_order_id or not order.get("clientOrderId"):
            raise LedgerError("native Binance order identity is missing/different")
        return self._report(intent, order["clientOrderId"], order, observed, [reference])

    def _report(self, intent, client_order_id, order, observed, references):
        market = self._market(intent.instrument)
        order_id = native_integer(order.get("orderId"))
        if (
            order.get("symbol") != market.symbol
            or order.get("clientOrderId") != client_order_id
            or order.get("side") != intent.side.upper()
            or order.get("type")
            != {"market": "MARKET", "limit": "LIMIT", "stop": "STOP_LOSS"}[intent.order_type]
            or decimal(order["origQty"]) != intent.quantity
            or order.get("orderListId", -1) != -1
        ):
            raise LedgerError("native Binance order differs from exact submitted intent")
        if (
            order.get("timeInForce") != intent.tif
            or (intent.order_type == "limit" and decimal(order["price"]) != intent.limit_price)
            or (intent.order_type == "stop" and decimal(order["stopPrice"]) != intent.stop_price)
        ):
            raise LedgerError("native Binance order price/time-in-force differs")
        cumulative = decimal(order["executedQty"])
        fills, seen = [], set()
        if cumulative:
            start = 0
            while True:
                trades, observed, reference = self._request(
                    "myTrades",
                    params={
                        "symbol": market.symbol,
                        "orderId": order_id,
                        "fromId": start,
                        "limit": 1000,
                    },
                )
                references.append(reference)
                if not isinstance(trades, list) or len(trades) > 1000:
                    raise LedgerError("invalid native trade page")
                last = start - 1
                for trade in trades:
                    identity = native_integer(trade["id"])
                    if (
                        identity in seen
                        or identity < start
                        or identity <= last
                        or trade.get("orderId") != order_id
                        or trade.get("symbol") != market.symbol
                        or trade.get("isBuyer") is not (intent.side == "buy")
                    ):
                        raise LedgerError("native trade identity/pagination/side differs")
                    seen.add(identity)
                    last = identity
                    fills.append(
                        self._fill(
                            intent,
                            native_id=market.symbol + ":" + str(identity),
                            native_payload=trade,
                            quantity=trade["qty"],
                            price=trade["price"],
                            cost=trade["quoteQty"],
                            fee_asset=trade["commissionAsset"],
                            fee_quantity=trade["commission"],
                            executed_at=native_time(trade["time"]),
                            observed_at=observed,
                            evidence_id=reference,
                        )
                    )
                if len(trades) < 1000:
                    break
                start = last + 1
        quantity = exact_decimal_sum(fill.quantity for fill in fills)
        cost = exact_decimal_sum(fill.settlement.gross_quote for fill in fills)
        if quantity > cumulative:
            raise LedgerError(
                "native trades advanced beyond order acknowledgement; refresh exact order"
            )
        if quantity == cumulative and cost != decimal(order["cummulativeQuoteQty"]):
            raise LedgerError("native individual trade costs do not reconcile to order settlement")
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
        content = {"order": order, "references": references, "intent_hash": intent.content_hash()}
        identity = self.venue + "-order:" + digest(content)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, identity, canonical(content))
        return VenueOrderReport(
            venue=self.venue,
            client_order_id=client_order_id,
            venue_order_id=str(order_id),
            observed_at=observed,
            status=status,
            cumulative_quantity=cumulative,
            fills=tuple(fills),
            executions_complete=quantity == cumulative and status != "unknown",
            evidence_id=identity,
        )
