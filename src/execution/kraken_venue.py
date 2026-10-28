"""Native Kraken spot orders with actual trade/ledger settlement proof.

Kraken spot has no verified sandbox. Construction requires explicit live
authorization; tests inject offline HTTP transport and never contact a venue.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING
from decimal import Decimal as D

from src.domain.portfolio import exact_decimal_product, exact_decimal_sum
from src.execution.ccxt_transport import canonical, decimal, digest
from src.execution.ledger import LedgerError
from src.execution.spot_venue import CCXTSpotVenue, bounded
from src.execution.venue import DefinitiveRejection, VenueBalances, VenueOrderReport


def native_time(value):
    seconds = decimal(value)
    if seconds < 0:
        raise LedgerError("negative native Kraken event time")
    # Preserve the exact wire time in raw receipts; project to the first
    # representable microsecond, never an earlier fill/availability boundary.
    micros = exact_decimal_product(seconds, D(1000000)).to_integral_value(rounding=ROUND_CEILING)
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(micros))


def result(value):
    if (
        not isinstance(value, dict)
        or value.get("error") != []
        or not isinstance(value.get("result"), dict)
    ):
        raise LedgerError("complete successful native Kraken response required")
    return value["result"]


def order_type(intent):
    return {
        "market": "market",
        "limit": "limit",
        "stop": "stop-loss",
        "trailing_stop": "trailing-stop",
    }[intent.order_type]


def trailing_price(intent):
    value = intent.trail_percent if intent.trail_percent is not None else intent.trail_price
    if intent.order_type != "trailing_stop" or value is None:
        raise LedgerError("native Kraken trailing order requires exact immutable trail terms")
    return "+" + format(value, "f") + ("%" if intent.trail_percent is not None else "")


def require_trailing_price(intent, value):
    # A number alone cannot distinguish a quote offset from a percent trail.
    if not isinstance(value, str) or re.fullmatch(r"\+[0-9]+(?:\.[0-9]+)?%?", value) is None:
        raise LedgerError("native Kraken trailing price requires its explicit relative basis")
    percentage = value.endswith("%")
    expected = intent.trail_percent if percentage else intent.trail_price
    if expected is None or decimal(value[1:-1] if percentage else value[1:]) != expected:
        raise LedgerError("native Kraken trailing price differs from immutable intent")


class KrakenVenue(CCXTSpotVenue):
    venue = "kraken"

    def _request(self, path, *, api="private", params=None, submitting=False, missing=False):
        value, status, at, evidence = self.transport.request(path, api=api, params=params)
        if status == 200 and isinstance(value, dict) and value.get("error") == []:
            result(value)
            return value, at, evidence
        errors = value.get("error") if isinstance(value, dict) else None
        if missing and status == 200 and errors == ["EOrder:Unknown order"]:
            return None, at, evidence
        from src.execution.native_submission_terms import definitive_rejection

        if submitting and definitive_rejection(self.venue, status, value):
            raise DefinitiveRejection("Kraken rejected native order; receipt " + evidence)
        raise LedgerError("Kraken native response unresolved/rejected; receipt " + evidence)

    def _static_captures(self):
        yield "wallets", "ListWalletAccounts", "private", {}
        yield "balances", "BalanceEx", "private", {}
        assets = sorted({self.cash_asset, *(market.base_asset for market in self.markets.values())})
        yield "assets", "Assets", "public", {"asset": ",".join(assets)}
        for key, market in sorted(self.markets.items()):
            yield "market:" + key, "AssetPairs", "public", {"pair": market.native_id}
            yield "reference:" + key, "Trades", "public", {"pair": market.native_id, "count": 1}

    def _order_pages(self, path, *, client_id=None):
        params = {"trades": True, "consolidate_taker": False, "with_cursor": True}
        if client_id is not None:
            params["cl_ord_id"] = client_id
        pages, sources, seen = [], [], set()
        kind = "open" if path == "OpenOrders" else "closed"
        while True:
            value, at, source = self._request(path, params=params)
            page = result(value)
            if (
                not isinstance(page.get(kind), dict)
                or not isinstance(page.get("cursor"), dict)
                or "next" not in page["cursor"]
            ):
                raise LedgerError("native Kraken order page/cursor coverage incomplete")
            pages.append(value)
            sources.append(source)
            cursor = page["cursor"]["next"]
            if cursor is None:
                break
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise LedgerError("native Kraken cursor repeated or invalid")
            seen.add(cursor)
            params["cursor"] = cursor
        return pages, sources, at

    @bounded
    def capture(self):
        payload, sources = {}, {}
        for role, path, api, params in self._static_captures():
            value, _, source = self._request(path, api=api, params=params)
            payload[role], sources[role] = value, source
        pages, refs, _ = self._order_pages("OpenOrders")
        for index, (page, source) in enumerate(zip(pages, refs, strict=True)):
            payload["orders:" + str(index)], sources["orders:" + str(index)] = page, source
        return self._save_capture(payload, sources)

    def _require_capture_source(self, role, receipt):
        expected = {item[0]: item[1:] for item in self._static_captures()}
        if role.startswith("orders:"):
            path, api, params = (
                "OpenOrders",
                "private",
                {"trades": True, "consolidate_taker": False, "with_cursor": True},
            )
            actual = {key: val for key, val in receipt["parameters"].items() if key != "cursor"}
        elif role in expected:
            path, api, params = expected[role]
            actual = receipt["parameters"]
        else:
            raise LedgerError("unknown native Kraken capture role")
        self._require_endpoint(receipt, path, api)
        if actual != params:
            raise LedgerError("native Kraken capture parameters differ")

    @staticmethod
    def _require_endpoint(receipt, path, api="private"):
        if (
            receipt["endpoint"] != f"https://api.kraken.com/0/{api}/{path}"
            or receipt["api"] != api
            or receipt["method"] != ("POST" if api == "private" else "GET")
        ):
            raise LedgerError("native Kraken source endpoint/method differs")

    def _open_orders(self, capture):
        expected = {row[0] for row in self._static_captures()}
        pages = len(capture["payload"]) - len(expected)
        if pages < 1 or set(capture["payload"]) != expected | {
            "orders:" + str(i) for i in range(pages)
        }:
            raise LedgerError("native Kraken exact capture coverage required")
        orders, next_cursor, seen = {}, None, set()
        for index in range(pages):
            role = "orders:" + str(index)
            receipt, _ = self._source(capture["sources"][role])
            if receipt["parameters"].get("cursor") != next_cursor:
                raise LedgerError("native Kraken archived pagination chain changed")
            page = result(capture["payload"][role])
            if not isinstance(page.get("open"), dict) or set(orders) & set(page["open"]):
                raise LedgerError("native Kraken duplicate/invalid open order page")
            orders.update(page["open"])
            next_cursor = page["cursor"]["next"]
            if next_cursor is not None and (
                not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen
            ):
                raise LedgerError("native Kraken repeated cursor")
            seen.add(next_cursor)
            if (next_cursor is None) != (index == pages - 1):
                raise LedgerError("native Kraken incomplete/extra open-order pages")
        return orders

    def _balances(self, capture, identity):
        payload = capture["payload"]
        wallets = result(payload["wallets"])
        main = [row for row in wallets["accounts"] if row["type"] == "main"]
        if (
            len(main) != 1
            or main[0]["account_id"] != self.broker_account_id
            or main[0]["status"] != "active"
            or main[0]["flags"].get("active") is not True
            or wallets["cursor"]["next"] is not None
        ):
            raise LedgerError("native Kraken actual default main-wallet identity/coverage differs")
        # All native calls intentionally use that verified default main wallet;
        # subwallet selection cannot be inferred from a caller account label.
        assets = result(payload["assets"])
        if (
            assets[self.cash_asset].get("altname")
            != self.ledger.spec(self.account_id).base_currency
        ):
            raise LedgerError("native Kraken cash currency alias differs from canonical currency")
        native = {m.base_asset: key for key, m in self.markets.items()}
        quantities, cash, free = {}, D(0), D(0)
        balances = result(payload["balances"])
        for asset, row in balances.items():
            total, held = decimal(row["balance"]), decimal(row["hold_trade"])
            if (
                decimal(row.get("credit", 0))
                or decimal(row.get("credit_used", 0))
                or min(total, held) < 0
                or held > total
            ):
                raise LedgerError("native Kraken borrowing/negative spot funds unsupported")
            if asset == self.cash_asset:
                cash, free = total, exact_decimal_sum([total, held.copy_negate()])
            elif asset in native:
                quantities[native[asset]] = total
            elif total or held:
                raise LedgerError(
                    "unmapped Kraken asset/earn balance requires explicit reconciliation"
                )
        if self.cash_asset not in balances:
            raise LedgerError("native Kraken cash balance unavailable")
        for key in self.markets:
            self._pair(capture, key)
        orders = self._open_orders(capture)
        return VenueBalances(
            venue=self.venue,
            observed_at=capture["observed_at"],
            cash_base=cash,
            available_cash_base=free,
            quantities=quantities,
            open_client_order_ids=tuple(
                row.get("cl_ord_id") or "unowned:" + key for key, row in orders.items()
            ),
            evidence_id=identity,
        )

    def _pair(self, capture, key):
        market = self.markets[key]
        pairs = result(capture["payload"]["market:" + key])
        if set(pairs) != {market.native_id}:
            raise LedgerError("native Kraken pair coverage differs")
        pair = pairs[market.native_id]
        if (
            pair.get("base") != market.base_asset
            or pair.get("quote") != market.quote_asset
            or pair.get("altname") != market.symbol
            or pair.get("aclass_base") != "currency"
            or pair.get("aclass_quote") != "currency"
            or pair.get("lot") != "unit"
            or decimal(pair["lot_multiplier"]) != 1
        ):
            raise LedgerError("native Kraken spot pair identity/lot differs")
        return pair

    def rules(self, instrument, as_of):
        self._market(instrument)
        capture, identity = self._capture(as_of)
        self._balances(capture, identity)
        return self._rules[instrument.instrument_id]

    def _check_order(self, intent, as_of):
        capture, _ = self._capture(as_of)
        pair = self._pair(capture, intent.instrument.instrument_id)
        if pair["status"] != "online" and not (
            pair["status"] == "limit_only" and intent.order_type == "limit"
        ):
            raise LedgerError("native Kraken trading status disallows order")
        digits = pair["lot_decimals"]
        if (
            type(digits) is not int
            or not 0 <= digits <= 18
            or intent.quantity % D(1).scaleb(-digits)
            or intent.quantity < decimal(pair["ordermin"])
        ):
            raise LedgerError("native Kraken lot/minimum quantity violated")
        price = intent.limit_price if intent.order_type == "limit" else intent.stop_price
        if price is not None and price % decimal(pair["tick_size"]):
            raise LedgerError("native Kraken tick size violated")
        if intent.trail_price is not None and intent.trail_price % decimal(pair["tick_size"]):
            raise LedgerError("native Kraken trailing offset tick size violated")
        if intent.tif == "DAY" or intent.expires_at is not None:
            raise LedgerError(
                "Kraken native GTD expiry requires an explicit supported temporal mapping"
            )
        if intent.order_type in ("stop", "trailing_stop") and intent.tif != "GTC":
            raise LedgerError("conditional Kraken order requires GTC")
        if intent.order_type != "limit":
            market = self.markets[intent.instrument.instrument_id]
            native = result(capture["payload"]["reference:" + intent.instrument.instrument_id])
            if set(native) != {market.native_id, "last"} or len(native[market.native_id]) != 1:
                raise LedgerError("native Kraken individual last-trade evidence incomplete")
            trade = native[market.native_id][0]
            if not 0 <= (as_of - native_time(trade[2])).total_seconds() <= 30:
                raise LedgerError("native Kraken last trade is stale/future")
            price = decimal(trade[0])
        if exact_decimal_product(intent.quantity, price) < decimal(pair["costmin"]):
            raise LedgerError("native Kraken minimum cost violated")

    @bounded
    def submit(self, intent, client_order_id, rules, as_of):
        self._claim(intent, client_order_id, rules, as_of)
        from src.execution.native_submission_terms import submission_payload

        market = self._market(intent.instrument)
        params = submission_payload(
            self.venue, intent, client_order_id, symbol=market.symbol, native_id=market.native_id
        )
        native, _, _ = self._request("AddOrder", params=params, submitting=True)
        identities = result(native).get("txid")
        if (
            not isinstance(identities, list)
            or len(identities) != 1
            or not isinstance(identities[0], str)
        ):
            raise LedgerError("native Kraken submission did not provide one actual order ID")
        report = self.lookup(intent, client_order_id, identities[0], as_of)
        if report is None:
            raise LedgerError("native Kraken accepted identity awaits actual order lookup")
        return report

    @bounded
    def lookup(self, intent, client_order_id, venue_order_id, as_of):
        references, matches = [], {}
        if venue_order_id is not None:
            native, observed, source = self._request(
                "QueryOrders",
                params={
                    "txid": venue_order_id,
                    "cl_ord_id": client_order_id,
                    "trades": True,
                    "consolidate_taker": False,
                },
                missing=True,
            )
            if native is None:
                return None
            references.append(source)
            matches = result(native)
            if set(matches) - {venue_order_id}:
                raise LedgerError("native Kraken lookup returned another order")
        else:
            for path, field in (("OpenOrders", "open"), ("ClosedOrders", "closed")):
                pages, sources, observed = self._order_pages(path, client_id=client_order_id)
                references.extend(sources)
                for page in pages:
                    for key, row in result(page)[field].items():
                        if row.get("cl_ord_id") != client_order_id:
                            raise LedgerError("native Kraken client-filter result differs")
                        if key in matches and canonical(matches[key]) != canonical(row):
                            raise LedgerError(
                                "native Kraken order changed while collecting history; refresh"
                            )
                        matches[key] = row
        if not matches:
            return None
        if len(matches) != 1:
            raise LedgerError("native Kraken client identity matched multiple economic orders")
        order_id, order = next(iter(matches.items()))
        return self._report(intent, client_order_id, order_id, order, observed, references)

    @bounded
    def cancel(self, intent, client_order_id, venue_order_id, as_of):
        self._claim(
            intent, client_order_id, None, as_of, cancel=True, venue_order_id=venue_order_id
        )
        self._request("CancelOrder", params={"txid": venue_order_id}, missing=True)
        report = self.lookup(intent, client_order_id, venue_order_id, as_of)
        if report is None:
            raise LedgerError("native Kraken cancellation unresolved; retain reservation")
        return report

    @bounded
    def read_native_order(self, intent, venue_order_id):
        """Read source-bound native order history, including legacy userref orders."""
        native, observed, source = self._request(
            "QueryOrders",
            params={"txid": venue_order_id, "trades": True, "consolidate_taker": False},
            missing=True,
        )
        if native is None:
            return None
        matches = result(native)
        if set(matches) != {venue_order_id}:
            raise LedgerError("native Kraken lookup returned missing/different order")
        order = matches[venue_order_id]
        return self._report(
            intent,
            order.get("cl_ord_id"),
            venue_order_id,
            order,
            observed,
            [source],
            report_client_id=order.get("cl_ord_id") or "native-order:" + venue_order_id,
        )

    def _report(
        self, intent, client_id, order_id, order, observed, references, *, report_client_id=None
    ):
        market, description = self._market(intent.instrument), order["descr"]
        from src.execution.native_order_terms import require_kraken_order

        require_kraken_order(intent, client_id, order, market)
        cumulative = decimal(order["vol_exec"])
        ids = order.get("trades", [])
        if (
            not isinstance(ids, list)
            or any(not isinstance(key, str) or not key for key in ids)
            or len(set(ids)) != len(ids)
        ):
            raise LedgerError("native Kraken individual trade coverage invalid")
        fills, costs, fees = [], [], []
        complete = True
        for start in range(0, len(ids), 20):
            requested = ids[start : start + 20]
            native, observed, trade_source = self._request(
                "QueryTrades", params={"txid": ",".join(requested), "ledgers": True}
            )
            references.append(trade_source)
            trades = result(native)
            if set(trades) - set(requested):
                raise LedgerError("native Kraken returned unrequested trades")
            complete &= set(trades) == set(requested)
            for key in requested:
                if key not in trades:
                    continue
                trade = trades[key]
                if (
                    trade["ordertxid"] != order_id
                    or trade["pair"] != market.native_id
                    or trade["type"] != intent.side
                    or decimal(trade["margin"])
                    or trade["ordertype"]
                    not in (
                        ("market", description["ordertype"])
                        if intent.order_type in ("stop", "trailing_stop")
                        else (description["ordertype"],)
                    )
                ):
                    raise LedgerError("native Kraken trade order/pair/side/margin differs")
                proof, source_id, observed = self._trade_proof(key, trade, trade_source)
                charges = self._settlement(market, key, trade, proof["ledgers"])
                references.append(source_id)
                fills.append(
                    self._fill(
                        intent,
                        native_id=key,
                        native_payload=trade,
                        quantity=trade["vol"],
                        price=trade["price"],
                        cost=trade["cost"],
                        fee_asset=self.cash_asset,
                        fee_quantity=D(0),
                        asset_charges=charges,
                        executed_at=native_time(trade["time"]),
                        observed_at=observed,
                        evidence_id=source_id,
                    )
                )
                costs.append(decimal(trade["cost"]))
                fees.append(decimal(trade["fee"]))
        quantity = exact_decimal_sum(fill.quantity for fill in fills)
        if quantity > cumulative:
            raise LedgerError("native Kraken executions advanced beyond order receipt; refresh")
        complete &= quantity == cumulative
        if complete and (
            exact_decimal_sum(costs) != decimal(order["cost"])
            or exact_decimal_sum(fees) != decimal(order["fee"])
        ):
            raise LedgerError("native Kraken individual costs/quoted fees disagree with order")
        status = {
            "pending": "new",
            "open": "partial" if cumulative else "new",
            "closed": "filled",
            "canceled": "cancelled",
            "expired": "expired",
        }.get(order["status"], "unknown")
        content = {
            "order_id": order_id,
            "order": order,
            "references": references,
            "intent_hash": intent.content_hash(),
        }
        identity = "kraken-order:" + digest(content)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, identity, canonical(content))
        return VenueOrderReport(
            venue=self.venue,
            client_order_id=report_client_id or client_id,
            venue_order_id=order_id,
            observed_at=observed,
            status=status,
            cumulative_quantity=cumulative,
            fills=tuple(fills),
            executions_complete=bool(complete and status != "unknown"),
            evidence_id=identity,
        )

    def _trade_proof(self, trade_id, trade, source):
        ids = trade.get("ledgers")
        if (
            not isinstance(ids, list)
            or not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(key, str) or not key for key in ids)
        ):
            raise LedgerError("native Kraken actual trade ledger IDs unavailable")
        ledgers, refs = {}, []
        for start in range(0, len(ids), 20):
            requested = ids[start : start + 20]
            value, at, evidence = self._request("QueryLedgers", params={"id": ",".join(requested)})
            if set(result(value)) != set(requested):
                raise LedgerError("native Kraken actual debit ledger coverage incomplete")
            ledgers.update(result(value))
            for key, entry in result(value).items():
                self._bind_ledger_entry(key, entry, evidence)
            refs.append(evidence)
        proof = {
            "trade_id": trade_id,
            "trade": trade,
            "ledgers": ledgers,
            "trade_source": source,
            "ledger_sources": refs,
            "observed_at": at.isoformat(),
        }
        identity = "kraken-trade-proof:" + digest(proof)
        self._verify_trade_proof(proof)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, identity, canonical(proof))
        return proof, identity, at

    def _bind_ledger_entry(self, key, entry, source):
        identity = "kraken-ledger-entry:" + digest(key)
        with self.ledger.atomic():
            row = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, identity)
            ).fetchone()
            if row is not None:
                original = json.loads(row[0])
                receipt, raw = super()._source(original["source"])
                self._require_endpoint(receipt, "QueryLedgers")
                if (
                    original["id"] != key
                    or canonical(original["entry"]) != canonical(entry)
                    or canonical(result(raw).get(key)) != canonical(entry)
                ):
                    raise LedgerError(
                        "native Kraken ledger identity changed original posted content"
                    )
                return
            self.ledger.record_event(
                self.account_id, identity, canonical({"id": key, "entry": entry, "source": source})
            )

    def _source(self, evidence_id):
        if not evidence_id.startswith("kraken-trade-proof:"):
            return super()._source(evidence_id)
        row = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, evidence_id)
        ).fetchone()
        if row is None:
            raise LedgerError("Kraken trade proof missing")
        proof = json.loads(row[0])
        if evidence_id != "kraken-trade-proof:" + digest(proof):
            raise LedgerError("Kraken trade proof changed")
        self._verify_trade_proof(proof)
        return proof, proof

    def _verify_trade_proof(self, proof):
        receipt, raw = super()._source(proof["trade_source"])
        self._require_endpoint(receipt, "QueryTrades")
        if (
            receipt["parameters"].get("ledgers") is not True
            or proof["trade_id"] not in receipt["parameters"]["txid"].split(",")
            or canonical(result(raw).get(proof["trade_id"])) != canonical(proof["trade"])
        ):
            raise LedgerError("Kraken trade proof differs from original native bytes")
        clocks = [datetime.fromisoformat(receipt["observed_at"])]
        ledgers = {}
        for source in proof["ledger_sources"]:
            receipt, raw = super()._source(source)
            self._require_endpoint(receipt, "QueryLedgers")
            native = result(raw)
            if set(native) != set(receipt["parameters"]["id"].split(",")) or set(native) & set(
                ledgers
            ):
                raise LedgerError("Kraken ledger proof request/coverage differs")
            ledgers.update(native)
            clocks.append(datetime.fromisoformat(receipt["observed_at"]))
        if (
            canonical(ledgers) != canonical(proof["ledgers"])
            or set(ledgers) != set(proof["trade"]["ledgers"])
            or max(clocks) != datetime.fromisoformat(proof["observed_at"])
        ):
            raise LedgerError("Kraken debit proof bytes/coverage/clock changed")

    @staticmethod
    def _settlement(market, identity, trade, ledgers):
        amounts, fees = {}, {}
        for row in ledgers.values():
            if (
                row["refid"] != identity
                or row["type"] != "trade"
                or row.get("subtype", "") != ""
                or row["aclass"] != "currency"
                or decimal(row["time"]) != decimal(trade["time"])
            ):
                raise LedgerError(
                    "Kraken ledger entry differs from actual trade identity/type/time"
                )
            amount, fee = decimal(row["amount"]), decimal(row["fee"])
            if fee < 0:
                raise LedgerError("negative native fee needs explicit rebate accounting")
            asset = row["asset"]
            amounts.setdefault(asset, []).append(amount)
            fees.setdefault(asset, []).append(fee)
        actual = {key: exact_decimal_sum(value) for key, value in amounts.items()}
        sign = D(1) if trade["type"] == "buy" else D(-1)
        expected = {
            market.base_asset: exact_decimal_product(sign, decimal(trade["vol"])),
            market.quote_asset: exact_decimal_product(sign.copy_negate(), decimal(trade["cost"])),
        }
        if any(value != expected.get(key, D(0)) for key, value in actual.items()) or any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise LedgerError(
                "native Kraken debit/credit amounts do not equal gross trade settlement"
            )
        charges = [(key, exact_decimal_sum(value)) for key, value in sorted(fees.items())]
        # The trade's fee is a quoted-currency aggregate, not a debit currency.
        # Only compare directly when every actual debit was in that currency.
        if not any(value for key, value in charges if key != market.quote_asset) and sum(
            (value for key, value in charges if key == market.quote_asset), D(0)
        ) != decimal(trade["fee"]):
            raise LedgerError("native Kraken quote fee differs from posted ledger debit")
        return charges
