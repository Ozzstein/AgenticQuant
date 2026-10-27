"""Single-attempt Alpaca REST venue over the canonical durable execution book.

Capture account/assets before a decision cutoff, prepare/dispatch via
VenueExecution, then reconcile actual FILL activities and posted cash fees.
Network responses are retained without credential headers. No SDK retry loop,
average-price synthetic fills, portfolio cache fallback or assumed acceptance.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal as D
from functools import wraps
from threading import RLock
from urllib.parse import quote

import httpx
from pydantic import SecretStr

from src.domain.common import _utc
from src.domain.portfolio import AccountFee, Fill, OrderIntent
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.venue import DefinitiveRejection, VenueBalances, VenueOrderReport, VenueRules


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _decimal(value):
    if isinstance(value, (bool, float)) or value is None:
        raise LedgerError("remote monetary values require decimal text/integer boundaries")
    result = D(value)
    if not result.is_finite():
        raise LedgerError("nonfinite remote monetary value")
    return result


def _bounded(method):
    @wraps(method)
    def operation(self, *args, **kwargs):
        with self._operation():
            return method(self, *args, **kwargs)

    return operation


class AlpacaVenue:
    venue = "alpaca"
    owner = "direct"
    maximum_operation_delay_seconds = 30

    def __init__(
        self,
        ledger: PortfolioLedger,
        account_id: str,
        *,
        broker_account_id: str,
        api_key: str | SecretStr,
        api_secret: str | SecretStr,
        rules: dict[str, VenueRules],
        allow_live: bool = False,
        client: httpx.AsyncClient | None = None,
        clock=None,
        operation_timeout_seconds: float = 30,
        max_requests_per_operation: int = 128,
        maximum_response_bytes: int = 8 * 1024 * 1024,
    ):
        self.ledger, self.account_id = ledger, account_id
        spec = ledger.spec(account_id)
        if spec.mode not in ("paper", "live") or spec.base_currency != "USD":
            raise LedgerError("Alpaca equity venue requires a paper/live USD account")
        if spec.mode == "live" and not allow_live:
            raise LedgerError("live Alpaca requires independent explicit authorization")
        if (
            not broker_account_id
            or not rules
            or any(
                key != value.instrument.instrument_id
                or value.venue != self.venue
                or value.market_type != "equity"
                or value.instrument.quote_currency != "USD"
                for key, value in rules.items()
            )
        ):
            raise LedgerError("explicit remote account and eligible equity venue rules required")
        self.mode, self.broker_account_id, self._rules = spec.mode, broker_account_id, dict(rules)
        if (
            not math.isfinite(operation_timeout_seconds)
            or operation_timeout_seconds <= 0
            or type(max_requests_per_operation) is not int
            or max_requests_per_operation < 1
            or type(maximum_response_bytes) is not int
            or maximum_response_bytes < 1
        ):
            raise LedgerError("positive finite transport operation limits required")
        if client is not None and not isinstance(client, httpx.AsyncClient):
            raise LedgerError("injected Alpaca transport must be httpx.AsyncClient")
        self.operation_timeout_seconds = operation_timeout_seconds
        self.max_requests_per_operation = max_requests_per_operation
        self.maximum_response_bytes = maximum_response_bytes
        self._operation_lock = RLock()
        self._budget = None
        self._runner = asyncio.Runner()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self._secret = api_secret if isinstance(api_secret, SecretStr) else SecretStr(api_secret)
        if not self._key.get_secret_value() or not self._secret.get_secret_value():
            raise LedgerError("Alpaca credentials unavailable")
        self._url = (
            "https://paper-api.alpaca.markets"
            if self.mode == "paper"
            else "https://api.alpaca.markets"
        )
        with ledger.atomic():
            from src.execution.dividends import require_remote_income_account

            require_remote_income_account(ledger, account_id)
            ledger.record_event(
                account_id,
                "remote-binding:alpaca",
                _json(
                    {
                        "broker_account_id": broker_account_id,
                        "endpoint": self._url,
                        "fee_convention": "separately-posted-account-fees-1",
                    }
                ),
            )
            for key, rule in rules.items():
                ledger.record_event(
                    account_id,
                    "remote-instrument:alpaca:" + key,
                    _json(
                        {
                            "instrument": rule.instrument.model_dump(mode="json"),
                            "symbol": rule.symbol,
                        }
                    ),
                )
        self._client = client or httpx.AsyncClient(
            timeout=15, follow_redirects=False, trust_env=False
        )

    def close(self):
        with self._operation_lock:
            if not self._client.is_closed:
                self._runner.run(self._client.aclose())
            self._runner.close()

    @contextmanager
    def _operation(self):
        with self._operation_lock:
            outer = self._budget is None
            if outer:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    raise LedgerError(
                        "run synchronous venue operations outside the caller's event loop"
                    )
                self._budget = [time.monotonic() + self.operation_timeout_seconds, 0]
            try:
                yield
            finally:
                if outer:
                    self._budget = None

    async def _read_response(self, method, path, *, params, payload, remaining):
        # A true overall cancellation deadline covers connection, headers and
        # every streamed byte, even if the peer keeps a per-read timeout alive.
        async with asyncio.timeout(remaining):
            async with self._client.stream(
                method,
                self._url + path,
                params=params,
                json=payload,
                headers={
                    "APCA-API-KEY-ID": self._key.get_secret_value(),
                    "APCA-API-SECRET-KEY": self._secret.get_secret_value(),
                    "Accept-Encoding": "identity",
                },
                timeout=httpx.Timeout(min(15, remaining)),
                follow_redirects=False,
            ) as response:
                body = bytearray()
                truncated = False
                async for chunk in response.aiter_raw():
                    available = self.maximum_response_bytes - len(body)
                    body.extend(chunk[:available])
                    if len(chunk) > available:
                        truncated = True
                        break
                return (
                    response.status_code,
                    bytes(body),
                    response.headers.get("content-encoding", "identity"),
                    truncated,
                )

    def _now(self):
        observed = _utc(self._clock())
        last = self.ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id LIKE 'alpaca-http:%' ORDER BY rowid DESC LIMIT 1",
            (self.account_id,),
        ).fetchone()
        if last and datetime.fromisoformat(json.loads(last[0])["observed_at"]) > observed:
            raise LedgerError("Alpaca receipt observation clock regressed")
        return observed

    def _require_claim(self, intent, client_order_id, rules, as_of):
        now = self._now()
        if not 0 <= (now - _utc(as_of)).total_seconds() <= self.maximum_operation_delay_seconds:
            raise LedgerError("submission requires a recent actual operation cutoff")
        rules.require_order(intent, now)
        if (
            now - self.balances(now).observed_at
        ).total_seconds() > self.maximum_operation_delay_seconds:
            raise LedgerError("refresh actual remote funding before submission")
        if intent.expires_at is not None and now > intent.expires_at:
            raise DefinitiveRejection("order expired before actual submission")
        if not self.ledger.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='venue_outbox'"
        ).fetchone():
            raise LedgerError("Alpaca submission requires the durable execution owner")
        row = self.ledger.db.execute(
            "SELECT o.*,b.owner FROM venue_outbox o JOIN execution_books b ON b.account=o.account WHERE o.account=? AND o.intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if (
            row is None
            or row["owner"] != self.owner
            or row["venue"] != self.venue
            or row["state"] != "submitting"
            or row["client_id"] != client_order_id
            or VenueRules.model_validate_json(row["rules"]) != rules
            or self.ledger.order(self.account_id, intent.intent_id)["intent"] != intent
        ):
            raise LedgerError("Alpaca submission requires the exact durable submitting claim")
        with self.ledger.atomic():
            key = "alpaca-submit-attempt:" + intent.intent_id
            if self.ledger.db.execute(
                "SELECT 1 FROM events WHERE account=? AND id=?", (self.account_id, key)
            ).fetchone():
                raise LedgerError(
                    "Alpaca submission already attempted; reconcile by client identity"
                )
            self.ledger.record_event(
                self.account_id,
                key,
                _json(
                    {
                        "client_order_id": client_order_id,
                        "intent_hash": intent.content_hash(),
                        "at": self._now().isoformat(),
                    }
                ),
            )

    def _request(self, method, path, *, params=None, payload=None, missing=False, submitting=False):
        # httpx performs no implicit order retry. A timeout after POST acceptance
        # escapes to VenueExecution's durable ambiguous state.
        self._now()
        if self._budget is None:
            raise LedgerError("remote HTTP request requires a bounded operation")
        remaining = self._budget[0] - time.monotonic()
        if remaining <= 0 or self._budget[1] >= self.max_requests_per_operation:
            raise LedgerError("Alpaca operation deadline/request allowance exhausted")
        self._budget[1] += 1
        status, content, encoding, truncated = self._runner.run(
            self._read_response(
                method,
                path,
                params=params,
                payload=payload,
                remaining=remaining,
            )
        )
        observed = self._now()
        evidence = {
            "method": method,
            "path": path,
            "params": params,
            "request": payload,
            "status": status,
            "observed_at": observed.isoformat(),
            "content_encoding": encoding,
            "complete": not truncated,
        }
        try:
            evidence["response"] = content.decode("utf-8")
        except UnicodeDecodeError:
            evidence["response_base64"] = base64.b64encode(content).decode("ascii")
        evidence_id = "alpaca-http:" + _hash(evidence)
        with self.ledger.atomic():
            self.ledger.record_event(self.account_id, evidence_id, _json(evidence))
        if truncated:
            raise LedgerError("Alpaca response exceeds streaming byte cap; bounded prefix archived")
        if encoding not in ("", "identity"):
            raise LedgerError("unexpected compressed wire response retained; execution unverified")
        if time.monotonic() > self._budget[0]:
            raise LedgerError("Alpaca overall operation deadline exhausted")
        if status == 404 and missing:
            return None, observed, evidence_id
        if not 200 <= status < 300:
            # A duplicate client ID or uncertain gateway failure never proves
            # the original submission absent. Keep its reservation and look up.
            code = None
            try:
                code = json.loads(content).get("code")
            except (ValueError, AttributeError):
                pass
            if submitting and status in (400, 401, 403, 422) and str(code) != "42210000":
                raise DefinitiveRejection(
                    f"Alpaca rejected submission: HTTP {status}, evidence {evidence_id}"
                )
            raise LedgerError(f"Alpaca HTTP {status}; retained evidence {evidence_id}")
        value = json.loads(content, parse_float=D) if content else None
        return value, observed, evidence_id

    @_bounded
    def capture(self) -> VenueBalances:
        """Read actual remote state before choosing the next decision cutoff."""
        account, _, account_evidence = self._request("GET", "/v2/account")
        if account.get("id") != self.broker_account_id or account.get("currency") != "USD":
            raise LedgerError("remote account/currency identity differs from immutable binding")
        positions, _, positions_evidence = self._request("GET", "/v2/positions")
        orders, _, orders_evidence = self._request(
            "GET", "/v2/orders", params={"status": "open", "limit": 500, "nested": "false"}
        )
        if not isinstance(orders, list) or len(orders) >= 500:
            raise LedgerError("open-order coverage is incomplete; cannot certify exclusive account")
        assets, references = (
            {},
            {
                "account": account_evidence,
                "positions": positions_evidence,
                "orders": orders_evidence,
            },
        )
        for key, rule in self._rules.items():
            asset, _, reference = self._request("GET", "/v2/assets/" + quote(rule.symbol, safe=""))
            if asset.get("symbol") != rule.symbol or asset.get("class") != "us_equity":
                raise LedgerError("Alpaca asset identity/class mismatch")
            assets[key] = asset
            references["asset:" + key] = reference
        observed = self._now()
        capture = {
            "account": account,
            "positions": positions,
            "orders": orders,
            "assets": assets,
            "observed_at": observed.isoformat(),
            "references": references,
        }
        balances = self._balances(capture)
        self._verify_capture(capture, balances.evidence_id)
        with self.ledger.atomic():
            self.ledger.record_event(
                self.account_id, "alpaca-capture:" + _hash(capture), _json(capture)
            )
        return balances

    def _capture(self, as_of):
        as_of = _utc(as_of)
        for row in self.ledger.db.execute(
            "SELECT id,payload FROM events WHERE account=? AND id LIKE 'alpaca-capture:%' ORDER BY rowid DESC",
            (self.account_id,),
        ):
            capture = json.loads(row["payload"])
            if datetime.fromisoformat(capture["observed_at"]) <= as_of:
                self._verify_capture(capture, row["id"])
                return capture
        raise LedgerError("capture actual Alpaca account/assets before the decision cutoff")

    def _verify_capture(self, capture, evidence_id):
        if evidence_id != "alpaca-capture:" + _hash(capture):
            raise LedgerError("Alpaca capture content hash changed")
        if (
            capture["account"].get("id") != self.broker_account_id
            or capture["account"].get("currency") != "USD"
        ):
            raise LedgerError("Alpaca capture account identity changed")
        expected = {
            "account": ("/v2/account", capture["account"]),
            "positions": ("/v2/positions", capture["positions"]),
            "orders": ("/v2/orders", capture["orders"]),
        }
        for key, asset in capture["assets"].items():
            expected["asset:" + key] = ("/v2/assets/" + quote(asset["symbol"], safe=""), asset)
        if set(capture["references"]) != set(expected):
            raise LedgerError("Alpaca capture raw evidence coverage changed")
        for key, (path, value) in expected.items():
            reference = capture["references"][key]
            row = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?", (self.account_id, reference)
            ).fetchone()
            if row is None:
                raise LedgerError("Alpaca capture raw evidence missing")
            raw = json.loads(row[0])
            if (
                reference != "alpaca-http:" + _hash(raw)
                or raw["method"] != "GET"
                or raw["path"] != path
                or not 200 <= raw["status"] < 300
                or raw.get("complete") is False
                or datetime.fromisoformat(raw["observed_at"])
                > datetime.fromisoformat(capture["observed_at"])
                or _json(json.loads(raw["response"], parse_float=D)) != _json(value)
            ):
                raise LedgerError("Alpaca capture differs from retained raw evidence")

    def _balances(self, capture):
        symbols = {rule.symbol: rule.instrument.instrument_id for rule in self._rules.values()}
        quantities = {}
        for position in capture["positions"]:
            if position.get("symbol") not in symbols or position.get("side") != "long":
                raise LedgerError(
                    "unmapped/short remote holding requires explicit audited migration"
                )
            key = symbols[position["symbol"]]
            if key in quantities:
                raise LedgerError("duplicate remote position identity")
            quantities[key] = _decimal(position["qty"])
        cash = _decimal(capture["account"]["cash"])
        free = min(cash, _decimal(capture["account"]["non_marginable_buying_power"]))
        return VenueBalances(
            venue=self.venue,
            observed_at=capture["observed_at"],
            cash_base=cash,
            available_cash_base=free,
            quantities=quantities,
            open_client_order_ids=tuple(
                order.get("client_order_id") or "unowned:" + order["id"]
                for order in capture["orders"]
            ),
            evidence_id="alpaca-capture:" + _hash(capture),
        )

    def balances(self, as_of):
        return self._balances(self._capture(as_of))

    def rules(self, instrument, as_of):
        capture = self._capture(as_of)
        account, asset = capture["account"], capture["assets"][instrument.instrument_id]
        rule = self._rules[instrument.instrument_id]
        if (
            rule.instrument != instrument
            or account.get("status") != "ACTIVE"
            or any(
                account.get(key)
                for key in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
            )
        ):
            raise LedgerError("remote account blocked or instrument mismatch")
        if not asset.get("tradable") or asset.get("status") != "active":
            raise LedgerError("remote asset is not active/tradable")
        if rule.quantity_step < 1 and not asset.get("fractionable"):
            raise LedgerError("fractional venue rule conflicts with actual asset eligibility")
        return rule  # original externally supplied lot/tick evidence is never freshened

    def _activities(self, *, kind, order_id=None):
        params = {
            "activity_types": kind,
            "direction": "asc",
            "page_size": 100,
            "after": self.ledger.spec(self.account_id).opened_at.isoformat(),
        }
        if order_id:
            params["order_id"] = order_id
        activities, seen, references, observed = [], set(), [], None
        for _ in range(1000):
            page, observed, reference = self._request(
                "GET", "/v2/account/activities", params=dict(params)
            )
            references.append(reference)
            if not isinstance(page, list):
                raise LedgerError("invalid activity coverage response")
            for item in page:
                if (
                    item.get("activity_type") != kind
                    or not item.get("id")
                    or item.get("id") in seen
                ):
                    raise LedgerError("activity pagination/type conflict")
                seen.add(item["id"])
                activities.append(item)
                with self.ledger.atomic():
                    self.ledger.record_event(
                        self.account_id, "alpaca-activity:" + _hash(item), _json(item)
                    )
            if len(page) < 100:
                return activities, observed, references
            params["page_token"] = page[-1]["id"]
        raise LedgerError("activity pagination bound exceeded; execution coverage unverified")

    def _report(self, intent, client_order_id, order, observed, references):
        if (
            not order.get("id")
            or order.get("extended_hours") is not False
            or order.get("order_class") not in ("", "simple")
            or order.get("client_order_id") != client_order_id
            or order.get("symbol") != self._rules[intent.instrument.instrument_id].symbol
            or order.get("side") != intent.side
            or _decimal(order.get("qty")) != intent.quantity
            or order.get("type") != intent.order_type
            or order.get("time_in_force", "").upper() != intent.tif
        ):
            raise LedgerError("remote acknowledgement differs from exact intended order")
        for key in ("limit_price", "stop_price"):
            actual = order.get(key)
            if (None if actual is None else _decimal(actual)) != getattr(intent, key):
                raise LedgerError("remote acknowledgement price differs from intended order")
        cumulative = _decimal(order["filled_qty"])
        fills = []
        if cumulative:
            activities, observed, activity_refs = self._activities(
                kind="FILL", order_id=order["id"]
            )
            references = [*references, *activity_refs]
            for activity in activities:
                if (
                    activity.get("order_id") != order["id"]
                    or activity.get("side") != intent.side
                    or activity.get("symbol") != order["symbol"]
                ):
                    raise LedgerError("activity execution identity differs from order")
                fills.append(
                    Fill(
                        fill_id="alpaca:" + activity["id"],
                        intent_id=intent.intent_id,
                        account_id=self.account_id,
                        quantity=_decimal(activity["qty"]),
                        price=_decimal(activity["price"]),
                        fee=D(0),
                        executed_at=activity["transaction_time"],
                        event_id="alpaca-activity:" + _hash(activity),
                    )
                )
        statuses = {
            "new": "new",
            "accepted": "new",
            "pending_new": "new",
            "partially_filled": "partial",
            "filled": "filled",
            "canceled": "cancelled",
            "expired": "expired",
            "rejected": "rejected",
            "done_for_day": "unknown",
            "pending_cancel": "unknown",
            "pending_replace": "unknown",
            "replaced": "unknown",
            "accepted_for_bidding": "new",
            "stopped": "unknown",
            "suspended": "unknown",
            "calculated": "unknown",
        }
        status = statuses.get(order.get("status"), "unknown")
        quantity = sum((fill.quantity for fill in fills), D(0))
        if quantity > cumulative:
            raise LedgerError(
                "activities advanced beyond acknowledgement; refresh order before reconciliation"
            )
        content = {
            "order": order,
            "references": references,
            "fee_convention": "separately-posted-account-fees-1",
        }
        with self.ledger.atomic():
            evidence_id = "alpaca-order:" + _hash(content)
            self.ledger.record_event(self.account_id, evidence_id, _json(content))
        return VenueOrderReport(
            venue=self.venue,
            client_order_id=client_order_id,
            venue_order_id=order["id"],
            observed_at=observed,
            status=status,
            cumulative_quantity=cumulative,
            fills=tuple(fills),
            executions_complete=quantity == cumulative and status != "unknown",
            evidence_id=evidence_id,
        )

    @_bounded
    def submit(self, intent: OrderIntent, client_order_id, rules, as_of):
        if intent.account_id != self.account_id or intent.fx_rate != 1:
            raise LedgerError("Alpaca exact account/USD FX required")
        rules.require_order(intent, as_of)
        self._require_claim(intent, client_order_id, rules, as_of)
        payload = {
            "symbol": rules.symbol,
            "qty": str(intent.quantity),
            "side": intent.side,
            "type": intent.order_type,
            "time_in_force": intent.tif.lower(),
            "client_order_id": client_order_id,
            "extended_hours": False,
            "order_class": "simple",
        }
        for key in ("limit_price", "stop_price"):
            if getattr(intent, key) is not None:
                payload[key] = str(getattr(intent, key))
        order, observed, reference = self._request(
            "POST", "/v2/orders", payload=payload, submitting=True
        )
        return self._report(intent, client_order_id, order, observed, [reference])

    @_bounded
    def lookup(self, intent, client_order_id, venue_order_id, as_of):
        path = (
            "/v2/orders/" + quote(venue_order_id, safe="")
            if venue_order_id
            else "/v2/orders:by_client_order_id"
        )
        order, observed, reference = self._request(
            "GET",
            path,
            params=None if venue_order_id else {"client_order_id": client_order_id},
            missing=True,
        )
        if order is None:
            return None  # a 404 is not authorization to retry a possibly accepted POST
        if venue_order_id and order.get("id") != venue_order_id:
            raise LedgerError("Alpaca lookup returned a different venue order identity")
        return self._report(intent, client_order_id, order, observed, [reference])

    @_bounded
    def read_native_order(self, intent, venue_order_id):
        """Read actual executions for an externally owned, source-bound order ID.

        This does not submit, adopt, cancel or book anything. The execution owner
        must independently bind that broker ID to its durable original intent.
        """
        order, observed, reference = self._request(
            "GET", "/v2/orders/" + quote(venue_order_id, safe=""), missing=True
        )
        if order is None:
            return None
        if order.get("id") != venue_order_id or not order.get("client_order_id"):
            raise LedgerError("native Alpaca order identity is missing/different")
        return self._report(intent, order["client_order_id"], order, observed, [reference])

    @_bounded
    def cancel(self, intent, client_order_id, venue_order_id, as_of):
        if self._now() < _utc(as_of):
            raise LedgerError("cannot cancel against a future decision clock")
        row = self.ledger.db.execute(
            "SELECT o.*,b.owner FROM venue_outbox o JOIN execution_books b ON b.account=o.account WHERE o.account=? AND o.intent=?",
            (self.account_id, intent.intent_id),
        ).fetchone()
        if (
            row is None
            or row["owner"] != self.owner
            or row["venue"] != self.venue
            or row["state"] != "cancel_pending"
            or row["client_id"] != client_order_id
            or row["venue_id"] != venue_order_id
            or self.ledger.order(self.account_id, intent.intent_id)["intent"] != intent
        ):
            raise LedgerError("cancellation requires the exact durable execution owner claim")
        self._request("DELETE", "/v2/orders/" + quote(venue_order_id, safe=""))
        report = self.lookup(intent, client_order_id, venue_order_id, as_of)
        if report is None:
            raise LedgerError("cancellation outcome unresolved; retain reservation")
        return report

    @_bounded
    def posted_fees(self) -> tuple[AccountFee, ...]:
        activities, observed, _ = self._activities(kind="FEE")
        result = []
        for activity in activities:
            amount = _decimal(activity["net_amount"])
            if amount >= 0:
                raise LedgerError("fee refund/zero posting needs explicit classification")
            has_outbox = self.ledger.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='venue_outbox'"
            ).fetchone()
            row = (
                self.ledger.db.execute(
                    "SELECT intent FROM venue_outbox WHERE account=? AND venue=? AND venue_id=?",
                    (self.account_id, self.venue, activity.get("order_id")),
                ).fetchone()
                if has_outbox
                else None
            )
            if row is None and activity.get("order_id"):
                binding = self.ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (
                        self.account_id,
                        "lean-native-order:" + _hash([self.venue, activity["order_id"]]),
                    ),
                ).fetchone()
                if binding:
                    proof = json.loads(binding[0])
                    bound = self.ledger.order(self.account_id, proof["intent_id"])["intent"]
                    if (
                        proof["venue"] != self.venue
                        or proof["broker_order_id"] != activity["order_id"]
                        or proof["intent_hash"] != bound.content_hash()
                    ):
                        raise LedgerError("LEAN native posted fee attribution binding differs")
                    row = (bound.intent_id,)
            result.append(
                AccountFee(
                    fee_id="alpaca:" + activity["id"],
                    account_id=self.account_id,
                    amount=amount.copy_negate(),
                    currency="USD",
                    effective_date=date.fromisoformat(activity["date"][:10]),
                    observed_at=observed,
                    source_hash=_hash(activity),
                    intent_id=row[0] if row else None,
                )
            )
        return tuple(result)

    @_bounded
    def reconcile_fees(self, *, sleeve_book=None, attributions=None):
        """Book posted actual expenses; unknown attribution never invents ownership."""
        fees = self.posted_fees()
        has_sleeves = self.ledger.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sleeve_events'"
        ).fetchone()
        attributed = (
            has_sleeves
            and self.ledger.db.execute(
                "SELECT 1 FROM sleeve_events WHERE account=?", (self.account_id,)
            ).fetchone()
        )
        if attributed and (
            sleeve_book is None
            or sleeve_book.ledger is not self.ledger
            or sleeve_book.account_id != self.account_id
        ):
            raise LedgerError("posted fees require the canonical account's actual SleeveBook")
        with self.ledger.atomic():
            for fee in fees:
                if attributed:
                    if not attributions or fee.fee_id not in attributions:
                        raise LedgerError(
                            "explicit fee sleeve allocation/source evidence unavailable"
                        )
                    amounts, source_hash = attributions[fee.fee_id]
                    sleeve_book.apply_account_fee(fee, amounts, attribution_source_hash=source_hash)
                else:
                    self.ledger.apply_account_fee(fee)
        return fees
