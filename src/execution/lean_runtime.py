"""Actual QCAlgorithm integration. QuantConnect/.NET imports occur at attach time."""

from __future__ import annotations

import hashlib
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal as D
from threading import RLock

from src.domain.portfolio import PriceMark, exact_decimal_product
from src.execution.ccxt_transport import decimal
from src.execution.lean_venue import require_trailing_support
from src.execution.ledger import LedgerError
from src.execution.risk_controls import evaluate_order


def utc_datetime(value):
    """Convert an explicitly UTC .NET clock without a floating Unix timestamp."""
    ticks = int(value.Ticks) % 10000000
    from datetime import timedelta

    return datetime(
        int(value.Year),
        int(value.Month),
        int(value.Day),
        int(value.Hour),
        int(value.Minute),
        int(value.Second),
        tzinfo=timezone.utc,
    ) + timedelta(microseconds=(ticks + 9) // 10)


class LeanAlgorithmPort:
    """Narrow real runtime boundary; offline substitutes live only in tests."""

    def __init__(self, algorithm, deployment, symbols, *, allow_live=False):
        try:
            from QuantConnect import Extensions
            from QuantConnect.Algorithm import QCAlgorithm
            from QuantConnect.Configuration import Config
            from QuantConnect.Orders import OrderProperties
            from QuantConnect.Orders.TimeInForces import (
                DayTimeInForce,
                GoodTilCanceledTimeInForce,
            )
            from System import DateTime, Decimal
            from System.Globalization import CultureInfo
        except ImportError as exc:
            raise LedgerError("actual QuantConnect LEAN/.NET runtime is unavailable") from exc
        if (
            not isinstance(algorithm, QCAlgorithm)
            or not algorithm.LiveMode
            or Config.Get("live-mode-brokerage") != deployment.brokerage
        ):
            raise LedgerError("attach bridge only to the exact actual live-mode LEAN brokerage")
        if deployment.mode == "live" and not allow_live:
            raise LedgerError("LEAN live submission lacks independent authorization")
        if set(symbols) != {item.instrument.instrument_id for item in deployment.instruments}:
            raise LedgerError("LEAN subscribed symbol coverage differs")
        self.algorithm, self.deployment, self.symbols = algorithm, deployment, symbols
        self._decimal, self._datetime, self._culture = (
            Decimal,
            DateTime,
            CultureInfo.InvariantCulture,
        )
        self._properties, self._gtc, self._day = (
            OrderProperties,
            GoodTilCanceledTimeInForce,
            DayTimeInForce,
        )
        self._extensions = Extensions
        self.runtime_id = str(uuid.uuid4())
        self._identity = {
            "runtime_id": self.runtime_id,
            "deployment_hash": deployment.content_hash(),
            "live_mode": True,
            "engine_version": str(algorithm.GetType().Assembly.GetName().Version),
            "algorithm_id": str(algorithm.AlgorithmId),
        }
        if deployment.mode == "live":
            self._identity["native_credential_identity"] = self._local_broker_identity(
                Config, deployment.brokerage
            )
        self._ids = {str(value.ID): key for key, value in symbols.items()}
        if len(self._ids) != len(symbols):
            raise LedgerError("canonical mappings collapse onto one LEAN security")
        for item in deployment.instruments:
            security = algorithm.Securities[symbols[item.instrument.instrument_id]]
            if str(security.QuoteCurrency.Symbol) != item.instrument.quote_currency or str(
                security.Symbol.SecurityType
            ) != ("Crypto" if item.instrument.asset_class == "crypto" else "Equity"):
                raise LedgerError("actual LEAN security/quote identity differs")
            if (
                item.base_currency is not None
                and str(security.BaseCurrency.Symbol) != item.base_currency
            ):
                raise LedgerError("actual LEAN crypto base currency differs")

    @staticmethod
    def _local_broker_identity(config, brokerage):
        # The reviewed local JobQueue builds the actual LiveNodePacket from the
        # brokerage factory's Config fields. Cloud/custom job packets require a
        # separate actual packet authority; equal cash balances do not prove it.
        if config.Get("job-queue-handler") != "QuantConnect.Queues.JobQueue":
            raise LedgerError("LEAN live credential binding requires the reviewed local JobQueue")
        prefix = {
            "AlpacaBrokerage": "alpaca",
            "BinanceBrokerage": "binance",
            "KrakenBrokerage": "kraken",
        }[brokerage]
        key = config.Get(prefix + "-api-key")
        if not key or not config.Get(prefix + "-api-secret"):
            raise LedgerError("LEAN actual native credential binding is unavailable")
        if prefix == "alpaca" and (
            config.Get("alpaca-access-token")
            or config.Get("alpaca-paper-trading").lower() != "false"
        ):
            raise LedgerError("LEAN actual live Alpaca key authority/endpoint differs")
        if prefix == "binance" and (
            config.Get("binance-api-url", "https://api.binance.com") != "https://api.binance.com"
            or config.Get("binance-websocket-url", "wss://stream.binance.com:9443/ws")
            != "wss://stream.binance.com:9443/ws"
            or config.Get("binance-orders-websocket-url", "wss://ws-api.binance.com:9443/ws-api/v3")
            != "wss://ws-api.binance.com:9443/ws-api/v3"
        ):
            raise LedgerError("LEAN live Binance native network differs")
        return hashlib.sha256(key.encode()).hexdigest()

    def identity(self):
        return dict(self._identity)

    def now(self):
        return utc_datetime(self._datetime.UtcNow)

    def _number(self, value):
        # C# decimals use invariant text; accepting a Python float here would
        # silently collapse monetary precision at the runtime boundary.
        if isinstance(value, (float, bool)):
            raise LedgerError("LEAN monetary boundary must be actual decimal")
        return decimal(value.ToString(self._culture) if hasattr(value, "ToString") else value)

    def _native_decimal(self, value):
        original = decimal(value)
        native = self._decimal.Parse(format(original, "f"), self._culture)
        if self._number(native) != original:
            raise LedgerError("quantity/price is not exactly representable by LEAN System.Decimal")
        return native

    def restore_paper_state(self, ledger):
        """Seed the actual paper runtime from reconciled canonical balances.

        This is runtime initialization, not a fill or a canonical cash transfer.
        Live broker state can never be overwritten by this method.
        """
        account = self.deployment.account_id
        if self.deployment.mode != "paper" or ledger.spec(account).mode != "paper":
            raise LedgerError("only an actual local paper runtime can restore canonical balances")
        ledger.reconcile(account)
        if ledger.db.execute(
            "SELECT 1 FROM venue_outbox WHERE account=? AND state NOT IN "
            "('prepared','filled','cancelled','rejected','expired') LIMIT 1",
            (account,),
        ).fetchone():
            raise LedgerError(
                "LEAN paper restart has unresolved accepted orders; original proof required"
            )
        cash = ledger._account(account)["cash"]
        self.algorithm.SetCash(
            ledger.spec(account).base_currency, self._native_decimal(cash), self._native_decimal(1)
        )
        positions = []
        for row in ledger.db.execute(
            "SELECT instrument,quantity,basis FROM holdings WHERE account=?",
            (account,),
        ):
            quantity, basis = D(row["quantity"]), D(row["basis"])
            security = self.algorithm.Securities[self.symbols[row["instrument"]]]
            security.Holdings.SetHoldings(
                self._native_decimal(basis / quantity), self._native_decimal(quantity)
            )
            item = next(
                value
                for value in self.deployment.instruments
                if value.instrument.instrument_id == row["instrument"]
            )
            if item.base_currency is not None:
                # CryptoHolding.SetHoldings does not mutate CashBook. Its actual
                # base inventory must be restored separately, without a made-up FX price.
                self.algorithm.Portfolio.CashBook[item.base_currency].SetAmount(
                    self._native_decimal(quantity)
                )
            positions.append(dict(row))
        return {
            "runtime_id": self.runtime_id,
            "observed_at": self.now().isoformat(),
            "cash": cash,
            "positions": positions,
        }

    def _instrument(self, symbol):
        key = self._ids.get(str(symbol.ID))
        if key is None:
            raise LedgerError("unmapped actual LEAN symbol requires reconciliation")
        return key

    def _data_mark(self, key):
        security = self.algorithm.Securities[self.symbols[key]]
        data = security.GetLastData()
        if data is None:
            raise LedgerError("LEAN security has no actual observed price data")
        at = utc_datetime(self._extensions.ConvertToUtc(data.EndTime, security.Exchange.TimeZone))
        return {
            "instrument_id": key,
            "price": str(self._number(data.Value)),
            "as_of": at.isoformat(),
            "data_type": str(data.GetType().FullName),
            "data": str(data.ToString()),
        }

    def capture(self):
        now, portfolio = self.now(), self.algorithm.Portfolio
        cash = {str(row.Key): str(self._number(row.Value.Amount)) for row in portfolio.CashBook}
        unsettled = {
            str(row.Key): str(self._number(row.Value.Amount)) for row in portfolio.UnsettledCashBook
        }
        quantities, marks = {}, []
        for row in portfolio:
            quantity = self._number(row.Value.Quantity)
            if quantity:
                quantities[self._instrument(row.Key)] = str(quantity)
        for key in self.symbols:
            security = self.algorithm.Securities[self.symbols[key]]
            if security.GetLastData() is not None:
                marks.append(self._data_mark(key))
        opened = [
            {
                "client_order_id": str(order.Tag) or "unowned:" + str(order.Id),
                "ticket_id": int(order.Id),
                "broker_ids": [str(value) for value in order.BrokerId],
            }
            for order in self.algorithm.Transactions.GetOpenOrders()
        ]
        return {
            "runtime_id": self.runtime_id,
            "observed_at": now.isoformat(),
            "account_currency": str(portfolio.AccountCurrency),
            "cash_book": cash,
            "unsettled_cash": unsettled,
            "margin_remaining": str(self._number(portfolio.MarginRemaining)),
            "quantities": quantities,
            "marks": marks,
            "open_client_order_ids": [order["client_order_id"] for order in opened],
            "open_orders": opened,
        }

    def _day_expiry(self, key, submitted_utc):
        """Match actual DayTimeInForce using the subscribed security calendar."""
        security = self.algorithm.Securities[self.symbols[key]]
        hours = security.Exchange.Hours
        local = self._extensions.ConvertFromUtc(submitted_utc, hours.TimeZone)
        if str(security.Symbol.SecurityType) == "Crypto":
            # LEAN's crypto DAY implementation is a calendar-day boundary.
            # Only its documented UTC calendar is supported here.
            if str(hours.TimeZone.Id) != "UTC":
                raise LedgerError("LEAN crypto DAY requires the actual UTC calendar")
            close = local.Date.AddDays(1)
        elif str(security.Symbol.SecurityType) == "Equity":
            close = hours.GetLastDailyMarketClose(local, False)
        else:
            raise LedgerError("LEAN DAY security calendar is unsupported")
        return utc_datetime(self._extensions.ConvertToUtc(close, hours.TimeZone))

    def _order_properties(self, intent):
        properties = self._properties()
        if intent.tif == "GTC" and intent.expires_at is None:
            properties.TimeInForce = self._gtc()
        elif intent.tif == "DAY":
            if self.deployment.brokerage not in ("PaperBrokerage", "AlpacaBrokerage") or (
                self.deployment.brokerage == "AlpacaBrokerage"
                and intent.instrument.asset_class == "crypto"
            ):
                raise LedgerError("reviewed LEAN brokerage does not support this DAY order")
            now = self.now()
            submitted = self._datetime.Parse(now.isoformat(), self._culture).ToUniversalTime()
            if (
                intent.expires_at is None
                or now >= intent.expires_at
                or self._day_expiry(intent.instrument.instrument_id, submitted) != intent.expires_at
            ):
                raise LedgerError("LEAN DAY expiry differs from the actual security calendar")
            properties.TimeInForce = self._day()
        else:
            raise LedgerError("LEAN exact time-in-force/expiry mapping is unsupported")
        return properties

    def preflight(self, intent):
        require_trailing_support(self.deployment, intent)
        self._order_properties(intent)
        security = self.algorithm.Securities[self.symbols[intent.instrument.instrument_id]]
        if not security.IsTradable or not security.HasData or self.algorithm.IsWarmingUp:
            raise LedgerError("LEAN security is not currently ready/tradable")
        if intent.order_type == "market" and not security.Exchange.ExchangeOpen:
            raise LedgerError("closed-exchange market order would be converted by LEAN")
        mark = self._data_mark(intent.instrument.instrument_id)
        if not 0 <= (self.now() - datetime.fromisoformat(mark["as_of"])).total_seconds() <= 30:
            raise LedgerError("LEAN observed security price is stale/future")
        self._native_decimal(intent.quantity)
        for price in (intent.limit_price, intent.stop_price):
            if price is not None:
                self._native_decimal(price)
        if intent.order_type == "trailing_stop":
            self._native_decimal(
                exact_decimal_product(intent.trail_percent, D(".01"))
                if intent.trail_percent is not None
                else intent.trail_price
            )

    def submit(self, intent, client_id):
        require_trailing_support(self.deployment, intent)
        properties = self._order_properties(intent)
        quantity = self._native_decimal(
            intent.quantity if intent.side == "buy" else intent.quantity.copy_negate()
        )
        symbol = self.symbols[intent.instrument.instrument_id]
        if intent.order_type == "market":
            ticket = self.algorithm.MarketOrder(symbol, quantity, True, client_id, properties)
        elif intent.order_type == "limit":
            ticket = self.algorithm.LimitOrder(
                symbol, quantity, self._native_decimal(intent.limit_price), client_id, properties
            )
        elif intent.order_type == "trailing_stop":
            percentage = intent.trail_percent is not None
            amount = (
                exact_decimal_product(intent.trail_percent, D(".01"))
                if percentage
                else intent.trail_price
            )
            ticket = self.algorithm.TrailingStopOrder(
                symbol,
                quantity,
                self._native_decimal(amount),
                percentage,
                True,
                client_id,
                properties,
            )
        else:
            ticket = self.algorithm.StopMarketOrder(
                symbol, quantity, self._native_decimal(intent.stop_price), client_id, properties
            )
        return self._ticket(ticket)

    def lookup(self, client_id, ticket_id):
        if ticket_id is None:
            matches = [
                order
                for order in self.algorithm.Transactions.GetOrders()
                if str(order.Tag) == client_id
            ]
            if not matches:
                return None
            if len(matches) != 1:
                raise LedgerError("LEAN client identity matches multiple actual orders")
            ticket_id = int(matches[0].Id)
        ticket = self.algorithm.Transactions.GetOrderTicket(ticket_id)
        if ticket is None:
            return None
        payload = self._ticket(ticket)
        if payload["client_order_id"] != client_id:
            raise LedgerError("LEAN queried ticket has another client identity")
        return payload

    def lookup_broker_order(self, broker_order_id):
        """Actual imported runtime ticket, located only by its original native ID."""
        matches = [
            order
            for order in self.algorithm.Transactions.GetOrders()
            if broker_order_id in [str(value) for value in order.BrokerId]
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise LedgerError("native broker identity maps to multiple actual LEAN tickets")
        ticket = self.algorithm.Transactions.GetOrderTicket(int(matches[0].Id))
        if ticket is None:
            return None
        return self._ticket(ticket)

    def cancel(self, ticket_id, client_id):
        ticket = self.algorithm.Transactions.GetOrderTicket(ticket_id)
        if (
            ticket is None
            or str(self.algorithm.Transactions.GetOrderById(ticket_id).Tag) != client_id
        ):
            raise LedgerError("LEAN exact cancellation ticket unavailable")
        # A cancellation tag replaces Order.Tag in LEAN. Retain the immutable
        # client identity used for restart lookup instead of overwriting it.
        ticket.Cancel()

    def event(self, event):
        return {
            "event_id": int(event.Id),
            "ticket_id": int(event.OrderId),
            "instrument_id": self._instrument(event.Symbol),
            "occurred_at": utc_datetime(event.UtcTime).isoformat(),
            "status": str(event.Status),
            "fill_quantity": str(self._number(event.FillQuantity)),
            "fill_price": str(self._number(event.FillPrice)),
            "price_currency": str(event.FillPriceCurrency or ""),
            "fee_amount": str(self._number(event.OrderFee.Value.Amount)),
            "fee_currency": str(event.OrderFee.Value.Currency or ""),
            "fee_type": str(event.OrderFee.GetType().FullName),
            "is_assignment": bool(event.IsAssignment),
            "message": str(event.Message or ""),
        }

    def _ticket(self, ticket):
        order = self.algorithm.Transactions.GetOrderById(int(ticket.OrderId))
        if order is None:
            raise LedgerError("LEAN ticket has no actual order yet")
        if (
            getattr(order, "Contingency", None) is not None
            or getattr(order, "GroupOrderManager", None) is not None
        ):
            raise LedgerError("LEAN contingent/combo ticket requires explicit group ownership")
        kind = str(order.TimeInForce.GetType().FullName)
        if kind == "QuantConnect.Orders.TimeInForces.DayTimeInForce":
            expiry = {
                "time_in_force": "DAY",
                "expires_at": self._day_expiry(
                    self._instrument(order.Symbol), order.Time
                ).isoformat(),
            }
        elif kind == "QuantConnect.Orders.TimeInForces.GoodTilCanceledTimeInForce":
            expiry = {}  # Preserve existing ordinary ticket receipt shape.
        else:
            raise LedgerError("actual LEAN ticket has unsupported time-in-force")
        return {
            "runtime_id": self.runtime_id,
            "observed_at": self.now().isoformat(),
            "ticket_id": int(ticket.OrderId),
            "client_order_id": str(order.Tag),
            "instrument_id": self._instrument(order.Symbol),
            "quantity": str(self._number(ticket.Quantity)),
            "filled_quantity": str(self._number(ticket.QuantityFilled)),
            "status": str(ticket.Status),
            "order_type": str(order.Type),
            **expiry,
            "limit_price": str(self._number(order.LimitPrice))
            if str(order.Type) == "Limit"
            else None,
            "stop_price": str(self._number(order.StopPrice))
            if str(order.Type) in ("StopMarket", "TrailingStop")
            else None,
            **(
                {
                    "trailing_amount": str(self._number(order.TrailingAmount)),
                    "trailing_as_percentage": bool(order.TrailingAsPercentage),
                }
                if str(order.Type) == "TrailingStop"
                else {}
            ),
            "broker_ids": tuple(str(value) for value in order.BrokerId),
            "events": [self.event(event) for event in ticket.OrderEvents],
        }


class LeanOperatingBridge:
    """Drains existing admitted outbox entries inside the real LEAN algorithm."""

    def __init__(self, execution):
        self.execution = execution
        self.adapter = execution.adapters["lean"]
        if self.adapter.runtime is None:
            raise LedgerError("LEAN operating bridge needs its actual attached runtime")
        self.lock = RLock()
        self._health = None
        self._health_at = None

    def _operation(self):
        if self.adapter.mode != "live":
            return nullcontext()
        reader = self.adapter._native_reader()
        return reader._operation() if reader.venue == "alpaca" else reader.transport.operation()

    def _status(self, state, error=None):
        signature = (state, None if error is None else str(error))
        now = self.adapter.runtime.now()
        if (
            signature != self._health
            or self._health_at is None
            or (now - self._health_at).total_seconds() >= 10
        ):
            self.adapter._record(
                "runtime-status",
                {
                    "runtime_id": self.adapter.runtime.runtime_id,
                    "observed_at": now.isoformat(),
                    "status": state,
                    "error": signature[1],
                },
            )
            self._health = signature
            self._health_at = now
        return {"status": state, "error": signature[1]}

    def step(self):
        """Keep receiving actual broker events while an economic gate is closed."""
        try:
            self.pump()
        except LedgerError as exc:
            return self._status("blocked", exc)
        return self._status("ready")

    def on_order_event(self, event):
        # Retain the received native event even if reconciliation is currently
        # busy. Ticket history is checked as a whole before any economic fill.
        self.adapter._record(
            "callback",
            {
                "runtime_id": self.adapter.runtime.runtime_id,
                "observed_at": self.adapter.runtime.now().isoformat(),
                "event": self.adapter.runtime.event(event),
            },
        )
        if self.lock.acquire(blocking=False):
            try:
                with self._operation():
                    self.execution.reconcile_all(as_of=self.adapter.runtime.now())
            except LedgerError as exc:
                self._status("blocked", exc)
            finally:
                self.lock.release()

    def pump(self):
        with self.lock, self._operation():
            runtime, execution = self.adapter.runtime, self.execution
            execution.expire_orders(as_of=runtime.now())
            if execution.ledger.db.execute(
                "SELECT 1 FROM venue_outbox WHERE account=? AND state IN "
                "('submitting','ambiguous','cancel_pending','reconciliation_required') LIMIT 1",
                (execution.account_id,),
            ).fetchone():
                raise LedgerError(
                    "LEAN unresolved execution history blocks newer valuation/dispatch"
                )
            if self.adapter.mode == "live" and self.adapter.native_reader.venue == "alpaca":
                self.adapter.native_reader.reconcile_fees()
            observation = self.adapter.capture()
            execution.reconcile_balances(as_of=observation.observed_at)
            payload, _ = self.adapter._latest("balances", observation.observed_at)
            marks = {
                row["instrument_id"]: PriceMark(
                    instrument=self.adapter.instruments[row["instrument_id"]],
                    price=decimal(row["price"]),
                    as_of=datetime.fromisoformat(row["as_of"]),
                )
                for row in payload["marks"]
            }
            if any(mark.as_of > observation.observed_at for mark in marks.values()):
                raise LedgerError("LEAN price observation is future data")
            anchor = datetime.fromisoformat(
                execution.ledger._account(execution.account_id)["last_at"]
            )
            # A slow symbol's last observation is not a new event. Retain its
            # actual timestamp and keep the prior canonical mark; never redate
            # old data to the current pump clock.
            current_marks = [mark for mark in marks.values() if mark.as_of >= anchor]
            execution.ledger.mark(execution.account_id, current_marks)
            pending = list(
                execution.ledger.db.execute(
                    "SELECT intent,state FROM venue_outbox WHERE account=? AND state IN ('prepared','cancel_pending') ORDER BY rowid",
                    (execution.account_id,),
                )
            )
            for row in pending:
                intent = execution.ledger.order(execution.account_id, row["intent"])["intent"]
                at = runtime.now()
                if row["state"] == "cancel_pending":
                    execution.cancel(intent.intent_id, as_of=at)
                    continue
                if at < intent.eligible_at:
                    continue
                if intent.instrument.instrument_id not in marks:
                    raise LedgerError("LEAN pending order lacks an actual observed mark")
                mark = marks[intent.instrument.instrument_id]
                snapshot = execution.ledger.snapshot(
                    execution.account_id,
                    at,
                    execution.risk_policy.max_price_age_seconds,
                    persist=False,
                )
                current = intent.model_copy(
                    update={
                        "reference_price": mark.price,
                        "estimated_fee": max(
                            intent.estimated_fee,
                            execution.fill_policy.fee(
                                intent.instrument, intent.quantity, mark.price, mark.fx_rate
                            ),
                        ),
                    }
                )
                risk = evaluate_order(
                    current,
                    snapshot,
                    execution.ledger.pending(execution.account_id),
                    execution.admission_metadata(intent.intent_id),
                    execution.risk_policy,
                )
                if not risk.passed:
                    execution.cancel(intent.intent_id, as_of=at)
                    raise LedgerError(
                        "LEAN current projected risk rejected: " + "; ".join(risk.details.values())
                    )
                execution.dispatch(intent.intent_id, as_of=at)
            return observation
