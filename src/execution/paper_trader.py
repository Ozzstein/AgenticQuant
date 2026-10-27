"""Durable paper broker and an explicit adapter for legacy observed-quote callers.

Typed callers use ``execution.submit`` and ``execution.open_bar/close_bar``.
The legacy execute_order API represents an immediately observed executable
quote, not a historical close-price backtest. It shares the same ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.domain.common import Instrument
from src.domain.portfolio import AccountSpec, Fill, FillPolicy, OrderIntent, PriceMark, RiskPolicy
from src.execution.broker import compute_broker_metrics
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.simulator import PaperExecution
from src.utils.exceptions import OrderError
from src.utils.schemas import AssetClass, Order, OrderSide, OrderStatus, Portfolio, Position

D = Decimal


class PaperTrader:
    def __init__(
        self,
        initial_cash: float,
        slippage_bps: int = 5,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        *,
        ledger_path: str | Path | None = None,
        account_id="paper:default",
        opened_at: datetime | None = None,
        base_currency="USD",
        fill_policy: FillPolicy | None = None,
        risk_policy: RiskPolicy | None = None,
    ):
        if ledger_path is None:
            # Explicitly isolated legacy session; create_broker supplies the stable operating path.
            fd = tempfile.NamedTemporaryFile(
                prefix="quantagent-paper-", suffix=".sqlite3", delete=False
            )
            ledger_path = fd.name
            fd.close()
        self.ledger = PortfolioLedger(ledger_path)
        self.account_id = account_id
        try:
            spec = self.ledger.spec(account_id)
        except LedgerError:
            spec = AccountSpec(
                account_id=account_id,
                mode="paper",
                base_currency=base_currency,
                initial_cash=D(str(initial_cash)),
                opened_at=opened_at or datetime.now(timezone.utc),
            )
        if (
            spec.mode != "paper"
            or spec.base_currency != base_currency
            or spec.initial_cash != D(str(initial_cash))
        ):
            raise LedgerError("paper account initial conditions mismatch")
        self.ledger.create_account(spec)
        self.fill_policy = fill_policy or FillPolicy(
            equity_cost_bps=0,
            crypto_cost_bps=0,
            slippage_bps=slippage_bps,
            commission_per_unit=D(str(commission_per_share)),
            minimum_commission=D(str(min_commission)),
        )
        self.execution = PaperExecution(self.ledger, account_id, self.fill_policy, risk_policy)

    def _instrument(self, ticker):
        return Instrument(
            instrument_id="legacy:equity:" + ticker,
            symbol=ticker,
            asset_class="equity",
            quote_currency=self.ledger.spec(self.account_id).base_currency,
        )

    def execute_order(self, order: Order, market_prices: dict[str, float]) -> Order | None:
        if order.ticker not in market_prices:
            raise OrderError(f"No market price available for {order.ticker}")
        raw = D(str(market_prices[order.ticker]))
        if (
            not raw.is_finite()
            or raw <= 0
            or not D(str(order.quantity)).is_finite()
            or order.quantity <= 0
        ):
            raise OrderError("positive finite quantity/price required")
        if order.order_type.value == "TRAILING_STOP":
            raise OrderError(
                "attach trail_percent to the entry order; standalone trailing orders need a persisted peak"
            )
        side = order.side.value.lower()
        at = datetime.now(timezone.utc)
        decision = order.timestamp.astimezone(timezone.utc)
        # Legacy orders may predate account creation. This adapter establishes intent
        # receipt at the account's clock; immutable ID still includes caller timestamp.
        identity = {
            k: v
            for k, v in order.model_dump(mode="json").items()
            if k not in ("status", "fill_price")
        }
        intent_id = (
            "quote:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        )
        existing = self.ledger.db.execute(
            "SELECT payload FROM orders WHERE account=? AND id=?", (self.account_id, intent_id)
        ).fetchone()
        if existing:
            intent = OrderIntent.model_validate_json(existing[0])
            state = self.ledger.order(self.account_id, intent_id)
            if state["status"] == "filled":
                filled = next(
                    f for f in self.ledger.fills(self.account_id) if f.intent_id == intent_id
                )
                order.status, order.fill_price = OrderStatus.FILLED, float(filled.price)
                return order
        else:
            decision = max(
                decision, datetime.fromisoformat(self.ledger._account(self.account_id)["last_at"])
            )
            if decision >= at:
                at = decision + timedelta(microseconds=1)
            price = raw * (1 + self.fill_policy.slippage_bps / 10000 * (1 if side == "buy" else -1))
            qty = D(str(order.quantity))
            fee = self.fill_policy.fee(self._instrument(order.ticker), qty, price, D(1))
            intent = OrderIntent(
                intent_id=intent_id,
                account_id=self.account_id,
                run_id="legacy-quotes",
                instrument=self._instrument(order.ticker),
                side=side,
                quantity=qty,
                decision_at=decision,
                eligible_at=decision + timedelta(microseconds=1),
                reference_price=price,
                estimated_fee=fee,
                strategy_id="legacy-unattributed",
                order_type=order.order_type.value.lower(),
                limit_price=order.limit_price,
                stop_price=order.stop_loss_price,
                tif=order.time_in_force.value,
                expires_at=(at + timedelta(days=1)) if order.time_in_force.value == "DAY" else None,
            )
        marketable = True
        if intent.order_type == "limit":
            marketable = raw <= intent.limit_price if side == "buy" else raw >= intent.limit_price
        elif intent.order_type == "stop":
            marketable = raw >= intent.stop_price if side == "buy" else raw <= intent.stop_price
        if not marketable:
            # Legacy synchronous API has no pending acknowledgement; typed API does.
            order.status = OrderStatus.REJECTED
            return None
        price = raw * (1 + self.fill_policy.slippage_bps / 10000 * (1 if side == "buy" else -1))
        fee = self.fill_policy.fee(intent.instrument, intent.quantity, price, D(1))
        try:
            with self.ledger.atomic():
                self.ledger.submit(intent)
                self.ledger.apply_fill(
                    Fill(
                        fill_id=intent_id,
                        intent_id=intent_id,
                        account_id=self.account_id,
                        quantity=intent.quantity,
                        price=price,
                        fee=fee,
                        executed_at=at,
                        event_id=intent_id,
                    )
                )
                self.ledger.mark(
                    self.account_id, [PriceMark(instrument=intent.instrument, price=raw, as_of=at)]
                )
                if side == "buy" and any(
                    v is not None
                    for v in (order.take_profit_price, order.stop_loss_price, order.trail_percent)
                ):
                    protection = {
                        "ticker": order.ticker,
                        "take_profit": order.take_profit_price,
                        "stop_loss": order.stop_loss_price,
                        "trail": order.trail_percent,
                        "peak": str(raw),
                    }
                    self.ledger.db.execute(
                        "INSERT OR REPLACE INTO protections VALUES (?,?,?)",
                        (self.account_id, intent.instrument.instrument_id, json.dumps(protection)),
                    )
        except LedgerError:
            order.status = OrderStatus.REJECTED
            return None
        order.status, order.fill_price = OrderStatus.FILLED, float(price)
        return order

    def snapshot(self, market_prices: dict[str, float], *, as_of: datetime | None = None):
        at = as_of or datetime.now(timezone.utc)
        with self.ledger.atomic():
            for row in self.ledger.db.execute(
                "SELECT instrument,payload FROM protections WHERE account=?", (self.account_id,)
            ).fetchall():
                p = json.loads(row["payload"])
                if p["ticker"] not in market_prices:
                    continue
                price = D(str(market_prices[p["ticker"]]))
                peak = max(D(p["peak"]), price)
                p["peak"] = str(peak)
                self.ledger.db.execute(
                    "UPDATE protections SET payload=? WHERE account=? AND instrument=?",
                    (json.dumps(p), self.account_id, row["instrument"]),
                )
                stop = D(str(p["stop_loss"])) if p["stop_loss"] else D(0)
                if p["trail"]:
                    stop = max(stop, peak * (1 - D(str(p["trail"])) / 100))
                if price <= stop or (
                    p["take_profit"] is not None and price >= D(str(p["take_profit"]))
                ):
                    qty = self.ledger.quantity(self.account_id, row["instrument"])
                    if qty:
                        self.execute_order(
                            Order(ticker=p["ticker"], side=OrderSide.SELL, quantity=float(qty)),
                            market_prices,
                        )
                        at = max(
                            at,
                            datetime.fromisoformat(
                                self.ledger._account(self.account_id)["last_at"]
                            ),
                        )
            self.ledger.mark(
                self.account_id,
                [
                    PriceMark(instrument=self._instrument(t), price=D(str(p)), as_of=at)
                    for t, p in market_prices.items()
                    if not t.endswith("_adv")
                ],
            )
            return self.ledger.snapshot(
                self.account_id, at, self.execution.risk_policy.max_price_age_seconds
            )

    @property
    def portfolio(self) -> Portfolio:
        at = datetime.fromisoformat(self.ledger._account(self.account_id)["last_at"])
        snap = self.ledger.snapshot(
            self.account_id, at, self.execution.risk_policy.max_price_age_seconds, persist=False
        )
        return Portfolio(
            cash=float(snap.cash),
            nav=float(snap.nav),
            timestamp=at,
            positions={
                p.instrument.symbol: Position(
                    ticker=p.instrument.symbol,
                    quantity=float(p.quantity),
                    avg_cost=float(p.cost_basis / p.quantity),
                    current_price=float(p.price * p.fx_rate),
                    unrealized_pnl=float(p.market_value - p.cost_basis),
                    weight_pct=float(p.market_value / snap.nav * 100),
                    asset_class=AssetClass(p.instrument.asset_class.upper()),
                )
                for p in snap.positions
            },
        )

    @property
    def protected_tickers(self) -> set[str]:
        return {
            json.loads(r[0])["ticker"]
            for r in self.ledger.db.execute(
                "SELECT payload FROM protections WHERE account=?", (self.account_id,)
            )
        }

    @property
    def _trade_log(self):
        result = []
        for row in self.ledger.db.execute(
            "SELECT * FROM fills WHERE account=? ORDER BY at,rowid", (self.account_id,)
        ):
            fill = Fill.model_validate_json(row["payload"])
            order = self.ledger.order(self.account_id, fill.intent_id)["intent"]
            result.append(
                {
                    "timestamp": fill.executed_at.isoformat(),
                    "ticker": order.instrument.symbol,
                    "side": order.side.upper(),
                    "quantity": float(fill.quantity),
                    "fill_price": float(fill.price),
                    "commission": float(fill.fee),
                    "pnl": float(row["realized"]),
                }
            )
        return result

    @property
    def _nav_history(self):
        spec = self.ledger.spec(self.account_id)
        return [(spec.opened_at, float(spec.initial_cash))] + [
            (s.as_of, float(s.nav)) for s in self.ledger.history(self.account_id)
        ]

    def get_metrics(self, *, samples_per_year: int = 252):
        return compute_broker_metrics(
            self._nav_history, self._trade_log, samples_per_year=samples_per_year
        )

    def trade_log_to_csv(self, path: Path):
        rows = self._trade_log
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0])
                if rows
                else ["timestamp", "ticker", "side", "quantity", "fill_price", "commission", "pnl"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def portfolio_history_to_csv(self, path: Path):
        with Path(path).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "nav"])
            writer.writerows((t.isoformat(), n) for t, n in self._nav_history)
