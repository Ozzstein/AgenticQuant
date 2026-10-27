"""Observed-open simulator used unchanged by paper and historical replay."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from src.domain.portfolio import (
    ExecutionBar,
    Fill,
    FillPolicy,
    InstrumentRisk,
    OrderIntent,
    PriceMark,
    RiskPolicy,
)
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.risk_controls import evaluate_order


class PaperExecution:
    def __init__(
        self,
        ledger: PortfolioLedger,
        account_id: str,
        policy: FillPolicy | None = None,
        risk_policy: RiskPolicy | None = None,
    ):
        self.ledger = ledger
        self.account_id = account_id
        self.policy = policy or FillPolicy()
        self.risk_policy = risk_policy or RiskPolicy()

    def submit(self, intent: OrderIntent, metadata: dict[str, InstrumentRisk]):
        if intent.account_id != self.account_id:
            raise LedgerError("execution account mismatch")
        with self.ledger.atomic():
            # Retry before checking a changed portfolio; successful retries are idempotent.
            row = self.ledger.db.execute(
                "SELECT payload FROM orders WHERE account=? AND id=?",
                (self.account_id, intent.intent_id),
            ).fetchone()
            if row:
                return self.ledger.submit(intent)
            required_fee = self.policy.fee(
                intent.instrument, intent.quantity, intent.reference_price, intent.fx_rate
            )
            if intent.estimated_fee < required_fee:
                raise LedgerError("estimated fee understates the declared fill policy")
            snap = self.ledger.snapshot(
                self.account_id,
                intent.decision_at,
                self.risk_policy.max_price_age_seconds,
                persist=False,
            )
            result = evaluate_order(
                intent, snap, self.ledger.pending(self.account_id), metadata, self.risk_policy
            )
            if not result.passed:
                raise LedgerError("risk rejected: " + "; ".join(result.details.values()))
            return self.ledger.submit(intent)

    def open_bar(self, bar: ExecutionBar, metadata: dict[str, InstrumentRisk]) -> list[Fill]:
        """At the open, fill market/marketable-limit/triggered-stop orders only.

        Intrabar high/low are deliberately not used to invent execution order.
        Capacity is a fraction of an explicitly supplied opening volume. Replays
        of the same event cannot reuse its liquidity, including after restart.
        """
        with self.ledger.atomic():
            if not self.ledger.record_event(
                self.account_id, "open:" + bar.event_id, bar.model_dump_json()
            ):
                return []
            self.ledger.mark(
                self.account_id,
                [
                    PriceMark(
                        instrument=bar.instrument,
                        price=bar.open,
                        fx_rate=bar.fx_rate,
                        as_of=bar.opened_at,
                    )
                ],
            )
            capacity = (
                bar.opening_volume * self.policy.opening_participation
                if bar.tradable
                else Decimal(0)
            )
            fills = []
            for state in self.ledger.pending(self.account_id):
                order = state["intent"]
                if order.instrument.instrument_id != bar.instrument.instrument_id:
                    continue
                if order.expires_at and bar.opened_at > order.expires_at:
                    self.ledger.cancel(self.account_id, order.intent_id, bar.opened_at, "expired")
                    continue
                if order.eligible_at > bar.opened_at:
                    continue
                if order.instrument != bar.instrument:
                    raise LedgerError("bar instrument differs from order identity")
                price = bar.open * (
                    1 + self.policy.slippage_bps / 10000 * (1 if order.side == "buy" else -1)
                )
                marketable = True
                if order.order_type == "limit":
                    marketable = (
                        price <= order.limit_price
                        if order.side == "buy"
                        else price >= order.limit_price
                    )
                elif order.order_type == "stop":
                    marketable = (
                        bar.open >= order.stop_price
                        if order.side == "buy"
                        else bar.open <= order.stop_price
                    )
                meta = metadata.get(bar.instrument.instrument_id)
                if meta is None:
                    raise LedgerError("missing execution metadata")
                qty = (min(state["remaining"], capacity) / meta.lot_size).to_integral_value(
                    rounding=ROUND_DOWN
                ) * meta.lot_size
                if order.tif == "FOK" and qty < state["remaining"]:
                    qty = Decimal(0)
                if marketable and qty > 0:
                    fee = self.policy.fee(order.instrument, qty, price, bar.fx_rate)
                    # Recheck the entire remaining order at the new price, not a small
                    # slice that could sidestep the projected exposure constraints.
                    projected = order.model_copy(
                        update={
                            "reference_price": price,
                            "fx_rate": bar.fx_rate,
                            "quantity": state["remaining"],
                            "estimated_fee": self.policy.fee(
                                order.instrument, state["remaining"], price, bar.fx_rate
                            ),
                        }
                    )
                    snap = self.ledger.snapshot(
                        self.account_id,
                        bar.opened_at,
                        self.risk_policy.max_price_age_seconds,
                        persist=False,
                    )
                    risk = evaluate_order(
                        projected,
                        snap,
                        self.ledger.pending(self.account_id),
                        metadata,
                        self.risk_policy,
                    )
                    if not risk.passed:
                        self.ledger.cancel(
                            self.account_id,
                            order.intent_id,
                            bar.opened_at,
                            "fill risk: " + "; ".join(risk.details.values()),
                            rejected=True,
                        )
                        continue
                    fill = Fill(
                        fill_id=f"{order.intent_id}@{bar.event_id}",
                        intent_id=order.intent_id,
                        account_id=self.account_id,
                        quantity=qty,
                        price=price,
                        fee=fee,
                        fx_rate=bar.fx_rate,
                        executed_at=bar.opened_at,
                        event_id=bar.event_id,
                    )
                    try:
                        self.ledger.apply_fill(fill)
                    except LedgerError as exc:
                        self.ledger.cancel(
                            self.account_id, order.intent_id, bar.opened_at, str(exc), rejected=True
                        )
                    else:
                        capacity -= qty
                        fills.append(fill)
                if order.tif in ("IOC", "FOK"):
                    self.ledger.cancel(
                        self.account_id,
                        order.intent_id,
                        bar.opened_at,
                        "unfilled immediate quantity",
                    )
            return fills

    def close_bar(self, bar: ExecutionBar):
        with self.ledger.atomic():
            if not self.ledger.record_event(
                self.account_id, "close:" + bar.event_id, bar.model_dump_json()
            ):
                return
            self.ledger.mark(
                self.account_id,
                [
                    PriceMark(
                        instrument=bar.instrument,
                        price=bar.close,
                        fx_rate=bar.fx_rate,
                        as_of=bar.closed_at,
                    )
                ],
            )
            for state in self.ledger.pending(self.account_id):
                o = state["intent"]
                if o.expires_at and o.expires_at <= bar.closed_at:
                    self.ledger.cancel(self.account_id, o.intent_id, bar.closed_at, "expired")


def execution_bar_from_market(
    bar,
    *,
    event_id: str,
    opening_volume: Decimal,
    session_open,
    session_close,
    calendar_version: str,
    fx_rate: Decimal = Decimal(1),
) -> ExecutionBar:
    """Explicit adapter over data lane MarketBar, retaining Decimal precision.

    Opening volume and session boundaries must come from the data/calendar
    artifact; whole-day volume is not silently substituted for opening liquidity.
    """
    bar.require_execution_price()
    return ExecutionBar(
        event_id=event_id,
        instrument=bar.instrument,
        opened_at=bar.opened_at,
        closed_at=bar.closed_at,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        opening_volume=opening_volume,
        fx_rate=fx_rate,
        calendar=bar.calendar,
        calendar_version=calendar_version,
        session=bar.session,
        session_open=session_open,
        session_close=session_close,
    )
