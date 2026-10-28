"""Observed-open simulator used unchanged by paper and historical replay."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from src.domain.common import _utc
from src.domain.portfolio import (
    ExecutionBar,
    Fill,
    FillPolicy,
    InstrumentRisk,
    OpeningEvent,
    OpeningSourceIdentity,
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
        intent = OrderIntent.model_validate(intent.model_dump())
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

    def open_bar(
        self,
        bar: ExecutionBar,
        metadata: dict[str, InstrumentRisk],
        *,
        eligible_intent_ids: set[str] | None = None,
        intent_metadata: dict[str, dict[str, InstrumentRisk]] | None = None,
        source_identity: OpeningSourceIdentity | None = None,
    ) -> list[Fill]:
        """Historical completed-bar adapter over the one opening-fill primitive."""
        bar = ExecutionBar.model_validate(bar.model_dump())
        event = OpeningEvent.from_bar(bar)
        with self.ledger.atomic():
            old = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?",
                (self.account_id, "open:" + bar.event_id),
            ).fetchone()
            if old:
                if ExecutionBar.model_validate_json(old[0]) != bar:
                    raise LedgerError("completed opening bar identity changed")
                self.ledger.record_event(
                    self.account_id, "opening-event:" + event.event_id, event.model_dump_json()
                )
                return []
            fills = self.open_event(
                event,
                metadata,
                eligible_intent_ids=eligible_intent_ids,
                intent_metadata=intent_metadata,
                source_identity=source_identity,
            )
            self.ledger.record_event(self.account_id, "open:" + bar.event_id, bar.model_dump_json())
            return fills

    def open_event(
        self,
        event: OpeningEvent,
        metadata: dict[str, InstrumentRisk],
        *,
        eligible_intent_ids: set[str] | None = None,
        intent_metadata: dict[str, dict[str, InstrumentRisk]] | None = None,
        source_identity: OpeningSourceIdentity | None = None,
    ) -> list[Fill]:
        """At the open, fill market/marketable-limit/triggered-stop orders only.

        Intrabar high/low are deliberately not used to invent execution order.
        Capacity is a fraction of an explicitly supplied opening volume. Replays
        of the same event cannot reuse its liquidity, including after restart.
        """
        event = OpeningEvent.model_validate(event.model_dump())
        if self.ledger.spec(self.account_id).mode == "live":
            raise LedgerError("live accounts cannot receive simulated executions")
        with self.ledger.atomic():
            legacy = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?",
                (self.account_id, "open:" + event.event_id),
            ).fetchone()
            if legacy:
                if OpeningEvent.from_bar(ExecutionBar.model_validate_json(legacy[0])) != event:
                    raise LedgerError(
                        "opening print conflicts with previously consumed bar liquidity"
                    )
                self.ledger.record_event(
                    self.account_id, "opening-event:" + event.event_id, event.model_dump_json()
                )
                return []
            if not self.ledger.record_event(
                self.account_id, "opening-event:" + event.event_id, event.model_dump_json()
            ):
                return []
            from src.execution.opening_sources import require_new_event_source, source_binding

            require_new_event_source(self.ledger, self.account_id, event, source_identity)
            if (
                source_identity is not None
                and source_binding(self.ledger, self.account_id, source_identity) is not None
            ):
                raise LedgerError("native opening alias must settle through its observed receipt")
            self.ledger.mark(
                self.account_id,
                [
                    PriceMark(
                        instrument=event.instrument,
                        price=event.price,
                        fx_rate=event.fx_rate,
                        as_of=event.opened_at,
                    )
                ],
            )
            capacity = (
                event.opening_volume * self.policy.opening_participation
                if event.tradable
                else Decimal(0)
            )
            fills = []
            for state in self.ledger.pending(self.account_id):
                order = state["intent"]
                if eligible_intent_ids is not None and order.intent_id not in eligible_intent_ids:
                    continue
                if order.instrument.instrument_id != event.instrument.instrument_id:
                    continue
                if order.expires_at and event.opened_at >= order.expires_at:
                    self.ledger.cancel(self.account_id, order.intent_id, event.opened_at, "expired")
                    continue
                if order.eligible_at > event.opened_at:
                    continue
                if order.instrument != event.instrument:
                    raise LedgerError("bar instrument differs from order identity")
                order_metadata = (
                    metadata if intent_metadata is None else intent_metadata[order.intent_id]
                )
                price = event.price * (
                    1 + self.policy.slippage_bps / 10000 * (1 if order.side == "buy" else -1)
                )
                marketable = True
                if order.order_type == "limit":
                    marketable = (
                        price <= order.limit_price
                        if order.side == "buy"
                        else price >= order.limit_price
                    )
                elif order.order_type in ("stop", "trailing_stop"):
                    from src.execution.stops import observe_stop

                    marketable = (
                        event.tradable
                        and observe_stop(self.ledger, order, event).elected_event_id is not None
                    )
                meta = order_metadata.get(event.instrument.instrument_id)
                if meta is None:
                    raise LedgerError("missing execution metadata")
                qty = (min(state["remaining"], capacity) / meta.lot_size).to_integral_value(
                    rounding=ROUND_DOWN
                ) * meta.lot_size
                if order.tif == "FOK" and qty < state["remaining"]:
                    qty = Decimal(0)
                if marketable and qty > 0:
                    fee = self.policy.fee(order.instrument, qty, price, event.fx_rate)
                    # Recheck the entire remaining order at the new price, not a small
                    # slice that could sidestep the projected exposure constraints.
                    projected = order.model_copy(
                        update={
                            "reference_price": price,
                            "fx_rate": event.fx_rate,
                            "quantity": state["remaining"],
                            "estimated_fee": self.policy.fee(
                                order.instrument, state["remaining"], price, event.fx_rate
                            ),
                        }
                    )
                    snap = self.ledger.snapshot(
                        self.account_id,
                        event.opened_at,
                        self.risk_policy.max_price_age_seconds,
                        persist=False,
                    )
                    risk = evaluate_order(
                        projected,
                        snap,
                        self.ledger.pending(self.account_id),
                        order_metadata,
                        self.risk_policy,
                    )
                    if not risk.passed:
                        self.ledger.cancel(
                            self.account_id,
                            order.intent_id,
                            event.opened_at,
                            "fill risk: " + "; ".join(risk.details.values()),
                            rejected=True,
                        )
                        continue
                    fill = Fill(
                        fill_id=f"{order.intent_id}@{event.event_id}",
                        intent_id=order.intent_id,
                        account_id=self.account_id,
                        quantity=qty,
                        price=price,
                        fee=fee,
                        fx_rate=event.fx_rate,
                        executed_at=event.opened_at,
                        event_id=event.event_id,
                    )
                    try:
                        self.ledger.apply_fill(fill)
                    except LedgerError as exc:
                        self.ledger.cancel(
                            self.account_id,
                            order.intent_id,
                            event.opened_at,
                            str(exc),
                            rejected=True,
                        )
                    else:
                        capacity -= qty
                        fills.append(fill)
                if order.tif in ("IOC", "FOK") and (
                    order.order_type not in ("stop", "trailing_stop") or marketable
                ):
                    self.ledger.cancel(
                        self.account_id,
                        order.intent_id,
                        event.opened_at,
                        "unfilled immediate quantity",
                    )
            return fills

    def stop_state(self, intent_id: str):
        """Verified immutable local stop observation, including original election."""
        from src.execution.stops import read_stop_state

        return read_stop_state(self.ledger, self.account_id, intent_id)

    def close_bar(self, bar: ExecutionBar):
        bar = ExecutionBar.model_validate(bar.model_dump())
        with self.ledger.atomic():
            opening = OpeningEvent.from_bar(bar)
            consumed = self.ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?",
                (self.account_id, "opening-event:" + opening.event_id),
            ).fetchone()
            if consumed and OpeningEvent.model_validate_json(consumed[0]) != opening:
                raise LedgerError("completed bar differs from its consumed opening print")
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
            self.expire_orders(as_of=bar.closed_at)

    def expire_orders(self, *, as_of) -> tuple[str, ...]:
        """Cancel actually due orders at a causal boundary, without inventing a bar."""
        as_of = _utc(as_of)
        expired = []
        with self.ledger.atomic():
            for state in self.ledger.pending(self.account_id):
                order = state["intent"]
                if order.expires_at is not None and order.expires_at <= as_of:
                    self.ledger.cancel(self.account_id, order.intent_id, as_of, "expired")
                    expired.append(order.intent_id)
        return tuple(expired)


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
