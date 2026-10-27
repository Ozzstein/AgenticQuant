"""Transactional, restart-safe accounting for isolated paper/backtest accounts.

SQLite serializes writers with BEGIN IMMEDIATE. Economic identifiers are scoped
by account and immutable: retrying identical input is a no-op; changing input
under an existing identifier is an error. All cash and basis arithmetic is
Decimal, stored as text. Buy fees are capitalized into weighted-average basis.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from threading import RLock

from src.domain.common import Instrument, _utc
from src.domain.portfolio import (
    AccountFee,
    AccountSpec,
    ActionCoverage,
    CashDividend,
    CorporateAction,
    DividendEntitlement,
    DividendPayment,
    DividendReceivable,
    Fill,
    Holding,
    OrderIntent,
    PortfolioSnapshot,
    PriceMark,
    exact_decimal_sum,
)
from src.execution.fill_arithmetic import FillArithmetic

D = Decimal
ZERO = D(0)


class LedgerError(ValueError):
    """Invalid accounting transition or conflicting idempotency key."""


class PortfolioLedger:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = str(path)
        if self.path != ":memory:" and not read_only:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        source = Path(path).resolve().as_uri() + "?mode=ro" if read_only else self.path
        self.db = sqlite3.connect(
            source, uri=read_only, isolation_level=None, timeout=30, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self._lock = RLock()
        self._depth = 0
        self.db.execute("PRAGMA foreign_keys=ON")
        if read_only:
            self.db.execute("PRAGMA query_only=ON")
            return
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY, spec TEXT NOT NULL, cash TEXT NOT NULL,
            realized TEXT NOT NULL, fees TEXT NOT NULL, peak TEXT NOT NULL,
            last_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS holdings (
            account TEXT NOT NULL REFERENCES accounts(id), instrument TEXT NOT NULL,
            definition TEXT NOT NULL, quantity TEXT NOT NULL, basis TEXT NOT NULL,
            PRIMARY KEY(account,instrument));
        CREATE TABLE IF NOT EXISTS orders (
            account TEXT NOT NULL REFERENCES accounts(id), id TEXT NOT NULL,
            payload TEXT NOT NULL, status TEXT NOT NULL, filled TEXT NOT NULL,
            reserved TEXT NOT NULL, reason TEXT,
            PRIMARY KEY(account,id));
        CREATE TABLE IF NOT EXISTS fills (
            account TEXT NOT NULL, id TEXT NOT NULL, intent TEXT NOT NULL,
            payload TEXT NOT NULL, cash_delta TEXT NOT NULL, realized TEXT NOT NULL,
            notional TEXT NOT NULL, at TEXT NOT NULL,
            PRIMARY KEY(account,id), FOREIGN KEY(account,intent) REFERENCES orders(account,id));
        CREATE TABLE IF NOT EXISTS marks (
            account TEXT NOT NULL REFERENCES accounts(id), instrument TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(account,instrument));
        CREATE TABLE IF NOT EXISTS journal (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL REFERENCES accounts(id),
            id TEXT NOT NULL, kind TEXT NOT NULL, at TEXT NOT NULL, payload TEXT NOT NULL,
            cash_delta TEXT NOT NULL, UNIQUE(account,id));
        CREATE TABLE IF NOT EXISTS snapshots (
            account TEXT NOT NULL REFERENCES accounts(id), at TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(account,at));
        CREATE TABLE IF NOT EXISTS events (
            account TEXT NOT NULL REFERENCES accounts(id), id TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(account,id));
        CREATE TABLE IF NOT EXISTS protections (
            account TEXT NOT NULL REFERENCES accounts(id), instrument TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(account,instrument));
        CREATE TABLE IF NOT EXISTS runs (
            account TEXT NOT NULL REFERENCES accounts(id), id TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(account,id));
        CREATE TABLE IF NOT EXISTS dividend_terms (
            account TEXT NOT NULL REFERENCES accounts(id), action TEXT NOT NULL,
            revision TEXT NOT NULL, terms_hash TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(account,action,revision), UNIQUE(account,terms_hash));
        CREATE TABLE IF NOT EXISTS dividend_entitlements (
            account TEXT NOT NULL REFERENCES accounts(id), action TEXT NOT NULL,
            payload TEXT NOT NULL, outstanding TEXT NOT NULL, PRIMARY KEY(account,action));
        CREATE TABLE IF NOT EXISTS dividend_payments (
            account TEXT NOT NULL REFERENCES accounts(id), id TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(account,id));
        """)
        with self.atomic():
            if "taxes" not in {row[1] for row in self.db.execute("PRAGMA table_info(accounts)")}:
                self.db.execute("ALTER TABLE accounts ADD COLUMN taxes TEXT NOT NULL DEFAULT '0'")

    @contextmanager
    def atomic(self):
        """Nested savepoints allow a rejected fill to roll back independently."""
        with self._lock:
            level = self._depth
            self.db.execute("BEGIN IMMEDIATE" if level == 0 else f"SAVEPOINT tx{level}")
            self._depth += 1
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK" if level == 0 else f"ROLLBACK TO tx{level}")
                if level:
                    self.db.execute(f"RELEASE tx{level}")
                raise
            else:
                self.db.execute("COMMIT" if level == 0 else f"RELEASE tx{level}")
            finally:
                self._depth -= 1

    def close(self):
        self.db.close()

    @contextmanager
    def _reading(self):
        """One consistent read view, including a caller's uncommitted transaction."""
        with self._lock:
            outer = not self.db.in_transaction
            if outer:
                self.db.execute("BEGIN")
            try:
                yield
            except BaseException:
                if outer:
                    self.db.execute("ROLLBACK")
                raise
            else:
                if outer:
                    self.db.execute("COMMIT")

    def create_account(self, spec: AccountSpec):
        with self.atomic():
            existing = self.db.execute(
                "SELECT spec FROM accounts WHERE id=?", (spec.account_id,)
            ).fetchone()
            if existing:
                if AccountSpec.model_validate_json(existing[0]) != spec:
                    raise LedgerError("account already exists with different initial conditions")
                return
            self.db.execute(
                "INSERT INTO accounts(id,spec,cash,realized,fees,peak,last_at) VALUES (?,?,?,?,?,?,?)",
                (
                    spec.account_id,
                    spec.model_dump_json(),
                    str(spec.initial_cash),
                    "0",
                    "0",
                    str(spec.initial_cash),
                    spec.opened_at.isoformat(),
                ),
            )
            self._journal(
                spec.account_id,
                "initial",
                "deposit",
                spec.opened_at,
                spec.model_dump_json(),
                spec.initial_cash,
            )

    def spec(self, account: str) -> AccountSpec:
        return AccountSpec.model_validate_json(self._account(account)["spec"])

    def _account(self, account):
        row = self.db.execute("SELECT * FROM accounts WHERE id=?", (account,)).fetchone()
        if row is None:
            raise LedgerError(f"unknown account: {account}")
        return row

    def _clock(self, account: str, at: datetime):
        at = _utc(at)
        row = self._account(account)
        if at < datetime.fromisoformat(row["last_at"]):
            raise LedgerError("economic events cannot run backwards")
        self.db.execute("UPDATE accounts SET last_at=? WHERE id=?", (at.isoformat(), account))

    def _journal(self, account, event_id, kind, at, payload, delta):
        self.db.execute(
            "INSERT INTO journal(account,id,kind,at,payload,cash_delta) VALUES (?,?,?,?,?,?)",
            (account, event_id, kind, at.isoformat(), payload, str(delta)),
        )

    def record_event(self, account: str, event_id: str, payload: str) -> bool:
        """Call inside the same transaction as the event's mutations."""
        row = self.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?", (account, event_id)
        ).fetchone()
        if row:
            if row[0] != payload:
                raise LedgerError(f"conflicting event retry: {event_id}")
            return False
        self.db.execute("INSERT INTO events VALUES (?,?,?)", (account, event_id, payload))
        return True

    def submit(self, intent: OrderIntent):
        with self.atomic():
            old = self.db.execute(
                "SELECT payload FROM orders WHERE account=? AND id=?",
                (intent.account_id, intent.intent_id),
            ).fetchone()
            if old:
                if OrderIntent.model_validate_json(old[0]) != intent:
                    raise LedgerError("intent identifier reused with different content")
                return self.order(intent.account_id, intent.intent_id)
            self._clock(intent.account_id, intent.decision_at)
            spec = self.spec(intent.account_id)
            if intent.instrument.quote_currency == spec.base_currency and intent.fx_rate != 1:
                raise LedgerError("base currency instrument must have FX rate 1")
            reserved = ZERO
            if intent.side == "buy":
                reserved = (
                    intent.quantity * intent.reference_price * intent.fx_rate + intent.estimated_fee
                )
                if reserved > self.available_cash(intent.account_id):
                    raise LedgerError("insufficient unreserved cash")
            else:
                held = self.quantity(intent.account_id, intent.instrument.instrument_id)
                reserved_qty = sum(
                    (
                        o["remaining"]
                        for o in self.pending(intent.account_id)
                        if o["intent"].instrument.instrument_id == intent.instrument.instrument_id
                        and o["intent"].side == "sell"
                    ),
                    ZERO,
                )
                if intent.quantity > held - reserved_qty:
                    raise LedgerError("insufficient unreserved holdings")
            self.db.execute(
                "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                (
                    intent.account_id,
                    intent.intent_id,
                    intent.model_dump_json(),
                    "pending",
                    "0",
                    str(reserved),
                    None,
                ),
            )
            self._journal(
                intent.account_id,
                "order:" + intent.intent_id,
                "order",
                intent.decision_at,
                intent.model_dump_json(),
                ZERO,
            )
            return self.order(intent.account_id, intent.intent_id)

    def order(self, account: str, intent_id: str):
        row = self.db.execute(
            "SELECT * FROM orders WHERE account=? AND id=?", (account, intent_id)
        ).fetchone()
        if row is None:
            raise LedgerError("unknown order")
        intent = OrderIntent.model_validate_json(row["payload"])
        return {
            "intent": intent,
            "status": row["status"],
            "filled": D(row["filled"]),
            "remaining": exact_decimal_sum([intent.quantity, D(row["filled"]).copy_negate()]),
            "reserved": D(row["reserved"]),
            "reason": row["reason"],
        }

    def pending(self, account: str):
        rows = self.db.execute(
            "SELECT id FROM orders WHERE account=? AND status IN ('pending','partial') ORDER BY rowid",
            (account,),
        ).fetchall()
        return [self.order(account, r[0]) for r in rows]

    def available_cash(self, account: str):
        return exact_decimal_sum(
            [
                D(self._account(account)["cash"]),
                *(o["reserved"].copy_negate() for o in self.pending(account)),
            ]
        )

    def quantity(self, account: str, instrument_id: str):
        row = self.db.execute(
            "SELECT quantity FROM holdings WHERE account=? AND instrument=?",
            (account, instrument_id),
        ).fetchone()
        return D(row[0]) if row else ZERO

    def cancel(
        self, account: str, intent_id: str, at: datetime, reason="cancelled", rejected=False
    ):
        with self.atomic():
            order = self.order(account, intent_id)
            if order["status"] not in ("pending", "partial"):
                return
            self._clock(account, at)
            status = "rejected" if rejected else "cancelled"
            self.db.execute(
                "UPDATE orders SET status=?, reserved='0', reason=? WHERE account=? AND id=?",
                (status, reason, account, intent_id),
            )
            self._journal(account, "cancel:" + intent_id, status, _utc(at), reason, ZERO)

    def apply_fill(self, fill: Fill, *, observed_at: datetime | None = None) -> bool:
        """Apply an actual execution, optionally received after a newer market mark.

        Operational receipts may arrive late, but cannot reorder recorded money
        movements or invalidate published NAV. Such conflicts require explicit
        reconciliation; we never alter actual execution times to make them fit.
        """
        fill = Fill.model_validate(fill.model_dump())
        with self.atomic():
            old = self.db.execute(
                "SELECT payload FROM fills WHERE account=? AND id=?",
                (fill.account_id, fill.fill_id),
            ).fetchone()
            if old:
                if Fill.model_validate_json(old[0]) != fill:
                    raise LedgerError("fill identifier reused with different content")
                return False
            order = self.order(fill.account_id, fill.intent_id)
            intent = order["intent"]
            if order["status"] not in ("pending", "partial") or fill.quantity > order["remaining"]:
                raise LedgerError("fill exceeds open order quantity")
            if fill.executed_at < intent.eligible_at or (
                intent.expires_at and fill.executed_at > intent.expires_at
            ):
                raise LedgerError("fill outside eligible execution interval")
            if intent.order_type == "limit":
                if (intent.side == "buy" and fill.price > intent.limit_price) or (
                    intent.side == "sell" and fill.price < intent.limit_price
                ):
                    raise LedgerError("fill violates limit price")
            if (
                intent.instrument.quote_currency == self.spec(fill.account_id).base_currency
                and fill.fx_rate != 1
            ):
                raise LedgerError("base currency fill must have FX rate 1")
            booked_at = fill.executed_at
            if observed_at is not None:
                booked_at = _utc(observed_at)
                if self.spec(fill.account_id).mode == "backtest" or booked_at < fill.executed_at:
                    raise LedgerError("operational receipt must follow actual execution")
                latest = self.db.execute(
                    "SELECT at FROM journal WHERE account=? AND kind IN ('fill','split','dividend','delist','account_fee','dividend_entitlement','dividend_payment','inventory_fee') ORDER BY sequence DESC LIMIT 1",
                    (fill.account_id,),
                ).fetchone()
                if latest and datetime.fromisoformat(latest[0]) > fill.executed_at:
                    raise LedgerError("late execution conflicts with committed economic chronology")
                published = self.db.execute(
                    "SELECT at FROM snapshots WHERE account=? AND at>=? LIMIT 1",
                    (fill.account_id, fill.executed_at.isoformat()),
                ).fetchone()
                if published:
                    raise LedgerError(
                        "late execution requires explicit published NAV reconciliation"
                    )
                self.record_event(
                    fill.account_id,
                    "fill-receipt:" + fill.fill_id,
                    json.dumps(
                        {"fill_hash": fill.content_hash(), "observed_at": booked_at.isoformat()},
                        sort_keys=True,
                    ),
                )
            self._clock(fill.account_id, booked_at)
            row = self._account(fill.account_id)
            holding = self.db.execute(
                "SELECT * FROM holdings WHERE account=? AND instrument=?",
                (fill.account_id, intent.instrument.instrument_id),
            ).fetchone()
            qty = D(holding["quantity"]) if holding else ZERO
            basis = D(holding["basis"]) if holding else ZERO
            if (
                holding
                and Instrument.model_validate_json(holding["definition"]) != intent.instrument
            ):
                raise LedgerError("instrument identity changed without a security-master migration")
            notional = fill.base_notional
            arithmetic = FillArithmetic(fill)
            remaining = arithmetic.subtract(order["remaining"], fill.quantity)
            reserve = order["reserved"] * remaining / order["remaining"]
            realized = ZERO
            if intent.side == "buy":
                delta = arithmetic.subtract(arithmetic.negate(notional), fill.fee)
                other_reservations = (
                    D(row["cash"]) - self.available_cash(fill.account_id) - order["reserved"]
                )
                if arithmetic.add(D(row["cash"]), delta) < other_reservations + reserve:
                    raise LedgerError("fill would consume reserved cash or borrow")
                qty = arithmetic.add(qty, fill.quantity)
                basis = arithmetic.add(basis, arithmetic.add(notional, fill.fee))
            else:
                if fill.quantity > qty:
                    raise LedgerError("fill would create a short position")
                removed_basis = basis if fill.quantity == qty else basis * fill.quantity / qty
                delta = arithmetic.subtract(notional, fill.fee)
                if arithmetic.add(D(row["cash"]), delta) < 0:
                    raise LedgerError("sell fees exceed available cash")
                realized = arithmetic.subtract(delta, removed_basis)
                basis = arithmetic.subtract(basis, removed_basis)
                qty = arithmetic.subtract(qty, fill.quantity)
            self.db.execute(
                "UPDATE accounts SET cash=?,realized=?,fees=? WHERE id=?",
                (
                    str(arithmetic.add(D(row["cash"]), delta)),
                    str(arithmetic.add(D(row["realized"]), realized)),
                    str(arithmetic.add(D(row["fees"]), fill.fee)),
                    fill.account_id,
                ),
            )
            if qty:
                self.db.execute(
                    "INSERT OR REPLACE INTO holdings VALUES (?,?,?,?,?)",
                    (
                        fill.account_id,
                        intent.instrument.instrument_id,
                        intent.instrument.model_dump_json(),
                        str(qty),
                        str(basis),
                    ),
                )
            else:
                self.db.execute(
                    "DELETE FROM holdings WHERE account=? AND instrument=?",
                    (fill.account_id, intent.instrument.instrument_id),
                )
                self.db.execute(
                    "DELETE FROM protections WHERE account=? AND instrument=?",
                    (fill.account_id, intent.instrument.instrument_id),
                )
            self.db.execute(
                "UPDATE orders SET filled=?,reserved=?,status=? WHERE account=? AND id=?",
                (
                    str(arithmetic.add(order["filled"], fill.quantity)),
                    str(reserve),
                    "filled" if remaining == 0 else "partial",
                    fill.account_id,
                    fill.intent_id,
                ),
            )
            self.db.execute(
                "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?)",
                (
                    fill.account_id,
                    fill.fill_id,
                    fill.intent_id,
                    fill.model_dump_json(),
                    str(delta),
                    str(realized),
                    str(notional),
                    fill.executed_at.isoformat(),
                ),
            )
            self._journal(
                fill.account_id,
                "fill:" + fill.fill_id,
                "fill",
                fill.executed_at,
                fill.model_dump_json(),
                delta,
            )
            if fill.inventory_fees:
                from src.execution.inventory_fees import apply

                apply(self, fill, observed_at=booked_at)
            return True

    def mark(self, account: str, marks: list[PriceMark]):
        with self.atomic():
            for mark in sorted(marks, key=lambda m: m.as_of):
                self._clock(account, mark.as_of)
                if (
                    mark.instrument.quote_currency == self.spec(account).base_currency
                    and mark.fx_rate != 1
                ):
                    raise LedgerError("base currency mark must have FX rate 1")
                self.db.execute(
                    "INSERT OR REPLACE INTO marks VALUES (?,?,?)",
                    (account, mark.instrument.instrument_id, mark.model_dump_json()),
                )

    def apply_account_fee(self, fee: AccountFee) -> bool:
        """Book a separately posted actual base-currency expense exactly once."""
        with self.atomic():
            existing = self.db.execute(
                "SELECT payload FROM journal WHERE account=? AND id=?",
                (fee.account_id, "account-fee:" + fee.fee_id),
            ).fetchone()
            if existing:
                prior = AccountFee.model_validate_json(existing[0])
                if (
                    prior.model_dump(exclude={"observed_at"})
                    != fee.model_dump(exclude={"observed_at"})
                    or fee.observed_at < prior.observed_at
                ):
                    raise LedgerError("account fee identity/content changed")
                return False
            spec = self.spec(fee.account_id)
            if spec.base_currency != fee.currency:
                raise LedgerError("account fee requires actual base-currency amount")
            if fee.intent_id is not None:
                self.order(fee.account_id, fee.intent_id)
            self._clock(fee.account_id, fee.observed_at)
            row = self._account(fee.account_id)
            if fee.amount > self.available_cash(fee.account_id):
                raise LedgerError("posted fee exceeds unreserved cash; reconcile pending risk")
            if self.db.execute(
                "SELECT 1 FROM snapshots WHERE account=? AND at>=?",
                (fee.account_id, fee.observed_at.isoformat()),
            ).fetchone():
                raise LedgerError(
                    "posted fee conflicts with published NAV; use a later observed cutoff"
                )
            self.db.execute(
                "UPDATE accounts SET cash=?,realized=?,fees=? WHERE id=?",
                (
                    str(D(row["cash"]) - fee.amount),
                    str(D(row["realized"]) - fee.amount),
                    str(D(row["fees"]) + fee.amount),
                    fee.account_id,
                ),
            )
            self._journal(
                fee.account_id,
                "account-fee:" + fee.fee_id,
                "account_fee",
                fee.observed_at,
                fee.model_dump_json(),
                fee.amount.copy_negate(),
            )
            return True

    def snapshot(
        self, account: str, at: datetime, max_age_seconds: int, persist=True
    ) -> PortfolioSnapshot:
        at = _utc(at)
        with self.atomic():
            row = self._account(account)
            if at < datetime.fromisoformat(row["last_at"]):
                raise LedgerError("cannot value current state at an earlier time")
            positions = []
            for h in self.db.execute(
                "SELECT * FROM holdings WHERE account=? ORDER BY instrument", (account,)
            ):
                m = self.db.execute(
                    "SELECT payload FROM marks WHERE account=? AND instrument=?",
                    (account, h["instrument"]),
                ).fetchone()
                if m is None:
                    raise LedgerError(f"missing mark for {h['instrument']}")
                mark = PriceMark.model_validate_json(m[0])
                age = (at - mark.as_of).total_seconds()
                if not 0 <= age <= max_age_seconds:
                    raise LedgerError(f"stale/future mark for {h['instrument']}")
                positions.append(
                    Holding(
                        instrument=mark.instrument,
                        quantity=D(h["quantity"]),
                        cost_basis=D(h["basis"]),
                        price=mark.price,
                        fx_rate=mark.fx_rate,
                        marked_at=mark.as_of,
                        market_value=D(h["quantity"]) * mark.price * mark.fx_rate,
                        opened_at=self.position_opened_at(account, h["instrument"]),
                    )
                )
            receivables = self.dividend_receivables(account)
            nav = D(row["cash"]) + sum((p.market_value for p in positions), ZERO)
            if receivables:
                nav = exact_decimal_sum([nav, *(r.outstanding for r in receivables)])
            peak = max(D(row["peak"]), nav)
            turnover = sum(
                (
                    D(f[0])
                    for f in self.db.execute(
                        "SELECT notional FROM fills WHERE account=? AND substr(at,1,10)=?",
                        (account, at.date().isoformat()),
                    )
                ),
                ZERO,
            )
            snap = PortfolioSnapshot(
                account_id=account,
                as_of=at,
                cash=D(row["cash"]),
                reserved_cash=D(row["cash"]) - self.available_cash(account),
                nav=nav,
                high_water_mark=peak,
                realized_pnl=D(row["realized"]),
                fees=D(row["fees"]),
                turnover_today=turnover,
                positions=tuple(positions),
                receivables=tuple(receivables),
                withholding_taxes=D(row["taxes"]) if "taxes" in row.keys() else ZERO,
            )
            if persist:
                existing = self.db.execute(
                    "SELECT payload FROM snapshots WHERE account=? AND at=?",
                    (account, at.isoformat()),
                ).fetchone()
                if existing and PortfolioSnapshot.model_validate_json(existing[0]) != snap:
                    raise LedgerError("cannot rewrite an existing NAV observation")
                self._clock(account, at)
                self.db.execute("UPDATE accounts SET peak=? WHERE id=?", (str(peak), account))
                self.db.execute(
                    "INSERT OR IGNORE INTO snapshots VALUES (?,?,?)",
                    (account, at.isoformat(), snap.model_dump_json()),
                )
            return snap

    def register_cash_dividend(self, account, terms, *, as_of):
        from src.execution.dividends import register_terms

        return register_terms(self, account, terms, as_of=as_of)

    def cash_dividends(self, account, *, as_of) -> tuple[CashDividend, ...]:
        """Latest actual admitted terms per action, available by the requested cutoff."""
        from src.execution.dividends import admitted_terms

        with self._reading():
            return admitted_terms(self, account, as_of=as_of)

    def dividend_entitlements(self, account, *, as_of) -> tuple[DividendEntitlement, ...]:
        """Original durable ownership entries, independently reconciled before return."""
        from src.execution.dividends import recorded_entitlements

        with self._reading():
            return recorded_entitlements(self, account, as_of=as_of)

    def dividend_payments(self, account, *, as_of) -> tuple[DividendPayment, ...]:
        """Original payment receipts and clocks; later observation cannot replace them."""
        from src.execution.dividends import recorded_payments

        with self._reading():
            return recorded_payments(self, account, as_of=as_of)

    def record_action_coverage(self, account, coverage: ActionCoverage, *, as_of):
        """Retain actual coverage/gaps; recording is not proof of completeness."""
        coverage = ActionCoverage.model_validate(coverage.model_dump())
        as_of = _utc(as_of)
        if coverage.available_at > as_of or coverage.ended_at > as_of:
            raise LedgerError("action coverage cannot use future source availability/window")
        if coverage.account_id is not None and coverage.account_id != account:
            raise LedgerError("action coverage belongs to a different account")
        with self.atomic():
            self.spec(account)
            return self.record_event(
                account, "action-coverage:" + coverage.coverage_id, coverage.model_dump_json()
            )

    def accrue_dividend(self, account, action_id, *, at):
        from src.execution.dividends import accrue

        return accrue(self, account, action_id, at=at)

    def apply_dividend_payment(self, payment, *, as_of):
        from src.execution.dividends import apply_payment

        return apply_payment(self, payment, as_of=as_of)

    def apply_dividend_distribution(self, account, distribution, *, as_of):
        from src.execution.dividends import apply_distribution

        return apply_distribution(self, account, distribution, as_of=as_of)

    def dividend_receivables(self, account) -> list[DividendReceivable]:
        from src.execution.dividends import receivables

        return receivables(self, account)

    def apply_paper_dividend_distribution(self, account, distribution, *, policy, as_of):
        from src.execution.dividends import apply_paper_distribution

        return apply_paper_distribution(self, account, distribution, policy=policy, as_of=as_of)

    def position_opened_at(self, account: str, instrument_id: str) -> datetime | None:
        """Derive holding age from actual economics visible in this transaction.

        A split preserves age, a full exit resets it, and adding to a position
        does not pretend it is newly opened. No separate connection can see
        the same-step uncommitted fill used by a replay decision callback.
        """
        quantity, opened_at = ZERO, None
        for row in self.db.execute(
            "SELECT kind,payload FROM journal WHERE account=? ORDER BY sequence", (account,)
        ):
            if row["kind"] == "fill":
                fill = Fill.model_validate_json(row["payload"])
                intent = self.order(account, fill.intent_id)["intent"]
                if intent.instrument.instrument_id != instrument_id:
                    continue
                if intent.side == "buy":
                    if quantity == 0:
                        opened_at = fill.executed_at
                    quantity = FillArithmetic(fill).add(quantity, fill.quantity)
                else:
                    quantity = FillArithmetic(fill).subtract(quantity, fill.quantity)
            elif row["kind"] in ("split", "delist"):
                action = CorporateAction.model_validate_json(row["payload"])
                if action.instrument.instrument_id != instrument_id:
                    continue
                quantity = quantity * action.value if action.kind == "split" else ZERO
            elif row["kind"] == "inventory_fee":
                payload = json.loads(row["payload"])
                if payload["fee"]["instrument"]["instrument_id"] == instrument_id:
                    quantity = exact_decimal_sum(
                        [quantity, D(payload["fee"]["quantity"]).copy_negate()]
                    )
            if quantity == 0:
                opened_at = None
        return opened_at

    def corporate_action(self, account: str, action: CorporateAction):
        with self.atomic():
            if not self.record_event(
                account, "action:" + action.action_id, action.model_dump_json()
            ):
                return
            self._clock(account, action.effective_at)
            # Cancel resting orders before a split/delisting, rather than executing old units.
            if action.kind in ("split", "delist"):
                for order in self.pending(account):
                    if order["intent"].instrument.instrument_id == action.instrument.instrument_id:
                        self.cancel(
                            account,
                            order["intent"].intent_id,
                            action.effective_at,
                            "corporate action",
                        )
            row = self.db.execute(
                "SELECT * FROM holdings WHERE account=? AND instrument=?",
                (account, action.instrument.instrument_id),
            ).fetchone()
            delta = ZERO
            if row:
                qty, basis = D(row["quantity"]), D(row["basis"])
                if action.kind == "split":
                    self.db.execute(
                        "UPDATE holdings SET quantity=? WHERE account=? AND instrument=?",
                        (str(qty * action.value), account, action.instrument.instrument_id),
                    )
                    mark_row = self.db.execute(
                        "SELECT payload FROM marks WHERE account=? AND instrument=?",
                        (account, action.instrument.instrument_id),
                    ).fetchone()
                    if mark_row:
                        mark = PriceMark.model_validate_json(mark_row[0])
                        self.mark(
                            account,
                            [
                                PriceMark(
                                    instrument=mark.instrument,
                                    price=mark.price / action.value,
                                    fx_rate=mark.fx_rate,
                                    as_of=action.effective_at,
                                )
                            ],
                        )
                    self.db.execute(
                        "DELETE FROM protections WHERE account=? AND instrument=?",
                        (account, action.instrument.instrument_id),
                    )
                else:
                    delta = qty * action.value
                    acct = self._account(account)
                    realized = delta - basis if action.kind == "delist" else delta
                    self.db.execute(
                        "UPDATE accounts SET cash=?,realized=? WHERE id=?",
                        (
                            str(D(acct["cash"]) + delta),
                            str(D(acct["realized"]) + realized),
                            account,
                        ),
                    )
                    if action.kind == "delist":
                        self.db.execute(
                            "DELETE FROM holdings WHERE account=? AND instrument=?",
                            (account, action.instrument.instrument_id),
                        )
                        self.db.execute(
                            "DELETE FROM protections WHERE account=? AND instrument=?",
                            (account, action.instrument.instrument_id),
                        )
            self._journal(
                account,
                "action:" + action.action_id,
                action.kind,
                action.effective_at,
                action.model_dump_json(),
                delta,
            )

    def history(self, account: str) -> list[PortfolioSnapshot]:
        return [
            PortfolioSnapshot.model_validate_json(r[0])
            for r in self.db.execute(
                "SELECT payload FROM snapshots WHERE account=? ORDER BY at", (account,)
            )
        ]

    def fills(self, account: str) -> list[Fill]:
        return [
            Fill.model_validate_json(r[0])
            for r in self.db.execute(
                "SELECT payload FROM fills WHERE account=? ORDER BY at,rowid", (account,)
            )
        ]

    def reconcile(self, account: str) -> dict:
        """Independently replay persisted economic fills/actions to check cash/basis."""
        from src.execution.dividends import (
            current_terms,
            expected_entitlement,
            verify_payment,
            verify_projections,
        )

        cash, positions, realized, fees = ZERO, {}, ZERO, ZERO
        accrued, payments, outstanding, taxes = {}, {}, {}, ZERO
        expected_inventory_fees, seen_inventory_fees, instruments = {}, set(), {}
        pending_inventory_fees = []
        for row in self.db.execute(
            "SELECT * FROM journal WHERE account=? ORDER BY sequence", (account,)
        ):
            if pending_inventory_fees and row["kind"] != "inventory_fee":
                raise LedgerError("actual fill inventory fee coverage is incomplete or reordered")
            if row["kind"] in ("dividend_entitlement", "dividend_payment"):
                cash = exact_decimal_sum([cash, D(row["cash_delta"])])
            elif row["kind"] == "fill":
                f = Fill.model_validate_json(row["payload"])
                cash = FillArithmetic(f).add(cash, D(row["cash_delta"]))
            elif row["kind"] in ("deposit", "account_fee"):
                cash += D(row["cash_delta"])
            elif row["kind"] in ("dividend", "delist"):
                action = CorporateAction.model_validate_json(row["payload"])
                if positions.get(action.instrument.instrument_id, (ZERO, ZERO))[0]:
                    cash += D(row["cash_delta"])
                elif D(row["cash_delta"]):
                    raise LedgerError("action cash without historical ownership")
            elif D(row["cash_delta"]):
                raise LedgerError("nonmonetary journal entry cannot change cash")
            if row["kind"] == "fill":
                f = Fill.model_validate_json(row["payload"])
                arithmetic = FillArithmetic(f)
                intent = self.order(account, f.intent_id)["intent"]
                key = intent.instrument.instrument_id
                if key in instruments and instruments[key] != intent.instrument:
                    raise LedgerError("historical fill instrument identity changed")
                instruments[key] = intent.instrument
                qty, basis = positions.get(key, (ZERO, ZERO))
                amount = f.base_notional
                trade_realized = ZERO
                if intent.side == "buy":
                    movement = arithmetic.subtract(arithmetic.negate(amount), f.fee)
                    qty, basis = (
                        arithmetic.add(qty, f.quantity),
                        arithmetic.add(basis, arithmetic.add(amount, f.fee)),
                    )
                else:
                    removed = basis if f.quantity == qty else basis * f.quantity / qty
                    movement = arithmetic.subtract(amount, f.fee)
                    trade_realized = arithmetic.subtract(movement, removed)
                    qty, basis = (
                        arithmetic.subtract(qty, f.quantity),
                        arithmetic.subtract(basis, removed),
                    )
                realized = arithmetic.add(realized, trade_realized)
                stored_fill = self.db.execute(
                    "SELECT * FROM fills WHERE account=? AND id=?", (account, f.fill_id)
                ).fetchone()
                if (
                    f.account_id != account
                    or row["at"] != f.executed_at.isoformat()
                    or D(row["cash_delta"]) != movement
                    or stored_fill is None
                    or Fill.model_validate_json(stored_fill["payload"]) != f
                    or D(stored_fill["cash_delta"]) != movement
                    or D(stored_fill["realized"]) != trade_realized
                    or D(stored_fill["notional"]) != amount
                ):
                    raise LedgerError(
                        "fill cash/basis differs from exact persisted source economics"
                    )
                positions[key] = qty, basis
                fees = arithmetic.add(fees, f.fee)
                from src.execution.inventory_fees import fee_key

                for fee in f.inventory_fees:
                    identity = fee_key(f.fill_id, fee.instrument.instrument_id)
                    expected_inventory_fees[identity] = f.content_hash()
                    pending_inventory_fees.append(identity)
            elif row["kind"] == "inventory_fee":
                from src.execution.inventory_fees import fee_key, verify_payload

                payload = json.loads(row["payload"])
                key = payload["fee"]["instrument"]["instrument_id"]
                qty, basis = positions.get(key, (ZERO, ZERO))
                f, fee = verify_payload(self, account, payload, qty, basis)
                if instruments.get(key) != fee.instrument:
                    raise LedgerError("inventory fee historical instrument identity changed")
                if (
                    row["id"] != fee_key(f.fill_id, key)
                    or not pending_inventory_fees
                    or row["id"] != pending_inventory_fees[0]
                    or expected_inventory_fees.get(row["id"]) != f.content_hash()
                    or row["id"] in seen_inventory_fees
                    or row["at"] != f.executed_at.isoformat()
                    or D(row["cash_delta"]) != 0
                ):
                    raise LedgerError("inventory fee journal differs from its actual fill")
                pending_inventory_fees.pop(0)
                seen_inventory_fees.add(row["id"])
                positions[key] = (
                    exact_decimal_sum([qty, fee.quantity.copy_negate()]),
                    exact_decimal_sum([basis, D(payload["removed_basis"]).copy_negate()]),
                )
                fees = exact_decimal_sum([fees, fee.base_value])
                realized = exact_decimal_sum([realized, D(payload["removed_basis"]).copy_negate()])
            elif row["kind"] == "account_fee":
                fee = AccountFee.model_validate_json(row["payload"])
                if D(row["cash_delta"]) != fee.amount.copy_negate():
                    raise LedgerError("account fee cash entry differs from its source")
                realized -= fee.amount
                fees += fee.amount
            elif row["kind"] == "dividend_entitlement":
                entry = DividendEntitlement.model_validate_json(row["payload"])
                terms = current_terms(self, account, entry.action_id)
                if (
                    entry != expected_entitlement(self, account, terms)
                    or entry.account_id != account
                    or row["at"] != entry.at.isoformat()
                    or terms.available_at > entry.at
                    or D(row["cash_delta"]) != 0
                    or entry.action_id in accrued
                ):
                    raise LedgerError(
                        "dividend entitlement differs from historical source ownership"
                    )
                accrued[entry.action_id], outstanding[entry.action_id] = entry, entry.gross_amount
                realized = exact_decimal_sum([realized, entry.gross_amount])
            elif row["kind"] == "dividend_payment":
                payment = DividendPayment.model_validate_json(row["payload"])
                verify_payment(self, payment, as_of=payment.available_at)
                if (
                    payment.action_id not in accrued
                    or payment.payment_id in payments
                    or row["at"] != payment.available_at.isoformat()
                    or D(row["cash_delta"]) != payment.net_amount
                    or outstanding[payment.action_id] < payment.gross_amount
                ):
                    raise LedgerError("dividend payment journal does not reconcile")
                outstanding[payment.action_id] = exact_decimal_sum(
                    [outstanding[payment.action_id], payment.gross_amount.copy_negate()]
                )
                realized = exact_decimal_sum(
                    [realized, payment.fee.copy_negate(), payment.withholding_tax.copy_negate()]
                )
                fees = exact_decimal_sum([fees, payment.fee])
                taxes = exact_decimal_sum([taxes, payment.withholding_tax])
                payments[payment.payment_id] = payment
            elif row["kind"] in ("split", "dividend", "delist"):
                a = CorporateAction.model_validate_json(row["payload"])
                key = a.instrument.instrument_id
                qty, basis = positions.get(key, (ZERO, ZERO))
                if a.kind == "split":
                    positions[key] = qty * a.value, basis
                elif a.kind == "dividend":
                    realized += qty * a.value
                else:
                    realized += qty * a.value - basis
                    positions[key] = ZERO, ZERO
        stored = {
            r["instrument"]: (D(r["quantity"]), D(r["basis"]))
            for r in self.db.execute("SELECT * FROM holdings WHERE account=?", (account,))
        }
        expected = {k: v for k, v in positions.items() if v[0]}
        if set(expected_inventory_fees) != seen_inventory_fees:
            raise LedgerError("actual fill inventory fee coverage is incomplete")
        state = self._account(account)
        verify_projections(self, account, accrued, payments, outstanding)
        valid = (
            cash == D(state["cash"])
            and stored == expected
            and realized == D(state["realized"])
            and fees == D(state["fees"])
            and taxes == (D(state["taxes"]) if "taxes" in state.keys() else ZERO)
        )
        if not valid:
            raise LedgerError("ledger reconciliation failed")
        result = {
            "reconciled": True,
            "cash": str(cash),
            "realized_pnl": str(realized),
            "fees": str(fees),
            "positions": len(stored),
        }
        if accrued:
            result.update(
                receivables=str(exact_decimal_sum(outstanding.values())),
                withholding_taxes=str(taxes),
            )
        return result

    def save_run(self, account: str, run_id: str, payload: dict):
        with self.atomic():
            current = self.db.execute(
                "SELECT payload FROM runs WHERE account=? AND id=?", (account, run_id)
            ).fetchone()
            if current:
                old = json.loads(current[0])
                for key in ("manifest_hash", "manifest"):
                    if old.get(key) != payload.get(key):
                        raise LedgerError("run identity cannot replace its immutable manifest")
            self.db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
                (account, run_id, json.dumps(payload, sort_keys=True, allow_nan=False)),
            )
