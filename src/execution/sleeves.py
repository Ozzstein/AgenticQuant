"""Fill-derived sleeve ownership, internal netting and reconciled NAV projection.

This is not a second execution ledger: every external change references an
actual canonical fill/action. Internal trades and capital flows conserve the
book, are explicit events, and never pretend a requested order was filled.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, localcontext

from src.domain.common import _utc
from src.domain.portfolio import (
    AccountFee,
    CorporateAction,
    DividendEntitlement,
    DividendPayment,
    DividendReceivable,
    Fill,
    Holding,
    OrderIntent,
    PortfolioSnapshot,
    PortfolioTarget,
    PriceMark,
    SleeveSnapshot,
)
from src.execution.fill_arithmetic import FillArithmetic
from src.execution.ledger import LedgerError, PortfolioLedger
from src.execution.routing import intent_for_delta, target_to_intents

D = Decimal
ZERO = D(0)
CASH_QUANTUM = D(".00000001")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def exact_sum(values):
    """Exact addition of finite Decimal projection parts, without a tolerance."""
    values = list(values)
    if not values:
        return ZERO
    precision = (
        max(v.adjusted() for v in values)
        - min(v.as_tuple().exponent for v in values)
        + len(str(len(values)))
        + 3
    )
    with localcontext() as ctx:
        ctx.prec = max(28, precision)
        return sum(values, ZERO)


def partition(total, weights, *, quantum=CASH_QUANTUM):
    """Fixed quantum for all but the final sorted owner; exact residual once."""
    weights = {k: D(v) for k, v in sorted(weights.items()) if v > 0}
    if not weights:
        if total:
            raise LedgerError("cannot allocate an economic amount without owners")
        return {}
    denominator = exact_sum(weights.values())
    result = {}
    for key, weight in list(weights.items())[:-1]:
        result[key] = (total * weight / denominator).quantize(quantum, rounding=ROUND_DOWN)
    result[next(reversed(weights))] = exact_sum([total, exact_sum(result.values()).copy_negate()])
    return result


def quantities(total, weights, quantum):
    """Largest remainder in actual quantity units; stable lexical tie-break."""
    weights = {k: D(v) for k, v in sorted(weights.items()) if v > 0}
    if total == 0:
        return {k: ZERO for k in weights}
    denominator = exact_sum(weights.values())
    if denominator <= 0 or total % quantum:
        raise LedgerError("invalid exact quantity allocation")
    units = int(total / quantum)
    raw = {k: D(units) * v / denominator for k, v in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = units - sum(counts.values())
    if not 0 <= remainder < len(counts):
        raise LedgerError("quantity allocation precision is insufficient")
    for key in sorted(counts, key=lambda k: (-(raw[k] - counts[k]), k))[:remainder]:
        counts[key] += 1
    return {k: D(v) * quantum for k, v in counts.items()}


class SleeveBook:
    """Idempotent projection in the canonical ledger's SQLite transaction."""

    def __init__(self, ledger: PortfolioLedger, account_id: str):
        self.ledger, self.account_id = ledger, account_id
        ledger.spec(account_id)
        if ledger.db.execute("PRAGMA query_only").fetchone()[0]:
            return
        # Executescript would commit an enclosing replay transaction. Individual
        # DDL statements preserve that transaction and its crash boundary.
        for sql in (
            "CREATE TABLE IF NOT EXISTS sleeve_events(account TEXT NOT NULL, id TEXT NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(account,id))",
            "CREATE TABLE IF NOT EXISTS sleeve_intents(account TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(account,id))",
            "CREATE TABLE IF NOT EXISTS sleeve_fees(account TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(account,id))",
            "CREATE TABLE IF NOT EXISTS sleeve_observations(account TEXT NOT NULL, sleeve TEXT NOT NULL, at TEXT NOT NULL, source_hash TEXT NOT NULL, nav TEXT NOT NULL, flow TEXT NOT NULL, PRIMARY KEY(account,sleeve,at,source_hash))",
            "CREATE TABLE IF NOT EXISTS sleeve_snapshot_receipts(account TEXT NOT NULL, at TEXT NOT NULL, source_hash TEXT NOT NULL, payload TEXT NOT NULL, receipt_hash TEXT NOT NULL, PRIMARY KEY(account,at,source_hash))",
        ):
            ledger.db.execute(sql)

    def _event(self, event_id, at, kind, payload):
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        old = self.ledger.db.execute(
            "SELECT at,kind,payload FROM sleeve_events WHERE account=? AND id=?",
            (self.account_id, event_id),
        ).fetchone()
        if old:
            if tuple(old) != (at.isoformat(), kind, serialized):
                raise LedgerError("sleeve event identity reused with changed content")
            return False
        previous = self.ledger.db.execute(
            "SELECT at FROM sleeve_events WHERE account=? ORDER BY rowid DESC LIMIT 1",
            (self.account_id,),
        ).fetchone()
        if previous and datetime.fromisoformat(previous[0]) > at:
            raise LedgerError("sleeve events cannot run backwards; reconcile executions first")
        self.ledger.db.execute(
            "INSERT INTO sleeve_events VALUES (?,?,?,?,?)",
            (self.account_id, event_id, at.isoformat(), kind, serialized),
        )
        return True

    def initialize(self, initial_capital_by_sleeve: dict[str, Decimal]):
        capital = {k: D(str(v)) for k, v in sorted(initial_capital_by_sleeve.items())}
        spec = self.ledger.spec(self.account_id)
        if (
            not capital
            or any(not k or not v.is_finite() or v < 0 for k, v in capital.items())
            or exact_sum(capital.values()) != spec.initial_cash
        ):
            raise LedgerError(
                "explicit nonnegative sleeve capital must sum exactly to initial account cash"
            )
        with self.ledger.atomic():
            previous = self.ledger.db.execute(
                "SELECT 1 FROM sleeve_events WHERE account=? AND id='initial'", (self.account_id,)
            ).fetchone()
            if not previous and (
                self.ledger.fills(self.account_id) or self.ledger.pending(self.account_id)
            ):
                raise LedgerError(
                    "sleeves require an empty initial portfolio or an explicit audited migration"
                )
            self._event(
                "initial",
                spec.opened_at,
                "initial",
                {
                    "capital": {k: str(v) for k, v in capital.items()},
                    "account_hash": spec.content_hash(),
                },
            )
        return self

    def _state(self, *, rows=None, contexts=None):
        rows = (
            list(
                self.ledger.db.execute(
                    "SELECT * FROM sleeve_events WHERE account=? ORDER BY rowid", (self.account_id,)
                )
            )
            if rows is None
            else rows
        )
        if not rows or rows[0]["kind"] != "initial":
            raise LedgerError("sleeve capital has not been initialized")
        initial = json.loads(rows[0]["payload"])
        if initial["account_hash"] != self.ledger.spec(self.account_id).content_hash():
            raise LedgerError("sleeve initial account source changed")
        cash = {k: D(v) for k, v in initial["capital"].items()}
        state = {
            "cash": cash,
            "quantities": {k: defaultdict(lambda: ZERO) for k in cash},
            "opened": {k: {} for k in cash},
            "flow": dict.fromkeys(cash, ZERO),
            "fees": dict.fromkeys(cash, ZERO),
            "realized": dict.fromkeys(cash, ZERO),
            "taxes": dict.fromkeys(cash, ZERO),
            "dividends": {},
            "dividend_allocations": {},
            "receivables": {},
            "turnover": defaultdict(lambda: defaultdict(lambda: ZERO)),
            "last": dict.fromkeys(cash, None),
            "allocated": defaultdict(lambda: defaultdict(lambda: ZERO)),
            "rows": rows,
        }
        contexts = self._source_context() if contexts is None else contexts
        for row in rows[1:]:
            payload, at = json.loads(row["payload"]), datetime.fromisoformat(row["at"])
            if row["kind"] == "fill":
                stored = self.ledger.db.execute(
                    "SELECT * FROM fills WHERE account=? AND id=?",
                    (self.account_id, payload["fill_id"]),
                ).fetchone()
                if stored is None or digest(dict(stored)) != payload["source_hash"]:
                    raise LedgerError("sleeve source fill/economics changed or disappeared")
            if row["kind"] == "action":
                stored = self.ledger.db.execute(
                    "SELECT * FROM journal WHERE account=? AND id=?",
                    (self.account_id, payload["journal_id"]),
                ).fetchone()
                if stored is None or digest(dict(stored)) != payload["source_hash"]:
                    raise LedgerError("sleeve source action changed or disappeared")
            if row["kind"] in ("fill", "action"):
                journal_id = (
                    "fill:" + payload["fill_id"] if row["kind"] == "fill" else payload["journal_id"]
                )
                source, source_cash, action_realized, source_fees, source_realized, source_taxes = (
                    contexts[journal_id]
                )
                if payload != self._economic_payload(
                    source,
                    state,
                    source_cash,
                    action_realized,
                    source_fees,
                    source_realized,
                    source_taxes,
                ):
                    raise LedgerError(
                        "sleeve economics differ from deterministic source allocation"
                    )
            for name, amount in payload.get("cash", {}).items():
                cash[name] = exact_sum([cash[name], D(amount)])
            for name, changes in payload.get("quantities", {}).items():
                for key, amount in changes.items():
                    old = state["quantities"][name][key]
                    new = exact_sum([old, D(amount)])
                    if new < 0:
                        raise LedgerError("sleeve ownership became short")
                    state["quantities"][name][key] = new
                    if old == 0 and new > 0:
                        state["opened"][name][key] = at
                    if new == 0:
                        state["opened"][name].pop(key, None)
                    if D(amount) and row["kind"] in ("fill", "internal_transfer"):
                        state["last"][name] = at
            if "dividend_entitlement" in payload:
                entry = DividendEntitlement.model_validate(payload["dividend_entitlement"])
                state["dividends"][entry.action_id] = entry
                state["dividend_allocations"][entry.action_id] = payload["dividend_allocations"]
                state["receivables"][entry.action_id] = {}
            for action, amounts in payload.get("receivables", {}).items():
                for name, amount in amounts.items():
                    previous = state["receivables"][action].get(name, ZERO)
                    value = exact_sum([previous, D(amount)])
                    if value < 0:
                        raise LedgerError("sleeve dividend receivable cannot be negative")
                    state["receivables"][action][name] = value
            for field in ("flow", "fees", "realized", "taxes"):
                for name, amount in payload.get(field, {}).items():
                    state[field][name] = exact_sum([state[field][name], D(amount)])
            for name, amount in payload.get("turnover", {}).items():
                state["turnover"][at.date().isoformat()][name] = exact_sum(
                    [state["turnover"][at.date().isoformat()][name], D(amount)]
                )
            for name, amount in payload.get("allocations", {}).items():
                state["allocated"][payload["intent_id"]][name] += D(amount)
        if any(v < 0 for v in cash.values()):
            raise LedgerError(
                "a sleeve has negative cash; explicit authorized capital reconciliation required"
            )
        return state

    def _intent(self, intent_id):
        row = self.ledger.db.execute(
            "SELECT payload FROM sleeve_intents WHERE account=? AND id=?",
            (self.account_id, intent_id),
        ).fetchone()
        if row is None:
            raise LedgerError("canonical execution has no explicit sleeve allocation")
        return json.loads(row[0])

    def register_intent(
        self,
        intent: OrderIntent,
        allocations: dict[str, Decimal],
        *,
        source_hash: str,
        quantity_quantum: Decimal,
    ):
        """Persist allocation BEFORE submission, atomically with the target plan."""
        state = self._state()
        allocations = {k: D(str(v)) for k, v in sorted(allocations.items()) if D(str(v)) > 0}
        if (
            intent.account_id != self.account_id
            or set(allocations) - state["cash"].keys()
            or exact_sum(allocations.values()) != intent.quantity
            or quantity_quantum <= 0
        ):
            raise LedgerError("intent allocation does not match canonical quantity/owners")
        if len(source_hash) != 64:
            raise LedgerError("intent allocation requires source SHA256")
        payload = json.dumps(
            {
                "intent": intent.model_dump(mode="json"),
                "allocations": {k: str(v) for k, v in allocations.items()},
                "source_hash": source_hash,
                "quantity_quantum": str(quantity_quantum),
            },
            sort_keys=True,
        )
        with self.ledger.atomic():
            previous = self.ledger.db.execute(
                "SELECT payload FROM sleeve_intents WHERE account=? AND id=?",
                (self.account_id, intent.intent_id),
            ).fetchone()
            if previous and previous[0] != payload:
                raise LedgerError("immutable sleeve intent allocation changed")
            self.ledger.db.execute(
                "INSERT OR IGNORE INTO sleeve_intents VALUES (?,?,?)",
                (self.account_id, intent.intent_id, payload),
            )

    def _source_context(self, journals=None):
        contexts, basis, cash = {}, defaultdict(lambda: ZERO), ZERO
        held = defaultdict(lambda: ZERO)
        total_fees, total_realized, total_taxes = ZERO, ZERO, ZERO
        if journals is None:
            journals = self.ledger.db.execute(
                "SELECT * FROM journal WHERE account=? ORDER BY sequence", (self.account_id,)
            )
        for row in journals:
            if row["kind"] in ("dividend_entitlement", "dividend_payment"):
                cash = exact_sum([cash, D(row["cash_delta"])])
            elif row["kind"] == "fill":
                fill = Fill.model_validate_json(row["payload"])
                cash = FillArithmetic(fill).add(cash, D(row["cash_delta"]))
            elif row["kind"] in ("deposit", "account_fee"):
                cash += D(row["cash_delta"])
            elif row["kind"] in ("dividend", "delist"):
                action = CorporateAction.model_validate_json(row["payload"])
                if held[action.instrument.instrument_id]:
                    cash += D(row["cash_delta"])
                elif D(row["cash_delta"]):
                    raise LedgerError("action cash without historical sleeve ownership")
            elif D(row["cash_delta"]):
                raise LedgerError("nonmonetary sleeve source cannot change cash")
            realized = ZERO
            if row["kind"] == "fill":
                fill = Fill.model_validate_json(row["payload"])
                arithmetic = FillArithmetic(fill)
                order = self.ledger.order(self.account_id, fill.intent_id)["intent"]
                economic = self.ledger.db.execute(
                    "SELECT * FROM fills WHERE account=? AND id=?", (self.account_id, fill.fill_id)
                ).fetchone()
                total_fees = arithmetic.add(total_fees, fill.fee)
                total_realized = arithmetic.add(total_realized, D(economic["realized"]))
                key = order.instrument.instrument_id
                if order.side == "buy":
                    basis[key] = arithmetic.add(
                        basis[key], arithmetic.add(D(economic["notional"]), fill.fee)
                    )
                    held[key] = arithmetic.add(held[key], fill.quantity)
                else:
                    basis[key] = (
                        ZERO
                        if held[key] == fill.quantity
                        else arithmetic.subtract(
                            basis[key],
                            arithmetic.subtract(D(economic["cash_delta"]), D(economic["realized"])),
                        )
                    )
                    held[key] = arithmetic.subtract(held[key], fill.quantity)
            elif row["kind"] == "split":
                action = CorporateAction.model_validate_json(row["payload"])
                held[action.instrument.instrument_id] *= action.value
            elif row["kind"] in ("dividend", "delist"):
                action = CorporateAction.model_validate_json(row["payload"])
                realized = D(row["cash_delta"])
                if action.kind == "delist":
                    realized -= basis[action.instrument.instrument_id]
                    basis[action.instrument.instrument_id] = ZERO
                    held[action.instrument.instrument_id] = ZERO
                total_realized += realized
            elif row["kind"] == "account_fee":
                fee = AccountFee.model_validate_json(row["payload"])
                realized = fee.amount.copy_negate()
                total_fees += fee.amount
                total_realized += realized
            elif row["kind"] == "inventory_fee":
                from src.execution.inventory_fees import verify_payload

                payload = json.loads(row["payload"])
                key = payload["fee"]["instrument"]["instrument_id"]
                _, fee = verify_payload(
                    self.ledger, self.account_id, payload, held[key], basis[key]
                )
                held[key] = exact_sum([held[key], fee.quantity.copy_negate()])
                basis[key] = exact_sum([basis[key], D(payload["removed_basis"]).copy_negate()])
                realized = D(payload["removed_basis"]).copy_negate()
                total_fees = exact_sum([total_fees, fee.base_value])
                total_realized = exact_sum([total_realized, realized])
            elif row["kind"] == "dividend_entitlement":
                entry = DividendEntitlement.model_validate_json(row["payload"])
                realized = entry.gross_amount
                total_realized = exact_sum([total_realized, realized])
            elif row["kind"] == "dividend_payment":
                payment = DividendPayment.model_validate_json(row["payload"])
                realized = exact_sum(
                    [payment.fee.copy_negate(), payment.withholding_tax.copy_negate()]
                )
                total_realized = exact_sum([total_realized, realized])
                total_fees = exact_sum([total_fees, payment.fee])
                total_taxes = exact_sum([total_taxes, payment.withholding_tax])
            contexts[row["id"]] = (
                dict(row),
                cash,
                realized,
                total_fees,
                total_realized,
                total_taxes,
            )
        return contexts

    def _economic_payload(
        self,
        journal,
        state,
        canonical_cash,
        action_realized,
        canonical_fees,
        canonical_realized,
        canonical_taxes,
    ):
        kind = journal["kind"]
        payload = {"cash": {}, "quantities": {}}
        if kind == "fill":
            fill = Fill.model_validate_json(journal["payload"])
            order = self.ledger.order(self.account_id, fill.intent_id)["intent"]
            allocation = self._intent(fill.intent_id)
            if OrderIntent.model_validate(allocation["intent"]) != order:
                raise LedgerError("allocated intent source changed")
            remaining = {
                k: D(v) - state["allocated"][fill.intent_id][k]
                for k, v in allocation["allocations"].items()
            }
            quantum = min(
                D(allocation["quantity_quantum"]),
                D(1).scaleb(fill.quantity.as_tuple().exponent),
            )
            assigned = quantities(fill.quantity, remaining, quantum)
            if any(v > remaining[k] for k, v in assigned.items()):
                raise LedgerError("partial fill exceeds sleeve remaining quantity")
            economic = self.ledger.db.execute(
                "SELECT * FROM fills WHERE account=? AND id=?",
                (self.account_id, fill.fill_id),
            ).fetchone()
            payload.update(
                fill_id=fill.fill_id,
                intent_id=fill.intent_id,
                source_hash=digest(dict(economic)),
                intent_source_hash=digest(allocation),
                allocations={k: str(v) for k, v in assigned.items()},
            )
            payload["quantities"] = {
                k: {order.instrument.instrument_id: str(v if order.side == "buy" else -v)}
                for k, v in assigned.items()
            }
            for field, value in (
                ("cash", D(economic["cash_delta"])),
                ("fees", fill.fee),
                ("realized", D(economic["realized"])),
                ("turnover", D(economic["notional"])),
            ):
                payload[field] = {k: str(v) for k, v in partition(value, assigned).items()}
        elif kind == "account_fee":
            fee = AccountFee.model_validate_json(journal["payload"])
            plan = self.ledger.db.execute(
                "SELECT payload FROM sleeve_fees WHERE account=? AND id=?",
                (self.account_id, fee.fee_id),
            ).fetchone()
            if plan is None:
                raise LedgerError("actual account fee requires explicit sleeve attribution")
            plan = json.loads(plan[0])
            amounts = {key: D(value) for key, value in plan["amounts"].items()}
            if (
                plan["fee"] != fee.model_dump(mode="json")
                or set(amounts) - state["cash"].keys()
                or any(value < 0 for value in amounts.values())
                or exact_sum(amounts.values()) != fee.amount
            ):
                raise LedgerError("account fee attribution differs from canonical source")
            payload.update(
                journal_id=journal["id"],
                source_hash=digest(dict(journal)),
                attribution_hash=digest(plan),
            )
            payload["cash"] = {key: str(value.copy_negate()) for key, value in amounts.items()}
            payload["fees"] = {key: str(value) for key, value in amounts.items()}
            payload["realized"] = dict(payload["cash"])
        elif kind in ("dividend_entitlement", "dividend_payment"):
            from src.execution.dividend_projection import economic_payload

            payload = economic_payload(self, journal, state)
        elif kind == "inventory_fee":
            from src.execution.inventory_fees import sleeve_payload

            payload = sleeve_payload(self, journal, state)
        else:
            action = CorporateAction.model_validate_json(journal["payload"])
            key = action.instrument.instrument_id
            owned = {
                name: positions[key]
                for name, positions in state["quantities"].items()
                if positions[key] > 0
            }
            payload.update(journal_id=journal["id"], source_hash=digest(dict(journal)))
            if kind in ("split", "delist"):
                payload["quantities"] = {
                    k: {key: str(v * (action.value - 1) if kind == "split" else -v)}
                    for k, v in owned.items()
                }
            if D(journal["cash_delta"]):
                payload["cash"] = {
                    k: str(v) for k, v in partition(D(journal["cash_delta"]), owned).items()
                }
        if kind in ("dividend", "delist"):
            payload["realized"] = {k: str(v) for k, v in partition(action_realized, owned).items()}
        # Canonical arithmetic has an explicit finite Decimal operation order.
        # Allocate each exact residual once, never compare with a tolerance.
        for field, total in (
            ("cash", canonical_cash),
            ("fees", canonical_fees),
            ("realized", canonical_realized),
            ("taxes", canonical_taxes),
        ):
            projected = exact_sum(
                [*state[field].values(), *(D(v) for v in payload.get(field, {}).values())]
            )
            residual = exact_sum([total, projected.copy_negate()])
            if residual:
                values = payload.setdefault(field, {})
                owner = sorted(values or payload.get("cash") or state["cash"])[-1]
                values[owner] = str(exact_sum([D(values.get(owner, "0")), residual]))
                payload[field + "_rounding"] = {owner: str(residual)}
        return payload

    def sync(self):
        """Project each committed/current-transaction economic source exactly once."""
        with self.ledger.atomic():
            state = self._state()
            known = {row["id"] for row in state["rows"]}
            contexts = self._source_context()
            for journal in self.ledger.db.execute(
                "SELECT * FROM journal WHERE account=? ORDER BY sequence", (self.account_id,)
            ):
                kind = journal["kind"]
                if (
                    kind
                    not in (
                        "fill",
                        "split",
                        "dividend",
                        "delist",
                        "account_fee",
                        "dividend_entitlement",
                        "dividend_payment",
                        "inventory_fee",
                    )
                    or "source:" + journal["id"] in known
                ):
                    continue
                (
                    _,
                    canonical_cash,
                    action_realized,
                    canonical_fees,
                    canonical_realized,
                    canonical_taxes,
                ) = contexts[journal["id"]]
                payload = self._economic_payload(
                    journal,
                    state,
                    canonical_cash,
                    action_realized,
                    canonical_fees,
                    canonical_realized,
                    canonical_taxes,
                )
                at = datetime.fromisoformat(journal["at"])
                self._event(
                    "source:" + journal["id"], at, kind if kind == "fill" else "action", payload
                )
                state = self._state()
            actual_account = self.ledger._account(self.account_id)
            for field in ("fees", "realized", "taxes"):
                if exact_sum(state[field].values()) != (
                    D(actual_account[field]) if field in actual_account.keys() else ZERO
                ):
                    raise LedgerError("projected sleeve " + field + " does not reconcile")
            actual = D(actual_account["cash"])
            if exact_sum(state["cash"].values()) != actual:
                raise LedgerError("projected sleeve cash does not reconcile to canonical cash")
            actual_receivables = {
                r.entitlement.action_id: r.outstanding
                for r in self.ledger.dividend_receivables(self.account_id)
            }
            projected_receivables = {
                key: exact_sum(amounts.values())
                for key, amounts in state["receivables"].items()
                if exact_sum(amounts.values())
            }
            if actual_receivables != projected_receivables:
                raise LedgerError("projected sleeve receivables differ from canonical entitlements")
            projected_quantities = defaultdict(lambda: ZERO)
            for positions in state["quantities"].values():
                for key, qty in positions.items():
                    projected_quantities[key] = exact_sum([projected_quantities[key], qty])
            stored = {
                r["instrument"]: D(r["quantity"])
                for r in self.ledger.db.execute(
                    "SELECT * FROM holdings WHERE account=?", (self.account_id,)
                )
            }
            if {k: v for k, v in projected_quantities.items() if v} != stored:
                raise LedgerError(
                    "projected sleeve ownership does not reconcile to canonical holdings"
                )
            return state

    def accrue_dividend(self, action_id, *, at):
        with self.ledger.atomic():
            self.sync()
            result = self.ledger.accrue_dividend(self.account_id, action_id, at=at)
            self.sync()
            return result

    def apply_dividend_payment(self, payment, *, as_of):
        with self.ledger.atomic():
            self.sync()
            result = self.ledger.apply_dividend_payment(payment, as_of=as_of)
            self.sync()
            return result

    def apply_dividend_distribution(self, distribution, *, as_of):
        with self.ledger.atomic():
            self.sync()
            result = self.ledger.apply_dividend_distribution(
                self.account_id, distribution, as_of=as_of
            )
            self.sync()
            return result

    def apply_paper_dividend_distribution(self, distribution, *, policy, as_of):
        with self.ledger.atomic():
            self.sync()
            result = self.ledger.apply_paper_dividend_distribution(
                self.account_id, distribution, policy=policy, as_of=as_of
            )
            self.sync()
            return result

    def apply_account_fee(
        self, fee: AccountFee, amounts: dict[str, D], *, attribution_source_hash: str
    ):
        """Explicitly attribute a posted fee; capital transfers cannot be implied.

        The caller derives amounts from its actual source/ownership policy. This
        binding is immutable, exact and separately audited from vendor content.
        """
        if fee.account_id != self.account_id or len(attribution_source_hash) != 64:
            raise LedgerError("fee account and attribution source hash required")
        amounts = {key: D(str(value)) for key, value in sorted(amounts.items())}
        if (
            any(not value.is_finite() or value < 0 for value in amounts.values())
            or exact_sum(amounts.values()) != fee.amount
        ):
            raise LedgerError("fee attribution must exactly conserve its actual amount")
        with self.ledger.atomic():
            previous = self.ledger.db.execute(
                "SELECT payload FROM sleeve_fees WHERE account=? AND id=?",
                (self.account_id, fee.fee_id),
            ).fetchone()
            if previous:
                saved = json.loads(previous[0])
                old_fee = AccountFee.model_validate(saved["fee"])
                if (
                    saved["amounts"] != {k: str(v) for k, v in amounts.items()}
                    or saved["attribution_source_hash"] != attribution_source_hash
                    or old_fee.model_dump(exclude={"observed_at"})
                    != fee.model_dump(exclude={"observed_at"})
                    or old_fee.observed_at > fee.observed_at
                ):
                    raise LedgerError("immutable account fee attribution changed")
                self.ledger.apply_account_fee(fee)
                self.sync()
                return False
            state = self.sync()
            if set(amounts) - state["cash"].keys():
                raise LedgerError("unknown fee sleeve")
            reserved = dict.fromkeys(state["cash"], ZERO)
            for order in self.ledger.pending(self.account_id):
                allocation = self._intent(order["intent"].intent_id)
                remaining = {
                    k: D(v) - state["allocated"][order["intent"].intent_id][k]
                    for k, v in allocation["allocations"].items()
                }
                for name, value in partition(order["reserved"], remaining).items():
                    reserved[name] = exact_sum([reserved[name], value])
            if any(amount > state["cash"][key] - reserved[key] for key, amount in amounts.items()):
                raise LedgerError("fee attribution exceeds unreserved sleeve cash")
            self.ledger.db.execute(
                "INSERT INTO sleeve_fees VALUES (?,?,?)",
                (
                    self.account_id,
                    fee.fee_id,
                    json.dumps(
                        {
                            "fee": fee.model_dump(mode="json"),
                            "amounts": {k: str(v) for k, v in amounts.items()},
                            "attribution_source_hash": attribution_source_hash,
                        },
                        sort_keys=True,
                    ),
                ),
            )
            self.ledger.apply_account_fee(fee)
            self.sync()
            return True

    def transfer_cash(self, event_id, from_sleeve, to_sleeve, amount, at, source_hash):
        at, amount = _utc(at), D(str(amount))
        with self.ledger.atomic():
            state = self.sync()
            if (
                from_sleeve == to_sleeve
                or {from_sleeve, to_sleeve} - state["cash"].keys()
                or amount <= 0
                or len(source_hash) != 64
            ):
                raise LedgerError("invalid explicit sleeve capital transfer")
            if self.ledger.pending(self.account_id):
                raise LedgerError("settle/cancel pending orders before moving sleeve capital")
            payload = {
                "cash": {from_sleeve: str(-amount), to_sleeve: str(amount)},
                "flow": {from_sleeve: str(-amount), to_sleeve: str(amount)},
                "source_hash": source_hash,
            }
            exists = self.ledger.db.execute(
                "SELECT 1 FROM sleeve_events WHERE account=? AND id=?",
                (self.account_id, "transfer:" + event_id),
            ).fetchone()
            if not exists and state["cash"][from_sleeve] < amount:
                raise LedgerError(
                    "sleeve capital is invested/unavailable; no fictional cash transfer"
                )
            self._event("transfer:" + event_id, at, "capital_transfer", payload)

    def net_target(
        self, target: PortfolioTarget, metadata, fill_policy, *, max_price_age_seconds=345600
    ):
        """Return exact external intents after explicit internal ownership trades."""
        with self.ledger.atomic():
            state = self.sync()
            target = PortfolioTarget.model_validate(target.model_dump())
            if set(target.sleeve_contributions) != set(state["cash"]):
                raise LedgerError(
                    "every sleeve, including retiring cash/holding owners, needs a contribution"
                )
            event_id = "target:" + target.target_id
            old = self.ledger.db.execute(
                "SELECT payload FROM sleeve_events WHERE account=? AND id=?",
                (self.account_id, event_id),
            ).fetchone()
            if old:
                previous = json.loads(old[0])
                if previous["target"] != target.model_dump(mode="json"):
                    raise LedgerError("sleeve target identity changed")
                return [OrderIntent.model_validate(v) for v in previous["intents"]]
            if self.ledger.pending(self.account_id):
                raise LedgerError("reconcile/cancel outstanding orders before sleeve netting")
            snapshot = self.ledger.snapshot(
                self.account_id, target.decision_at, max_price_age_seconds, persist=False
            )
            marks = {
                m.instrument.instrument_id: m
                for m in (
                    PriceMark.model_validate_json(row[0])
                    for row in self.ledger.db.execute(
                        "SELECT payload FROM marks WHERE account=?", (self.account_id,)
                    )
                )
            }
            orders = target_to_intents(
                target,
                snapshot,
                marks,
                metadata,
                fill_policy,
                max_price_age_seconds=max_price_age_seconds,
            )
            final = {p.instrument.instrument_id: p.quantity for p in snapshot.positions}
            for order in orders:
                final[order.instrument.instrument_id] = final.get(
                    order.instrument.instrument_id, ZERO
                ) + (order.quantity if order.side == "buy" else -order.quantity)
            allocations = {}
            net_orders = []
            for key in sorted(
                set(final) | {k for c in target.sleeve_contributions.values() for k in c}
            ):
                desired = (
                    quantities(
                        final.get(key, ZERO),
                        {s: c.get(key, ZERO) for s, c in target.sleeve_contributions.items()},
                        metadata[key].lot_size,
                    )
                    if final.get(key, ZERO)
                    else {}
                )
                preserved = {
                    name: values[key]
                    for name, values in target.sleeve_preserve_quantities.items()
                    if key in values
                }
                if any(qty != state["quantities"][name][key] for name, qty in preserved.items()):
                    raise LedgerError("preserved quantity differs from actual sleeve ownership")
                if preserved:
                    flexible = {
                        name: c.get(key, ZERO)
                        for name, c in target.sleeve_contributions.items()
                        if name not in preserved and c.get(key, ZERO) > 0
                    }
                    remainder = max(ZERO, final.get(key, ZERO) - exact_sum(preserved.values()))
                    desired = (
                        {**quantities(remainder, flexible, metadata[key].lot_size), **preserved}
                        if flexible
                        else preserved
                    )
                current_quantity = exact_sum(
                    state["quantities"][name][key] for name in state["cash"]
                )
                net_delta = exact_sum(desired.values()) - current_quantity
                if net_delta:
                    net_orders.append(
                        intent_for_delta(target, snapshot, marks[key], net_delta, fill_policy)
                    )
                delta = {
                    s: desired.get(s, ZERO) - state["quantities"][s][key] for s in state["cash"]
                }
                buyers = {s: v for s, v in delta.items() if v > 0}
                sellers = {s: -v for s, v in delta.items() if v < 0}
                for seller in sorted(sellers):
                    for buyer in sorted(buyers):
                        qty = min(sellers[seller], buyers[buyer])
                        if not qty:
                            continue
                        cash = qty * marks[key].price * marks[key].fx_rate
                        if state["cash"][buyer] < cash:
                            raise LedgerError(
                                "internal buyer lacks sleeve cash; explicit capital transfer or settled sale required"
                            )
                        payload = {
                            "cash": {seller: str(cash), buyer: str(-cash)},
                            "quantities": {seller: {key: str(-qty)}, buyer: {key: str(qty)}},
                            "target_hash": target.content_hash(),
                            "mark": marks[key].model_dump(mode="json"),
                            "source_hash": digest(target.sleeve_sources),
                        }
                        self._event(
                            f"cross:{target.target_id}:{key}:{seller}:{buyer}",
                            target.decision_at,
                            "internal_transfer",
                            payload,
                        )
                        sellers[seller] -= qty
                        buyers[buyer] -= qty
                        state = self._state()
                allocations[key] = {
                    "buy": {s: v for s, v in buyers.items() if v},
                    "sell": {s: v for s, v in sellers.items() if v},
                }
            orders = sorted(
                net_orders, key=lambda o: (o.side != "sell", o.instrument.instrument_id)
            )
            available_by_owner = dict(state["cash"])
            for order in orders:
                key = order.instrument.instrument_id
                owner_quantities = allocations[key][order.side]
                if order.side == "buy":
                    for owner, amount in partition(
                        order.quantity * order.reference_price * order.fx_rate
                        + order.estimated_fee,
                        owner_quantities,
                    ).items():
                        if available_by_owner[owner] < amount:
                            raise LedgerError(
                                "external purchase exceeds sleeve cash including costs"
                            )
                        available_by_owner[owner] = exact_sum(
                            [available_by_owner[owner], amount.copy_negate()]
                        )
                self.register_intent(
                    order,
                    owner_quantities,
                    source_hash=target.content_hash(),
                    quantity_quantum=metadata[key].lot_size,
                )
            self._event(
                event_id,
                target.decision_at,
                "target",
                {
                    "target": target.model_dump(mode="json"),
                    "intents": [o.model_dump(mode="json") for o in orders],
                    "source_hash": target.content_hash(),
                },
            )
            return orders

    def observation(self, at, *, source_hash=None) -> dict[str, SleeveSnapshot] | None:
        """Read one exact historical full receipt, never project current balances."""
        from src.execution.sleeve_observations import read_observation

        return read_observation(self, _utc(at), source_hash=source_hash)

    def snapshot(self, at, max_price_age_seconds=345600, *, persist=True):
        from src.execution.sleeve_observations import persist_observation

        at = _utc(at)
        with self.ledger.atomic():
            state = self.sync()
            canonical = self.ledger.snapshot(
                self.account_id, at, max_price_age_seconds, persist=False
            )
            pending = self.ledger.pending(self.account_id)
            history = list(
                self.ledger.db.execute(
                    "SELECT * FROM sleeve_observations WHERE account=? AND at<=? ORDER BY rowid",
                    (self.account_id, at.isoformat()),
                )
            )
            output = self._project_snapshot(at, state, canonical, pending, history)
            if persist:
                persist_observation(self, canonical, state, pending, history, output)
            return output

    def _project_snapshot(self, at, state, canonical, pending, history):
        """The same projection for current publication and frozen receipt checks."""
        if any(datetime.fromisoformat(row["at"]) > at for row in state["rows"]):
            raise LedgerError("sleeve observation cannot include future ownership events")
        names = sorted(state["cash"])
        positions = {name: [] for name in names}
        for held in canonical.positions:
            key = held.instrument.instrument_id
            owners = {
                name: state["quantities"][name][key]
                for name in names
                if state["quantities"][name][key]
            }
            values, basis = (
                partition(held.market_value, owners),
                partition(held.cost_basis, owners),
            )
            for name, qty in owners.items():
                positions[name].append(
                    Holding(
                        **{
                            **held.model_dump(),
                            "quantity": qty,
                            "market_value": values[name],
                            "cost_basis": basis[name],
                            "opened_at": state["opened"][name][key],
                        }
                    )
                )
        reserved = dict.fromkeys(names, ZERO)
        for order in pending:
            allocation = self._intent(order["intent"].intent_id)
            remaining = {
                k: D(v) - state["allocated"][order["intent"].intent_id][k]
                for k, v in allocation["allocations"].items()
            }
            for name, amount in partition(order["reserved"], remaining).items():
                reserved[name] = exact_sum([reserved[name], amount])
        nav = {
            name: exact_sum(
                [
                    state["cash"][name],
                    *(p.market_value for p in positions[name]),
                    *(amounts.get(name, ZERO) for amounts in state["receivables"].values()),
                ]
            )
            for name in names
        }
        rounding = exact_sum([canonical.nav, exact_sum(nav.values()).copy_negate()])
        # Finite canonical Decimal valuation can differ from exact sums of
        # its components. Assign that residual only to an economic owner;
        # an empty sleeve cannot gain funding or acquire a negative NAV.
        rounding_owner = None
        if rounding:
            candidates = [
                name for name in names if nav[name] > 0 and exact_sum([nav[name], rounding]) >= 0
            ]
            if not candidates:
                raise LedgerError("no funded sleeve can absorb canonical valuation residual")
            rounding_owner = candidates[-1]
            nav[rounding_owner] = exact_sum([nav[rounding_owner], rounding])
        source = [
            {
                "id": r["id"],
                "at": r["at"],
                "kind": r["kind"],
                "payload": json.loads(r["payload"]),
            }
            for r in state["rows"]
        ]
        source_hash = digest(
            {"events": source, "canonical_snapshot": canonical.model_dump(mode="json")}
        )
        output = {}
        for name in names:
            high_water = max(
                [nav[name]]
                + [
                    exact_sum([D(row["nav"]), state["flow"][name], D(row["flow"]).copy_negate()])
                    for row in history
                    if row["sleeve"] == name
                ]
            )
            leaf = PortfolioSnapshot(
                account_id=self.account_id + "/sleeve/" + name,
                as_of=at,
                cash=state["cash"][name],
                reserved_cash=reserved[name],
                nav=nav[name],
                high_water_mark=high_water,
                realized_pnl=state["realized"][name],
                fees=state["fees"][name],
                turnover_today=state["turnover"][at.date().isoformat()][name],
                positions=tuple(positions[name]),
                receivables=tuple(
                    DividendReceivable(
                        entitlement=DividendEntitlement.model_validate(
                            state["dividend_allocations"][action][name]
                        ),
                        outstanding=amounts[name],
                    )
                    for action, amounts in sorted(state["receivables"].items())
                    if amounts.get(name, ZERO) > 0
                ),
                withholding_taxes=state["taxes"][name],
            )
            output[name] = SleeveSnapshot(
                portfolio=leaf,
                net_capital_flow=state["flow"][name],
                last_rebalanced_at=state["last"][name],
                source_hash=source_hash,
                source_event_ids=tuple(r["id"] for r in state["rows"]),
                valuation_rounding=rounding if name == rounding_owner else ZERO,
            )
        if (
            exact_sum(s.portfolio.nav for s in output.values()) != canonical.nav
            or exact_sum(s.portfolio.cash for s in output.values()) != canonical.cash
        ):
            raise LedgerError("sleeve NAV/cash reconciliation failed")
        return output

    def performance_period(self, opening, ended_at, *, available_at=None):
        """Verified kwargs for AttributedPerformanceStore.record_nav.

        Its flow convention is end-of-period. Intermediate capital transfers
        require an observation boundary; they cannot be mislabeled investment
        return. Internal ownership trades are investments, not capital flows.
        """
        ended_at = _utc(ended_at)
        available_at = _utc(available_at) if available_at else ended_at
        if available_at < ended_at:
            raise LedgerError("performance cannot be available before its closing observation")
        closing = self.snapshot(ended_at)
        if set(opening) != set(closing):
            raise LedgerError("all actual sleeves need an opening performance observation")
        result = {}
        for name, start in opening.items():
            started_at = start.portfolio.as_of
            if started_at >= ended_at:
                raise LedgerError("performance interval must advance time")
            receipt = self.ledger.db.execute(
                "SELECT nav,flow FROM sleeve_observations WHERE account=? AND sleeve=? AND at=? AND source_hash=?",
                (self.account_id, name, started_at.isoformat(), start.source_hash),
            ).fetchone()
            if receipt is None or (D(receipt[0]), D(receipt[1])) != (
                start.portfolio.nav,
                start.net_capital_flow,
            ):
                raise LedgerError(
                    "opening performance observation is not bound to this sleeve book"
                )
            for row in self.ledger.db.execute(
                "SELECT id,at FROM sleeve_events WHERE account=? AND kind='capital_transfer' AND at>=? AND at<=?",
                (self.account_id, started_at.isoformat(), ended_at.isoformat()),
            ):
                if (
                    row["id"] not in start.source_event_ids
                    and datetime.fromisoformat(row["at"]) != ended_at
                ):
                    raise LedgerError(
                        "split reporting interval at every intermediate capital transfer"
                    )
            end = closing[name]
            source_id = f"sleeve-period:{self.account_id}:{name}:{started_at.isoformat()}:{ended_at.isoformat()}"
            result[name] = dict(
                started_at=started_at,
                ended_at=ended_at,
                available_at=available_at,
                opening_nav=start.portfolio.nav,
                closing_nav=end.portfolio.nav,
                end_flow=exact_sum([end.net_capital_flow, start.net_capital_flow.copy_negate()]),
                source_event_id=source_id,
                source_event_hash=digest(
                    {
                        "opening": start.model_dump(mode="json"),
                        "closing": end.model_dump(mode="json"),
                    }
                ),
            )
        return result
