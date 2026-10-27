"""Entitlements and payments in the canonical ledger transaction/journal.

No account cache, inferred payable-date cash or reconstructed target ownership.
This module is part of PortfolioLedger; sleeves project these same journal rows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal as D

from src.domain.common import _utc
from src.domain.portfolio import (
    CashDividend,
    CorporateAction,
    DividendEntitlement,
    DividendPayment,
    DividendReceivable,
    Fill,
    PaperDividendPolicy,
    exact_decimal_product,
    exact_decimal_sum,
)
from src.execution.ledger import LedgerError


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def terms_history(ledger, account, action_id):
    chain = []
    for row in ledger.db.execute(
        "SELECT * FROM dividend_terms WHERE account=? AND action=? ORDER BY rowid",
        (account, action_id),
    ):
        terms = CashDividend.model_validate_json(row["payload"])
        event = ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (account, "dividend-terms:" + row["terms_hash"]),
        ).fetchone()
        admission = ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (account, "dividend-admission:" + row["terms_hash"]),
        ).fetchone()
        admitted = json.loads(admission[0]) if admission else {}
        if (
            terms.content_hash() != row["terms_hash"]
            or terms.action_id != row["action"]
            or terms.revision_id != row["revision"]
            or terms.currency != ledger.spec(account).base_currency
            or event is None
            or CashDividend.model_validate_json(event[0]) != terms
            or admitted.get("terms_hash") != terms.content_hash()
            or not terms.available_at
            <= datetime.fromisoformat(admitted.get("as_of", "0001-01-01T00:00:00+00:00"))
            <= terms.entitlement_at
            or terms.supersedes_hash != (chain[-1].content_hash() if chain else None)
            or (
                chain
                and (
                    terms.available_at < chain[-1].available_at
                    or terms.instrument != chain[-1].instrument
                )
            )
        ):
            raise LedgerError("dividend terms/revision source chain changed")
        chain.append(terms)
    return chain


def current_terms(ledger, account, action_id):
    chain = terms_history(ledger, account, action_id)
    if not chain:
        raise LedgerError("dividend terms unavailable")
    return chain[-1]


def register_terms(ledger, account, terms: CashDividend, *, as_of):
    terms = CashDividend.model_validate(terms.model_dump())
    as_of = _utc(as_of)
    with ledger.atomic():
        chain = terms_history(ledger, account, terms.action_id)
        for old in chain:
            if old.revision_id == terms.revision_id:
                if old != terms:
                    raise LedgerError("dividend revision identity changed")
                return False
        if (
            terms.currency != ledger.spec(account).base_currency
            or terms.available_at > as_of
            or as_of > terms.entitlement_at
            or terms.entitlement_at < ledger.spec(account).opened_at
            or terms.supersedes_hash != (chain[-1].content_hash() if chain else None)
            or (
                chain
                and (
                    chain[-1].instrument != terms.instrument
                    or chain[-1].available_at > terms.available_at
                )
            )
        ):
            raise LedgerError(
                "dividend requires causal same-currency terms and an exact revision chain"
            )
        if ledger.db.execute(
            "SELECT 1 FROM dividend_entitlements WHERE account=? AND action=?",
            (account, terms.action_id),
        ).fetchone():
            raise LedgerError("post-entitlement revision requires explicit audited reconciliation")
        if as_of < datetime.fromisoformat(ledger._account(account)["last_at"]):
            raise LedgerError("dividend admission cannot backdate the account clock")
        ledger.db.execute(
            "INSERT INTO dividend_terms VALUES (?,?,?,?,?)",
            (
                account,
                terms.action_id,
                terms.revision_id,
                terms.content_hash(),
                terms.model_dump_json(),
            ),
        )
        ledger.record_event(
            account, "dividend-terms:" + terms.content_hash(), terms.model_dump_json()
        )
        ledger.record_event(
            account,
            "dividend-admission:" + terms.content_hash(),
            json.dumps(
                {"terms_hash": terms.content_hash(), "as_of": as_of.isoformat()}, sort_keys=True
            ),
        )
        return True


def ownership_before(ledger, account, instrument, at):
    quantity, sources = D(0), []
    for row in ledger.db.execute(
        "SELECT id,kind,at,payload FROM journal WHERE account=? ORDER BY sequence", (account,)
    ):
        if datetime.fromisoformat(row["at"]) >= at:
            continue
        if row["kind"] == "fill":
            fill = Fill.model_validate_json(row["payload"])
            intent = ledger.order(account, fill.intent_id)["intent"]
            if intent.instrument.instrument_id != instrument.instrument_id:
                continue
            if intent.instrument != instrument:
                raise LedgerError("dividend instrument differs from historical ownership")
            quantity += fill.quantity if intent.side == "buy" else fill.quantity.copy_negate()
        elif row["kind"] in ("split", "delist"):
            action = CorporateAction.model_validate_json(row["payload"])
            if action.instrument != instrument:
                continue
            quantity = quantity * action.value if action.kind == "split" else D(0)
        else:
            continue
        sources.append(dict(row))
    return quantity, _hash(sources)


def _fence(ledger, account, at):
    if ledger.db.execute(
        "SELECT 1 FROM snapshots WHERE account=? AND at>=?", (account, at.isoformat())
    ).fetchone():
        raise LedgerError("dividend cannot rewrite published NAV")
    ledger._clock(account, at)


def entitlement(ledger, account, action_id):
    row = ledger.db.execute(
        "SELECT payload FROM dividend_entitlements WHERE account=? AND action=?",
        (account, action_id),
    ).fetchone()
    if row is None:
        raise LedgerError("actual dividend ownership entitlement unavailable")
    result = DividendEntitlement.model_validate_json(row[0])
    if result.account_id != account or result.action_id != action_id:
        raise LedgerError("dividend entitlement identity changed")
    return result


def expected_entitlement(ledger, account, terms):
    quantity, ownership_hash = ownership_before(
        ledger, account, terms.instrument, terms.entitlement_at
    )
    return DividendEntitlement(
        account_id=account,
        action_id=terms.action_id,
        terms_hash=terms.content_hash(),
        instrument=terms.instrument,
        at=terms.entitlement_at,
        quantity=quantity,
        gross_amount=exact_decimal_product(quantity, terms.cash_per_unit),
        currency=terms.currency,
        ownership_hash=ownership_hash,
    )


def accrue(ledger, account, action_id, *, at):
    at = _utc(at)
    with ledger.atomic():
        terms = current_terms(ledger, account, action_id)
        if at != terms.entitlement_at or terms.available_at > at:
            raise LedgerError("dividend entitlement must use the actual causal pre-ex boundary")
        expected = expected_entitlement(ledger, account, terms)
        if ledger.db.execute(
            "SELECT 1 FROM dividend_entitlements WHERE account=? AND action=?", (account, action_id)
        ).fetchone():
            if entitlement(ledger, account, action_id) != expected:
                raise LedgerError("historical dividend ownership changed")
            return expected
        _fence(ledger, account, at)
        if ledger.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sleeve_events'"
        ).fetchone():
            for row in ledger.db.execute(
                "SELECT payload FROM sleeve_events WHERE account=? AND at=? AND kind='internal_transfer'",
                (account, at.isoformat()),
            ):
                changes = json.loads(row[0]).get("quantities", {})
                if any(D(v.get(terms.instrument.instrument_id, "0")) for v in changes.values()):
                    raise LedgerError(
                        "accrue entitlement before same-boundary sleeve ownership changes"
                    )
        for row in ledger.db.execute(
            "SELECT kind,payload FROM journal WHERE account=? AND at=? AND kind IN ('fill','split','delist')",
            (account, at.isoformat()),
        ):
            changed = (
                ledger.order(account, Fill.model_validate_json(row["payload"]).intent_id)[
                    "intent"
                ].instrument
                if row["kind"] == "fill"
                else CorporateAction.model_validate_json(row["payload"]).instrument
            )
            if changed == terms.instrument:
                raise LedgerError(
                    "accrue entitlement before same-boundary fills or ownership changes"
                )
        if ledger.quantity(account, terms.instrument.instrument_id) != expected.quantity:
            raise LedgerError("accrue entitlement before same-boundary fills or ownership changes")
        ledger.db.execute(
            "INSERT INTO dividend_entitlements VALUES (?,?,?,?)",
            (account, action_id, expected.model_dump_json(), str(expected.gross_amount)),
        )
        state = ledger._account(account)
        ledger.db.execute(
            "UPDATE accounts SET realized=? WHERE id=?",
            (str(exact_decimal_sum([D(state["realized"]), expected.gross_amount])), account),
        )
        ledger._journal(
            account,
            "dividend-entitlement:" + action_id,
            "dividend_entitlement",
            at,
            expected.model_dump_json(),
            D(0),
        )
        return expected


def verify_payment(ledger, payment, *, as_of):
    spec = ledger.spec(payment.account_id)
    original = entitlement(ledger, payment.account_id, payment.action_id)
    terms = current_terms(ledger, payment.account_id, payment.action_id)
    if (
        payment.currency != spec.base_currency
        or payment.terms_hash != original.terms_hash
        or terms.content_hash() != original.terms_hash
        or original.content_hash() != payment.entitlement_hash
        or payment.available_at > as_of
        or payment.available_at < original.at
        or payment.effective_date < original.at.date()
        or (terms.payable_date is not None and payment.effective_date < terms.payable_date)
        or (spec.mode == "backtest") != (payment.origin == "replay-distribution")
        or (payment.origin == "paper-distribution" and spec.mode != "paper")
    ):
        raise LedgerError("payment differs from actual entitlement, mode, terms or source clocks")
    if payment.origin in ("replay-distribution", "paper-distribution"):
        from src.domain.portfolio import DividendDistribution

        policy = None
        if payment.origin == "paper-distribution":
            require_local_paper(ledger, payment.account_id)
            policy_row = ledger.db.execute(
                "SELECT payload FROM events WHERE account=? AND id=?",
                (payment.account_id, "paper-dividend-policy:" + payment.sizing_policy_hash),
            ).fetchone()
            policy = PaperDividendPolicy.model_validate_json(policy_row[0]) if policy_row else None
            identity = (
                ledger.db.execute(
                    "SELECT payload FROM events WHERE account=? AND id=?",
                    (payment.account_id, "paper-dividend-policy-id:" + policy.policy_id),
                ).fetchone()
                if policy
                else None
            )
            if (
                policy is None
                or policy.content_hash() != payment.sizing_policy_hash
                or identity is None
                or PaperDividendPolicy.model_validate_json(identity[0]) != policy
            ):
                raise LedgerError("local paper distribution requires exact retained sizing policy")
        source = ledger.db.execute(
            "SELECT payload FROM events WHERE account=? AND id=?",
            (
                payment.account_id,
                "dividend-distribution:" + payment.payment_id.removeprefix("distribution:"),
            ),
        ).fetchone()
        if (
            source is None
            or DividendDistribution.model_validate_json(source[0]).payment(
                original, paper_policy=policy
            )
            != payment
        ):
            raise LedgerError("simulation payment requires the exact actual per-unit distribution")
    elif ledger.db.execute(
        "SELECT 1 FROM events WHERE account=? AND id LIKE 'paper-dividend-policy:%' LIMIT 1",
        (payment.account_id,),
    ).fetchone():
        raise LedgerError("local paper distribution account cannot mix broker-posted credits")
    return original


def require_local_paper(ledger, account):
    if (
        ledger.spec(account).mode != "paper"
        or ledger.db.execute(
            "SELECT 1 FROM events WHERE account=? AND id LIKE 'remote-binding:%' LIMIT 1",
            (account,),
        ).fetchone()
    ):
        raise LedgerError("per-unit paper sizing requires an exclusively local paper account")
    if ledger.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_books'"
    ).fetchone():
        book = ledger.db.execute(
            "SELECT owner,configuration FROM execution_books WHERE account=?", (account,)
        ).fetchone()
        if book and (
            book["owner"] != "direct" or json.loads(book["configuration"])["venues"] != ["paper"]
        ):
            raise LedgerError("per-unit paper sizing requires the local PaperVenue owner")
    for row in ledger.db.execute(
        "SELECT payload FROM dividend_payments WHERE account=?", (account,)
    ):
        if DividendPayment.model_validate_json(row[0]).origin != "paper-distribution":
            raise LedgerError("local paper sizing cannot mix actual broker-posted credits")


def require_remote_income_account(ledger, account):
    if ledger.db.execute(
        "SELECT 1 FROM events WHERE account=? AND id LIKE 'paper-dividend-policy:%' LIMIT 1",
        (account,),
    ).fetchone():
        raise LedgerError(
            "local paper distribution account cannot acquire a remote execution owner"
        )


def apply_paper_distribution(ledger, account, distribution, *, policy, as_of):
    from src.domain.portfolio import DividendDistribution

    policy = PaperDividendPolicy.model_validate(policy.model_dump())
    distribution = DividendDistribution.model_validate(distribution.model_dump())
    with ledger.atomic():
        require_local_paper(ledger, account)
        original = entitlement(ledger, account, distribution.action_id)
        payment = distribution.payment(original, paper_policy=policy)
        ledger.record_event(
            account, "paper-dividend-policy-id:" + policy.policy_id, policy.model_dump_json()
        )
        ledger.record_event(
            account, "paper-dividend-policy:" + policy.content_hash(), policy.model_dump_json()
        )
        ledger.record_event(
            account,
            "dividend-distribution:" + distribution.distribution_id,
            distribution.model_dump_json(),
        )
        return apply_payment(ledger, payment, as_of=as_of)


def apply_distribution(ledger, account, distribution, *, as_of):
    from src.domain.portfolio import DividendDistribution

    distribution = DividendDistribution.model_validate(distribution.model_dump())
    with ledger.atomic():
        if ledger.spec(account).mode != "backtest":
            raise LedgerError("per-unit distribution sizing is exclusive to historical replay")
        original = entitlement(ledger, account, distribution.action_id)
        ledger.record_event(
            account,
            "dividend-distribution:" + distribution.distribution_id,
            distribution.model_dump_json(),
        )
        return apply_payment(ledger, distribution.payment(original), as_of=as_of)


def apply_payment(ledger, payment: DividendPayment, *, as_of):
    payment = DividendPayment.model_validate(payment.model_dump())
    as_of = _utc(as_of)
    with ledger.atomic():
        verify_payment(ledger, payment, as_of=as_of)
        old = ledger.db.execute(
            "SELECT payload FROM dividend_payments WHERE account=? AND id=?",
            (payment.account_id, payment.payment_id),
        ).fetchone()
        if old:
            original = DividendPayment.model_validate_json(old[0])
            clocks = {"observed_at", "available_at"}
            if (
                original.model_dump(exclude=clocks) != payment.model_dump(exclude=clocks)
                or original.observed_at > payment.observed_at
                or original.available_at > payment.available_at
            ):
                raise LedgerError("dividend payment identity changed")
            return False
        row = ledger.db.execute(
            "SELECT outstanding FROM dividend_entitlements WHERE account=? AND action=?",
            (payment.account_id, payment.action_id),
        ).fetchone()
        remaining = exact_decimal_sum([D(row[0]), payment.gross_amount.copy_negate()])
        if remaining < 0:
            raise LedgerError("dividend payment exceeds unsettled entitlement")
        _fence(ledger, payment.account_id, payment.available_at)
        state = ledger._account(payment.account_id)
        ledger.db.execute(
            "UPDATE accounts SET cash=?,realized=?,fees=?,taxes=? WHERE id=?",
            (
                str(exact_decimal_sum([D(state["cash"]), payment.net_amount])),
                str(
                    exact_decimal_sum(
                        [
                            D(state["realized"]),
                            payment.fee.copy_negate(),
                            payment.withholding_tax.copy_negate(),
                        ]
                    )
                ),
                str(exact_decimal_sum([D(state["fees"]), payment.fee])),
                str(exact_decimal_sum([D(state["taxes"]), payment.withholding_tax])),
                payment.account_id,
            ),
        )
        ledger.db.execute(
            "UPDATE dividend_entitlements SET outstanding=? WHERE account=? AND action=?",
            (str(remaining), payment.account_id, payment.action_id),
        )
        ledger.db.execute(
            "INSERT INTO dividend_payments VALUES (?,?,?)",
            (payment.account_id, payment.payment_id, payment.model_dump_json()),
        )
        ledger._journal(
            payment.account_id,
            "dividend-payment:" + payment.payment_id,
            "dividend_payment",
            payment.available_at,
            payment.model_dump_json(),
            payment.net_amount,
        )
        return True


def receivables(ledger, account):
    if not ledger.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dividend_entitlements'"
    ).fetchone():
        return []  # Read-only pre-extension archive; no newly inferred entitlement.
    return [
        DividendReceivable(
            entitlement=DividendEntitlement.model_validate_json(row["payload"]),
            outstanding=D(row["outstanding"]),
        )
        for row in ledger.db.execute(
            "SELECT * FROM dividend_entitlements WHERE account=? ORDER BY action", (account,)
        )
        if D(row["outstanding"]) > 0
    ]


def verify_projections(ledger, account, accrued, payments, outstanding):
    if not ledger.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dividend_entitlements'"
    ).fetchone():
        if accrued or payments:
            raise LedgerError("dividend durable projections absent")
        return
    actual = {
        row["action"]: (
            DividendEntitlement.model_validate_json(row["payload"]),
            D(row["outstanding"]),
        )
        for row in ledger.db.execute(
            "SELECT * FROM dividend_entitlements WHERE account=?", (account,)
        )
    }
    if actual != {key: (value, outstanding[key]) for key, value in accrued.items()}:
        raise LedgerError("dividend receivable projection differs from exact journal")
    stored = {
        row["id"]: DividendPayment.model_validate_json(row["payload"])
        for row in ledger.db.execute("SELECT * FROM dividend_payments WHERE account=?", (account,))
    }
    if stored != payments:
        raise LedgerError("dividend payment projection differs from exact journal")
