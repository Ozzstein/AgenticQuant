"""Deterministic sleeve projection of the canonical entitlement/payment journal."""

from decimal import Decimal as D

from src.domain.portfolio import DividendEntitlement, DividendPayment, exact_decimal_product
from src.execution.ledger import LedgerError


def bounded_partition(total, weights):
    from src.execution.sleeves import exact_sum, partition

    weights = {k: v for k, v in sorted(weights.items()) if v > 0}
    if total < 0 or total > exact_sum(weights.values()):
        raise LedgerError("dividend allocation exceeds its original owners' receivables")
    if total == exact_sum(weights.values()):
        return dict(weights)
    allocated = partition(total, weights)
    for key in allocated:
        excess = exact_sum([allocated[key], weights[key].copy_negate()])
        if excess <= 0:
            continue
        allocated[key] = weights[key]
        for other in allocated:
            room = exact_sum([weights[other], allocated[other].copy_negate()])
            portion = min(room, excess)
            allocated[other] = exact_sum([allocated[other], portion])
            excess = exact_sum([excess, portion.copy_negate()])
        if excess:
            raise LedgerError("dividend residual has no entitled owner")
    if exact_sum(allocated.values()) != total:
        raise LedgerError("dividend owner allocation does not conserve the exact amount")
    return allocated


def economic_payload(book, journal, state):
    from src.execution.dividends import current_terms
    from src.execution.sleeves import digest, exact_sum

    payload = {
        "cash": {},
        "quantities": {},
        "journal_id": journal["id"],
        "source_hash": digest(dict(journal)),
    }
    if journal["kind"] == "dividend_entitlement":
        original = DividendEntitlement.model_validate_json(journal["payload"])
        terms = current_terms(book.ledger, book.account_id, original.action_id)
        owned = {
            name: quantities[original.instrument.instrument_id]
            for name, quantities in state["quantities"].items()
            if quantities[original.instrument.instrument_id] > 0
        }
        if (
            exact_sum(owned.values()) != original.quantity
            or original.action_id in state["dividends"]
        ):
            raise LedgerError("sleeve pre-entitlement ownership differs from canonical entitlement")
        parts = {
            name: exact_decimal_product(quantity, terms.cash_per_unit)
            for name, quantity in owned.items()
        }
        if exact_sum(parts.values()) != original.gross_amount:
            raise LedgerError("sleeve dividend income does not conserve exact entitlement")
        entries = {}
        for name, quantity in owned.items():
            entry = DividendEntitlement(
                account_id=book.account_id + "/sleeve/" + name,
                action_id=original.action_id,
                terms_hash=original.terms_hash,
                instrument=original.instrument,
                at=original.at,
                quantity=quantity,
                gross_amount=parts[name],
                currency=original.currency,
                parent_entitlement_hash=original.content_hash(),
                ownership_hash=digest(
                    {"parent": original.content_hash(), "sleeve": name, "quantity": str(quantity)}
                ),
            )
            entries[name] = entry.model_dump(mode="json")
        payload.update(
            dividend_entitlement=original.model_dump(mode="json"),
            dividend_allocations=entries,
            receivables={original.action_id: {k: str(v) for k, v in parts.items()}},
            realized={k: str(v) for k, v in parts.items()},
        )
    else:
        payment = DividendPayment.model_validate_json(journal["payload"])
        original = state["dividends"].get(payment.action_id)
        if original is None or original.content_hash() != payment.entitlement_hash:
            raise LedgerError("dividend payment lacks original sleeve entitlement")
        gross = bounded_partition(payment.gross_amount, state["receivables"][payment.action_id])
        fees = bounded_partition(payment.fee, gross)
        after_fee = {k: exact_sum([v, fees.get(k, D(0)).copy_negate()]) for k, v in gross.items()}
        taxes = bounded_partition(payment.withholding_tax, after_fee)
        net = {k: exact_sum([v, taxes.get(k, D(0)).copy_negate()]) for k, v in after_fee.items()}
        payload.update(
            receivables={payment.action_id: {k: str(v.copy_negate()) for k, v in gross.items()}},
            cash={k: str(v) for k, v in net.items()},
            fees={k: str(v) for k, v in fees.items()},
            taxes={k: str(v) for k, v in taxes.items()},
            realized={
                k: str(exact_sum([fees.get(k, D(0)), taxes.get(k, D(0))]).copy_negate())
                for k in gross
            },
        )
    return payload
