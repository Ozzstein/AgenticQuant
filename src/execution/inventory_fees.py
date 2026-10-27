"""Actual asset debits attached to canonical fills, never invented cash conversions."""

import json
from datetime import datetime
from decimal import Decimal as D

from src.domain.common import Instrument
from src.domain.portfolio import Fill, InventoryFee, exact_decimal_sum
from src.execution.ledger import LedgerError


def fee_key(fill_id, instrument_id):
    # JSON-encoded structured IDs avoid collisions when native IDs contain ':'.
    return "inventory-fee:" + json.dumps([fill_id, instrument_id], separators=(",", ":"))


def economic_payload(fill, fee, quantity, basis):
    if quantity < fee.quantity or quantity <= 0:
        raise LedgerError("actual inventory fee exceeds owned quantity")
    removed = basis if quantity == fee.quantity else basis * fee.quantity / quantity
    return {
        "fill_id": fill.fill_id,
        "fill_hash": fill.content_hash(),
        "fee": fee.model_dump(mode="json"),
        "removed_basis": str(removed),
        "base_value": str(fee.base_value),
    }


def verify_payload(ledger, account, payload, quantity, basis):
    row = ledger.db.execute(
        "SELECT payload FROM fills WHERE account=? AND id=?", (account, payload["fill_id"])
    ).fetchone()
    if row is None:
        raise LedgerError("inventory fee lacks its actual source fill")
    fill = Fill.model_validate_json(row[0])
    fee = InventoryFee.model_validate(payload["fee"])
    if (
        fill.account_id != account
        or fee not in fill.inventory_fees
        or payload != economic_payload(fill, fee, quantity, basis)
    ):
        raise LedgerError("inventory fee differs from exact source/basis economics")
    receipt = ledger.db.execute(
        "SELECT payload FROM events WHERE account=? AND id=?",
        (account, "fill-receipt:" + fill.fill_id),
    ).fetchone()
    observed_at = fill.executed_at
    if receipt:
        source = json.loads(receipt[0])
        observed_at = datetime.fromisoformat(source["observed_at"])
        if source["fill_hash"] != fill.content_hash() or observed_at < fill.executed_at:
            raise LedgerError("inventory fee operational fill receipt changed")
    require_source(ledger, fill, fee, observed_at)
    return fill, fee


def require_source(ledger, fill, fee, observed_at):
    if fee.available_at > observed_at:
        raise LedgerError("inventory fee source is unavailable at actual accounting receipt")
    if (
        fee.instrument.quote_currency == ledger.spec(fill.account_id).base_currency
        and fee.fx_rate != 1
    ):
        raise LedgerError("base-currency inventory fee must have FX rate 1")


def apply(ledger, fill, *, observed_at):
    for fee in fill.inventory_fees:
        require_source(ledger, fill, fee, observed_at)
        row = ledger.db.execute(
            "SELECT * FROM holdings WHERE account=? AND instrument=?",
            (fill.account_id, fee.instrument.instrument_id),
        ).fetchone()
        if row is None or Instrument.model_validate_json(row["definition"]) != fee.instrument:
            raise LedgerError("inventory fee requires an actually owned exact instrument")
        quantity, basis = D(row["quantity"]), D(row["basis"])
        payload = economic_payload(fill, fee, quantity, basis)
        remaining = exact_decimal_sum([quantity, fee.quantity.copy_negate()])
        reserved = exact_decimal_sum(
            state["remaining"]
            for state in ledger.pending(fill.account_id)
            if state["intent"].side == "sell" and state["intent"].instrument == fee.instrument
        )
        if remaining < reserved:
            raise LedgerError("inventory fee conflicts with outstanding sell quantities")
        if remaining:
            ledger.db.execute(
                "UPDATE holdings SET quantity=?,basis=? WHERE account=? AND instrument=?",
                (
                    str(remaining),
                    str(exact_decimal_sum([basis, D(payload["removed_basis"]).copy_negate()])),
                    fill.account_id,
                    fee.instrument.instrument_id,
                ),
            )
        else:
            ledger.db.execute(
                "DELETE FROM holdings WHERE account=? AND instrument=?",
                (fill.account_id, fee.instrument.instrument_id),
            )
            ledger.db.execute(
                "DELETE FROM protections WHERE account=? AND instrument=?",
                (fill.account_id, fee.instrument.instrument_id),
            )
        state = ledger._account(fill.account_id)
        ledger.db.execute(
            "UPDATE accounts SET fees=?,realized=? WHERE id=?",
            (
                str(exact_decimal_sum([D(state["fees"]), fee.base_value])),
                str(
                    exact_decimal_sum(
                        [D(state["realized"]), D(payload["removed_basis"]).copy_negate()]
                    )
                ),
                fill.account_id,
            ),
        )
        ledger._journal(
            fill.account_id,
            fee_key(fill.fill_id, fee.instrument.instrument_id),
            "inventory_fee",
            fill.executed_at,
            json.dumps(payload, sort_keys=True),
            D(0),
        )


def sleeve_payload(book, journal, state):
    from src.execution.sleeves import digest, partition, quantities

    source = json.loads(journal["payload"])
    fee = InventoryFee.model_validate(source["fee"])
    original = next(
        (
            json.loads(row["payload"])
            for row in state["rows"]
            if row["kind"] == "fill" and json.loads(row["payload"])["fill_id"] == source["fill_id"]
        ),
        None,
    )
    if original is None:
        raise LedgerError("inventory fee lacks its actual sleeve fill ownership")
    assigned = quantities(
        fee.quantity,
        {k: D(v) for k, v in original["allocations"].items()},
        D(1).scaleb(fee.quantity.as_tuple().exponent),
    )
    key = fee.instrument.instrument_id
    if any(state["quantities"][name][key] < amount for name, amount in assigned.items()):
        raise LedgerError("fill owner lacks actual fee asset; explicit ownership transfer required")
    return {
        "journal_id": journal["id"],
        "source_hash": digest(dict(journal)),
        "fill_allocation_hash": digest(original),
        "cash": {},
        "quantities": {name: {key: str(amount.copy_negate())} for name, amount in assigned.items()},
        "fees": {name: str(value) for name, value in partition(fee.base_value, assigned).items()},
        "realized": {
            name: str(value)
            for name, value in partition(D(source["removed_basis"]).copy_negate(), assigned).items()
        },
    }
