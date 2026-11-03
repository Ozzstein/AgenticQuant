"""Fail-closed assertions for retained actual-engine lifecycle evidence.

Unit tests of this reader are contract tests, not runtime acceptance. Runtime
acceptance requires engine output, its SQLite ledger, and launch inspection.
"""

import argparse
import json
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path

from src.domain.portfolio import exact_decimal_product, exact_decimal_sum
from src.execution.ledger import PortfolioLedger


def validate_result(result):
    snapshots = result["snapshots"]
    prepared = snapshots["prepared"]
    resting, before, final = (snapshots[key] for key in ("resting", "before_cancel", "cancelled"))
    original = resting["ticket"]
    client, ticket = original["client_order_id"], original["ticket_id"]
    assert client and ticket > 0
    reserve = D(prepared["reserved"])
    assert reserve > 0
    for snapshot in (resting, before, final):
        native = snapshot["ticket"]
        assert native["ticket_id"] == ticket and native["client_order_id"] == client
        assert snapshot["state"]["client_id"] == client
        assert native["order_type"] == "Limit" and D(native["limit_price"]) == D("1")
        assert D(native["quantity"]) == 1 and D(native["filled_quantity"]) == 0
        assert D(snapshot["remaining"]) == 1 and snapshot["fills"] == []
        assert snapshot["capture"]["quantities"] == {}
        assert D(snapshot["capture"]["cash_book"]["USD"]) == D("100000")
    for snapshot in (resting, before):
        assert snapshot["ticket"]["status"] == "Submitted"
        assert snapshot["state"]["state"] == "new"
        assert D(snapshot["reserved"]) == reserve
        assert D(snapshot["available_cash"]) == D("100000") - reserve
        assert snapshot["capture"]["open_client_order_ids"] == [client]
    assert (
        datetime.fromisoformat(before["capture"]["observed_at"])
        - datetime.fromisoformat(resting["capture"]["observed_at"])
    ).total_seconds() >= 2
    assert final["state"]["state"] == "cancelled"
    assert final["ticket"]["status"] == "Canceled"
    assert D(final["reserved"]) == 0 and D(final["available_cash"]) == D("100000")
    assert final["capture"]["open_orders"] == []
    events = final["ticket"]["events"]
    assert [event["status"] for event in events] == ["Submitted", "CancelPending", "Canceled"]
    for event in events:
        assert event["ticket_id"] == ticket and D(event["fill_quantity"]) == 0
        assert D(event["fee_amount"]) == 0
        assert event in result["events"]  # Actual callbacks also retained by bridge.
    assert result["audit"]["status"] == "PASS"
    assert result["audit"]["blocking_codes"] == []
    assert result["audit"]["lineage"] == "lean-paper"
    assert result["reconciliation"]["reconciled"]
    assert result["fill_model"] == "QuantConnect.Orders.Fills.ImmediateFillModel"


def verify(run):
    record = json.loads((run.parent / "launch.json").read_text())
    from order_lifecycle_launch import IMAGE

    assert record["image"] == IMAGE
    assert record["exit_code"] == 0 and record["cleanup_exit"] == 0
    assert not record.get("timed_out", False)
    inspection = record["inspection"]
    assert inspection["State"]["ExitCode"] == 0 and not inspection["State"]["OOMKilled"]
    host = inspection["HostConfig"]
    assert host["NetworkMode"] == "none" and host["CapDrop"] == ["ALL"]
    assert "no-new-privileges" in host["SecurityOpt"]
    assert host["Memory"] == 4 * 1024**3 and host["MemorySwap"] == 4 * 1024**3
    assert host["NanoCpus"] == 2 * 10**9 and host["PidsLimit"] == 256
    assert any(
        mount["Destination"] == "/app/src" and not mount["RW"] for mount in inspection["Mounts"]
    )
    result = json.loads((run / "result.json").read_text())
    validate_result(result)
    exact = json.loads((run / "exact-decimal.json").read_text())
    assert exact["original"] == exact["returned"] == "100000.000000000000000000001"
    clock = json.loads((run / "exact-clock.json").read_text())
    assert clock["ticks"] == 638712864000000001
    assert clock["projected"] == "2025-01-01T00:00:00.000001+00:00"
    ledger = PortfolioLedger(run / "ledger.sqlite", read_only=True)
    try:
        account = result["snapshots"]["cancelled"]["state"]["account"]
        order = ledger.order(account, "lifecycle-limit")
        intent = order["intent"]
        expected_reserve = exact_decimal_sum(
            [
                exact_decimal_product(
                    exact_decimal_product(intent.quantity, intent.reference_price), intent.fx_rate
                ),
                intent.estimated_fee,
            ]
        )
        assert D(result["snapshots"]["prepared"]["reserved"]) == expected_reserve
        assert order["reserved"] == 0 and order["status"] == "cancelled"
        assert ledger.available_cash(account) == D("100000")
        assert ledger.fills(account) == []
        assert ledger.reconcile(account)["reconciled"]
        audit = ledger.execution_compliance(account)
        assert audit.status == "PASS" and audit.lineage == "lean-paper"
        return {"status": "PASS", "audit": audit.model_dump(mode="json")}
    finally:
        ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    evidence = verify(args.run)
    (args.run.parent / "verification.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))
