"""Actual-engine resting limit cancellation; synthetic feed, unchanged fill model."""

import json
import time
from datetime import timedelta
from decimal import Decimal as D
from pathlib import Path

from smoke_baseline import SmokeAlgorithm

from src.domain.portfolio import InstrumentRisk, OrderIntent
from src.execution.lean_algorithm import AIQuantStackAlgorithm


class SmokeAlgorithm(SmokeAlgorithm):
    def Initialize(self):
        super().Initialize()  # Includes the real CLR decimal and clock checks.
        self.resting = None
        self.cancel_requested = False
        self.snapshots = {}

    def snapshot(self):
        execution = self._qal_execution
        order = execution.ledger.order(execution.account_id, "lifecycle-limit")
        state = execution.state("lifecycle-limit")
        ticket = self._qal_port.lookup(state["client_id"], None)
        return {
            "state": state,
            "reserved": str(order["reserved"]),
            "remaining": str(order["remaining"]),
            "available_cash": str(execution.ledger.available_cash(execution.account_id)),
            "ticket": ticket,
            "capture": self._qal_port.capture(),
            "fills": execution.ledger.fills(execution.account_id),
        }

    def OnData(self, data):
        AIQuantStackAlgorithm.OnData(self, data)
        if self.finished:
            return
        execution, port = self._qal_execution, self._qal_port
        item = port.deployment.instruments[0]
        security = self.Securities[port.symbols[item.instrument.instrument_id]]
        now = port.now()
        if not self.prepared and security.HasData:
            price = port._money(security, "Price")
            assert price > D("1")
            intent = OrderIntent(
                intent_id="lifecycle-limit",
                account_id=execution.account_id,
                run_id="synthetic-lean-lifecycle",
                instrument=item.instrument,
                side="buy",
                quantity="1",
                order_type="limit",
                limit_price="1.00",
                reference_price=price,
                estimated_fee="1",
                decision_at=now,
                eligible_at=now + timedelta(seconds=1),
                strategy_id="synthetic-lifecycle",
            )
            risk = InstrumentRisk(
                instrument=item.instrument, sector="synthetic", adv_base="1000000000", as_of=now
            )
            execution.prepare(intent, {item.instrument.instrument_id: risk}, as_of=now)
            self.snapshots["prepared"] = self.snapshot()
            self.prepared = True
        if self.prepared:
            execution.reconcile_all(as_of=now)
            state = execution.state("lifecycle-limit")
            if state["state"] == "new" and not self.cancel_requested:
                current = self.snapshot()
                ticket = current["ticket"]
                if ticket and ticket["status"] == "Submitted":
                    assert D(ticket["filled_quantity"]) == 0
                    assert D(current["reserved"]) > 0
                    assert port._money(security, "Price") > D("1")
                    if self.resting is None:
                        self.resting = now
                        self.snapshots["resting"] = current
                    elif now - self.resting >= timedelta(seconds=2):
                        self.snapshots["before_cancel"] = current
                        self.cancel_requested = True
                        execution.cancel("lifecycle-limit", as_of=now)
            if state["state"] == "cancelled":
                self.snapshots["cancelled"] = self.snapshot()
                reconciliation = execution.ledger.reconcile(execution.account_id)
                audit = execution.ledger.execution_compliance(execution.account_id)
                Path("/smoke/result.json").write_text(
                    json.dumps(
                        {
                            "scope": "synthetic offline actual LEAN paper cancellation",
                            "snapshots": self.snapshots,
                            "events": self.events,
                            "audit": audit.model_dump(mode="json"),
                            "reconciliation": reconciliation,
                            "runtime": port.identity(),
                            "fill_model": str(security.FillModel.GetType().FullName),
                        },
                        default=str,
                        indent=2,
                    )
                )
                self.finished = True
                self.Quit("canonical resting limit cancellation reconciled")
        if time.monotonic() - self.started > 50 and not self.finished:
            Path("/smoke/blocked.json").write_text(
                json.dumps({"snapshots": self.snapshots, "events": self.events}, default=str)
            )
            self.Quit("lifecycle timeout without accepted cancellation")
