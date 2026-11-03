"""Actual offline LEAN acceptance; no fabricated tickets or recovery authority."""

import json
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

from src.domain.portfolio import InstrumentRisk, OrderIntent
from src.execution.lean_algorithm import AIQuantStackAlgorithm

ROOT = Path("/smoke")


class RecoveryAlgorithm(AIQuantStackAlgorithm):
    def save(self, name, value):
        (ROOT / name).write_text(json.dumps(value, default=str, indent=2))

    def snapshot(self):
        execution = self._qal_execution
        return {
            "runtime": self._qal_port.identity(),
            "native_orders": [str(order.Id) for order in self.Transactions.GetOrders()],
            "bindings": [
                dict(row)
                for row in execution.ledger.db.execute("SELECT * FROM lean_ticket_bindings")
            ],
            "attempts": [
                dict(row)
                for row in execution.ledger.db.execute(
                    "SELECT * FROM events WHERE id LIKE 'lean-submit-attempt:%'"
                )
            ],
            "fills": execution.ledger.fills(execution.account_id),
            "outbox": [
                dict(row) for row in execution.ledger.db.execute("SELECT * FROM venue_outbox")
            ],
            "pending": execution.ledger.pending(execution.account_id),
        }

    def Initialize(self):
        self.phase = (ROOT / "phase").read_text().strip()
        self.started = time.monotonic()
        self.prepared_ids = set()
        self.done = False
        try:
            super().Initialize()
        except Exception as exc:
            self.save("initialization-error.json", {"error": str(exc), "snapshot": self.snapshot()})
            raise
        self.save("initialized.json", self.snapshot())

    def prepare(self, name, limit=False):
        now = self._qal_port.now()
        item = self._qal_port.deployment.instruments[0]
        security = self.Securities[self._qal_port.symbols[item.instrument.instrument_id]]
        if security.Price <= 0:
            return
        intent = OrderIntent(
            intent_id=name,
            account_id=self._qal_execution.account_id,
            run_id="synthetic-recovery-acceptance",
            instrument=item.instrument,
            side="buy",
            quantity="1",
            reference_price=str(self._qal_port._money(security, "Price")),
            estimated_fee="1",
            decision_at=now,
            eligible_at=now + timedelta(seconds=1),
            strategy_id="synthetic-recovery",
            **({"order_type": "limit", "limit_price": "1"} if limit else {}),
        )
        risk = InstrumentRisk(
            instrument=item.instrument, sector="synthetic", adv_base="1000000000", as_of=now
        )
        result = self._qal_execution.prepare(
            intent, {item.instrument.instrument_id: risk}, as_of=now
        )
        self.save(name + "-prepared.json", result)
        self.prepared_ids.add(name)

    def OnData(self, data):
        if self.done:
            return
        super().OnData(data)
        if self.phase == "completed-retry":
            before = self.snapshot()
            retries = [
                self._qal_execution.dispatch("recovery-filled", as_of=self._qal_port.now())
                for _ in range(3)
            ]
            self._qal_execution.reconcile_all(as_of=self._qal_port.now())
            self.save(
                "retry-result.json",
                {"before": before, "retries": retries, "after": self.snapshot()},
            )
            self.done = True
            self.Quit("completed receipt retries finished")
            return
        if "recovery-filled" not in self.prepared_ids:
            self.prepare("recovery-filled")
        elif self._qal_execution.state("recovery-filled")["state"] == "filled":
            if "recovery-pending" not in self.prepared_ids:
                before = self.snapshot()
                retries = [
                    self._qal_execution.dispatch("recovery-filled", as_of=self._qal_port.now())
                    for _ in range(3)
                ]
                self._qal_execution.reconcile_all(as_of=self._qal_port.now())
                self.save(
                    "retry-result.json",
                    {"before": before, "retries": retries, "after": self.snapshot()},
                )
                self.save("completed-checkpoint.json", self.snapshot())
                with sqlite3.connect(str(ROOT / "completed.sqlite")) as target:
                    self._qal_execution.ledger.db.backup(target)
                self.prepare("recovery-pending", limit=True)
            elif self._qal_execution.state("recovery-pending")["state"] == "new":
                self.save("pending-checkpoint.json", self.snapshot())
                self.done = True
                self.Quit("original accepted pending ticket retained")
        if time.monotonic() - self.started > 60:
            self.save("timeout.json", self.snapshot())
            self.done = True
            self.Quit("acceptance timeout")
