"""SYNTHETIC acceptance fixture: no signals, external data, or live capital."""

import json
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from src.domain.portfolio import InstrumentRisk, OrderIntent
from src.execution.lean_algorithm import AIQuantStackAlgorithm


class SmokeAlgorithm(AIQuantStackAlgorithm):
    def Initialize(self):
        self.started = time.monotonic()
        self.prepared = False
        self.events = []
        self.finished = False
        super().Initialize()
        from QuantConnect.Securities import Cash

        exact = Decimal("100000.000000000000000000001")
        original = self._qal_port._native_decimal(exact)
        cash = Cash("USD", original, self._qal_port._native_decimal(1))
        restored = self._qal_port._money(cash, "Amount")
        assert restored == exact
        Path("/smoke/exact-decimal.json").write_text(
            json.dumps(
                {
                    "original": str(exact),
                    "returned": str(restored),
                    "python_property_type": str(type(cash.Amount)),
                }
            )
        )
        from QuantConnect.Data.Market import Tick
        from System import DateTime, DateTimeKind

        tick = Tick()
        ticks = 638712864000000001
        tick.Time = DateTime(ticks, DateTimeKind.Utc)
        assert self._qal_port._clock_reader.read(tick, "Time") == ticks
        assert self._qal_port._clock(tick, "Time").isoformat() == "2025-01-01T00:00:00.000001+00:00"
        Path("/smoke/exact-clock.json").write_text(
            json.dumps(
                {"ticks": ticks, "projected": self._qal_port._clock(tick, "Time").isoformat()}
            )
        )
        Path("/smoke/initialized.json").write_text(json.dumps(self._qal_port.identity()))

    def OnData(self, data):
        super().OnData(data)
        if self.finished:
            return
        state = self._qal_bridge.step()
        if state["status"] == "ready" and not self.prepared:
            item = self._qal_port.deployment.instruments[0]
            security = self.Securities[self._qal_port.symbols[item.instrument.instrument_id]]
            if security.Price <= 0:
                return
            now = self._qal_port.now()
            intent = OrderIntent(
                intent_id="smoke-buy",
                account_id=self._qal_execution.account_id,
                run_id="synthetic-lean-runtime-smoke",
                instrument=item.instrument,
                side="buy",
                quantity="1",
                reference_price=str(self._qal_port._money(security, "Price")),
                estimated_fee="1",
                decision_at=now,
                eligible_at=now + timedelta(seconds=1),
                strategy_id="synthetic-smoke",
            )
            risk = InstrumentRisk(
                instrument=item.instrument, sector="synthetic", adv_base="1000000000", as_of=now
            )
            prepared = self._qal_execution.prepare(
                intent, {item.instrument.instrument_id: risk}, as_of=now
            )
            Path("/smoke/prepared.json").write_text(json.dumps(prepared, default=str))
            self.prepared = True
        if self.prepared:
            order = self._qal_execution.state("smoke-buy")
            if order["state"] == "filled":
                self._qal_execution.ledger.reconcile(self._qal_execution.account_id)
                audit = self._qal_execution.ledger.execution_compliance(
                    self._qal_execution.account_id
                )
                Path("/smoke/result.json").write_text(
                    json.dumps(
                        {
                            "scope": "synthetic offline actual LEAN paper engine",
                            "order": order,
                            "events": self.events,
                            "audit": audit.model_dump(mode="json"),
                            "runtime": self._qal_port.identity(),
                        },
                        default=str,
                        indent=2,
                    )
                )
                self.finished = True
                self.Quit("synthetic paper fill received and reconciled")
        if time.monotonic() - self.started > 50 and not self.finished:
            Path("/smoke/blocked.json").write_text(
                json.dumps(
                    {"health": state, "prepared": self.prepared, "events": self.events}, default=str
                )
            )
            self.Quit("smoke timeout without accepted fill")

    def OnOrderEvent(self, event):
        self.events.append(self._qal_port.event(event))
        super().OnOrderEvent(event)
