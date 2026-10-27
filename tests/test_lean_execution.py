"""Offline native runtime fixtures; these do not claim actual LEAN execution."""

import copy
import json
import subprocess
import sys
from datetime import timedelta
from decimal import Decimal as D

import pytest

from src.domain.portfolio import AccountSpec, FillPolicy, PriceMark
from src.execution.broker import create_typed_lean_execution
from src.execution.lean_bridge import LeanLaunchConfig, prepare_project, verify_project
from src.execution.lean_runtime import LeanOperatingBridge
from src.execution.lean_venue import LeanDeployment, LeanInstrument
from src.execution.ledger import LedgerError
from src.execution.venue import VenueRules
from tests.test_durable_accounting import A, B, T, intent, metadata


class RuntimeFixture:
    def __init__(self, deployment):
        self.deployment = deployment
        self.runtime_id, self.at = "fixture-lean-process-1", T
        self.cash, self.held, self.price = D(100000), D(0), D(100)
        self.ticket, self.calls, self.lose_ack = None, [], False

    @property
    def instrument(self):
        return self.deployment.instruments[0].instrument

    def identity(self):
        result = {
            "runtime_id": self.runtime_id,
            "deployment_hash": self.deployment.content_hash(),
            "live_mode": True,
            "engine_version": "fixture-runtime-only",
            "algorithm_id": "fixture",
        }
        if self.deployment.mode == "live":
            import hashlib

            result["native_credential_identity"] = hashlib.sha256(
                b"fixture-key-not-a-credential"
            ).hexdigest()
        return result

    def now(self):
        return self.at

    def capture(self):
        opened = (
            [self.ticket["client_order_id"]]
            if self.ticket and self.ticket["status"] in ("Submitted", "PartiallyFilled")
            else []
        )
        return {
            "runtime_id": self.runtime_id,
            "observed_at": self.at.isoformat(),
            "account_currency": "USD",
            "cash_book": {
                "USD": str(self.cash),
                **({"BTC": str(self.held)} if self.instrument.asset_class == "crypto" else {}),
            },
            "unsettled_cash": {},
            "margin_remaining": str(self.cash),
            "quantities": {self.instrument.instrument_id: str(self.held)} if self.held else {},
            "marks": [
                {
                    "instrument_id": self.instrument.instrument_id,
                    "price": str(self.price),
                    "as_of": self.at.isoformat(),
                    "data_type": "fixture",
                    "data": "fixture",
                }
            ],
            "open_client_order_ids": opened,
            "open_orders": [
                {"client_order_id": client, "ticket_id": 1, "broker_ids": ["native-fixture-ticket"]}
                for client in opened
            ],
        }

    def preflight(self, order):
        assert order.instrument == self.instrument

    def submit(self, order, client):
        self.calls.append(("submit", client))
        sign = D(1) if order.side == "buy" else D(-1)
        self.ticket = {
            "runtime_id": self.runtime_id,
            "ticket_id": 1,
            "client_order_id": client,
            "observed_at": self.at.isoformat(),
            "instrument_id": self.instrument.instrument_id,
            "quantity": str(sign * order.quantity),
            "filled_quantity": "0",
            "order_type": {"market": "Market", "limit": "Limit", "stop": "StopMarket"}[
                order.order_type
            ],
            "limit_price": str(order.limit_price) if order.limit_price else None,
            "stop_price": str(order.stop_price) if order.stop_price else None,
            "status": "Submitted",
            "broker_ids": ["native-fixture-ticket"],
            "events": [self._event(1, "0", "0", "0", "Submitted")],
        }
        if self.lose_ack:
            raise RuntimeError("fixture runtime accepted, acknowledgement lost")
        return copy.deepcopy(self.ticket)

    def lookup(self, client, ticket):
        if self.ticket is None:
            return None
        result = copy.deepcopy(self.ticket)
        result["observed_at"] = self.at.isoformat()
        return result

    def cancel(self, ticket, client):
        self.calls.append(("cancel", client))
        self.ticket["status"] = "Canceled"
        self.ticket["events"].append(
            self._event(len(self.ticket["events"]) + 1, "0", "0", "0", "Canceled")
        )

    def _event(self, identity, quantity, price, fee, status):
        return {
            "event_id": identity,
            "ticket_id": 1,
            "instrument_id": self.instrument.instrument_id,
            "occurred_at": self.at.isoformat(),
            "status": status,
            "fill_quantity": quantity,
            "fill_price": price,
            "price_currency": "USD",
            "fee_amount": fee,
            "fee_currency": "USD",
            "fee_type": "QuantConnect.Orders.Fees.OrderFee",
            "is_assignment": False,
            "message": "fixture",
        }

    def add(self, quantity, price, fee):
        signed = D(quantity) if D(self.ticket["quantity"]) > 0 else -D(quantity)
        self.cash -= signed * D(price) + D(fee)
        self.held += signed
        self.price = D(price)
        filled = D(self.ticket["filled_quantity"]) + signed
        status = "Filled" if filled == D(self.ticket["quantity"]) else "PartiallyFilled"
        self.ticket["filled_quantity"], self.ticket["status"] = str(filled), status
        self.ticket["events"].append(
            self._event(len(self.ticket["events"]) + 1, str(signed), price, fee, status)
        )


def setup(path, *, instrument=A, runtime=None, mode="paper"):
    deployment = LeanDeployment(
        deployment_id="fixture-deployment",
        account_id=mode + ":lean",
        broker_account_id="local-paper-fixture" if mode == "paper" else "remote-fixture",
        brokerage="PaperBrokerage" if mode == "paper" else "AlpacaBrokerage",
        mode=mode,
        instruments=(
            LeanInstrument(
                instrument=instrument,
                ticker=instrument.symbol,
                market="usa" if instrument.asset_class == "equity" else "coinbase",
                base_currency="BTC" if instrument.asset_class == "crypto" else None,
            ),
        ),
    )
    runtime = runtime or RuntimeFixture(deployment)
    account = AccountSpec(
        account_id=mode + ":lean",
        mode=mode,
        base_currency="USD",
        initial_cash=D(100000),
        opened_at=T,
    )
    rules = {
        instrument.instrument_id: VenueRules(
            venue="lean",
            instrument=instrument,
            symbol=instrument.symbol,
            available_at=T,
            quantity_step=D(1),
            price_tick=D(".01"),
            active=True,
            market_type="spot" if instrument.asset_class == "crypto" else "equity",
        )
    }
    execution = create_typed_lean_execution(
        ledger_path=path / "lean.sqlite",
        account=account,
        deployment=deployment,
        rules=rules,
        runtime=runtime,
        fill_policy=FillPolicy(equity_cost_bps=0, crypto_cost_bps=0),
        allow_live=mode == "live",
    )
    order = intent(instrument=instrument).model_copy(update={"account_id": account.account_id})
    return runtime, execution, execution.adapters["lean"], order


def admit(runtime, execution, adapter, order):
    adapter.capture()
    execution.reconcile_balances(as_of=runtime.at)
    execution.prepare(
        order, {order.instrument.instrument_id: metadata(T, order.instrument)}, as_of=T
    )
    runtime.at = order.eligible_at
    return execution.dispatch(order.intent_id, as_of=runtime.at)


@pytest.mark.parametrize("instrument", [A, B])
def test_native_partial_receipts_restart_duplicate_and_losing_nav(tmp_path, instrument):
    runtime, execution, adapter, order = setup(tmp_path, instrument=instrument)
    assert admit(runtime, execution, adapter, order)["state"] == "new"
    runtime.at += timedelta(seconds=1)
    runtime.add("3", "100", ".3")
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "partial"
    first = execution.ledger.fills(order.account_id)[0]
    execution.ledger.close()
    runtime, execution, adapter, order = setup(tmp_path, instrument=instrument, runtime=runtime)
    runtime.at += timedelta(seconds=1)
    execution.reconcile_all(as_of=runtime.at)
    assert execution.ledger.fills(order.account_id) == [first]
    runtime.at += timedelta(seconds=1)
    runtime.add("7", "90", ".63")
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "filled"
    adapter.capture()
    assert execution.reconcile_balances(as_of=runtime.at)["reconciled"]
    execution.ledger.mark(
        order.account_id, [PriceMark(instrument=instrument, price=D(90), as_of=runtime.at)]
    )
    final = execution.ledger.snapshot(order.account_id, runtime.at, 0)
    assert final.cash == D("99069.07") and final.nav == D("99969.07") and final.fees == D(".93")
    assert execution.ledger.reconcile(order.account_id)["reconciled"]
    assert len([row for row in runtime.calls if row[0] == "submit"]) == 1


def test_durable_attempt_recovers_lost_ack_without_new_ticket(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    runtime.lose_ack = True
    assert admit(runtime, execution, adapter, order)["state"] == "ambiguous"
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "ambiguous"
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "new"
    runtime.at += timedelta(seconds=1)
    runtime.add("3", "100", ".3")
    assert execution.cancel(order.intent_id, as_of=runtime.at)["state"] == "cancelled"
    assert execution.ledger.quantity(order.account_id, A.instrument_id) == 3
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 0
    assert len([row for row in runtime.calls if row[0] == "submit"]) == 1


def test_new_runtime_cannot_invent_restart_fill_history_or_resubmit(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    admit(runtime, execution, adapter, order)
    runtime.runtime_id = "different-real-process-fixture"
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "ambiguous"
    assert execution.dispatch(order.intent_id, as_of=runtime.at)["state"] == "ambiguous"
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    assert len([row for row in runtime.calls if row[0] == "submit"]) == 1


def test_missing_native_event_prefix_does_not_become_performance(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    admit(runtime, execution, adapter, order)
    runtime.at += timedelta(seconds=1)
    runtime.add("10", "100", "1")
    runtime.ticket["events"].pop(0)
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "reconciliation_required"
    assert not execution.ledger.fills(order.account_id)
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000


def test_native_event_id_change_fails_and_original_economics_remain(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    admit(runtime, execution, adapter, order)
    runtime.at += timedelta(seconds=1)
    runtime.add("3", "100", ".3")
    execution.reconcile_all(as_of=runtime.at)
    runtime.ticket["events"][1]["fee_amount"] = ".4"
    with pytest.raises(LedgerError, match="event identity changed"):
        adapter.lookup(order, execution.state(order.intent_id)["client_id"], None, runtime.at)
    assert execution.ledger.reconcile(order.account_id)["fees"] == "0.3"


def test_operating_bridge_drains_actual_admitted_outbox_and_rechecks_current_risk(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    adapter.capture()
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    bridge = LeanOperatingBridge(execution)
    bridge.pump()
    assert not runtime.calls
    runtime.at = order.eligible_at
    bridge.pump()
    assert execution.state(order.intent_id)["state"] == "new"
    runtime.at += timedelta(seconds=1)
    runtime.add("10", "100", "1")
    bridge.pump()
    assert execution.state(order.intent_id)["state"] == "filled"
    assert execution.ledger.reconcile(order.account_id)["fees"] == "1"


def test_current_price_gap_blocks_prepared_order_before_runtime_submission(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    adapter.capture()
    execution.prepare(order, {A.instrument_id: metadata()}, as_of=T)
    runtime.at, runtime.price = order.eligible_at, D(2000)
    with pytest.raises(LedgerError, match="current projected risk rejected"):
        LeanOperatingBridge(execution).pump()
    assert not runtime.calls
    assert execution.state(order.intent_id)["state"] == "cancelled"


def test_project_is_real_outbox_entry_and_does_not_claim_runtime_acceptance(tmp_path):
    _, execution, adapter, _ = setup(tmp_path)
    config = LeanLaunchConfig(
        ledger_path=str(tmp_path / "lean.sqlite"),
        account=execution.ledger.spec(execution.account_id),
        deployment=adapter.deployment,
        rules=adapter._rules,
    )
    project = tmp_path / "project"
    report = prepare_project(config, project)
    assert report["status"] == "prepared" and report["runtime_verified"] is False
    compile((project / "main.py").read_text(), "main.py", "exec")
    assert "src.execution.lean_algorithm" in (project / "main.py").read_text()
    assert (
        LeanLaunchConfig.model_validate_json((project / "qal-execution.json").read_text()) == config
    )
    assert verify_project(project / "qal-execution.json") == config
    proof_path = project / "execution-manifest.json"
    proof = json.loads(proof_path.read_text())
    proof["implementation_hashes"]["src/execution/lean_venue.py"] = "0" * 64
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="implementation changed"):
        verify_project(project / "qal-execution.json")
    with pytest.raises(FileExistsError):
        prepare_project(config, project)


def test_missing_runtime_cli_is_explicit_failure():
    import importlib.util

    if importlib.util.find_spec("QuantConnect") is not None:
        pytest.skip("environment supplies actual LEAN bindings")
    process = subprocess.run(
        [sys.executable, "-m", "src.execution.lean_bridge", "check-runtime"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result["status"] == "failed" and result["runtime_verified"] is False


def test_lost_ack_cannot_be_adopted_by_new_runtime_before_ticket_binding(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    runtime.lose_ack = True
    assert admit(runtime, execution, adapter, order)["state"] == "ambiguous"
    assert not execution.ledger.db.execute("SELECT 1 FROM lean_ticket_bindings").fetchone()
    runtime.runtime_id = "other-process"
    runtime.ticket["runtime_id"] = runtime.runtime_id
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "ambiguous"
    assert execution.ledger.order(order.account_id, order.intent_id)["reserved"] == 1000
    assert not execution.ledger.fills(order.account_id)


def test_unresolved_execution_blocks_newer_marks_and_dispatch(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    admit(runtime, execution, adapter, order)
    runtime.ticket = None
    previous = execution.ledger._account(order.account_id)["last_at"]
    runtime.at += timedelta(seconds=3)
    with pytest.raises(LedgerError, match="unresolved execution history"):
        LeanOperatingBridge(execution).pump()
    assert execution.ledger._account(order.account_id)["last_at"] == previous


def test_runtime_stays_blocked_until_actual_late_history_then_publishes_ready(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    admit(runtime, execution, adapter, order)
    ticket, runtime.ticket = runtime.ticket, None
    runtime.at += timedelta(seconds=2)
    bridge = LeanOperatingBridge(execution)
    assert bridge.step()["status"] == "blocked"
    assert adapter.runtime_status(runtime.at)["status"] == "blocked"
    runtime.ticket = ticket
    runtime.add("10", "100", "1")
    assert bridge.step()["status"] == "ready"
    runtime.at += timedelta(seconds=31)
    with pytest.raises(LedgerError, match="status receipt is stale"):
        adapter.runtime_status(runtime.at)
    assert bridge.step()["status"] == "ready"
    assert adapter.runtime_status(runtime.at)["observed_at"] == runtime.at.isoformat()
    assert len(execution.ledger.fills(order.account_id)) == 1


def test_original_ticket_receipt_tampering_is_not_absent_history(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path)
    admit(runtime, execution, adapter, order)
    key, payload = execution.ledger.db.execute(
        "SELECT id,payload FROM events WHERE id LIKE 'lean-ticket:%' LIMIT 1"
    ).fetchone()
    changed = json.loads(payload)
    changed["quantity"] = "20"
    execution.ledger.db.execute(
        "UPDATE events SET payload=? WHERE id=?", (json.dumps(changed), key)
    )
    adapter.runtime = None
    with pytest.raises(LedgerError, match="receipt identity changed"):
        adapter.lookup(order, execution.state(order.intent_id)["client_id"], None, runtime.at)


def test_modified_crypto_fill_quantity_reconstructs_actual_gross_and_fee_once(tmp_path):
    runtime, execution, adapter, order = setup(tmp_path, instrument=B)
    admit(runtime, execution, adapter, order)
    runtime.at += timedelta(seconds=1)
    runtime.add("10", "100", "0")
    runtime.held = D("9.99")
    runtime.ticket["filled_quantity"] = "9.99"
    runtime.ticket["events"][-1].update(
        fill_quantity="9.99",
        fee_amount=".01",
        fee_currency="BTC",
        fee_type="QuantConnect.Orders.Fees.ModifiedFillQuantityOrderFee",
    )
    assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "filled"
    fill = execution.ledger.fills(order.account_id)[0]
    assert fill.quantity == 10 and fill.fee == 0 and fill.inventory_fees[0].quantity == D(".01")
    adapter.capture()
    assert execution.reconcile_balances(as_of=runtime.at)["cash"] == "99000"
    execution.ledger.mark(
        order.account_id, [PriceMark(instrument=B, price=D(100), as_of=runtime.at)]
    )
    snapshot = execution.ledger.snapshot(order.account_id, runtime.at, 0)
    assert snapshot.nav == 99999 and snapshot.fees == 1
    assert execution.ledger.reconcile(order.account_id)["reconciled"]
    original = runtime.capture()
    original["cash_book"]["BTC"] = "10"
    with pytest.raises(LedgerError, match="cash-book inventory differs"):
        adapter._balances(original, "fixture")


def test_paper_restart_restores_exact_cash_crypto_inventory_without_new_fills(tmp_path):
    from types import SimpleNamespace

    from src.execution.lean_runtime import LeanAlgorithmPort

    runtime, execution, adapter, order = setup(tmp_path, instrument=B)
    admit(runtime, execution, adapter, order)
    runtime.at += timedelta(seconds=1)
    runtime.add("10", "100", "1")
    execution.reconcile_all(as_of=runtime.at)
    before = execution.ledger.fills(order.account_id)
    applied = []
    algorithm = SimpleNamespace(
        SetCash=lambda *args: applied.append(("cash", args)),
        Securities={
            "symbol": SimpleNamespace(
                Holdings=SimpleNamespace(
                    SetHoldings=lambda *args: applied.append(("holdings", args))
                )
            )
        },
        Portfolio=SimpleNamespace(
            CashBook={
                "BTC": SimpleNamespace(SetAmount=lambda amount: applied.append(("base", amount)))
            }
        ),
    )
    port = object.__new__(LeanAlgorithmPort)
    port.deployment, port.algorithm, port.symbols = (
        adapter.deployment,
        algorithm,
        {B.instrument_id: "symbol"},
    )
    port._native_decimal, port.now, port.runtime_id = D, runtime.now, "next-process"
    restored = port.restore_paper_state(execution.ledger)
    assert restored["cash"] == "98999"
    assert applied == [
        ("cash", ("USD", D(98999), D(1))),
        ("holdings", (D("100.1"), D(10))),
        ("base", D(10)),
    ]
    assert execution.ledger.fills(order.account_id) == before


def test_native_decimal_refuses_loss_and_cancel_keeps_original_tag():
    from types import SimpleNamespace

    from src.execution.lean_runtime import LeanAlgorithmPort

    calls = []
    port = object.__new__(LeanAlgorithmPort)
    port._culture = None
    port._decimal = SimpleNamespace(Parse=lambda text, culture: D(text).quantize(D(".01")))
    with pytest.raises(LedgerError, match="not exactly representable"):
        port._native_decimal(D(".123"))
    port.algorithm = SimpleNamespace(
        Transactions=SimpleNamespace(
            GetOrderTicket=lambda identity: SimpleNamespace(
                Cancel=lambda *args: calls.append(args)
            ),
            GetOrderById=lambda identity: SimpleNamespace(Tag="original-client"),
        )
    )
    port.cancel(1, "original-client")
    assert calls == [()]


def test_live_native_execution_proof_overrides_lean_fee_model_and_binds_postings(tmp_path):
    import httpx

    from src.execution.alpaca_venue import AlpacaVenue
    from tests.test_alpaca_venue import Wire

    class LiveWire(Wire):
        def handle(self, request):
            assert request.url.host == "api.alpaca.markets"
            # Reuse the offline response bodies; no socket or authenticated call.
            request.url = request.url.copy_with(host="paper-api.alpaca.markets")
            return super().handle(request)

    runtime, execution, adapter, order = setup(tmp_path, mode="live")
    wire = LiveWire()
    native = AlpacaVenue(
        execution.ledger,
        order.account_id,
        broker_account_id="remote-fixture",
        api_key="fixture-key-not-a-credential",
        api_secret="fixture-secret-not-a-credential",
        rules={
            A.instrument_id: adapter._rules[A.instrument_id].model_copy(update={"venue": "alpaca"})
        },
        allow_live=True,
        client=httpx.AsyncClient(transport=httpx.MockTransport(wire.transport)),
        clock=lambda: runtime.at,
    )
    adapter.native_reader = native
    original_submit = runtime.submit

    def submit(order, client):
        original_submit(order, client)
        runtime.ticket["broker_ids"] = ["remote-order-1"]
        wire.order = {
            "id": "remote-order-1",
            "client_order_id": "actual-native-client",
            "symbol": "A",
            "qty": "10",
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "extended_hours": False,
            "order_class": "simple",
            "filled_qty": "0",
            "status": "new",
            "limit_price": None,
            "stop_price": None,
        }
        return copy.deepcopy(runtime.ticket)

    runtime.submit = submit
    try:
        assert admit(runtime, execution, adapter, order)["state"] == "new"
        runtime.at += timedelta(seconds=1)
        runtime.add("10", "100", "7")  # Runtime FeeModel estimate, not a posted charge.
        wire.add_fill("actual-execution", "10", "100", 2)
        wire.order.update(filled_qty="10", status="filled")
        wire.cash, wire.positions = "99000", [{"symbol": "A", "side": "long", "qty": "10"}]
        assert execution.reconcile_all(as_of=runtime.at)[0]["state"] == "filled"
        fill = execution.ledger.fills(order.account_id)[0]
        assert fill.fill_id == "alpaca:actual-execution" and fill.fee == 0
        assert execution.ledger.reconcile(order.account_id)["fees"] == "0"
        with pytest.raises(LedgerError, match="native account cash/holdings/cutoff differs"):
            adapter.capture()
        # Simulate the runtime's subsequent actual account synchronization.
        runtime.cash = D(wire.cash)
        adapter.capture()
        assert execution.reconcile_balances(as_of=runtime.at)["reconciled"]
        runtime.at += timedelta(seconds=1)
        wire.fees = [
            {
                "activity_type": "FEE",
                "id": "posted-1",
                "net_amount": "-.03",
                "order_id": "remote-order-1",
                "date": runtime.at.date().isoformat(),
            }
        ]
        runtime.cash, wire.cash = D("98999.97"), "98999.97"
        LeanOperatingBridge(execution).pump()
        fees = native.posted_fees()
        assert len(fees) == 1 and fees[0].intent_id == order.intent_id
        assert execution.ledger.reconcile(order.account_id)["fees"] == "0.03"
        assert not any(method != "GET" for method, *_ in wire.calls)
        # Caller-supplied balance summaries cannot replace native raw evidence.
        proof, key = adapter._latest("balances", runtime.at)
        proof["native_balances"]["cash_base"] = "100000"
        with pytest.raises(LedgerError, match="differ from original account proof"):
            adapter._balances(proof, key)
    finally:
        native.close()


def test_live_local_packet_identity_rejects_custom_jobs_alt_auth_and_networks():
    from types import SimpleNamespace

    from src.execution.lean_runtime import LeanAlgorithmPort

    values = {
        "job-queue-handler": "QuantConnect.Queues.JobQueue",
        "alpaca-api-key": "fixture-key",
        "alpaca-api-secret": "fixture-secret",
        "alpaca-paper-trading": "false",
    }
    config = SimpleNamespace(Get=lambda key, default="": values.get(key, default))
    identity = LeanAlgorithmPort._local_broker_identity(config, "AlpacaBrokerage")
    assert len(identity) == 64
    values["alpaca-access-token"] = "fixture-alternate-authority"
    with pytest.raises(LedgerError, match="key authority/endpoint differs"):
        LeanAlgorithmPort._local_broker_identity(config, "AlpacaBrokerage")
    values.pop("alpaca-access-token")
    values["job-queue-handler"] = "another-job-source"
    with pytest.raises(LedgerError, match="reviewed local JobQueue"):
        LeanAlgorithmPort._local_broker_identity(config, "AlpacaBrokerage")
    values.update(
        {
            "job-queue-handler": "QuantConnect.Queues.JobQueue",
            "binance-api-key": "fixture-key",
            "binance-api-secret": "fixture-secret",
            "binance-api-url": "https://testnet.binance.vision",
        }
    )
    with pytest.raises(LedgerError, match="native network differs"):
        LeanAlgorithmPort._local_broker_identity(config, "BinanceBrokerage")
