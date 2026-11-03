"""Algorithm lifecycle regression; actual engine acceptance is recorded separately."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.execution.ledger import LedgerError


@pytest.fixture
def algorithm(monkeypatch):
    imports = ModuleType("AlgorithmImports")
    imports.QCAlgorithm = type("QCAlgorithm", (), {})
    for name in ("DataNormalizationMode", "Resolution", "TimeZones"):
        setattr(imports, name, SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "AlgorithmImports", imports)
    spec = importlib.util.spec_from_file_location(
        "paper_initialization_fixture",
        Path(__file__).resolve().parents[1] / "src/execution/lean_algorithm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    value = module.AIQuantStackAlgorithm()
    value._qal_paper_restore_pending = True
    value.IsWarmingUp = False
    value._qal_execution = SimpleNamespace(
        ledger=object(),
        adapters={"lean": SimpleNamespace(_record=lambda *args: calls.append(("record", args)))},
    )
    value._qal_port = SimpleNamespace(
        restore_paper_state=lambda ledger: calls.append(("restore", ledger))
    )
    monkeypatch.setattr(
        module,
        "LeanOperatingBridge",
        lambda execution: (
            calls.append(("bridge", execution))
            or SimpleNamespace(step=lambda: {"status": "ready", "error": None})
        ),
    )
    return value, calls


def test_after_setup_restores_before_enabling_bridge_exactly_once(algorithm):
    value, calls = algorithm
    value.OnWarmupFinished()
    assert [row[0] for row in calls] == ["restore", "record", "bridge"]
    assert value._qal_paper_restore_pending is False
    value.OnWarmupFinished()
    value.OnData(None)
    assert len(calls) == 3


def test_no_dispatch_before_initialization_finishes(algorithm):
    value, calls = algorithm
    with pytest.raises(LedgerError, match="initialization is not complete"):
        value.OnData(None)
    assert calls == [] and not hasattr(value, "_qal_bridge")


def test_failed_restore_never_enables_dispatch(algorithm):
    value, calls = algorithm

    def fail(ledger):
        raise LedgerError("original proof required")

    value._qal_port.restore_paper_state = fail
    with pytest.raises(LedgerError, match="original proof"):
        value.OnWarmupFinished()
    assert value._qal_paper_restore_pending and not hasattr(value, "_qal_bridge")
    assert calls == []


def test_live_callback_never_restores_paper_cash(algorithm):
    value, calls = algorithm
    value._qal_paper_restore_pending = False
    value.OnWarmupFinished()
    assert calls == []


def test_original_pending_order_rejected_before_any_runtime_mutation(tmp_path):
    from src.execution.lean_runtime import LeanAlgorithmPort
    from tests.test_lean_execution import admit, setup

    runtime, execution, adapter, order = setup(tmp_path)
    try:
        admit(runtime, execution, adapter, order)
        before = execution.ledger.order(order.account_id, order.intent_id)
        port = object.__new__(LeanAlgorithmPort)
        port.deployment = adapter.deployment
        # No algorithm or CLR constructor: validation must fail before accessing either.
        with pytest.raises(LedgerError, match="unresolved accepted orders"):
            port.restore_paper_state(execution.ledger)
        assert execution.ledger.order(order.account_id, order.intent_id) == before
    finally:
        execution.ledger.close()


@pytest.mark.parametrize(
    "mode,brokerage", [("live", "PaperBrokerage"), ("paper", "AlpacaBrokerage")]
)
def test_nonlocal_or_live_restore_denied_before_runtime_access(tmp_path, mode, brokerage):
    from src.execution.lean_runtime import LeanAlgorithmPort
    from tests.test_lean_execution import setup

    _, execution, adapter, _ = setup(tmp_path)
    try:
        port = object.__new__(LeanAlgorithmPort)
        port.deployment = SimpleNamespace(
            account_id=execution.account_id, mode=mode, brokerage=brokerage
        )
        with pytest.raises(LedgerError, match="only an actual local paper runtime"):
            port.restore_paper_state(execution.ledger)
    finally:
        execution.ledger.close()


@pytest.mark.parametrize("currency,rate", [("EUR", "1"), ("USD", "0.5")])
def test_wrong_base_currency_or_conversion_rejects_before_cash_write(tmp_path, currency, rate):
    from decimal import Decimal

    from src.execution.lean_runtime import LeanAlgorithmPort
    from tests.test_lean_execution import setup

    _, execution, adapter, _ = setup(tmp_path)
    calls = []

    class CashBook(dict):
        AccountCurrency = currency

    try:
        port = object.__new__(LeanAlgorithmPort)
        port.deployment = adapter.deployment
        port._money = lambda source, name: Decimal(getattr(source, name))
        port.algorithm = SimpleNamespace(
            Portfolio=SimpleNamespace(
                CashBook=CashBook(
                    {"USD": SimpleNamespace(ConversionRate=rate, SetAmount=calls.append)}
                )
            )
        )
        with pytest.raises(LedgerError, match="currency/conversion differs"):
            port.restore_paper_state(execution.ledger)
        assert calls == []
    finally:
        execution.ledger.close()
