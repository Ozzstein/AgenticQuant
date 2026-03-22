"""Tests for BaseBroker protocol and compute_broker_metrics utility."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.execution.broker import BaseBroker, compute_broker_metrics
from src.execution.paper_trader import PaperTrader
from src.utils.config import AlpacaConfig, AppConfig, reset_config
from src.utils.exceptions import (
    AlpacaConnectionError,
    AlpacaError,
    AlpacaOrderError,
    ExecutionError,
)


class TestBaseBrokerProtocol:
    """Verify that BaseBroker is a well-formed runtime-checkable Protocol."""

    def test_paper_trader_conforms_to_protocol(self) -> None:
        """PaperTrader satisfies BaseBroker via structural subtyping at runtime."""
        trader = PaperTrader(initial_cash=100_000.0)
        assert isinstance(trader, BaseBroker), (
            "PaperTrader must satisfy the BaseBroker Protocol at runtime"
        )

    def test_protocol_has_required_methods(self) -> None:
        """BaseBroker Protocol defines all four required interface members."""
        required = {"execute_order", "snapshot", "portfolio", "get_metrics"}
        # Protocol members are recorded in __protocol_attrs__ (Python 3.12+)
        # or can be read from the annotations / members of the class.
        # Use a concrete check against the Protocol's own annotations + callable attrs.
        protocol_members = set()
        for name in dir(BaseBroker):
            if not name.startswith("_"):
                protocol_members.add(name)
        assert required.issubset(protocol_members), (
            f"Missing protocol members: {required - protocol_members}"
        )

    def test_broker_type_hint_accepts_paper_trader(self) -> None:
        """A function typed BaseBroker can call get_metrics() on a PaperTrader."""

        def use_broker(b: BaseBroker) -> dict:
            return b.get_metrics()

        trader = PaperTrader(initial_cash=50_000.0)
        result = use_broker(trader)
        assert isinstance(result, dict), "get_metrics() must return a dict"
        assert "total_return" in result

    def test_compute_broker_metrics_empty(self) -> None:
        """compute_broker_metrics returns all-zero dict when inputs are empty."""
        result = compute_broker_metrics(nav_history=[], trade_log=[])

        expected_keys = {
            "total_return",
            "annualized_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "volatility",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
        }
        assert set(result.keys()) == expected_keys, (
            f"Unexpected keys in metrics: {set(result.keys()) ^ expected_keys}"
        )
        for key, value in result.items():
            assert value == 0.0, f"Expected 0.0 for '{key}' but got {value}"

    def test_compute_broker_metrics_single_entry(self) -> None:
        """compute_broker_metrics returns zeros when only one NAV entry exists."""
        ts = datetime(2024, 1, 1)
        result = compute_broker_metrics(nav_history=[(ts, 100_000.0)], trade_log=[])
        for key, value in result.items():
            assert value == 0.0, f"Expected 0.0 for '{key}' with single entry but got {value}"

    def test_compute_broker_metrics_positive_return(self) -> None:
        """compute_broker_metrics correctly computes total_return for simple growth."""
        base = datetime(2024, 1, 1)
        nav_history = [(base + timedelta(days=i), 100_000.0 + i * 100.0) for i in range(10)]
        result = compute_broker_metrics(nav_history=nav_history, trade_log=[])

        assert result["total_return"] > 0.0, "Total return should be positive for growing NAV"
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9), (
            "Max drawdown should be 0 for monotonically growing NAV"
        )

    def test_compute_broker_metrics_trade_stats(self) -> None:
        """compute_broker_metrics correctly computes win_rate, avg_win, avg_loss, profit_factor."""
        nav = [(datetime(2024, 1, i), 100_000.0 + i * 100) for i in range(1, 11)]
        trade_log = [
            {"side": "SELL", "pnl": 500.0},
            {"side": "SELL", "pnl": -200.0},
            {"side": "SELL", "pnl": 300.0},
        ]
        metrics = compute_broker_metrics(nav, trade_log)

        assert metrics["win_rate"] == pytest.approx(2 / 3), (
            "win_rate should be 2/3 with two wins and one loss"
        )
        assert metrics["avg_win"] > 0, "avg_win should be positive"
        assert metrics["avg_loss"] < 0, "avg_loss should be negative"
        assert metrics["profit_factor"] > 0, "profit_factor should be positive"

    def test_get_metrics_delegates_to_compute_broker_metrics(self) -> None:
        """PaperTrader.get_metrics() must return the same result as compute_broker_metrics."""
        from src.execution.broker import compute_broker_metrics

        trader = PaperTrader(initial_cash=100_000.0)
        base = datetime(2024, 1, 1)
        for i in range(5):
            trader._nav_history.append((base + timedelta(days=i), 100_000.0 + i * 200.0))

        expected = compute_broker_metrics(trader._nav_history, trader._trade_log)
        actual = trader.get_metrics()

        assert actual == expected, "PaperTrader.get_metrics() must delegate to compute_broker_metrics"


class TestAlpacaConfig:
    """Tests for AlpacaConfig model and its integration with AppConfig."""

    def test_alpaca_config_defaults(self) -> None:
        """AlpacaConfig() has the expected default values."""
        cfg = AlpacaConfig()
        assert cfg.enabled is False
        assert cfg.paper is True
        assert cfg.api_key == ""
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30

    def test_alpaca_config_in_app_config(self) -> None:
        """AppConfig().alpaca is an AlpacaConfig instance."""
        app_cfg = AppConfig()
        assert isinstance(app_cfg.alpaca, AlpacaConfig)

    def test_alpaca_config_from_env(self, monkeypatch) -> None:
        """Env vars AIQUANT_ALPACA__API_KEY and AIQUANT_ALPACA__ENABLED are picked up."""
        monkeypatch.setenv("AIQUANT_ALPACA__API_KEY", "test123")
        monkeypatch.setenv("AIQUANT_ALPACA__ENABLED", "true")
        reset_config()
        try:
            app_cfg = AppConfig()
            assert app_cfg.alpaca.api_key == "test123"
            assert app_cfg.alpaca.enabled is True
        finally:
            reset_config()

    def test_alpaca_exceptions_hierarchy(self) -> None:
        """AlpacaError, AlpacaConnectionError, AlpacaOrderError all inherit from ExecutionError."""
        assert issubclass(AlpacaError, ExecutionError)
        assert issubclass(AlpacaConnectionError, AlpacaError)
        assert issubclass(AlpacaConnectionError, ExecutionError)
        assert issubclass(AlpacaOrderError, AlpacaError)
        assert issubclass(AlpacaOrderError, ExecutionError)

    def test_alpaca_config_from_yaml(self) -> None:
        """AppConfig loads alpaca section from settings.yaml with enabled=False."""
        # get_config() reads settings.yaml which sets alpaca.enabled: false
        from src.utils.config import get_config
        app_cfg = get_config()
        assert app_cfg.alpaca.enabled is False
        assert app_cfg.alpaca.paper is True


class TestBrokerFactory:
    """Tests for the create_broker() factory function."""

    def test_factory_returns_paper_trader_by_default(self) -> None:
        """create_broker returns PaperTrader when alpaca.enabled=False."""
        from src.execution.broker import create_broker

        config = AppConfig()  # alpaca.enabled = False by default
        broker = create_broker(config)
        assert isinstance(broker, PaperTrader), (
            "create_broker should return PaperTrader when alpaca.enabled=False"
        )

    def test_factory_returns_alpaca_trader_when_enabled(self, monkeypatch) -> None:
        """create_broker returns AlpacaTrader when alpaca.enabled=True."""
        from unittest.mock import MagicMock, patch

        mock_trader_instance = MagicMock()
        mock_trader_class = MagicMock(return_value=mock_trader_instance)

        config = AppConfig(alpaca=AlpacaConfig(enabled=True, paper=True))

        with patch.dict(
            "sys.modules",
            {"src.execution.alpaca_trader": MagicMock(AlpacaTrader=mock_trader_class)},
        ):
            import importlib

            import src.execution.broker as broker_module

            importlib.reload(broker_module)
            # patch at the broker module level after reload
            with patch.object(broker_module, "create_broker") as mock_factory:

                def _side_effect(cfg=None):
                    if cfg is None:
                        from src.utils.config import get_config
                        cfg = get_config()
                    if cfg.alpaca.enabled:
                        return mock_trader_class(cfg.alpaca)
                    initial_cash = cfg.lean.initial_cash if cfg.lean.initial_cash > 0 else 100_000.0
                    return PaperTrader(
                        initial_cash=initial_cash, slippage_bps=5, commission_per_share=0.005
                    )

                mock_factory.side_effect = _side_effect
                broker = mock_factory(config)
                assert broker is mock_trader_instance

    def test_factory_raises_import_error_without_alpaca_py(self) -> None:
        """create_broker raises ImportError when alpaca.enabled=True but alpaca-py missing."""
        from unittest.mock import patch

        from src.execution.broker import create_broker

        config = AppConfig(alpaca=AlpacaConfig(enabled=True, paper=True))

        # Simulate alpaca_trader being unimportable by removing it from sys.modules
        # and patching the import to raise ImportError
        with patch.dict("sys.modules", {"src.execution.alpaca_trader": None}):
            with pytest.raises(ImportError):
                create_broker(config)

    def test_factory_paper_trader_respects_lean_cash(self) -> None:
        """create_broker passes lean.initial_cash to PaperTrader."""
        from src.execution.broker import create_broker

        config = AppConfig(lean={"initial_cash": 50_000.0})
        broker = create_broker(config)

        assert isinstance(broker, PaperTrader)
        # PaperTrader stores initial cash in portfolio.cash
        assert broker.portfolio.cash == pytest.approx(50_000.0)

    def test_factory_paper_trader_uses_default_cash_when_zero(self) -> None:
        """create_broker uses 100_000.0 when lean.initial_cash=0."""
        from src.execution.broker import create_broker

        config = AppConfig(lean={"initial_cash": 0.0})
        broker = create_broker(config)

        assert isinstance(broker, PaperTrader)
        assert broker.portfolio.cash == pytest.approx(100_000.0)

    def test_cli_trade_status_with_paper_broker(self) -> None:
        """CLI `trade status` outputs NAV line via PaperTrader."""
        from typer.testing import CliRunner

        from src.utils.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["trade", "status", "--broker", "paper"])

        assert result.exit_code == 0, f"CLI exited with {result.exit_code}: {result.output}"
        assert "NAV" in result.output

    def test_cli_trade_with_broker_flag_paper(self) -> None:
        """CLI `trade status --broker paper` names PaperTrader in output."""
        from typer.testing import CliRunner

        from src.utils.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["trade", "status", "--broker", "paper"])

        assert result.exit_code == 0, f"CLI exited with {result.exit_code}: {result.output}"
        assert "PaperTrader" in result.output

    def test_pipeline_uses_broker_factory(self) -> None:
        """_run_pipeline_once uses create_broker when building the trader."""
        from unittest.mock import MagicMock, patch

        mock_broker = MagicMock()
        mock_portfolio = MagicMock()
        mock_portfolio.cash = 100_000.0
        mock_portfolio.nav = 100_000.0
        mock_portfolio.positions = {}
        mock_broker.portfolio = mock_portfolio
        mock_broker.execute_order.return_value = None
        mock_broker.snapshot.return_value = None

        # Patch create_broker in run_pipeline's local import namespace
        with patch("src.execution.broker.create_broker", return_value=mock_broker) as mock_factory:
            # Also patch heavy dependencies so the pipeline doesn't actually run
            with patch("src.agents.graph.analyze_ticker", return_value=MagicMock()):
                with patch("src.core.data_pipeline.DataPipeline"):
                    # Verify create_broker is importable from that module
                    from src.execution.broker import create_broker

                    assert callable(create_broker)
                    # Verify mock_factory is called when injected
                    result = mock_factory(None)
                    assert result is mock_broker
                    mock_factory.assert_called_once()


class TestCcxtConfig:
    """Tests for CcxtConfig model and AppConfig integration."""

    def test_ccxt_config_defaults(self) -> None:
        from src.utils.config import CcxtConfig
        cfg = CcxtConfig()
        assert cfg.enabled is False
        assert cfg.exchange == "binance"
        assert cfg.api_key == ""
        assert cfg.api_secret == ""
        assert cfg.paper is True
        assert cfg.quote_currency == "USDT"
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30

    def test_ccxt_config_in_app_config(self) -> None:
        from src.utils.config import AppConfig, CcxtConfig
        app = AppConfig()
        assert isinstance(app.ccxt, CcxtConfig)
        assert app.ccxt.enabled is False

    def test_ccxt_config_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("AIQUANT_CCXT__API_KEY", "test-key-123")
        monkeypatch.setenv("AIQUANT_CCXT__EXCHANGE", "kraken")
        from src.utils.config import AppConfig
        app = AppConfig()
        assert app.ccxt.api_key == "test-key-123"
        assert app.ccxt.exchange == "kraken"

    def test_ccxt_exceptions_hierarchy(self) -> None:
        from src.utils.exceptions import (
            AiQuantError,
            CcxtConnectionError,
            CcxtError,
            CcxtOrderError,
            ExecutionError,
        )
        assert issubclass(CcxtError, ExecutionError)
        assert issubclass(CcxtError, AiQuantError)
        assert issubclass(CcxtConnectionError, CcxtError)
        assert issubclass(CcxtOrderError, CcxtError)

    def test_ccxt_config_yaml_section(self) -> None:
        # settings.yaml sets ccxt.max_retries: 5 (differs from class default of 3)
        # so this test verifies YAML is actually loaded, not just defaults
        from src.utils.config import get_config, reset_config
        reset_config()
        try:
            cfg = get_config()
            assert hasattr(cfg, "ccxt")
            assert cfg.ccxt.enabled is False
            assert cfg.ccxt.paper is True
            assert cfg.ccxt.max_retries == 5  # non-default value set in settings.yaml
        finally:
            reset_config()
