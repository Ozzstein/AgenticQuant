"""Tests for utils: config, logger, schemas, exceptions."""

from __future__ import annotations

import pytest

from src.utils.config import AppConfig, get_config, reset_config
from src.utils.exceptions import (
    AgentError,
    AiQuantError,
    BacktestError,
    ConfigError,
    DataPipelineError,
    ExecutionError,
    FactorError,
    LLMError,
    ModelError,
    OrderError,
    RiskBreachError,
    ToolError,
)
from src.utils.logger import get_logger, reset_logger, setup_logger
from src.utils.schemas import (
    AgentReport,
    AnalysisResult,
    AssetClass,
    BacktestMetrics,
    BacktestResult,
    BenchmarkComparison,
    Decision,
    FactorDefinition,
    ModelSpec,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    Signal,
    SignalDirection,
    ValidationCheck,
    ValidationResult,
    ValidationVerdict,
    WalkForwardFold,
)


# --- Config Tests ---


class TestConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.project_name == "QuantAgentLab"
        assert config.log_level == "INFO"
        assert config.qlib.region == "us"
        assert config.agents.llm_provider == "anthropic"
        assert config.crypto.exchange == "binance"
        assert config.risk.max_position_weight == 10.0

    def test_config_override(self):
        config = AppConfig(log_level="DEBUG", anthropic_api_key="test-key")
        assert config.log_level == "DEBUG"
        assert config.anthropic_api_key == "test-key"

    def test_log_level_validation(self):
        with pytest.raises(ValueError, match="log_level"):
            AppConfig(log_level="INVALID")

    def test_log_level_case_insensitive(self):
        config = AppConfig(log_level="debug")
        assert config.log_level == "DEBUG"

    def test_nested_config(self):
        config = AppConfig()
        assert config.qlib.dataset == "Alpha158"
        assert config.agents.max_debate_rounds == 2
        assert config.models.topk == 30
        assert config.monitoring.schedule_time == "16:00"

    def test_config_singleton(self):
        reset_config()
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_config_reset(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2


# --- Logger Tests ---


class TestLogger:
    def test_setup_logger(self):
        reset_logger()
        setup_logger("DEBUG")
        log = get_logger("test")
        assert log is not None

    def test_setup_with_file(self, tmp_path):
        reset_logger()
        setup_logger("INFO", str(tmp_path / "logs"))
        log = get_logger("test")
        assert log is not None

    def test_logger_idempotent(self):
        reset_logger()
        setup_logger("INFO")
        setup_logger("INFO")  # should not add duplicate handlers


# --- Exception Tests ---


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(DataPipelineError, AiQuantError)
        assert issubclass(FactorError, AiQuantError)
        assert issubclass(ModelError, AiQuantError)
        assert issubclass(BacktestError, AiQuantError)
        assert issubclass(AgentError, AiQuantError)
        assert issubclass(LLMError, AgentError)
        assert issubclass(ToolError, AgentError)
        assert issubclass(ExecutionError, AiQuantError)
        assert issubclass(RiskBreachError, ExecutionError)
        assert issubclass(OrderError, ExecutionError)
        assert issubclass(ConfigError, AiQuantError)

    def test_raise_and_catch(self):
        with pytest.raises(AiQuantError):
            raise DataPipelineError("test error")

        with pytest.raises(AgentError):
            raise LLMError("LLM failed")

        with pytest.raises(ExecutionError):
            raise RiskBreachError("Position too large")


# --- Schema Tests ---


class TestDecision:
    def test_enum_values(self):
        assert Decision.STRONG_BUY.value == "STRONG_BUY"
        assert Decision.HOLD.value == "HOLD"

    def test_enum_from_string(self):
        assert Decision("BUY") == Decision.BUY


class TestAgentReport:
    def test_valid_report(self):
        report = AgentReport(
            agent_name="fundamental",
            decision=Decision.BUY,
            confidence=85.0,
            reasoning="Strong growth metrics.",
        )
        assert report.confidence == 85.0
        assert report.data_points == {}

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            AgentReport(
                agent_name="test",
                decision=Decision.BUY,
                confidence=150.0,
                reasoning="test",
            )

        with pytest.raises(ValueError):
            AgentReport(
                agent_name="test",
                decision=Decision.BUY,
                confidence=-10.0,
                reasoning="test",
            )


class TestAnalysisResult:
    def test_full_result(self, sample_analysis_result):
        r = sample_analysis_result
        assert r.ticker == "AAPL"
        assert r.decision == Decision.BUY
        assert r.confidence == 75.0
        assert len(r.agent_reports) == 2
        assert "fundamental" in r.agent_reports

    def test_serialization(self, sample_analysis_result):
        json_str = sample_analysis_result.model_dump_json()
        assert "AAPL" in json_str
        restored = AnalysisResult.model_validate_json(json_str)
        assert restored.ticker == "AAPL"


class TestSignal:
    def test_signal_creation(self):
        s = Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=0.8, source="model")
        assert s.strength == 0.8

    def test_strength_bounds(self):
        with pytest.raises(ValueError):
            Signal(ticker="AAPL", direction=SignalDirection.LONG, strength=1.5)


class TestOrder:
    def test_market_order(self):
        o = Order(ticker="AAPL", side=OrderSide.BUY, quantity=100)
        assert o.order_type == OrderType.MARKET
        assert o.status == OrderStatus.PENDING

    def test_limit_order(self):
        o = Order(
            ticker="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=400.0,
        )
        assert o.limit_price == 400.0


class TestPortfolio:
    def test_portfolio_defaults(self):
        p = Portfolio()
        assert p.cash == 100_000.0
        assert p.nav == 100_000.0
        assert len(p.positions) == 0

    def test_portfolio_with_positions(self, sample_portfolio):
        assert len(sample_portfolio.positions) == 2
        assert "AAPL" in sample_portfolio.positions


class TestBacktest:
    def test_metrics_defaults(self):
        m = BacktestMetrics()
        assert m.sharpe_ratio == 0.0
        assert m.total_trades == 0

    def test_backtest_result(self):
        r = BacktestResult(
            metrics=BacktestMetrics(sharpe_ratio=1.5, total_return=0.25),
            daily_returns=[0.01, -0.005, 0.003],
            walk_forward_results=[
                WalkForwardFold(
                    fold_id=1,
                    train_start="2020-01-01",
                    train_end="2021-12-31",
                    test_start="2022-01-01",
                    test_end="2022-03-31",
                    in_sample_sharpe=2.0,
                    oos_sharpe=1.2,
                )
            ],
        )
        assert r.metrics.sharpe_ratio == 1.5
        assert len(r.walk_forward_results) == 1

    def test_validation_result(self):
        v = ValidationResult(
            checks=[
                ValidationCheck(name="look_ahead", passed=True, details="OK"),
                ValidationCheck(name="cost_sensitivity", passed=False, details="Sharpe drops >50%"),
            ],
            verdict=ValidationVerdict.CAUTION,
            summary="Minor concerns.",
        )
        assert v.passed_count == 1
        assert v.failed_count == 1


class TestBenchmark:
    def test_comparison(self):
        b = BenchmarkComparison(
            strategy_sharpe=1.8,
            spy_sharpe=0.9,
            monte_carlo_percentile=85.0,
            excess_sharpe_vs_spy=0.9,
        )
        assert b.excess_sharpe_vs_spy == 0.9


class TestFactorDefinition:
    def test_factor(self):
        f = FactorDefinition(
            name="RSI_14",
            expression="RSI($close, 14)",
            category="momentum",
            ic_mean=0.03,
            icir=0.5,
        )
        assert f.name == "RSI_14"


class TestModelSpec:
    def test_model_spec(self):
        m = ModelSpec(
            name="LightGBM",
            params={"n_estimators": 500, "learning_rate": 0.05},
            feature_set=["RSI_14", "MACD"],
        )
        assert m.name == "LightGBM"
        assert len(m.feature_set) == 2


def test_model_config_ensemble_defaults():
    """ModelConfig has ensemble defaults: 3 models, ic_weighted, 0.15 val fraction."""
    from src.utils.config import ModelConfig
    cfg = ModelConfig()
    assert cfg.ensemble_models == ["LightGBM", "XGBoost", "Linear"]
    assert cfg.ensemble_method == "ic_weighted"
    assert cfg.ensemble_val_fraction == 0.15


def test_model_config_ensemble_custom():
    """ModelConfig accepts custom ensemble settings."""
    from src.utils.config import ModelConfig
    cfg = ModelConfig(
        ensemble_models=["LightGBM", "Linear"],
        ensemble_method="equal",
        ensemble_val_fraction=0.2,
    )
    assert len(cfg.ensemble_models) == 2
    assert cfg.ensemble_method == "equal"
    assert cfg.ensemble_val_fraction == 0.2
