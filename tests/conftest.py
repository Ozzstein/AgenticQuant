"""Shared test fixtures for QuantAgentLab."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.utils.config import AppConfig, reset_config
from src.utils.schemas import (
    AgentReport,
    AnalysisResult,
    AssetClass,
    Decision,
    Portfolio,
    Position,
)


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Reset config singleton between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _cleanup_generated_artifacts():
    """Remove generated output artifacts that interfere with unit tests."""
    from pathlib import Path
    _ARTIFACTS = [
        Path("outputs/best_model_config.yaml"),
    ]
    for p in _ARTIFACTS:
        p.unlink(missing_ok=True)
    yield
    for p in _ARTIFACTS:
        p.unlink(missing_ok=True)


@pytest.fixture
def mock_config() -> AppConfig:
    """Return a test AppConfig with defaults."""
    return AppConfig(
        log_level="DEBUG",
        output_dir="/tmp/aiquant_test/outputs",
        log_dir="/tmp/aiquant_test/logs",
        anthropic_api_key="test-key-123",
        finnhub_api_key="test-finnhub-key",
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate synthetic 252-day OHLCV data for 5 stocks."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    frames = []
    for ticker in tickers:
        base_price = np.random.uniform(100, 500)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.03, len(dates)),
                "low": prices * np.random.uniform(0.97, 1.00, len(dates)),
                "close": prices,
                "volume": np.random.randint(1_000_000, 50_000_000, len(dates)),
            },
            index=dates,
        )
        df["ticker"] = ticker
        frames.append(df)

    result = pd.concat(frames)
    result.index.name = "date"
    return result


@pytest.fixture
def mock_llm():
    """Return a mock LLM that returns structured AgentReport."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="Test analysis complete.",
    )
    mock.with_structured_output.return_value = mock
    return mock


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Return a sample portfolio with positions."""
    return Portfolio(
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=100,
                avg_cost=150.0,
                current_price=175.0,
                unrealized_pnl=2500.0,
                weight_pct=35.0,
                asset_class=AssetClass.EQUITY,
            ),
            "MSFT": Position(
                ticker="MSFT",
                quantity=50,
                avg_cost=300.0,
                current_price=350.0,
                unrealized_pnl=2500.0,
                weight_pct=35.0,
                asset_class=AssetClass.EQUITY,
            ),
        },
        cash=15_000.0,
        nav=50_000.0,
        timestamp=datetime(2024, 1, 15, 16, 0),
    )


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    """Return a sample AnalysisResult."""
    return AnalysisResult(
        ticker="AAPL",
        asset_class=AssetClass.EQUITY,
        decision=Decision.BUY,
        confidence=75.0,
        target_price=200.0,
        stop_loss=160.0,
        position_size_pct=5.0,
        time_horizon="3M",
        reasoning="Strong fundamentals with positive momentum.",
        agent_reports={
            "fundamental": AgentReport(
                agent_name="fundamental",
                decision=Decision.BUY,
                confidence=80.0,
                reasoning="Strong revenue growth.",
                data_points={"pe_ratio": 28.5, "revenue_growth": 0.12},
            ),
            "technical": AgentReport(
                agent_name="technical",
                decision=Decision.BUY,
                confidence=70.0,
                reasoning="Bullish trend confirmed.",
                data_points={"rsi": 55.0, "above_sma200": True},
            ),
        },
        risk_flags=["High valuation"],
        catalysts=["Upcoming product launch"],
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Return a temporary output directory."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out
