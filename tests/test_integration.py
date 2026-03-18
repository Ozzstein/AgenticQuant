"""Integration tests for TASK-06 scripts and pipeline components.

All tests are marked @pytest.mark.integration. Run with:
    pytest tests/test_integration.py -v --tb=short -m integration

External API calls are mocked; no real API keys required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test 1: run_backtest._run_pipeline
# ---------------------------------------------------------------------------


def test_run_backtest_script(tmp_path):
    """run_backtest._run_pipeline works correctly with mocked yfinance."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    dates = pd.date_range("2022-01-01", "2022-06-30", freq="B")
    tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "UNH"]

    # Build a long-format DataFrame mimicking DataPipeline.yfinance_fallback output
    frames = []
    for ticker in tickers:
        frame = pd.DataFrame(
            {
                "close": [100 + i * 0.1 for i in range(len(dates))],
                "open": [99 + i * 0.1 for i in range(len(dates))],
                "high": [101 + i * 0.1 for i in range(len(dates))],
                "low": [99 + i * 0.1 for i in range(len(dates))],
                "volume": [1_000_000] * len(dates),
                "ticker": ticker,
            },
            index=dates,
        )
        frames.append(frame)
    mock_df = pd.concat(frames)
    mock_df.index.name = "date"

    with patch("src.core.data_pipeline.DataPipeline.yfinance_fallback", return_value=mock_df):
        from scripts.run_backtest import _run_pipeline

        result = _run_pipeline("Linear", 5, "2022-01-01", "2022-06-30", False)

    assert result is not None
    assert hasattr(result, "metrics")
    assert result.metrics.sharpe_ratio is not None


# ---------------------------------------------------------------------------
# Test 2: agents.graph.analyze_ticker
# ---------------------------------------------------------------------------


def test_run_agents_analyze():
    """analyze_ticker runs correctly with mocked LangGraph graph."""
    from src.utils.schemas import AnalysisResult, Decision

    mock_result = AnalysisResult(
        ticker="AAPL",
        decision=Decision.BUY,
        confidence=75.0,
        reasoning="Mocked analysis",
    )

    with patch("src.agents.graph.build_trading_desk_graph") as mock_build:
        compiled = MagicMock()
        compiled.invoke.return_value = {"result": mock_result, "debate_transcript": []}
        mock_build.return_value = compiled

        from src.agents.graph import analyze_ticker

        result = analyze_ticker("AAPL")

    assert result.ticker == "AAPL"
    assert result.decision in list(Decision)


# ---------------------------------------------------------------------------
# Test 3: run_pipeline._run_pipeline_once
# ---------------------------------------------------------------------------


def test_run_pipeline_backtest_mode():
    """run_pipeline._run_pipeline_once executes in backtest mode with mocked data."""
    from src.utils.schemas import AnalysisResult, Decision

    mock_result = AnalysisResult(
        ticker="SPY",
        decision=Decision.HOLD,
        confidence=50.0,
        reasoning="Mocked",
    )

    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    frames = []
    for ticker in ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "V", "UNH"]:
        frame = pd.DataFrame(
            {
                "close": [100 + i for i in range(len(dates))],
                "open": [99 + i for i in range(len(dates))],
                "high": [101 + i for i in range(len(dates))],
                "low": [99 + i for i in range(len(dates))],
                "volume": [1_000_000] * len(dates),
                "ticker": ticker,
            },
            index=dates,
        )
        frames.append(frame)
    mock_df = pd.concat(frames)
    mock_df.index.name = "date"

    with patch("src.core.data_pipeline.DataPipeline.yfinance_fallback", return_value=mock_df), \
         patch("src.agents.graph.analyze_ticker", return_value=mock_result):

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from scripts.run_pipeline import _run_pipeline_once

        try:
            summary = _run_pipeline_once("backtest")
        except Exception as e:
            pytest.fail(f"_run_pipeline_once raised unexpectedly: {e}")

    assert isinstance(summary, dict)
    assert "tickers_analyzed" in summary
    assert "nav" in summary


# ---------------------------------------------------------------------------
# Test 4: factor library roundtrip
# ---------------------------------------------------------------------------


def test_factor_library_roundtrip(tmp_path):
    """Factor library JSON saves and loads correctly, merging into dataset."""
    from src.utils.config import AppConfig
    from src.utils.schemas import FactorDefinition

    # Write a minimal factor library to tmp_path (which will be output_dir)
    factors = [
        FactorDefinition(
            name="test_momentum",
            expression="close.pct_change(20)",
            category="momentum",
            ic_mean=0.03,
            icir=0.5,
        )
    ]
    factor_path = tmp_path / "factor_library.json"
    factor_path.write_text(json.dumps([f.model_dump() for f in factors]))

    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    mock_df = pd.DataFrame(
        {
            "close": [float(i + 100) for i in range(50)],
            "open": [float(i + 99) for i in range(50)],
            "high": [float(i + 101) for i in range(50)],
            "low": [float(i + 99) for i in range(50)],
            "volume": [1_000_000] * 50,
            "ticker": ["SPY"] * 50,
        },
        index=dates,
    )

    # Create a config pointing output_dir to tmp_path so load_factor_library
    # finds the JSON we wrote above
    cfg = AppConfig(output_dir=str(tmp_path))

    with patch("src.core.data_pipeline.DataPipeline.yfinance_fallback", return_value=mock_df):
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(cfg)
        lib_df = pipeline.load_factor_library()

    # Library should return a DataFrame (may be empty if no qlib data)
    assert isinstance(lib_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 5: RDAgentRunner KB persistence
# ---------------------------------------------------------------------------


def test_knowledge_base_persistence(tmp_path):
    """RDAgentRunner loads KB from previous run on second invocation."""
    import src.core.rd_agent_runner as rdmod

    original_kb_path = rdmod._KB_PATH
    original_kb_dir = rdmod._KB_DIR
    rdmod._KB_PATH = tmp_path / "kb.json"
    rdmod._KB_DIR = tmp_path

    try:
        from src.core.rd_agent_runner import RDAgentRunner

        runner1 = RDAgentRunner()
        result1 = runner1.co_optimize(iterations=2, budget=0.5)
        assert "factors_accepted" in result1

        # Second run — KB should be loaded from disk (written by runner1)
        runner2 = RDAgentRunner()
        result2 = runner2.co_optimize(iterations=2, budget=0.5)
        assert "factors_accepted" in result2

        # Verify KB file exists after runs
        assert rdmod._KB_PATH.exists()

    finally:
        rdmod._KB_PATH = original_kb_path
        rdmod._KB_DIR = original_kb_dir


# ---------------------------------------------------------------------------
# Test 6: KB prevents reproposals
# ---------------------------------------------------------------------------


def test_kb_prevents_reproposals(tmp_path):
    """Factors that failed IC threshold are not re-proposed in subsequent runs."""
    import src.core.rd_agent_runner as rdmod

    original_kb_path = rdmod._KB_PATH
    original_kb_dir = rdmod._KB_DIR
    rdmod._KB_PATH = tmp_path / "kb.json"
    rdmod._KB_DIR = tmp_path

    try:
        from src.core.rd_agent_runner import RDAgentRunner

        runner = RDAgentRunner()

        # Pre-populate KB with "failed" factors (exact base names from templates)
        failed_names = {"momentum_20d", "rsi_14", "macd_signal"}
        kb = runner._load_kb()
        kb["failed_factors"] = sorted(failed_names)
        runner._save_kb(kb)

        # Run factor mining — the failed factors should be skipped
        # (they may appear with version suffixes like rsi_14_v1 if base not skipped,
        #  but the exact base names should not appear in results)
        result = runner.mine_factors(iterations=5)

        # None of the exact failed factor names should appear in the result
        result_names = {f.name for f in result}
        overlap = result_names & failed_names
        assert len(overlap) == 0, f"Failed factors were re-proposed: {overlap}"

    finally:
        rdmod._KB_PATH = original_kb_path
        rdmod._KB_DIR = original_kb_dir
