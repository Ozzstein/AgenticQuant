"""Unit tests for DataPipeline and RDAgentRunner."""

from __future__ import annotations

import builtins
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from src.utils.schemas import FactorDefinition

# Capture the real built-in import so side-effect helpers can delegate to it.
_real_import = builtins.__import__


def _import_blocker(*blocked: str):
    """Return an import side-effect that raises ImportError for *blocked* names."""

    def _side_effect(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"No module named '{name}'")
        return _real_import(name, *args, **kwargs)

    return _side_effect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 10) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [102.0] * n,
            "volume": [1_000_000] * n,
            "ticker": ["SPY"] * n,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# DataPipeline — Qlib init
# ---------------------------------------------------------------------------


class TestInitQlib:
    def test_init_qlib_success(self, mock_config):
        """init_qlib calls qlib.init with provider_uri and region."""
        from src.core.data_pipeline import DataPipeline

        mock_qlib = MagicMock()
        with patch.dict("sys.modules", {"qlib": mock_qlib}):
            pipeline = DataPipeline(config=mock_config)
            pipeline.init_qlib()

        mock_qlib.init.assert_called_once_with(
            provider_uri=mock_config.qlib.provider_uri,
            region=mock_config.qlib.region,
        )
        assert pipeline._qlib_initialized is True

    def test_init_qlib_graceful_no_qlib(self, mock_config):
        """init_qlib does not raise when pyqlib is not installed."""
        from src.core.data_pipeline import DataPipeline

        with patch.dict("sys.modules", {"qlib": None}):
            pipeline = DataPipeline(config=mock_config)
            with patch("builtins.__import__", side_effect=_import_blocker("qlib")):
                pipeline.init_qlib()

        assert pipeline._qlib_initialized is False

    def test_init_qlib_raises_data_pipeline_error_on_other_error(self, mock_config):
        """init_qlib wraps unexpected errors in DataPipelineError."""
        from src.core.data_pipeline import DataPipeline
        from src.utils.exceptions import DataPipelineError

        mock_qlib = MagicMock()
        mock_qlib.init.side_effect = RuntimeError("bad provider uri")

        with patch.dict("sys.modules", {"qlib": mock_qlib}):
            pipeline = DataPipeline(config=mock_config)
            with pytest.raises(DataPipelineError, match="bad provider uri"):
                pipeline.init_qlib()


# ---------------------------------------------------------------------------
# DataPipeline — get_features / yfinance fallback
# ---------------------------------------------------------------------------


class TestGetFeatures:
    def test_get_features_falls_back_to_yfinance(self, mock_config):
        """When Qlib is not initialized, get_features delegates to yfinance_fallback."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        assert pipeline._qlib_initialized is False

        expected_df = _make_ohlcv()
        with patch.object(pipeline, "yfinance_fallback", return_value=expected_df) as mock_fb:
            result = pipeline.get_features(start="2023-01-01", end="2023-12-31")

        mock_fb.assert_called_once_with(start="2023-01-01", end="2023-12-31")
        assert result is expected_df

    def test_get_features_uses_config_dates_by_default(self, mock_config):
        """get_features uses config test_start/test_end when no dates supplied."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        expected_df = _make_ohlcv()

        with patch.object(pipeline, "yfinance_fallback", return_value=expected_df) as mock_fb:
            pipeline.get_features()

        mock_fb.assert_called_once_with(
            start=mock_config.qlib.test_start,
            end=mock_config.qlib.test_end,
        )


class TestYfinanceFallback:
    def test_yfinance_fallback_returns_dataframe(self, mock_config):
        """yfinance_fallback returns a non-empty DataFrame when download succeeds."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        mock_df = _make_ohlcv()

        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            result = pipeline.yfinance_fallback(
                tickers=["SPY"], start="2023-01-01", end="2023-06-30"
            )

        mock_dl.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_yfinance_fallback_uses_config_dates(self, mock_config):
        """yfinance_fallback falls back to config test_start/test_end."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        mock_df = _make_ohlcv()

        with patch("yfinance.download", return_value=mock_df) as mock_dl:
            pipeline.yfinance_fallback()

        assert mock_dl.call_args is not None
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["start"] == mock_config.qlib.test_start
        assert kwargs["end"] == mock_config.qlib.test_end

    def test_yfinance_fallback_empty_response(self, mock_config):
        """yfinance_fallback returns an empty DataFrame with expected columns on empty response."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        empty_df = pd.DataFrame()

        with patch("yfinance.download", return_value=empty_df):
            result = pipeline.yfinance_fallback(tickers=["FAKE"])

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_yfinance_fallback_multi_ticker_normalisation(self, mock_config):
        """yfinance_fallback handles multi-level columns from yfinance correctly."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        tickers = ["SPY", "QQQ"]
        dates = pd.bdate_range("2023-01-01", periods=5)
        # Build a MultiIndex DataFrame (Price × Ticker) as yfinance produces
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], tickers],
            names=["Price", "Ticker"],
        )
        data = [[100, 200, 102, 202, 101, 201, 99, 199, 1_000_000, 2_000_000]] * 5
        mock_multi = pd.DataFrame(data, index=dates, columns=columns)

        with patch("yfinance.download", return_value=mock_multi):
            result = pipeline.yfinance_fallback(tickers=tickers)

        assert isinstance(result, pd.DataFrame)
        assert "ticker" in result.columns


# ---------------------------------------------------------------------------
# DataPipeline — custom factors & merge
# ---------------------------------------------------------------------------


class TestCustomFactors:
    def test_create_dataset_with_custom_factors_none(self, mock_config):
        """Accepts None without raising; returns base features DataFrame when Qlib unavailable."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        base = pd.DataFrame(
            {"open": [100.0], "close": [101.0]},
            index=pd.to_datetime(["2023-01-02"]),
        )
        # Patch yfinance_fallback to avoid real network calls
        with patch.object(pipeline, "yfinance_fallback", return_value=base):
            result = pipeline.create_dataset_with_custom_factors(custom_factors=None)
        # Qlib not available → returns base features DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_create_dataset_with_custom_factors_df(self, mock_config):
        """Accepts a DataFrame without raising; merges with base features when Qlib unavailable."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        base = pd.DataFrame(
            {"open": [100.0], "close": [101.0]},
            index=pd.to_datetime(["2023-01-02"]),
        )
        factors_df = pd.DataFrame({"factor_a": [1.0], "factor_b": [0.5]})
        with patch.object(pipeline, "yfinance_fallback", return_value=base):
            result = pipeline.create_dataset_with_custom_factors(custom_factors=factors_df)
        # Qlib not available → returns merged DataFrame
        assert isinstance(result, pd.DataFrame)


class TestMergeCryptoData:
    def test_merge_crypto_data(self, mock_config):
        """merge_crypto_data combines equity and crypto DataFrames on date index."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)

        equity_df = pd.DataFrame(
            {"close": [100.0, 101.0], "ticker": ["SPY", "SPY"]},
            index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
        )
        crypto_df = pd.DataFrame(
            {"close": [20_000.0, 21_000.0], "ticker": ["BTC-USD", "BTC-USD"]},
            index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
        )

        merged = pipeline.merge_crypto_data(equity_df, crypto_df)

        assert len(merged) == 4
        assert set(merged["ticker"]) == {"SPY", "BTC-USD"}

    def test_merge_crypto_data_sorted_by_date(self, mock_config):
        """merge_crypto_data output is sorted by date index."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)

        equity_df = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime(["2023-01-05"]),
        )
        crypto_df = pd.DataFrame(
            {"close": [20_000.0]},
            index=pd.to_datetime(["2023-01-02"]),
        )

        merged = pipeline.merge_crypto_data(equity_df, crypto_df)

        assert merged.index[0] < merged.index[-1]


# ---------------------------------------------------------------------------
# DataPipeline — mock_config fixture integration
# ---------------------------------------------------------------------------


class TestDataPipelineWithMockConfig:
    def test_data_pipeline_with_mock_config(self, mock_config):
        """DataPipeline initialises correctly from the mock_config fixture."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline(config=mock_config)
        assert pipeline.config is mock_config
        assert pipeline._qlib_initialized is False

    def test_data_pipeline_default_config(self):
        """DataPipeline accepts no config and falls back to get_config()."""
        from src.core.data_pipeline import DataPipeline

        pipeline = DataPipeline()
        assert pipeline.config is not None
        assert hasattr(pipeline.config, "qlib")


# ---------------------------------------------------------------------------
# RDAgentRunner — stub mode
# ---------------------------------------------------------------------------


class TestRDAgentRunnerStub:
    def test_rdagent_runner_stub_mode(self):
        """run_factor_search returns [] when rdagent is not installed."""
        from src.core.rd_agent_runner import RDAgentRunner

        with patch.dict("sys.modules", {"rdagent": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("rdagent")):
                runner = RDAgentRunner()

        assert runner._rd_agent_available is False
        assert runner.run_factor_search() == []

    def test_rdagent_runner_stub_model_search(self):
        """run_model_search returns {} when rdagent is not installed."""
        from src.core.rd_agent_runner import RDAgentRunner

        with patch.dict("sys.modules", {"rdagent": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("rdagent")):
                runner = RDAgentRunner()

        assert runner.run_model_search() == {}

    def test_rdagent_runner_save_factor_library(self, tmp_path):
        """save_factor_library writes a valid JSON file."""
        from src.core.rd_agent_runner import RDAgentRunner
        from src.utils.config_loader import FullAppConfig

        lib_dir = tmp_path / "factor_library"
        cfg = FullAppConfig(
            rd_agent={"factor_library_dir": str(lib_dir), "best_model_config_path": str(tmp_path / "model.yaml")}
        )

        runner = RDAgentRunner(config=cfg)
        factors = [
            FactorDefinition(name="momentum_5d", expression="Ref($close,5)/$close-1", ic_mean=0.03),
            FactorDefinition(name="vol_20d", expression="Std($close,20)", ic_mean=0.01),
        ]
        runner.save_factor_library(factors)

        json_path = lib_dir / "factor_library.json"
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert len(loaded) == 2
        assert loaded[0]["name"] == "momentum_5d"

    def test_rdagent_runner_save_model_config(self, tmp_path):
        """save_model_config writes a valid YAML file."""
        from src.core.rd_agent_runner import RDAgentRunner
        from src.utils.config_loader import FullAppConfig

        yaml_path = tmp_path / "best_model_config.yaml"
        cfg = FullAppConfig(
            rd_agent={
                "factor_library_dir": str(tmp_path / "factors"),
                "best_model_config_path": str(yaml_path),
            }
        )

        runner = RDAgentRunner(config=cfg)
        model_cfg = {"model": "LightGBM", "n_estimators": 300, "learning_rate": 0.05}
        runner.save_model_config(model_cfg)

        assert yaml_path.exists()
        loaded = yaml.safe_load(yaml_path.read_text())
        assert loaded["model"] == "LightGBM"
        assert loaded["n_estimators"] == 300


