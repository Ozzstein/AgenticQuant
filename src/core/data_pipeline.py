"""Qlib data pipeline with yfinance fallback."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import DataPipelineError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Module-level factor computation helpers
# ------------------------------------------------------------------


def _eval_factor_expression(expr: str, name: str, close: pd.Series, volume: pd.Series | None) -> pd.Series:
    """Compatibility adapter to the exact shared expression interpreter."""
    from src.data.features.expressions import evaluate_expression
    frame = pd.DataFrame({"close": close})
    if volume is not None:
        frame["volume"] = volume
    return evaluate_expression(expr, frame).rename(name)


def _compute_factor(df: pd.DataFrame, factor: Any) -> pd.Series:
    from src.data.features.expressions import evaluate_expression
    return evaluate_expression(factor.expression, df).rename(factor.name)


def instrument_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct an unambiguous date/instrument join key; never join by date alone."""
    if isinstance(frame.index, pd.MultiIndex):
        if not frame.index.is_unique:
            raise DataPipelineError("duplicate date/instrument rows")
        return frame
    identity = next((c for c in ("instrument_id", "instrument", "ticker") if c in frame), None)
    if identity is None:
        raise DataPipelineError("instrument identity is required for panel joins")
    result = frame.copy()
    result.index = pd.MultiIndex.from_arrays([frame.index, frame[identity]], names=["datetime", "instrument"])
    if not result.index.is_unique:
        raise DataPipelineError("duplicate date/instrument rows")
    return result


class DataPipeline:
    """Qlib-backed data pipeline with graceful yfinance fallback.

    All Qlib imports are lazy (inside methods) to allow the class to be
    instantiated even when pyqlib is not installed.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._qlib_initialized = False

    # ------------------------------------------------------------------
    # Qlib initialisation
    # ------------------------------------------------------------------

    def init_qlib(self) -> None:
        """Initialize Qlib with settings from config.

        Logs a warning and continues gracefully when pyqlib is not installed.

        Raises:
            DataPipelineError: On any non-ImportError failure during Qlib init.
        """
        try:
            import qlib  # noqa: PLC0415

            qlib.init(
                provider_uri=self.config.qlib.provider_uri,
                region=self.config.qlib.region,
            )
            self._qlib_initialized = True
            logger.info(
                "Qlib initialized: uri={}, region={}",
                self.config.qlib.provider_uri,
                self.config.qlib.region,
            )
        except ImportError:
            raise DataPipelineError("pyqlib is required for the configured Qlib scenario") from None
        except Exception as exc:
            raise DataPipelineError(f"Qlib initialisation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def get_dataset(self, dataset_type: str = "Alpha158") -> Any:
        """Create a Qlib DatasetH for the requested dataset type.

        Args:
            dataset_type: ``"Alpha158"`` or ``"Alpha360"``.

        Returns:
            A Qlib ``DatasetH`` instance, or ``None`` when Qlib is unavailable.
        """
        if not self._qlib_initialized:
            raise DataPipelineError("Qlib must be initialized with an explicit verified US bundle")

        try:
            from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
            from qlib.data.dataset import DatasetH  # noqa: PLC0415

            if dataset_type not in ("Alpha158", "Alpha360"):
                raise DataPipelineError(f"Unsupported Qlib dataset: {dataset_type}")
            handler_cls = Alpha158
            if dataset_type == "Alpha360":
                from qlib.contrib.data.handler import Alpha360  # noqa: PLC0415

                handler_cls = Alpha360

            handler = handler_cls(
                instruments=self.config.qlib.universe,
                start_time=self.config.qlib.train_start,
                end_time=self.config.qlib.test_end,
                fit_start_time=self.config.qlib.train_start,
                fit_end_time=self.config.qlib.train_end,
                infer_processors=[{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                                  {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
            )
            dataset = DatasetH(
                handler=handler,
                segments={
                    "train": (self.config.qlib.train_start, self.config.qlib.train_end),
                    "valid": (self.config.qlib.valid_start, self.config.qlib.valid_end),
                    "test": (self.config.qlib.test_start, self.config.qlib.test_end),
                },
            )
            logger.info("Qlib DatasetH ({}) created.", dataset_type)
            return dataset
        except Exception as exc:
            raise DataPipelineError(f"Qlib dataset construction failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Feature retrieval
    # ------------------------------------------------------------------

    def get_features(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return a feature matrix, preferring Qlib with yfinance as fallback.

        Args:
            start: Start date string (``YYYY-MM-DD``). Defaults to config test_start.
            end:   End date string (``YYYY-MM-DD``). Defaults to config test_end.

        Returns:
            DataFrame with date/ticker columns or a Qlib-style MultiIndex.
        """
        start = start or self.config.qlib.test_start
        end = end or self.config.qlib.test_end

        if self._qlib_initialized:
            try:
                from qlib.data import D  # noqa: PLC0415

                fields = ["$open", "$high", "$low", "$close", "$volume"]
                names = ["open", "high", "low", "close", "volume"]
                df: pd.DataFrame = D.features(
                    D.instruments(self.config.qlib.universe),
                    fields,
                    start_time=start,
                    end_time=end,
                )
                df.columns = names
                logger.info(
                    "Qlib features retrieved: {} rows, start={}, end={}",
                    len(df),
                    start,
                    end,
                )
                return df
            except Exception as exc:
                raise DataPipelineError(f"Qlib feature fetch failed: {exc}") from exc

        logger.info("Using yfinance fallback for features: start={}, end={}", start, end)
        return self.yfinance_fallback(start=start, end=end)

    # ------------------------------------------------------------------
    # yfinance fallback
    # ------------------------------------------------------------------

    def yfinance_fallback(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        period: str | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance when Qlib data is unavailable.

        Args:
            tickers:  Ticker symbols. Defaults to a small benchmark set.
            start:    Start date string. Defaults to config test_start.
            end:      End date string. Defaults to config test_end.
            interval: yfinance interval string (e.g. ``"1d"``, ``"1h"``, ``"4h"``).
                      Intraday intervals (< ``"1d"``) are limited to the last 60 days
                      by the yfinance API.
            period:   yfinance period shorthand (e.g. ``"2d"``, ``"5d"``). When set,
                      takes precedence over ``start``/``end``.

        Returns:
            Standardised DataFrame with columns: open, high, low, close, volume, ticker.
        """
        import yfinance as yf  # noqa: PLC0415

        if not tickers:
            raise DataPipelineError("explicit instruments required; no benchmark-universe substitution")

        logger.info(
            "yfinance download: tickers={}, interval={}, start={}, end={}, period={}",
            tickers,
            interval,
            start,
            end,
            period,
        )

        if period is not None:
            raw = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)
        else:
            start = start or self.config.qlib.test_start
            end = end or self.config.qlib.test_end
            raw = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=False, progress=False)

        if raw.empty:
            logger.warning("yfinance returned empty DataFrame.")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "ticker"])

        # Normalise multi-level columns produced by yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            frames: list[pd.DataFrame] = []
            for ticker in tickers:
                try:
                    sub = raw.xs(ticker, axis=1, level=1).copy()
                except KeyError:
                    continue
                sub.columns = [c.lower() for c in sub.columns]
                sub["ticker"] = ticker
                frames.append(sub)
            df = pd.concat(frames) if frames else pd.DataFrame()
        else:
            df = raw
            df.columns = [c.lower() for c in df.columns]
            if len(tickers) == 1:
                df["ticker"] = tickers[0]

        df.index.name = "date"
        logger.info("yfinance fallback: {} rows returned.", len(df))
        return df

    # ------------------------------------------------------------------
    # Custom factor integration
    # ------------------------------------------------------------------

    def load_factor_library(self, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Load factor library from outputs/factor_library.json and compute factor values.

        Loads the JSON file produced by RDAgentRunner.save_factor_library().
        For each FactorDefinition with a non-empty expression, attempts to compute
        the factor value on yfinance data. Returns a DataFrame of computed factor values.

        Returns:
            DataFrame with factor values indexed by (date, ticker). Empty DataFrame
            if the library file is missing or no factors can be computed.
        """
        import json
        from pathlib import Path

        from src.utils.schemas import FactorDefinition

        factor_library_path = Path(self.config.output_dir) / "factor_library.json"
        if not factor_library_path.exists():
            logger.info("Factor library not found at {}.", factor_library_path)
            return pd.DataFrame()

        try:
            with factor_library_path.open() as fh:
                raw = json.load(fh)
        except Exception as exc:
            raise DataPipelineError(f"Invalid factor library: {exc}") from exc

        factors: list[FactorDefinition] = []
        for item in raw:
            try:
                factors.append(FactorDefinition(**item))
            except Exception as exc:
                raise DataPipelineError(f"Invalid factor definition: {exc}") from exc

        if not factors:
            logger.info("Factor library loaded but empty.")
            return pd.DataFrame()

        logger.info("Factor library loaded: {} factors.", len(factors))

        # Get base OHLCV data to compute factors on
        if base_df is None:
            base_df = self.get_features()
        base_df = instrument_index(base_df)
        if base_df.empty:
            return pd.DataFrame()

        # Compute each factor on the base data
        # For now: evaluate simple numpy/pandas expressions
        # Supported factor types based on expression keywords
        factor_frames: list[pd.DataFrame] = []

        for factor in factors:
            try:
                factor_col = _compute_factor(base_df, factor)
                if factor_col is not None:
                    factor_frames.append(factor_col.rename(factor.name))
            except Exception as exc:
                raise DataPipelineError(f"Factor {factor.name} failed exactly as specified: {exc}") from exc

        if not factor_frames:
            return pd.DataFrame()

        result = pd.concat(factor_frames, axis=1)
        logger.info("Factor library computed: {} factors, {} rows.", len(factor_frames), len(result))
        return result

    def create_dataset_with_custom_factors(
        self,
        custom_factors: pd.DataFrame | None = None,
    ) -> Any:
        """Create a dataset merging Qlib features with custom factor library factors.

        If custom_factors is None, loads the factor library from disk using
        load_factor_library(). Merges with base yfinance features on the date index.

        Args:
            custom_factors: Optional DataFrame of additional alpha factors.
                            If None, loads from outputs/factor_library.json.

        Returns:
            Qlib DatasetH when Qlib is available, else a plain DataFrame
            with base features merged with any custom factors.
        """
        base_df = instrument_index(self.get_features())
        if custom_factors is None:
            custom_factors = self.load_factor_library(base_df)
        if custom_factors.empty:
            return base_df
        custom_factors = instrument_index(custom_factors) if not isinstance(custom_factors.index, pd.MultiIndex) else custom_factors
        if not custom_factors.index.is_unique:
            raise DataPipelineError("custom factors contain duplicate instrument/date keys")
        if not custom_factors.index.isin(base_df.index).all():
            raise DataPipelineError("custom factors contain observations outside base data")
        try:
            return base_df.join(custom_factors, how="left", validate="one_to_one")
        except Exception as exc:
            raise DataPipelineError(f"Custom factor join failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Crypto / equity merge
    # ------------------------------------------------------------------

    def merge_crypto_data(
        self,
        equity_df: pd.DataFrame,
        crypto_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge equity and crypto DataFrames on their date index.

        Args:
            equity_df: Equity OHLCV DataFrame indexed by date.
            crypto_df: Crypto OHLCV DataFrame indexed by date.

        Returns:
            Concatenated DataFrame sorted by date.
        """
        logger.info(
            "Merging equity ({} rows) and crypto ({} rows) data.",
            len(equity_df),
            len(crypto_df),
        )
        merged = pd.concat([equity_df, crypto_df]).sort_index()
        return merged
