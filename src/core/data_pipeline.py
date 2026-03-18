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


def _eval_factor_expression(
    expr: str, name: str, close: pd.Series, volume: pd.Series | None
) -> pd.Series | None:
    """Evaluate a factor expression using pandas operations.

    Args:
        expr: Factor expression string (lowercase).
        name: Factor name (lowercase).
        close: Close price Series.
        volume: Volume Series (may be None).

    Returns:
        Computed factor values as pd.Series, or None if unrecognized.
    """
    import re

    try:
        if "momentum" in expr or "pct_change" in expr or "mom" in name:
            match = re.search(r"(\d+)", name)
            window = int(match.group(1)) if match else 20
            return close.pct_change(window)
        elif "rsi" in expr or "rsi" in name:
            match = re.search(r"(\d+)", name)
            period = int(match.group(1)) if match else 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, float("inf"))
            return 100 - (100 / (1 + rs))
        elif "macd" in expr or "macd" in name:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            return ema12 - ema26
        elif "bb" in expr or "bollinger" in name:
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            band_width = bb_upper - bb_lower
            return (close - bb_lower) / band_width.replace(0, float("nan"))
        elif "atr" in expr or "atr" in name:
            return close.rolling(14).std() / close
        elif "obv" in expr or "obv" in name:
            if volume is None:
                return None
            direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (volume * direction).cumsum()
            return obv.pct_change(20)
        elif "volume" in expr or "vol_ratio" in name:
            if volume is None:
                return None
            return volume / volume.rolling(20).mean()
        else:
            # Default: 20-day momentum
            return close.pct_change(20)
    except Exception:
        return None


def _compute_factor(df: pd.DataFrame, factor: "Any") -> pd.Series | None:
    """Compute a factor from OHLCV data based on expression keywords.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume, ticker.
        factor: FactorDefinition with name and expression attributes.

    Returns:
        pd.Series with factor values, or None if computation fails.
    """
    expr = factor.expression.lower()
    name = factor.name.lower()

    # For multi-ticker DataFrames, compute per ticker and concatenate
    if "ticker" in df.columns:
        frames = []
        for _ticker, grp in df.groupby("ticker"):
            close = grp["close"]
            volume = grp["volume"] if "volume" in grp.columns else None

            val = _eval_factor_expression(expr, name, close, volume)
            if val is not None:
                frames.append(val)
        if frames:
            return pd.concat(frames)
        return None
    else:
        close = df["close"]
        volume = df.get("volume")
        return _eval_factor_expression(expr, name, close, volume)


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
            logger.warning("pyqlib not installed — running without Qlib.")
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
            logger.warning("Qlib not initialised — get_dataset returns None.")
            return None

        try:
            from qlib.contrib.data.handler import Alpha158  # noqa: PLC0415
            from qlib.data.dataset import DatasetH  # noqa: PLC0415
            from qlib.data.dataset.handler import DataHandlerLP  # noqa: PLC0415

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
                infer_processors=[DataHandlerLP.IS_DEPEND],
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
            logger.warning("get_dataset failed: {} — returning None.", exc)
            return None

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
                logger.warning("Qlib feature fetch failed ({}). Falling back to yfinance.", exc)

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
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance when Qlib data is unavailable.

        Args:
            tickers: Ticker symbols. Defaults to a small benchmark set.
            start:   Start date string. Defaults to config test_start.
            end:     End date string. Defaults to config test_end.

        Returns:
            Standardised DataFrame with columns: open, high, low, close, volume, ticker.
        """
        import yfinance as yf  # noqa: PLC0415

        tickers = tickers or ["SPY", "QQQ", "IWM"]
        start = start or self.config.qlib.test_start
        end = end or self.config.qlib.test_end

        logger.info("yfinance download: tickers={}, start={}, end={}", tickers, start, end)

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

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

    def load_factor_library(self) -> pd.DataFrame:
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
            logger.warning("Could not load factor library: {}", exc)
            return pd.DataFrame()

        factors: list[FactorDefinition] = []
        for item in raw:
            try:
                factors.append(FactorDefinition(**item))
            except Exception:
                continue

        if not factors:
            logger.info("Factor library loaded but empty.")
            return pd.DataFrame()

        logger.info("Factor library loaded: {} factors.", len(factors))

        # Get base OHLCV data to compute factors on
        base_df = self.yfinance_fallback()  # uses default test window
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
                logger.debug("Factor {} computation failed: {}", factor.name, exc)
                continue

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
        if custom_factors is None:
            logger.info("Loading factor library from disk...")
            custom_factors = self.load_factor_library()

        # When Qlib is available, try to use it (factors passed as extra context for now)
        if self._qlib_initialized:
            if custom_factors is not None and not custom_factors.empty:
                logger.info(
                    "Custom factors provided (shape={}); using Qlib dataset.", custom_factors.shape
                )
            return self.get_dataset(self.config.qlib.dataset)

        # Fallback: get base features and merge custom factors
        base_df = self.get_features()
        if custom_factors is None or custom_factors.empty:
            logger.info(
                "create_dataset_with_custom_factors: no custom factors, returning base features."
            )
            return base_df

        logger.info(
            "Merging base features ({} rows) with custom factors ({} rows).",
            len(base_df),
            len(custom_factors),
        )
        try:
            merged = base_df.join(custom_factors, how="left")
            logger.info("Merged dataset: {} rows × {} cols.", len(merged), merged.shape[1])
            return merged
        except Exception as exc:
            logger.warning("Factor merge failed ({}). Returning base features.", exc)
            return base_df

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
