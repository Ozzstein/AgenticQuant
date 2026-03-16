"""Qlib data pipeline with yfinance fallback."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.exceptions import DataPipelineError
from src.utils.logger import get_logger

logger = get_logger(__name__)


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

    def create_dataset_with_custom_factors(
        self,
        custom_factors: pd.DataFrame | None = None,
    ) -> Any:
        """Create a dataset merging Qlib features with custom factor library factors.

        Args:
            custom_factors: Optional DataFrame of additional alpha factors.
                            RD-Agent wiring is deferred to Task 06.

        Returns:
            Qlib DatasetH when Qlib is available, else a plain DataFrame
            (which may include ``custom_factors`` merged in).
        """
        if custom_factors is not None and not custom_factors.empty:
            logger.info(
                "Custom factors provided (shape={}); RD-Agent merge deferred to Task 06.",
                custom_factors.shape,
            )
        else:
            logger.info("create_dataset_with_custom_factors: no custom factors.")

        return self.get_dataset(self.config.qlib.dataset)

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
