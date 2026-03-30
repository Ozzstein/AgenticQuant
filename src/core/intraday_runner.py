"""Lightweight intraday pipeline runner for sub-daily signal generation and execution."""

from __future__ import annotations

from loguru import logger

from src.utils.config import AppConfig
from src.utils.schemas import IntradayRunResult, Signal, SignalDirection


_TIMEFRAME_MAP: dict[int, str] = {
    60: "1h",
    240: "4h",
    1440: "1d",
}


def _minutes_to_timeframe(minutes: int) -> str:
    """Map interval_minutes to a CCXT/yfinance timeframe string.

    Falls back to the nearest entry for non-standard values.

    Args:
        minutes: Interval in minutes (e.g. 60, 240, 1440).

    Returns:
        Timeframe string (e.g. "1h", "4h", "1d").
    """
    if minutes in _TIMEFRAME_MAP:
        return _TIMEFRAME_MAP[minutes]
    # Nearest match
    closest = min(_TIMEFRAME_MAP.keys(), key=lambda k: abs(k - minutes))
    return _TIMEFRAME_MAP[closest]


class IntradayRunner:
    """Lightweight intraday pipeline: fetch OHLCV → momentum score → signals → orders.

    Skips ML retraining and LLM analysis for speed. Suitable for sub-hourly execution.

    Args:
        config: AppConfig with intraday section configured.
        broker: Optional broker for order execution. If None, signals are generated
            but no orders are placed.
    """

    def __init__(self, config: AppConfig, broker=None) -> None:
        self._config = config
        self._broker = broker

    def run(self) -> IntradayRunResult:
        """Execute one intraday cycle: fetch → score → signals → orders → snapshot.

        Returns:
            IntradayRunResult summarising the cycle outcome.
        """
        cfg = self._config.intraday
        timeframe = _minutes_to_timeframe(cfg.interval_minutes)

        try:
            universe = self._resolve_universe()
            if not universe:
                return IntradayRunResult(
                    universe=[],
                    timeframe=timeframe,
                    signals_generated=0,
                    orders_placed=0,
                    orders_rejected=0,
                    nav=self._nav(),
                    status="skipped",
                    error="empty universe",
                )

            prices = self._fetch_prices(universe, timeframe)
            scores = self._score_tickers(prices)
            signals = self._build_signals(scores, cfg.top_n)

            orders_placed = 0
            orders_rejected = 0

            if self._broker is not None and signals:
                from src.execution.risk_controls import check_order  # noqa: PLC0415
                from src.execution.signal_translator import signals_to_orders  # noqa: PLC0415

                orders = signals_to_orders(signals, self._broker.portfolio, prices)
                for order in orders:
                    result = check_order(order, self._broker.portfolio, prices, self._config.risk)
                    if result.passed:
                        filled = self._broker.execute_order(order, prices)
                        if filled is not None:
                            orders_placed += 1
                        else:
                            orders_rejected += 1
                    else:
                        orders_rejected += 1
                        logger.debug(
                            "IntradayRunner: order rejected by risk — {}", result.failed_checks
                        )
                self._broker.snapshot(prices)

            logger.info(
                "IntradayRunner: {} signals, {} placed, {} rejected",
                len(signals),
                orders_placed,
                orders_rejected,
            )
            return IntradayRunResult(
                universe=universe,
                timeframe=timeframe,
                signals_generated=len(signals),
                orders_placed=orders_placed,
                orders_rejected=orders_rejected,
                nav=self._nav(),
                status="completed",
            )

        except Exception as exc:
            logger.error("IntradayRunner.run failed: {}", exc)
            return IntradayRunResult(
                universe=[],
                timeframe=timeframe,
                signals_generated=0,
                orders_placed=0,
                orders_rejected=0,
                nav=self._nav(),
                status="failed",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_universe(self) -> list[str]:
        """Return the ticker universe, falling back to CryptoPipeline top-N."""
        cfg = self._config.intraday
        if cfg.universe:
            return list(cfg.universe)
        # Default: top crypto tickers
        try:
            from src.core.crypto_pipeline import CryptoPipeline  # noqa: PLC0415

            pipeline = CryptoPipeline(self._config)
            return pipeline.build_universe()[: cfg.top_n]
        except Exception as exc:
            logger.warning("IntradayRunner: could not build crypto universe: {}", exc)
            return []

    def _fetch_prices(self, universe: list[str], timeframe: str) -> dict[str, float]:
        """Fetch latest close price for each ticker.

        Args:
            universe: List of ticker symbols.
            timeframe: OHLCV timeframe string (e.g. "1h", "4h").

        Returns:
            Mapping of ticker → latest close price.
        """
        prices: dict[str, float] = {}
        cfg = self._config.intraday

        # Lazily instantiate pipelines once — not inside the per-ticker loop.
        crypto_pipeline = None
        equity_pipeline = None

        for ticker in universe:
            try:
                if "/" in ticker or cfg.crypto_only:
                    if crypto_pipeline is None:
                        from src.core.crypto_pipeline import CryptoPipeline  # noqa: PLC0415

                        crypto_pipeline = CryptoPipeline(self._config)
                    symbol = (
                        ticker
                        if "/" in ticker
                        else f"{ticker}/{self._config.crypto.quote_currency}"
                    )
                    df = crypto_pipeline.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
                    if not df.empty:
                        prices[ticker] = float(df["close"].iloc[-1])
                else:
                    if equity_pipeline is None:
                        from src.core.data_pipeline import DataPipeline  # noqa: PLC0415

                        equity_pipeline = DataPipeline(self._config)
                    df = equity_pipeline.yfinance_fallback([ticker], interval=timeframe, period="2d")
                    if not df.empty:
                        close_col = (
                            ("Close", ticker) if ("Close", ticker) in df.columns else "Close"
                        )
                        prices[ticker] = float(df[close_col].dropna().iloc[-1])
            except Exception as exc:
                logger.warning("IntradayRunner: failed to fetch price for {}: {}", ticker, exc)

        return prices

    def _score_tickers(self, prices: dict[str, float]) -> dict[str, float]:
        """Compute a simple momentum score for each ticker.

        Uses the raw price as a proxy — in production this would use rolling returns
        computed from the full OHLCV series fetched in _fetch_prices.

        Args:
            prices: Ticker → latest close price.

        Returns:
            Ticker → score (higher = more bullish).
        """
        # Normalise prices to 0-1 range as a simple relative ranking score
        if not prices:
            return {}
        vals = list(prices.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {t: 0.5 for t in prices}
        return {t: (p - min_v) / (max_v - min_v) for t, p in prices.items()}

    def _build_signals(self, scores: dict[str, float], top_n: int) -> list[Signal]:
        """Generate LONG signals for top-N tickers, FLAT for the rest.

        Args:
            scores: Ticker → momentum score.
            top_n: Number of top tickers to go long.

        Returns:
            List of Signal objects.
        """
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        signals: list[Signal] = []
        for i, (ticker, score) in enumerate(ranked):
            direction = SignalDirection.LONG if i < top_n else SignalDirection.FLAT
            signals.append(
                Signal(ticker=ticker, direction=direction, strength=score, source="intraday")
            )
        return signals

    def _nav(self) -> float:
        """Return current portfolio NAV, or 0.0 if no broker."""
        if self._broker is None:
            return 0.0
        return self._broker.portfolio.nav
