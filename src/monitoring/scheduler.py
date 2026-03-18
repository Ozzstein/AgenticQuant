"""Pipeline scheduler using APScheduler."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger


class PipelineScheduler:
    """Wraps the daily pipeline with APScheduler scheduling.

    Attributes:
        config: AppConfig instance.
        _scheduler: APScheduler BackgroundScheduler (or None if not available).
        _lock_file: Path to lock file preventing concurrent runs.
    """

    def __init__(self, config=None) -> None:
        from src.utils.config import get_config

        self.config = config or get_config()
        self._scheduler = None
        self._lock_file = Path("outputs/.pipeline.lock")
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()
            logger.info("APScheduler initialized.")
        except ImportError:
            logger.warning("APScheduler not installed — scheduler will run manually only.")

    def run_once(self) -> dict:
        """Run the pipeline once immediately.

        Returns:
            dict with: status, tickers_analyzed, orders_placed, timestamp
        """
        if self._lock_file.exists():
            logger.warning("Pipeline lock file exists — skipping (another run in progress).")
            return {"status": "skipped", "reason": "lock_file_exists"}

        self._lock_file.touch()
        try:
            logger.info("PipelineScheduler: running pipeline once...")
            # Import and run the core pipeline steps
            from src.agents.graph import analyze_ticker
            from src.core.data_pipeline import DataPipeline

            config = self.config
            _UNIVERSE = ["SPY", "AAPL", "MSFT"]  # small universe for scheduler runs

            pipeline = DataPipeline(config)
            df = pipeline.yfinance_fallback(
                tickers=_UNIVERSE,
                start=config.qlib.test_start,
                end=config.qlib.test_end,
            )

            tickers_analyzed = 0
            if not df.empty:
                for ticker in _UNIVERSE:
                    try:
                        analyze_ticker(ticker)  # graceful stubs if no API key
                        tickers_analyzed += 1
                    except Exception as exc:
                        logger.warning("scheduler run_once: {} failed: {}", ticker, exc)

            result = {
                "status": "completed",
                "tickers_analyzed": tickers_analyzed,
                "orders_placed": 0,
                "timestamp": datetime.now().isoformat(),
            }
            logger.info("Pipeline run completed: {}", result)
            return result
        except Exception as exc:
            logger.error("Pipeline run failed: {}", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._lock_file.unlink(missing_ok=True)

    def start(self, schedule_time: str | None = None) -> None:
        """Schedule the pipeline to run daily at schedule_time.

        Args:
            schedule_time: Time string "HH:MM" in config.monitoring.schedule_timezone.
        """
        if self._scheduler is None:
            logger.warning("APScheduler not available — cannot schedule.")
            return

        time_str = schedule_time or self.config.monitoring.schedule_time
        hour, minute = time_str.split(":")
        self._scheduler.add_job(
            self.run_once,
            trigger="cron",
            hour=int(hour),
            minute=int(minute),
            timezone=self.config.monitoring.schedule_timezone,
            misfire_grace_time=300,
        )
        self._scheduler.start()
        logger.info(
            "Pipeline scheduled daily at {} {}", time_str, self.config.monitoring.schedule_timezone
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown()
            logger.info("PipelineScheduler stopped.")
