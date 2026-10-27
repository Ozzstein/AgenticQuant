"""Check installed native libraries and Parquet without provider calls."""

from __future__ import annotations

import importlib
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger


def main() -> None:
    """Fail on missing/broken native dependencies or a corrupt Parquet round trip."""
    for module, distribution in (
        ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
        ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("xgboost", "xgboost"),
        ("cvxpy", "cvxpy"), ("riskfolio", "riskfolio-lib"), ("hmmlearn", "hmmlearn"),
        ("httpx", "httpx"), ("mcp", "mcp"), ("exchange_calendars", "exchange-calendars"),
        ("qlib", "pyqlib"), ("streamlit", "streamlit"), ("alpaca", "alpaca-py"),
    ):
        importlib.import_module(module)
        logger.info("Imported {} {}", distribution, version(distribution))

    import pandas as pd

    original = pd.DataFrame({"instrument_id": ["equity:one", "crypto:two"], "price": [10.25, 20.5]})
    with TemporaryDirectory(prefix="quantagent-env-") as directory:
        path = Path(directory) / "roundtrip.parquet"
        original.to_parquet(path, engine="pyarrow", index=False)
        pd.testing.assert_frame_equal(original, pd.read_parquet(path, engine="pyarrow"))
    logger.info("Native imports and Parquet round trip passed; provider calls were not exercised")


if __name__ == "__main__":
    main()
