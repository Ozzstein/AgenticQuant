"""Closed credential lookup for read-only refresh transports.

Configuration uses the application's documented precedence. Unprefixed vendor
environment names remain a compatibility fallback; values never enter a task,
source manifest or operating receipt.
"""

from __future__ import annotations

import os

from src.utils.config import get_config


def provider_credential(name: str) -> str:
    config = get_config()
    values = {
        "fmp": (config.fmp_api_key, "FMP_API_KEY"),
        "fred": (config.fred_api_key, "FRED_API_KEY"),
        "finnhub": (config.finnhub_api_key, "FINNHUB_API_KEY"),
        "coingecko": (config.coingecko_api_key, "COINGECKO_API_KEY"),
        "sec": (config.sec_user_agent, "SEC_USER_AGENT"),
        "alpaca_key": (config.alpaca.api_key, "ALPACA_API_KEY"),
        "alpaca_secret": (config.alpaca.api_secret, "ALPACA_API_SECRET"),
    }
    if name not in values:
        raise ValueError("unsupported read-only provider credential")
    configured, legacy = values[name]
    return configured or os.getenv(legacy, "")
