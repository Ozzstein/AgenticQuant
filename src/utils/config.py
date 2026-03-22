"""Application configuration using Pydantic BaseSettings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class QlibConfig(BaseModel):
    provider_uri: str = str(PROJECT_ROOT / "data" / "qlib_data")
    region: str = "us"
    dataset: str = "Alpha158"
    train_start: str = "2010-01-01"
    train_end: str = "2022-12-31"
    valid_start: str = "2022-01-01"
    valid_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"
    universe: str = "us_all"
    model_name: str = "LightGBM"
    model_params: dict[str, int | float | str] = Field(default_factory=lambda: {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "num_leaves": 63
    })
    strategy_topk: int = 30
    strategy_n_drop: int = 5
    account: float = 100_000_000.0
    cost_fixed: float = 0.0
    cost_rate: float = 0.0002


class AgentConfig(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    deep_think_model: str = "claude-opus-4-20250514"
    quick_think_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.2
    max_tokens: int = 4096
    max_debate_rounds: int = 2
    debate_confidence_threshold: float = 20.0
    debate_trigger_spread: float = 40.0
    fundamental_weight: float = 0.35
    quant_weight: float = 0.25
    technical_weight: float = 0.20
    sentiment_weight: float = 0.10
    risk_weight: float = 0.10
    # Per-agent enable flags
    fundamental_enabled: bool = True
    sentiment_enabled: bool = True
    technical_enabled: bool = True
    risk_manager_enabled: bool = True
    strategist_enabled: bool = True
    # Cost controls
    max_cost_per_analysis: float = 0.50
    cache_ttl_seconds: int = 3600
    tools_enabled: list[str] = Field(default_factory=lambda: [
        "get_stock_price", "get_stock_history", "get_financials",
        "get_company_news", "get_market_news", "get_analyst_ratings",
        "get_insider_trades", "get_sec_filings", "get_earnings_calendar",
        "get_earnings_estimate", "get_fear_greed_index", "get_vix",
        "get_sector_performance", "get_options_data",
        "get_crypto_price", "get_crypto_history", "get_crypto_fundamentals",
        "get_crypto_fear_greed", "get_crypto_on_chain",
    ])


class CryptoConfig(BaseModel):
    exchange: str = "binance"
    top_n_universe: int = 20
    quote_currency: str = "USDT"
    max_weight_pct: float = 5.0
    total_crypto_min_pct: float = 5.0
    total_crypto_max_pct: float = 25.0


class LeanConfig(BaseModel):
    enabled: bool = False
    project_path: str = ""
    deploy_on_signal: bool = False
    initial_cash: float = 100_000.0
    broker: str = "paper"
    commission: float = 0.001
    slippage: float = 0.001
    rebalance_frequency: str = "daily"


class MonitoringConfig(BaseModel):
    telegram_token: str = ""
    telegram_chat_id: str = ""
    schedule_time: str = "16:00"
    schedule_timezone: str = "US/Eastern"
    health_alert_threshold: int = 60
    drawdown_alert_pct: float = 10.0
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_to: str | None = None


class ModelConfig(BaseModel):
    default_model: str = "LightGBM"
    topk: int = 30
    walk_forward_months: int = 3
    embargo_days: int = 5
    transaction_cost_bps: float = 10.0
    retrain_frequency: str = "monthly"
    ensemble_models: list[str] = Field(
        default_factory=lambda: ["LightGBM", "XGBoost", "Linear"]
    )
    ensemble_method: str = "ic_weighted"
    ensemble_val_fraction: float = 0.15


class RiskConfig(BaseModel):
    max_position_weight: float = 10.0
    max_drawdown_halt: float = 15.0
    max_daily_turnover: float = 25.0
    min_liquidity_adv: float = 1_000_000.0
    max_market_impact_pct: float = 1.0
    min_cash_pct: float = 5.0


class StrategyConfig(BaseModel):
    """Strategy selection and lifecycle configuration."""

    selection_method: str = "regime_based"  # regime_based | performance_weighted | bandit | manual
    forced_strategy: str = ""  # for manual override
    blend_enabled: bool = False
    min_backtest_sharpe: float = 0.5
    drift_threshold: float = 0.50  # alert if live Sharpe < 50% of backtest Sharpe
    drift_window_days: int = 60
    default_strategy: str = "momentum_topk"


class MacroRegimeConfig(BaseModel):
    """HMM regime detector configuration."""

    hmm_states: int = 4
    confidence_threshold: float = 0.7
    min_regime_days: int = 3
    retrain_frequency: str = "quarterly"
    normalization_window: int = 252
    signals: list[str] = Field(default_factory=lambda: [
        "vix_level",
        "vix_roc_10d",
        "yield_curve_10y2y",
        "sp500_breadth",
        "sp500_realized_vol_20d",
        "sp500_momentum_20d",
        "dxy_roc_20d",
        "credit_spread_proxy",
    ])


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = {"env_prefix": "AIQUANT_", "env_nested_delimiter": "__"}

    project_name: str = "QuantAgentLab"
    log_level: str = "INFO"
    output_dir: str = str(PROJECT_ROOT / "outputs")
    log_dir: str = str(PROJECT_ROOT / "logs")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    finnhub_api_key: str = ""
    coingecko_api_key: str = ""

    qlib: QlibConfig = Field(default_factory=QlibConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    lean: LeanConfig = Field(default_factory=LeanConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    macro_regime: MacroRegimeConfig = Field(default_factory=MacroRegimeConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v_upper


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from YAML file if it exists."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_dotenv(path: Path) -> None:
    """Load .env file into environment if it exists."""
    if path.exists():
        from dotenv import load_dotenv
        load_dotenv(path)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get singleton application config. Reads settings.yaml + .env, then env vars."""
    config_dir = PROJECT_ROOT / "config"
    _load_dotenv(config_dir / ".env")

    yaml_data = _load_yaml_config(config_dir / "settings.yaml")
    # Strip keys unknown to AppConfig so rd_agent (and any future FullAppConfig-only
    # sections) do not cause extra-field validation errors.
    known_fields = set(AppConfig.model_fields)
    yaml_data = {k: v for k, v in yaml_data.items() if k in known_fields}
    return AppConfig(**yaml_data)


def reset_config() -> None:
    """Clear cached config (for testing)."""
    get_config.cache_clear()
