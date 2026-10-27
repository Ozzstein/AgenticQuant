"""Application configuration using Pydantic BaseSettings."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, EnvSettingsSource

from src.domain.common import NonEmptyString, UtcTimestamp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class QlibConfig(BaseModel):
    provider_uri: str = str(PROJECT_ROOT / "data" / "qlib_data")
    region: str = "us"
    dataset: str = "Alpha158"
    train_start: str = "2010-01-01"
    train_end: str = "2021-12-31"
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

    @model_validator(mode="after")
    def chronological_segments(self) -> QlibConfig:
        """Reject overlapping or inverted inclusive research segments."""
        boundaries = [date.fromisoformat(getattr(self, name)) for name in (
            "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"
        )]
        train_start, train_end, valid_start, valid_end, test_start, test_end = boundaries
        if not train_start <= train_end < valid_start <= valid_end < test_start <= test_end:
            raise ValueError("Qlib train, validation and test segments must be ordered and disjoint")
        return self


class ModelPricing(BaseModel):
    """Explicit versioned model rates; absence is not free inference."""

    model_config = {"extra": "forbid", "frozen": True, "allow_inf_nan": False}
    provider: NonEmptyString
    model: NonEmptyString
    price_version: NonEmptyString
    source_uri: NonEmptyString
    effective_at: UtcTimestamp
    input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(ge=0)


class AgentConfig(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    deep_think_model: str = "claude-opus-4-20250514"
    quick_think_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.2
    max_tokens: int = 4096
    analysis_timeout_seconds: float = Field(default=180.0, gt=0)
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
    max_cost_per_analysis: float = Field(default=0.50, gt=0)
    pricing: list[ModelPricing] = Field(default_factory=list)
    budget_store_path: str = ""
    max_cost_per_day_usd: Decimal = Field(default=Decimal("10"), gt=0)
    max_requests_per_analysis: int = Field(default=30, gt=0)
    max_requests_per_day: int = Field(default=600, gt=0)
    max_tokens_per_analysis: int = Field(default=100_000, gt=0)
    max_tokens_per_day: int = Field(default=1_000_000, gt=0)
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
    walk_forward_months: int = Field(default=6, ge=1)
    embargo_days: int = Field(default=5, ge=0)
    transaction_cost_bps: float = 10.0
    retrain_frequency: str = "monthly"
    ensemble_models: list[str] = Field(
        default_factory=lambda: ["LightGBM", "XGBoost", "Linear"]
    )
    ensemble_method: Literal["ic_weighted", "equal", "custom"] = "ic_weighted"
    ensemble_val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)


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


class AlpacaConfig(BaseModel):
    """Alpaca broker configuration."""

    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True         # True = paper-api.alpaca.markets; False = live (gated)
    base_url: str = ""         # Override URL (empty = auto from paper flag)
    max_retries: int = 3
    timeout_seconds: int = 30


class CcxtConfig(BaseModel):
    """CCXT crypto exchange broker configuration.

    Controls live crypto order execution via any CCXT-compatible exchange.
    paper=True uses the exchange's sandbox/testnet API when available.
    Live mode requires AIQUANT_CCXT_LIVE=true env var (double opt-in).
    """

    enabled: bool = False
    exchange: str = "binance"        # Any valid ccxt exchange id (e.g. binance, kraken, coinbase)
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True               # True = sandbox/testnet; False = live (gated)
    quote_currency: str = "USDT"     # Base quote currency for symbol formatting (BTC → BTC/USDT)
    max_retries: int = 3
    timeout_seconds: int = 30


class MemoryConfig(BaseModel):
    """Agent memory persistence configuration."""

    enabled: bool = True
    max_history: int = 10
    store_path: Path = PROJECT_ROOT / "data" / "agent_memory" / "analysis_log.jsonl"


class IntradayConfig(BaseModel):
    """Configuration for sub-daily intraday pipeline runs."""

    enabled: bool = False
    interval_minutes: int = 240
    crypto_only: bool = True
    universe: list[str] = Field(default_factory=list)
    top_n: int = 5


class OperationsConfig(BaseModel):
    store_path: str = ''
    evidence_store_path: str = ''
    research_registry_path: str = ''
    attribution_path: str = ''
    poll_seconds: int = Field(default=60,ge=1,le=3600)
    max_cycle_age_seconds: int = Field(default=3600,ge=1,le=86400)
    # Only exact declared baselines can bypass learned-candidate activation.
    approved_baseline_hashes: list[str] = Field(default_factory=list)
    approved_model_hashes: list[str] = Field(default_factory=list)

    @field_validator('approved_baseline_hashes','approved_model_hashes')
    @classmethod
    def exact_artifact_identities(cls, values):
        if len(values) != len(set(values)) or any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in values):
            raise ValueError('approval lists require unique exact SHA256 artifact hashes')
        return values


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = {"env_prefix": "AIQUANT_", "env_nested_delimiter": "__"}

    project_name: str = "QuantAgentLab"
    log_level: str = "INFO"
    output_dir: str = str(PROJECT_ROOT / "outputs")
    log_dir: str = str(PROJECT_ROOT / "logs")
    paper_ledger_path: str = ""
    paper_account_id: str = Field(default="paper:default", min_length=1)
    paper_base_currency: str = Field(default="USD", pattern=r"^[A-Z]{3,10}$")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    finnhub_api_key: str = ""
    coingecko_api_key: str = ""
    fmp_api_key: str = ""

    qlib: QlibConfig = Field(default_factory=QlibConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    lean: LeanConfig = Field(default_factory=LeanConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    macro_regime: MacroRegimeConfig = Field(default_factory=MacroRegimeConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    ccxt: CcxtConfig = Field(default_factory=CcxtConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    intraday: IntradayConfig = Field(default_factory=IntradayConfig)
    operations: OperationsConfig = Field(default_factory=OperationsConfig)

    @model_validator(mode="after")
    def persistent_paper_account(self) -> AppConfig:
        """Derive the ledger location after resolving the output directory."""
        if not self.paper_ledger_path:
            self.paper_ledger_path = str(Path(self.output_dir) / "paper-ledger.sqlite3")
        if not self.agents.budget_store_path:
            self.agents.budget_store_path = str(Path(self.output_dir) / "agent-budgets.sqlite3")
        for name,filename in (('store_path','operations.sqlite3'),('evidence_store_path','evidence.sqlite3'),
                              ('research_registry_path','research.sqlite3'),('attribution_path','strategy-performance.sqlite3')):
            if not getattr(self.operations,name):
                setattr(self.operations,name,str(Path(self.output_dir)/filename))
        if self.paper_ledger_path == ":memory:" or not self.paper_ledger_path.strip():
            raise ValueError("paper_ledger_path must identify a persistent database")
        if not self.paper_account_id.strip():
            raise ValueError("paper_account_id must not be blank")
        return self

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
            result = yaml.safe_load(f) or {}
        if not isinstance(result, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")
        return result
    return {}


def _load_dotenv(path: Path) -> None:
    """Load .env file into environment if it exists."""
    if path.exists():
        from dotenv import load_dotenv
        load_dotenv(path)


SettingsType = TypeVar("SettingsType", bound=AppConfig)


def _merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_settings(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(settings_type: type[SettingsType]) -> SettingsType:
    """Resolve defaults < YAML < dotenv < process environment for either schema.

    Passing YAML as BaseSettings constructor arguments would give it precedence
    over environment variables. Resolve environment overrides explicitly first,
    recursively preserving unaffected YAML fields in nested configuration.
    """
    config_dir = PROJECT_ROOT / "config"
    _load_dotenv(config_dir / ".env")
    yaml_data = _load_yaml_config(config_dir / "settings.yaml")
    known_fields = set(settings_type.model_fields)
    yaml_data = {k: v for k, v in yaml_data.items() if k in known_fields}
    overrides = EnvSettingsSource(settings_type)()
    return settings_type(**_merge_settings(yaml_data, overrides))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get singleton application configuration with environment overrides."""
    return load_settings(AppConfig)


def reset_config() -> None:
    """Clear cached config (for testing)."""
    get_config.cache_clear()
    # Avoid importing the extended loader while it is importing this module.
    import sys

    extended = sys.modules.get("src.utils.config_loader")
    if extended is not None and hasattr(extended, "get_full_config"):
        extended.get_full_config.cache_clear()
