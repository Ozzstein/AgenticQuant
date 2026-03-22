"""Extended configuration loader with RDAgentConfig and FullAppConfig."""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field

# Re-export everything from src.utils.config so both import paths work.
from src.utils.config import (  # noqa: F401
    PROJECT_ROOT,
    AgentConfig,
    AppConfig,
    CryptoConfig,
    LeanConfig,
    ModelConfig,
    MonitoringConfig,
    QlibConfig,
    RiskConfig,
    get_config,
    reset_config,
)


class RDAgentConfig(BaseModel):
    """Configuration for RD-Agent factor and model discovery."""

    enabled: bool = False
    factor_library_dir: str = "outputs/factor_library"
    best_model_config_path: str = "outputs/best_model_config.yaml"
    factor_iterations: int = 50
    min_ic: float = 0.02
    min_icir: float = 0.3
    min_backtest_sharpe: float = 0.5
    budget: int = 10
    model_iterations: int = 20
    schedule: str = "weekly"


class FullAppConfig(AppConfig):
    """AppConfig extended with RDAgentConfig."""

    rd_agent: RDAgentConfig = Field(default_factory=RDAgentConfig)


@lru_cache(maxsize=1)
def get_full_config() -> FullAppConfig:
    """Get singleton FullAppConfig (extends AppConfig with RDAgentConfig)."""
    import yaml

    base = get_config()
    data = base.model_dump()

    # Re-read YAML to pick up rd_agent block — AppConfig drops unknown keys
    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    if yaml_path.exists():
        with open(yaml_path) as _f:
            yaml_data: dict = yaml.safe_load(_f) or {}
        if "rd_agent" in yaml_data:
            data["rd_agent"] = yaml_data["rd_agent"]

    return FullAppConfig(**data)


def reset_full_config() -> None:
    """Clear cached full config (for testing)."""
    get_full_config.cache_clear()
