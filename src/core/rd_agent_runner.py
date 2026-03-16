"""Graceful stub for RD-Agent factor and model discovery.

Real implementation is deferred to Task 06.  When rdagent is not installed,
all methods degrade safely and return empty results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.logger import get_logger
from src.utils.schemas import FactorDefinition

logger = get_logger(__name__)


class RDAgentRunner:
    """Stub runner for RD-Agent factor/model discovery.

    Attributes:
        config: Full application config including RD-Agent settings.
        _rd_agent_available: ``True`` when the ``rdagent`` package is importable.
    """

    def __init__(self, config: FullAppConfig | None = None) -> None:
        self.config = config or get_full_config()
        self._rd_agent_available = False
        try:
            import rdagent  # noqa: F401, PLC0415

            self._rd_agent_available = True
            logger.info("RD-Agent package detected.")
        except ImportError:
            logger.info("RD-Agent not installed. Using stub mode.")

    # ------------------------------------------------------------------
    # Factor discovery
    # ------------------------------------------------------------------

    def run_factor_search(self, n_iterations: int | None = None) -> list[FactorDefinition]:
        """Discover new alpha factors via RD-Agent.

        Args:
            n_iterations: Number of search iterations. Defaults to
                          ``config.rd_agent.factor_iterations``.

        Returns:
            List of discovered :class:`~src.utils.schemas.FactorDefinition` objects.
            Returns an empty list when RD-Agent is not installed.
        """
        if not self._rd_agent_available:
            logger.warning("RD-Agent not available. Skipping factor search.")
            return []

        # Real implementation deferred to Task 06.
        iterations = n_iterations or self.config.rd_agent.factor_iterations
        logger.info("RD-Agent factor search: {} iterations (stub).", iterations)
        return []

    # ------------------------------------------------------------------
    # Model optimisation
    # ------------------------------------------------------------------

    def run_model_search(self, n_iterations: int | None = None) -> dict[str, Any]:
        """Optimise model hyperparameters via RD-Agent.

        Args:
            n_iterations: Number of optimisation rounds. Defaults to
                          ``config.rd_agent.model_iterations``.

        Returns:
            Dict of optimised model parameters.
            Returns an empty dict when RD-Agent is not installed.
        """
        if not self._rd_agent_available:
            logger.warning("RD-Agent not available. Skipping model search.")
            return {}

        # Real implementation deferred to Task 06.
        iterations = n_iterations or self.config.rd_agent.model_iterations
        logger.info("RD-Agent model search: {} iterations (stub).", iterations)
        return {}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_factor_library(self, factors: list[FactorDefinition]) -> None:
        """Persist discovered factors to ``outputs/factor_library.json``.

        Args:
            factors: List of :class:`~src.utils.schemas.FactorDefinition` to save.
        """
        output_path = Path(self.config.rd_agent.factor_library_dir) / "factor_library.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [f.model_dump() for f in factors]
        with output_path.open("w") as fh:
            json.dump(data, fh, indent=2)

        logger.info("Factor library saved to {}. ({} factors)", output_path, len(factors))

    def save_model_config(self, config: dict[str, Any]) -> None:
        """Persist optimised model config to ``outputs/best_model_config.yaml``.

        Args:
            config: Dict of model hyperparameters to save.
        """
        output_path = Path(self.config.rd_agent.best_model_config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as fh:
            yaml.safe_dump(config, fh, default_flow_style=False)

        logger.info("Model config saved to {}.", output_path)
