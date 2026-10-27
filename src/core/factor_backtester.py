"""Factor replay against frozen point-in-time inputs and actual ledger fills."""
from __future__ import annotations

from pathlib import Path

from src.core.backtester import ReplayResult
from src.core.validation import ValidationSuite
from src.core.walk_forward import WalkForwardRunner
from src.domain.market import FeatureSpec
from src.research.datasets import ResearchInputs
from src.utils.config_loader import FullAppConfig, get_full_config
from src.utils.schemas import BacktestValidationResult, CheckVerdict, FactorDefinition


class FactorBacktester:
    def __init__(self, config: FullAppConfig | None = None):
        self.config = config or get_full_config()

    def run(self, factor: FactorDefinition, *, inputs: ResearchInputs, ledger_path: str | Path,
            artifact_dir: str | Path, provenance: dict | None = None,
            model_name: str = "Linear") -> ReplayResult:
        """Fit fresh models per fold; account chronology comes from shared replay."""
        specs = (FeatureSpec(name=factor.name, expression=factor.expression),)
        data = inputs.materialize(specs)
        if data.features.empty:
            raise ValueError("no finite point-in-time eligible factor observations")
        return WalkForwardRunner(self.config).run(
            data.features, data.labels, model_name=model_name,
            label_end_times=data.label_available_at, feature_available_at=data.feature_available_at,
            feature_spec_hash=data.feature_spec_hash, data_snapshot_hash=data.snapshot_hash,
            ledger_path=ledger_path, artifact_dir=artifact_dir, universe_version=inputs.universe_version,
            walk_months=inputs.walk_months, embargo_days=inputs.embargo_days,
            validation_months=inputs.validation_months, initial_train_months=inputs.initial_train_months,
            provenance=provenance, **inputs.replay_kwargs(),
        )

    def validate(self, factor: FactorDefinition, *, inputs: ResearchInputs | None = None,
                 ledger_path: str | Path | None = None, artifact_dir: str | Path | None = None,
                 provenance: dict | None = None, cost_results=None, cost_artifacts=None, benchmark=None) -> BacktestValidationResult:
        if inputs is None or ledger_path is None or artifact_dir is None:
            raise ValueError("FactorBacktester requires ResearchInputs, ledger_path and artifact_dir; "
                             "ticker-only or label-only evaluation cannot establish performance")
        result = self.run(factor, inputs=inputs, ledger_path=ledger_path, artifact_dir=artifact_dir,
                          provenance=provenance)
        report = ValidationSuite().validate(result, ledger_path=ledger_path,
                                            cost_results=cost_results, cost_artifacts=cost_artifacts, benchmark=benchmark)
        sharpe = result.metrics.sharpe_ratio
        checks = {c.name: c.verdict == CheckVerdict.PASS for c in report.checks}
        passed = (report.overall == "RELIABLE" and sharpe is not None
                  and sharpe > self.config.rd_agent.min_backtest_sharpe and not inputs.synthetic)
        return BacktestValidationResult(
            factor_name=factor.name, passed=passed, sharpe=sharpe,
            max_drawdown=result.metrics.max_drawdown, checks=checks,
            reason="passed" if passed else "synthetic_data" if inputs.synthetic else
            "backtest_rejected" if report.overall != "RELIABLE" else "low_sharpe",
        )

    def validate_batch(self, factors: list[FactorDefinition], *, inputs: ResearchInputs,
                       output_dir: str | Path) -> list[BacktestValidationResult]:
        """Independent ledgers/models per candidate; approval needs registry evidence."""
        output = Path(output_dir)
        return [self.validate(factor, inputs=inputs,
                              ledger_path=output / str(i) / "ledger.sqlite3",
                              artifact_dir=output / str(i) / "models") for i, factor in enumerate(factors)]
