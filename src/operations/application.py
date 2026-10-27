"""Configured application entry points over the shared operating services."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from src.core.strategy_tracker import AttributedPerformanceStore
from src.data.evidence.store import EvidenceStore
from src.data.operating_plan import OperatingPreparationRecipe, prepare_operating_data
from src.domain.common import NonEmptyString, UtcTimestamp, VersionedContract
from src.domain.market import MarketSnapshot
from src.domain.portfolio import UniverseEligibility
from src.execution.ledger import PortfolioLedger
from src.operations.contracts import PaperCyclePlan, PaperCycleTemplate
from src.operations.operating_fetch import load_operating_sources
from src.operations.refresh import RefreshPlan, RefreshService
from src.operations.scheduler import ScheduleSupervisor
from src.operations.store import OperationsStore
from src.research.evidence import SourceBinding
from src.research.registry import MissingDevelopmentEvaluation, ResearchRegistry


@contextmanager
def paper_cycle_service(config):
    """One persistent account and store set for CLI and scheduled execution."""
    from src.operations.cycle import PaperCycleService

    operations = OperationsStore(config.operations.store_path)
    evidence = EvidenceStore(config.operations.evidence_store_path)
    registry = ResearchRegistry(config.operations.research_registry_path)
    ledger = PortfolioLedger(config.paper_ledger_path)
    try:
        yield PaperCycleService(
            config,
            operations_store=operations,
            evidence_store=evidence,
            registry=registry,
            ledger=ledger,
            attribution_store=AttributedPerformanceStore(config.operations.attribution_path),
            artifact_dir=Path(config.output_dir) / "operating-artifacts",
            resolver=registry.resolve,
        )
    finally:
        ledger.close()


def execute_cycle(plan, config, *, dispatch_only=False):
    from src.operations.contracts import PaperCyclePlan

    plan = PaperCyclePlan.model_validate(plan)
    if plan.inputs.account.account_id != config.paper_account_id:
        raise ValueError("cycle account differs from configured persistent account")
    with paper_cycle_service(config) as service:
        if dispatch_only:
            return service.dispatch_pending(plan)
        return service.run(plan).model_dump(mode="json")


def prepare_cycle(
    template,
    macro_dataset,
    config,
    *,
    refresh_run_id,
    run_id,
    scheduled_for,
    contextual_sources=(),
    income_authorities=(),
):
    from src.operations.contracts import PaperCycleTemplate
    from src.research.regime_experiments import FrozenEvidence, MacroDatasetSpec

    template = PaperCycleTemplate.model_validate(template)
    if template.account.account_id != config.paper_account_id:
        raise ValueError("template account differs from configured persistent account")
    dataset = MacroDatasetSpec.model_validate(macro_dataset)
    with paper_cycle_service(config) as service:
        sources = load_operating_sources(service.operations, service.evidence, refresh_run_id)
        return service.prepare(
            template,
            refresh_run_id=refresh_run_id,
            run_id=run_id,
            scheduled_for=scheduled_for,
            economic_sources=sources,
            macro_dataset=dataset,
            contextual_sources=tuple(
                FrozenEvidence.model_validate(value) for value in contextual_sources
            ),
            income_authorities=tuple(
                SourceBinding.model_validate(value) for value in income_authorities
            ),
        )


class CyclePreparationRequest(VersionedContract):
    template: PaperCycleTemplate
    recipe: OperatingPreparationRecipe
    refresh_run_id: NonEmptyString
    run_id: NonEmptyString
    scheduled_for: UtcTimestamp
    breadth_snapshot: MarketSnapshot
    universes: tuple[UniverseEligibility, ...] = Field(min_length=1)
    universe_sources: tuple[SourceBinding, ...] = Field(min_length=1)
    income_authorities: tuple[SourceBinding, ...] = ()


def build_cycle(request, config):
    """Derive macro/contextual IDs from exact completed task receipts, then freeze H."""
    request = CyclePreparationRequest.model_validate(request)
    if request.template.account.account_id != config.paper_account_id:
        raise ValueError("template account differs from configured persistent account")
    with paper_cycle_service(config) as service:
        prepared = prepare_operating_data(
            service.operations,
            service.evidence,
            request.refresh_run_id,
            request.recipe,
            breadth_snapshot=request.breadth_snapshot,
            universes=request.universes,
            universe_sources=request.universe_sources,
            training_prefix=request.template.macro_recipe.training_dataset,
        )
        plan = service.prepare(
            request.template,
            refresh_run_id=request.refresh_run_id,
            run_id=request.run_id,
            scheduled_for=request.scheduled_for,
            economic_sources=prepared.economic_sources,
            macro_dataset=prepared.macro_dataset,
            contextual_sources=prepared.contextual_sources,
            income_authorities=request.income_authorities,
        )
        proof = service.registry.archive_contract(prepared)
        return plan.model_copy(
            update={"provenance": (*plan.provenance, ("operating_preparation_id", proof))}
        )


class ScheduledCycle(VersionedContract):
    kind: Literal["paper_cycle"]
    plan: PaperCyclePlan


class ScheduledDispatch(VersionedContract):
    kind: Literal["dispatch_pending"]
    scheduled_for: UtcTimestamp
    plan: PaperCyclePlan


class ScheduledRefresh(VersionedContract):
    kind: Literal["data_refresh"]
    plan: RefreshPlan


class ScheduledResearch(VersionedContract):
    kind: Literal["research_trial"]
    scheduled_for: UtcTimestamp
    trial_id: NonEmptyString
    simulations: int = Field(default=1000, ge=1000, le=10000)


ScheduledWork = Annotated[
    ScheduledCycle | ScheduledDispatch | ScheduledRefresh | ScheduledResearch,
    Field(discriminator="kind"),
]
WORK = TypeAdapter(ScheduledWork)


def execute_scheduled(path: Path, scheduled_for: datetime, config, *, lane, authorities=None):
    """Dispatch a frozen, closed job type against configured authoritative stores.

    Research has a separate worker lane so a long experiment never blocks the
    operating scheduler. It can execute registered development trials, but has
    no automatic holdout admission or promotion authority.
    """
    job = WORK.validate_json(path.read_bytes())
    expected_lane = "research" if job.kind == "research_trial" else "operating"
    if lane != expected_lane:
        raise ValueError("scheduled job belongs to a different execution lane")
    actual_schedule = (
        job.plan.scheduled_for if job.kind in ("paper_cycle", "data_refresh") else job.scheduled_for
    )
    if actual_schedule != scheduled_for:
        raise ValueError("scheduled input window differs from durable schedule claim")
    if job.kind == "paper_cycle":
        return execute_cycle(job.plan, config)
    if job.kind == "dispatch_pending":
        receipt = execute_cycle(job.plan, config, dispatch_only=True)
        return {"status": "completed", "kind": job.kind, "receipt": receipt}
    if job.kind == "data_refresh":
        from src.operations.cycle import cycle_config_hash

        if job.plan.context.config_hash != cycle_config_hash(config):
            raise ValueError("refresh configuration differs from configured scheduler")
        return RefreshService(
            OperationsStore(config.operations.store_path),
            EvidenceStore(config.operations.evidence_store_path),
        ).run(job.plan)
    from src.research.registry import content_hash
    from src.research.runner import ResearchRunner

    registry = ResearchRegistry(config.operations.research_registry_path)
    trial = registry.trial(job.trial_id)
    if trial.purpose != "development" or trial.registered_at > scheduled_for:
        raise ValueError("scheduled research requires a previously registered development trial")
    # Only a fully persisted independently verified outcome can be replayed.
    # An incomplete execution keeps its immutable output admission and requires
    # explicit recovery; changing the schedule cannot restart it under a new path.
    try:
        evaluation = registry.development_evaluation(job.trial_id)
    except MissingDevelopmentEvaluation:
        output = Path(config.output_dir) / "research-runs" / content_hash(job.trial_id.encode())
        evaluation = ResearchRunner(
            registry,
            EvidenceStore(config.operations.evidence_store_path),
            config,
            authorities=authorities,
        ).run(job.trial_id, output_dir=output, simulations=job.simulations)
    return {
        "status": "completed",
        "kind": job.kind,
        "trial_id": job.trial_id,
        "evaluation": evaluation.model_dump(mode="json"),
    }


def schedule_supervisor(config, *, lane="operating", authorities=None):
    if lane not in ("operating", "research"):
        raise ValueError("unknown scheduler lane")
    return ScheduleSupervisor(
        OperationsStore(config.operations.store_path),
        lambda path, at: execute_scheduled(path, at, config, lane=lane, authorities=authorities),
    )
