"""Canonical CLI/service/scheduler integration with actual durable stores."""

import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.run_operations import app
from src.operations.application import (
    ScheduledCycle,
    ScheduledResearch,
    execute_scheduled,
    paper_cycle_service,
)
from src.operations.scheduler import ScheduleSpec, ScheduleSupervisor
from src.operations.store import OperationsStore, canonical, digest
from src.utils.config import AppConfig

pytest_plugins = ["tests.test_operating_cycle"]


def test_configured_cli_aliases_preserve_source_block_and_no_economic_admission(
    runtime, monkeypatch, tmp_path
):
    cfg = runtime.config
    cfg.paper_ledger_path = str(runtime.ledger.path)
    cfg.paper_account_id = runtime.plan().inputs.account.account_id
    cfg.operations.store_path = str(runtime.ops.path)
    cfg.operations.evidence_store_path = str(runtime.archive.store.path)
    cfg.operations.research_registry_path = str(runtime.service.registry.path)
    cfg.operations.attribution_path = str(runtime.service.attribution.path)
    monkeypatch.setattr("src.utils.config.get_config", lru_cache()(lambda: cfg))
    clock = [runtime.plan().inputs.context.as_of]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr("src.operations.cycle.datetime", Clock)
    from src.utils.cli import app as canonical_cli

    calls = len(runtime.calls)
    for number, (cli, args) in enumerate(
        ((app, ["cycle"]), (canonical_cli, ["pipeline", "run"]), (canonical_cli, ["trade", "run"]))
    ):
        plan = runtime.plan(run_id=f"configured-source-block-{number}")
        clock[0] = plan.inputs.context.as_of
        path = tmp_path / f"blocked-{number}.json"
        path.write_text(plan.model_dump_json())
        response = CliRunner().invoke(cli, [*args, "--input", str(path)])
        assert response.exit_code == 2, response.output
        result = json.loads(response.output)
        assert result["status"] == "blocked" and "income policy" in result["reason"]
        assert (
            runtime.ops.status(plan.inputs.context.run_id)["stages"]["account"]["state"]
            == "pending"
        )
    assert len(runtime.calls) == calls


def test_configured_factory_and_all_cli_aliases_replay_actual_completed_cycle(
    runtime, monkeypatch, tmp_path
):
    cfg = runtime.config
    cfg.paper_ledger_path = str(runtime.ledger.path)
    cfg.paper_account_id = runtime.plan().inputs.account.account_id
    cfg.operations.store_path = str(runtime.ops.path)
    cfg.operations.evidence_store_path = str(runtime.archive.store.path)
    cfg.operations.research_registry_path = str(runtime.service.registry.path)
    cfg.operations.attribution_path = str(runtime.service.attribution.path)
    plan = runtime.plan()
    original = runtime.service.run(plan, now=plan.inputs.context.as_of)
    assert original.status == "completed", original.reason
    calls = len(runtime.calls)
    monkeypatch.setattr("src.utils.config.get_config", lru_cache()(lambda: cfg))
    path = tmp_path / "plan.json"
    path.write_text(plan.model_dump_json())
    from src.utils.cli import app as canonical_cli

    for cli, arguments in (
        (app, ["cycle"]),
        (canonical_cli, ["pipeline", "run"]),
        (canonical_cli, ["trade", "run"]),
    ):
        response = CliRunner().invoke(cli, [*arguments, "--input", str(path)])
        assert response.exit_code == 0, response.output
        assert json.loads(response.output) == original.model_dump(mode="json")
    job = ScheduledCycle(kind="paper_cycle", plan=plan)
    path.write_text(job.model_dump_json())
    assert execute_scheduled(
        path, plan.scheduled_for, cfg, lane="operating"
    ) == original.model_dump(mode="json")
    assert len(runtime.calls) == calls
    with pytest.raises(ValueError, match="window"):
        execute_scheduled(path, plan.scheduled_for + timedelta(seconds=1), cfg, lane="operating")
    with pytest.raises(ValueError, match="lane"):
        execute_scheduled(path, plan.scheduled_for, cfg, lane="research")
    response = CliRunner().invoke(canonical_cli, ["optimize", "--run-id", original.run_id])
    assert response.exit_code == 0, response.output
    assert "covariance_sampling_hash" in json.loads(response.output)["optimizer"]
    # Configured service uses one actual account and closes its owned connection.
    with paper_cycle_service(cfg) as service:
        assert service.ledger.spec(cfg.paper_account_id) == plan.inputs.account
        assert service.operations.status(original.run_id)["status"] == "completed"
    bad = plan.model_copy(
        update={
            "inputs": plan.inputs.model_copy(
                update={
                    "account": plan.inputs.account.model_copy(update={"account_id": "paper:other"}),
                    "context": plan.inputs.context.model_copy(update={"account_id": "paper:other"}),
                }
            )
        }
    )
    path.write_text(bad.model_dump_json())
    response = CliRunner().invoke(app, ["cycle", "--input", str(path)])
    assert response.exit_code == 1 and "configured persistent account" in response.output


def test_schedule_lanes_and_legacy_hash_survive_restart(tmp_path):
    from tests.test_operations_scheduler import T

    store = OperationsStore(tmp_path / "ops.db")
    operating = ScheduleSpec(
        schedule_id="ops",
        starts_at=T,
        kind="interval",
        interval_seconds=60,
        input_directory=str(tmp_path / "in"),
    )
    research = operating.model_copy(
        update={"schedule_id": "research", "execution_lane": "research"}
    )
    assert "execution_lane" not in operating.model_dump(mode="json")
    assert operating.content_hash() == digest(canonical(operating.model_dump(mode="json")))
    calls = []
    for spec in (operating, research):
        path = spec.input_path(T)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"job": spec.schedule_id}))
    supervisor = ScheduleSupervisor(
        store,
        lambda p, at: calls.append(json.loads(p.read_text())["job"]) or {"status": "completed"},
    )
    for spec in (operating, research):
        supervisor.register(spec)
    assert supervisor.tick(T, lane="operating")[0]["status"] == "completed"
    assert calls == ["ops"]
    restarted = ScheduleSupervisor(OperationsStore(store.path), supervisor.handler)
    assert restarted.tick(T, lane="operating") == []
    assert restarted.tick(T, lane="research")[0]["status"] == "completed"
    assert calls == ["ops", "research"]


def test_scheduled_research_rejects_holdout_before_outcome_access(tmp_path, monkeypatch):
    from src.research.registry import ResearchRegistry
    from tests.test_research_registry import T, admitted

    cfg = AppConfig(
        _env_file=None, output_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs")
    )
    registry = ResearchRegistry(cfg.operations.research_registry_path)
    _, trial, _, _ = admitted(registry)
    job = ScheduledResearch(kind="research_trial", scheduled_for=T, trial_id=trial.trial_id)
    path = tmp_path / "job.json"
    path.write_text(job.model_dump_json())
    monkeypatch.setattr(
        ResearchRegistry, "development_evaluation", lambda *_: pytest.fail("holdout outcome read")
    )
    with pytest.raises(ValueError, match="development"):
        execute_scheduled(path, T, cfg, lane="research")
    assert not (Path(cfg.output_dir) / "research-runs").exists()


def test_invalid_cycle_and_retired_argument_paths_fail_visibly(tmp_path, monkeypatch):
    from scripts.run_pipeline import _run_pipeline_once
    from src.utils.cli import app as canonical_cli

    cfg = AppConfig(
        _env_file=None, output_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs")
    )
    monkeypatch.setattr("src.utils.config.get_config", lru_cache()(lambda: cfg))
    path = tmp_path / "bad.json"
    path.write_text("{}")
    assert CliRunner().invoke(app, ["cycle", "--input", str(path)]).exit_code == 1
    assert CliRunner().invoke(canonical_cli, ["trade", "run"]).exit_code == 2
    with pytest.raises(ValueError, match="explicit PaperCyclePlan"):
        _run_pipeline_once("backtest", config=cfg)
    assert not Path(cfg.paper_ledger_path).exists()
