"""Real schedule/store/refresh execution; only HTTP boundary uses fixtures."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.data.evidence.store import EvidenceStore
from src.operations import refresh
from src.operations.application import WORK
from src.operations.refresh import FetchTask, RefreshPlan, RefreshService
from src.operations.rolling import RefreshClockBinding, RollingRefreshTemplate
from src.operations.scheduler import ScheduleSpec, ScheduleSupervisor
from src.operations.store import OperationsStore, canonical
from tests.test_operations_refresh import context, real_adapter_fetcher, task


def template(*, bindings=(), tasks=None):
    ctx = context()
    return RollingRefreshTemplate(
        plan=RefreshPlan(context=ctx, scheduled_for=ctx.as_of, tasks=tasks or (task(),)),
        bindings=bindings,
    )


def schedule(value, at):
    return ScheduleSpec(
        schedule_id="rolling",
        starts_at=at,
        kind="interval",
        interval_seconds=60,
        max_lag_seconds=180,
        refresh_template=value,
    )


def test_actual_rolling_adapter_two_windows_and_restart_without_mutable_input(
    tmp_path, monkeypatch
):
    from src.operations import application
    from src.operations.cycle import cycle_config_hash
    from src.utils.config import AppConfig

    calls, client = real_adapter_fetcher(monkeypatch)
    ops = OperationsStore(tmp_path / "ops.sqlite")
    evidence = EvidenceStore(tmp_path / "evidence.sqlite")
    config = AppConfig()
    config.operations.store_path = str(ops.path)
    config.operations.evidence_store_path = str(evidence.path)
    monkeypatch.setattr(
        application,
        "RefreshService",
        lambda ops, evidence: RefreshService(ops, evidence, fetcher=refresh._fetch),
    )
    plans = []

    def handler(path, at):
        job = WORK.validate_json(path.read_bytes())
        assert job.kind == "data_refresh" and job.plan.scheduled_for == at
        plans.append(job.plan)
        return application.execute_scheduled(path, at, config, lane="operating")

    initial = datetime.now(timezone.utc) - timedelta(seconds=61)
    value = template()
    value = value.model_copy(
        update={
            "plan": value.plan.model_copy(
                update={
                    "context": value.plan.context.model_copy(
                        update={"config_hash": cycle_config_hash(config)}
                    )
                }
            )
        }
    )
    spec = schedule(value, initial)
    scheduler = ScheduleSupervisor(ops, handler)
    scheduler.register(spec)
    try:
        for _ in range(2):
            invoked = datetime.now(timezone.utc)
            assert scheduler.tick(invoked)[0]["status"] == "completed"
            plan = plans[-1]
            assert plan.context.as_of == invoked
            result = ops.artifact(
                ops.status(plan.context.run_id)["stages"]["report"]["output_hash"]
            )
            source = result["fetches"]["financials"]["sources"][0]
            record = evidence.get(source["evidence_id"]).record
            assert record.requested_as_of == invoked < record.available_at
            with pytest.raises(ValueError, match="unavailable"):
                evidence.get(record.evidence_id, as_of=invoked)
        assert plans[0].context.run_id != plans[1].context.run_id
        assert len(calls) == 2
        restarted = ScheduleSupervisor(OperationsStore(ops.path), handler)
        restarted.register(spec)
        assert restarted.tick(datetime.now(timezone.utc)) == []
        assert len(calls) == 2
    finally:
        client.close()


def test_failed_post_refresh_tick_reuses_exact_plan_and_no_provider_retry(tmp_path, monkeypatch):
    calls, client = real_adapter_fetcher(monkeypatch)
    ops, evidence = (
        OperationsStore(tmp_path / "ops.sqlite"),
        EvidenceStore(tmp_path / "data.sqlite"),
    )
    plans = []

    def handler(path, at):
        job = WORK.validate_json(path.read_bytes())
        plans.append(job.plan)
        result = RefreshService(ops, evidence, fetcher=refresh._fetch).run(job.plan)
        if len(plans) == 1:
            raise RuntimeError("lost scheduler completion after durable refresh")
        return result

    now = datetime.now(timezone.utc)
    spec = schedule(template(), now)
    supervisor = ScheduleSupervisor(ops, handler)
    supervisor.register(spec)
    try:
        assert supervisor.tick(now)[0]["status"] == "failed"
        assert supervisor.tick(now + timedelta(seconds=1))[0]["status"] == "recovery_required"
        with ops.connection() as db:
            frozen = db.execute("SELECT input_hash FROM schedule_ticks").fetchone()[0]
        supervisor.resume(spec.schedule_id, expected_input_hash=frozen)
        assert supervisor.tick(now + timedelta(seconds=10))[0]["status"] == "completed"
        assert plans[0] == plans[1] and len(calls) == 1
    finally:
        client.close()


def test_temporal_bindings_keep_schedule_and_invocation_clocks_separate():
    task = FetchTask(
        task_id="macro",
        provider="fred",
        operation="observations",
        arguments_json=canonical(
            {"series_id": "DGS10", "start": "2000-01-01", "end": "2000-01-02"}
        ).decode(),
    )
    value = template(
        tasks=(task,),
        bindings=(
            RefreshClockBinding(
                task_id="macro", argument="start", anchor="scheduled_for", offset_seconds=-86400
            ),
            RefreshClockBinding(task_id="macro", argument="end", anchor="invoked_at"),
        ),
    )
    at = datetime(2026, 3, 8, 23, 59, tzinfo=timezone.utc)
    invoked = at + timedelta(minutes=2)
    plan = value.materialize("clock", scheduled_for=at, invoked_at=invoked)
    assert json.loads(plan.tasks[0].arguments_json) == {
        "series_id": "DGS10",
        "start": "2026-03-07",
        "end": "2026-03-09",
    }
    assert plan.context.as_of == invoked and plan.scheduled_for == at
    assert value.plan.context.as_of != invoked  # immutable seed remains original
    assert (
        value.materialize(
            "clock", scheduled_for=at, invoked_at=invoked + timedelta(seconds=1)
        ).context.run_id
        == plan.context.run_id
    )
    assert (
        value.materialize("other", scheduled_for=at, invoked_at=invoked).context.run_id
        != plan.context.run_id
    )


@pytest.mark.parametrize("operation", ["funding", "open_interest"])
def test_native_derivative_windows_remain_exact_utc(operation):
    from src.domain.common import Instrument

    instrument = Instrument(
        instrument_id="binance:BTCUSD:perp",
        symbol="BTC/USD:BTC",
        asset_class="crypto",
        quote_currency="USD",
        venue="binance",
    )
    ctx = context()
    native = FetchTask(
        task_id="derivative",
        provider="ccxt",
        operation=operation,
        instrument=instrument,
        arguments_json=canonical(
            {"since": (ctx.as_of - timedelta(hours=1)).isoformat(), "end": ctx.as_of.isoformat()}
        ).decode(),
    )
    value = template(
        tasks=(native,),
        bindings=(
            RefreshClockBinding(
                task_id="derivative", argument="since", anchor="scheduled_for", offset_seconds=-3600
            ),
            RefreshClockBinding(task_id="derivative", argument="end", anchor="invoked_at"),
        ),
    )
    plan = value.materialize(
        "derivative", scheduled_for=ctx.as_of, invoked_at=ctx.as_of + timedelta(seconds=7)
    )
    assert json.loads(plan.tasks[0].arguments_json) == {
        "since": (ctx.as_of - timedelta(hours=1)).isoformat(),
        "end": (ctx.as_of + timedelta(seconds=7)).isoformat(),
    }


@pytest.mark.parametrize(
    "mutation", ["unknown", "duplicate", "snapshot", "unsupported", "missing_argument"]
)
def test_unapproved_clock_substitutions_fail(mutation):
    data = template().model_dump()
    binding = {"task_id": "financials", "argument": "start", "anchor": "invoked_at"}
    if mutation == "snapshot":
        data["plan"]["context"]["data_snapshot_ids"] = ("old-approval",)
    else:
        data["bindings"] = [binding]
        if mutation == "unknown":
            binding["task_id"] = "missing"
        elif mutation == "duplicate":
            data["bindings"] *= 2
        elif mutation == "missing_argument":
            data["plan"]["tasks"][0].update(provider="fred", operation="observations")
    with pytest.raises(ValueError):
        RollingRefreshTemplate.model_validate(data)


def test_legacy_schedule_hash_shape_and_closed_source_choice(tmp_path):
    from tests.test_operations_scheduler import interval

    legacy = interval(tmp_path)
    assert "refresh_template" not in legacy.model_dump(mode="json")
    assert ScheduleSpec.model_validate_json(legacy.model_dump_json()) == legacy
    values = legacy.model_dump()
    values["refresh_template"] = template().model_dump()
    with pytest.raises(ValueError, match="exactly one"):
        ScheduleSpec.model_validate(values)
    with pytest.raises(ValueError, match="operating lane"):
        ScheduleSpec.model_validate(
            {**schedule(template(), legacy.starts_at).model_dump(), "execution_lane": "research"}
        )


def test_expired_rolling_window_never_fetches_and_changes_cannot_reset_schedule(tmp_path):
    ops = OperationsStore(tmp_path / "ops.sqlite")
    spec = schedule(template(), datetime.now(timezone.utc))
    supervisor = ScheduleSupervisor(ops, lambda *_: pytest.fail("expired window cannot fetch"))
    supervisor.register(spec)
    assert supervisor.tick(spec.starts_at + timedelta(seconds=181))[0]["status"] == "missed"
    from src.operations.store import OperationError

    with pytest.raises(OperationError, match="immutable"):
        supervisor.register(spec.model_copy(update={"max_lag_seconds": 200}))


def test_future_naive_and_reversed_windows_reject():
    value = template()
    at = value.plan.context.as_of
    with pytest.raises(ValueError, match="future"):
        value.materialize("future", scheduled_for=at, invoked_at=at - timedelta(seconds=1))
    with pytest.raises(ValueError, match="timezone"):
        value.materialize("naive", scheduled_for=at, invoked_at=at.replace(tzinfo=None))
    with pytest.raises(ValueError):
        RefreshClockBinding(task_id="one", argument="start", anchor="invoked_at", offset_seconds=1)
    task = FetchTask(
        task_id="calendar",
        provider="operating",
        operation="alpaca_calendar",
        arguments_json=canonical({"start": "2000-01-01", "end": "2000-01-02"}).decode(),
    )
    reversed_window = template(
        tasks=(task,),
        bindings=(
            RefreshClockBinding(task_id="calendar", argument="start", anchor="invoked_at"),
            RefreshClockBinding(
                task_id="calendar", argument="end", anchor="invoked_at", offset_seconds=-86400
            ),
        ),
    )
    with pytest.raises(ValueError, match="reversed"):
        reversed_window.materialize("reversed", scheduled_for=at, invoked_at=at)
