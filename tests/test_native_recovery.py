"""Actual SQLite/bundle recovery; Docker and authenticated HTTP are protocol fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from src.research import native_mounts
from src.research.native import NativeSupervisor
from src.research.native_recovery import (
    NativeRecoveryService,
    RecoveryError,
    bind_container,
    record_launch,
)
from src.research.native_runtime.overlay import UPSTREAM_REVISION
from src.research.registry import RegistryError, ResearchRegistry
from tests.native_budget_fixtures import pending, receipt
from tests.test_native_discovery_exposure import mounted_request


@pytest.fixture(autouse=True)
def isolate_authority(tmp_path, monkeypatch):
    monkeypatch.setattr(
        native_mounts, "_authority_path", lambda: tmp_path / "host-control/mounts.db"
    )


class DockerBoundary:
    def __init__(self, req, launch, now):
        self.calls = []
        self.container_id = "a" * 64
        self.image = {
            "Id": launch.image_id,
            "RepoDigests": [req.image],
            "Config": {
                "Env": ["QUANTAGENT_NATIVE_SANDBOX=1"],
                "Entrypoint": ["python", "-m", "src.research.native_runtime.entrypoint"],
                "Cmd": None,
                "Labels": {},
                "WorkingDir": "/work",
            },
        }
        self.network = {"Id": launch.network_id, "Name": req.network, "Internal": True}
        self.container = {
            "Id": self.container_id,
            "Name": "/quantagent-rd-" + req.run_id,
            "Created": now.isoformat(),
            "Image": launch.image_id,
            "Config": {
                **copy.deepcopy(self.image["Config"]),
                "Image": req.image,
                "User": launch.user,
                "Labels": {
                    "quantagent.native-run": req.run_id,
                    "quantagent.request": req.content_hash(),
                },
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "Privileged": False,
                "CapDrop": ["ALL"],
                "CapAdd": None,
                "SecurityOpt": ["no-new-privileges"],
                "AutoRemove": True,
                "RestartPolicy": {"Name": "no"},
                "PidsLimit": 256,
                "Memory": req.memory_mb * 1024**2,
                "NanoCpus": req.cpus * 1_000_000_000,
                "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=512m"},
                "PortBindings": {},
                "NetworkMode": req.network,
                "LogConfig": {"Type": "none"},
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": m.source,
                    "Destination": m.destination,
                    "RW": not m.read_only,
                    "Propagation": "rprivate",
                }
                for m in launch.mounts
            ],
            "NetworkSettings": {"Networks": {req.network: {"NetworkID": launch.network_id}}},
            "State": {
                "Status": "created",
                "Running": False,
                "ExitCode": 0,
                "OOMKilled": False,
                "StartedAt": "0001-01-01T00:00:00Z",
                "FinishedAt": "0001-01-01T00:00:00Z",
                "Error": "",
            },
            "RestartCount": 0,
        }
        self.before_stop = None
        self.stop_error = False
        self.error = False
        self.inspect_count = 0
        self.on_inspect = None

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if self.error:
            raise RuntimeError("offline daemon unavailable")
        if args[:2] == ("container", "ls"):
            return self.container["Id"] + "\n" if self.container else ""
        if args[:2] == ("container", "inspect"):
            self.inspect_count += 1
            if self.on_inspect:
                self.on_inspect(self.inspect_count)
            return json.dumps([self.container])
        if args[:2] == ("image", "inspect"):
            return json.dumps([self.image])
        if args[:2] == ("network", "inspect"):
            return json.dumps([self.network])
        if args[:2] in (("container", "stop"), ("container", "rm")):
            if self.before_stop:
                self.before_stop()
            assert args[-1] == self.container_id
            self.container = None
            if self.stop_error:
                raise RuntimeError("response lost after stop")
            return self.container_id + "\n"
        pytest.fail("Unexpected Docker action: " + repr(args))




@pytest.fixture
def job(tmp_path):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "research.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    exposure = supervisor.admit_discovery(req, registry)
    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    token = tmp_path / "private/bearer"
    token.parent.mkdir()
    token.write_text("scoped-test-bearer-never-persist-this-value")
    token.chmod(0o600)
    budget = receipt(req, now)
    preflight = {
        "upstream_revision": UPSTREAM_REVISION,
        "image_id": "sha256:" + "b" * 64,
        "network_id": "c" * 64,
        "scenario_hash": req.scenario.content_hash(),
        "bundle_hash": req.scenario.bundle_hash,
        "data_snapshot_hash": hashlib.sha256(
            json.dumps(
                json.loads(registry.blob(exposure.bundle_hash))["snapshot"], sort_keys=True
            ).encode()
        ).hexdigest(),
        "gateway_receipt": budget,
    }
    work = supervisor.root / req.run_id
    work.mkdir()
    for name, value in {
        "request.json": {**req.model_dump(mode="json"), "native_args": req.native_args()},
        "scenario.json": {**req.scenario.model_dump(mode="json"), "provider_uri": "/data/qlib"},
        "admission.json": {
            **preflight,
            "discovery_exposure": exposure.model_dump(mode="json"),
            "discovery_exposure_hash": exposure.content_hash(),
        },
    }.items():
        (work / name).write_text(json.dumps(value, indent=2))
    launch = record_launch(
        supervisor,
        req,
        registry,
        gateway_token_path=token,
        preflight=preflight,
        exposure=exposure,
        clock=lambda: now,
    )
    docker = DockerBoundary(req, launch, now + timedelta(seconds=1))
    binding = bind_container(
        supervisor,
        registry,
        req.run_id,
        docker.container_id,
        docker=docker,
        clock=lambda: now + timedelta(seconds=2),
    )

    def clock():
        return now + timedelta(seconds=20)

    def gateway_receipt(*args):
        value = copy.deepcopy(budget)
        value["accounting"].update(as_of=clock().isoformat(), day_id=clock().date().isoformat())
        return value

    service = NativeRecoveryService(
        supervisor,
        registry,
        clock=clock,
        docker=docker,
        gateway_receipt=gateway_receipt,
    )
    return SimpleNamespace(
        req=req,
        registry=registry,
        supervisor=supervisor,
        launch=launch,
        binding=binding,
        docker=docker,
        service=service,
        work=work,
        now=now,
        budget=budget,
        clock=clock,
        token=token,
    )


def finish(job, code=0):
    job.docker.container["State"].update(
        Status="exited",
        Running=False,
        ExitCode=code,
        StartedAt=(job.now + timedelta(seconds=3)).isoformat(),
        FinishedAt=(job.now + timedelta(seconds=10)).isoformat(),
    )
    (job.work / "watchdog.json").write_text(
        json.dumps(
            {
                "status": "discovery_complete",
                "exit_code": 0,
                "elapsed_seconds": 7,
                "output_bytes": 200,
                "max_seconds": job.req.max_seconds,
                "max_output_bytes": job.req.max_output_bytes,
            }
        )
    )
    (job.work / "research.txt").write_text(
        "Synthetic discovered candidate; unvalidated and quarantined."
    )


def test_actual_registry_restart_recovers_only_original_clean_exit(job):
    finish(job)
    registry = ResearchRegistry(job.registry.path)
    service = NativeRecoveryService(
        job.supervisor,
        registry,
        clock=job.clock,
        docker=job.docker,
        gateway_receipt=job.service.gateway_receipt,
    )
    event = service.reconcile(job.req.run_id)
    assert event.reason == "completion_recovered" and event.cost_state == "audited"
    result = registry.native_result(job.req.run_id)
    assert result["returncode"] == 0 and result["activation"] == "quarantined"
    assert result["internal_search_trials"] is None and result["recovered"] is True
    assert json.loads((job.work / "result.json").read_bytes()) == result
    assert registry.native_recoveries(job.req.run_id)[0] == event.model_dump(mode="json")
    assert job.token.read_text() not in json.dumps(result)
    assert not any("start" in args or "create" in args for args in job.docker.calls)
    job.docker.container = None
    (job.work / "result.json").unlink()
    again = service.reconcile(job.req.run_id)
    assert again.reason == "authoritative_result_retained" and again.sequence == 2
    assert registry.native_result(job.req.run_id) == result
    assert (job.work / "result.json").exists()


def test_auto_removed_container_and_forged_local_success_are_not_completion(job):
    finish(job)
    job.docker.container = None
    (job.work / "result.json").write_text('{"status":"discovery_complete","returncode":0}')
    event = job.service.reconcile(job.req.run_id)
    assert (
        event.observation == "absent"
        and event.cost_state == "unknown"
        and event.result_hash is None
    )
    with pytest.raises(RegistryError, match="no authoritative"):
        job.registry.native_result(job.req.run_id)
    assert job.registry.discovery_exposure("native:" + job.req.run_id).internal_trials is None


@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "image",
        "mount-path",
        "mount-rw",
        "extra-mount",
        "network",
        "env",
        "restart",
        "privileged",
        "extra-network",
        "name",
    ],
)
def test_changed_container_boundaries_never_stop_or_publish(job, mutation):
    finish(job)
    c = job.docker.container
    if mutation == "id":
        c["Id"] = "e" * 64
    elif mutation == "image":
        c["Image"] = "sha256:" + "e" * 64
    elif mutation == "mount-path":
        c["Mounts"][0]["Source"] = str(job.work.parent)
    elif mutation == "mount-rw":
        c["Mounts"][1]["RW"] = True
    elif mutation == "extra-mount":
        c["Mounts"].append({"Type": "volume", "Destination": "/hidden"})
    elif mutation == "network":
        job.docker.network["Id"] = "e" * 64
    elif mutation == "env":
        c["Config"]["Env"].append("OPENAI_API_KEY=must-not-leak")
    elif mutation == "restart":
        c["RestartCount"] = 1
    elif mutation == "privileged":
        c["HostConfig"]["Privileged"] = True
    elif mutation == "extra-network":
        c["NetworkSettings"]["Networks"]["internet"] = {"NetworkID": "e" * 64}
    elif mutation == "name":
        c["Name"] = "/replacement"
    event = job.service.reconcile(job.req.run_id, stop=True)
    assert event.observation == "ambiguous" and event.result_hash is None
    assert not any(
        args[:2] in (("container", "stop"), ("container", "rm")) for args in job.docker.calls
    )
    assert "must-not-leak" not in event.model_dump_json()


@pytest.mark.parametrize("source", ["frozen", "request", "symlink", "extra-frozen"])
def test_source_corruption_blocks_before_any_docker_action(job, source, tmp_path):
    frozen = job.supervisor.root / ".inputs" / job.req.run_id
    if source == "request":
        (job.work / "request.json").write_text("{}")
    elif source == "frozen":
        target = next((frozen / "qlib/features").rglob("*.bin"))
        target.chmod(0o644)
        target.write_bytes(b"changed")
    elif source == "symlink":
        (job.work / "request.json").unlink()
        (job.work / "request.json").symlink_to(job.token)
    else:
        frozen.chmod(0o755)
        (frozen / "unadmitted").write_text("unknown observations")
    job.docker.calls.clear()
    event = job.service.reconcile(job.req.run_id, stop=True)
    assert event.reason == "source_changed" and job.docker.calls == []


def test_stop_records_intent_before_exact_id_and_retains_cost_uncertainty(job):
    job.docker.container["State"].update(
        Status="running", Running=True, StartedAt=(job.now + timedelta(seconds=3)).isoformat()
    )

    def before():
        assert job.registry.native_recoveries(job.req.run_id)[-1]["action"] == "stop_requested"

    job.docker.before_stop = before
    event = job.service.reconcile(job.req.run_id, stop=True)
    assert (
        event.action == "stopped"
        and event.observation == "absent"
        and event.cost_state == "unknown"
    )
    assert event.sequence == 2
    again = job.service.reconcile(job.req.run_id)
    assert again.reason == "auto_removed_exit_unknown"
    assert sum(a[:2] == ("container", "stop") for a in job.docker.calls) == 1


def test_lost_stop_response_is_durable_and_not_blindly_retried(job):
    job.docker.container["State"].update(Status="running", Running=True)
    job.docker.stop_error = True
    event = job.service.reconcile(job.req.run_id, stop=True)
    assert event.action == "stop_uncertain"
    assert job.service.reconcile(job.req.run_id).observation == "absent"
    assert sum(a[:2] == ("container", "stop") for a in job.docker.calls) == 1


def test_daemon_and_gateway_outages_are_distinct_durable_events(job):
    job.docker.error = True
    assert job.service.reconcile(job.req.run_id).reason == "docker_unavailable"
    job.docker.error = False
    finish(job)

    def outage(*a):
        raise httpx.ReadTimeout("synthetic HTTP boundary timeout")

    job.service.gateway_receipt = outage
    assert job.service.reconcile(job.req.run_id).reason == "gateway_unavailable"
    assert len(job.registry.native_recoveries(job.req.run_id)) == 2


def test_uncertain_usage_cannot_be_released_or_completed(job):
    finish(job)
    pending(job.budget)
    before = copy.deepcopy(job.budget)
    event = job.service.reconcile(job.req.run_id)
    assert event.reason == "cost_uncertain" and job.budget == before
    assert not (job.work / "result.json").exists()
    # Only a later actual gateway receipt can resolve usage; recovery does no settlement.
    job.budget.update(receipt(job.req, job.now))
    assert job.service.reconcile(job.req.run_id).reason == "completion_recovered"


@pytest.mark.parametrize("change", ["model", "negative", "nan", "missing", "inconsistent"])
def test_malformed_or_wrong_model_budget_cannot_complete(job, change):
    finish(job)
    if change == "model":
        job.budget["models"]["chat"]["model"] = "unapproved"
    elif change == "negative":
        job.budget["uncertain"]["requests"] = -1
    elif change == "nan":
        job.budget["charged"]["cost_usd"] = "NaN"
    elif change == "missing":
        job.budget.pop("uncertain")
    else:
        job.budget["actual"]["input_tokens"] = 11
    assert job.service.reconcile(job.req.run_id).reason == "gateway_unavailable"


@pytest.mark.parametrize(
    "failure", ["nonzero", "oom", "watchdog", "output-link", "result-conflict"]
)
def test_failed_exit_and_untrusted_output_never_become_success(job, failure):
    finish(job, code=1 if failure == "nonzero" else 0)
    if failure == "oom":
        job.docker.container["State"]["OOMKilled"] = True
    elif failure == "watchdog":
        (job.work / "watchdog.json").unlink()
    elif failure == "output-link":
        (job.work / "leak").symlink_to(job.token)
    elif failure == "result-conflict":
        (job.work / "result.json").write_text("{}")
    event = job.service.reconcile(job.req.run_id)
    assert event.result_hash is None
    assert event.reason in ("exit_failed", "artifacts_unavailable", "result_conflict")


def test_readonly_reconciliations_serialize_without_relaunch(job):
    job.docker.container["State"].update(Status="running", Running=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        events = list(pool.map(lambda _: job.service.reconcile(job.req.run_id), range(2)))
    assert sorted(e.sequence for e in events) == [1, 2]
    assert all(e.observation == "running" for e in events)
    assert len(job.registry.native_recoveries(job.req.run_id)) == 2


def test_original_ambient_bundle_changes_cannot_rewrite_frozen_launch(job):
    Path(job.req.scenario.provider_uri, "manifest.json").write_text("{}")
    finish(job)
    assert job.service.reconcile(job.req.run_id).reason == "completion_recovered"


def test_legacy_exposure_without_launch_remains_ambiguous(tmp_path):
    req, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "research.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    supervisor.admit_discovery(req, registry)
    service = NativeRecoveryService(
        supervisor, registry, docker=lambda *a, **k: pytest.fail("no legacy authority")
    )
    event = service.reconcile(req.run_id, stop=True)
    assert event.reason == "legacy_launch_unbound" and event.observation == "ambiguous"


def test_binding_after_process_start_is_rejected(tmp_path, job):
    job.docker.container["State"].update(Status="running", Running=True)
    with pytest.raises(RecoveryError, match="never-started"):
        bind_container(
            job.supervisor,
            job.registry,
            job.req.run_id,
            job.docker.container_id,
            docker=job.docker,
            clock=job.clock,
        )


def test_artifact_change_during_reconciliation_is_not_success(job):
    finish(job)
    initial = job.docker.inspect_count

    def change(count):
        if count == initial + 2:
            (job.work / "research.txt").write_text("Changed after first output inventory")

    job.docker.on_inspect = change
    assert job.service.reconcile(job.req.run_id).reason == "artifacts_unavailable"
    with pytest.raises(RegistryError, match="no authoritative"):
        job.registry.native_result(job.req.run_id)


def test_created_job_can_only_be_removed_by_bound_id_without_start(job):
    event = job.service.reconcile(job.req.run_id, stop=True)
    assert event.action == "stopped" and event.observation == "absent"
    assert ("container", "rm", job.binding.container_id) in job.docker.calls
    assert not any(a[:2] == ("container", "start") for a in job.docker.calls)


def test_ambiguous_docker_json_is_an_event_not_false_absence(job):
    actual = job.service.docker

    def malformed(*args, **kwargs):
        if args[:2] == ("container", "inspect"):
            return "{not-json}"
        return actual(*args, **kwargs)

    job.service.docker = malformed
    event = job.service.reconcile(job.req.run_id)
    assert event.observation == "ambiguous" and event.result_hash is None


def test_private_extra_gateway_fields_are_not_archived_in_recovered_result(job):
    finish(job)
    job.budget["private_debug"] = "private-fixture-must-not-persist"
    job.budget["models"]["chat"]["private_debug"] = "private-fixture-must-not-persist"
    event = job.service.reconcile(job.req.run_id)
    assert event.reason == "completion_recovered"
    result = job.registry.native_result(job.req.run_id)
    assert "private-fixture-must-not-persist" not in json.dumps(result)
    assert (
        "private-fixture-must-not-persist"
        not in job.registry.blob(event.gateway_receipt_hash).decode()
    )


@pytest.mark.parametrize('case', ['reservation-floor', 'run-breach', 'day-breach', 'stale'])
def test_recovery_requires_fresh_clear_v2_scope_not_legacy_arithmetic(job, case):
    from tests.native_budget_fixtures import project

    finish(job)
    run = job.budget['accounting']['run']
    run.update(conservative_tokens=100, conservative_cost_usd='0.10')
    job.budget['accounting']['day'] = copy.deepcopy(run)
    if case.endswith('breach'):
        scope = job.budget['accounting']['run' if case == 'run-breach' else 'day']
        scope.update(breached=True, breach_call_ids=['original-call'],
                     breach_reasons=['token_reservation_exceeded'])
    project(job.budget)
    if case == 'stale':
        job.service.gateway_receipt = lambda *args: copy.deepcopy(job.budget)
    event = job.service.reconcile(job.req.run_id)
    if case == 'reservation-floor':
        assert event.reason == 'completion_recovered'
        result = job.registry.native_result(job.req.run_id)
        saved = result['gateway_final_receipt']
        assert saved['charged']['cost_usd'] == '0.10'
        assert saved['actual']['cost_usd'] == '0.01'
        assert saved['uncertain']['requests'] == 0
    else:
        assert event.reason == ('gateway_unavailable' if case == 'stale' else 'budget_breached')
        assert not (job.work/'result.json').exists()
        with pytest.raises(RegistryError):
            job.registry.native_result(job.req.run_id)
        if case != 'stale':
            assert event.cost_state == 'audited'
            assert event.gateway_receipt_hash is not None
