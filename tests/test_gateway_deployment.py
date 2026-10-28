"""Filesystem/process tests and Docker boundary fixtures; no provider access."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.agents.budget import AgentBudgetError
from src.decision.research_gateway import GatewayConfig, ResearchGateway
from src.research.gateway_deployment import (
    GatewayDeployment,
    GatewayDeploymentManager,
    private_secret,
)
from src.research.native import NativeRequest, NativeSupervisor
from src.research.native_runtime.watchdog import supervise
from src.research.registry import ResearchRegistry
from src.utils.exceptions import BacktestError
from tests.test_native_rd_adapter import request as native_request


@pytest.fixture
def spec():
    price = dict(
        price_version="fixture-only",
        source_uri="https://example.test/prices",
        effective_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        input_usd_per_million="1",
        output_usd_per_million="2",
    )
    config = GatewayConfig(
        run_id="native-test",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        budget_store_path="/state/budgets.sqlite3",
        chat={**price, "model": "pinned-chat"},
        embedding={**price, "model": "pinned-embedding", "output_usd_per_million": "0"},
        max_requests=100,
        max_tokens=100000,
        max_cost_usd="5",
        max_requests_per_day=200,
        max_tokens_per_day=200000,
        max_cost_per_day_usd="10",
    )
    return GatewayDeployment(
        config=config, image="test/gateway@sha256:" + "2" * 64, receipt_port=18879
    )


def bound_request(spec):
    return NativeRequest.model_validate(
        {
            **native_request().model_dump(mode="json"),
            "run_id": spec.config.run_id,
            "gateway_config_hash": spec.config.content_hash(),
            "network": spec.network,
            "gateway_receipt_url": spec.receipt_url,
        }
    )


@pytest.fixture
def private_key(tmp_path):
    path = tmp_path / "test-key"
    path.write_text("fixture-only-not-a-provider-key")
    path.chmod(0o600)
    return path


class DockerBoundary:
    def __init__(self, manager, spec):
        self.manager, self.spec, self.calls = manager, spec, []
        self.container = None
        self.network = False

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        if args[0] == "info":
            return "fixture-engine"
        if args[:2] == ("image", "inspect"):
            return '[{"Id":"fixture-image"}]'
        if args[:2] == ("network", "ls"):
            return "fixture-network" if self.network else ""
        if args[:2] == ("network", "create"):
            self.network = True
            return "fixture-network"
        if args[:2] == ("network", "inspect"):
            internal = args[2] == self.spec.network
            return json.dumps(
                [
                    {
                        "Internal": internal,
                        "Driver": "bridge",
                        "Labels": {"quantagent.deployment": self.spec.content_hash()},
                    }
                ]
            )
        if args[0] == "ps":
            return "fixture-container" if self.container else ""
        if args[0] == "inspect":
            return json.dumps([self.container])
        if args[0] == "create":
            mounts = []
            for i, arg in enumerate(args):
                if arg == "--mount":
                    parts = args[i + 1].split(",")
                    values = dict(part.split("=", 1) for part in parts if "=" in part)
                    mounts.append(
                        dict(
                            Type="bind",
                            Source=values["src"],
                            Destination=values["dst"],
                            RW="readonly" not in parts,
                        )
                    )
            self.container = {
                "Config": {
                    "Labels": {"quantagent.deployment": self.spec.content_hash()},
                    "Image": self.spec.image,
                    "User": f"{os.getuid()}:{os.getgid()}",
                    "Entrypoint": ["python"],
                    "Cmd": ["-m", "src.research.native_runtime.gateway_entrypoint"],
                    "Env": [],
                },
                "HostConfig": {
                    "ReadonlyRootfs": True,
                    "Privileged": False,
                    "CapDrop": ["ALL"],
                    "CapAdd": [],
                    "SecurityOpt": ["no-new-privileges"],
                    "PortBindings": {
                        "8787/tcp": [
                            {"HostIp": "127.0.0.1", "HostPort": str(self.spec.receipt_port)}
                        ]
                    },
                },
                "Mounts": mounts,
                "NetworkSettings": {"Networks": {self.spec.network: {}}},
                "State": {"Running": False},
            }
            return "fixture-container"
        if args[:2] == ("network", "connect"):
            self.container["NetworkSettings"]["Networks"][args[2]] = {}
            return ""
        if args[0] in ("start", "stop"):
            self.container["State"]["Running"] = args[0] == "start"
            return ""
        raise AssertionError(args)


def install_boundary(monkeypatch, manager, spec):
    docker = DockerBoundary(manager, spec)
    monkeypatch.setattr(manager, "_docker", docker)
    real_client = httpx.Client

    def handle(request):
        assert request.url == spec.receipt_url + "/v1/budget"
        assert request.headers["Authorization"] == "Bearer " + private_secret(
            manager.root / spec.config.run_id / "bearer"
        )
        return httpx.Response(
            200,
            json={
                "run_id": spec.config.run_id,
                "config_hash": spec.config.content_hash(),
                "ready": True,
                "provider_configured": True,
            },
        )

    monkeypatch.setattr(
        "src.research.gateway_deployment.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    return docker


def test_private_key_permissions_and_original_symlink(tmp_path, private_key):
    assert private_secret(private_key) == "fixture-only-not-a-provider-key"
    link = tmp_path / "link"
    link.symlink_to(private_key)
    with pytest.raises(ValueError, match="private file"):
        private_secret(link)
    private_key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        private_secret(private_key)


def test_start_stop_restart_preserves_bearer_and_shared_budget(
    monkeypatch, tmp_path, spec, private_key
):
    manager = GatewayDeploymentManager(tmp_path / "gateways")
    docker = install_boundary(monkeypatch, manager, spec)
    first = manager.start(spec, provider_key_path=private_key)
    token = (manager.root / spec.config.run_id / "bearer").read_bytes()
    database = manager.root / "state" / "budgets.sqlite3"
    database.write_bytes(b"fixture-existing-budget-content")
    manager.stop(spec)
    second = manager.start(spec, provider_key_path=private_key)
    assert first == second and (manager.root / spec.config.run_id / "bearer").read_bytes() == token
    assert database.read_bytes() == b"fixture-existing-budget-content"
    assert sum(call[0] == "create" for call in docker.calls) == 1
    create = next(call for call in docker.calls if call[0] == "create")
    assert not any(
        "fixture-only-not-a-provider-key" in value or token.decode() in value for value in create
    )
    assert str(manager.root / "state") in " ".join(create)
    assert "127.0.0.1:18879:8787" in create
    bad = spec.model_copy(update={"config": spec.config.model_copy(update={"max_requests": 101})})
    with pytest.raises(BacktestError, match="immutable artifact"):
        manager.start(bad, provider_key_path=private_key)


@pytest.mark.parametrize(
    "change", ["mount", "network", "credential_env", "privileged", "port", "entrypoint"]
)
def test_restart_rejects_container_drift(monkeypatch, tmp_path, spec, private_key, change):
    manager = GatewayDeploymentManager(tmp_path / "gateways")
    docker = install_boundary(monkeypatch, manager, spec)
    manager.start(spec, provider_key_path=private_key)
    value = docker.container
    if change == "mount":
        value["Mounts"][0]["RW"] = True
    elif change == "network":
        value["NetworkSettings"]["Networks"]["unapproved"] = {}
    elif change == "credential_env":
        value["Config"]["Env"] = ["OPENAI_API_KEY=fixture"]
    elif change == "privileged":
        value["HostConfig"]["Privileged"] = True
    elif change == "port":
        value["HostConfig"]["PortBindings"]["8787/tcp"][0]["HostIp"] = "0.0.0.0"
    elif change == "entrypoint":
        value["Config"]["Entrypoint"] = ["sh"]
    with pytest.raises(ValueError):
        manager.start(spec, provider_key_path=private_key)


def test_daily_budget_is_actual_shared_state_and_caps_cannot_change(
    monkeypatch, tmp_path, spec, private_key
):
    manager = GatewayDeploymentManager(tmp_path / "gateways")
    install_boundary(monkeypatch, manager, spec)
    manager.start(spec, provider_key_path=private_key)
    other = spec.model_copy(
        update={
            "config": spec.config.model_copy(
                update={"run_id": "second", "max_requests_per_day": 201}
            )
        }
    )
    with pytest.raises(BacktestError, match="immutable artifact"):
        manager.start(other, provider_key_path=private_key)
    local = spec.config.model_copy(
        update={
            "budget_store_path": str(manager.root / "state" / "budgets.sqlite3"),
            "max_requests_per_day": 1,
        }
    )
    one = ResearchGateway(local, bearer="a" * 48, provider_key="fixture")
    one._budget.reserve_bound(local.chat.model, 10, 1)
    two = ResearchGateway(
        local.model_copy(update={"run_id": "second"}), bearer="b" * 48, provider_key="fixture"
    )
    with pytest.raises(AgentBudgetError, match="exhausted"):
        two._budget.reserve_bound(local.chat.model, 10, 1)


def test_native_mounts_cannot_expose_gateway_secrets(monkeypatch, tmp_path, spec, private_key):
    manager = GatewayDeploymentManager(tmp_path / "gateways")
    monkeypatch.setattr(
        manager, "start", lambda *a, **k: pytest.fail("must fail before starting gateway")
    )
    req = bound_request(spec)
    for work in (manager.root, manager.root / "jobs", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            manager.run_native(spec, req, provider_key_path=private_key, workdir=work)
    unsafe = req.model_copy(
        update={"scenario": req.scenario.model_copy(update={"provider_uri": str(manager.root)})}
    )
    with pytest.raises(ValueError, match="disjoint"):
        manager.run_native(spec, unsafe, provider_key_path=private_key, workdir=tmp_path / "jobs")


def test_cleanup_failure_does_not_mask_start_failure(monkeypatch, tmp_path, spec, private_key):
    manager = GatewayDeploymentManager(tmp_path / "gateways")

    def failed_start(*a, **k):
        raise ValueError("original-start-error")

    def failed_stop(*a, **k):
        raise RuntimeError("cleanup-error")

    monkeypatch.setattr(manager, "start", failed_start)
    monkeypatch.setattr(manager, "stop", failed_stop)
    with pytest.raises(ValueError, match="original-start-error") as info:
        manager.run_native(
            spec, bound_request(spec), provider_key_path=private_key, workdir=tmp_path / "jobs",
            registry=ResearchRegistry(tmp_path / "registry.db")
        )
    assert "cleanup" in info.value.__notes__[0]


def test_missing_created_container_is_explicit(monkeypatch, tmp_path, spec, private_key):
    manager = GatewayDeploymentManager(tmp_path / "gateways")
    install_boundary(monkeypatch, manager, spec)
    monkeypatch.setattr(manager, "_inspect", lambda _: None)
    with pytest.raises(RuntimeError, match="absent"):
        manager.start(spec, provider_key_path=private_key)


def test_watchdog_executes_real_child_and_bounds_time_and_output(tmp_path):
    for name, code, seconds, limit, expected in (
        ("ok", 'print("actual child")', 5, 100000, "discovery_complete"),
        ("timeout", "import time; time.sleep(30)", 0.15, 100000, "timeout"),
        ("output", 'print("x"*50000)', 5, 2048, "output_limit"),
        ("failed", "raise SystemExit(7)", 5, 100000, "failed"),
    ):
        work = tmp_path / name
        work.mkdir()
        result = supervise(
            [sys.executable, "-c", code],
            work,
            max_seconds=seconds,
            max_output_bytes=limit,
            poll_seconds=0.02,
        )
        assert result["status"] == expected
        assert json.loads((work / "watchdog.json").read_text()) == result
    assert (tmp_path / "ok" / "native-stdout.log").read_text() == "actual child\n"


def test_native_manifest_mounts_are_read_only(tmp_path):
    request = native_request()
    command = NativeSupervisor(tmp_path / "jobs").command(
        request, tmp_path / "work", tmp_path / "token"
    )
    for name in ("request.json", "scenario.json", "admission.json"):
        assert f"type=bind,src={tmp_path}/work/{name},dst=/work/{name},readonly" in command
    assert "--log-driver" in command


@pytest.mark.parametrize("changed_model", [False, True])
def test_native_result_requires_actual_final_budget_receipt(
    monkeypatch, tmp_path, private_key, changed_model
):
    from tests.native_launch_fixtures import install_native_boundary
    from tests.test_native_discovery_exposure import mounted_request
    request, _ = mounted_request(tmp_path)
    registry = ResearchRegistry(tmp_path / "registry.db")
    supervisor = NativeSupervisor(tmp_path / "jobs")
    boundary = install_native_boundary(monkeypatch, supervisor, request, registry)
    boundary.token = private_key
    def after_start():
        boundary.budget['ready'] = False
        if changed_model:
            boundary.budget['observed_models'] = ['wrong']
    boundary.start_hook = after_start
    result = supervisor.run(request, gateway_token_path=private_key, registry=registry)
    assert result["status"] == ("audit_unavailable" if changed_model else "discovery_complete")
    saved = json.loads((supervisor.root / request.run_id / "result.json").read_text())
    assert (saved['gateway_final_receipt'] is None) == changed_model
    if not changed_model:
        assert saved['gateway_final_receipt']['actual'] == boundary.budget['actual']
    assert saved["activation"] == "quarantined"
    assert saved["discovery_exposure_hash"] == registry.discovery_exposure("native:" + request.run_id).content_hash()
    assert saved["internal_search_trials"] is None


def test_managed_cli_exposes_lifecycle_and_validates_before_start(tmp_path):
    from typer.testing import CliRunner

    from scripts.run_rd_agent import app

    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    for name in ("managed-native", "gateway-start", "gateway-stop"):
        assert name in help_result.stdout
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}")
    result = runner.invoke(
        app,
        ["gateway-stop", "--deployment", str(wrong), "--gateway-state", str(tmp_path / "state")],
    )
    assert result.exit_code == 1
