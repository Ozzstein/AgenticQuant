"""Reconcile an admitted native job; never restart discovery or forgive exposure.

Docker is the exit authority. A missing auto-removed container is not a successful
exit. The registry owns immutable launch, container and sequenced recovery records.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.research.native import NativeRequest, NativeSupervisor
from src.research.native_mounts import NativeMountAuthority, canonical_path
from src.research.native_runtime.overlay import UPSTREAM_REVISION
from src.research.registry import RegistryError, ResearchRegistry, canonical, content_hash


class NativeFile(VersionedContract):
    tree: Literal["work", "frozen"]
    path: NonEmptyString
    blob_hash: Sha256
    size: int = Field(ge=0)

    @model_validator(mode="after")
    def relative_name(self):
        from pathlib import PurePosixPath

        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.path or "\\" in self.path:
            raise ValueError("native file needs an exact relative POSIX path")
        return self


class NativeMount(VersionedContract):
    source: NonEmptyString
    destination: NonEmptyString
    read_only: bool


class NativeLaunch(VersionedContract):
    run_id: NonEmptyString
    request_hash: Sha256
    discovery_exposure_hash: Sha256
    account_id: str | None = None  # Original dataset provenance, never trading authority.
    recorded_at: UtcTimestamp
    registry_path: NonEmptyString
    supervisor_root: NonEmptyString
    gateway_token_path: NonEmptyString  # Path only; never credential bytes or a token hash.
    image_id: NonEmptyString
    network_id: NonEmptyString
    user: NonEmptyString
    files: tuple[NativeFile, ...]
    mounts: tuple[NativeMount, ...]
    auto_remove: Literal[True] = True
    policy: Literal["native-recovery-1"] = "native-recovery-1"

    @model_validator(mode="after")
    def unique_files(self):
        if len({(f.tree, f.path) for f in self.files}) != len(self.files):
            raise ValueError("duplicate launch file")
        if len({m.destination for m in self.mounts}) != len(self.mounts):
            raise ValueError("duplicate launch mount")
        if {f.path for f in self.files if f.tree == "work"} != {
            "request.json",
            "scenario.json",
            "admission.json",
        }:
            raise ValueError("exact original native control files required")
        return self


class NativeContainerBinding(VersionedContract):
    run_id: NonEmptyString
    request_hash: Sha256
    discovery_exposure_hash: Sha256
    launch_hash: Sha256
    container_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: UtcTimestamp
    bound_at: UtcTimestamp
    identity_hash: Sha256

    @model_validator(mode="after")
    def chronology(self):
        if self.bound_at < self.created_at:
            raise ValueError("container binding precedes actual creation")
        return self


class NativeRecoveryEvent(VersionedContract):
    run_id: NonEmptyString
    request_hash: Sha256
    discovery_exposure_hash: Sha256
    sequence: int = Field(ge=1)
    previous_hash: Sha256 | None = None
    observed_at: UtcTimestamp
    launch_hash: Sha256 | None = None
    container_binding_hash: Sha256 | None = None
    observation: Literal["absent", "created", "running", "exited", "unavailable", "ambiguous"]
    action: Literal["observe", "stop_requested", "stopped", "stop_uncertain"] = "observe"
    reason: Literal[
        "observed",
        "legacy_launch_unbound",
        "container_unbound",
        "source_changed",
        "identity_changed",
        "docker_unavailable",
        "inspection_ambiguous",
        "auto_removed_exit_unknown",
        "stop_pending",
        "stop_acknowledged",
        "stop_unknown",
        "exit_failed",
        "gateway_unavailable",
        "cost_uncertain",
        "artifacts_unavailable",
        "completion_recovered",
        "authoritative_result_retained",
        "result_conflict",
    ]
    inspection_hash: Sha256 | None = None
    artifact_manifest_hash: Sha256 | None = None
    gateway_receipt_hash: Sha256 | None = None
    result_hash: Sha256 | None = None
    cost_state: Literal["unknown", "uncertain", "audited"] = "unknown"
    exposure_retained: Literal[True] = True
    activation: Literal["quarantined"] = "quarantined"
    internal_search_trials: None = None

    @model_validator(mode="after")
    def chain(self):
        if (self.sequence == 1) != (self.previous_hash is None):
            raise ValueError("recovery events require exact append-only sequencing")
        return self


class RecoveryError(ValueError):
    """Original authority or immutable input bytes cannot be verified."""


def _now():
    return datetime.now(timezone.utc)


def _json(raw):
    return json.loads(
        raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
    )


def _archive(registry, value):
    return registry.archive(canonical(value))


@contextmanager
def native_run_lock(registry: ResearchRegistry, run_id: str):
    """Coordinate host admission/finalization and recovery, never mount the lock."""
    directory = registry.path.parent / (registry.path.name + ".native-locks")
    NativeMountAuthority().protect(directory)
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / (content_hash(run_id.encode()) + ".lock")
    canonical_path(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RecoveryError("invalid recovery lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _authority(registry, run_id):
    if not isinstance(registry, ResearchRegistry):
        raise RecoveryError("authoritative research registry required")
    exposure = registry.discovery_exposure("native:" + run_id)
    request = NativeRequest.model_validate_json(registry.blob(exposure.request_hash))
    if request.run_id != run_id or request.content_hash() != exposure.request_hash:
        raise RecoveryError("original request identity differs")
    return request, exposure


def _tree(root):
    """Enumerate ordinary files only. No symlink, hardlink, device or FIFO reads."""
    canonical_path(root)
    if not root.is_dir():
        raise RecoveryError("original input tree missing")
    result = {}
    for path in sorted(root.rglob("*")):
        canonical_path(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
        elif not path.is_dir():
            raise RecoveryError("unsupported native file")
    return result


def _frozen_expected(supervisor, request, registry, exposure):
    bundle = _json(registry.blob(exposure.bundle_hash))
    expected = {"qlib/" + k: v for k, v in bundle["files"].items()}
    expected["qlib/manifest.json"] = exposure.bundle_hash
    inherited = []
    if exposure.document_manifest_hash:
        document = _json(registry.blob(exposure.document_manifest_hash))
        expected.update({"sources/" + k: v["sha256"] for k, v in document["files"].items()})
    for reference in request.knowledge:
        data = supervisor.knowledge_bytes(reference, registry=registry)
        expected["knowledge/" + reference.artifact + ".pkl"] = content_hash(data)
        inherited.append(
            registry.discovery_exposure("native:" + reference.source_run).content_hash()
        )
    if request.ui_source_run:
        _, result, prior, _ = supervisor._prior_run(request.ui_source_run, registry)
        inherited.append(prior.content_hash())
        expected.update({k: v for k, v in result["files"].items() if k.startswith("log/")})
    if tuple(sorted(inherited)) != exposure.inherited_exposures:
        raise RecoveryError("inherited frozen sources differ")
    return expected, bundle


def _read(path, limit=1024 * 1024 * 1024):
    return NativeSupervisor._ordinary_bytes(path, limit)


def _mounts(supervisor, request, token):
    work = supervisor.root / request.run_id
    frozen = supervisor.root / ".inputs" / request.run_id
    pairs = [
        (work, "/work", False),
        (frozen / "qlib", "/data/qlib", True),
        (token, "/run/gateway-token", True),
    ]
    pairs += [
        (work / name, "/work/" + name, True)
        for name in ("request.json", "scenario.json", "admission.json")
    ]
    if request.document:
        pairs.append((frozen / "sources", "/sources", True))
    if request.knowledge:
        pairs.append((frozen / "knowledge", "/work/knowledge-input", True))
    if request.command == "ui":
        pairs.append((frozen / "log", "/work/log", True))
    return tuple(
        NativeMount(source=str(canonical_path(p)), destination=d, read_only=r) for p, d, r in pairs
    )


def record_launch(
    supervisor, request, registry, *, gateway_token_path, preflight, exposure, clock=None
) -> NativeLaunch:
    """Archive original inputs before Docker create; contains no bearer bytes."""
    original, authoritative = _authority(registry, request.run_id)
    if original != request or authoritative != exposure:
        raise RecoveryError("launch admission differs")
    token = canonical_path(gateway_token_path)
    supervisor.reserve_mounts(request, registry, control_paths=(token,))
    expected, bundle = _frozen_expected(supervisor, request, registry, exposure)
    work = supervisor.root / request.run_id
    frozen = supervisor.root / ".inputs" / request.run_id
    actual = _tree(frozen)
    if set(actual) != set(expected):
        raise RecoveryError("frozen file coverage differs")
    controls = {
        name: _read(work / name, 16 * 1024 * 1024)
        for name in ("request.json", "scenario.json", "admission.json")
    }
    if _json(controls["request.json"]) != {
        **request.model_dump(mode="json"),
        "native_args": request.native_args(),
    }:
        raise RecoveryError("original runtime request differs")
    if _json(controls["scenario.json"]) != {
        **request.scenario.model_dump(mode="json"),
        "provider_uri": "/data/qlib",
    }:
        raise RecoveryError("original runtime scenario differs")
    admission = {
        **preflight,
        "discovery_exposure": exposure.model_dump(mode="json"),
        "discovery_exposure_hash": exposure.content_hash(),
    }
    if _json(controls["admission.json"]) != admission:
        raise RecoveryError("original preflight bytes differ")
    if (
        preflight["upstream_revision"] != UPSTREAM_REVISION
        or preflight["scenario_hash"] != request.scenario.content_hash()
        or preflight["bundle_hash"] != request.scenario.bundle_hash
        or preflight["data_snapshot_hash"]
        != hashlib.sha256(json.dumps(bundle["snapshot"], sort_keys=True).encode()).hexdigest()
    ):
        raise RecoveryError("preflight identity differs")
    now = (clock or _now)()
    supervisor.verify_receipt(request, preflight["gateway_receipt"], now=now)
    records = []
    for name, path in actual.items():
        raw = _read(path)
        if content_hash(raw) != expected[name]:
            raise RecoveryError("frozen source bytes differ")
        records.append(
            NativeFile(tree="frozen", path=name, blob_hash=registry.archive(raw), size=len(raw))
        )
    for name, raw in controls.items():
        records.append(
            NativeFile(tree="work", path=name, blob_hash=registry.archive(raw), size=len(raw))
        )
    launch = NativeLaunch(
        run_id=request.run_id,
        request_hash=exposure.request_hash,
        discovery_exposure_hash=exposure.content_hash(),
        account_id=bundle["snapshot"]["context"].get("account_id"),
        recorded_at=now,
        registry_path=str(registry.path),
        supervisor_root=str(supervisor.root),
        gateway_token_path=str(token),
        image_id=preflight["image_id"],
        network_id=preflight["network_id"],
        user=f"{os.getuid()}:{os.getgid()}",
        files=tuple(sorted(records, key=lambda f: (f.tree, f.path))),
        mounts=_mounts(supervisor, request, token),
    )
    registry.record_native_launch(launch.model_dump(mode="json"))
    return launch


def _load_launch(supervisor, registry, request, exposure):
    raw = registry.native_launch(request.run_id)
    if raw is None:
        return None
    launch = NativeLaunch.model_validate(raw)
    if (
        launch.request_hash != exposure.request_hash
        or launch.discovery_exposure_hash != exposure.content_hash()
        or launch.registry_path != str(registry.path)
        or launch.supervisor_root != str(supervisor.root)
    ):
        raise RecoveryError("launch authority differs")
    supervisor.reserve_mounts(request, registry, control_paths=(Path(launch.gateway_token_path),))
    expected, bundle = _frozen_expected(supervisor, request, registry, exposure)
    if launch.account_id != bundle["snapshot"]["context"].get("account_id"):
        raise RecoveryError("original dataset account differs")
    if launch.mounts != _mounts(supervisor, request, Path(launch.gateway_token_path)):
        raise RecoveryError("launch mounts differ")
    frozen = supervisor.root / ".inputs" / request.run_id
    actual = _tree(frozen)
    if set(actual) != set(expected):
        raise RecoveryError("frozen coverage changed")
    if {f.path: f.blob_hash for f in launch.files if f.tree == "frozen"} != expected:
        raise RecoveryError("frozen authority differs")
    for item in launch.files:
        path = (
            actual[item.path]
            if item.tree == "frozen"
            else supervisor.root / request.run_id / item.path
        )
        raw = _read(path, item.size)
        archived = registry.blob(item.blob_hash)
        if raw != archived or len(raw) != item.size:
            raise RecoveryError("original admitted bytes changed")
    return launch


def _one(docker, *args):
    values = _json(docker(*args, timeout=15))
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise RecoveryError("ambiguous Docker inspection")
    return values[0]


def _instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise RecoveryError("Docker clock lacks timezone")
    return result.astimezone(timezone.utc)


def _identity(launch, request, container, image, network):
    """Validate the actual Docker protocol; return a credential-free projection."""
    config, host = container["Config"], container["HostConfig"]
    image_config = image["Config"]
    expected_labels = {
        **(image_config.get("Labels") or {}),
        "quantagent.native-run": request.run_id,
        "quantagent.request": request.content_hash(),
    }
    if (
        container["Name"] != "/quantagent-rd-" + request.run_id
        or container["Image"] != launch.image_id
        or image["Id"] != launch.image_id
        or request.image not in image.get("RepoDigests", [])
        or config["Image"] != request.image
        or config.get("Labels") != expected_labels
        or config.get("User") != launch.user
        or config.get("Entrypoint") != image_config.get("Entrypoint")
        or config.get("Cmd") != image_config.get("Cmd")
        or config.get("WorkingDir", "") != image_config.get("WorkingDir", "")
        or config.get("Env", []) != image_config.get("Env", [])
        or any(
            e.partition("=")[0] in {"OPENAI_API_KEY", "RESEARCH_GATEWAY_BEARER"}
            for e in config.get("Env", [])
        )
    ):
        raise RecoveryError("container image/config identity changed")
    expected = {(m.source, m.destination, not m.read_only) for m in launch.mounts}
    mounts = container["Mounts"]
    binds = [m for m in mounts if m["Type"] == "bind"]
    actual = {(str(canonical_path(m["Source"])), m["Destination"], m["RW"]) for m in binds}
    if (
        actual != expected
        or len(actual) != len(binds)
        or any(m.get("Propagation", "rprivate") != "rprivate" for m in binds)
        or any(
            m["Type"] != "bind" and (m["Type"] != "tmpfs" or m["Destination"] != "/tmp")
            for m in mounts
        )
    ):
        raise RecoveryError("container mounts changed")
    ports = (
        {"19899/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(request.ui_port)}]}
        if request.command == "ui"
        else {}
    )
    if (
        not host.get("ReadonlyRootfs")
        or host.get("Privileged")
        or host.get("CapDrop") != ["ALL"]
        or host.get("CapAdd")
        or host.get("SecurityOpt") != ["no-new-privileges"]
        or host.get("AutoRemove") is not True
        or host.get("RestartPolicy", {}).get("Name", "no") not in ("", "no")
        or host.get("PidsLimit") != 256
        or host.get("Memory") != request.memory_mb * 1024 * 1024
        or host.get("NanoCpus") != request.cpus * 1_000_000_000
        or host.get("Tmpfs") != {"/tmp": "rw,nosuid,nodev,size=512m"}
        or (host.get("PortBindings") or {}) != ports
        or host.get("PublishAllPorts")
        or host.get("NetworkMode") != request.network
        or host.get("PidMode", "")
        or host.get("UsernsMode", "")
        or host.get("IpcMode", "private") not in ("private", "")
        or host.get("LogConfig", {}).get("Type") != "none"
        or any(
            host.get(k)
            for k in (
                "Binds",
                "VolumesFrom",
                "Devices",
                "DeviceRequests",
                "DeviceCgroupRules",
                "Links",
                "ExtraHosts",
            )
        )
    ):
        raise RecoveryError("container isolation changed")
    networks = container["NetworkSettings"]["Networks"]
    if (
        network["Id"] != launch.network_id
        or network["Name"] != request.network
        or network.get("Internal") is not True
        or set(networks) != {request.network}
    ):
        raise RecoveryError("container network identity changed")
    attached = networks[request.network].get("NetworkID")
    if attached != launch.network_id and not (
        container["State"]["Status"] == "created" and attached == ""
    ):
        raise RecoveryError("container network attachment differs")
    if container.get("RestartCount", 0) != 0:
        raise RecoveryError("native execution restarted")
    return {
        "container_id": container["Id"],
        "created_at": _instant(container["Created"]).isoformat(),
        "name": container["Name"],
        "image_id": launch.image_id,
        "network_id": launch.network_id,
        "launch_hash": launch.content_hash(),
        "config_hash": content_hash(canonical(config)),
        "mounts": [m.model_dump(mode="json") for m in launch.mounts],
    }


def bind_container(supervisor, registry, run_id, container_id, *, docker=None, clock=None):
    """Bind only a newly created, never-started container before generated code."""
    docker = docker or supervisor._docker
    request, exposure = _authority(registry, run_id)
    launch = _load_launch(supervisor, registry, request, exposure)
    if launch is None:
        raise RecoveryError("original launch required")
    container = _one(docker, "container", "inspect", container_id)
    identity = _identity(
        launch,
        request,
        container,
        _one(docker, "image", "inspect", request.image),
        _one(docker, "network", "inspect", request.network),
    )
    state = container["State"]
    if (
        container["Id"] != container_id
        or state["Status"] != "created"
        or state.get("Running")
        or not str(state.get("StartedAt", "")).startswith("0001-01-01")
    ):
        raise RecoveryError("binding requires original never-started container")
    binding = NativeContainerBinding(
        run_id=run_id,
        request_hash=exposure.request_hash,
        discovery_exposure_hash=exposure.content_hash(),
        launch_hash=launch.content_hash(),
        container_id=container_id,
        created_at=_instant(container["Created"]),
        bound_at=(clock or _now)(),
        identity_hash=_archive(registry, identity),
    )
    if binding.created_at < launch.recorded_at:
        raise RecoveryError("container predates launch")
    registry.record_native_container(binding.model_dump(mode="json"))
    return binding


def _budget_receipt(request, raw, now):
    """Keep only the public budget schema; reject unknown accounting, never settle."""
    NativeSupervisor.verify_receipt(request, raw, now=now, final=True)
    if raw["schema_version"] != "1.0" or any(
        type(raw[k]) is not bool for k in ("ready", "provider_configured", "streaming", "blocked")
    ):
        raise RecoveryError("invalid public receipt schema")
    value = {
        k: raw[k]
        for k in (
            "schema_version",
            "run_id",
            "config_hash",
            "expires_at",
            "ready",
            "provider_configured",
            "streaming",
            "blocked",
            "observed_models",
            "limits",
            "day_limits",
        )
    }
    value["models"] = {
        scope: {k: raw["models"][scope][k] for k in ("provider", "model")}
        for scope in ("chat", "embedding")
    }
    counters = {
        "charged": ("requests", "tokens", "cost_usd"),
        "actual": ("requests_with_usage", "input_tokens", "output_tokens", "cost_usd"),
        "uncertain": ("requests", "reserved_tokens", "reserved_cost_usd"),
    }
    for group, keys in counters.items():
        value[group] = {}
        for key in keys:
            item = raw[group][key]
            if "cost" in key:
                number = Decimal(str(item))
                if not number.is_finite() or number < 0:
                    raise RecoveryError("invalid budget cost")
                value[group][key] = str(number)
            else:
                if type(item) is not int or item < 0:
                    raise RecoveryError("invalid budget usage")
                value[group][key] = item
    charged, actual, uncertain = (value[k] for k in counters)
    if (
        charged["requests"] != actual["requests_with_usage"] + uncertain["requests"]
        or charged["tokens"]
        != actual["input_tokens"] + actual["output_tokens"] + uncertain["reserved_tokens"]
        or Decimal(charged["cost_usd"])
        != Decimal(actual["cost_usd"]) + Decimal(uncertain["reserved_cost_usd"])
        or charged["requests"] > request.gateway_limits.requests
        or charged["tokens"] > request.gateway_limits.tokens
        or Decimal(charged["cost_usd"]) > request.gateway_limits.cost_usd
    ):
        raise RecoveryError("inconsistent budget accounting")
    return value


def _artifacts(work, request):
    """Hash outputs without importing code/pickle or returning log/secret contents."""
    records, total = [], 0
    files = _tree(work)
    if len(files) > 100000:
        raise RecoveryError("artifact file allowance exceeded")
    for name, path in files.items():
        if name == "result.json":
            continue
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RecoveryError("unsupported output file")
            total += before.st_size
            if total > request.max_output_bytes:
                raise RecoveryError("artifact byte allowance exceeded")
            key = hashlib.file_digest(handle, "sha256").hexdigest()
            after = os.fstat(fd)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise RecoveryError("artifact changed while reading")
        records.append({"path": name, "sha256": key, "size": before.st_size})
    watchdog = _json(_read(work / "watchdog.json", 65536))
    if (
        watchdog["status"] != "discovery_complete"
        or watchdog["exit_code"] != 0
        or watchdog["max_seconds"] != request.max_seconds
        or watchdog["max_output_bytes"] != request.max_output_bytes
        or not 0 <= watchdog["elapsed_seconds"] <= request.max_seconds
        or not 0 <= watchdog["output_bytes"] <= request.max_output_bytes
    ):
        raise RecoveryError("original watchdog did not complete within admitted limits")
    return {"files": records, "total_bytes": total}


def _result(registry, run_id):
    try:
        return registry.native_result(run_id)
    except RegistryError as exc:
        if str(exc) == "native run has no authoritative completion record":
            return None
        raise


def _publish_result(work, result):
    """Idempotent projection of registry authority; never replace conflicting data."""
    path = work / "result.json"
    payload = json.dumps(result, indent=2).encode()
    canonical_path(path)
    if path.exists():
        if _json(_read(path, 16 * 1024 * 1024)) != result:
            raise RecoveryError("local result conflicts with authoritative completion")
        return
    # Link an fsynced temporary file atomically without overwriting another writer.
    import tempfile

    fd, temporary = tempfile.mkstemp(prefix=".recovery-result-", dir=work)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _json(_read(path, 16 * 1024 * 1024)) != result:
                raise RecoveryError("concurrent result differs") from None
    finally:
        os.unlink(temporary)
    fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class NativeRecoveryService:
    """Inspect/reconcile original authority. `stop=True` is the only mutation of Docker.

    Injected callables replace protocol boundaries in offline tests, not production
    provider/evidence verification. No route starts or recreates a native job.
    """

    def __init__(self, supervisor, registry, *, clock=None, docker=None, gateway_receipt=None):
        if not isinstance(supervisor, NativeSupervisor) or not isinstance(
            registry, ResearchRegistry
        ):
            raise RecoveryError("actual supervisor and registry required")
        self.supervisor, self.registry = supervisor, registry
        self.clock = clock or _now
        self.docker = docker or supervisor._docker
        self.gateway_receipt = gateway_receipt or supervisor.gateway_receipt

    def _append(self, request, exposure, launch, binding, **fields):
        history = self.registry.native_recoveries(request.run_id)
        previous = NativeRecoveryEvent.model_validate(history[-1]) if history else None
        event = NativeRecoveryEvent(
            run_id=request.run_id,
            request_hash=exposure.request_hash,
            discovery_exposure_hash=exposure.content_hash(),
            sequence=len(history) + 1,
            previous_hash=previous.content_hash() if previous else None,
            observed_at=self.clock(),
            launch_hash=launch.content_hash() if launch else None,
            container_binding_hash=binding.content_hash() if binding else None,
            **fields,
        )
        self.registry.append_native_recovery(event.model_dump(mode="json"))
        return event

    def _list(self, *, name=None, identity=None):
        selector = "name=^/" + name + "$" if name else "id=" + identity
        values = self.docker(
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            selector,
            "--format",
            "{{.ID}}",
            timeout=15,
        ).split()
        if len(values) > 1 or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values
        ):
            raise RecoveryError("ambiguous Docker resource enumeration")
        return values

    def _inspect(self, launch, request, binding):
        named = self._list(name="quantagent-rd-" + request.run_id)
        by_id = self._list(identity=binding.container_id) if binding else []
        if not named and not by_id:
            return None, None
        if binding is None or named != [binding.container_id] or by_id != named:
            raise RecoveryError("original named container identity differs")
        value = _one(self.docker, "container", "inspect", binding.container_id)
        identity = _identity(
            launch,
            request,
            value,
            _one(self.docker, "image", "inspect", request.image),
            _one(self.docker, "network", "inspect", request.network),
        )
        if (
            value["Id"] != binding.container_id
            or _instant(value["Created"]) != binding.created_at
            or canonical(identity) != self.registry.blob(binding.identity_hash)
        ):
            raise RecoveryError("original container binding changed")
        state = value["State"]
        status = state["Status"]
        if (
            status not in ("created", "running", "exited")
            or state.get("Paused")
            or state.get("Restarting")
            or state.get("Dead")
            or bool(state.get("Running")) != (status == "running")
        ):
            raise RecoveryError("ambiguous native process state")
        snapshot = {
            "identity": identity,
            "state": {
                k: state.get(k)
                for k in ("Status", "Running", "ExitCode", "OOMKilled", "StartedAt", "FinishedAt")
            },
            "has_error": bool(state.get("Error")),
        }
        return value, snapshot

    def reconcile(self, run_id, *, stop=False) -> NativeRecoveryEvent:
        with native_run_lock(self.registry, run_id):
            return self._reconcile(run_id, stop=stop)

    def _reconcile(self, run_id, *, stop):
        request, exposure = _authority(self.registry, run_id)
        launch_raw, binding_raw = (
            self.registry.native_launch(run_id),
            self.registry.native_container(run_id),
        )
        launch = NativeLaunch.model_validate(launch_raw) if launch_raw else None
        binding = NativeContainerBinding.model_validate(binding_raw) if binding_raw else None

        def emit(**fields):
            return self._append(request, exposure, launch, binding, **fields)

        if launch is None:
            return emit(observation="ambiguous", reason="legacy_launch_unbound")
        try:
            _load_launch(self.supervisor, self.registry, request, exposure)
        except (ValueError, OSError, KeyError):
            return emit(observation="ambiguous", reason="source_changed")
        try:
            container, snapshot = self._inspect(launch, request, binding)
        except RuntimeError:
            return emit(observation="unavailable", reason="docker_unavailable")
        except (ValueError, KeyError, TypeError, OSError):
            return emit(observation="ambiguous", reason="identity_changed")
        original = _result(self.registry, run_id)
        original_hash = _archive(self.registry, original) if original else None
        if container is None:
            if original:
                try:
                    _publish_result(self.supervisor.root / run_id, original)
                except (ValueError, OSError):
                    return emit(
                        observation="absent", reason="result_conflict", result_hash=original_hash
                    )
            return emit(
                observation="absent",
                reason="authoritative_result_retained" if original else "auto_removed_exit_unknown",
                result_hash=original_hash,
            )
        inspection_hash = _archive(self.registry, snapshot)
        status = container["State"]["Status"]
        if original:
            return emit(
                observation=status,
                reason="authoritative_result_retained",
                inspection_hash=inspection_hash,
                result_hash=original_hash,
            )
        if stop and status in ("created", "running"):
            # Commit intent BEFORE crossing the side-effect boundary. A crash
            # leaves stop_pending; the next call only observes unless explicitly asked.
            emit(
                observation=status,
                action="stop_requested",
                reason="stop_pending",
                inspection_hash=inspection_hash,
            )
            try:
                if status == "created":
                    response = self.docker("container", "rm", binding.container_id, timeout=15)
                else:
                    response = self.docker(
                        "container", "stop", "--time", "5", binding.container_id, timeout=15
                    )
                if response.strip() != binding.container_id:
                    raise RecoveryError("ambiguous stop acknowledgement")
            except (RuntimeError, ValueError, OSError):
                return emit(
                    observation="unavailable",
                    action="stop_uncertain",
                    reason="stop_unknown",
                    inspection_hash=inspection_hash,
                )
            # No inferred exit code: --rm can remove the process before this read.
            try:
                after, state = self._inspect(launch, request, binding)
            except (RuntimeError, ValueError, KeyError, TypeError, OSError):
                return emit(
                    observation="unavailable", action="stop_uncertain", reason="stop_unknown"
                )
            if after is not None and after["State"]["Status"] != "exited":
                return emit(observation="ambiguous", action="stop_uncertain", reason="stop_unknown")
            return emit(
                observation="exited" if after else "absent",
                action="stopped",
                reason="stop_acknowledged",
                inspection_hash=_archive(self.registry, state) if state else inspection_hash,
            )
        if status != "exited":
            return emit(observation=status, reason="observed", inspection_hash=inspection_hash)
        state = container["State"]
        if state["ExitCode"] != 0 or state.get("OOMKilled") or state.get("Error"):
            return emit(observation="exited", reason="exit_failed", inspection_hash=inspection_hash)
        try:
            started, finished = _instant(state["StartedAt"]), _instant(state["FinishedAt"])
            if not binding.bound_at <= started <= finished <= self.clock():
                raise RecoveryError("process chronology differs")
            receipt = _budget_receipt(
                request,
                self.gateway_receipt(request, Path(launch.gateway_token_path)),
                self.clock(),
            )
        except (
            RuntimeError,
            ValueError,
            OSError,
            KeyError,
            TypeError,
            httpx.HTTPError,
            InvalidOperation,
        ):
            return emit(
                observation="exited", reason="gateway_unavailable", inspection_hash=inspection_hash
            )
        receipt_hash = _archive(self.registry, receipt)
        if any(Decimal(str(v)) != 0 for v in receipt["uncertain"].values()):
            return emit(
                observation="exited",
                reason="cost_uncertain",
                cost_state="uncertain",
                inspection_hash=inspection_hash,
                gateway_receipt_hash=receipt_hash,
            )
        work = self.supervisor.root / run_id
        try:
            artifacts = _artifacts(work, request)
            # Re-read after hashing: another start/mutation cannot be called a stable exit.
            after, snapshot_after = self._inspect(launch, request, binding)
            if (
                after is None
                or snapshot_after != snapshot
                or _artifacts(work, request) != artifacts
            ):
                raise RecoveryError("exit/artifacts changed while reconciling")
            _load_launch(self.supervisor, self.registry, request, exposure)
        except (RuntimeError, ValueError, OSError, KeyError, TypeError):
            return emit(
                observation="exited",
                reason="artifacts_unavailable",
                cost_state="audited",
                inspection_hash=inspection_hash,
                gateway_receipt_hash=receipt_hash,
            )
        artifact_hash = _archive(self.registry, artifacts)
        result = {
            "status": "discovery_complete",
            "returncode": 0,
            "run_id": run_id,
            "elapsed_seconds": (finished - started).total_seconds(),
            "request_hash": request.content_hash(),
            "upstream_revision": UPSTREAM_REVISION,
            "files": {f["path"]: f["sha256"] for f in artifacts["files"]},
            "activation": "quarantined",
            "discovery_exposure_hash": exposure.content_hash(),
            "internal_search_trials": None,
            "cleanup_error": None,
            "gateway_final_receipt": receipt,
            "gateway_audit_error": None,
            "recovered": True,
            "container_id": binding.container_id,
            "exit_evidence_hash": inspection_hash,
            "artifact_manifest_hash": artifact_hash,
        }
        if (work / "result.json").exists():
            return emit(
                observation="exited",
                reason="result_conflict",
                inspection_hash=inspection_hash,
                artifact_manifest_hash=artifact_hash,
                gateway_receipt_hash=receipt_hash,
                cost_state="audited",
            )
        self.registry.record_native_result(result)
        result_hash = _archive(self.registry, result)
        _publish_result(work, result)
        return emit(
            observation="exited",
            reason="completion_recovered",
            inspection_hash=inspection_hash,
            artifact_manifest_hash=artifact_hash,
            gateway_receipt_hash=receipt_hash,
            result_hash=result_hash,
            cost_state="audited",
        )
