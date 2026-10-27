"""Explicit Docker lifecycle for the metered native research gateway.

Only the gateway joins an egress network. Native jobs join its separate internal
network and receive only a scoped bearer. Admission files and budget state survive
container restarts; resources with a different identity are never adopted.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import Field, model_validator

from src.core.backtester import write_immutable
from src.decision.research_gateway import GatewayConfig
from src.domain.common import NonEmptyString, VersionedContract
from src.operations.store import OperationsStore
from src.research.native import NativeRequest, NativeSupervisor


class GatewayDeployment(VersionedContract):
    config: GatewayConfig
    image: str = Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")
    receipt_port: int = Field(ge=1024, le=65535)
    egress_network: NonEmptyString = "bridge"
    memory_mb: int = Field(default=512, ge=256, le=4096)
    startup_seconds: int = Field(default=30, ge=1, le=120)

    @model_validator(mode="after")
    def fixed_paths(self):
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.config.run_id):
            raise ValueError("gateway run ID must be a bounded container identity")
        if self.config.budget_store_path != "/state/budgets.sqlite3":
            raise ValueError("container gateway budget path must be /state/budgets.sqlite3")
        if self.egress_network in ("host", "none") or self.egress_network == self.network:
            raise ValueError("gateway needs a separate bounded egress network")
        return self

    @property
    def network(self):
        return "quantagent-internal-" + self.config.run_id

    @property
    def container(self):
        return "quantagent-gateway-" + self.config.run_id

    @property
    def receipt_url(self):
        return f"http://127.0.0.1:{self.receipt_port}"


def private_secret(path: Path) -> str:
    # Inspect the requested path, never a resolved symlink target. O_NOFOLLOW
    # also closes the final-component race between checking and opening it.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError("secret must be an owned ordinary private file") from exc
    with os.fdopen(fd, "r") as handle:
        info = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError("secret must be an owned ordinary file with mode 0600 or stricter")
        value = handle.read(16386).strip()
    if not value or len(value) > 16384 or "\n" in value or "\r" in value:
        raise ValueError("secret file must contain one bounded nonempty value")
    return value


def _private_write(path: Path, value: str):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())


def _private_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("gateway state requires an owned private directory without symlinks")


def _disjoint(left: Path, right: Path):
    left, right = left.resolve(), right.resolve()
    return not (left.is_relative_to(right) or right.is_relative_to(left))


class GatewayDeploymentManager:
    def __init__(self, root: str | Path):
        path = Path(root).absolute()
        from src.research.native_mounts import NativeMountAuthority
        NativeMountAuthority().protect(path)
        _private_directory(path)
        self.root = path.resolve()
        self.operations = OperationsStore(self.root / "lifecycle.sqlite3")

    def _docker(self, *args, **kwargs):
        return NativeSupervisor._docker(*args, **kwargs)

    def command(self, spec: GatewayDeployment, directory: Path) -> list[str]:
        mounts = [
            (directory / "config.json", "/run/gateway-config.json", True),
            (directory / "bearer", "/run/secrets/bearer", True),
            (directory / "provider", "/run/secrets/provider", True),
            (self.root / "state", "/state", False),
        ]
        if any("," in str(path) for path, _, _ in mounts):
            raise ValueError("Docker mount paths cannot contain commas")
        args = [
            "create",
            "--name",
            spec.container,
            "--label",
            "quantagent.deployment=" + spec.content_hash(),
            "--label",
            "quantagent.run=" + spec.config.run_id,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            f"{spec.memory_mb}m",
            "--cpus",
            "1",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            spec.network,
            "--network-alias",
            "gateway",
            "--publish",
            f"127.0.0.1:{spec.receipt_port}:8787",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=32m",
        ]
        for path, destination, readonly in mounts:
            args += [
                "--mount",
                f"type=bind,src={path},dst={destination}" + (",readonly" if readonly else ""),
            ]
        return [
            *args,
            "--entrypoint",
            "python",
            spec.image,
            "-m",
            "src.research.native_runtime.gateway_entrypoint",
        ]

    def _inspect(self, spec):
        ids = self._docker(
            "ps", "--all", "--filter", "name=^/" + spec.container + "$", "--format", "{{.ID}}"
        ).split()
        if not ids:
            return None
        if len(ids) != 1:
            raise ValueError("gateway container identity is ambiguous")
        value = json.loads(self._docker("inspect", ids[0]))[0]
        if value["Config"].get("Labels", {}).get("quantagent.deployment") != spec.content_hash():
            raise ValueError("existing gateway belongs to a different immutable deployment")
        return value

    def start(self, spec: GatewayDeployment, *, provider_key_path: str | Path) -> dict:
        from src.research.native_mounts import NativeMountAuthority
        NativeMountAuthority().protect(Path(provider_key_path))
        key = private_secret(Path(provider_key_path))
        if spec.config.expires_at <= datetime.now(timezone.utc):
            raise ValueError("gateway configuration has expired")
        with self.operations.exclusive("gateway:" + spec.config.run_id):
            directory = self.root / spec.config.run_id
            _private_directory(directory)
            write_immutable(directory / "deployment.json", spec.model_dump_json(indent=2))
            write_immutable(directory / "config.json", spec.config.model_dump_json(indent=2))
            # Every managed run shares actual UTC-day usage. A fresh run cannot
            # silently create another independent daily allowance.
            with self.operations.exclusive("gateway:shared-budget-policy"):
                _private_directory(self.root / "state")
                write_immutable(
                    self.root / "day-policy.json",
                    json.dumps(
                        {
                            "requests": spec.config.max_requests_per_day,
                            "tokens": spec.config.max_tokens_per_day,
                            "cost_usd": str(spec.config.max_cost_per_day_usd),
                        },
                        sort_keys=True,
                    ),
                )
            if not (directory / "bearer").exists():
                _private_write(directory / "bearer", secrets.token_urlsafe(48))
            token = private_secret(directory / "bearer")
            if not (directory / "provider").exists():
                _private_write(directory / "provider", key)
            if not secrets.compare_digest(private_secret(directory / "provider"), key):
                raise ValueError(
                    "existing run has a different provider credential; explicit new admission required"
                )
            self._docker("info", "--format", "{{.ServerVersion}}")
            self._docker("image", "inspect", spec.image)
            egress = json.loads(self._docker("network", "inspect", spec.egress_network))[0]
            if egress.get("Internal") or egress.get("Driver") not in ("bridge",):
                raise ValueError("gateway egress requires an ordinary external bridge network")
            networks = self._docker(
                "network", "ls", "--filter", "name=^" + spec.network + "$", "--format", "{{.ID}}"
            ).split()
            if not networks:
                self._docker(
                    "network",
                    "create",
                    "--internal",
                    "--label",
                    "quantagent.deployment=" + spec.content_hash(),
                    spec.network,
                )
            internal = json.loads(self._docker("network", "inspect", spec.network))[0]
            if (
                not internal.get("Internal")
                or internal.get("Labels", {}).get("quantagent.deployment") != spec.content_hash()
            ):
                raise ValueError("native network is not the exact isolated deployment network")
            container = self._inspect(spec)
            if container is None:
                self._docker(*self.command(spec, directory))
                container = self._inspect(spec)
            if container is None:
                raise RuntimeError("created gateway container is absent from Docker inspection")
            self._verify_container(spec, directory, container)
            networks = container["NetworkSettings"]["Networks"]
            if set(networks) - {spec.network, spec.egress_network} or spec.network not in networks:
                raise ValueError("gateway has an undeclared network attachment")
            if spec.egress_network not in networks:
                self._docker("network", "connect", spec.egress_network, spec.container)
            if not container["State"]["Running"]:
                self._docker("start", spec.container)
            deadline = time.monotonic() + spec.startup_seconds
            receipt = None
            with httpx.Client(timeout=2, trust_env=False, follow_redirects=False) as client:
                while time.monotonic() < deadline:
                    try:
                        response = client.get(
                            spec.receipt_url + "/v1/budget",
                            headers={"Authorization": "Bearer " + token},
                        )
                        response.raise_for_status()
                        receipt = response.json()
                        break
                    except (httpx.HTTPError, ValueError):
                        time.sleep(0.2)
            if receipt is None:
                raise RuntimeError("gateway did not return a bounded authenticated startup receipt")
            if (
                receipt.get("run_id") != spec.config.run_id
                or receipt.get("config_hash") != spec.config.content_hash()
                or not receipt.get("ready")
                or receipt.get("provider_configured") is not True
            ):
                raise ValueError("gateway receipt does not match ready immutable admission")
            value = {
                "status": "ready",
                "deployment_hash": spec.content_hash(),
                "config_hash": spec.config.content_hash(),
                "network": spec.network,
                "container": spec.container,
                "receipt_url": spec.receipt_url,
                "gateway_url": "http://gateway:8787",
                "receipt": receipt,
            }
            self.operations.archive(value)
            return value

    def _verify_container(self, spec, directory, container):
        config, host = container["Config"], container["HostConfig"]
        expected = {
            str(directory / "config.json"): ("/run/gateway-config.json", False),
            str(directory / "bearer"): ("/run/secrets/bearer", False),
            str(directory / "provider"): ("/run/secrets/provider", False),
            str(self.root / "state"): ("/state", True),
        }
        actual = {
            m["Source"]: (m["Destination"], m["RW"])
            for m in container["Mounts"]
            if m["Type"] == "bind"
        }
        if actual != expected or len(actual) != sum(
            m["Type"] == "bind" for m in container["Mounts"]
        ):
            raise ValueError("gateway mount identity differs from immutable deployment")
        if (
            config["Image"] != spec.image
            or config.get("User") != f"{os.getuid()}:{os.getgid()}"
            or config.get("Entrypoint") != ["python"]
            or config.get("Cmd") != ["-m", "src.research.native_runtime.gateway_entrypoint"]
            or any(
                item.partition("=")[0] in {"OPENAI_API_KEY", "RESEARCH_GATEWAY_BEARER"}
                for item in config.get("Env", [])
            )
            or not host.get("ReadonlyRootfs")
            or host.get("Privileged")
            or host.get("CapDrop") != ["ALL"]
            or host.get("CapAdd")
            or "no-new-privileges" not in host.get("SecurityOpt", [])
            or host.get("PortBindings")
            != {"8787/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(spec.receipt_port)}]}
        ):
            raise ValueError("gateway runtime isolation differs from immutable deployment")

    def stop(self, spec: GatewayDeployment) -> dict:
        with self.operations.exclusive("gateway:" + spec.config.run_id):
            container = self._inspect(spec)
            if container is not None and container["State"]["Running"]:
                self._docker("stop", "--time", "5", spec.container, timeout=15)
            receipt = {
                "status": "stopped",
                "deployment_hash": spec.content_hash(),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "budget_state_retained": True,
            }
            self.operations.archive(receipt)
            return receipt

    def run_native(
        self, spec: GatewayDeployment, request: NativeRequest, *, provider_key_path, workdir, registry=None
    ) -> dict:
        if (
            request.run_id != spec.config.run_id
            or request.gateway_config_hash != spec.config.content_hash()
            or request.network != spec.network
            or request.gateway_url != "http://gateway:8787"
            or request.gateway_receipt_url != spec.receipt_url
        ):
            raise ValueError("native request does not bind the exact managed gateway")
        paths = [Path(workdir), Path(request.scenario.provider_uri)]
        if request.document:
            paths.append(Path(request.document))
        if any(not _disjoint(path, self.root) for path in paths):
            raise ValueError(
                "native work/data/document mounts must be disjoint from private gateway state"
            )
        from src.research.registry import ResearchRegistry
        if not isinstance(registry, ResearchRegistry):
            raise ValueError("native execution requires the authoritative research registry")
        supervisor=NativeSupervisor(workdir)
        supervisor.reserve_mounts(request,registry,control_paths=(self.root,Path(provider_key_path),
                                                                  self.root/spec.config.run_id/'bearer'))
        failure = None
        try:
            receipt = self.start(spec, provider_key_path=provider_key_path)
            NativeSupervisor.verify_receipt(
                request, receipt["receipt"], now=datetime.now(timezone.utc)
            )
            return supervisor.run(
                request, gateway_token_path=self.root / spec.config.run_id / "bearer",registry=registry
            )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # Exact-label checks prevent stopping an unrelated container.
            try:
                self.stop(spec)
            except Exception as exc:
                self.operations.archive(
                    {
                        "status": "cleanup_failed",
                        "deployment_hash": spec.content_hash(),
                        "error_type": type(exc).__name__,
                    }
                )
                if failure is None:
                    raise
                failure.add_note(
                    "Gateway cleanup also failed; inspect its owned container before retrying."
                )
