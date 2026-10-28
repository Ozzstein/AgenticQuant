"""Docker protocol fixture around real supervisor admission and registry code."""

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

from src.research import native
from src.research.native_recovery import NativeLaunch
from tests.test_native_recovery import DockerBoundary, receipt


def install_native_boundary(monkeypatch, supervisor, request, registry):
    state = SimpleNamespace(calls=[], start_hook=None, create_error=None, remove_error=None,
                            interrupted=False, missing_watchdog=False)
    projection = SimpleNamespace(image_id="sha256:" + "b" * 64, network_id="c" * 64,
                                 user=f"{os.getuid()}:{os.getgid()}", mounts=())
    state.docker = DockerBoundary(request, projection, datetime.now(timezone.utc))
    state.budget = receipt(request, datetime.now(timezone.utc))

    def docker(*args, **kwargs):
        state.calls.append(args)
        if args[0] == "info":
            return "fixture-only"
        if args[0] == "create":
            launch = NativeLaunch.model_validate(registry.native_launch(request.run_id))
            assert registry.native_container(request.run_id) is None
            work = supervisor.root / request.run_id
            expected = supervisor.command(request, work, state.token,
                                          frozen_root=supervisor.root / ".inputs" / request.run_id)
            assert list(args[1:]) == expected[2:]
            if state.create_error:
                raise state.create_error
            state.docker = DockerBoundary(request, launch, datetime.now(timezone.utc))
            return state.docker.container_id + "\n"
        if args[:2] == ("rm", "--force"):
            assert args[-1] == state.docker.container_id
            if state.remove_error:
                raise state.remove_error
            state.process.returncode = -9
            return args[-1]
        return state.docker(*args, **kwargs)

    class Process:
        def __init__(self):
            self.returncode = None if state.interrupted else 0
            self.polled = False

        def poll(self):
            if state.interrupted and not self.polled:
                self.polled = True
                raise KeyboardInterrupt("original interruption")
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def start(command, *, stdout, stderr):
        bound = registry.native_container(request.run_id)
        assert bound is not None
        assert command == ["docker", "start", "--attach", bound["container_id"]]
        state.calls.append(tuple(command))
        if state.start_hook:
            state.start_hook()
        work = supervisor.root / request.run_id
        (work / "discovery.txt").write_text("Unvalidated synthetic discovery artifact")
        if not state.missing_watchdog:
            (work / "watchdog.json").write_text(json.dumps({
                "status": "discovery_complete", "exit_code": 0, "elapsed_seconds": 0.1,
                "output_bytes": 100, "max_seconds": request.max_seconds,
                "max_output_bytes": request.max_output_bytes,
            }))
        state.process = Process()
        return state.process

    monkeypatch.setattr(supervisor, "_docker", docker)
    def gateway_receipt(*args, **kwargs):
        now = datetime.now(timezone.utc)
        state.budget["accounting"].update(as_of=now.isoformat(), day_id=now.date().isoformat())
        return state.budget

    monkeypatch.setattr(supervisor, "gateway_receipt", gateway_receipt)
    monkeypatch.setattr(native, "subprocess", SimpleNamespace(
        Popen=start, run=native.subprocess.run, TimeoutExpired=native.subprocess.TimeoutExpired))
    return state
