"""Container-local limits that do not depend on the host Docker client."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path


def output_size(work: Path) -> int:
    total = 0
    for root, _, files in os.walk(work, followlinks=False):
        for name in files:
            try:
                info = (Path(root) / name).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def supervise(
    command: list[str],
    work: Path,
    *,
    max_seconds: float,
    max_output_bytes: int,
    poll_seconds: float = 0.1,
) -> dict:
    if not 0 < max_seconds <= 86400 or not 1024 <= max_output_bytes or not 0 < poll_seconds <= 1:
        raise ValueError("explicit positive bounded runtime and output limits required")
    started = time.monotonic()
    interrupted = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, lambda number, frame: interrupted.append(number))
    process = None
    status, exit_code = "failed", 1
    try:
        # Child logs remain in the counted output directory even if the host
        # supervisor disappears. No unbounded Docker logging stream is needed.
        with (
            (work / "native-stdout.log").open("wb") as stdout,
            (work / "native-stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, start_new_session=True
            )
            while True:
                code = process.poll()
                if interrupted:
                    status, exit_code = "interrupted", 130
                    break
                if time.monotonic() - started >= max_seconds:
                    status, exit_code = "timeout", 124
                    break
                if output_size(work) > max_output_bytes:
                    status, exit_code = "output_limit", 125
                    break
                if code is not None:
                    status, exit_code = ("discovery_complete" if code == 0 else "failed"), code
                    break
                time.sleep(poll_seconds)
    finally:
        if process is not None:
            # Terminate background descendants even when the direct child has
            # already exited. Container PID 1 exits immediately after this.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result = {
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - started,
        "output_bytes": output_size(work),
        "max_seconds": max_seconds,
        "max_output_bytes": max_output_bytes,
    }
    # Replace instead of following a path that generated code could have linked.
    temporary = work / (".watchdog-" + str(os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, work / "watchdog.json")
    return result
