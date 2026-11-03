"""Bounded, isolated execution of an already freshly prepared lifecycle fixture."""

import argparse
import json
import subprocess
import uuid
from pathlib import Path

IMAGE = "sha256:94ee09ba94c575561b11bc19cc6f5406f507d1f2fa436449b13ea3532612b2d2"
DOCKER = [
    "docker",
    "--host",
    "unix:///var/run/docker.sock",
    "--config",
    "/tmp/qal-lean-docker-config",
]


def launch(run, checkout):
    name = "qal-lifecycle-" + uuid.uuid4().hex
    record = {"image": IMAGE, "container_name": name, "scope": "synthetic offline paper only"}
    evidence = run.parent
    (evidence / "launch.json").write_text(json.dumps(record))
    try:
        cid = subprocess.check_output(
            DOCKER
            + [
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--cpus",
                "2",
                "--pids-limit",
                "256",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "OMP_NUM_THREADS=1",
                "--env",
                "HOME=/smoke",
                "--mount",
                f"type=bind,src={checkout / 'src'},dst=/app/src,readonly",
                "--mount",
                f"type=bind,src={run},dst=/smoke",
                IMAGE,
                "--config",
                "/smoke/engine.json",
            ],
            timeout=30,
            text=True,
        ).strip()
        record["container_id"] = cid
        with (evidence / "engine-console.log").open("wb") as console:
            try:
                record["exit_code"] = subprocess.run(
                    DOCKER + ["start", "--attach", cid],
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    timeout=150,
                ).returncode
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
        subprocess.run(DOCKER + ["kill", name], capture_output=True, timeout=15)
        record["inspection"] = json.loads(
            subprocess.check_output(DOCKER + ["inspect", name], timeout=15)
        )[0]
    finally:
        # The name is known even if create times out after daemon-side creation.
        try:
            cleanup = subprocess.run(
                DOCKER + ["rm", "--force", name], capture_output=True, text=True, timeout=15
            )
            record["cleanup_exit"] = cleanup.returncode
            record["cleanup_stderr"] = cleanup.stderr
        finally:
            (evidence / "launch.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.run.resolve(), args.checkout.resolve())
    print({key: value for key, value in result.items() if key != "inspection"})
    if result.get("exit_code") != 0 or result.get("cleanup_exit") != 0:
        raise SystemExit(1)
