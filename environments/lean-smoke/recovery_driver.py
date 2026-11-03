"""Three bounded actual engines. Run with root project interpreter and checkout PYTHONPATH."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
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
CHECKOUT = Path(__file__).resolve().parents[2]


def launch(run):
    name = "qal-recovery-" + uuid.uuid4().hex
    record = {"image": IMAGE, "container_name": name, "source": str(CHECKOUT)}
    try:
        command = DOCKER + [
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
            f"type=bind,src={CHECKOUT}/src,dst=/app/src,readonly",
            "--mount",
            f"type=bind,src={run},dst=/smoke",
            IMAGE,
            "--config",
            "/smoke/engine.json",
        ]
        record["command"] = command
        created = subprocess.run(command, capture_output=True, text=True, timeout=30)
        record["create"] = {
            "exit": created.returncode,
            "stdout": created.stdout,
            "stderr": created.stderr,
        }
        created.check_returncode()
        cid = created.stdout.strip()
        record["container_id"] = cid
        with (run / "console.log").open("wb") as output:
            try:
                record["attach_exit"] = subprocess.run(
                    DOCKER + ["start", "--attach", cid],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=150,
                ).returncode
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                subprocess.run(DOCKER + ["kill", cid], capture_output=True, timeout=15)
        record["inspection"] = json.loads(
            subprocess.check_output(DOCKER + ["inspect", cid], timeout=15)
        )[0]
    finally:
        cleanup = subprocess.run(
            DOCKER + ["rm", "--force", name], capture_output=True, text=True, timeout=15
        )
        record["cleanup"] = {
            "exit": cleanup.returncode,
            "stdout": cleanup.stdout,
            "stderr": cleanup.stderr,
        }
        absent = subprocess.run(
            DOCKER + ["inspect", name], capture_output=True, text=True, timeout=15
        )
        record["after_cleanup"] = {"exit": absent.returncode, "stderr": absent.stderr}
        (run / "launch.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-completed-restart", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    origin = args.output / "origin"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare.py")),
            "--source",
            str(args.source),
            "--output",
            str(origin),
        ],
        check=True,
    )
    engine = json.loads((origin / "engine.json").read_text())
    engine["algorithm-type-name"] = "RecoveryAlgorithm"
    (origin / "engine.json").write_text(json.dumps(engine, indent=2))
    shutil.copyfile(Path(__file__).with_name("recovery_algorithm.py"), origin / "main.py")
    files = list((CHECKOUT / "src").rglob("*.py")) + list(
        Path(__file__).parent.glob("recovery_*.py")
    )
    manifest = {
        str(p.relative_to(CHECKOUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True
                ).strip(),
                "files": manifest,
            },
            indent=2,
        )
    )
    (origin / "phase").write_text("origin")
    # Only the new run tree is writable to the image UID; its parent remains private.
    for p in [origin, *origin.rglob("*")]:
        p.chmod(0o777 if p.is_dir() else 0o666)
    launch(origin)
    if not (origin / "pending-checkpoint.json").exists():
        raise RuntimeError("origin did not reach both checkpoints; no restart runs attempted")
    for phase, database in [
        ("completed-retry", origin / "completed.sqlite"),
        ("pending-restart", origin / "ledger.sqlite"),
    ]:
        if phase == "completed-retry" and args.skip_completed_restart:
            continue
        run = args.output / phase
        run.mkdir(mode=0o777)
        run.chmod(0o777)
        for name in ["engine.json", "qal-execution.json", "execution-manifest.json", "main.py"]:
            shutil.copyfile(origin / name, run / name)
        # SQLite backup preserves the real receipt history, without table edits.
        with (
            sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source,
            sqlite3.connect(run / "ledger.sqlite") as target,
        ):
            source.backup(target)
        (run / "phase").write_text(phase)
        for p in run.iterdir():
            p.chmod(0o666)
        launch(run)
    inventory = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    (args.output / "sha256.json").write_text(json.dumps(inventory, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
