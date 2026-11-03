"""One bounded restart from a retained original timeout; never submits a new intent."""

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from recovery_driver import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    origin = args.output / "origin"
    previous = json.loads((origin / "launch.json").read_text())
    assert previous["cleanup"]["exit"] == 0
    assert (origin / "timeout.json").is_file()
    run = args.output / "pending-restart"
    run.mkdir(mode=0o777, exist_ok=False)
    run.chmod(0o777)
    for name in ["engine.json", "qal-execution.json", "execution-manifest.json", "main.py"]:
        shutil.copyfile(origin / name, run / name)
    with (
        sqlite3.connect(f"file:{origin / 'ledger.sqlite'}?mode=ro", uri=True) as source,
        sqlite3.connect(run / "ledger.sqlite") as target,
    ):
        source.backup(target)
    (run / "phase").write_text("pending-restart")
    for p in run.iterdir():
        p.chmod(0o666)
    (args.output / "resume-source.py").write_bytes(Path(__file__).read_bytes())
    launch(run)
    inventory = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file() and p.name != "sha256.json"
    }
    (args.output / "sha256.json").write_text(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
