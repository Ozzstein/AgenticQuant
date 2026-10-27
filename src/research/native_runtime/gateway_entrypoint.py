"""Container-local secret loading; Docker configuration contains no credentials."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("gateway entrypoint requires its dedicated container")
    config = json.loads(Path("/run/gateway-config.json").read_text())
    expires = datetime.fromisoformat(config["expires_at"].replace("Z", "+00:00"))
    remaining = (expires - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise RuntimeError("gateway scope has expired")
    environment = {
        **os.environ,
        "RESEARCH_GATEWAY_BEARER": Path("/run/secrets/bearer").read_text().strip(),
        "OPENAI_API_KEY": Path("/run/secrets/provider").read_text().strip(),
    }
    if not environment["RESEARCH_GATEWAY_BEARER"] or not environment["OPENAI_API_KEY"]:
        raise RuntimeError("gateway secret file is empty")
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.decision.research_gateway",
            "--config",
            "/run/gateway-config.json",
            "--host",
            "0.0.0.0",
            "--port",
            "8787",
        ],
        env=environment,
    )
    try:
        child.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
