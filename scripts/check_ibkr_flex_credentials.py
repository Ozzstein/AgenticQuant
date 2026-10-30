"""Check 1Password injection without printing credentials or contacting IBKR."""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "config" / "ibkr-flex.op.env"

# Only this subprocess receives the resolved IBKR fields. Drop the Service
# Account credential before starting Python, rather than exposing it to the check.
CHECK_CODE = """
import os, re, sys
query = os.environ.get('IBKR_FLEX_QUERY_ID', '')
token = os.environ.get('IBKR_FLEX_TOKEN', '')
valid = bool(re.fullmatch(r'[0-9]+', query)) and bool(token.strip())
valid = valid and not token.startswith('op://')
valid = valid and 'OP_SERVICE_ACCOUNT_TOKEN' not in os.environ
sys.exit(0 if valid else 2)
"""


def check_credentials(service_token: str, *, op_path=None, runner=subprocess.run) -> int:
    """Use exact configured references; never relay CLI stdout/stderr to the user."""
    executable = op_path or shutil.which("op") or str(Path.home() / ".local/bin/op")
    if not service_token.strip() or not Path(executable).is_file():
        print("Missing Service Account credential or 1Password CLI.", file=sys.stderr)
        return 1
    environment = dict(os.environ)
    for name in ("OP_CONNECT_HOST", "OP_CONNECT_TOKEN", "IBKR_FLEX_TOKEN", "IBKR_FLEX_QUERY_ID"):
        environment.pop(name, None)
    environment["OP_SERVICE_ACCOUNT_TOKEN"] = service_token
    try:
        result = runner(
            [
                executable,
                "run",
                "--env-file",
                str(REFERENCES),
                "--",
                "/usr/bin/env",
                "-u",
                "OP_SERVICE_ACCOUNT_TOKEN",
                sys.executable,
                "-c",
                CHECK_CODE,
            ],
            env=environment,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("1Password credential check unavailable; no IBKR request made.", file=sys.stderr)
        return 1
    finally:
        environment.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
    if result.returncode:
        print(
            "Credential check failed. Check Service Account read access to Agents, "
            "the exact item/field names, and that Query ID is numeric. "
            "Provider diagnostic text was withheld to protect credentials.",
            file=sys.stderr,
        )
        return 1
    print("IBKR Flex token and numeric Query ID loaded successfully; values withheld.")
    print("No IBKR request was made. This does not verify the IBKR token or report coverage.")
    return 0


def main() -> int:
    if not sys.stdin.isatty():
        print("Run this command in your own interactive terminal.", file=sys.stderr)
        return 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            credential = getpass.getpass("1Password SERVICE ACCOUNT token (input hidden): ")
    except (getpass.GetPassWarning, EOFError):
        print("A terminal with hidden input is required; check cancelled.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCredential check cancelled.", file=sys.stderr)
        return 130
    return check_credentials(credential)


if __name__ == "__main__":
    raise SystemExit(main())
