"""Check private 1Password injection; optionally download an IBKR Flex report."""

from __future__ import annotations

import argparse
import getpass
import json
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
if 'OP_SERVICE_ACCOUNT_TOKEN' in os.environ:
    sys.exit(23)
if not re.fullmatch(r'[0-9]+', query):
    sys.exit(21)
if not token.strip() or token.startswith('op://'):
    sys.exit(22)
sys.exit(0)
"""

FIELD_CODE = """
import os, re, sys
name = sys.argv[1]
value = os.environ.get(name, '')
valid = bool(re.fullmatch(r'[0-9]+', value)) if name == 'IBKR_FLEX_QUERY_ID' else bool(value.strip())
valid = valid and not value.startswith('op://') and 'OP_SERVICE_ACCOUNT_TOKEN' not in os.environ
sys.exit(0 if valid else 2)
"""


def _invoke(runner, command, environment, *, timeout=30):
    try:
        return runner(command, env=environment, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _diagnose(executable, environment, runner):
    """Emit fixed labels only; never interpolate provider output or field values."""
    auth = _invoke(runner, [executable, "user", "get", "--me", "--format=json"], environment)
    if auth is None:
        print("AUTHENTICATION_UNAVAILABLE: CLI/network timeout or launch failure.", file=sys.stderr)
        return
    if auth.returncode:
        print(
            "AUTHENTICATION_FAILED: Could not verify 1Password authentication. "
            "Use the 1Password Service Account token, not the IBKR token; "
            "also check that the Service Account is active.",
            file=sys.stderr,
        )
        return
    print("1PASSWORD_AUTHENTICATION_OK", file=sys.stderr)
    fields = (
        ("IBKR_FLEX_QUERY_ID", "op://Agents/IBKR Felx Queries/credential"),
        ("IBKR_FLEX_TOKEN", "op://Agents/IBKR Flex Web Service /credential"),
    )
    all_valid = True
    for name, reference in fields:
        field_environment = {**environment, name: reference}
        result = _invoke(
            runner,
            [
                executable,
                "run",
                "--",
                "/usr/bin/env",
                "-u",
                "OP_SERVICE_ACCOUNT_TOKEN",
                sys.executable,
                "-c",
                FIELD_CODE,
                name,
            ],
            field_environment,
        )
        if result is None:
            status = "CHECK_UNAVAILABLE"
        elif result.returncode == 0:
            status = "LOADED_AND_VALID"
        elif result.returncode == 2:
            status = "READ_OR_VALIDATION_FAILED"
        else:
            status = "READ_FAILED"
        all_valid = all_valid and status == "LOADED_AND_VALID"
        print(name + ": " + status, file=sys.stderr)
        field_environment.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
    if all_valid:
        print(
            "COMBINED_CONFIGURATION_FAILED: Individual references worked; report these labels for review.",
            file=sys.stderr,
        )
    print(
        "If a field failed: check Agents vault access (Read Items), "
        "exact item/field labels and any section containing the field. "
        "Copy Secret Reference from 1Password supplies the exact path; share only that pointer.",
        file=sys.stderr,
    )


def _fetch(executable, environment, runner):
    result = _invoke(
        runner,
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
            str(ROOT / "scripts/fetch_ibkr_flex_report.py"),
        ],
        environment,
        timeout=180,
    )
    try:
        if result is None:
            raise ValueError
        report = json.loads(result.stdout)
        if report.get("status") == "failed" and result.returncode:
            reason = report.get("reason", "")
            allowed = {
                "private_directory_invalid",
                "private_directory_permissions",
                "unsupported_xml",
                "invalid_xml_report",
                "invalid_provider_error",
                "credentials_invalid",
                "deadline_exceeded",
                "response_bytes_exceeded",
                "transport_failed_no_retry",
                "transport_tls_certificate_verification_failed",
                "transport_dns_resolution_failed",
                "transport_connection_reset",
                "transport_connect_timeout",
                "transport_read_timeout",
                "transport_proxy_error",
                "transport_connect_error",
                "transport_remote_protocol_error",
                "credential_echo_refused",
                "generation_response_invalid",
                "xml_activity_report_required",
                "statement_scope_missing",
                "report_not_ready_bounded_polling_stopped",
                "service_account_not_isolated",
                "capture_unavailable",
            }
            import re

            if reason not in allowed and not re.fullmatch(
                r"(?:ibkr_error_[0-9]{4}|http_status_[0-9]{3})", reason
            ):
                raise ValueError
            print("IBKR retrieval stopped: " + reason, file=sys.stderr)
            return 1
        if result.returncode or report.get("status") != "downloaded":
            raise ValueError
        location = Path(report["directory"])
        base = Path.home() / ".local/share/quantagentlab/ibkr-flex"
        if (
            not location.is_absolute()
            or not location.is_relative_to(base)
            or ".." in location.parts
        ):
            raise ValueError
        if type(report["statements"]) is not int or report["statements"] < 1:
            raise ValueError
        print("IBKR XML report downloaded. Private capture folder:")
        print(location)
        print("Statements:", report["statements"])
        print("Income completeness and payment linkage have not yet been verified.")
        return 0
    except (ValueError, KeyError, TypeError):
        print("Report worker unavailable; diagnostic content withheld.", file=sys.stderr)
        return 1


def check_credentials(
    service_token: str, *, op_path=None, runner=subprocess.run, fetch=False
) -> int:
    """Use exact configured references; never relay CLI stdout/stderr to the user."""
    executable = op_path or shutil.which("op") or str(Path.home() / ".local/bin/op")
    if not service_token.strip() or not Path(executable).is_file():
        print("Missing Service Account credential or 1Password CLI.", file=sys.stderr)
        return 1
    # Do not let unrelated op:// environment references load other vault fields.
    environment = {
        name: os.environ[name]
        for name in (
            "HOME",
            "PATH",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "XDG_CONFIG_HOME",
            "USER",
            "LOGNAME",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        )
        if name in os.environ and not os.environ[name].startswith("op://")
    }
    environment["OP_SERVICE_ACCOUNT_TOKEN"] = service_token
    try:
        result = _invoke(
            runner,
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
            environment,
        )
        if result is None:
            print("1Password credential check unavailable; no IBKR request made.", file=sys.stderr)
            return 1
        if result.returncode:
            print(
                "Credential check failed; running bounded diagnostics. No IBKR request made.",
                file=sys.stderr,
            )
            _diagnose(executable, environment, runner)
            return 1
        if fetch:
            return _fetch(executable, environment, runner)
    finally:
        environment.pop("OP_SERVICE_ACCOUNT_TOKEN", None)
    print("IBKR Flex token and numeric Query ID loaded successfully; values withheld.")
    print("No IBKR request was made. This does not verify the IBKR token or report coverage.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download the configured XML report into a private capture folder.",
    )
    args = parser.parse_args()
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
    return check_credentials(credential, fetch=args.fetch)


if __name__ == "__main__":
    raise SystemExit(main())
