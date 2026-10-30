"""No real secrets, 1Password requests or IBKR calls in these boundary tests."""

import getpass
import subprocess
import sys
import warnings

import pytest

from scripts.check_ibkr_flex_credentials import CHECK_CODE, REFERENCES, check_credentials


@pytest.mark.parametrize("failure", [False, True, "timeout"])
def test_cli_boundary_hides_diagnostics_and_drops_service_token(
    tmp_path, monkeypatch, capsys, failure
):
    executable = tmp_path / "op"
    executable.touch()
    monkeypatch.setenv("OP_CONNECT_TOKEN", "old-connect-secret")
    monkeypatch.setenv("OP_CONNECT_HOST", "https://wrong.invalid")
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "old-ibkr-secret")

    def run(command, **kwargs):
        assert "synthetic-service-secret" not in command
        assert kwargs["env"]["OP_SERVICE_ACCOUNT_TOKEN"] == "synthetic-service-secret"
        assert not {"OP_CONNECT_HOST", "OP_CONNECT_TOKEN", "IBKR_FLEX_TOKEN"} & kwargs["env"].keys()
        assert command[command.index("--") + 1 :][:3] == [
            "/usr/bin/env",
            "-u",
            "OP_SERVICE_ACCOUNT_TOKEN",
        ]
        assert kwargs["capture_output"] and kwargs["timeout"] == 90
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 90, output="leaked-provider-secret")
        return subprocess.CompletedProcess(
            command, int(failure), "leaked-provider-secret", "leaked-provider-secret"
        )

    assert check_credentials(
        "synthetic-service-secret", op_path=str(executable), runner=run
    ) == int(bool(failure))
    output = capsys.readouterr()
    assert "secret" not in output.out + output.err


@pytest.mark.parametrize(
    "query,token,service,success",
    [
        ("12345", "synthetic-ibkr-token", False, True),
        ("op://unresolved", "synthetic-ibkr-token", False, False),
        ("12345", "op://unresolved", False, False),
        ("12345", "", False, False),
        ("12345", "synthetic-ibkr-token", True, False),
    ],
)
def test_child_requires_resolved_fields_without_account_token(query, token, service, success):
    environment = {"IBKR_FLEX_QUERY_ID": query, "IBKR_FLEX_TOKEN": token}
    if service:
        environment["OP_SERVICE_ACCOUNT_TOKEN"] = "synthetic-service-secret"
    result = subprocess.run(
        [sys.executable, "-c", CHECK_CODE], env=environment, capture_output=True
    )
    assert (result.returncode == 0) == success
    assert result.stdout == result.stderr == b""


def test_reference_file_contains_only_exact_user_supplied_pointers():
    rows = [row for row in REFERENCES.read_text().splitlines() if row and not row.startswith("#")]
    assert rows == [
        'IBKR_FLEX_QUERY_ID="op://Agents/IBKR Flex Queries/Query ID"',
        'IBKR_FLEX_TOKEN="op://Agents/IBKR Flex Web Service/Flex token"',
    ]


def test_hidden_prompt_cannot_fall_back_to_echo(monkeypatch, capsys):
    from scripts import check_ibkr_flex_credentials as checker

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def unavailable(_):
        warnings.warn("cannot disable echo", getpass.GetPassWarning)
        pytest.fail("prompt must stop before any echo fallback")

    monkeypatch.setattr(getpass, "getpass", unavailable)
    assert checker.main() == 1
    assert "cancelled" in capsys.readouterr().err
