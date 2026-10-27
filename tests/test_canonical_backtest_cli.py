import json

from typer.testing import CliRunner

from src.utils.cli import app


def test_canonical_backtest_delegates_all_operations():
    result = CliRunner().invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0
    assert all(command in result.output for command in ("run", "compare", "validate"))


def test_invalid_bundle_fails_with_durable_diagnostic(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{}')
    output = tmp_path / "replay.json"
    result = CliRunner().invoke(app, ["backtest", "run", "--input", str(bundle),
                                     "--ledger", str(tmp_path / "ledger.sqlite3"), "--output", str(output)])
    assert result.exit_code == 1
    assert not output.exists()
    failure = json.loads(output.with_suffix(".json.failure.json").read_text())
    assert failure["status"] == "failed"
    assert "input bundle missing" in failure["error"]
