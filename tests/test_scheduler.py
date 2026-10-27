"""Retired date-only scheduler routes to durable explicit schedules."""
from typer.testing import CliRunner

from scripts.run_operations import schedule_app
from src.scheduler import app


def test_scheduler_alias_is_the_canonical_cli():
    assert app is schedule_app
    result=CliRunner().invoke(app,['--help'])
    assert result.exit_code == 0
    for command in ('register','tick','serve','resume'):
        assert command in result.output


def test_simulated_day_cannot_publish_successful_run_state():
    result=CliRunner().invoke(app,['--simulate-day','--dry-run'])
    assert result.exit_code == 2
