"""Account-bound coverage cannot certify another book's actual cash postings."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from src.domain.portfolio import ActionCoverage
from src.execution.ledger import LedgerError
from tests.test_durable_accounting import A, T, account


def coverage(**changes):
    return ActionCoverage(
        **{
            "coverage_id": "actions",
            "instrument_ids": (A.instrument_id,),
            "venue": "alpaca",
            "scope": "custodial-credit-ledger",
            "account_id": "backtest:test",
            "started_at": T,
            "ended_at": T + timedelta(days=1),
            "available_at": T + timedelta(days=1),
            "event_types": ("cash_dividend_payments",),
            "completeness": "complete",
            "gaps": (),
            "source_id": "exact-credit-source",
            "source_hash": "c" * 64,
            **changes,
        }
    )


def require(value, account_id=None):
    return value.require_complete(
        instrument_ids=(A.instrument_id,),
        started_at=T,
        ended_at=T + timedelta(days=1),
        as_of=T + timedelta(days=1),
        scope=value.scope,
        event_types=value.event_types,
        account_id=account_id,
    )


def test_custodial_scope_needs_exact_account_and_explicit_use():
    with pytest.raises(ValidationError, match="exact account"):
        coverage(account_id=None)
    actual = coverage()
    require(actual, "backtest:test")
    for requested in (None, "paper:other", "live:other"):
        with pytest.raises(ValueError, match="economic action coverage"):
            require(actual, requested)
    assert (
        ActionCoverage.model_validate_json(actual.model_dump_json()).content_hash()
        == actual.content_hash()
    )


def test_account_independent_event_stream_stays_generic_but_specific_stream_cannot_be_reused():
    public = coverage(
        scope="effective-event-stream",
        account_id=None,
        event_types=("cash_dividend_terms", "cash_dividend_distributions"),
    )
    assert "account_id" not in public.model_dump(mode="json")
    assert (
        ActionCoverage.model_validate_json(public.model_dump_json()).content_hash()
        == public.content_hash()
    )
    require(public)
    require(public, "paper:local")
    specific = public.model_copy(update={"account_id": "paper:local"})
    require(specific, "paper:local")
    with pytest.raises(ValueError, match="economic action coverage"):
        require(specific, "paper:other")


def test_recording_mismatched_account_coverage_fails_before_durable_claim(tmp_path):
    ledger = account(tmp_path / "coverage.sqlite")
    cutoff = T + timedelta(days=1)
    with pytest.raises(LedgerError, match="different account"):
        ledger.record_action_coverage(
            "backtest:test", coverage(account_id="live:other"), as_of=cutoff
        )
    assert (
        ledger.db.execute(
            "SELECT count(*) FROM events WHERE id LIKE 'action-coverage:%'"
        ).fetchone()[0]
        == 0
    )
    assert ledger.record_action_coverage("backtest:test", coverage(), as_of=cutoff)
    assert not ledger.record_action_coverage("backtest:test", coverage(), as_of=cutoff)
    ledger.close()
