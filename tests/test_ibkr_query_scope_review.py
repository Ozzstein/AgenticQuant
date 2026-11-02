"""Synthetic local configuration reviews; no provider or income acceptance."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data import ibkr_flex
from src.data.ibkr_flex import BASE_GAPS, FlexIntakeError, read_flex_query_scope_review

CUTOFF = datetime(2026, 10, 31, tzinfo=timezone.utc)
ACQUIRED = "2026-10-30T14:00:00+00:00"
ACCOUNT = "synthetic-private-account"
CONFIG = b"opaque synthetic screenshot bytes: private query, token, /private/path"
REPORT = (
    '<FlexQueryResponse><FlexStatements count="1">'
    f'<FlexStatement accountId="{ACCOUNT}" fromDate="20261001" toDate="20261030">'
    '<CashTransactions/><Trades><Trade quantity="123.456"/></Trades>'
    "</FlexStatement></FlexStatements></FlexQueryResponse>"
).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def put(path, raw):
    path.write_bytes(raw)
    path.chmod(0o600)


@pytest.fixture
def review_files(tmp_path):
    """Write only synthetic files, including original fixed receipt clocks."""
    tmp_path.chmod(0o700)
    ack = b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>123</ReferenceCode></FlexStatementResponse>"
    for index, (operation, raw) in enumerate((("SendRequest", ack), ("GetStatement", REPORT)), 1):
        name = f"{index:02d}-{operation}.xml"
        put(tmp_path / name, raw)
        receipt = {
            "raw_file": name,
            "operation": operation,
            "http_status": 200,
            "request_version": "3",
            "raw_sha256": digest(raw),
            "started_at": ACQUIRED,
            "available_at": ACQUIRED,
            "query_id_sha256": digest(b"synthetic-query"),
        }
        put(tmp_path / (name + ".receipt.json"), json.dumps(receipt).encode())
    put(
        tmp_path / "capture.json",
        json.dumps(
            {
                "status": "downloaded",
                "report_file": "02-GetStatement.xml",
                "directory": str(tmp_path),
                "requests": 2,
                "statements": 1,
            }
        ).encode(),
    )
    capture = ibkr_flex.read_flex_capture(tmp_path, as_of=CUTOFF)
    put(tmp_path / "configuration.bin", CONFIG)
    unknown = {"mode": "unknown", "values": []}
    declaration = {
        "schema_version": "ibkr-flex-operator-scope-1",
        "scope_authority": "operator_declared_not_provider_certified",
        "configuration_sha256": digest(CONFIG),
        "report_sha256": digest(REPORT),
        "capture_binding_sha256": capture.source_binding_hash,
        "query_id_sha256": capture.query_id_hash,
        "declared_at": "2026-10-30T16:00:00+02:00",
        "configuration_recorded_at": "2026-10-01T10:00:00+02:00",
        "query_version": "private operator version",
        "requested_sections": {"CashTransactions": ["amount"], "OpenDividendAccruals": None},
        "scope": {
            "accounts": {"mode": "selected", "values": [ACCOUNT]},
            "models": unknown,
            "asset_classes": unknown,
            "currencies": unknown,
            "security_ids": unknown,
            "period": {"from_date": "20261001", "to_date": "20261030"},
            "timezone": "Europe/Rome",
            "date_format": "yyyyMMdd",
            "time_format": None,
            "detail_level": "detail",
            "currency_mode": "native",
            "output_format": "XML",
        },
    }
    return tmp_path, declaration


def load(files, mutate=None, **kwargs):
    directory, declaration = files
    if mutate:
        mutate(declaration)
    raw = json.dumps(declaration).encode()
    put(directory / "declaration.json", raw)
    return read_flex_query_scope_review(
        directory,
        configuration_path=directory / "configuration.bin",
        declaration_path=directory / "declaration.json",
        expected_declaration_hash=kwargs.pop("expected_declaration_hash", digest(raw)),
        as_of=kwargs.pop("as_of", CUTOFF),
        **kwargs,
    )


def test_own_import_and_exact_immutable_binding(review_files):
    assert Path(ibkr_flex.__file__).resolve() == Path.cwd() / "src/data/ibkr_flex.py"
    review = load(review_files)
    assert review.configuration_hash == digest(CONFIG)
    assert review.capture.report.raw_hash == digest(REPORT)
    assert review.declaration_hash == digest(review.declaration_bytes)
    with pytest.raises(FrozenInstanceError):
        review.configuration_hash = "altered"
    assert review.safe_summary()["account_selection_matches"] is True
    assert review.safe_summary()["period_matches"] is True
    assert load(review_files).binding_hash == review.binding_hash


def test_requested_returned_empty_missing_and_unrequested(review_files):
    summary = load(review_files).safe_summary()
    sections = summary["sections"]
    assert sections["CashTransactions"]["requested"] is True
    assert sections["CashTransactions"]["status"] == "empty"
    assert sections["OpenDividendAccruals"]["requested"] is True
    assert sections["OpenDividendAccruals"]["status"] == "absent"
    assert sections["OpenDividendAccruals"]["statements_missing_section"] == 1
    assert sections["Trades"]["requested"] is False
    assert sections["Trades"]["status"] == "populated"


def test_safe_summary_and_clocks_never_restamped(review_files):
    review = load(review_files)
    summary = review.safe_summary()
    public = json.dumps(summary)
    for private in (
        ACCOUNT,
        "123.456",
        "private operator version",
        "/private/path",
        "token",
        str(review_files[0]),
        "Europe/Rome",
        "yyyyMMdd",
    ):
        assert private not in public
    assert summary["available_at"] == ACQUIRED
    assert summary["declared_at"] == ACQUIRED
    later = load(review_files, as_of=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert later.safe_summary() == summary
    assert later.binding_hash == review.binding_hash


@pytest.mark.parametrize(
    "target",
    [
        "configuration.bin",
        "02-GetStatement.xml",
        "02-GetStatement.xml.receipt.json",
        "capture.json",
    ],
)
def test_exact_bytes_tamper_rejected(review_files, target):
    directory, _ = review_files
    put(directory / target, (directory / target).read_bytes() + b" ")
    with pytest.raises(FlexIntakeError):
        load(review_files)


def test_declaration_pin_rejects_even_whitespace_tampering(review_files):
    review = load(review_files)
    directory, _ = review_files
    put(directory / "declaration.json", review.declaration_bytes + b" ")
    with pytest.raises(FlexIntakeError, match="scope_declaration_hash_mismatch"):
        read_flex_query_scope_review(
            directory,
            configuration_path=directory / "configuration.bin",
            declaration_path=directory / "declaration.json",
            expected_declaration_hash=review.declaration_hash,
            as_of=CUTOFF,
        )


@pytest.mark.parametrize(
    "key", ["configuration_sha256", "report_sha256", "capture_binding_sha256", "query_id_sha256"]
)
def test_binding_pins_cannot_be_substituted(review_files, key):
    with pytest.raises(FlexIntakeError, match="scope_binding_mismatch"):
        load(review_files, lambda d: d.update({key: "0" * 64}))


@pytest.mark.parametrize(
    "change",
    [
        {"declared_at": "2026-11-01T00:00:00Z"},
        {"declared_at": "2026-09-01T00:00:00Z"},
        {"declared_at": "2026-10-30T14:00:00"},
        {"configuration_recorded_at": "invalid-private-clock"},
    ],
)
def test_bad_clocks_fail_closed(review_files, change):
    with pytest.raises(FlexIntakeError):
        load(review_files, lambda d: d.update(change))


def test_capture_cutoff_still_enforced(review_files):
    with pytest.raises(FlexIntakeError, match="capture_clock_mismatch"):
        load(review_files, as_of=datetime(2026, 10, 29, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    "flag",
    [
        "complete",
        "source_authenticated",
        "payment_linked",
        "income_admission",
        "canonical_income_records_created",
    ],
)
def test_caller_authority_flags_rejected_at_every_level(review_files, flag):
    for level in (None, "scope"):
        directory, declaration = review_files
        with pytest.raises(FlexIntakeError, match="scope_structure_invalid"):
            load(
                (directory, declaration),
                lambda d: (d if level is None else d[level]).update({flag: True}),
            )
        (declaration if level is None else declaration[level]).pop(flag)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("scope"),
        lambda d: d.update(scope_authority="provider_certified"),
        lambda d: d.update(requested_sections={"PrivateUnsupportedSection": None}),
        lambda d: d.update(requested_sections={"CashTransactions": []}),
        lambda d: d["scope"].update(accounts={"mode": "all", "values": [ACCOUNT]}),
        lambda d: d["scope"].update(period={"from_date": "20261031", "to_date": "20261001"}),
    ],
)
def test_closed_declaration_rejects_missing_ambiguous_and_unknown(review_files, mutate):
    with pytest.raises(FlexIntakeError):
        load(review_files, mutate)


def test_mismatched_scope_is_observation_and_never_authorizes(review_files):
    def mismatch(d):
        d["scope"]["accounts"]["values"] = ["another-private-account"]
        d["scope"]["period"]["from_date"] = "20260901"

    summary = load(review_files, mismatch).safe_summary()
    assert summary["account_selection_matches"] is False
    assert summary["period_matches"] is False
    assert summary["income_admission"] == "blocked"
    assert summary["canonical_income_records_created"] == 0
    assert summary["blocking_codes"] == list(BASE_GAPS)
    assert summary["source_authentication"] == "not_independently_verified"
    assert summary["scope_authority"] == "operator_declared_not_provider_certified"


def test_fully_matching_scope_still_never_authorizes(review_files):
    summary = load(review_files).safe_summary()
    assert summary["income_admission"] == "blocked"
    assert summary["canonical_income_records_created"] == 0
    assert summary["blocking_codes"] == list(BASE_GAPS)


def test_private_file_policy_reused(review_files):
    load(review_files)
    directory, declaration = review_files
    (directory / "configuration.bin").chmod(0o644)
    with pytest.raises(FlexIntakeError, match="private_source_invalid"):
        load((directory, declaration))


def test_partial_multistatement_sections_keep_missing_and_empty_counts(review_files):
    directory, declaration = review_files
    mixed = REPORT.replace(b'count="1"', b'count="2"').replace(
        b"</FlexStatements>",
        b'<FlexStatement accountId="second-private-account" fromDate="20261001" '
        b'toDate="20261030"><OpenDividendAccruals/></FlexStatement></FlexStatements>',
    )
    put(directory / "02-GetStatement.xml", mixed)
    receipt_path = directory / "02-GetStatement.xml.receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["raw_sha256"] = digest(mixed)
    put(receipt_path, json.dumps(receipt).encode())
    manifest_path = directory / "capture.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["statements"] = 2
    put(manifest_path, json.dumps(manifest).encode())
    # Consistently rewriting local capture metadata still cannot reuse an old scope binding.
    with pytest.raises(FlexIntakeError, match="scope_binding_mismatch"):
        load(review_files)
    capture = ibkr_flex.read_flex_capture(directory, as_of=CUTOFF)
    declaration.update(
        report_sha256=digest(mixed), capture_binding_sha256=capture.source_binding_hash
    )
    summary = load(review_files).safe_summary()
    section = summary["sections"]["OpenDividendAccruals"]
    assert section["status"] == "empty"
    assert section["statements_missing_section"] == 1
    assert section["statements_with_empty_section"] == 1
    assert section["declared_field_count"] is None
    assert summary["sections"]["CashTransactions"]["declared_field_count"] == 1
    assert summary["income_admission"] == "blocked"


def test_unknown_selection_is_not_a_match_and_retrospective_clock_is_preserved(review_files):
    def retrospective(d):
        d["scope"]["accounts"] = {"mode": "unknown", "values": []}
        d["configuration_recorded_at"] = "2026-10-30T15:00:00Z"
        d["declared_at"] = "2026-10-30T16:00:00Z"

    summary = load(review_files, retrospective).safe_summary()
    assert summary["account_selection_matches"] is None
    assert summary["available_at"] == ACQUIRED
    assert summary["configuration_recorded_at"] == "2026-10-30T15:00:00+00:00"
    assert summary["historical_configuration_effectiveness"] == "not_verified"
