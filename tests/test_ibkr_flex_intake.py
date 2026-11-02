"""Synthetic native-shaped XML, not broker payment/coverage acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from scripts.fetch_ibkr_flex_report import FlexCaptureError, fetch_report, private_directory
from src.data.ibkr_flex import FlexIntakeError, parse_flex_report, read_flex_capture

ACCOUNT = "synthetic-private-account"
NOW = datetime.now(timezone.utc) + timedelta(minutes=5)
ACK = b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>12345</ReferenceCode></FlexStatementResponse>"
PENDING = b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1019</ErrorCode></FlexStatementResponse>"


def report(sections="", *, account=ACCOUNT, end="20261030"):
    return (
        f'<FlexQueryResponse queryName="private-query"><FlexStatements count="1">'
        f'<FlexStatement accountId="{account}" fromDate="20261001" toDate="{end}">'
        f"{sections}</FlexStatement></FlexStatements></FlexQueryResponse>"
    ).encode()


def cash(**overrides):
    attrs = {
        "accountId": ACCOUNT,
        "currency": "USD",
        "amount": "100.000000000000000001",
        "type": "Dividends",
        "transactionID": "private-transaction",
        "actionID": "private-action",
        "dateTime": "20261030;120000",
        "description": "private-description",
        "conid": "777",
    }
    attrs.update(overrides)
    return "<CashTransaction " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>"


def accrual(**overrides):
    attrs = {
        "accountId": ACCOUNT,
        "currency": "USD",
        "conid": "777",
        "quantity": "100",
        "grossRate": "1",
        "grossAmount": "100",
        "netAmount": "85",
        "tax": "15",
        "fee": "0",
        "exDate": "20261020",
        "payDate": "20261030",
        "code": "Po",
    }
    attrs.update(overrides)
    return "<ChangeInDividendAccrual " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>"


def capture(tmp_path, raw=None, *, pending=False):
    directory = private_directory(tmp_path / "captures")
    responses = [ACK, *([PENDING] if pending else []), raw or report()]
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=responses.pop(0)))
    ) as client:
        fetch_report(client, "synthetic-token", "999", directory, sleep=lambda _: None)
    return directory


def rewrite(path, value):
    path.write_text(json.dumps(value))


def test_native_observations_exact_decimals_source_identity_and_private_summary():
    raw = report(
        "<CashTransactions>"
        + cash()
        + "</CashTransactions><ChangeInDividendAccruals>"
        + accrual()
        + "</ChangeInDividendAccruals><OpenDividendAccruals/>"
    )
    parsed = parse_flex_report(raw, expected_account_id=ACCOUNT)
    row = parsed.statements[0].sections[0].rows[0]
    assert dict(row.decimal_values)["amount"] == Decimal("100.000000000000000001")
    assert row.value("transactionID") == "private-transaction"
    assert row.value("dateTime") == "20261030;120000"  # No fabricated UTC availability.
    assert parsed.raw_hash == hashlib.sha256(raw).hexdigest()
    assert row.pointer.endswith("/CashTransactions/CashTransaction[1]")
    summary = parsed.safe_summary()
    assert summary["income_admission"] == "blocked"
    assert summary["canonical_income_records_created"] == 0
    public = json.dumps(summary)
    for private in (
        ACCOUNT,
        "private-transaction",
        "private-action",
        "private-description",
        "100.000000000000000001",
        "20261030",
    ):
        assert private not in public


def test_absent_blank_and_explicit_zero_are_distinct_not_invented_deductions():
    raw = report(
        "<ChangeInDividendAccruals>"
        + accrual(tax="").replace('fee="0"', "")
        + "</ChangeInDividendAccruals>"
    )
    row = parse_flex_report(raw).statements[0].sections[0].rows[0]
    assert row.field_status("tax") == "blank" and row.field_status("fee") == "absent"
    assert "fee" not in dict(row.decimal_values) and "tax" not in dict(row.decimal_values)
    assert {"tax", "fee"} <= set(row.missing_fields)
    zero = (
        parse_flex_report(
            report("<ChangeInDividendAccruals>" + accrual(tax="0") + "</ChangeInDividendAccruals>")
        )
        .statements[0]
        .sections[0]
        .rows[0]
    )
    assert zero.field_status("tax") == "value" and dict(zero.decimal_values)["tax"] == 0


def test_accrual_reversal_and_open_accrual_never_become_payments():
    raw = report(
        "<ChangeInDividendAccruals>"
        + accrual(code="Re")
        + '</ChangeInDividendAccruals><OpenDividendAccruals><OpenDividendAccrual quantity="100" grossAmount="100"/></OpenDividendAccruals>'
    )
    result = parse_flex_report(raw).safe_summary()
    assert result["record_categories"] == {
        "accrual_reversal_not_payment": 1,
        "open_accrual_not_payment": 1,
    }
    assert result["income_admission"] == "blocked"


def test_empty_absent_unknown_and_summary_rows_are_explicit():
    result = parse_flex_report(
        report(
            "<CashTransactions>"
            + cash(accountId="-", levelOfDetail="SUMMARY")
            + '</CashTransactions><OpenDividendAccruals/><UnreviewedVendorSection><Claim complete="true"/></UnreviewedVendorSection>'
        )
    ).safe_summary()
    assert result["sections"]["OpenDividendAccruals"]["status"] == "empty"
    assert result["sections"]["ChangeInDividendAccruals"]["status"] == "absent"
    assert result["record_categories"]["summary_not_posting"] == 1
    assert "dividend_cash_observation" not in result["record_categories"]
    assert result["unsupported_sections"] == 1 and result["income_admission"] == "blocked"


def test_unsupported_rows_and_normalized_success_flags_cannot_grant_authority():
    raw = report(
        '<CashTransactions><InventedPayment type="Dividends" amount="100" complete="true"/></CashTransactions><ActionCoverage complete="true"/>'
    )
    parsed = parse_flex_report(raw)
    assert parsed.safe_summary()["record_categories"] == {"unsupported_record": 1}
    assert parsed.safe_summary()["income_admission"] == "blocked"
    assert parsed.safe_summary()["unsupported_sections"] == 1


def test_original_leaf_pointer_uses_each_tags_own_position():
    raw = report("<CashTransactions><UnreviewedSummary/>" + cash() + cash() + "</CashTransactions>")
    rows = parse_flex_report(raw).statements[0].sections[0].rows
    assert rows[1].pointer.endswith("/CashTransaction[1]")
    assert rows[2].pointer.endswith("/CashTransaction[2]")


def test_preserves_multiple_currencies_duplicates_and_correction_legs_without_matching():
    rows = cash() + cash(currency="EUR", amount="85")
    raw = report(
        "<CashTransactions>"
        + rows
        + '</CashTransactions><StmtFunds><StatementOfFundsLine activityCode="CO" amount="-100"/><StatementOfFundsLine activityCode="FRTAX" amount="-15"/></StmtFunds>'
    )
    result = parse_flex_report(raw)
    assert len(result.statements[0].sections[0].rows) == 2
    assert [r.value("currency") for r in result.statements[0].sections[0].rows] == ["USD", "EUR"]
    assert result.safe_summary()["record_categories"]["funds_correction_observation"] == 1
    assert result.safe_summary()["canonical_income_records_created"] == 0


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "1e100", " 0 ", "1,000", "9" * 129])
def test_invalid_amount_fails_without_echoing_native_values(amount):
    with pytest.raises(FlexIntakeError, match="invalid_numeric_field"):
        parse_flex_report(
            report("<CashTransactions>" + cash(amount=amount) + "</CashTransactions>")
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'<!DOCTYPE x [<!ENTITY y "secret">]><x>&y;</x>',
        b"\xff",
        b"<FlexQueryResponse>\x00</FlexQueryResponse>",
        b"<FlexQueryResponse/>",
        b"<FlexStatementResponse/>",
    ],
)
def test_unsafe_or_wrong_xml_is_not_a_report(raw):
    with pytest.raises(FlexIntakeError):
        parse_flex_report(raw)


@pytest.mark.parametrize(
    "change", ["row-account", "expected-account", "duplicate", "nested", "date", "count"]
)
def test_identity_structure_and_statement_bounds_fail_closed(change):
    raw = report("<CashTransactions>" + cash() + "</CashTransactions>")
    expected = ACCOUNT
    if change == "row-account":
        raw = report(
            "<CashTransactions>"
            + cash(accountId="different-private-account")
            + "</CashTransactions>"
        )
    elif change == "expected-account":
        expected = "wrong-account"
    elif change == "duplicate":
        raw = report("<CashTransactions/><CashTransactions/>")
    elif change == "nested":
        raw = report(
            "<CashTransactions><CashTransaction><Nested/></CashTransaction></CashTransactions>"
        )
    elif change == "date":
        raw = report(end="20260930")
    elif change == "count":
        raw = raw.replace(b'count="1"', b'count="2"')
    with pytest.raises(FlexIntakeError):
        parse_flex_report(raw, expected_account_id=expected)


def test_real_fetch_to_reader_chain_is_read_only_and_uses_acquisition_cutoff(tmp_path):
    directory = capture(
        tmp_path, report("<CashTransactions>" + cash() + "</CashTransactions>"), pending=True
    )
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    captured = read_flex_capture(directory, as_of=NOW)
    assert len(captured.receipt_hashes) == 3
    assert captured.available_at <= NOW
    assert captured.safe_summary()["capture_consistency"] == "verified"
    assert captured.safe_summary()["source_authentication"] == "not_independently_verified"
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    with pytest.raises(FlexIntakeError, match="capture_clock_mismatch"):
        read_flex_capture(directory, as_of=datetime(2020, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(FlexIntakeError, match="aware_cutoff_required"):
        read_flex_capture(directory, as_of=datetime(2020, 1, 1))


@pytest.mark.parametrize(
    "tamper",
    [
        "raw",
        "query",
        "clock",
        "status",
        "count",
        "operation",
        "report-path",
        "symlink",
        "hardlink",
        "permissions",
        "missing-receipt",
        "ack",
        "intermediate",
        "duplicate-json",
    ],
)
def test_capture_corruption_and_path_aliases_are_rejected(tmp_path, tamper):
    directory = capture(tmp_path, pending=True)
    manifest_path = directory / "capture.json"
    manifest = json.loads(manifest_path.read_bytes())
    final = directory / manifest["report_file"]
    receipt_path = directory / (final.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_bytes())
    if tamper == "raw":
        final.write_bytes(report("<CashTransactions/>"))
    elif tamper == "query":
        receipt["query_id_sha256"] = "a" * 64
    elif tamper == "clock":
        receipt["available_at"] = "2099-01-01T00:00:00+00:00"
    elif tamper == "status":
        receipt["http_status"] = 404
    elif tamper == "count":
        manifest["requests"] = 1
    elif tamper == "operation":
        receipt["operation"] = "SendRequest"
    elif tamper == "report-path":
        manifest["report_file"] = "../private-report.xml"
    elif tamper == "symlink":
        target = directory / "alias.xml"
        final.rename(target)
        final.symlink_to(target)
    elif tamper == "hardlink":
        (directory / "alias.xml").hardlink_to(final)
    elif tamper == "permissions":
        final.chmod(0o644)
    elif tamper == "missing-receipt":
        receipt_path.unlink()
    elif tamper in ("ack", "intermediate"):
        name = "01-SendRequest.xml" if tamper == "ack" else "02-GetStatement.xml"
        path = directory / name
        body = b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1015</ErrorCode></FlexStatementResponse>"
        path.write_bytes(body)
        rpath = directory / (name + ".receipt.json")
        r = json.loads(rpath.read_bytes())
        r["raw_sha256"] = hashlib.sha256(body).hexdigest()
        rewrite(rpath, r)
    elif tamper == "duplicate-json":
        manifest_path.write_bytes(b'{"status":"downloaded","status":"downloaded"}')
    if tamper not in ("missing-receipt",):
        rewrite(receipt_path, receipt)
    if tamper != "duplicate-json":
        rewrite(manifest_path, manifest)
    with pytest.raises(FlexIntakeError):
        read_flex_capture(directory, as_of=NOW)


def test_original_reference_resume_chain_reverified_and_original_ack_tamper_blocks(tmp_path):
    source = private_directory(tmp_path / "captures")
    responses = [ACK, *([PENDING] * 8)]
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=responses.pop(0)))
    ) as client:
        with pytest.raises(FlexCaptureError):
            fetch_report(client, "synthetic-token", "999", source, sleep=lambda _: None)
    resumed = private_directory(source.parent)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=report()))
    ) as client:
        fetch_report(
            client,
            "synthetic-token",
            "999",
            resumed,
            sleep=lambda _: None,
            resume_generation=source.parent,
        )
    original = read_flex_capture(resumed, as_of=NOW)
    assert original.safe_summary()["capture_consistency"] == "verified"
    ack_receipt = source / "01-SendRequest.xml.receipt.json"
    metadata = json.loads(ack_receipt.read_bytes())
    metadata["test-only-extra-metadata"] = "kept-in-source-identity"
    rewrite(ack_receipt, metadata)
    assert read_flex_capture(resumed, as_of=NOW).source_binding_hash != original.source_binding_hash
    (source / "01-SendRequest.xml").write_bytes(ACK + b" ")
    with pytest.raises(FlexIntakeError, match="resume_acknowledgement_mismatch"):
        read_flex_capture(resumed, as_of=NOW)


def test_cli_summary_never_prints_financial_details_or_arbitrary_errors(
    tmp_path, monkeypatch, capsys
):
    from scripts import inspect_ibkr_flex_report as cli

    directory = capture(tmp_path, report("<CashTransactions>" + cash() + "</CashTransactions>"))
    monkeypatch.setattr("sys.argv", ["inspect", "--capture", str(directory)])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert ACCOUNT not in output and "private-description" not in output
    assert json.loads(output)["income_admission"] == "blocked"
    (directory / "capture.json").unlink()
    assert cli.main() == 1
    output = capsys.readouterr().out
    assert str(directory) not in output and ACCOUNT not in output
