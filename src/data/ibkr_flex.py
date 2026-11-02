"""Bounded, read-only IBKR Flex intake. Observations never authorize income.

Keep native strings/identities and absent/blank/zero distinctions. Receipt hashes
establish local capture consistency, not broker signatures, query completeness,
pre-ex availability, deduction applicability, or business-level payment joins.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.data.private_files import PrivateFileError, read_private

MAX_BYTES = 8 * 1024 * 1024
MAX_NODES = 100_000
SECTIONS = {
    "CashReport": "CashReportCurrency",
    "StmtFunds": "StatementOfFundsLine",
    "CashTransactions": "CashTransaction",
    "ChangeInDividendAccruals": "ChangeInDividendAccrual",
    "OpenDividendAccruals": "OpenDividendAccrual",
    "CorporateActions": "CorporateAction",
    "Trades": "Trade",
    "SecuritiesInfo": "SecurityInfo",
    "OpenPositions": "OpenPosition",
}
DECIMALS = {
    "amount",
    "debit",
    "credit",
    "balance",
    "fxRateToBase",
    "quantity",
    "grossRate",
    "grossAmount",
    "netAmount",
    "tax",
    "fee",
    "tradeGross",
    "tradeTax",
    "tradeCommission",
    "tradeQuantity",
    "tradePrice",
    "startingCash",
    "endingCash",
    "dividends",
    "withholdingTax",
}
ACCRUAL_FIELDS = (
    "currency",
    "conid",
    "exDate",
    "payDate",
    "quantity",
    "grossRate",
    "grossAmount",
    "netAmount",
    "tax",
    "fee",
)
BASE_GAPS = (
    "query_configuration_and_filters_not_verified",
    "complete_window_and_late_amendments_not_verified",
    "pre_ex_terms_and_entitlement_convention_not_verified",
    "payment_deduction_and_revision_linkage_not_verified",
    "account_mode_and_live_posting_not_authenticated_by_reader",
)


class FlexIntakeError(ValueError):
    """Only code-owned labels, never provider content or filesystem paths."""


def _fail(code):
    raise FlexIntakeError(code)


def _day(value):
    try:
        if not re.fullmatch(r"[0-9]{8}|[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            _fail("invalid_statement_date")
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        _fail("invalid_statement_date")


def _instant(value):
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            _fail("invalid_receipt_clock")
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError):
        _fail("invalid_receipt_clock")


@dataclass(frozen=True)
class FlexRow:
    pointer: str
    tag: str
    attributes: tuple[tuple[str, str], ...] = field(repr=False)
    decimal_values: tuple[tuple[str, Decimal], ...] = field(repr=False)
    category: str
    missing_fields: tuple[str, ...]

    def value(self, name: str) -> str | None:
        """None means absent; an empty string means selected but blank."""
        return dict(self.attributes).get(name)

    def field_status(self, name: str) -> str:
        value = self.value(name)
        return "absent" if value is None else "blank" if value == "" else "value"


@dataclass(frozen=True)
class FlexSection:
    name: str
    rows: tuple[FlexRow, ...]


@dataclass(frozen=True)
class FlexStatement:
    account_id: str = field(repr=False)
    started_on: date
    ended_on: date
    attributes: tuple[tuple[str, str], ...] = field(repr=False)
    sections: tuple[FlexSection, ...]


@dataclass(frozen=True)
class FlexReport:
    raw_hash: str
    statements: tuple[FlexStatement, ...]

    def safe_summary(self) -> dict:
        """No account IDs, descriptions, amounts, native dates, or arbitrary text."""
        sections = {}
        categories = Counter()
        for name in SECTIONS:
            selected = [s for st in self.statements for s in st.sections if s.name == name]
            rows = [r for s in selected for r in s.rows]
            categories.update(r.category for r in rows)
            sections[name] = {
                "statements_with_section": len(selected),
                "status": "absent" if not selected else "empty" if not rows else "populated",
                "rows": len(rows),
                "missing_selected_values": sorted({f for r in rows for f in r.missing_fields}),
            }
        unsupported = sum(s.name not in SECTIONS for st in self.statements for s in st.sections)
        return {
            "schema_version": "ibkr-flex-intake-1",
            "statements": len(self.statements),
            "sections": sections,
            "record_categories": dict(sorted(categories.items())),
            "unsupported_sections": unsupported,
            "income_admission": "blocked",
            "canonical_income_records_created": 0,
            "blocking_codes": list(BASE_GAPS),
        }


def _category(section, row):
    if section == "CashTransactions":
        return {
            "Dividends": "dividend_cash_observation",
            "Withholding Tax": "withholding_cash_observation",
            "Payment In Lieu Of Dividends": "payment_in_lieu_observation",
        }.get(row.get("type"), "other_cash_observation")
    if section == "ChangeInDividendAccruals":
        return {"Po": "accrual_posting", "Re": "accrual_reversal_not_payment"}.get(
            row.get("code"), "accrual_other_observation"
        )
    if section == "OpenDividendAccruals":
        return "open_accrual_not_payment"
    if section == "StmtFunds":
        return {
            "DIV": "funds_dividend_observation",
            "FRTAX": "funds_tax_observation",
            "PIL": "funds_payment_in_lieu_observation",
            "CO": "funds_correction_observation",
            "CA": "funds_cancellation_observation",
        }.get(row.get("activityCode"), "other_funds_observation")
    return "other_observation"


def _xml(raw: bytes):
    if not raw or len(raw) > MAX_BYTES:
        _fail("report_size_invalid")
    try:
        text = raw.decode("utf-8-sig")
        if "\x00" in text or re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
            _fail("unsupported_xml")
        root = ET.fromstring(text)
    except (UnicodeError, ET.ParseError):
        _fail("invalid_xml")
    stack = [(root, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > 8:
            _fail("xml_structure_limit")
        stack.extend((child, depth + 1) for child in node)
    return root


def parse_flex_report(raw: bytes, *, expected_account_id: str | None = None) -> FlexReport:
    """Parse original UTF-8 XML; preserve date-only/native timestamps as supplied."""
    root = _xml(raw)
    if root.tag != "FlexQueryResponse":
        _fail("activity_response_required")
    if len(root) != 1 or root[0].tag != "FlexStatements" or not len(root[0]):
        _fail("statement_container_invalid")
    if "count" in root[0].attrib and root[0].get("count") != str(len(root[0])):
        _fail("statement_count_mismatch")
    statements = []
    for index, statement in enumerate(root[0], 1):
        account = statement.get("accountId", "")
        if statement.tag != "FlexStatement" or not account:
            _fail("statement_identity_missing")
        if expected_account_id is not None and account != expected_account_id:
            _fail("statement_account_mismatch")
        start, end = _day(statement.get("fromDate", "")), _day(statement.get("toDate", ""))
        if start > end:
            _fail("statement_interval_invalid")
        sections, names = [], set()
        for section in statement:
            if section.tag in names:
                _fail("duplicate_statement_section")
            names.add(section.tag)
            rows = []
            positions = Counter()
            for row in section:
                positions[row.tag] += 1
                position = positions[row.tag]
                if len(row):
                    _fail("unsupported_nested_record")
                summary = row.get("levelOfDetail") == "SUMMARY"
                if row.get("accountId") not in (None, "", account) and not (
                    summary and row.get("accountId") == "-"
                ):
                    _fail("row_account_mismatch")
                decimals = []
                for name, value in row.attrib.items():
                    if name in DECIMALS and value != "":
                        if len(value) > 128 or not re.fullmatch(
                            r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", value
                        ):
                            _fail("invalid_numeric_field")
                        decimals.append((name, Decimal(value)))
                supported = section.tag in SECTIONS and row.tag == SECTIONS[section.tag]
                required = (
                    ACCRUAL_FIELDS
                    if section.tag in ("ChangeInDividendAccruals", "OpenDividendAccruals")
                    else ("currency", "amount", "type", "transactionID")
                    if section.tag == "CashTransactions"
                    else ("currency", "amount", "balance", "transactionID")
                    if section.tag == "StmtFunds"
                    else ()
                )
                rows.append(
                    FlexRow(
                        pointer=f"/FlexQueryResponse/FlexStatements/FlexStatement[{index}]/{section.tag}/{row.tag}[{position}]",
                        tag=row.tag,
                        attributes=tuple(sorted(row.attrib.items())),
                        decimal_values=tuple(sorted(decimals)),
                        category=(
                            "summary_not_posting"
                            if summary
                            else _category(section.tag, row)
                            if supported
                            else "unsupported_record"
                        ),
                        missing_fields=("transaction_posting_detail",)
                        if summary
                        else tuple(name for name in required if not row.get(name)),
                    )
                )
            sections.append(FlexSection(section.tag, tuple(rows)))
        statements.append(
            FlexStatement(
                account, start, end, tuple(sorted(statement.attrib.items())), tuple(sections)
            )
        )
    return FlexReport(hashlib.sha256(raw).hexdigest(), tuple(statements))


def _private_read(path: Path, limit: int = MAX_BYTES) -> bytes:
    try:
        return read_private(path, limit)
    except PrivateFileError as exc:
        _fail(str(exc))
    except OSError:
        _fail("private_source_unavailable")


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_json_field")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        if not isinstance(result, dict):
            _fail("invalid_capture_metadata")
        return result
    except (UnicodeError, ValueError):
        _fail("invalid_capture_metadata")


@dataclass(frozen=True)
class CapturedFlexReport:
    report: FlexReport
    available_at: datetime
    receipt_hashes: tuple[str, ...]
    capture_manifest_hash: str
    query_id_hash: str
    source_binding_hash: str

    def safe_summary(self):
        return {
            **self.report.safe_summary(),
            "capture_consistency": "verified",
            "source_authentication": "not_independently_verified",
            "available_at": self.available_at.isoformat(),
            "requests": len(self.receipt_hashes),
        }


def read_flex_capture(directory: str | Path, *, as_of: datetime) -> CapturedFlexReport:
    """Read only; verify local raw/receipt chain and actual acquisition cutoff.

    Caller-supplied manifests/hashes are not external provenance attestations.
    The source remains diagnostic and is deliberately not admitted by income.py.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        _fail("aware_cutoff_required")
    directory = Path(directory)
    manifest_raw = _private_read(directory / "capture.json", 64 * 1024)
    manifest = _json(manifest_raw)
    name = manifest.get("report_file", "")
    if (
        manifest.get("status") != "downloaded"
        or not isinstance(name, str)
        or not re.fullmatch(r"[0-9]{2}-GetStatement\.xml", name)
    ):
        _fail("download_manifest_invalid")
    if manifest.get("directory") != str(directory) or type(manifest.get("requests")) is not int:
        _fail("download_manifest_invalid")
    receipts = sorted(directory.glob("*.xml.receipt.json"))
    if not 1 <= len(receipts) <= 9 or manifest["requests"] != len(receipts):
        _fail("capture_receipt_count_mismatch")
    hashes, bodies, operations = [], {}, []
    inherited_hashes = []
    previous = first_started = None
    query_hash = None
    for index, path in enumerate(receipts, 1):
        receipt_raw = _private_read(path, 64 * 1024)
        receipt = _json(receipt_raw)
        raw_file = receipt.get("raw_file", "")
        operation = receipt.get("operation")
        if (
            operation not in ("SendRequest", "GetStatement")
            or raw_file != f"{index:02d}-{operation}.xml"
            or path.name != raw_file + ".receipt.json"
        ):
            _fail("capture_operation_invalid")
        if receipt.get("http_status") != 200 or receipt.get("request_version") != "3":
            _fail("capture_receipt_invalid")
        raw = _private_read(directory / raw_file)
        if hashlib.sha256(raw).hexdigest() != receipt.get("raw_sha256"):
            _fail("capture_raw_hash_mismatch")
        started, available = (
            _instant(receipt.get("started_at")),
            _instant(receipt.get("available_at")),
        )
        if (
            started > available
            or available > as_of
            or (previous is not None and started < previous)
        ):
            _fail("capture_clock_mismatch")
        previous = available
        if first_started is None:
            first_started = started
        current_hash = receipt.get("query_id_sha256", "")
        if (
            not isinstance(current_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", current_hash)
            or (query_hash is not None and query_hash != current_hash)
        ):
            _fail("capture_query_mismatch")
        query_hash = current_hash
        operations.append(operation)
        hashes.append(hashlib.sha256(receipt_raw).hexdigest())
        bodies[raw_file] = raw
    if operations == ["GetStatement"] * len(operations):
        resume_raw = _private_read(directory / "resume.json", 64 * 1024)
        resume = _json(resume_raw)
        if not isinstance(resume.get("directory"), str):
            _fail("resume_source_invalid")
        source = Path(resume["directory"])
        if (
            source.parent != directory.parent
            or source == directory
            or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}", source.name)
        ):
            _fail("resume_source_invalid")
        ack_raw = _private_read(source / "01-SendRequest.xml")
        ack_receipt_raw = _private_read(source / "01-SendRequest.xml.receipt.json", 64 * 1024)
        ack_receipt = _json(ack_receipt_raw)
        inherited_hashes = [
            hashlib.sha256(raw).hexdigest() for raw in (resume_raw, ack_receipt_raw, ack_raw)
        ]
        if (
            resume.get("acknowledgement_sha256") != hashlib.sha256(ack_raw).hexdigest()
            or ack_receipt.get("raw_sha256") != hashlib.sha256(ack_raw).hexdigest()
            or ack_receipt.get("query_id_sha256") != query_hash
            or ack_receipt.get("operation") != "SendRequest"
            or ack_receipt.get("raw_file") != "01-SendRequest.xml"
            or ack_receipt.get("http_status") != 200
            or ack_receipt.get("request_version") != "3"
            or not _instant(ack_receipt.get("started_at"))
            <= _instant(ack_receipt.get("available_at"))
            <= first_started
        ):
            _fail("resume_acknowledgement_mismatch")
    elif operations[0] == "SendRequest" and operations[1:] == ["GetStatement"] * (
        len(operations) - 1
    ):
        ack_raw = bodies["01-SendRequest.xml"]
    else:
        _fail("capture_operation_invalid")
    ack = _xml(ack_raw)
    if (
        ack.tag != "FlexStatementResponse"
        or ack.findtext("Status") != "Success"
        or ack.find("ErrorCode") is not None
        or not re.fullmatch(r"[0-9]+", ack.findtext("ReferenceCode", ""))
    ):
        _fail("capture_acknowledgement_invalid")
    if name not in bodies or name != receipts[-1].name.removesuffix(".receipt.json"):
        _fail("capture_final_report_mismatch")
    for body_name, raw in bodies.items():
        if body_name.endswith("-GetStatement.xml") and body_name != name:
            pending = _xml(raw)
            if pending.tag != "FlexStatementResponse" or pending.findtext("ErrorCode") not in {
                "1004",
                "1005",
                "1006",
                "1007",
                "1008",
                "1009",
                "1019",
                "1021",
            }:
                _fail("capture_intermediate_response_invalid")
    report = parse_flex_report(bodies[name])
    if type(manifest.get("statements")) is not int or manifest["statements"] != len(
        report.statements
    ):
        _fail("capture_statement_count_mismatch")
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    binding = json.dumps(
        {
            "policy": "ibkr-flex-local-capture-consistency-1",
            "manifest_hash": manifest_hash,
            "receipt_hashes": hashes,
            "inherited_hashes": inherited_hashes,
            "report_hash": report.raw_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return CapturedFlexReport(
        report,
        previous,
        tuple(hashes),
        manifest_hash,
        query_hash,
        hashlib.sha256(binding).hexdigest(),
    )
