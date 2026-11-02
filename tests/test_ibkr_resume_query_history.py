"""Desired behavior for retained query history; synthetic files, no network."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import fetch_ibkr_flex_report as worker
from src.data import ibkr_flex, private_files


def original(base, suffix, second, query, reference):
    directory = base / ("20260101T000000Z-" + suffix * 32)
    directory.mkdir(mode=0o700)
    raw = (
        "<FlexStatementResponse><Status>Success</Status>"
        f"<ReferenceCode>{reference}</ReferenceCode></FlexStatementResponse>"
    ).encode()
    receipt = {
        "operation": "SendRequest",
        "http_status": 200,
        "raw_file": "01-SendRequest.xml",
        "request_version": "3",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "query_id_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "started_at": f"2026-01-01T00:00:{second:02d}.100000+00:00",
        "available_at": f"2026-01-01T00:00:{second:02d}.200000+00:00",
    }
    worker.write_private(directory / "01-SendRequest.xml", raw)
    worker.write_private(
        directory / "01-SendRequest.xml.receipt.json", json.dumps(receipt).encode()
    )
    return directory


@pytest.mark.parametrize("older_query", ["888", "999"])
def test_resume_current_latest_with_valid_retained_history(tmp_path, socket_disabled, older_query):
    original(tmp_path, "f", 1, older_query, "111")
    latest = original(tmp_path, "0", 2, "999", "222")
    reference, source = worker.latest_generation(tmp_path, "999")
    assert reference == "222"
    assert source["directory"] == str(latest)


def test_newest_other_query_must_not_fall_back(tmp_path, socket_disabled):
    original(tmp_path, "f", 1, "999", "111")
    original(tmp_path, "0", 2, "888", "222")
    with pytest.raises(worker.FlexCaptureError, match="resume_source_unavailable_or_invalid"):
        worker.latest_generation(tmp_path, "999")


def test_corrupt_other_query_history_still_fails_closed(tmp_path, socket_disabled):
    older = original(tmp_path, "f", 1, "888", "111")
    original(tmp_path, "0", 2, "999", "222")
    (older / "01-SendRequest.xml").write_bytes(b"corrupt-synthetic-ack")
    with pytest.raises(worker.FlexCaptureError, match="resume_source_unavailable_or_invalid"):
        worker.latest_generation(tmp_path, "999")


@pytest.mark.parametrize("invalid_hash", ["", "x" * 64, "a" * 63, None, 123, ["a" * 64]])
def test_older_query_hash_must_have_valid_syntax(tmp_path, socket_disabled, invalid_hash):
    older = original(tmp_path, "f", 1, "888", "111")
    original(tmp_path, "0", 2, "999", "222")
    path = older / "01-SendRequest.xml.receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["query_id_sha256"] = invalid_hash
    path.write_text(json.dumps(receipt))
    with pytest.raises(worker.FlexCaptureError, match="resume_source_unavailable_or_invalid"):
        worker.latest_generation(tmp_path, "999")


def test_review_imports_are_local():
    checkout = Path(__file__).resolve().parents[1]
    for module in (worker, ibkr_flex, private_files):
        assert Path(module.__file__).resolve().is_relative_to(checkout)
