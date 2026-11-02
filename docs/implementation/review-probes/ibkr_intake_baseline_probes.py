"""SYNTHETIC review probes: assert baseline defects, not desired fixed behavior.

No credentials, real accounts, provider calls, or financial capture files.
Race hooks schedule filesystem replacement only; original validation APIs run.
"""

from pathlib import Path

import httpx
import pytest

from scripts import fetch_ibkr_flex_report as worker
from src.data import ibkr_flex as intake

ACK = b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>111</ReferenceCode></FlexStatementResponse>"
PENDING = b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1019</ErrorCode></FlexStatementResponse>"


def pending_capture(base, ack=ACK):
    directory = worker.private_directory(base)
    responses = [ack, *([PENDING] * 8)]
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=responses.pop(0)))
        ) as client,
        pytest.raises(worker.FlexCaptureError, match="bounded_polling_stopped"),
    ):
        worker.fetch_report(
            client, "synthetic-review-token", "999", directory, sleep=lambda _: None
        )
    return directory


def test_review_reader_parent_replacement_bypasses_no_link_check(tmp_path, monkeypatch):
    directory = tmp_path / "capture"
    directory.mkdir(mode=0o700)
    target = directory / "synthetic.txt"
    target.write_bytes(b"synthetic-original")
    target.chmod(0o600)
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    (replacement / target.name).write_bytes(b"synthetic-replacement")
    (replacement / target.name).chmod(0o600)
    original_open = intake.os.open

    def swap_then_open(path, flags, *args, **kwargs):
        if path == target:
            directory.rename(tmp_path / "retained")
            directory.symlink_to(replacement, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(intake.os, "open", swap_then_open)
    assert intake._private_read(target) == b"synthetic-replacement"
    assert directory.is_symlink()


def test_review_resume_leaf_replacement_bypasses_no_link_check(tmp_path, monkeypatch):
    source = pending_capture(tmp_path / "captures")
    ack_path = source / "01-SendRequest.xml"
    replacement = tmp_path / "synthetic-ack.xml"
    replacement.write_bytes(ACK)
    replacement.chmod(0o600)
    original_read = Path.read_bytes

    def swap_then_read(path):
        if path == ack_path:
            path.unlink()
            path.symlink_to(replacement)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", swap_then_read)
    reference, _ = worker.latest_generation(source.parent, "999")
    assert reference == "111"
    assert ack_path.is_symlink()


def test_review_latest_generation_uses_random_uuid_order_within_second(tmp_path):
    base = tmp_path / "captures"
    older = pending_capture(base)
    older.rename(base / ("20260101T000000Z-" + "f" * 32))
    newer = pending_capture(base, ACK.replace(b">111<", b">222<"))
    newer.rename(base / ("20260101T000000Z-" + "0" * 32))
    # Names model two generations in the same second; receipts retain true order.
    reference, _ = worker.latest_generation(base, "999")
    assert reference == "111"  # Older acknowledgement wins despite later receipt.


@pytest.mark.parametrize(
    "raw,label",
    [
        (b"x" * (intake.MAX_BYTES + 1), "report_size_invalid"),
        (b"<x>" * 10 + b"</x>" * 10, "xml_structure_limit"),
        (b"<x>" + b"<y/>" * intake.MAX_NODES + b"</x>", "xml_structure_limit"),
    ],
)
def test_review_reader_rejects_bounded_synthetic_resource_overflows(raw, label):
    with pytest.raises(intake.FlexIntakeError, match=label):
        intake.parse_flex_report(raw)
