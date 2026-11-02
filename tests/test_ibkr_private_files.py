"""Synthetic regressions for the three review findings and private IO boundaries."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fetch_ibkr_flex_report as worker
from src.data import ibkr_flex as intake
from src.data import private_files as storage

ACK = b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>111</ReferenceCode></FlexStatementResponse>"


@pytest.fixture(autouse=True)
def no_sockets(socket_disabled):
    pass


def private_file(directory, name="raw", data=b"original"):
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def generation(base, suffix, start, end, reference="111"):
    directory = base / ("20260101T000000Z-" + suffix * 32)
    directory.mkdir(mode=0o700, parents=True)
    raw = ACK.replace(b">111<", f">{reference}<".encode())
    private_file(directory, "01-SendRequest.xml", raw)
    receipt = dict(
        operation="SendRequest",
        http_status=200,
        raw_file="01-SendRequest.xml",
        request_version="3",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        query_id_sha256=hashlib.sha256(b"999").hexdigest(),
        started_at="2026-01-01T00:00:00." + start + "+00:00",
        available_at="2026-01-01T00:00:00." + end + "+00:00",
    )
    private_file(directory, "01-SendRequest.xml.receipt.json", json.dumps(receipt).encode())
    return directory


def test_review_parent_swap_at_actual_leaf_open_keeps_original(tmp_path, monkeypatch):
    path = private_file(tmp_path / "capture")
    replacement = private_file(tmp_path / "replacement", data=b"replacement")
    original_open = os.open
    fired = []

    def swap(name, flags, *args, **kwargs):
        if name == path.name and kwargs.get("dir_fd") is not None:
            fired.append(True)
            path.parent.rename(tmp_path / "retained")
            path.parent.symlink_to(replacement.parent, target_is_directory=True)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", swap)
    assert intake._private_read(path) == b"original"
    assert fired == [True] and path.parent.is_symlink()


def test_review_resume_leaf_swap_at_actual_open_refused(tmp_path, monkeypatch):
    base = tmp_path / "captures"
    base.mkdir(mode=0o700)
    source = generation(base, "a", "100000", "200000")
    replacement = private_file(tmp_path / "outside", data=ACK)
    original_open = os.open
    fired = []

    def swap(name, flags, *args, **kwargs):
        if name == "01-SendRequest.xml" and kwargs.get("dir_fd") is not None:
            fired.append(True)
            (source / name).unlink()
            (source / name).symlink_to(replacement)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", swap)
    with pytest.raises(worker.FlexCaptureError, match="resume_source_unavailable_or_invalid"):
        worker.latest_generation(base, "999")
    assert fired == [True]


def test_review_same_second_order_uses_receipts_not_uuid_or_mtime(tmp_path):
    base = tmp_path / "captures"
    base.mkdir(mode=0o700)
    older = generation(base, "f", "100000", "200000")
    newer = generation(base, "0", "300000", "400000", "222")
    os.utime(older, (2000000000, 2000000000))
    os.utime(newer, (1, 1))
    reference, source = worker.latest_generation(base, "999")
    assert reference == "222" and source["directory"] == str(newer)


@pytest.mark.parametrize(
    "start,end",
    [("100000", "200000"), ("150000", "250000"), ("200000", "300000"), ("400000", "300000")],
)
def test_equal_overlapping_or_backwards_clocks_refused(tmp_path, start, end):
    base = tmp_path / "captures"
    base.mkdir(mode=0o700)
    generation(base, "a", "100000", "200000")
    generation(base, "b", start, end)
    with pytest.raises(worker.FlexCaptureError, match="resume_source_unavailable_or_invalid"):
        worker.latest_generation(base, "999")


@pytest.mark.parametrize(
    "change",
    [
        "file_mode",
        "parent_mode",
        "hardlink",
        "fifo",
        "directory",
        "leaf_link",
        "parent_link",
        "oversize",
    ],
)
def test_private_read_rejects_unsafe_objects(tmp_path, change):
    path = private_file(tmp_path / "private")
    if change == "file_mode":
        path.chmod(0o640)
    elif change == "parent_mode":
        path.parent.chmod(0o750)
    elif change == "hardlink":
        os.link(path, tmp_path / "alias")
    elif change in ("fifo", "directory", "leaf_link"):
        path.unlink()
        if change == "fifo":
            os.mkfifo(path, 0o600)
        elif change == "directory":
            path.mkdir(mode=0o700)
        else:
            target = private_file(tmp_path / "target")
            path.symlink_to(target)
    elif change == "parent_link":
        path.parent.rename(tmp_path / "retained")
        path.parent.symlink_to(tmp_path / "retained", target_is_directory=True)
    with pytest.raises(intake.FlexIntakeError):
        intake._private_read(path, 2 if change == "oversize" else 100)


@pytest.mark.parametrize("target", ["file", "parent"])
def test_exact_opened_object_owner_checked(tmp_path, monkeypatch, target):
    path = private_file(tmp_path / "private")
    original_fstat = os.fstat

    def foreign(fd):
        info = original_fstat(fd)
        if (target == "parent") == os.path.isdir(f"/proc/self/fd/{fd}"):
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_uid=os.getuid() + 1,
                st_nlink=info.st_nlink,
                st_size=info.st_size,
            )
        return info

    monkeypatch.setattr(storage.os, "fstat", foreign)
    with pytest.raises(intake.FlexIntakeError):
        intake._private_read(path)


def test_write_parent_swap_stays_in_opened_directory(tmp_path, monkeypatch):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    original_open = os.open
    fired = []

    def swap(name, flags, *args, **kwargs):
        if name == "new" and kwargs.get("dir_fd") is not None:
            fired.append(True)
            directory.rename(tmp_path / "retained")
            directory.symlink_to(replacement, target_is_directory=True)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", swap)
    worker.write_private(directory / "new", b"synthetic-private")
    assert fired == [True]
    assert not list(replacement.iterdir())
    assert (tmp_path / "retained/new").read_bytes() == b"synthetic-private"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "shared_parent", "fifo"])
def test_writer_never_overwrites_aliases(tmp_path, kind):
    path = private_file(tmp_path / "private")
    outside = private_file(tmp_path / "outside", data=b"untouched")
    path.unlink()
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.parent.chmod(0o755)
    with pytest.raises(worker.FlexCaptureError):
        worker.write_private(path, b"synthetic-private")
    assert outside.read_bytes() == b"untouched"


def test_creation_swap_before_directory_open_refused(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    original_open = os.open
    fired = []

    def swap(name, flags, *args, **kwargs):
        if name == "base" and kwargs.get("dir_fd") is not None:
            fired.append(True)
            base.rename(tmp_path / "retained")
            base.symlink_to(outside, target_is_directory=True)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", swap)
    with pytest.raises(worker.FlexCaptureError):
        worker.private_directory(base)
    assert fired == [True] and not list(outside.iterdir())


@pytest.mark.parametrize("path", [Path("relative/raw"), Path("/tmp/../raw")])
def test_invalid_paths_fail_closed(path):
    with pytest.raises(intake.FlexIntakeError):
        intake._private_read(path)


def test_unsupported_capability_fails_closed(tmp_path, monkeypatch):
    path = private_file(tmp_path / "private")
    monkeypatch.setattr(storage, "_DIR_FD_SUPPORTED", False)
    with pytest.raises(intake.FlexIntakeError):
        intake._private_read(path)
    with pytest.raises(worker.FlexCaptureError):
        worker.write_private(path.parent / "new", b"private")


def test_descriptor_cleanup_on_success_and_failures(tmp_path, monkeypatch):
    path = private_file(tmp_path / "private")
    before = set(os.listdir("/proc/self/fd"))
    for _ in range(20):
        assert intake._private_read(path) == b"original"
        with pytest.raises(intake.FlexIntakeError):
            intake._private_read(path.parent / "absent")
        with pytest.raises(intake.FlexIntakeError):
            intake._private_read(path, 2)
        with pytest.raises(worker.FlexCaptureError):
            worker.write_private(path, b"refuse overwrite")
    worker.write_private(path.parent / "new", b"ok")
    monkeypatch.setattr(storage.os, "write", lambda *a: (_ for _ in ()).throw(OSError()))
    with pytest.raises(worker.FlexCaptureError):
        worker.write_private(path.parent / "failed", b"private")
    assert set(os.listdir("/proc/self/fd")) == before


def test_bounded_read_detects_growth_after_initial_fstat(tmp_path, monkeypatch):
    path = private_file(tmp_path / "private", data=b"small")
    original_read = os.read
    fired = []

    def grow(fd, count):
        if not fired:
            fired.append(True)
            path.write_bytes(b"x" * 101)
        return original_read(fd, count)

    monkeypatch.setattr(storage.os, "read", grow)
    with pytest.raises(intake.FlexIntakeError):
        intake._private_read(path, 100)
    assert fired == [True]


@pytest.mark.parametrize("failure", ["read", "fstat", "fsync"])
def test_descriptor_cleanup_on_io_errors(tmp_path, monkeypatch, failure):
    path = private_file(tmp_path / "private")
    before = set(os.listdir("/proc/self/fd"))

    def fail(*args):
        raise OSError("synthetic-private-error")

    monkeypatch.setattr(storage.os, failure, fail)
    if failure == "fsync":
        with pytest.raises(worker.FlexCaptureError, match="capture_unavailable"):
            worker.write_private(path.parent / "new", b"synthetic")
    else:
        with pytest.raises(intake.FlexIntakeError, match="private_source_unavailable"):
            intake._private_read(path)
    assert set(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize(
    "change",
    [
        "missing_clock",
        "naive_clock",
        "duplicate_json",
        "hash",
        "receipt_fifo",
        "parent_mode",
        "completed_latest",
    ],
)
def test_ordering_requires_verified_original_receipts(tmp_path, change):
    base = tmp_path / "captures"
    base.mkdir(mode=0o700)
    source = generation(base, "a", "100000", "200000")
    path = source / "01-SendRequest.xml.receipt.json"
    receipt = json.loads(path.read_bytes())
    if change == "missing_clock":
        del receipt["started_at"]
    elif change == "naive_clock":
        receipt["started_at"] = "2026-01-01T00:00:00"
    elif change == "hash":
        receipt["raw_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    if change == "duplicate_json":
        path.write_text(path.read_text()[:-1] + ', "started_at": "invalid"}')
    elif change == "receipt_fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif change == "parent_mode":
        source.chmod(0o755)
    elif change == "completed_latest":
        private_file(source, "capture.json", b"{}")
    with pytest.raises(worker.FlexCaptureError):
        worker.latest_generation(base, "999")


def test_review_original_fetch_probe_selects_newer_retained_send_receipt(tmp_path):
    import httpx

    base = tmp_path / "captures"
    pending = b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1019</ErrorCode></FlexStatementResponse>"
    sources = []
    for reference, suffix in ((b"111", "f"), (b"222", "0")):
        source = worker.private_directory(base)
        responses = [ACK.replace(b">111<", b">" + reference + b"<"), *([pending] * 8)]
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, content=responses.pop(0))
                )
            ) as client,
            pytest.raises(worker.FlexCaptureError, match="bounded_polling_stopped"),
        ):
            worker.fetch_report(client, "synthetic-token", "999", source, sleep=lambda _: None)
        renamed = base / ("20260101T000000Z-" + suffix * 32)
        source.rename(renamed)
        sources.append(renamed)
    reference, metadata = worker.latest_generation(base, "999")
    assert reference == "222" and metadata["directory"] == str(sources[-1])
