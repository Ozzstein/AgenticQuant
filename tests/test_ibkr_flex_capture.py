"""Real HTTPX boundary over synthetic XML; no credentials or external calls."""

import json
import socket
import ssl
import stat

import httpx
import pytest

from scripts.fetch_ibkr_flex_report import (
    MAX_BYTES,
    ORIGIN,
    FlexCaptureError,
    fetch_report,
    network_check,
    parse_xml,
    private_directory,
    transport_reason,
)

TOKEN = "synthetic-private-flex-token"
ACK = b"<FlexStatementResponse><Status>Success</Status><ReferenceCode>98765</ReferenceCode><url>https://hostile.invalid/steal</url></FlexStatementResponse>"
REPORT = b'<FlexQueryResponse><FlexStatements><FlexStatement accountId="private-account" fromDate="20261001" toDate="20261028"><CashTransactions/></FlexStatement></FlexStatements></FlexQueryResponse>'


def error(code):
    return f"<FlexStatementResponse><Status>Fail</Status><ErrorCode>{code}</ErrorCode></FlexStatementResponse>".encode()


def capture(tmp_path, responses):
    requests, delays = [], []
    directory = private_directory(tmp_path / "private")

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url).split("?")[0] in (ORIGIN + "SendRequest", ORIGIN + "GetStatement")
        assert request.url.params["t"] == TOKEN
        assert request.url.params["v"] == "3"
        return httpx.Response(200, content=responses.pop(0))

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    return directory, client, requests, delays


def test_actual_transport_retains_raw_private_receipts_and_ignores_supplied_url(tmp_path):
    directory, client, requests, delays = capture(tmp_path, [ACK, error("1019"), REPORT])
    with client:
        result = fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    assert [r.url.params["q"] for r in requests] == ["12345", "98765", "98765"]
    assert delays == [7, 7]
    assert result["statements"] == 1 and result["income_admission"] == "not_verified"
    assert (directory / result["report_file"]).read_bytes() == REPORT
    for path in directory.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert TOKEN.encode() not in path.read_bytes()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    receipt = json.loads(next(directory.glob("*.receipt.json")).read_text())
    assert receipt["started_at"] <= receipt["available_at"]
    assert set(receipt) == {
        "operation",
        "started_at",
        "available_at",
        "http_status",
        "raw_file",
        "raw_sha256",
        "query_id_sha256",
        "request_version",
    }


@pytest.mark.parametrize("code", ["1012", "1013", "1014", "1015", "1018"])
def test_generation_rejections_are_archived_without_resubmission(tmp_path, code):
    directory, client, requests, delays = capture(tmp_path, [error(code)])
    with client, pytest.raises(FlexCaptureError, match="ibkr_error_" + code):
        fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    assert len(requests) == 1 and not delays
    assert not (directory / "capture.json").exists()


def test_pending_is_bounded_and_preserves_all_same_reference_receipts(tmp_path):
    directory, client, requests, delays = capture(tmp_path, [ACK, *([error("1019")] * 8)])
    with client, pytest.raises(FlexCaptureError, match="bounded_polling_stopped"):
        fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    assert len(requests) == 9 and len(delays) == 8
    assert len(list(directory.glob("*.receipt.json"))) == 9
    assert not (directory / "capture.json").exists()


def test_resume_reuses_original_reference_without_generation_or_overwrite(tmp_path):
    directory, client, requests, delays = capture(tmp_path, [ACK, *([error("1019")] * 8)])
    with client, pytest.raises(FlexCaptureError, match="bounded_polling_stopped"):
        fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    resumed = private_directory(directory.parent)
    requests.clear()

    def handler(request):
        requests.append(request)
        assert request.url.path.endswith("/GetStatement")
        assert request.url.params["q"] == "98765"
        return httpx.Response(200, content=REPORT)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_report(
            client,
            TOKEN,
            "12345",
            resumed,
            sleep=lambda _: None,
            resume_generation=directory.parent,
        )
    assert len(requests) == 1 and result["statements"] == 1
    assert json.loads((resumed / "resume.json").read_text())["directory"] == str(directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


@pytest.mark.parametrize("change", ["query", "raw", "permissions", "symlink", "complete"])
def test_resume_invalid_source_never_contacts_provider(tmp_path, change):
    directory, client, _, delays = capture(tmp_path, [ACK, *([error("1019")] * 8)])
    with client, pytest.raises(FlexCaptureError):
        fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    ack = directory / "01-SendRequest.xml"
    query = "67890" if change == "query" else "12345"
    if change == "raw":
        ack.write_bytes(ACK + b" ")
    elif change == "permissions":
        ack.chmod(0o644)
    elif change == "symlink":
        target = directory / "original.xml"
        ack.rename(target)
        ack.symlink_to(target)
    elif change == "complete":
        (directory / "capture.json").write_text("{}")
    resumed = private_directory(directory.parent)
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as client:
        with pytest.raises(FlexCaptureError):
            fetch_report(
                client,
                TOKEN,
                query,
                resumed,
                sleep=lambda _: None,
                resume_generation=directory.parent,
            )
    assert calls == []


def test_paper_worker_uses_separate_capture_and_resume_namespace(tmp_path, monkeypatch, capsys):
    import scripts.fetch_ibkr_flex_report as worker

    monkeypatch.setattr(worker.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(worker.sys, "argv", ["fetch_ibkr_flex_report.py", "--paper"])
    monkeypatch.setenv("IBKR_FLEX_TOKEN", TOKEN)
    monkeypatch.setenv("IBKR_FLEX_QUERY_ID", "12345")
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    actual_client = httpx.Client
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, content=ACK if request.url.path.endswith("SendRequest") else REPORT
        )

    monkeypatch.setattr(
        worker.httpx,
        "Client",
        lambda **kwargs: actual_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    actual_fetch = worker.fetch_report
    monkeypatch.setattr(
        worker, "fetch_report", lambda *a, **kw: actual_fetch(*a, sleep=lambda _: None, **kw)
    )
    assert worker.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        worker.Path(result["directory"]).parent
        == tmp_path / ".local/share/quantagentlab/ibkr-flex/paper"
    )
    assert len(calls) == 2
    assert TOKEN not in json.dumps(result)


def test_overflow_never_archives_truncated_success(tmp_path):
    directory, client, requests, delays = capture(tmp_path, [ACK, b"x" * (MAX_BYTES + 1)])
    with client, pytest.raises(FlexCaptureError, match="response_bytes_exceeded"):
        fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    assert len(requests) == 2
    assert len(list(directory.glob("*.receipt.json"))) == 1


@pytest.mark.parametrize(
    "raw",
    [
        b'<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><x>&e;</x>',
        b"<x>\x00</x>",
        b"\xff\xfe<x/>",
    ],
)
def test_unsafe_or_unsupported_xml_refused(raw):
    with pytest.raises(FlexCaptureError):
        parse_xml(raw)


def test_response_echo_cannot_persist_flex_token(tmp_path):
    directory, client, requests, delays = capture(tmp_path, [TOKEN.encode()])
    with client, pytest.raises(FlexCaptureError, match="credential_echo_refused"):
        fetch_report(client, TOKEN, "12345", directory, sleep=delays.append)
    assert not list(directory.iterdir())


def test_private_output_refuses_symlinks_and_shared_permissions(tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(FlexCaptureError, match="permissions"):
        private_directory(public)
    link = tmp_path / "alias"
    link.symlink_to(public)
    with pytest.raises(FlexCaptureError, match="invalid"):
        private_directory(link)


@pytest.mark.parametrize(
    "cause,label",
    [
        (ConnectionResetError(104, "PRIVATE-TOKEN"), "connection_reset"),
        (ssl.SSLCertVerificationError(1, "PRIVATE-TOKEN"), "tls_certificate_verification_failed"),
        (socket.gaierror(-2, "PRIVATE-TOKEN"), "dns_resolution_failed"),
    ],
)
def test_transport_diagnostics_never_echo_exception_message(cause, label):
    exc = httpx.ConnectError("https://broker.invalid/?t=PRIVATE-TOKEN")
    exc.__cause__ = cause
    assert transport_reason(exc) == "transport_" + label


def test_public_network_probe_uses_no_credentials_or_report_routes(monkeypatch):
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "PRIVATE-TOKEN")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/" and not request.url.query
        assert "authorization" not in request.headers
        assert b"PRIVATE-TOKEN" not in request.content
        if request.url.host == "www.1password.dev":
            return httpx.Response(200)
        raise httpx.ConnectError("PRIVATE-TOKEN") from ConnectionResetError(104, "PRIVATE-TOKEN")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = network_check(client)
    assert len(requests) == 3
    assert result == {
        "CONTROL_HTTPS": "HTTP_200",
        "IBKR_WEBSITE": "transport_connection_reset",
        "IBKR_FLEX_HOST": "transport_connection_reset",
    }
