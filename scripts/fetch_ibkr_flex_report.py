"""Private diagnostic capture only; never admits income or places orders."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import re
import socket
import ssl
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import httpx

# The launcher executes this file directly, with a sanitized environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import private_files  # noqa: E402
from src.data.ibkr_flex import FlexIntakeError, _instant, _json  # noqa: E402

ORIGIN = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/"
MAX_BYTES = 8 * 1024 * 1024
PENDING = {"1004", "1005", "1006", "1007", "1008", "1009", "1019", "1021"}


class FlexCaptureError(ValueError):
    """Only code-owned, public status labels may leave this module."""


def transport_reason(exc):
    """Classify exception types/codes, never authenticated URLs or messages."""
    cause, seen = exc, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, ssl.SSLCertVerificationError):
            return "transport_tls_certificate_verification_failed"
        if isinstance(cause, socket.gaierror):
            return "transport_dns_resolution_failed"
        if (
            isinstance(cause, ConnectionResetError)
            or getattr(cause, "errno", None) == errno.ECONNRESET
        ):
            return "transport_connection_reset"
        cause = cause.__cause__ or cause.__context__
    for kind, label in (
        (httpx.ConnectTimeout, "connect_timeout"),
        (httpx.ReadTimeout, "read_timeout"),
        (httpx.ProxyError, "proxy_error"),
        (httpx.ConnectError, "connect_error"),
        (httpx.RemoteProtocolError, "remote_protocol_error"),
    ):
        if isinstance(exc, kind):
            return "transport_" + label
    return "transport_failed_no_retry"


def network_check(client):
    """Fixed public probes with no query parameters, tokens or report requests."""
    result = {}
    for label, url in (
        ("CONTROL_HTTPS", "https://www.1password.dev/"),
        ("IBKR_WEBSITE", "https://www.interactivebrokers.com/"),
        ("IBKR_FLEX_HOST", "https://ndcdyn.interactivebrokers.com/"),
    ):
        try:
            with client.stream("GET", url, timeout=8) as response:
                result[label] = "HTTP_" + str(response.status_code)
        except httpx.HTTPError as exc:
            result[label] = transport_reason(exc)
    return result


def private_directory(base: Path) -> Path:
    try:
        with private_files.private_directory(base, create=True) as parent:
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
            os.mkdir(name, 0o700, dir_fd=parent)
            with private_files.private_child(parent, name):
                pass
        return base / name
    except private_files.PrivateFileError as exc:
        code = "permissions" if str(exc) == "private_source_permissions" else "invalid"
        raise FlexCaptureError("private_directory_" + code) from None
    except OSError:
        raise FlexCaptureError("private_directory_invalid") from None


def write_private(path: Path, data: bytes):
    try:
        private_files.write_private(path, data)
    except (OSError, private_files.PrivateFileError):
        raise FlexCaptureError("capture_unavailable") from None


def parse_xml(raw: bytes):
    try:
        text = raw.decode("utf-8-sig")
        if "\x00" in text or re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
            raise FlexCaptureError("unsupported_xml")
        return ET.fromstring(text)
    except (UnicodeError, ET.ParseError):
        raise FlexCaptureError("invalid_xml_report") from None


def _error(root):
    code = root.findtext("ErrorCode")
    if code is not None:
        if not re.fullmatch(r"[0-9]{4}", code):
            raise FlexCaptureError("invalid_provider_error")
        return code
    return None


def latest_generation(base: Path, query_id: str):
    """Read the latest original acknowledgement, without exposing its reference."""
    try:
        candidates = []
        with private_files.private_directory(base) as parent:
            for name in os.listdir(parent):
                if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}", name):
                    continue
                with private_files.private_child(parent, name) as source:
                    # Resume-only directories have no original SendRequest.
                    names = os.listdir(source)
                    if "01-SendRequest.xml" not in names:
                        continue
                    raw = private_files.read_at(source, "01-SendRequest.xml", MAX_BYTES)
                    receipt = _json(
                        private_files.read_at(source, "01-SendRequest.xml.receipt.json", 64 * 1024)
                    )
                    if (
                        receipt.get("operation") != "SendRequest"
                        or receipt.get("http_status") != 200
                        or receipt.get("raw_file") != "01-SendRequest.xml"
                        or receipt.get("request_version") != "3"
                        or receipt.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
                        or receipt.get("query_id_sha256")
                        != hashlib.sha256(query_id.encode()).hexdigest()
                    ):
                        raise ValueError
                    started = _instant(receipt.get("started_at"))
                    available = _instant(receipt.get("available_at"))
                    if started > available:
                        raise ValueError
                    root = parse_xml(raw)
                    reference = root.findtext("ReferenceCode", "")
                    if (
                        root.tag != "FlexStatementResponse"
                        or root.findtext("Status") != "Success"
                        or _error(root)
                        or not re.fullmatch(r"[0-9]+", reference)
                    ):
                        raise ValueError
                    candidates.append(
                        (
                            started,
                            available,
                            name,
                            reference,
                            receipt["raw_sha256"],
                            "capture.json" in names,
                        )
                    )
        if not candidates:
            raise ValueError
        candidates.sort(key=lambda candidate: candidate[0])
        # Require non-overlapping, strictly ordered original request intervals.
        # UUIDs, directory timestamps and mtimes carry no acquisition ordering.
        for older, newer in zip(candidates, candidates[1:]):
            if older[0] == newer[0] or older[1] >= newer[0]:
                raise ValueError
        _, _, name, reference, digest, completed = candidates[-1]
        if completed:
            raise FlexCaptureError("resume_latest_already_downloaded")
        return reference, {
            "directory": str(base / name),
            "acknowledgement_sha256": digest,
        }
    except FlexCaptureError:
        raise
    except (OSError, ValueError, KeyError, TypeError, FlexIntakeError):
        raise FlexCaptureError("resume_source_unavailable_or_invalid") from None


def fetch_report(
    client,
    token,
    query_id,
    directory,
    *,
    sleep=time.sleep,
    monotonic=time.monotonic,
    resume_generation=None,
):
    """One generation request, at most eight same-reference retrieval attempts."""
    if not re.fullmatch(r"[0-9]+", query_id) or not token.strip() or token.startswith("op://"):
        raise FlexCaptureError("credentials_invalid")
    deadline = monotonic() + 120
    receipts = []

    def request(operation, identity):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise FlexCaptureError("deadline_exceeded")
        started = datetime.now(timezone.utc).isoformat()
        try:
            with client.stream(
                "GET",
                ORIGIN + operation,
                params={"t": token, "q": identity, "v": "3"},
                timeout=min(15, remaining),
            ) as response:
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if monotonic() > deadline:
                        raise FlexCaptureError("deadline_exceeded")
                    if len(raw) + len(chunk) > MAX_BYTES:
                        raise FlexCaptureError("response_bytes_exceeded")
                    raw.extend(chunk)
                status = response.status_code
        except httpx.HTTPError as exc:
            # Exceptions contain authenticated request URLs. Never relay them.
            raise FlexCaptureError(transport_reason(exc)) from None
        raw = bytes(raw)
        if token.encode() in raw:
            raise FlexCaptureError("credential_echo_refused")
        name = f"{len(receipts) + 1:02d}-{operation}.xml"
        write_private(directory / name, raw)
        receipt = {
            "operation": operation,
            "started_at": started,
            "available_at": datetime.now(timezone.utc).isoformat(),
            "http_status": status,
            "raw_file": name,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "query_id_sha256": hashlib.sha256(query_id.encode()).hexdigest(),
            "request_version": "3",
        }
        write_private(
            directory / (name + ".receipt.json"), json.dumps(receipt, sort_keys=True).encode()
        )
        receipts.append(receipt)
        if status != 200:
            raise FlexCaptureError("http_status_" + str(status))
        return parse_xml(raw), name

    if resume_generation is not None:
        reference, source = latest_generation(resume_generation, query_id)
        write_private(directory / "resume.json", json.dumps(source, sort_keys=True).encode())
    else:
        root, _ = request("SendRequest", query_id)
        code = _error(root)
        if code:
            raise FlexCaptureError("ibkr_error_" + code)
        reference = root.findtext("ReferenceCode", "")
        if (
            root.tag != "FlexStatementResponse"
            or root.findtext("Status") != "Success"
            or not re.fullmatch(r"[0-9]+", reference)
        ):
            raise FlexCaptureError("generation_response_invalid")
    # Use the documented fixed endpoint, never a server-supplied redirect/URL.
    for _ in range(8):
        if monotonic() + 7 >= deadline:
            raise FlexCaptureError("deadline_exceeded")
        sleep(7)  # <=9 requests total, spaced below IBKR's 10/minute/token ceiling.
        root, name = request("GetStatement", reference)
        code = _error(root)
        if code in PENDING:
            continue
        if code:
            raise FlexCaptureError("ibkr_error_" + code)
        if root.tag != "FlexQueryResponse":
            raise FlexCaptureError("xml_activity_report_required")
        statements = root.findall("./FlexStatements/FlexStatement")
        if not statements:
            raise FlexCaptureError("statement_scope_missing")
        result = {
            "status": "downloaded",
            "directory": str(directory),
            "report_file": name,
            "statements": len(statements),
            "requests": len(receipts),
            "income_admission": "not_verified",
        }
        write_private(directory / "capture.json", json.dumps(result, sort_keys=True).encode())
        return result
    raise FlexCaptureError("report_not_ready_bounded_polling_stopped")


def main():
    # Authenticated Flex URLs have the token in their query. Suppress HTTP logs.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).disabled = True
    if sys.argv[1:] == ["--network-check"]:
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            for label, value in network_check(client).items():
                print(label + ": " + value)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    try:
        if "OP_SERVICE_ACCOUNT_TOKEN" in os.environ:
            raise FlexCaptureError("service_account_not_isolated")
        base = Path.home() / ".local/share/quantagentlab/ibkr-flex"
        directory = private_directory(base / "paper" if args.paper else base)
        with httpx.Client(
            trust_env=False,
            follow_redirects=False,
            headers={"User-Agent": "QuantAgentLab-Flex/0.1"},
        ) as client:
            result = fetch_report(
                client,
                os.environ.get("IBKR_FLEX_TOKEN", ""),
                os.environ.get("IBKR_FLEX_QUERY_ID", ""),
                directory,
                resume_generation=directory.parent if args.resume_latest else None,
            )
    except FlexCaptureError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"status": "failed", "reason": "capture_unavailable"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
