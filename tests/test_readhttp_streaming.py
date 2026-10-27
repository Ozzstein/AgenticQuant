"""Bounded actual HTTPX streams and a constrained loopback HTTP boundary."""

import gzip
import socket
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from src.data.adapters.base import ProviderError, ReadHTTP
from src.data.adapters.context import CoinGeckoAdapter
from src.data.evidence.store import EvidenceStore
from src.domain.common import RunContext


class ResponseStream(httpx.SyncByteStream):
    """Test-only network chunks, without pre-buffering a Response.content body."""

    def __init__(self, chunks):
        self.chunks, self.reads, self.closed = chunks, 0, False

    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def transport(responses):
    calls = []

    def get(request):
        calls.append(request)
        status, headers, stream = responses[len(calls) - 1]
        return httpx.Response(status, headers=headers, stream=stream)

    return httpx.Client(transport=httpx.MockTransport(get)), calls


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1024", None])
def test_decoded_limit_requires_explicit_positive_integer(limit):
    with pytest.raises(ValueError, match="maximum_response_bytes"):
        ReadHTTP("source", "https://example.org", maximum_response_bytes=limit)


@pytest.mark.parametrize("chunks", [[], [b"abcd"], [b"a", b"bc", b"d"]])
def test_complete_body_within_or_at_limit_is_unchanged_and_closed(chunks):
    stream = ResponseStream(chunks)
    client, calls = transport([(200, {}, stream)])
    with client:
        http = ReadHTTP(
            "source", "https://example.org", client=client, maximum_response_bytes=4, min_interval=0
        )
        assert http.get(
            "/actual", params={"symbol": "BTCUSD"}, headers={"X-Test": "value"}
        ) == b"".join(chunks)
        assert stream.closed and http.requests == len(calls) == 1
        assert calls[0].method == "GET" and calls[0].url.params["symbol"] == "BTCUSD"
        assert calls[0].headers["X-Test"] == "value"


def test_overflow_stops_before_tail_closes_without_retry_and_releases_client():
    oversized = ResponseStream([b"ab", b"cde", AssertionError("overflow must stop before tail")])
    following = ResponseStream([b"safe"])
    client, calls = transport([(200, {"Content-Length": "1"}, oversized), (200, {}, following)])
    with client:
        http = ReadHTTP(
            "source",
            "https://example.org",
            client=client,
            maximum_response_bytes=4,
            min_interval=0,
            attempts=3,
        )
        with pytest.raises(ProviderError, match="response_bytes_exceeded") as error:
            http.get("/actual", headers={"X-Secret": "fixture-only-not-a-key"})
        assert "fixture-only-not-a-key" not in str(error.value)
        assert oversized.closed and oversized.reads == 2
        assert http.requests == len(calls) == 1
        assert http.get("/next") == b"safe" and following.closed


def test_capture_never_archives_oversized_partial_body_as_evidence(tmp_path):
    stream = ResponseStream([b'{"data":' + b" " * 128, AssertionError("body rejected before tail")])
    client, _ = transport([(200, {}, stream)])
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    store = EvidenceStore(tmp_path / "bounded.sqlite")
    context = RunContext(
        run_id="bounded-capture", mode="paper", as_of=now, config_hash="0" * 64, code_version="test"
    )
    with client:
        http = ReadHTTP(
            "coingecko",
            "https://api.coingecko.com/api/v3",
            client=client,
            maximum_response_bytes=64,
            min_interval=0,
        )
        adapter = CoinGeckoAdapter(store, http=http, clock=lambda: now)
        with pytest.raises(ProviderError, match="response_bytes_exceeded"):
            adapter.current_dominance(context)
        assert stream.closed and stream.reads == 1
        assert not store.query(as_of=now).records


@pytest.mark.parametrize("overflow", [False, True])
def test_limit_counts_decoded_gzip_bytes_not_content_length(overflow):
    body = b"a" * (1025 if overflow else 1024)
    wire = gzip.compress(body)
    assert len(wire) < 1024
    stream = ResponseStream([wire[:7], wire[7:]])
    client, _ = transport(
        [(200, {"Content-Encoding": "gzip", "Content-Length": str(len(wire))}, stream)]
    )
    with client:
        http = ReadHTTP(
            "source",
            "https://example.org",
            client=client,
            maximum_response_bytes=1024,
            min_interval=0,
        )
        if overflow:
            with pytest.raises(ProviderError, match="response_bytes_exceeded"):
                http.get("/compressed")
        else:
            assert http.get("/compressed") == body
        assert stream.closed


@pytest.mark.parametrize("overflow", [False, True])
def test_documented_native_400_remains_exact_and_subject_to_same_cap(overflow):
    body = b'{"code":-2043,"msg":"Reference price not set"}'
    stream = ResponseStream([body])
    client, calls = transport([(400, {}, stream)])
    with client:
        http = ReadHTTP(
            "binance",
            "https://api.binance.com",
            client=client,
            maximum_response_bytes=len(body) - int(overflow),
            min_interval=0,
        )
        if overflow:
            with pytest.raises(ProviderError, match="response_bytes_exceeded"):
                http.get("/api/v3/referencePrice", allowed_error_statuses=(400,))
        else:
            assert http.get("/api/v3/referencePrice", allowed_error_statuses=(400,)) == body
        assert stream.closed and len(calls) == 1


@pytest.mark.parametrize(
    "status,code", [(400, "http_failure"), (401, "credentials"), (403, "entitlement")]
)
def test_rejected_status_closes_without_consuming_arbitrary_error_body(status, code):
    stream = ResponseStream([AssertionError("rejected body need not be read")])
    client, _ = transport([(status, {}, stream)])
    with client:
        http = ReadHTTP(
            "source", "https://example.org", client=client, maximum_response_bytes=1, min_interval=0
        )
        with pytest.raises(ProviderError, match=code):
            http.get("/failure")
        assert stream.closed and stream.reads == 0


@pytest.mark.parametrize("status,code", [(429, "rate_limited"), (503, "provider_unavailable")])
def test_retriable_status_responses_close_at_each_bounded_attempt(status, code):
    streams = [ResponseStream([AssertionError("status body not needed")]) for _ in range(3)]
    client, calls = transport([(status, {}, s) for s in streams])
    sleeps = []
    with client:
        http = ReadHTTP(
            "source",
            "https://example.org",
            client=client,
            attempts=3,
            min_interval=0,
            sleep=sleeps.append,
        )
        with pytest.raises(ProviderError, match=code):
            http.get("/busy")
        assert len(calls) == http.requests == 3 and sleeps == [1, 2]
        assert all(s.closed and s.reads == 0 for s in streams)


@pytest.mark.parametrize("recover", [False, True])
def test_partial_stream_failure_closes_and_never_combines_attempt_bodies(recover):
    failed = ResponseStream([b"partial", httpx.ReadError("fixture-private-error")])
    following = ResponseStream([b"complete"])
    client, calls = transport([(200, {}, failed), (200, {}, following)])
    with client:
        http = ReadHTTP(
            "source",
            "https://example.org",
            client=client,
            attempts=2 if recover else 1,
            min_interval=0,
            sleep=lambda _: None,
        )
        if recover:
            assert http.get("/actual") == b"complete" and following.closed
            assert len(calls) == 2
        else:
            with pytest.raises(ProviderError, match="transport_unavailable") as error:
                http.get("/actual")
            assert "fixture-private-error" not in str(error.value)
        assert failed.closed


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_actual_http_boundary_closes_unfinished_overflow_and_caps_decoded_bytes(pytestconfig):
    # No vendor or external network permission: the only allowed TCP peer is
    # this ephemeral loopback server. Prove the guard before opening it.
    if pytestconfig.pluginmanager.hasplugin("socket"):
        import pytest_socket

        with socket.socket() as blocked:
            with pytest.warns(UserWarning, match="203.0.113.1"):
                with pytest.raises(pytest_socket.SocketConnectBlockedError):
                    blocked.connect(("203.0.113.1", 443))
    disconnected = threading.Event()
    native = b'{"code":-2043,"msg":"Reference price not set"}'

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/unfinished":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"800\r\n" + b"a" * 2048 + b"\r\n")
                self.wfile.flush()
                # No end-of-response: an eager client waits here. A bounded
                # streaming client rejects the first chunk and closes now.
                self.connection.settimeout(2)
                try:
                    if self.connection.recv(1) == b"":
                        disconnected.set()
                except (ConnectionResetError, BrokenPipeError):
                    disconnected.set()
                except TimeoutError:
                    pass
                self.close_connection = True
                return
            body = gzip.compress(b"a" * 2048) if self.path == "/gzip" else native
            self.send_response(200 if self.path == "/gzip" else 400)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            if self.path == "/gzip":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False) as client:
            http = ReadHTTP(
                "local-fixture",
                f"http://127.0.0.1:{server.server_port}",
                client=client,
                maximum_response_bytes=1024,
                attempts=1,
                timeout=3,
                min_interval=0,
            )
            with pytest.raises(ProviderError, match="response_bytes_exceeded"):
                http.get("/unfinished")
            assert disconnected.wait(2)
            with pytest.raises(ProviderError, match="response_bytes_exceeded"):
                http.get("/gzip")
            assert http.get("/native", allowed_error_statuses=(400,)) == native
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
