"""``freeai serve``: a local OpenAI-compatible endpoint backed by :class:`~freeai.FreeAI`.

Any tool that speaks the OpenAI API (SDKs, IDE plugins, agents) can point its base URL at
``http://127.0.0.1:8787/v1`` and get free-model failover without code changes.

Threat model: the proxy spends the quota of every API key it was started with, so whoever can
reach it can use those keys. It binds to loopback by default and refuses any other address unless
a bearer token is set (compared in constant time). Clients never send provider keys; key strings
are redacted from error bodies. Request bodies are capped at 16 MiB. Configure OpenAI SDK clients
with ``max_retries=0``: the proxy already fails over, and its 429 means every free model is busy.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import math
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ._http import logger
from .client import FreeAI
from .compat import chunk_dict, completion_dict, model_dict, parse_model
from .errors import ExhaustedError, FreeAIError, OutputLimitError, StreamInterrupted
from .types import StreamChunk

__all__ = ["MAX_BODY_BYTES", "ProxyServer", "serve"]

MAX_BODY_BYTES = 16 * 1024 * 1024


class ProxyServer(ThreadingHTTPServer):
    """Threaded HTTP server; one :class:`FreeAI` instance (thread-safe) serves every request."""

    daemon_threads = True

    def __init__(self, ai: FreeAI, address: tuple[str, int], *, token: str | None = None) -> None:
        if not token and not _is_loopback(address[0]):
            raise ValueError(f"refusing to listen on {address[0]} without a token: anyone could spend your quota")
        self.ai = ai
        self.token = token or None
        self._token_bytes = None if self.token is None else self.token.encode()
        self._secrets = [p.api_key for p in ai.providers if p.api_key and len(p.api_key) >= 8]
        super().__init__(address, _Handler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        return text


class _Handler(BaseHTTPRequestHandler):
    server: ProxyServer
    protocol_version = "HTTP/1.1"
    _started = False
    server_version = "freeai"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("proxy %s - " + format, self.address_string(), *args)

    # -- routes ----------------------------------------------------------------------------

    # One handler instance serves every request of a keep-alive connection, so per-request state
    # is reset here, and any early exit that leaves a body unread closes the connection.

    def do_GET(self) -> None:
        self._started = False
        if not self._authorized():
            return
        path = self._path()
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [model_dict(m) for m in self.server.ai.models()]})
        elif path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok", "cooldowns": self.server.ai.status()})
        else:
            self._error(404, f"no route for GET {path}", "invalid_request_error", "not_found")

    def do_POST(self) -> None:
        self._started = False
        if not self._authorized():
            return
        path = self._path()
        if path != "/v1/chat/completions":
            self.close_connection = True  # the body was not read
            self._error(404, f"no route for POST {path}", "invalid_request_error", "not_found")
            return
        body = self._read_json()
        if body is None:
            return
        messages = body.pop("messages", None)
        model = body.pop("model", None)
        stream = body.pop("stream", False)
        if not isinstance(messages, list) or not messages:
            self._error(400, "messages must be a non-empty list", "invalid_request_error", "invalid_messages")
            return
        if model is not None and not isinstance(model, str):
            self._error(400, "model must be a string", "invalid_request_error", "invalid_model")
            return
        if not isinstance(stream, bool):
            # ``bool("false")`` is True; only JSON booleans are accepted.
            self._error(400, "stream must be a boolean", "invalid_request_error", "invalid_stream")
            return
        try:
            if stream:
                self._stream(messages, model, body)
            else:
                response = self.server.ai.chat(messages, model=parse_model(model), **body)
                origin = {"X-FreeAI-Provider": response.provider, "X-FreeAI-Model": response.model}
                self._json(200, completion_dict(response), origin)
        except ExhaustedError as exc:
            self._exhausted(exc)
        except OutputLimitError as exc:
            self._error(400, str(exc), "invalid_request_error", "output_limit")
        except (TypeError, ValueError) as exc:
            self._error(400, str(exc), "invalid_request_error", "invalid_parameter")
        except FreeAIError as exc:
            self._error(502, str(exc), "server_error", "upstream_error")

    def _stream(self, messages: list[Any], model: str | None, params: dict[str, Any]) -> None:
        chunks = self.server.ai.stream(messages, model=parse_model(model), **params)
        first = _first(chunks)  # failover happens here, so its errors still become JSON responses
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "close"}
        if first is not None:
            headers |= {"X-FreeAI-Provider": first.provider, "X-FreeAI-Model": first.model}
        self._start(200, headers)
        self.close_connection = True
        try:
            if first is not None:
                self._event(chunk_dict(first))
            for chunk in chunks:
                self._event(chunk_dict(chunk))
        except StreamInterrupted as exc:
            self._event({"error": {"message": self.server.redact(str(exc)), "type": "server_error"}})
        except (BrokenPipeError, ConnectionResetError):
            if hasattr(chunks, "close"):
                chunks.close()  # client went away: release the model's slot
            return
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- helpers ---------------------------------------------------------------------------

    def _path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _authorized(self) -> bool:
        expected = self.server._token_bytes
        if expected is None:
            return True
        scheme, _, given = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(given.strip().encode(), expected):
            return True
        self.close_connection = True  # any request body stays unread
        self._error(
            401,
            "missing or invalid bearer token",
            "invalid_request_error",
            "invalid_api_key",
            {"WWW-Authenticate": "Bearer"},
        )
        return False

    def _read_json(self) -> dict[str, Any] | None:
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self.close_connection = True
            self._error(411, "Content-Length required", "invalid_request_error", "length_required")
            return None
        size = int(length)
        if size > MAX_BODY_BYTES:
            self.close_connection = True
            self._error(413, "request body too large", "invalid_request_error", "body_too_large")
            return None
        try:
            body = json.loads(self.rfile.read(size) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(400, "body is not valid JSON", "invalid_request_error", "invalid_json")
            return None
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object", "invalid_request_error", "invalid_json")
            return None
        return body

    def _exhausted(self, exc: ExhaustedError) -> None:
        detail = {
            "attempts": [
                {"provider": a.provider, "model": a.model, "kind": a.kind, "status": a.status} for a in exc.attempts
            ]
        }
        if exc.reason.startswith("no free model matches"):
            self._error(400, str(exc), "invalid_request_error", "model_not_found", detail=detail)
        elif exc.next_available_in is not None:
            retry = {"Retry-After": str(max(1, math.ceil(exc.next_available_in)))}
            self._error(429, str(exc), "rate_limit_error", "all_models_busy", retry, detail)
        else:
            self._error(503, str(exc), "server_error", "all_models_failed", detail=detail)

    def _error(
        self,
        status: int,
        message: str,
        kind: str,
        code: str,
        headers: Mapping[str, str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self._started:
            logger.warning("proxy error after the response started: %s", message)
            return
        error: dict[str, Any] = {"message": self.server.redact(message), "type": kind, "code": code}
        if detail:
            error["x_freeai"] = detail
        self._json(status, {"error": error}, headers)

    def _json(self, status: int, body: Any, headers: Mapping[str, str] | None = None) -> None:
        data = json.dumps(body, default=str, separators=(",", ":")).encode()
        self._start(status, {"Content-Type": "application/json", "Content-Length": str(len(data)), **(headers or {})})
        self.wfile.write(data)

    def _start(self, status: int, headers: Mapping[str, str]) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        if self.close_connection and "Connection" not in headers:
            # Say so, or a keep-alive client sends its next request into the closing socket.
            self.send_header("Connection", "close")
        self.end_headers()
        self._started = True

    def _event(self, payload: dict[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(payload, default=str, separators=(",", ":")).encode() + b"\n\n")
        self.wfile.flush()


def _first(chunks: Iterator[StreamChunk]) -> StreamChunk | None:
    try:
        return next(chunks)
    except StopIteration:
        return None


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(ai: FreeAI, *, host: str = "127.0.0.1", port: int = 8787, token: str | None = None) -> None:
    """Run the proxy until interrupted."""
    with ProxyServer(ai, (host, port), token=token) as server:
        logger.warning("freeai proxy listening on %s (providers: %s)", server.url, [p.name for p in ai.providers])
        print(f"freeai: OpenAI-compatible endpoint at {server.url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            ai.close()
