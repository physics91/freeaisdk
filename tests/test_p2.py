"""P2 surfaces: aliases, the OpenAI-shaped facade, the local proxy (real socket), CLI, OTel mapping."""

import asyncio
import json
import socket
import threading

import httpx
import pytest

from freeai import FailureKind, FreeAI, OpenRouter, Provider
from freeai import server as server_module
from freeai.__main__ import main
from freeai.aio import AsyncFreeAI
from freeai.compat import AsyncOpenAI, OpenAI, parse_model
from freeai.otel import attempt_attributes
from freeai.server import MAX_BODY_BYTES, ProxyServer
from freeai.types import Attempt, ChatResponse

from .conftest import FakeServer, delta, error, ok, sse

ALPHA, BETA = "alpha/one:free", "beta/two:free"


def plain(server: FakeServer, models, api_key: str = "k") -> Provider:
    return Provider("p", "https://p.example/v1", api_key=api_key, models=list(models), http_client=server.client())


# -- aliases and facade ------------------------------------------------------------------------


def test_aliases_order_without_excluding(server):
    ai = FreeAI(plain(server, ["other", "big-120b", "tiny-flash-lite"]), max_wait=0)
    assert ai.chat("hi", model="fast").requested_model == "tiny-flash-lite"
    assert ai.chat("hi", model="smart").requested_model == "big-120b"
    assert ai.chat("hi", model="auto").requested_model == "other"
    only_other = FreeAI(plain(FakeServer(), ["other"]), max_wait=0)
    assert only_other.chat("hi", model="coding").requested_model == "other"  # alias never blocks a request
    custom = FreeAI(plain(FakeServer(), ["a", "b"]), max_wait=0, aliases={"mine": ["b"]})
    assert custom.chat("hi", model="mine").requested_model == "b"


def test_parse_model():
    assert parse_model(None) is None
    assert parse_model("auto") == "auto"
    assert parse_model("a, b") == ("a", "b")
    assert parse_model(["x"]) == ("x",)


def test_openai_facade(make_ai, server):
    server.on(ALPHA, error(500, "down"))
    client = OpenAI(make_ai())
    reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}], temperature=0)
    assert reply["object"] == "chat.completion"
    assert reply["choices"][0]["message"]["content"] == "hello"
    assert reply["x_freeai"] == {"provider": "openrouter", "requested_model": BETA, "attempts": 2}

    server.on(BETA, sse(delta("o"), delta("k")))
    chunks = list(client.chat.completions.create(model=BETA, messages=[{"role": "user", "content": "hi"}], stream=True))
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "ok"
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}
    assert client.models.list()["data"][0]["id"] == f"openrouter:{ALPHA}"


def test_openai_facade_stream_close_releases_slot(clock, server):
    server.on("m1", sse(delta("x"), delta("y"), delta(finish="stop")), ok(text="second"))
    provider = Provider(
        "p",
        "https://p.example/v1",
        api_key="k",
        models=["m1"],
        max_concurrency=1,
        http_client=server.client(),
    )
    ai = FreeAI([provider], clock=clock, sleep=clock.sleep, max_wait=0)
    stream = OpenAI(ai).chat.completions.create(messages=[{"role": "user", "content": "hi"}], stream=True)
    assert next(stream)["choices"][0]["delta"]["content"] == "x"
    stream.close()
    assert ai.chat("hi").text == "second"


def test_async_openai_facade_stream_aclose_releases_slot(clock, server):
    server.on("m1", sse(delta("x"), delta("y"), delta(finish="stop")), ok(text="second"))
    provider = Provider(
        "p",
        "https://p.example/v1",
        api_key="k",
        models=["m1"],
        max_concurrency=1,
        http_client=server.client(),
        async_http_client=server.aclient(),
    )

    async def asleep(seconds: float) -> None:
        clock.sleep(seconds)

    async def main() -> str:
        ai = AsyncFreeAI([provider], clock=clock, sleep=asleep, max_wait=0)
        stream = await AsyncOpenAI(ai).chat.completions.create(
            messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert (await stream.__anext__())["choices"][0]["delta"]["content"] == "x"
        await stream.aclose()
        try:
            return (await ai.chat("hi")).text
        finally:
            await ai.aclose()

    assert asyncio.run(main()) == "second"


def test_async_openai_models_list(server, clock):
    provider = OpenRouter(api_key="sk-test", http_client=server.client(), async_http_client=server.aclient())

    async def main() -> dict:
        client = AsyncOpenAI(AsyncFreeAI([provider], clock=clock, max_wait=0))
        try:
            return await client.models.list()
        finally:
            await client.close()

    listed = asyncio.run(main())
    assert listed["object"] == "list"
    assert {row["id"] for row in listed["data"]} >= {f"openrouter:{ALPHA}", f"openrouter:{BETA}"}


# -- proxy over a real socket ------------------------------------------------------------------


@pytest.fixture
def proxy(make_ai):
    servers = []

    def start(ai=None, token=None):
        instance = ProxyServer(ai or make_ai(), ("127.0.0.1", 0), token=token)
        threading.Thread(target=instance.serve_forever, daemon=True).start()
        servers.append(instance)
        return instance

    yield start
    for instance in servers:
        instance.shutdown()
        instance.server_close()


def test_proxy_chat_models_and_stream(proxy, server):
    url = proxy().url
    with httpx.Client(base_url=url, timeout=10) as http:
        models = http.get("/models").json()
        assert models["object"] == "list" and models["data"][0]["owned_by"] == "openrouter"

        reply = http.post("/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
        assert reply.status_code == 200
        assert reply.headers["x-freeai-provider"] == "openrouter"
        assert reply.json()["choices"][0]["message"]["content"] == "hello"

        server.on(ALPHA, sse(delta(role=True), delta("str"), delta("eam"), delta(finish="stop")))
        with http.stream(
            "POST", "/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
        ) as response:
            assert response.headers["content-type"] == "text/event-stream"
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        text = "".join(json.loads(line[6:])["choices"][0]["delta"].get("content", "") for line in lines[:-1])
        assert text == "stream"


def test_proxy_stream_interrupted_omits_done(proxy, server):
    failing = {
        "error": {"code": 502, "message": "provider crashed"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    server.on(ALPHA, sse(delta("partial"), failing))
    url = proxy().url
    with httpx.Client(base_url=url, timeout=10) as http:
        with http.stream(
            "POST", "/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] != "data: [DONE]"
    assert all(line != "data: [DONE]" for line in lines)
    payload = json.loads(lines[-1][6:])
    assert payload["error"]["type"] == "server_error"
    assert "interrupted" in payload["error"]["message"].lower() or "crashed" in payload["error"]["message"].lower()


def test_proxy_stream_output_limit_emits_error_without_done(proxy, server):
    thinking = {"model": "x", "choices": [{"index": 0, "delta": {"reasoning": "..."}}]}
    server.on(ALPHA, sse(thinking, delta(finish="length")))
    url = proxy().url
    with httpx.Client(base_url=url, timeout=10) as http:
        with http.stream(
            "POST",
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 16},
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert all(line != "data: [DONE]" for line in lines)
    payload = json.loads(lines[-1][6:])
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "output_limit"


def test_proxy_maps_exhaustion_to_429_with_retry_after(proxy, server):
    for model in (ALPHA, BETA, "gamma/three:free", "openrouter/free"):
        server.on(model, error(429, "busy", headers={"Retry-After": "120"}, provider_name="X"))
    url = proxy().url
    reply = httpx.post(f"{url}/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}, timeout=10)
    assert reply.status_code == 429
    assert reply.headers["retry-after"] == "120"
    body = reply.json()["error"]
    assert body["type"] == "rate_limit_error"
    assert [a["kind"] for a in body["x_freeai"]["attempts"]] == [FailureKind.RATE_LIMIT] * 4


def test_proxy_request_validation(proxy, monkeypatch):
    url = proxy().url
    assert httpx.post(f"{url}/chat/completions", json={"messages": []}).status_code == 400
    assert (
        httpx.post(
            f"{url}/chat/completions", content=b"{not json", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )
    # Raw JSON so stream stays a string; httpx would coerce Python bools.
    fake_stream = httpx.post(
        f"{url}/chat/completions",
        content=b'{"messages":[{"role":"user","content":"x"}],"stream":"false"}',
        headers={"content-type": "application/json"},
    )
    assert fake_stream.status_code == 400 and fake_stream.json()["error"]["code"] == "invalid_stream"
    bad = httpx.post(
        f"{url}/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "models": ["paid"]}
    )
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_parameter"
    nomatch = httpx.post(
        f"{url}/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "model": "nope/*"}
    )
    assert nomatch.status_code == 400 and nomatch.json()["error"]["code"] == "model_not_found"
    assert httpx.get(f"{url}/nowhere").status_code == 404
    monkeypatch.setattr(server_module, "MAX_BODY_BYTES", 50)
    big = httpx.post(f"{url}/chat/completions", json={"messages": [{"role": "user", "content": "x" * 100}]})
    assert big.status_code == 413


def test_proxy_keep_alive_connection_survives_errors(proxy):
    url = proxy().url
    message = {"messages": [{"role": "user", "content": "hi"}]}
    with httpx.Client(base_url=url, timeout=5) as http:  # one persistent connection, like OpenAI SDKs
        assert http.post("/chat/completions", json=message).status_code == 200
        assert http.post("/chat/completions", json={"messages": []}).status_code == 400  # error after a success
        assert http.post("/nowhere", json=message).status_code == 404  # unread body: connection is closed
        assert http.post("/chat/completions", json=message).status_code == 200


def test_proxy_rejected_token_does_not_poison_the_connection(proxy):
    url = proxy(token="s3cret").url
    with httpx.Client(base_url=url, timeout=5) as http:
        wrong = http.post("/chat/completions", json={"messages": [{"role": "user", "content": "x" * 1000}]})
        assert wrong.status_code == 401
        right = http.get("/models", headers={"Authorization": "Bearer s3cret"})
        assert right.status_code == 200


def test_proxy_announces_every_connection_it_closes(proxy):
    # Without "Connection: close" a keep-alive client may reuse the socket the server is closing
    # and get "Server disconnected without sending a response" (seen on CI, timing dependent).
    url = proxy(token="s3cret").url
    auth = {"Authorization": "Bearer s3cret"}
    message = {"messages": [{"role": "user", "content": "hi"}]}
    with httpx.Client(base_url=url, timeout=5) as http:
        kept = http.get("/models", headers=auth)
        unauthorized = http.post("/chat/completions", json=message)
        no_route = http.post("/nowhere", json=message, headers=auth)
        no_length = http.post("/chat/completions", content=iter([b"{}"]), headers=auth)  # chunked
    assert kept.status_code == 200 and kept.headers.get("connection") != "close"
    for reply, status in ((unauthorized, 401), (no_route, 404), (no_length, 411)):
        assert reply.status_code == status
        assert reply.headers.get("connection") == "close", status
    # A declared body over the limit is refused before it is sent (httpx will not send a short one).
    host, port = httpx.URL(url).host, httpx.URL(url).port
    with socket.create_connection((host, port), timeout=5) as raw:
        raw.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer s3cret\r\n"
            b"Content-Length: %d\r\n\r\n" % (MAX_BODY_BYTES + 1)
        )
        head = raw.recv(65536).split(b"\r\n\r\n", 1)[0].decode().lower()
    assert head.startswith("http/1.1 413") and "\r\nconnection: close" in head


def test_proxy_token_and_bind_policy(proxy, make_ai):
    url = proxy(token="s3cret").url
    assert httpx.get(f"{url}/models").status_code == 401
    assert httpx.get(f"{url}/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert httpx.get(f"{url}/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    with pytest.raises(ValueError, match="without a token"):
        ProxyServer(make_ai(), ("0.0.0.0", 0))


def test_proxy_redacts_provider_keys(proxy):
    upstream = FakeServer()
    upstream.on("m1", error(401, "Incorrect API key provided: sk-live-0123456789"))
    ai = FreeAI(plain(upstream, ["m1"], api_key="sk-live-0123456789"), max_wait=0)
    url = proxy(ai).url
    reply = httpx.post(f"{url}/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert reply.status_code == 503
    assert "sk-live-0123456789" not in reply.text and "[redacted]" in reply.text


# -- CLI and OTel --------------------------------------------------------------------------------


def test_cli_errors_without_network(monkeypatch, capsys):
    from freeai import PRESETS

    for preset in PRESETS.values():
        for name in preset.env:
            monkeypatch.delenv(name, raising=False)
    assert main(["models"]) == 2
    assert "no provider API key" in capsys.readouterr().err
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert main(["serve", "--host", "0.0.0.0", "--port", "0"]) == 2  # no token: refused before binding
    assert "without a token" in capsys.readouterr().err
    assert main(["models"]) == 0
    assert "groq:openai/gpt-oss-120b" in capsys.readouterr().out


def test_cli_max_tokens_zero_is_forwarded(monkeypatch, capsys):
    seen: dict = {}

    class _FakeAI:
        @classmethod
        def from_env(cls, **kwargs):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def chat(self, prompt, **params):
            seen.update(params)
            return ChatResponse(
                text="ok",
                provider="p",
                model="m",
                requested_model="m",
                message={"role": "assistant", "content": "ok"},
                finish_reason="stop",
                usage=None,
                raw={},
            )

    monkeypatch.setattr("freeai.__main__.FreeAI", _FakeAI)
    assert main(["chat", "hi", "--max-tokens", "0"]) == 0
    assert seen.get("max_tokens") == 0
    assert "ok" in capsys.readouterr().out


def test_attempt_attributes_follow_genai_conventions():
    ok = Attempt("groq", "m", True, 0.5, served_model="m-2026", usage={"prompt_tokens": 3, "completion_tokens": 2})
    attributes = attempt_attributes(ok)
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == "groq"
    assert attributes["gen_ai.response.model"] == "m-2026"
    assert (attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.output_tokens"]) == (3, 2)
    floated = attempt_attributes(
        Attempt("groq", "m", True, 0.1, usage={"prompt_tokens": 3.0, "completion_tokens": 2.0})
    )
    assert (floated["gen_ai.usage.input_tokens"], floated["gen_ai.usage.output_tokens"]) == (3, 2)
    as_str = attempt_attributes(
        Attempt("groq", "m", True, 0.1, usage={"prompt_tokens": "3", "completion_tokens": "2.0"})
    )
    assert (as_str["gen_ai.usage.input_tokens"], as_str["gen_ai.usage.output_tokens"]) == (3, 2)
    assert "gen_ai.usage.input_tokens" not in attempt_attributes(
        Attempt("groq", "m", True, 0.1, usage={"prompt_tokens": True})
    )
    failed = attempt_attributes(Attempt("p", "m", False, 0.1, kind=FailureKind.RATE_LIMIT, status=429, cooldown=7))
    assert failed["error.type"] == "rate_limit" and failed["http.response.status_code"] == 429


def test_otel_spans_per_attempt(make_ai, server):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from freeai.otel import instrument

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    seen = []
    server.on(ALPHA, error(500, "down"))
    ai = make_ai(on_attempt=seen.append)
    instrument(ai, tracer_provider=provider)
    ai.chat("hi")
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == [f"chat {ALPHA}", f"chat {BETA}"]
    assert spans[0].status.status_code.name == "ERROR"
    assert spans[1].attributes["gen_ai.response.model"] == BETA
    assert len(seen) == 2  # the previous callback still runs
