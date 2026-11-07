import json

import httpx
import pytest

from freeai import FailureKind, OutputLimitError, StreamInterrupted
from freeai._http import Headers
from freeai._plan import Plan
from freeai._sse import is_event_stream

from .conftest import completion, delta, error, sse

ALPHA, BETA = "alpha/one:free", "beta/two:free"


def test_is_event_stream_content_type():
    assert is_event_stream(Headers({})) is False
    assert is_event_stream(Headers({"content-type": ""})) is False
    assert is_event_stream(Headers({"content-type": "   "})) is False
    assert is_event_stream(Headers({"content-type": "application/json"})) is False
    assert is_event_stream(Headers({"content-type": "Text/Event-Stream"})) is True
    assert is_event_stream(Headers({"content-type": "text/event-stream; charset=utf-8"})) is True
    assert is_event_stream(Headers({"content-type": "text/event-stream;charset=utf-8"})) is True


def test_stream_json_without_content_type_yields_text(make_ai, server):
    body = completion("secret")
    server.on(ALPHA, httpx.Response(200, content=json.dumps(body).encode(), headers={}))
    assert "".join(c.text for c in make_ai().stream("hi")) == "secret"


def test_stream_accepts_mixed_case_event_stream_content_type(make_ai, server):
    lines = [f"data: {json.dumps(delta('ok'))}", "data: [DONE]"]
    server.on(
        ALPHA,
        httpx.Response(
            200,
            content=("\n\n".join(lines) + "\n\n").encode(),
            headers={"content-type": "Text/Event-Stream"},
        ),
    )
    assert "".join(c.text for c in make_ai().stream("hi")) == "ok"


def test_stream_yields_text(make_ai, server):
    server.on(ALPHA, sse(delta(role=True), delta("Hel"), delta("lo"), delta(finish="stop")))
    chunks = list(make_ai().stream("hi"))
    assert "".join(c.text for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert server.calls[0]["stream"] is True


def test_stream_output_limit_raises_without_failover(make_ai, server):
    server.on(ALPHA, sse(delta(role=True), delta(finish="length")))
    server.on(BETA, sse(delta("ok")))
    with pytest.raises(OutputLimitError, match="max_tokens"):
        list(make_ai().stream("hi", max_tokens=16))
    assert server.models_called == [ALPHA]


def test_stream_reasoning_only_length_raises_output_limit(make_ai, server):
    thinking = {"model": "x", "choices": [{"index": 0, "delta": {"reasoning": "..."}}]}
    server.on(ALPHA, sse(thinking, delta(finish="length")))
    server.on(BETA, sse(delta("ok")))
    with pytest.raises(OutputLimitError, match="max_tokens"):
        list(make_ai().stream("hi", max_tokens=16))
    assert server.models_called == [ALPHA]


def test_error_event_before_output_fails_over(make_ai, server):
    failing = {
        "error": {"code": 429, "message": "upstream busy", "metadata": {"provider_name": "X"}},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    server.on(ALPHA, sse(delta(role=True), failing))
    server.on(BETA, sse(delta("ok"), delta(finish="stop")))
    seen = []
    chunks = list(make_ai(on_attempt=seen.append).stream("hi"))
    assert "".join(c.text for c in chunks) == "ok"
    assert {c.model for c in chunks} == {"x"}
    assert [(a.model, a.kind) for a in seen] == [(ALPHA, FailureKind.RATE_LIMIT), (BETA, None)]


def test_http_error_at_stream_start_fails_over(make_ai, server):
    server.on(ALPHA, error(503, "no provider"))
    server.on(BETA, sse(delta("ok")))
    assert "".join(c.text for c in make_ai().stream("hi")) == "ok"
    assert server.models_called == [ALPHA, BETA]


def test_error_after_output_raises_with_partial_text(make_ai, server):
    failing = {
        "error": {"code": 502, "message": "provider crashed"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    server.on(ALPHA, sse(delta("partial "), delta("answer"), failing))
    received = []
    with pytest.raises(StreamInterrupted) as info:
        for chunk in make_ai().stream("hi"):
            received.append(chunk.text)
    assert info.value.partial_text == "partial answer"
    assert "".join(received) == "partial answer"
    assert server.models_called == [ALPHA]


def test_stream_failure_forwards_chunk_usage(make_ai, server, monkeypatch):
    captured: dict = {}
    original = Plan.failed

    def wrap(self, model, failure, started, snapshots=(), *, usage=None):
        captured["usage"] = usage
        return original(self, model, failure, started, snapshots, usage=usage)

    monkeypatch.setattr(Plan, "failed", wrap)
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    failing = {
        "error": {"code": 502, "message": "provider crashed"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    server.on(ALPHA, sse({**delta("partial"), "usage": usage}, failing))
    with pytest.raises(StreamInterrupted):
        list(make_ai().stream("hi"))
    assert captured["usage"] == usage


def test_empty_stream_fails_over(make_ai, server):
    server.on(ALPHA, sse(delta(role=True), delta(finish="stop")))
    server.on(BETA, sse(delta("ok")))
    assert "".join(c.text for c in make_ai().stream("hi")) == "ok"


def test_whitespace_only_stream_fails_over(make_ai, server):
    from freeai.types import StreamChunk

    assert StreamChunk(text="  \n", provider="p", model="m").has_output is False
    assert StreamChunk(text="", reasoning="  \n", provider="p", model="m").has_output is False
    server.on(ALPHA, sse(delta(role=True), delta("  \n"), delta(finish="stop")))
    server.on(BETA, sse(delta("ok"), delta(finish="stop")))
    assert "".join(c.text for c in make_ai().stream("hi")) == "ok"
    assert server.models_called == [ALPHA, BETA]


def test_whitespace_only_reasoning_stream_fails_over(make_ai, server):
    thinking = {"model": "x", "choices": [{"index": 0, "delta": {"reasoning": "  \n"}}]}
    server.on(ALPHA, sse(delta(role=True), thinking, delta(finish="stop")))
    server.on(BETA, sse(delta("ok"), delta(finish="stop")))
    assert "".join(c.text for c in make_ai().stream("hi")) == "ok"
    assert server.models_called == [ALPHA, BETA]


def test_reasoning_only_stop_stream_fails_over_like_chat(make_ai, server):
    thinking = {"model": "x", "choices": [{"index": 0, "delta": {"reasoning": "think"}}]}
    server.on(ALPHA, sse(thinking, delta(finish="stop")))
    server.on(BETA, sse(delta("ok"), delta(finish="stop")))
    chunks = list(make_ai().stream("hi"))
    assert "".join(c.text for c in chunks) == "ok"
    assert server.models_called == [ALPHA, BETA]


def test_stream_keeps_wire_served_model_across_model_less_events(make_ai, server):
    content = {"model": "real/served:free", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    seen = []
    server.on(ALPHA, sse(content, finish))
    chunks = list(make_ai(on_attempt=seen.append).stream("hi"))
    assert [c.model for c in chunks] == ["real/served:free", "real/served:free"]
    assert seen[-1].served_model == "real/served:free"


def test_stream_validates_arguments_eagerly(make_ai):
    with pytest.raises(TypeError):
        make_ai().stream("hi", stream=True)
