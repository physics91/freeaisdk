import pytest

from freeai import FailureKind, StreamInterrupted

from .conftest import delta, error, sse

ALPHA, BETA = "alpha/one:free", "beta/two:free"


def test_stream_yields_text(make_ai, server):
    server.on(ALPHA, sse(delta(role=True), delta("Hel"), delta("lo"), delta(finish="stop")))
    chunks = list(make_ai().stream("hi"))
    assert "".join(c.text for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert server.calls[0]["stream"] is True


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


def test_empty_stream_fails_over(make_ai, server):
    server.on(ALPHA, sse(delta(role=True), delta(finish="stop")))
    server.on(BETA, sse(delta("ok")))
    assert "".join(c.text for c in make_ai().stream("hi")) == "ok"


def test_stream_validates_arguments_eagerly(make_ai):
    with pytest.raises(TypeError):
        make_ai().stream("hi", stream=True)
