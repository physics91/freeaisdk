import time

import httpx
import pytest

from freeai import (
    PRESETS,
    BillingWarning,
    ExhaustedError,
    FailureKind,
    FreeAI,
    Gemini,
    Groq,
    OpenRouter,
    OutputLimitError,
    Provider,
)
from freeai.limits import EDGE_MARGIN
from freeai.providers import GROQ_FREE_MODELS

from .conftest import FakeServer, error, ok

ALPHA, BETA, GAMMA = "alpha/one:free", "beta/two:free", "gamma/three:free"


def test_upstream_429_swaps_model_and_benches_only_that_model(make_ai, server):
    server.on(ALPHA, error(429, "rate-limited upstream", headers={"Retry-After": "7"}, provider_name="Chutes"))
    ai = make_ai()

    response = ai.chat("hi")

    assert response.requested_model == BETA
    assert [(a.model, a.ok, a.kind) for a in response.attempts] == [
        (ALPHA, False, FailureKind.RATE_LIMIT),
        (BETA, True, None),
    ]
    assert response.attempts[0].cooldown == 7
    assert ai.status()["models"] == {f"openrouter:{ALPHA}": 7}
    assert ai.status()["providers"] == {}

    server.calls.clear()
    ai.chat("again")
    assert server.models_called == [BETA]  # alpha is still cooling down


def test_platform_429_benches_whole_provider_and_uses_another_provider(make_ai, server, clock):
    reset_ms = str(int((time.time() + 30) * 1000))  # OpenRouter-style epoch milliseconds
    server.on(
        ALPHA,
        error(
            429,
            "Rate limit exceeded: free-models-per-min",
            headers={
                "X-RateLimit-Limit": "20",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": reset_ms,
            },
        ),
    )
    backup_server = FakeServer()
    backup = Provider(
        "backup", "https://backup.example/v1", api_key="k", models=["m1"], http_client=backup_server.client()
    )
    ai = make_ai(backup)

    response = ai.chat("hi")

    assert response.provider == "backup"
    assert server.models_called == [ALPHA]  # no other OpenRouter model was burned
    assert response.attempts[0].kind is FailureKind.PROVIDER_RATE_LIMIT
    assert ai.status()["providers"]["openrouter"] == pytest.approx(30, abs=2)


def test_error_inside_http_200_is_a_failure(make_ai, server):
    server.on(ALPHA, httpx.Response(200, json={"error": {"code": 502, "message": "upstream died"}}))
    response = make_ai().chat("hi")
    assert response.requested_model == BETA
    assert response.attempts[0].kind is FailureKind.UNAVAILABLE


def test_empty_completion_swaps_model(make_ai, server):
    server.on(ALPHA, ok(""))
    response = make_ai().chat("hi")
    assert response.requested_model == BETA
    assert response.attempts[0].kind is FailureKind.EMPTY


def test_network_error_swaps_model(make_ai, server):
    def explode(body):
        raise httpx.ConnectTimeout("timed out")

    server.on(ALPHA, explode)
    response = make_ai().chat("hi")
    assert response.requested_model == BETA
    assert response.attempts[0].kind is FailureKind.UNAVAILABLE


def test_invalid_key_disables_provider(make_ai, server):
    server.on(ALPHA, error(401, "No auth credentials found"))
    ai = make_ai()
    with pytest.raises(ExhaustedError, match="check the API keys") as info:
        ai.chat("hi")
    assert [a.kind for a in info.value.attempts] == [FailureKind.AUTH]
    assert info.value.next_available_in is None

    server.calls.clear()
    with pytest.raises(ExhaustedError):
        ai.chat("hi")
    assert server.calls == []  # never retried


def test_missing_key_fails_without_sending(server, clock):
    ai = FreeAI(OpenRouter(api_key="", http_client=server.client()), clock=clock, sleep=clock.sleep, max_wait=0)
    with pytest.raises(ExhaustedError, match="missing API key"):
        ai.chat("hi")
    assert server.calls == []


def test_repeated_bad_request_skips_only_that_provider(make_ai, server):
    for model in (ALPHA, BETA, GAMMA, "openrouter/free"):
        server.on(model, error(400, "messages: field required"))
    with pytest.raises(ExhaustedError, match="every candidate model failed") as info:
        make_ai().chat("hi")
    assert len(info.value.attempts) == 2  # two rejections from one provider stop that provider

    server.calls.clear()
    backup = Provider(
        "backup", "https://backup.example/v1", api_key="k", models=["m1"], http_client=FakeServer().client()
    )
    response = make_ai(backup).chat("hi")
    assert response.provider == "backup"
    assert server.models_called == [ALPHA, BETA]


def test_402_quota_benches_provider(make_ai, server):
    server.on(ALPHA, error(402, "Insufficient credits"))
    with pytest.raises(ExhaustedError):
        make_ai().chat("hi")
    assert server.models_called == [ALPHA]


def test_waits_for_short_cooldown_within_budget(make_ai, server, clock):
    server.on(ALPHA, error(429, "slow down", headers={"Retry-After": "5"}, provider_name="X"), ok("finally"))
    ai = make_ai(max_wait=30)

    response = ai.chat("hi", model=ALPHA)

    assert response.text == "finally"
    assert clock.sleeps == [5.0]
    assert len(response.attempts) == 2


def test_gives_up_when_cooldown_exceeds_wait_budget(make_ai, server):
    server.on(ALPHA, error(429, "slow down", headers={"Retry-After": "120"}, provider_name="X"))
    with pytest.raises(ExhaustedError) as info:
        make_ai(max_wait=30).chat("hi", model=ALPHA)
    assert info.value.next_available_in == pytest.approx(120)


def test_proactive_rpm_budget_prevents_platform_429(make_ai, server, clock):
    ai = make_ai(openrouter={"rpm": 2})
    ai.chat("1")
    ai.chat("2")
    with pytest.raises(ExhaustedError, match="cooling down or rate limited"):
        ai.chat("3")
    assert len(server.calls) == 2  # the third request never left the client

    waiting = make_ai(openrouter={"rpm": 1}, max_wait=90)
    waiting.chat("1")
    waiting.chat("2")
    assert clock.sleeps == [60.0 + EDGE_MARGIN]  # one per minute: every slot is regained quota


def test_tools_narrow_candidates(make_ai, server):
    server.on(ALPHA, error(503, "no provider"))
    response = make_ai().chat("weather?", tools=[{"type": "function", "function": {"name": "w", "parameters": {}}}])
    assert server.models_called == [ALPHA, "openrouter/free"]  # beta/gamma lack tool support
    assert response.requested_model == "openrouter/free"


def test_image_input_narrows_candidates(make_ai, server):
    image = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    }
    make_ai().chat([image])
    assert server.models_called == [BETA]


def test_prefer_and_model_patterns_control_order(make_ai, server):
    make_ai(prefer=["gamma/*"]).chat("hi")
    assert server.models_called == [GAMMA]

    server.calls.clear()
    server.on(BETA, error(500, "down"))
    make_ai().chat("hi", model=[BETA, "openrouter:alpha/*"])
    assert server.models_called == [BETA, ALPHA]


def test_round_robin_spreads_load(make_ai, server):
    ai = make_ai(strategy="round_robin")
    for _ in range(3):
        ai.chat("hi")
    assert server.models_called == [ALPHA, BETA, GAMMA]


def test_payload_passthrough_and_reserved_params(make_ai, server):
    ai = make_ai()
    ai.chat("hi", system="be brief", temperature=0.2, max_tokens=50)
    sent = server.calls[0]
    assert sent["model"] == ALPHA
    assert sent["temperature"] == 0.2 and sent["max_tokens"] == 50
    assert sent["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    assert "models" not in sent

    with pytest.raises(TypeError):
        ai.chat("hi", models=["paid/model"])  # would let OpenRouter fall back to paid models


def test_model_gone_refreshes_catalog(make_ai, server):
    server.on(ALPHA, error(404, "No endpoints found"))
    ai = make_ai()
    ai.chat("hi")
    assert server.catalog_fetches == 1
    ai.chat("hi")
    assert server.catalog_fetches == 2


def test_no_matching_model(make_ai):
    with pytest.raises(ExhaustedError, match="no free model matches"):
        make_ai().chat("hi", require=["web_search_options"])


def test_on_attempt_callback(make_ai, server):
    seen = []
    server.on(ALPHA, error(500, "down"))
    make_ai(on_attempt=seen.append).chat("hi")
    assert [(a.model, a.ok) for a in seen] == [(ALPHA, False), (BETA, True)]


def test_context_exceeded_moves_to_a_larger_window(make_ai, server):
    server.on(GAMMA, error(400, "This model's maximum context length is 8000 tokens"))
    response = make_ai(prefer=["gamma/*"]).chat("long prompt")
    assert server.models_called == [GAMMA, ALPHA]
    assert response.attempts[0].kind is FailureKind.CONTEXT_EXCEEDED
    assert response.attempts[0].cooldown == 0


def test_output_limit_raises_without_failover(make_ai, server):
    exhausted = {
        "model": ALPHA,
        "choices": [
            {"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "", "reasoning": "..."}}
        ],
    }
    server.on(ALPHA, httpx.Response(200, json=exhausted))
    ai = make_ai()
    with pytest.raises(OutputLimitError, match="max_tokens"):
        ai.chat("hi", max_tokens=16)
    assert server.models_called == [ALPHA]
    assert ai.status()["models"] == {}


def test_daily_quota_benches_provider_until_reset_and_resyncs(make_ai, server):
    reset_ms = str(int((time.time() + 3 * 3600) * 1000))
    server.on(
        ALPHA,
        error(
            429,
            "Rate limit exceeded: free-models-per-day. Add 10 credits",
            headers=None,
            headers_in_body={"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset_ms},
        ),
    )
    ai = make_ai()
    with pytest.raises(ExhaustedError) as info:
        ai.chat("hi")
    assert server.models_called == [ALPHA]  # no other free model was burned
    assert info.value.attempts[0].kind is FailureKind.QUOTA_EXHAUSTED
    assert ai.status()["providers"]["openrouter"] == pytest.approx(3 * 3600, abs=5)
    assert [p for p in server.paths if p.endswith("/key")] == ["/api/v1/key", "/api/v1/key"]  # warm-up + resync


def test_quota_sync_sets_daily_budget(make_ai, server):
    server.daily = {"used": 50, "limit": 50, "remaining": 0}
    with pytest.raises(ExhaustedError, match="cooling down or rate limited"):
        make_ai().chat("hi")
    assert server.calls == []  # the daily budget was known to be spent before sending anything


def test_discovery_prefers_models_user_when_keyed(make_ai, server):
    make_ai().models()
    assert server.paths[0] == "/api/v1/models/user"


def test_discovery_falls_back_to_next_catalog_path(server):
    class FallingServer(FakeServer):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/models/user"):
                self.paths.append(path)
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return super().__call__(request)

    falling = FallingServer(server.catalog)
    models = OpenRouter(api_key="sk-test", http_client=falling.client()).list_models()
    assert falling.paths == ["/api/v1/models/user", "/api/v1/models"]
    assert models


def test_openrouter_requires_parameters_for_tools(make_ai, server):
    tools = [{"type": "function", "function": {"name": "w", "parameters": {}}}]
    make_ai().chat("weather?", tools=tools, n=3)
    assert server.calls[0]["provider"] == {"require_parameters": True}
    assert "n" not in server.calls[0]


def test_empty_capability_params_do_not_narrow():
    from freeai._prepare import ALIASES, PreparePolicy
    from freeai.providers.openrouter import OpenRouter

    prepare = PreparePolicy(aliases=ALIASES, allow_training=True, by_name={})
    for empty in (
        {"tools": []},
        {"tool_choice": ""},
        {"tool_choice": "none"},
        {"tool_choice": {"type": "none"}},
        {"response_format": {}},
        {"response_format": {"type": "text"}},
        {"reasoning": False},
        {"response_format": False},
        {"reasoning": 0},
    ):
        request = prepare.prepare("hi", None, None, (), None, dict(empty))
        assert request.requirements.parameters == frozenset()
        body = OpenRouter(api_key="k").prepare_payload("m", {"messages": [], **empty})
        assert "provider" not in body
    request = prepare.prepare(
        "hi", None, None, (), None, {"tools": [{"type": "function", "function": {"name": "w", "parameters": {}}}]}
    )
    assert "tools" in request.requirements.parameters
    assert "reasoning" in prepare.prepare("hi", None, None, (), None, {"reasoning": True}).requirements.parameters
    assert (
        "response_format"
        in prepare.prepare(
            "hi", None, None, (), None, {"response_format": {"type": "json_object"}}
        ).requirements.parameters
    )


def test_openrouter_provider_field_ignores_non_dict():
    from freeai.providers.openrouter import OpenRouter

    router = OpenRouter(api_key="k")
    body = router.prepare_payload(
        "m",
        {"messages": [], "tools": [{"type": "function", "function": {"name": "w", "parameters": {}}}], "provider": "x"},
    )
    assert body["provider"] == {"require_parameters": True}
    denied = router.deny_training({"messages": [], "provider": "x"})
    assert denied["provider"] == {"data_collection": "deny"}
    merged = router.deny_training({"messages": [], "provider": {"order": ["a"]}})
    assert merged["provider"] == {"order": ["a"], "data_collection": "deny"}


def test_openrouter_free_model_tolerates_bad_architecture():
    from freeai.providers.openrouter import OpenRouter

    router = OpenRouter(api_key="k")
    priced = {"prompt": "0", "completion": "0"}
    assert router._free_model({"id": "x:free", "pricing": priced, "architecture": "bad"}, "2099-01-01") is not None
    assert (
        router._free_model(
            {"id": "x:free", "pricing": priced, "architecture": {"output_modalities": ["audio"]}}, "2099-01-01"
        )
        is None
    )


def test_from_env_enables_only_free_tiers(monkeypatch):
    for preset in PRESETS.values():
        for name in preset.env:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("CEREBRAS_API_KEY", "c")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    assert [p.name for p in FreeAI.from_env().providers] == ["groq"]
    with pytest.warns(BillingWarning):
        everything = FreeAI.from_env(tiers=("free", "trial", "paid"))
    assert [p.name for p in everything.providers] == ["groq", "cerebras", "mistral"]


def test_preset_payload_adaptation():
    messages = [{"role": "user", "content": "hi", "name": "kim"}]
    sent = Groq(api_key="k").prepare_payload("openai/gpt-oss-20b", {"messages": messages, "logprobs": True})
    assert sent == {"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "hi"}]}
    gem = Gemini(api_key="k")
    assert "temperature" not in gem.prepare_payload("gemini-3.8-flash", {"messages": [], "temperature": 0.2})
    assert gem.prepare_payload("gemini-3.1-flash-lite", {"messages": [], "temperature": 0.2})["temperature"] == 0.2
    assert [m.id for m in Groq(api_key="k").list_models()] == list(GROQ_FREE_MODELS)
