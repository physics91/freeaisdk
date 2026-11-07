import httpx
import pytest

from freeai import CooldownPolicy, FailureKind, Gemini, Groq, OpenRouter, Provider, Scope
from freeai._http import _zone, error_object, retry_hint, seconds_until_midnight
from freeai._sse import iter_sse
from freeai.providers._classify import _effective_status

NOW = 1_700_000_000.0


@pytest.mark.parametrize(
    ("headers", "error", "expected"),
    [
        ({"retry-after": "12"}, {}, 12.0),
        ({"retry-after-ms": "1500"}, {}, 1.5),
        ({"retry-after": "Tue, 14 Nov 2023 22:13:50 GMT"}, {}, 30.0),
        ({"x-ratelimit-reset": str(int((NOW + 45) * 1000))}, {}, 45.0),  # epoch ms
        ({"x-ratelimit-reset": str(int(NOW + 20))}, {}, 20.0),  # epoch s
        ({"x-ratelimit-reset": "3"}, {}, 3.0),  # relative s
        ({}, {"metadata": {"headers": {"X-RateLimit-Reset": str(int((NOW + 60) * 1000))}}}, 60.0),  # OpenRouter body
        ({}, {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "40s"}]}, 40.0),
        ({}, {"message": "Please retry in 40.15s."}, 40.15),
        ({}, {"message": "Please try again in 2m59.56s. Visit ..."}, 179.56),
        ({}, {"message": "retry after 30 seconds"}, 30.0),
        ({}, {}, None),
        ({"retry-after": "soon"}, {}, None),
    ],
)
def test_retry_hint_sources(headers, error, expected):
    result = retry_hint(httpx.Headers(headers), error, now=NOW)
    assert result == (pytest.approx(expected) if expected is not None else None)


def test_reset_headers_are_opt_in_per_dialect():
    groq_like = {"x-ratelimit-reset-requests": "2m59.56s"}
    assert retry_hint(groq_like, {}, reset_headers=()) is None
    assert retry_hint(groq_like, {}, reset_headers=("x-ratelimit-reset-requests",)) == pytest.approx(179.56)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"error": {"message": "openai style"}}, "openai style"),
        ([{"error": {"code": 429, "message": "gemini list"}}], "gemini list"),
        ({"error": "plain text"}, "plain text"),
        ({"message": "cerebras flat", "type": "too_many_requests_error", "code": "x"}, "cerebras flat"),
        ({"object": "error", "message": "mistral object"}, "mistral object"),
        ({"detail": "Unauthorized"}, "Unauthorized"),
        ({"detail": {"error": {"message": "nested detail"}}}, "nested detail"),
        ({"success": False, "errors": [{"code": 3036, "message": "daily neurons"}]}, "daily neurons"),
    ],
)
def test_error_object_normalizes_platform_shapes(body, message):
    assert error_object(body)["message"] == message


def test_error_object_ignores_success_bodies():
    assert error_object({"choices": [], "message": "x", "type": "y"}) == {}
    assert error_object({"success": True, "errors": [], "result": {}}) == {}


def test_sse_follows_the_spec():
    lines = [
        ": keepalive",
        'data: {"a":',
        "data: 1}",
        "",
        "event: error",
        'data: {"message": "boom"}',
        "",
        "id: 3",
        "",
        'data: {"b": 2}',
        "",
        "data: [DONE]",
        "",
        'data: {"c": 3}',
        "",
    ]
    assert list(iter_sse(lines)) == [{"a": 1}, {"error": {"message": "boom"}}, {"b": 2}]


def test_sse_tolerates_missing_blank_lines():
    assert list(iter_sse(['data: {"a": 1}', 'data: {"b": 2}', "data: [DONE]"])) == [{"a": 1}, {"b": 2}]
    assert list(iter_sse(['data: {"tail": true}'])) == [{"tail": True}]


def kind_scope(provider, status, body, headers=None):
    failure = provider.classify(status, headers or {}, body)
    return failure.kind, failure.scope


def test_openrouter_classification():
    router = OpenRouter(api_key="k")
    account_min = {
        "error": {
            "code": 429,
            "message": "Rate limit exceeded: free-models-per-min. ",
            "metadata": {"provider_name": "Targon"},
        }
    }
    assert kind_scope(router, 200, account_min) == (FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER)
    daily = {
        "error": {
            "code": 429,
            "message": "Rate limit exceeded: free-models-per-day. Add 10 credits",
            "metadata": {
                "headers": {
                    "X-RateLimit-Limit": "50",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "4102444800000",
                },
                "provider_name": None,
            },
        }
    }
    failure = router.classify(429, {}, daily)
    assert (failure.kind, failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER)
    assert failure.retry_after > 86400  # reset in 2100: capped later by the cooldown policy
    upstream = {
        "error": {
            "code": 429,
            "message": "Provider returned error",
            "metadata": {"raw": "x is temporarily rate-limited upstream", "provider_name": "Chutes"},
        }
    }
    assert kind_scope(router, 429, upstream) == (FailureKind.RATE_LIMIT, Scope.MODEL)
    assert kind_scope(router, 429, {"error": {"code": 429, "message": "limit_rpd/google/x/uuid"}}) == (
        FailureKind.QUOTA_EXHAUSTED,
        Scope.MODEL,
    )
    embedded = {
        "error": {
            "code": 429,
            "message": "Rate limit exceeded",
            "metadata": {"headers": {"X-RateLimit-Limit": "20"}, "provider_name": None},
        }
    }
    assert kind_scope(router, 429, embedded) == (FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER)
    assert kind_scope(router, 429, "<html>Too Many Requests</html>") == (
        FailureKind.PROVIDER_RATE_LIMIT,
        Scope.PROVIDER,
    )
    in_flight = {"error": {"code": 402, "metadata": {"limit_source": "openrouter_in_flight_budget"}}}
    assert kind_scope(router, 402, in_flight) == (FailureKind.PROVIDER_RATE_LIMIT, Scope.PROVIDER)
    assert kind_scope(router, 402, {"error": {"code": 402}}) == (FailureKind.QUOTA, Scope.PROVIDER)
    policy = {"error": {"code": 404, "message": "No endpoints found matching your data policy."}}
    assert kind_scope(router, 404, policy) == (FailureKind.NOT_ENTITLED, Scope.MODEL)


def test_generic_classification():
    generic = Provider("g", "https://g.example/v1", api_key="k")
    assert kind_scope(generic, 429, {"error": {"message": "x"}}) == (FailureKind.RATE_LIMIT, Scope.MODEL)
    rate_200 = {"error": {"type": "rate_limit_exceeded", "code": "rate_limit_exceeded"}}
    assert generic.classify(200, {}, rate_200).kind is FailureKind.RATE_LIMIT
    assert generic.classify(200, {}, {"error": {"message": "?"}}).status == 502
    assert generic.classify(500, {}, "<html>oops</html>").kind is FailureKind.UNAVAILABLE
    assert generic.classify(429, {}, "<html>Too Many Requests</html>").kind is FailureKind.RATE_LIMIT
    too_long = {"error": {"message": "This model's maximum context length is 8192 tokens"}}
    assert kind_scope(generic, 400, too_long) == (FailureKind.CONTEXT_EXCEEDED, Scope.REQUEST)
    bad_key = {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}}
    assert kind_scope(generic, 400, bad_key) == (FailureKind.AUTH, Scope.PROVIDER)
    assert kind_scope(generic, 401, {"detail": "Unauthorized"}) == (FailureKind.AUTH, Scope.PROVIDER)
    tier = {"object": "error", "message": "Model not available on your subscription tier"}
    assert kind_scope(generic, 403, tier) == (FailureKind.NOT_ENTITLED, Scope.MODEL)
    assert kind_scope(generic, 403, {"error": {"message": "flagged"}}) == (FailureKind.REFUSED, Scope.REQUEST)
    business_code = {"error": {"code": "1302", "message": "High concurrency"}}
    assert generic.classify(429, {}, business_code).status == 429  # not mistaken for HTTP 1302


def test_hints_only_apply_to_rate_limits_and_503():
    generic = Provider("g", "https://g.example/v1", api_key="k")
    assert generic.classify(500, {"retry-after": "40"}, {"error": {"message": "x"}}).retry_after is None
    assert generic.classify(503, {"retry-after": "40"}, {"error": {"message": "x"}}).retry_after == 40


def test_groq_daily_limit_and_headers():
    groq = Groq(api_key="k")
    rpd = {
        "error": {
            "message": "Rate limit reached for model `openai/gpt-oss-20b` on requests per day (RPD): "
            "Limit 1000, Used 1000. Please try again in 1h2m3s.",
            "type": "requests",
        }
    }
    failure = groq.classify(429, {"x-ratelimit-reset-requests": "5s"}, rpd)
    assert (failure.kind, failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.MODEL)
    assert failure.retry_after == pytest.approx(3723)  # from the message, not the unrelated header
    tpm = {
        "error": {"message": "on tokens per minute (TPM): Limit 8000. Please try again in 5.289s.", "type": "tokens"}
    }
    failure = groq.classify(429, {"retry-after": "6"}, tpm)
    assert (failure.kind, failure.retry_after) == (FailureKind.RATE_LIMIT, 6)


def test_gemini_quota_details_and_project_errors():
    gemini = Gemini(api_key="k")
    per_day = [
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "quota",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "40s"},
                ],
            }
        }
    ]
    failure = gemini.classify(429, {}, per_day)
    assert (failure.kind, failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.MODEL)
    assert failure.retry_after == pytest.approx(seconds_until_midnight("America/Los_Angeles"), abs=5)
    per_minute = {
        "error": {
            "code": 429,
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "PerMinutePerModel"}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
            ],
        }
    }
    failure = gemini.classify(429, {}, per_minute)
    assert (failure.kind, failure.retry_after) == (FailureKind.RATE_LIMIT, 12)
    verify = {"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "FREE_TIER_VERIFICATION_REQUIRED"}}
    assert kind_scope(gemini, 403, verify) == (FailureKind.AUTH, Scope.PROVIDER)


def test_cooldown_policy():
    policy = CooldownPolicy(rate_limit=10, max_backoff=35)
    assert [policy.duration(FailureKind.RATE_LIMIT, n, None) for n in (1, 2, 3)] == [10, 20, 35]
    assert policy.duration(FailureKind.RATE_LIMIT, 1, 0) == 1.0  # Retry-After: 0 still waits a second
    assert policy.duration(FailureKind.AUTH, 1, None) == float("inf")
    assert policy.duration(FailureKind.QUOTA_EXHAUSTED, 1, 7200) == 7200
    assert policy.duration(FailureKind.CONTEXT_EXCEEDED, 1, None) == 0
    assert policy.duration(FailureKind.OUTPUT_LIMIT, 1, None) == 0
    assert policy.duration(FailureKind.NOT_ENTITLED, 1, None) == 86400


def test_cooldown_ignores_corrupt_entries():
    from freeai.cooldown import cooldown_fail, cooldown_failures, cooldown_ready_at, cooldown_snapshot

    tx = {"cd:p:x": "bad", "cd:p:y": {"until": 100.0, "failures": 2}}
    assert cooldown_ready_at(tx, "cd:p:x") == 0.0
    assert cooldown_failures(tx, "cd:p:x") == 0
    assert cooldown_fail(tx, "cd:p:x", FailureKind.RATE_LIMIT, None, 1.0, CooldownPolicy()) == 30.0
    assert tx["cd:p:x"] == {"until": 31.0, "failures": 1}
    assert cooldown_snapshot(tx, "cd:p:", 0.0) == {"x": 31.0, "y": 100.0}
    tx["cd:p:z"] = {"until": "nope", "failures": "x"}
    assert cooldown_ready_at(tx, "cd:p:z") == 0.0
    assert cooldown_failures(tx, "cd:p:z") == 0
    assert "z" not in cooldown_snapshot(tx, "cd:p:", 0.0)


def test_zone_is_cached():
    assert _zone("America/Los_Angeles") is _zone("America/Los_Angeles")
    assert _zone("UTC") is _zone("UTC")


def test_effective_status_accepts_float_code():
    assert _effective_status({"code": 429.0}, 200) == 429
    assert _effective_status({"code": "503"}, 200) == 503
    assert _effective_status({"code": "429.0"}, 200) == 429
    assert _effective_status({"code": " 503 "}, 200) == 503
    assert _effective_status({"code": "429.5"}, 200) == 502
    assert _effective_status({"code": True}, 200) == 502


def test_effective_status_prefers_non_200_http_over_body_code():
    assert _effective_status({"code": 400, "message": "Rate limit exceeded"}, 429) == 429
    generic = Provider("g", "https://g.example/v1", api_key="k", models=["m"])
    failure = generic.classify(
        429,
        {"retry-after": "20"},
        {"error": {"code": 400, "message": "Rate limit exceeded. Please try again in 20s."}},
    )
    assert failure.kind is FailureKind.RATE_LIMIT
    assert failure.status == 429
    assert failure.retry_after == 20
    assert failure.scope is Scope.MODEL
    assert kind_scope(OpenRouter(api_key="k"), 200, {"error": {"code": 429.0, "message": "free-models-per-min"}}) == (
        FailureKind.PROVIDER_RATE_LIMIT,
        Scope.PROVIDER,
    )
    assert kind_scope(OpenRouter(api_key="k"), 200, {"error": {"code": "429.0", "message": "free-models-per-min"}}) == (
        FailureKind.PROVIDER_RATE_LIMIT,
        Scope.PROVIDER,
    )
