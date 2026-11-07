"""New platform presets: catalogs, business error codes, environment detection."""

import time

import httpx
import pytest

from freeai import (
    NVIDIA,
    OVH,
    PRESETS,
    ZAI,
    BillingWarning,
    Cloudflare,
    Cohere,
    FailureKind,
    FreeAI,
    FreeAIError,
    Kilo,
    Requesty,
    SambaNova,
    Scope,
)
from freeai.limits import seconds_until

from .conftest import FakeServer


def test_cloudflare_needs_account_and_classifies_neuron_exhaustion(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(FreeAIError, match="account_id"):
        Cloudflare(api_key="t")
    cf = Cloudflare(api_key="t", account_id="acc")
    assert cf.base_url == "https://api.cloudflare.com/client/v4/accounts/acc/ai/v1"

    daily = {"success": False, "errors": [{"code": 3036, "message": "You have used up your daily free allocation"}]}
    failure = cf.classify(429, {}, daily)
    assert (failure.kind, failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER)
    assert failure.retry_after == pytest.approx(seconds_until("day", time.time(), "UTC"), abs=5)
    # JSON may decode the code as float; must still hit business_codes (provider scope).
    daily_float = {"success": False, "errors": [{"code": 3036.0, "message": "daily free allocation"}]}
    float_failure = cf.classify(429, {}, daily_float)
    assert (float_failure.kind, float_failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER)
    daily_str = {"success": False, "errors": [{"code": "3036.0", "message": "daily free allocation"}]}
    str_failure = cf.classify(429, {}, daily_str)
    assert (str_failure.kind, str_failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER)

    busy = cf.classify(429, {}, {"errors": [{"code": 3040, "message": "Out of capacity"}], "success": False})
    assert (busy.kind, busy.scope) == (FailureKind.UNAVAILABLE, Scope.MODEL)


def test_zai_business_codes_and_flush_time():
    zai = ZAI(api_key="k")
    assert zai.max_concurrency == 1
    rate = zai.classify(429, {}, {"error": {"code": "1302", "message": "Rate limit reached for requests"}})
    assert (rate.kind, rate.scope) == (FailureKind.RATE_LIMIT, Scope.MODEL)
    fair = zai.classify(429, {}, {"error": {"code": "1313", "message": "fair use"}})
    assert fair.kind is FailureKind.PROVIDER_RATE_LIMIT
    unknown_model = zai.classify(400, {}, {"error": {"code": "1211", "message": "Unknown Model"}})
    assert unknown_model.kind is FailureKind.MODEL_GONE

    soon = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 3600 + 600))  # +10 min in UTC+8
    usage = zai.classify(
        429,
        {},
        {"error": {"code": "1308", "message": f"Usage limit reached for 5 hours. Your limit will reset at {soon}"}},
    )
    assert (usage.kind, usage.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER)
    assert usage.retry_after == pytest.approx(600, abs=5)


def test_kilo_keeps_only_free_models_and_reads_training_flag():
    catalog = {
        "data": [
            {"id": "kilo-auto/free", "isFree": True, "mayTrainOnYourPrompts": True, "context_length": 256000},
            {"id": "nvidia/x:free", "isFree": True, "mayTrainOnYourPrompts": False, "supported_parameters": ["tools"]},
            {"id": "anthropic/paid", "isFree": False},
        ]
    }
    server = FakeServer(catalog)
    kilo = Kilo(http_client=server.client())
    models = kilo.list_models()
    assert [(m.id, m.is_router, m.trains_on_prompts) for m in models] == [
        ("nvidia/x:free", False, False),
        ("kilo-auto/free", True, True),
    ]
    assert kilo.keyless and any(limit.per == "hour" for limit in kilo.limits)  # anonymous: 200 req/hour
    assert not any(limit.per == "hour" for limit in Kilo(api_key="k").limits)


def test_requesty_filters_zero_priced_models():
    catalog = {
        "data": [
            {"id": "free/one", "input_price": 0, "output_price": 0, "context_window": 32000},
            {"id": "paid/two", "input_price": 0.000001, "output_price": 0.000002},
            {"id": "half/three", "input_price": "0", "output_price": "0.1"},
        ]
    }
    models = Requesty(api_key="k", http_client=FakeServer(catalog).client()).list_models()
    assert [(m.id, m.context_length) for m in models] == [("free/one", 32000)]


def test_preset_limits_and_tiers():
    sambanova = SambaNova(api_key="k")
    assert {(lim.amount, lim.per, lim.unit, lim.scope) for lim in sambanova.limits} >= {
        (20, "day", "requests", "model")
    }
    assert Cohere.tier == "trial"
    assert NVIDIA.trains_on_prompts is True
    assert OVH().keyless and OVH.trains_on_prompts is False


def test_cohere_monthly_trial_quota():
    failure = Cohere(api_key="k").classify(
        429, {}, {"message": "You are using a Trial key, which is limited to 1000 API calls / month."}
    )
    assert (failure.kind, failure.scope) == (FailureKind.QUOTA_EXHAUSTED, Scope.PROVIDER)
    assert failure.retry_after == pytest.approx(seconds_until("month", time.time(), "UTC"), abs=5)


def test_from_env_keyless_presets_are_opt_in(monkeypatch):
    for preset in PRESETS.values():
        for name in preset.env:
            monkeypatch.delenv(name, raising=False)
    with pytest.raises(FreeAIError, match="include_keyless"):
        FreeAI.from_env()
    names = [p.name for p in FreeAI.from_env(include_keyless=True).providers]
    assert names == ["kilo", "ovh"]

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    assert "cloudflare" not in [p.name for p in FreeAI.from_env(include_keyless=True).providers]  # needs both vars
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("COHERE_API_KEY", "c")
    assert [p.name for p in FreeAI.from_env().providers] == ["cloudflare"]
    with pytest.warns(BillingWarning):
        assert "cohere" in [p.name for p in FreeAI.from_env(tiers=("free", "trial")).providers]


def test_keyless_provider_sends_without_authorization():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "hi"}}]})

    ovh = OVH(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    response = FreeAI(ovh, max_wait=0).chat("ping")
    assert response.text == "hi" and seen == [None]
