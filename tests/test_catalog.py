import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from freeai import AsyncFreeAI, CatalogDrift, FreeAI, Groq, OpenRouter, Provider
from freeai.types import ModelInfo

from .conftest import FakeServer


def keys(models):
    return [m.key for m in models]


def test_discovers_only_free_text_chat_models(make_ai):
    ai = make_ai()
    assert keys(ai.models()) == [
        "openrouter:alpha/one:free",
        "openrouter:beta/two:free",
        "openrouter:gamma/three:free",
        "openrouter:openrouter/free",  # routers are the last resort
    ]


def test_zero_priced_non_free_ids_are_opt_in(make_ai):
    ai = make_ai(openrouter={"include_zero_priced": True, "include_routers": False})
    assert "openrouter:stealth/zero" in keys(ai.models())
    assert "openrouter:openrouter/free" not in keys(ai.models())
    assert "openrouter:paid/model" not in keys(ai.models())


def test_capability_filters(make_ai):
    ai = make_ai()
    assert keys(ai.models(require=["tools"])) == ["openrouter:alpha/one:free", "openrouter:openrouter/free"]
    assert keys(ai.models(input_modalities=["image"])) == ["openrouter:beta/two:free", "openrouter:openrouter/free"]
    assert "openrouter:gamma/three:free" not in keys(ai.models(min_context=16000))


def test_exclusions_can_be_disabled(make_ai):
    ai = make_ai(exclude=())
    assert "openrouter:nvidia/x-content-safety:free" in keys(ai.models())


def test_catalog_is_cached(server: FakeServer):
    ai = FreeAI(OpenRouter(api_key="k", http_client=server.client()))
    ai.models()
    ai.models()
    assert server.catalog_fetches == 1
    ai.models(refresh=True)
    assert server.catalog_fetches == 2


def test_list_models_copy_isolates_cache_and_snapshot_is_shared(server: FakeServer):
    provider = OpenRouter(api_key="k", http_client=server.client())
    first = provider.list_models()
    snapshot = provider.models_snapshot()
    first.clear()
    assert provider.list_models()  # public copy mutation must not empty the cache
    assert provider.models_snapshot() is snapshot  # routing reuses the immutable snapshot
    assert tuple(provider.list_models()) == snapshot


def test_sync_catalog_single_flight_and_lock_not_held_during_discover():
    fetches = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    class Slow(Provider):
        def __init__(self) -> None:
            super().__init__("p", "https://example.test", api_key="k", keyless=False)

        def _discover(self) -> list[ModelInfo]:
            fetches["n"] += 1
            started.set()
            assert release.wait(1.0)
            return [ModelInfo(self.name, "m")]

    provider = Slow()

    def snapshot() -> tuple[ModelInfo, ...]:
        return provider.models_snapshot()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(snapshot) for _ in range(4)]
        assert started.wait(1.0)
        acquired = provider._catalog_lock.acquire(timeout=0.05)
        assert acquired  # discover must not hold _catalog_lock across I/O
        provider._catalog_lock.release()
        release.set()
        results = [future.result() for future in futures]
    assert fetches["n"] == 1
    assert all(result == (ModelInfo("p", "m"),) for result in results)


def test_sync_pools_fetches_cold_catalogs_in_parallel():
    delay = 0.15

    class Slow(Provider):
        def __init__(self, name: str) -> None:
            super().__init__(name, "https://example.test", api_key="k", keyless=False)

        def _discover(self) -> list[ModelInfo]:
            time.sleep(delay)
            return [ModelInfo(self.name, "m")]

    ai = FreeAI([Slow("a"), Slow("b")], max_wait=0.0)
    started = time.perf_counter()
    pools = ai._pools()
    elapsed = time.perf_counter() - started
    assert set(pools) == {"a", "b"}
    assert elapsed < delay * 1.7  # wall clock ~max(delay), not the sequential sum


def test_sync_plan_overlaps_pools_with_quota_refresh():
    delay = 0.15
    from freeai.limits import RateSnapshot

    class Both(Provider):
        def __init__(self, name: str) -> None:
            super().__init__(name, "https://example.test", api_key="k", keyless=False)

        def _discover(self) -> list[ModelInfo]:
            time.sleep(delay)
            return [ModelInfo(self.name, "m")]

        def _quota_path(self) -> str | None:
            return "/quota"

        def refresh_quota(self):
            time.sleep(delay)
            return [RateSnapshot("day", "requests", 10, 9, "provider")]

    ai = FreeAI([Both("a")], max_wait=0.0)
    request = ai._request("hi", None, None, (), None, {})
    started = time.perf_counter()
    ai._plan(request, None, None)
    elapsed = time.perf_counter() - started
    assert elapsed < delay * 1.7  # catalog and quota overlap on first plan


def test_sync_warmups_refresh_quota_in_parallel():
    delay = 0.15
    from freeai.limits import RateSnapshot

    class SlowQuota(Provider):
        def __init__(self, name: str) -> None:
            super().__init__(name, "https://example.test", api_key="k", models=[ModelInfo(name, "m")], keyless=False)

        def _quota_path(self) -> str | None:
            return "/quota"

        def refresh_quota(self):
            time.sleep(delay)
            return [RateSnapshot("day", "requests", 10, 9, "provider")]

    ai = FreeAI([SlowQuota("a"), SlowQuota("b")], max_wait=0.0)
    warmups = ai._router.claim_warmups({"a", "b"})
    started = time.perf_counter()
    ai._observe_warmups(warmups)
    elapsed = time.perf_counter() - started
    assert elapsed < delay * 1.7


def _listing(*ids: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": model_id} for model_id in ids]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_audit_reports_where_a_pinned_list_went_stale(clock):
    live = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "new/model-1", "whisper-large-v3", "Qwen3Guard-Gen-8B")
    groq = Groq(api_key="k", http_client=_listing(*live))
    discovering = OpenRouter(api_key="sk-test", http_client=FakeServer().client())
    ai = FreeAI([discovering, groq], clock=clock)

    assert ai.audit() == [
        CatalogDrift(
            "groq",
            pinned=("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"),
            missing=("openai/gpt-oss-20b",),
            unpinned=("new/model-1",),  # speech and guard models are not chat candidates
        )
    ]
    assert [m.id for m in groq.list_models()] == list(groq.catalog_drift().pinned)  # routing still uses the pins


def test_audit_survives_a_catalog_it_cannot_read(clock):
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "nope"})

    groq = Groq(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    (drift,) = FreeAI(groq, clock=clock).audit()
    assert (drift.missing, drift.unpinned, drift.changed) == ((), (), False)
    assert "403" in (drift.error or "")


def test_async_audit_and_cli(clock, capsys, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-120b"}, {"id": "new/model-1"}]})

    async def main() -> list[CatalogDrift]:
        groq = Groq(api_key="k", async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        async with AsyncFreeAI(groq) as ai:
            return await ai.audit()

    (drift,) = asyncio.run(main())
    assert drift.missing == ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    assert drift.unpinned == ("new/model-1",)

    from freeai import __main__ as cli

    groq = Groq(api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(cli.FreeAI, "from_env", classmethod(lambda cls, **kwargs: FreeAI(groq)))
    assert cli.main(["audit"]) == 1  # a pinned model is gone
    out = capsys.readouterr().out
    assert "groq: 3 pinned, differs from the catalog" in out
    assert "no longer listed: openai/gpt-oss-20b, qwen/qwen3.8-27b" in out
    assert "listed but not pinned (not shown to be free): new/model-1" in out
