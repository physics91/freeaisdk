import threading
import time
from concurrent.futures import ThreadPoolExecutor

from freeai import FreeAI, OpenRouter, Provider
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
            super().__init__(
                name, "https://example.test", api_key="k", models=[ModelInfo(name, "m")], keyless=False
            )

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
