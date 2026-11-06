from freeai import FreeAI, OpenRouter

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
