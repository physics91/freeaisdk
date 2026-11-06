"""Real-network checks against OpenRouter. Run with FREEAI_LIVE=1; chat needs OPENROUTER_API_KEY."""

import os

import pytest

from freeai import ExhaustedError, FailureKind, FreeAI, OpenRouter

pytestmark = pytest.mark.skipif(os.environ.get("FREEAI_LIVE") != "1", reason="set FREEAI_LIVE=1 to hit the network")


def test_live_catalog_lists_free_models():
    with FreeAI(OpenRouter(api_key="")) as ai:
        models = ai.models()
    assert models, "OpenRouter returned no free models"
    assert all(m.id.endswith(":free") or m.is_router for m in models)


def test_live_invalid_key_is_classified_as_auth():
    ai = FreeAI(OpenRouter(api_key="sk-or-v1-invalid"), max_wait=0)
    with ai, pytest.raises(ExhaustedError) as info:
        ai.chat("ping", max_tokens=5)
    assert info.value.attempts[0].kind is FailureKind.AUTH
    assert info.value.attempts[0].status == 401


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_chat_and_stream():
    with FreeAI(OpenRouter(), max_wait=20) as ai:
        response = ai.chat("Reply with the single word: pong", max_tokens=200)
        assert response.text.strip()
        print(f"chat served by {response.model} after {len(response.attempts)} attempt(s)")
        streamed = "".join(chunk.text for chunk in ai.stream("Count from 1 to 3.", max_tokens=200))
        assert streamed.strip()


def test_live_anonymous_catalogs_parse():
    """Kilo, OVHcloud, Requesty and NVIDIA list models without a key."""
    from freeai import Kilo, Requesty

    kilo = Kilo().list_models()
    assert kilo and all(m.trains_on_prompts is not None for m in kilo)
    print("kilo free:", [m.id for m in kilo])
    requesty = Requesty(api_key="").list_models()
    print("requesty zero-priced:", [m.id for m in requesty])


def test_live_anonymous_chat_ovh():
    """One real completion on OVHcloud's anonymous tier (2 req/min per model), then one async."""
    import asyncio

    from freeai import OVH, AsyncFreeAI

    with FreeAI(OVH(), max_wait=65, max_attempts=4) as ai:
        response = ai.chat("Reply with the single word: pong", max_tokens=300)
    tried = [(a.model, a.kind, a.status) for a in response.attempts]
    print(f"ovh sync: {response.model} -> {response.text.strip()[:60]!r}; attempts {tried}")
    assert response.text.strip()

    async def main():
        async with AsyncFreeAI(OVH(), max_wait=65, max_attempts=4, prefer=["gpt-oss-20b"]) as ai:
            chunks = [c.text async for c in ai.stream("Count from 1 to 3.", max_tokens=300)]
        return "".join(chunks)

    streamed = asyncio.run(main())
    print(f"ovh async stream: {streamed.strip()[:60]!r}")
    assert streamed.strip()
