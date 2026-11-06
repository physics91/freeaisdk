"""Named provider factories for ``FreeAI.from_env()``."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..errors import FreeAIError
from ._impl import Provider, Tier
from .gemini import Gemini
from .groq import Groq
from .openrouter import OpenRouter
from .platforms import NVIDIA, OVH, ZAI, Cerebras, Cloudflare, Cohere, Kilo, Mistral, Requesty, SambaNova


def groq(api_key: str | None = None, **kwargs: Any) -> Groq:
    return Groq(api_key, **kwargs)


def gemini(api_key: str | None = None, **kwargs: Any) -> Gemini:
    return Gemini(api_key, **kwargs)


def cerebras(api_key: str | None = None, **kwargs: Any) -> Cerebras:
    return Cerebras(api_key, **kwargs)


def mistral(api_key: str | None = None, **kwargs: Any) -> Mistral:
    return Mistral(api_key, **kwargs)


@dataclass(frozen=True, slots=True)
class Preset:
    """A provider ``FreeAI.from_env()`` can build.

    It is enabled when every ``env`` variable is set, or, for ``keyless`` presets, when the caller
    opts into anonymous access with ``include_keyless=True``.
    """

    env: tuple[str, ...]
    factory: Callable[[], Provider]
    tier: Tier
    keyless: bool = False


# Order is the default priority: providers with their own capacity first, gateways after.
PRESETS: dict[str, Preset] = {
    "openrouter": Preset(("OPENROUTER_API_KEY",), OpenRouter, "free"),
    "groq": Preset(("GROQ_API_KEY",), Groq, "free"),
    "gemini": Preset(("GEMINI_API_KEY",), Gemini, "free"),
    "cloudflare": Preset(("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"), Cloudflare, "free"),
    "zai": Preset(("ZAI_API_KEY",), ZAI, "free"),
    "sambanova": Preset(("SAMBANOVA_API_KEY",), SambaNova, "free"),
    "nvidia": Preset(("NVIDIA_API_KEY",), NVIDIA, "free"),
    "kilo": Preset(("KILO_API_KEY",), Kilo, "free", keyless=True),
    "requesty": Preset(("REQUESTY_API_KEY",), Requesty, "free"),
    "ovh": Preset((), OVH, "free", keyless=True),
    "cohere": Preset(("COHERE_API_KEY",), Cohere, "trial"),
    "cerebras": Preset(("CEREBRAS_API_KEY",), Cerebras, "trial"),
    "mistral": Preset(("MISTRAL_API_KEY",), Mistral, "paid"),
}


def build_from_env(tiers: Iterable[Tier], include_keyless: bool) -> list[Provider]:
    """Providers for every preset whose environment variables are set (see ``from_env``)."""
    allowed = set(tiers)
    providers: list[Provider] = []
    for preset in PRESETS.values():
        if preset.tier not in allowed:
            continue
        keyed = bool(preset.env) and all(os.environ.get(name) for name in preset.env)
        if keyed or (preset.keyless and include_keyless):
            providers.append(preset.factory())
    if not providers:
        names = sorted({name for p in PRESETS.values() if p.tier in allowed for name in p.env})
        hint = "" if include_keyless else " (or pass include_keyless=True for anonymous endpoints)"
        raise FreeAIError(f"no provider API key found; set one of: {', '.join(names)}{hint}")
    return providers
