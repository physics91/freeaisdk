"""Named provider factories for ``FreeAI.from_env()``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._impl import (
    NVIDIA,
    OVH,
    ZAI,
    Cerebras,
    Cloudflare,
    Cohere,
    Gemini,
    Kilo,
    Mistral,
    Provider,
    Requesty,
    SambaNova,
    Tier,
)
from .groq import Groq
from .openrouter import OpenRouter


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
