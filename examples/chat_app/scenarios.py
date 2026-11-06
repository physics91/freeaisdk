"""Free-tier platform profiles for load tests, shared by the mock upstream and the sample app.

The limits are the published free-tier numbers as of 2026-11-06 (see the README section on
providers). Gemini's per-model numbers come from a third-party source, and Z.ai's rule of one
concurrent request is unverified. Latency and token-rate profiles are assumptions, not measurements.
Each profile uses the real SDK preset class with ``base_url`` pointed at the mock, so the
preset's declared limits, payload rules and error classification are what gets exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from freeai import NVIDIA, OVH, ZAI, Cloudflare, Gemini, Groq, Kilo, OpenRouter, Provider, Requesty, SambaNova
from freeai.providers import (
    CLOUDFLARE_FREE_MODELS,
    GROQ_FREE_MODELS,
    NVIDIA_FREE_MODELS,
    OVH_ANONYMOUS_MODELS,
    SAMBANOVA_FREE_MODELS,
    ZAI_FREE_MODELS,
)

SCENARIOS = ("all", "keyless", "instant", "slow")


@dataclass(frozen=True)
class ModelLimits:
    rpm: float | None = None
    rpd: float | None = None
    tpm: float | None = None


@dataclass(frozen=True)
class Platform:
    name: str
    style: str  # error and header dialect: openrouter, groq, gemini, cloudflare, zai, kong, generic
    models: dict[str, ModelLimits]
    account_rpm: float | None = None
    account_rph: float | None = None
    account_rpd: float | None = None
    concurrency: int | None = None
    neurons_per_day: float | None = None  # Cloudflare's daily budget
    day_tz: str = "UTC"
    ttft: tuple[float, float] = (0.5, 1.5)  # seconds, uniform
    tps: tuple[float, float] | None = (40.0, 100.0)  # output tokens/second; None sends everything at once
    output_tokens: tuple[int, int] = (80, 300)
    error_rate: float = 0.02  # injected 503s
    keyed: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# Cloudflare neurons per million input/output tokens (official for these three, others estimated).
NEURONS = {
    "@cf/qwen/qwen3-30b-a3b-fp8": (4625, 30475),
    "@cf/openai/gpt-oss-120b": (31818, 68182),
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": (26668, 204805),
}
NEURONS_DEFAULT = (20000, 60000)

OPENROUTER_MODELS = (
    "meta-llama/llama-4-maverick:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "deepseek/deepseek-v4.1-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
)
KILO_MODELS = ("nvidia/nemotron-3.5-lightning:free", "cohere/north-mini-code:free", "reflection/beam-free")
REQUESTY_MODELS = ("google/gemma-4-31b-it", "nvidia/nemotron-3-super-120b-a12b", "novita/ling-3.1-flash")
GEMINI_LIMITS = {  # third-party numbers: Google no longer publishes free-tier limits
    "gemini-3.8-flash": ModelLimits(5, 20),
    "gemini-3.6-flash": ModelLimits(5, 20),
    "gemini-3.5-flash-lite": ModelLimits(10, 500),
    "gemini-3.1-flash-lite": ModelLimits(10, 500),
    "gemini-3-flash-preview": ModelLimits(5, 20),
}


GROQ_TEMPLATE_TOKENS = {"openai/gpt-oss-120b": 74, "openai/gpt-oss-20b": 74, "qwen/qwen3.8-27b": 15}
# Output reserve Groq pre-charged for requests without max_tokens, measured on 2026-11-06
# (reports/load-test/groq-reserve-probe.jsonl and groq-sequence-real.jsonl). gpt-oss reserves were
# mostly the length of one of the model's recent completions; qwen's showed no pattern.
# fmt: off
_GPT_OSS_RESERVES = (  # both gpt-oss models; 16 and 394 were seen on 120b, the rest on 20b
    16, 35, 35, 35, 35, 46, 234, 245, 275, 295, 359, 359, 359, 359, 359, 394,
    458, 484, 585, 593, 618, 636, 753, 789, 789, 789, 789, 831,
)
# fmt: on
_QWEN_RESERVES = (2, 155, 157, 174, 179, 274, 402, 528, 540, 746, 794, 797, 961)
GROQ_RESERVES = {
    "openai/gpt-oss-120b": _GPT_OSS_RESERVES,
    "openai/gpt-oss-20b": _GPT_OSS_RESERVES,
    "qwen/qwen3.8-27b": _QWEN_RESERVES,
}


def _each(models: tuple[str, ...], limits: ModelLimits) -> dict[str, ModelLimits]:
    return dict.fromkeys(models, limits)


def platforms(scenario: str) -> list[Platform]:
    ovh = Platform("ovh", "kong", _each(OVH_ANONYMOUS_MODELS, ModelLimits(rpm=2)), keyed=False)
    kilo = Platform("kilo", "generic", _each(KILO_MODELS, ModelLimits()), account_rph=200, keyed=False)
    if scenario == "keyless":
        return [ovh, kilo]
    if scenario == "instant":  # no latency, no limits, no errors: measures app and SDK overhead only
        models = tuple(f"model-{i}" for i in range(8))
        return [
            Platform(
                "instant",
                "generic",
                _each(models, ModelLimits()),
                ttft=(0, 0),
                tps=None,
                output_tokens=(40, 40),
                error_rate=0.0,
            )
        ]
    if scenario == "slow":  # realistic latency, no limits: how many concurrent streams the app holds
        models = tuple(f"model-{i}" for i in range(8))
        return [
            Platform(
                "slow",
                "generic",
                _each(models, ModelLimits()),
                ttft=(0.5, 1.5),
                tps=(50, 100),
                output_tokens=(80, 300),
                error_rate=0.0,
            )
        ]
    if scenario != "all":
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return [
        Platform(
            "openrouter",
            "openrouter",
            _each(OPENROUTER_MODELS, ModelLimits()),
            account_rpm=20,
            account_rpd=50,
            ttft=(0.8, 2.5),
            tps=(40, 90),
        ),
        Platform(
            "groq",
            "groq",
            _each(GROQ_FREE_MODELS, ModelLimits(rpm=30, rpd=1000, tpm=8000)),
            ttft=(0.15, 0.4),
            tps=(300, 600),
            # Measured 2026-11-06: chat template tokens per model, and the output reserves seen
            # when a request sets no max_tokens. gpt-oss draws from its recent completions once
            # it has some (see GROQ_RESERVES), qwen always from the measured list.
            extra={
                "template": GROQ_TEMPLATE_TOKENS,
                "reserve": GROQ_RESERVES,
                "recent_reserve": ("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
            },
        ),
        Platform("gemini", "gemini", GEMINI_LIMITS, day_tz="America/Los_Angeles", ttft=(0.5, 1.5), tps=(100, 200)),
        Platform(
            "cloudflare",
            "cloudflare",
            _each(CLOUDFLARE_FREE_MODELS, ModelLimits(rpm=300)),
            neurons_per_day=10_000,
            ttft=(0.4, 1.2),
            tps=(40, 100),
        ),
        Platform("zai", "zai", _each(ZAI_FREE_MODELS, ModelLimits()), concurrency=1, ttft=(0.8, 2.5), tps=(30, 70)),
        Platform(
            "sambanova",
            "generic",
            _each(SAMBANOVA_FREE_MODELS, ModelLimits(rpm=20, rpd=20)),
            ttft=(0.3, 0.8),
            tps=(150, 400),
        ),
        Platform(
            "nvidia", "generic", _each(NVIDIA_FREE_MODELS, ModelLimits()), account_rpm=40, ttft=(0.6, 2.0), tps=(30, 80)
        ),
        Platform(
            "requesty", "generic", _each(REQUESTY_MODELS, ModelLimits()), account_rpd=200, ttft=(0.8, 2.0), tps=(40, 90)
        ),
        kilo,
        ovh,
    ]


_PRESETS: dict[str, Any] = {
    "openrouter": OpenRouter,
    "groq": Groq,
    "gemini": Gemini,
    "cloudflare": Cloudflare,
    "zai": ZAI,
    "sambanova": SambaNova,
    "nvidia": NVIDIA,
    "requesty": Requesty,
    "kilo": Kilo,
    "ovh": OVH,
}


def build_providers(mock_url: str, scenario: str, **options: Any) -> list[Provider]:
    """SDK presets pointed at the mock: same classes, declared limits and classifiers as production.

    Model lists are pinned to the mock's, so catalog discovery is not exercised.
    """
    providers: list[Provider] = []
    for platform in platforms(scenario):
        url = f"{mock_url.rstrip('/')}/{platform.name}/v1"
        models = list(platform.models)
        preset = _PRESETS.get(platform.name)
        if preset is None:
            providers.append(Provider(platform.name, url, api_key="mock-key", models=models, tier="free", **options))
        else:
            key = "mock-key-" + platform.name if platform.keyed else None
            providers.append(preset(key, base_url=url, models=models, **options))
    return providers
