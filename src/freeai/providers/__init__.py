"""OpenAI-compatible providers and free-tier presets."""

from ._impl import CONNECTION_LIMITS, Completion, Provider, SnapshotSink, Tier, Verdict
from .gemini import GEMINI_FREE_MODELS, Gemini
from .groq import GROQ_FREE_MODELS, Groq
from .openrouter import OPENROUTER_BASE_URL, OpenRouter
from .platforms import (
    CEREBRAS_TRIAL_MODELS,
    CLOUDFLARE_FREE_MODELS,
    COHERE_TRIAL_MODELS,
    NVIDIA,
    NVIDIA_FREE_MODELS,
    OVH,
    OVH_ANONYMOUS_MODELS,
    SAMBANOVA_FREE_MODELS,
    ZAI,
    ZAI_FREE_MODELS,
    Cerebras,
    Cloudflare,
    Cohere,
    Kilo,
    Mistral,
    Requesty,
    SambaNova,
)
from .presets import PRESETS, Preset, cerebras, gemini, groq, mistral

__all__ = [
    "CEREBRAS_TRIAL_MODELS",
    "CLOUDFLARE_FREE_MODELS",
    "COHERE_TRIAL_MODELS",
    "CONNECTION_LIMITS",
    "GEMINI_FREE_MODELS",
    "GROQ_FREE_MODELS",
    "NVIDIA",
    "NVIDIA_FREE_MODELS",
    "OPENROUTER_BASE_URL",
    "OVH",
    "OVH_ANONYMOUS_MODELS",
    "PRESETS",
    "SAMBANOVA_FREE_MODELS",
    "ZAI",
    "ZAI_FREE_MODELS",
    "Cerebras",
    "Cloudflare",
    "Cohere",
    "Completion",
    "Gemini",
    "Groq",
    "Kilo",
    "Mistral",
    "OpenRouter",
    "Preset",
    "Provider",
    "Requesty",
    "SambaNova",
    "SnapshotSink",
    "Tier",
    "Verdict",
    "cerebras",
    "gemini",
    "groq",
    "mistral",
]
