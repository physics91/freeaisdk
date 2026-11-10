"""Groq free-tier provider."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any, ClassVar

from .._http import Headers
from ..errors import FailureKind, ProviderFailure, Scope
from ..limits import RateSnapshot, Window, otpm, rpd, rpm, tpd, tpm
from ._impl import _VERIFIED, Provider, Tier, Verdict

GROQ_FREE_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
_MEASURED = "2026-11-09"
_LIMIT_REACHED_RE = re.compile(r"\((RPM|OTPM)\): Limit (\d+), Used (\d+)")


def _without(message: Any, key: str) -> Any:
    if isinstance(message, dict) and key in message:
        return {k: v for k, v in message.items() if k != key}
    return message


class Groq(Provider):
    """Groq free plan: per-model limits (30 RPM, 1K RPD, 8K TPM, 200K TPD for the free models).

    ``x-ratelimit-*-requests`` headers describe the *daily* window and ``*-tokens`` the minute
    window; they overwrite the local counters when a request is accepted. Neither resets at a
    calendar boundary: ``x-ratelimit-reset-requests`` is 86.4 s per request used today. Measured on
    2026-11-06 (``reports/load-test/groq-sequence-real.jsonl``, ``groq-daily-refill-real.jsonl``),
    the daily count gets one request back per whole 86.4 s since the previous request, and the
    remainder is dropped when a request arrives: requests 13 s apart got nothing back, requests
    300 s apart always got 3. Under steady traffic the daily count therefore does not refill. The
    SDK treats the window as rolling and resyncs it from every response; once it is spent, the next
    request waits 86.4 s (plus ``EDGE_MARGIN``) after the last reading, when Groq has refilled one.
    ``retry-after`` (sent with 429s) is the only retry hint used.

    Accepting a request pre-charges its prompt (chat template included) plus ``max_tokens`` against
    the minute window and refunds the unused output when it ends. Measured on 2026-11-06 on the
    three free models (``reports/load-test/groq-reserve-probe.jsonl``): the template added 15 to 74
    tokens to a short prompt, hence ``prompt_overhead_tokens``. Without ``max_tokens`` the output
    reserve was 2 to 961 tokens: on gpt-oss mostly the length of a recent completion, on qwen with
    no pattern. It cannot be predicted, so the SDK keeps its generic 512; set ``max_tokens`` on Groq
    for an exact pre-charge. qwen also limits output tokens per minute (1000): one of 12 qwen
    requests without ``max_tokens`` got a 429 "Request too large" against it, while a request with
    ``max_tokens`` 1500 was accepted, so what triggers it is unknown. As Groq words it as a problem
    of the request, it skips the model for that request only.

    The minute's request count works like the daily one, one back per whole 2 s since the last
    accepted request with the remainder dropped, and a rejected request costs nothing and leaves that
    clock alone (``reports/load-test/groq-rpm-refill-real.jsonl``). Sends at 1.5 s passed
    ``x.x.x.x.``, 20 of 40; 16 sends at 3.5 s saved nothing for the burst after them. No header reports
    this window, so a 429's body is the only reading, and the preset declares it ``whole``.

    Output tokens per minute also have no header. They seem to refill continuously (the body's "try
    again in" matches 1000 per minute) and Groq refuses a request when its count plus an estimate would
    not fit. That estimate is not ``max_tokens``: it was 80 to 103 with ``max_tokens`` 256 on answers of
    about 50 tokens, hence ``output_reserve_tokens``. Only qwen was seen with this limit.
    """

    tier: Tier = "free"
    trains_on_prompts = False
    key_env = "GROQ_API_KEY"
    reset_headers = ()
    header_windows: ClassVar[Mapping[str, Window]] = {"requests": "day", "tokens": "minute"}
    prompt_overhead_tokens = 80
    output_reserve_tokens = 100
    default_models = GROQ_FREE_MODELS
    default_limits = (
        rpm(30, whole=True, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        rpd(1000, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        tpm(8000, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        tpd(200_000, source="console.groq.com/docs/rate-limits", last_verified=_VERIFIED),
        otpm(1000, models=("qwen/qwen3.8-27b",), source="429 response bodies", last_verified=_MEASURED),
    )
    drop_params = frozenset({"logprobs", "logit_bias", "top_logprobs"})

    def __init__(
        self,
        api_key: str | None = None,
        *,
        name: str = "groq",
        base_url: str = "https://api.groq.com/openai/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, base_url, api_key=api_key, **kwargs)

    def prepare_payload(self, model_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = super().prepare_payload(model_id, payload)
        messages = body.get("messages", [])
        if any(isinstance(message, dict) and "name" in message for message in messages):
            body["messages"] = [_without(message, "name") for message in messages]
        return body

    def rate_snapshots(self, headers: Mapping[str, str]) -> list[RateSnapshot]:
        return [dataclasses.replace(s, rolling=True) if s.per == "day" else s for s in super().rate_snapshots(headers)]

    def classify(self, status: int, headers: Mapping[str, str], body: Any) -> ProviderFailure:
        failure = super().classify(status, headers, body)
        match = _LIMIT_REACHED_RE.search(failure.message) if status == 429 else None
        if match:
            limit, used = float(match[2]), float(match[3])
            if match[1] == "RPM":
                failure.snapshots.append(RateSnapshot("minute", "requests", limit, limit - used, whole=True))
            else:
                failure.snapshots.append(RateSnapshot("minute", "output_tokens", limit, limit - used))
        return failure

    def _rate_limit_verdict(self, headers: Headers, error: dict[str, Any], text: str) -> Verdict:
        # "Request too large for model ... on tokens per minute (TPM)" or "... on output tokens per
        # minute (OTPM)" (a 429 seen on 2026-11-06): Groq says this request itself is too large and
        # asks to reduce it. Skip the model for this request instead of benching it for everyone.
        if "request too large" in text:
            return FailureKind.BAD_REQUEST, Scope.REQUEST
        return super()._rate_limit_verdict(headers, error, text)
