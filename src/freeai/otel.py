"""OpenTelemetry spans for every attempt (experimental; ``pip install "freeai[otel]"``).

One CLIENT span per attempt, named ``chat {model}``, with the GenAI semantic-convention attributes.
Those conventions are still in Development status, so the attribute names live only in this
module. Prompt and completion text are never recorded. Wrap calls in your own span to group the
attempts of one logical request.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from ._routing_types import _usage_number
from .types import Attempt

__all__ = ["attempt_attributes", "instrument"]

AttributeValue = str | bool | int | float


def attempt_attributes(attempt: Attempt) -> dict[str, AttributeValue]:
    """GenAI span attributes for one attempt (no OpenTelemetry import needed)."""
    attributes: dict[str, AttributeValue] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": attempt.provider,
        "gen_ai.request.model": attempt.model,
        "freeai.attempt.ok": attempt.ok,
    }
    if math.isfinite(attempt.latency):
        attributes["freeai.attempt.latency_s"] = attempt.latency
    if attempt.served_model:
        attributes["gen_ai.response.model"] = attempt.served_model
    usage = attempt.usage if isinstance(attempt.usage, Mapping) else {}
    for source, target in (
        ("prompt_tokens", "gen_ai.usage.input_tokens"),
        ("completion_tokens", "gen_ai.usage.output_tokens"),
    ):
        number = _usage_number(usage.get(source))
        if number is not None:
            attributes[target] = int(number)
    if not attempt.ok:
        attributes["error.type"] = str(attempt.kind or "unknown")
        if attempt.scope is not None:
            attributes["freeai.failure.scope"] = str(attempt.scope)
        attributes["freeai.cooldown_s"] = attempt.cooldown
    if attempt.status is not None:
        attributes["http.response.status_code"] = attempt.status
    return attributes


def instrument(ai: Any, *, tracer_provider: Any = None) -> None:
    """Emit a span per attempt from a :class:`~freeai.FreeAI` or :class:`~freeai.AsyncFreeAI`.

    Chains to any existing ``on_attempt`` callback. Requires ``opentelemetry-api``.
    """
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    tracer = trace.get_tracer("freeai", tracer_provider=tracer_provider)
    router = ai._router
    previous = router.on_attempt

    def emit(attempt: Attempt) -> None:
        end = time.time_ns()
        latency = attempt.latency if math.isfinite(attempt.latency) else 0.0
        span = tracer.start_span(
            f"chat {attempt.model}",
            kind=SpanKind.CLIENT,
            start_time=end - int(max(0.0, latency) * 1e9),
            attributes=attempt_attributes(attempt),
        )
        if not attempt.ok:
            span.set_status(Status(StatusCode.ERROR, attempt.message or str(attempt.kind)))
        span.end(end_time=end)
        if previous is not None:
            previous(attempt)

    router.on_attempt = emit
