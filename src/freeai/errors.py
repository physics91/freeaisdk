"""Failure classification and the exceptions raised by freeai."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .limits import RateSnapshot
    from .types import Attempt


class FailureKind(StrEnum):
    """Why a single attempt failed. Drives cooldown duration; :class:`Scope` drives what is benched."""

    RATE_LIMIT = "rate_limit"  # short-window limit (per minute), usually on one model
    PROVIDER_RATE_LIMIT = "provider_rate_limit"  # short-window limit shared by the whole account
    QUOTA_EXHAUSTED = "quota_exhausted"  # daily/monthly allowance used up: wait for the reset
    UNAVAILABLE = "unavailable"  # timeouts, network errors, 408, 5xx, capacity
    MODEL_GONE = "model_gone"  # 404: the model was removed or renamed
    NOT_ENTITLED = "not_entitled"  # this account/plan/privacy setting cannot use the model
    BAD_REQUEST = "bad_request"  # 400/413/422: this provider rejected the request
    CONTEXT_EXCEEDED = "context_exceeded"  # prompt too long for this model
    REFUSED = "refused"  # 403: moderation or guardrail block
    EMPTY = "empty"  # completion arrived without any content
    OUTPUT_LIMIT = "output_limit"  # max_tokens spent before any visible output (reasoning models)
    REJECTED = "rejected"  # the caller's reject_response hook refused an otherwise valid answer
    AUTH = "auth"  # 401, missing key, unverified project: the provider is unusable
    QUOTA = "quota"  # 402: no credits / negative balance


class Scope(StrEnum):
    """What a failure benches."""

    MODEL = "model"  # this model only
    PROVIDER = "provider"  # every model of the provider (account-wide condition)
    REQUEST = "request"  # nothing; skip the model for the current request only


DEFAULT_SCOPE: dict[FailureKind, Scope] = {
    FailureKind.RATE_LIMIT: Scope.MODEL,
    FailureKind.PROVIDER_RATE_LIMIT: Scope.PROVIDER,
    FailureKind.QUOTA_EXHAUSTED: Scope.MODEL,
    FailureKind.UNAVAILABLE: Scope.MODEL,
    FailureKind.MODEL_GONE: Scope.MODEL,
    FailureKind.NOT_ENTITLED: Scope.MODEL,
    FailureKind.BAD_REQUEST: Scope.REQUEST,
    FailureKind.CONTEXT_EXCEEDED: Scope.REQUEST,
    FailureKind.REFUSED: Scope.REQUEST,
    FailureKind.EMPTY: Scope.MODEL,
    FailureKind.OUTPUT_LIMIT: Scope.REQUEST,
    FailureKind.REJECTED: Scope.REQUEST,
    FailureKind.AUTH: Scope.PROVIDER,
    FailureKind.QUOTA: Scope.PROVIDER,
}

# Retrying the same model with the same request is pointless for these.
SKIP_FOR_REQUEST = frozenset(
    {
        FailureKind.BAD_REQUEST,
        FailureKind.CONTEXT_EXCEEDED,
        FailureKind.REFUSED,
        FailureKind.EMPTY,
        FailureKind.REJECTED,
    }
)


class FreeAIError(Exception):
    """Base class for every error raised by freeai."""


class BillingWarning(UserWarning):
    """A provider that may bill or consume trial credit was enabled."""


class ProviderFailure(FreeAIError):
    """One attempt against one model failed. Handled internally by the failover loop."""

    def __init__(
        self,
        kind: FailureKind,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        scope: Scope | None = None,
        snapshots: list[RateSnapshot] | None = None,
        reached: bool = True,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after = retry_after
        self.scope = scope or DEFAULT_SCOPE[kind]
        self.snapshots = snapshots or []  # limit state reported alongside the failure
        # False for transport failures before any HTTP response; local request quota is refundable.
        self.reached = reached


class ExhaustedError(FreeAIError):
    """No model could serve the request within the attempt and wait budget."""

    def __init__(
        self,
        reason: str,
        attempts: Iterable[Attempt],
        *,
        next_available_in: float | None = None,
    ) -> None:
        self.reason = reason
        self.attempts = list(attempts)
        self.next_available_in = next_available_in
        super().__init__(self._describe())

    def _describe(self) -> str:
        parts = [self.reason]
        if self.attempts:
            counts = Counter(str(a.kind) for a in self.attempts if not a.ok)
            parts.append("failures: " + ", ".join(f"{k}x{n}" for k, n in counts.items()))
            parts.append(f"last: {self.attempts[-1].message}")
        if self.next_available_in is not None:
            parts.append(f"next model available in {self.next_available_in:.1f}s")
        return "; ".join(parts)


class OutputLimitError(FreeAIError):
    """The model used its whole ``max_tokens`` budget (typically on reasoning) before answering.

    Retrying or switching models does not fix this; raise ``max_tokens`` or lower the reasoning effort.
    """

    def __init__(self, message: str, attempts: Iterable[Attempt], raw: dict[str, Any] | None = None) -> None:
        self.attempts = list(attempts)
        self.raw = raw
        super().__init__(message)


class StreamInterrupted(FreeAIError):
    """A stream failed after output was already delivered, so it cannot be moved to another model."""

    def __init__(self, failure: ProviderFailure, partial_text: str, attempts: Iterable[Attempt]) -> None:
        self.failure = failure
        self.partial_text = partial_text
        self.attempts = list(attempts)
        super().__init__(f"stream interrupted after partial output: {failure.message}")
