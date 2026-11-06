"""In-process retry throttle for failover sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetryBudget:
    """gRPC-style retry throttle: server failures spend a token, successes earn ``token_ratio``.

    While fewer than half the tokens remain, a request stops failing over after a server failure,
    so an upstream outage does not turn every call into a sweep over every free endpoint.
    """

    max_tokens: float = 10.0
    token_ratio: float = 0.1
    tokens: float = field(default=-1.0)

    def __post_init__(self) -> None:
        if self.tokens < 0:
            self.tokens = self.max_tokens

    def allows_retry(self) -> bool:
        return self.tokens > self.max_tokens / 2

    def on_failure(self) -> None:
        self.tokens = max(0.0, self.tokens - 1.0)

    def on_success(self) -> None:
        self.tokens = min(self.max_tokens, self.tokens + self.token_ratio)
