"""Claude clients and the cost accounting that makes a billable run reportable.

Every metrics row carries `api_cost_usd`, and a cost nobody counted is a cost
nobody can defend. Usage is accumulated per run rather than estimated afterwards
from call counts, because retries, cache hits and structured-output overhead all
move the real number away from the arithmetic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from kontrakt_guard.config import Settings

# USD per million tokens, as published for the pinned model ids. These are the
# one number here that lives outside the repository's control: if Anthropic
# changes list prices, a recorded api_cost_usd becomes an estimate under the old
# prices rather than a lie, which is why the model id is recorded beside it.
PRICES: dict[str, tuple[float, float]] = {
    # (input, output)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}


@dataclass
class Usage:
    """Token and cost totals for one run, aggregated across models.

    Thread-safe because the grading node fans out over chunks: independent
    judgements are the point of grading per chunk, and unsynchronised `+=` on a
    float would quietly lose some of them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    per_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, message: AIMessage, model: str) -> None:
        meta = message.usage_metadata
        read_in = int(meta["input_tokens"]) if meta else 0
        read_out = int(meta["output_tokens"]) if meta else 0

        # An unpriced model must not silently cost nothing. Charging the most
        # expensive known rate keeps the estimate conservative — a run that looks
        # cheaper than it was is the failure that matters here.
        rate_in, rate_out = PRICES.get(model, max(PRICES.values()))

        with self._lock:
            self.input_tokens += read_in
            self.output_tokens += read_out
            self.calls += 1
            self.cost_usd += (read_in * rate_in + read_out * rate_out) / 1_000_000
            self.per_model[model] = self.per_model.get(model, 0) + 1

    def summary(self) -> str:
        models = ", ".join(f"{m} x{n}" for m, n in sorted(self.per_model.items()))
        return (
            f"{self.calls} calls ({models}), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out, "
            f"${self.cost_usd:.4f}"
        )


def cheap_model(settings: Settings, temperature: float = 0.0) -> ChatAnthropic:
    """Haiku, for the high-volume steps: query rewriting and chunk grading."""
    return _model(settings.model_cheap, settings, temperature, max_tokens=1024)


def strong_model(settings: Settings, temperature: float = 0.0) -> ChatAnthropic:
    """Sonnet, for the answer itself and for faithfulness judging."""
    return _model(settings.model_strong, settings, temperature, max_tokens=2048)


def _model(name: str, settings: Settings, temperature: float, max_tokens: int) -> ChatAnthropic:
    # Field names are those of the installed langchain-anthropic 1.x (`model`,
    # `max_tokens`, `default_request_timeout`). The 0.x names that most tutorials
    # use — `model_name`, `max_tokens_to_sample` — are not the same fields here.
    #
    # temperature=0 throughout. Not because it makes generation deterministic —
    # it does not — but because a metric recorded under sampling varies run to
    # run for reasons unrelated to any change being measured.
    return ChatAnthropic(
        model=name,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        default_request_timeout=60.0,
    )
