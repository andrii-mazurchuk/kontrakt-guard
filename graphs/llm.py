"""Claude clients and the cost accounting that makes a billable run reportable.

Every metrics row carries `api_cost_usd`, and a cost nobody counted is a cost
nobody can defend. Usage is accumulated per run rather than estimated afterwards
from call counts, because retries, cache hits and structured-output overhead all
move the real number away from the arithmetic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import SecretStr

from graphs.cassette import (
    PLACEHOLDER_API_KEY,
    REPLAY_FLAG,
    CassetteMiss,
    active_cassette,
    open_cassette,
    request_payload,
)
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

    # LIVE calls only. `cost_usd` is a statement about money actually spent, and
    # it stays one — a replayed call costs $0.00 and is counted separately.
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    # Calls served from a cassette. `avoided_cost_usd` is what the replayed calls
    # would have cost at list price, which is the argument for the cassette
    # existing — never a number that may be recorded as a run's cost.
    replayed_calls: int = 0
    replayed_input_tokens: int = 0
    replayed_output_tokens: int = 0
    avoided_cost_usd: float = 0.0

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
        cost = (read_in * rate_in + read_out * rate_out) / 1_000_000

        # Detected here rather than passed in, so no call site has to know
        # whether the process is replaying. `structured_call` is unchanged.
        replayed = bool(message.response_metadata.get(REPLAY_FLAG))

        with self._lock:
            if replayed:
                self.replayed_calls += 1
                self.replayed_input_tokens += read_in
                self.replayed_output_tokens += read_out
                self.avoided_cost_usd += cost
            else:
                self.input_tokens += read_in
                self.output_tokens += read_out
                self.calls += 1
                self.cost_usd += cost
            self.per_model[model] = self.per_model.get(model, 0) + 1

    def summary(self) -> str:
        models = ", ".join(f"{m} x{n}" for m, n in sorted(self.per_model.items()))
        line = (
            f"{self.calls} calls ({models}), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out, "
            f"${self.cost_usd:.4f}"
        )
        if self.replayed_calls:
            line += (
                f"  [+{self.replayed_calls} replayed, "
                f"{self.replayed_input_tokens:,} in / {self.replayed_output_tokens:,} out, "
                f"${self.avoided_cost_usd:.4f} avoided]"
            )
        return line


# Models that reject the `temperature` parameter outright, with
# "`temperature` is deprecated for this model." — a 400, not a warning. Found by
# a one-question smoke test before the full gold-set run, which is the entire
# argument for spending two cents before spending two dollars.
#
# An explicit set rather than a try/retry: it is greppable, and a new model that
# also rejects the parameter fails on the first call with a message that names
# the cause, rather than silently costing an extra round trip per call.
TEMPERATURE_UNSUPPORTED = frozenset({"claude-sonnet-5"})


def cheap_model(settings: Settings, temperature: float = 0.0) -> ChatAnthropic:
    """Haiku, for the high-volume steps: query rewriting and chunk grading."""
    return _model(settings.model_cheap, settings, temperature, max_tokens=1024)


def strong_model(settings: Settings, temperature: float = 0.0) -> ChatAnthropic:
    """Sonnet, for the answer itself and for faithfulness judging."""
    return _model(settings.model_strong, settings, temperature, max_tokens=2048)


class CassetteChatAnthropic(ChatAnthropic):
    """`ChatAnthropic` with `_generate` routed through a cassette.

    The seam is here — one level below `with_structured_output` — on purpose.
    Wrapping `structured_call` instead would leak (`rewrite_question` and the
    audit flow do not go through it) and would force recording *parsed* Pydantic
    objects, so replay would skip langchain's real parser and stop testing the
    thing it exists to test. Recording the raw `AIMessage` means a replayed run
    still parses, still validates, and still fails the same way a live one would.

    No pydantic field is added: the cassette is a process singleton fetched from
    `graphs.cassette`, which avoids declaring state on a model whose
    `model_config` langchain owns.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cassette = active_cassette()
        if cassette is None:
            return super()._generate(messages, stop, run_manager, **kwargs)

        payload = request_payload(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            messages=messages,
            kwargs=kwargs,
        )

        # `record` deliberately does not consult the cassette: re-recording is
        # meant to produce a genuinely live run, not a run half-served from the
        # tape it is replacing.
        if cassette.mode != "record":
            recorded = cassette.get(payload)
            if recorded is not None:
                return ChatResult(generations=[ChatGeneration(message=recorded)])

            if cassette.mode == "replay":
                # Never falls through to the network. A replay that quietly
                # bought the missing calls would produce a number nobody
                # budgeted for, from a cassette nobody has finished recording.
                raise CassetteMiss(
                    f"cassette '{cassette.name}' has no recorded response for this call.\n"
                    f"{cassette.nearest_hint(payload)}\n"
                    "  re-record with --cassette record, or top up with --cassette auto."
                )

        result = super()._generate(messages, stop, run_manager, **kwargs)
        message = result.generations[0].message
        if isinstance(message, AIMessage):
            cassette.put(payload, message)
        return result

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        # Loud rather than silent: an async path that quietly bypassed the
        # cassette would bill a run the operator believed was free.
        raise RuntimeError("cassette mode does not support async/streaming")

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cassette mode does not support async/streaming")


def _model(name: str, settings: Settings, temperature: float, max_tokens: int) -> ChatAnthropic:
    # Field names are those of the installed langchain-anthropic 1.x (`model`,
    # `max_tokens`, `default_request_timeout`). The 0.x names that most tutorials
    # use — `model_name`, `max_tokens_to_sample` — are not the same fields here.
    #
    # temperature=0 where the model still accepts it. Not because it makes
    # generation deterministic — it does not — but because a metric recorded
    # under sampling varies run to run for reasons unrelated to any change being
    # measured. Where the model rejects it, the parameter is omitted entirely:
    # langchain only sends it when set, so `None` is the correct absence.
    cassette = open_cassette(
        settings.cassette_mode, str(settings.cassette_dir), settings.cassette_name
    )
    if cassette is None:
        client = ChatAnthropic
        api_key = settings.anthropic_api_key
    else:
        client = CassetteChatAnthropic
        # Replay never opens a socket, so it must not require a key — and
        # substituting one guarantees the real key is nowhere near the tape.
        api_key = (
            SecretStr(PLACEHOLDER_API_KEY)
            if settings.cassette_mode == "replay"
            else settings.anthropic_api_key
        )

    return client(
        model=name,
        anthropic_api_key=api_key,
        temperature=None if name in TEMPERATURE_UNSUPPORTED else temperature,
        max_tokens=max_tokens,
        default_request_timeout=60.0,
    )
