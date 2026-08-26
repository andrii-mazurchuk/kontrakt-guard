"""Cassette record/replay. Free: no network, no database, no API key.

The seam under test is `graphs.llm.CassetteChatAnthropic._generate`, so the fake
"network" is a monkeypatched `ChatAnthropic._generate` — everything above it,
including langchain's real structured-output parser, runs for real. That is the
point of recording raw `AIMessage`s rather than parsed objects: a replayed call
still has to parse.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from graphs import cassette as cassette_module
from graphs.cassette import (
    Cassette,
    CassetteCorruption,
    CassetteMiss,
    cassette_key,
    deserialise_ai_message,
    request_payload,
    reset_cassette_cache,
    serialise_ai_message,
)
from graphs.llm import PRICES, Usage, cheap_model
from graphs.qa_flow import Grade, structured_call
from kontrakt_guard.config import Settings

CHEAP = "claude-haiku-4-5-20251001"
SENTINEL_KEY = "sk-ant-SENTINEL-do-not-record-me"


@pytest.fixture(autouse=True)
def _fresh_cassette():
    """The cassette is a process singleton; each test needs its own."""
    reset_cassette_cache()
    yield
    reset_cassette_cache()


def settings_for(tmp_path, mode: str, name: str = "t") -> Settings:
    return Settings(
        anthropic_api_key=SENTINEL_KEY,
        cassette_mode=mode,
        cassette_name=name,
        cassette_dir=tmp_path / "cassettes",
    )


def tool_message(reason: str = "wprost o tym stanowi", model: str = CHEAP) -> AIMessage:
    """What Claude returns for a structured `Grade` — tool call and all."""
    args = {"relevant": True, "reason": reason}
    return AIMessage(
        content=[{"type": "tool_use", "id": "toolu_1", "name": "Grade", "input": args}],
        tool_calls=[{"name": "Grade", "args": args, "id": "toolu_1", "type": "tool_call"}],
        response_metadata={"model_name": model, "stop_reason": "tool_use"},
        usage_metadata={"input_tokens": 800, "output_tokens": 40, "total_tokens": 840},
        id="msg_live",
    )


def fake_network(monkeypatch, message: AIMessage | None = None) -> list[Any]:
    """Replace the real HTTP call. Returns the list of calls that reached it."""
    seen: list[Any] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        seen.append(messages)
        reply = message if message is not None else tool_message()
        return ChatResult(generations=[ChatGeneration(message=reply)])

    monkeypatch.setattr(ChatAnthropic, "_generate", _generate)
    return seen


def grade_messages(question: str = "czy okres próbny może trwać 6 miesięcy?"):
    return [SystemMessage(content="Jesteś asystentem prawnym."), HumanMessage(content=question)]


# --- serialisation ------------------------------------------------------------


def test_an_ai_message_survives_the_round_trip():
    """Every field the flow reads downstream, including the ones easy to forget."""
    original = tool_message()
    restored = deserialise_ai_message(serialise_ai_message(original))

    assert restored.content == original.content
    assert restored.tool_calls == original.tool_calls
    assert restored.usage_metadata == original.usage_metadata
    assert restored.response_metadata["model_name"] == CHEAP
    assert restored.id == "msg_live"
    # The replay flag is added on the way out and never written on the way in.
    assert restored.response_metadata["_cassette_replayed"] is True
    assert "_cassette_replayed" not in serialise_ai_message(restored)["response_metadata"]


def test_list_of_blocks_content_is_preserved_exactly():
    blocks: list[str | dict[Any, Any]] = [
        {"type": "text", "text": "Art. 25 § 2"},
        {"type": "text", "text": "drugi blok"},
    ]
    restored = deserialise_ai_message(serialise_ai_message(AIMessage(content=blocks)))
    assert restored.content == blocks


# --- the key ------------------------------------------------------------------


def payload_for(messages, model: str = CHEAP, **kwargs):
    return request_payload(
        model=model, temperature=0.0, max_tokens=1024, stop=None, messages=messages, kwargs=kwargs
    )


def test_the_same_call_produces_the_same_key():
    assert cassette_key(payload_for(grade_messages())) == cassette_key(
        payload_for(grade_messages())
    )


def test_message_ids_are_excluded_from_the_key():
    """THE trap this whole module is built around.

    LangChain and LangGraph stamp a fresh random UUID on every message they
    construct. A key computed over `message.id` is therefore different on every
    single run, which turns a fully recorded cassette into a 100% miss rate —
    and, in `auto` mode, into paying full price for a run that appears cached.
    """
    first = grade_messages()
    second = grade_messages()
    first[0].id = "run-11111111-1111-1111-1111-111111111111"
    second[0].id = "run-22222222-2222-2222-2222-222222222222"
    first[1].id = "abc"
    second[1].id = "xyz"

    assert cassette_key(payload_for(first)) == cassette_key(payload_for(second))


def test_one_changed_character_changes_the_key():
    a = cassette_key(payload_for(grade_messages("czy okres próbny może trwać 6 miesięcy?")))
    b = cassette_key(payload_for(grade_messages("czy okres próbny może trwać 5 miesięcy?")))
    assert a != b


def test_a_different_schema_changes_the_key():
    """Structured output is tool calling, so a schema change IS a request change."""
    old = payload_for(grade_messages(), tools=[{"name": "Grade", "input_schema": {"x": 1}}])
    new = payload_for(grade_messages(), tools=[{"name": "Grade", "input_schema": {"x": 2}}])
    assert cassette_key(old) != cassette_key(new)


def test_a_different_model_changes_the_key():
    assert cassette_key(payload_for(grade_messages(), model=CHEAP)) != cassette_key(
        payload_for(grade_messages(), model="claude-sonnet-5")
    )


def test_the_key_is_never_truncated():
    assert len(cassette_key(payload_for(grade_messages()))) == 64


# --- record then replay -------------------------------------------------------


def test_a_replayed_run_reproduces_the_recorded_one_for_free(tmp_path, monkeypatch):
    """The headline claim: same parsed result, same per-model tally, $0.00."""
    seen = fake_network(monkeypatch)

    recording = Usage()
    model = cheap_model(settings_for(tmp_path, "record"))
    live = structured_call(model, Grade, grade_messages(), recording)

    assert isinstance(live, Grade)
    assert len(seen) == 1
    assert recording.cost_usd > 0

    reset_cassette_cache()
    replay_usage = Usage()
    replayed_model = cheap_model(settings_for(tmp_path, "replay"))
    replayed = structured_call(replayed_model, Grade, grade_messages(), replay_usage)

    # Parsed by langchain's own parser from the recorded raw message, not
    # deserialised from a stored Pydantic object.
    assert replayed == live
    assert len(seen) == 1, "replay must not reach the network"

    assert replay_usage.per_model == recording.per_model
    assert replay_usage.calls == 0
    assert replay_usage.cost_usd == 0.0
    assert replay_usage.replayed_calls == 1
    assert replay_usage.avoided_cost_usd == pytest.approx(recording.cost_usd)


def test_a_replay_miss_raises_and_never_calls_through(tmp_path, monkeypatch):
    seen = fake_network(monkeypatch)
    (tmp_path / "cassettes" / "t").mkdir(parents=True)
    (tmp_path / "cassettes" / "t" / "calls.jsonl").write_text("", encoding="utf-8")

    model = cheap_model(settings_for(tmp_path, "replay"))
    with pytest.raises(CassetteMiss) as caught:
        structured_call(model, Grade, grade_messages(), Usage())

    assert seen == [], "replay must never fall through to the network"
    # The hint distinguishes "prompt changed" from "question never recorded".
    assert "system-message hash" in str(caught.value)


def test_auto_mode_buys_only_the_misses(tmp_path, monkeypatch):
    seen = fake_network(monkeypatch)

    usage = Usage()
    model = cheap_model(settings_for(tmp_path, "auto"))
    structured_call(model, Grade, grade_messages(), usage)
    assert len(seen) == 1
    assert usage.calls == 1

    # Same call again, same process: now served from the pool it just appended.
    structured_call(model, Grade, grade_messages(), usage)
    assert len(seen) == 1
    assert usage.replayed_calls == 1

    reset_cassette_cache()
    replayed = Usage()
    replay_model = cheap_model(settings_for(tmp_path, "replay"))
    structured_call(replay_model, Grade, grade_messages(), replayed)
    assert len(seen) == 1
    assert replayed.replayed_calls == 1


# --- concurrency --------------------------------------------------------------


def test_thirty_two_threads_write_a_well_formed_file(tmp_path, monkeypatch):
    """Grading fans out over chunks and qa_eval fans out over questions.

    An unsynchronised append would interleave partial lines and corrupt the tape
    for every future run, which is the failure that only shows up under load.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        question = messages[-1].content
        return ChatResult(generations=[ChatGeneration(message=tool_message(f"reason-{question}"))])

    monkeypatch.setattr(ChatAnthropic, "_generate", _generate)

    questions = [f"pytanie numer {i}" for i in range(32)]
    model = cheap_model(settings_for(tmp_path, "record"))

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(
            pool.map(lambda q: structured_call(model, Grade, grade_messages(q), Usage()), questions)
        )

    path = tmp_path / "cassettes" / "t" / "calls.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    entries = [json.loads(line) for line in lines]
    assert len({e["key"] for e in entries}) == 32

    reset_cassette_cache()
    replay_model = cheap_model(settings_for(tmp_path, "replay"))

    def replay(question: str) -> str:
        graded = structured_call(replay_model, Grade, grade_messages(question), Usage())
        assert graded is not None
        return graded.reason

    with ThreadPoolExecutor(max_workers=32) as pool:
        reasons = list(pool.map(replay, questions))

    assert reasons == [f"reason-{q}" for q in questions]


# --- corruption ---------------------------------------------------------------


def write_calls(tmp_path, *lines: str) -> None:
    root = tmp_path / "cassettes" / "t"
    root.mkdir(parents=True, exist_ok=True)
    (root / "calls.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def test_a_request_contradicting_its_key_is_corruption(tmp_path):
    """Guards against a key-version mistake or a truncation bug served silently."""
    payload = payload_for(grade_messages())
    liar = {
        "key": cassette_key(payload),
        "recorded_at": "2026-08-23T00:00:00+00:00",
        "request": payload_for(grade_messages("zupełnie inne pytanie")),
        "response": serialise_ai_message(tool_message()),
    }
    write_calls(tmp_path, json.dumps(liar))

    tape = Cassette("replay", tmp_path / "cassettes", "t")
    with pytest.raises(CassetteCorruption) as caught:
        tape.get(payload)
    assert "recorded" in str(caught.value) and "current" in str(caught.value)
    tape.close()


def test_a_truncated_final_line_is_warned_about_and_dropped(tmp_path):
    """A `kill -9` mid-write must cost the last call, not the whole cassette."""
    payload = payload_for(grade_messages())
    good = json.dumps(
        {
            "key": cassette_key(payload),
            "recorded_at": "2026-08-23T00:00:00+00:00",
            "request": payload,
            "response": serialise_ai_message(tool_message()),
        }
    )
    write_calls(tmp_path, good, '{"key": "half-written", "reque')

    with pytest.warns(RuntimeWarning, match="unparseable line"):
        tape = Cassette("replay", tmp_path / "cassettes", "t")

    assert tape.n_entries == 1
    assert tape.get(payload) is not None
    tape.close()


def test_a_stale_index_count_is_a_warning_not_an_error(tmp_path):
    """kill -9 skips the index rewrite while the body is already flushed."""
    payload = payload_for(grade_messages())
    write_calls(
        tmp_path,
        json.dumps(
            {
                "key": cassette_key(payload),
                "recorded_at": "2026-08-23T00:00:00+00:00",
                "request": payload,
                "response": serialise_ai_message(tool_message()),
            }
        ),
    )
    index = tmp_path / "cassettes" / "t" / "index.json"
    index.write_text(
        json.dumps({"cassette_version": 1, "key_version": 1, "n_entries": 99}), encoding="utf-8"
    )

    with pytest.warns(RuntimeWarning, match="n_entries"):
        tape = Cassette("replay", tmp_path / "cassettes", "t")
    assert tape.n_entries == 1
    tape.close()


def test_a_key_version_mismatch_is_a_hard_error(tmp_path):
    write_calls(tmp_path)
    index = tmp_path / "cassettes" / "t" / "index.json"
    index.write_text(json.dumps({"cassette_version": 1, "key_version": 99}), encoding="utf-8")

    with pytest.raises(CassetteCorruption, match="re-record"):
        Cassette("replay", tmp_path / "cassettes", "t")


# --- secrets ------------------------------------------------------------------


def test_no_api_key_reaches_the_recorded_files(tmp_path, monkeypatch):
    """The bodies are gitignored, but index.json is committed. Neither may leak."""
    fake_network(monkeypatch)
    model = cheap_model(settings_for(tmp_path, "record"))
    structured_call(model, Grade, grade_messages(), Usage())
    reset_cassette_cache()  # closes the handle and writes index.json

    root = tmp_path / "cassettes" / "t"
    for name in ("calls.jsonl", "index.json"):
        assert SENTINEL_KEY not in (root / name).read_text(encoding="utf-8")
        assert "sk-ant" not in (root / name).read_text(encoding="utf-8")


# --- the default --------------------------------------------------------------


def test_the_cassette_is_off_by_default():
    """The replay seam must not be able to reach production via a changed default.

    `is`, not `isinstance`: the subclass would pass an isinstance check, and the
    whole point is that under default settings it is never constructed.
    """
    assert Settings().cassette_mode == "off"
    assert type(cheap_model(Settings())) is ChatAnthropic


def test_async_and_streaming_are_refused_loudly(tmp_path, monkeypatch):
    fake_network(monkeypatch)
    model = cheap_model(settings_for(tmp_path, "record"))
    with pytest.raises(RuntimeError, match="async/streaming"):
        model._stream(grade_messages())


def test_multiple_responses_for_one_key_are_pooled_and_reported(tmp_path):
    """Generation is not deterministic even at temperature 0."""
    tape = Cassette("record", tmp_path / "cassettes", "t")
    payload = payload_for(grade_messages())
    tape.put(payload, tool_message("pierwszy"))
    tape.put(payload, tool_message("drugi"))
    tape.put(payload, tool_message("pierwszy"))  # identical: not appended twice

    assert tape.stats()["entries"] == 2
    assert tape.stats()["keys_with_multiple_responses"] == 1
    # Replay always serves the first, so two replays are bit-identical.
    served = tape.get(payload)
    assert served is not None and served.tool_calls[0]["args"]["reason"] == "pierwszy"
    tape.close()


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown cassette mode"):
        cassette_module.get_cassette("rewind", "cassettes", "t")


def test_replay_restores_the_model_name_from_the_request(tmp_path):
    """Found by the first live recording, and it was not cosmetic.

    `model_name` is absent from the response entirely: langchain adds it to
    `response_metadata` after `_generate` returns, from the `llm_output` the
    real client builds. A replayed result carries no `llm_output`, so the
    enrichment never runs.

    `Usage` keys `per_model` on that field and prices an unknown model at the
    most expensive known rate, so 970 Haiku calls were billed as Sonnet and the
    reported avoided cost came out at $6.68 against a real $3.23.
    """
    cassette = Cassette("record", tmp_path / "cassettes", "m")
    payload = request_payload(
        "claude-haiku-4-5-20251001", 0.0, 1024, None, [HumanMessage(content="pytanie")], {}
    )
    # As Anthropic actually returns it: a provider, and no model name.
    recorded = AIMessage(
        content="ok",
        response_metadata={"model_provider": "anthropic"},
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    cassette.put(payload, recorded)
    cassette.close()

    replayed = Cassette("replay", tmp_path / "cassettes", "m").get(payload)
    assert replayed is not None
    assert replayed.response_metadata["model_name"] == "claude-haiku-4-5-20251001"

    usage = Usage()
    usage.record(replayed, str(replayed.response_metadata.get("model_name") or ""))
    assert usage.per_model == {"claude-haiku-4-5-20251001": 1}
    # Priced as Haiku, not at the unknown-model fallback rate.
    rate_in, rate_out = PRICES["claude-haiku-4-5-20251001"]
    assert usage.avoided_cost_usd == pytest.approx((10 * rate_in + 2 * rate_out) / 1_000_000)
