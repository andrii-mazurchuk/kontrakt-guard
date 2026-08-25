"""Record and replay Claude calls, so an eval can be re-run for $0.00.

A full Layer 1b run is ~1100 API calls and $3.23. Almost every reason to run it
again — a refactor, a renamed field, a test of the harness itself, a demo — has
nothing to do with what the model would say, and paying three dollars to find
that out again is a tax on touching the code.

So every request/response pair is written to an append-only JSONL file once, and
served from disk thereafter. Only a deliberate re-record costs money.

The design constraints worth knowing before changing anything here:

- **The key excludes `message.id`.** LangChain and LangGraph stamp fresh UUIDs on
  messages, so including them makes every key unique and every replay a total
  miss. `tests/test_cassette.py` names this trap explicitly.
- **Responses are stored as raw `AIMessage` fields, not as parsed objects.**
  Replay therefore runs langchain's real structured-output parser over the
  recorded content, rather than skipping it — which is what keeps a replayed run
  a test of the code and not merely of the cassette.
- **Serialisation is field by field, not `langchain_core.load.dumpd`.** The dumpd
  envelope is coupled to the langchain version and is opaque to read in a diff.
- **Every append is flushed under the lock, immediately.** `evals/qa_eval.py`
  carries a scar from a run that died at question 90 and lost the 89 already paid
  for; a cassette that serialised at exit would repeat exactly that.

This module deliberately does not import `langchain_anthropic`, so it can be
tested without touching the client, the network or an API key.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import subprocess
import sys
import threading
import warnings
from contextlib import suppress
from datetime import UTC, datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import IO, Any, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage

Mode = Literal["off", "record", "replay", "auto"]

# Bumped whenever `request_payload` changes shape. A cassette recorded under a
# different key version cannot be matched against, so it is a hard error rather
# than a silent stream of misses.
KEY_VERSION = 1

# Bumped whenever the on-disk entry or index shape changes.
CASSETTE_VERSION = 1

CALLS_FILE = "calls.jsonl"
INDEX_FILE = "index.json"

# Marks a message as having come from disk. Set on deserialise, never written to
# disk, and read by `graphs.llm.Usage.record` so that no call site has to know
# whether it is replaying.
REPLAY_FLAG = "_cassette_replayed"

# Replay never opens a socket, so the client only needs *a* key, not a real one.
# Substituting one also guarantees the real key cannot reach the cassette files.
PLACEHOLDER_API_KEY = "sk-ant-cassette-replay-no-network"

# How much of a message is shown in a miss report.
HINT_CHARS = 120


# The `noqa: N818` on both classes below: ruff wants an `Error` suffix, and
# `CassetteMissError` reads worse everywhere it is caught. These two names are
# part of the reviewed design, so the naming rule loses here rather than silently
# renaming an exception that other modules catch by name.
class CassetteMiss(RuntimeError):  # noqa: N818
    """Replay was asked for a call that was never recorded."""


class CassetteCorruption(RuntimeError):  # noqa: N818
    """The cassette contradicts itself and cannot be trusted to replay."""


# --- the canonical request ----------------------------------------------------


def request_payload(
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stop: list[str] | None,
    messages: list[BaseMessage],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Everything about a call that can change the response, and nothing else.

    `message.id` is excluded, and that exclusion is the single most load-bearing
    line in this module: langchain assigns random UUIDs to messages, so a key
    computed over them is different on every run and a cassette recorded with
    them replays at exactly 0%.

    `tools` and `tool_choice` are included because structured output is
    implemented as tool calling — a schema change is a request change, and a
    cassette recorded against the old schema must not be served for the new one.
    """
    return {
        "key_version": KEY_VERSION,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": list(stop) if stop else None,
        "tools": kwargs.get("tools"),
        "tool_choice": kwargs.get("tool_choice"),
        "messages": [
            {
                "type": message.type,
                "content": message.content,
                "name": message.name,
                "additional_kwargs": message.additional_kwargs,
            }
            for message in messages
        ],
    }


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cassette_key(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical request. Full 64 hex — never truncated.

    Truncation would trade a vanishing collision probability for a failure mode
    that surfaces as one question quietly answered with another question's
    recorded response, which is indistinguishable from a real result.
    """
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _content_sha(content: object) -> str:
    blob = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# --- message serialisation ----------------------------------------------------


def serialise_ai_message(message: AIMessage) -> dict[str, Any]:
    """An `AIMessage` reduced to the fields that survive a round trip.

    Explicit rather than `langchain_core.load.dumpd`: the dumpd envelope embeds
    the class path and a serialisation version, so a langchain upgrade can
    invalidate an otherwise perfectly good recording, and the result is not
    readable in a diff.
    """
    metadata = {k: v for k, v in message.response_metadata.items() if k != REPLAY_FLAG}
    return {
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
        "response_metadata": metadata,
        "usage_metadata": message.usage_metadata,
        "tool_calls": message.tool_calls,
        "invalid_tool_calls": message.invalid_tool_calls,
        "id": message.id,
    }


def deserialise_ai_message(blob: dict[str, Any]) -> AIMessage:
    """Rebuild the message, flagged so cost accounting knows it was free."""
    metadata = dict(blob.get("response_metadata") or {})
    metadata[REPLAY_FLAG] = True
    return AIMessage(
        content=blob.get("content", ""),
        additional_kwargs=dict(blob.get("additional_kwargs") or {}),
        response_metadata=metadata,
        usage_metadata=blob.get("usage_metadata"),
        tool_calls=list(blob.get("tool_calls") or []),
        invalid_tool_calls=list(blob.get("invalid_tool_calls") or []),
        id=blob.get("id"),
    )


# --- the cassette -------------------------------------------------------------


class Cassette:
    """One directory of recorded calls, opened for a single process.

    Thread-safe: the grading node fans out over chunks and `evals/qa_eval.py`
    fans out over questions, so both reads and appends happen concurrently.
    """

    def __init__(self, mode: Mode, directory: Path, name: str) -> None:
        self.mode: Mode = mode
        self.name = name
        self.root = directory / name
        self.calls_path = self.root / CALLS_FILE
        self.index_path = self.root / INDEX_FILE

        self._lock = threading.Lock()
        self._pools: dict[str, list[AIMessage]] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._system_hashes: dict[str, str] = {}
        self._served: dict[str, int] = {}
        self._handle: IO[str] | None = None

        self.hits = 0
        self.misses = 0
        self.appended = 0
        self.recorded_at = ""

        self._load()
        self._check_index()

        if mode in ("record", "auto"):
            self.root.mkdir(parents=True, exist_ok=True)
            self._handle = self.calls_path.open("a", encoding="utf-8", newline="\n")

    # -- loading ---------------------------------------------------------------

    def _load(self) -> None:
        if not self.calls_path.exists():
            return
        raw = self.calls_path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A process killed mid-write leaves a partial final line. The
                # rest of the cassette is intact and worth keeping.
                warnings.warn(
                    f"{self.calls_path}: dropping unparseable line {number} of {len(lines)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            self._absorb(entry)

    def _absorb(self, entry: dict[str, Any]) -> None:
        key = entry.get("key")
        request = entry.get("request")
        response = entry.get("response")
        if not isinstance(key, str) or not isinstance(request, dict) or response is None:
            warnings.warn(
                f"{self.calls_path}: dropping malformed entry", RuntimeWarning, stacklevel=2
            )
            return
        self._pools.setdefault(key, []).append(deserialise_ai_message(response))
        self._requests[key] = request
        self._system_hashes[key] = _system_hash(request)
        self.recorded_at = str(entry.get("recorded_at") or self.recorded_at)

    def _check_index(self) -> None:
        if not self.index_path.exists():
            return
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CassetteCorruption(f"{self.index_path} is not valid JSON: {exc}") from exc

        for field, current in (
            ("key_version", KEY_VERSION),
            ("cassette_version", CASSETTE_VERSION),
        ):
            recorded = index.get(field)
            if recorded is not None and int(recorded) != current:
                raise CassetteCorruption(
                    f"{self.name} was recorded under {field} {recorded}, current is {current} "
                    "— re-record it."
                )

        # Advisory only. A `kill -9` skips the index rewrite while the body is
        # already flushed, so a stale count describes a perfectly good cassette.
        counted = self.n_entries
        claimed = index.get("n_entries")
        if claimed is not None and int(claimed) != counted:
            warnings.warn(
                f"{self.index_path}: n_entries says {claimed}, body has {counted} "
                "(index is advisory; using the body)",
                RuntimeWarning,
                stacklevel=2,
            )

        environment = _environment()
        for field in (
            "model_cheap",
            "model_strong",
            "retrieval_config_hash",
            "corpus_manifest_sha",
        ):
            recorded = index.get(field)
            now = environment.get(field)
            if recorded and now and recorded != now:
                warnings.warn(
                    f"{self.name}: {field} was {recorded} at record time, now {now}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # -- serving ---------------------------------------------------------------

    @property
    def n_entries(self) -> int:
        return sum(len(pool) for pool in self._pools.values())

    @property
    def n_keys(self) -> int:
        return len(self._pools)

    def get(self, payload: dict[str, Any]) -> AIMessage | None:
        """The recorded response for this request, or None if never recorded.

        Always serves the *first* recorded response for a key, so two replays of
        the same cassette produce byte-identical runs.
        """
        key = cassette_key(payload)
        with self._lock:
            pool = self._pools.get(key)
            if not pool:
                self.misses += 1
                return None
            stored = self._requests.get(key)
            if stored is not None and stored != payload:
                raise CassetteCorruption(
                    f"key {key} was recorded for a different request.\n"
                    f"  recorded: {_canonical(stored)[:400]}\n"
                    f"  current:  {_canonical(payload)[:400]}"
                )
            self.hits += 1
            self._served[key] = self._served.get(key, 0) + 1
            return pool[0].model_copy(deep=True)

    def put(self, payload: dict[str, Any], message: AIMessage) -> None:
        """Append a response, unless an identical one is already pooled.

        A pool rather than a single slot because generation is not deterministic
        even at temperature 0: the same request really can produce two different
        answers, and pretending otherwise would silently discard one of them.
        """
        key = cassette_key(payload)
        blob = serialise_ai_message(message)
        entry = {
            "key": key,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "request": payload,
            "response": blob,
        }
        with self._lock:
            pool = self._pools.setdefault(key, [])
            if any(serialise_ai_message(existing) == blob for existing in pool):
                return
            pool.append(deserialise_ai_message(blob))
            self._requests[key] = payload
            self._system_hashes[key] = _system_hash(payload)
            self.appended += 1
            if self._handle is not None:
                self._handle.write(
                    json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                # Flushed here, not at exit: a run that dies at question 90 must
                # keep the 89 responses already paid for.
                self._handle.flush()

    # -- diagnostics -----------------------------------------------------------

    def nearest_hint(self, payload: dict[str, Any]) -> str:
        """Enough context to tell *why* a request missed, without guessing.

        Deliberately not a fuzzy matcher. The one distinction worth automating is
        "the system prompt changed" versus "this question was never recorded",
        and a count of entries sharing the system-message hash makes it.
        """
        messages = payload.get("messages") or []
        lines = [
            f"  model: {payload.get('model')}",
            f"  messages: {len(messages)}",
        ]
        for message in messages:
            content = message.get("content")
            preview = content if isinstance(content, str) else json.dumps(content, default=str)
            preview = preview.replace("\n", " ")[:HINT_CHARS]
            lines.append(f"    [{message.get('type')}] {_content_sha(content)} {preview}")

        system = _system_hash(payload)
        shared = sum(1 for value in self._system_hashes.values() if value == system)
        lines.append(
            f"  {shared} of {self.n_keys} recorded requests share this system-message hash "
            f"({system})"
        )
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        with self._lock:
            multi = sum(1 for pool in self._pools.values() if len(pool) > 1)
        return {
            "keys": self.n_keys,
            "entries": self.n_entries,
            "hits": self.hits,
            "misses": self.misses,
            "appended": self.appended,
            "keys_with_multiple_responses": multi,
        }

    def summary(self) -> str:
        stats = self.stats()
        line = (
            f"cassette {self.name} [{self.mode}]: {stats['hits']} replayed, "
            f"{stats['misses']} missed, {stats['appended']} newly recorded, "
            f"{stats['entries']} entries over {stats['keys']} keys"
        )
        if stats["keys_with_multiple_responses"]:
            line += f", {stats['keys_with_multiple_responses']} keys with >1 response"
        return line

    # -- closing ---------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            handle.flush()
            handle.close()
        if self.mode in ("record", "auto"):
            self._write_index()

    def _write_index(self) -> None:
        input_tokens = 0
        output_tokens = 0
        per_model: dict[str, tuple[int, int]] = {}
        for pool in self._pools.values():
            for message in pool:
                meta = message.usage_metadata
                if not meta:
                    continue
                model = str(message.response_metadata.get("model_name") or "")
                seen_in, seen_out = per_model.get(model, (0, 0))
                per_model[model] = (
                    seen_in + int(meta["input_tokens"]),
                    seen_out + int(meta["output_tokens"]),
                )
                input_tokens += int(meta["input_tokens"])
                output_tokens += int(meta["output_tokens"])

        index = {
            "cassette_version": CASSETTE_VERSION,
            "key_version": KEY_VERSION,
            "name": self.name,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **_environment(),
            "n_entries": self.n_entries,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "recorded_cost_usd": _cost_of(per_model),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )


def _system_hash(payload: dict[str, Any]) -> str:
    for message in payload.get("messages") or []:
        if message.get("type") == "system":
            return _content_sha(message.get("content"))
    return _content_sha(None)


def _cost_of(per_model: dict[str, tuple[int, int]]) -> float:
    """List price of what this cassette cost to record.

    Imported lazily so this module stays free of the Anthropic client, and
    treated as best-effort: a cassette that cannot price itself is still a
    perfectly usable cassette.
    """
    try:
        from graphs.llm import PRICES
    except Exception:  # pragma: no cover - only reachable on a broken install
        return 0.0
    total = 0.0
    for model, (tokens_in, tokens_out) in per_model.items():
        rate_in, rate_out = PRICES.get(model, max(PRICES.values()))
        total += (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000
    return total


@lru_cache(maxsize=1)
def _environment() -> dict[str, str]:
    """Provenance for the index, on a best-effort basis.

    Every lookup here can legitimately fail — no git, no corpus manifest, no
    `.env` — and none of them is worth failing a cassette over, so each degrades
    to an empty string rather than raising.
    """
    environment = dict.fromkeys(
        ("commit", "model_cheap", "model_strong", "retrieval_config_hash", "corpus_manifest_sha"),
        "",
    )
    with suppress(Exception):
        from kontrakt_guard.config import get_settings

        settings = get_settings()
        environment["model_cheap"] = settings.model_cheap
        environment["model_strong"] = settings.model_strong
        environment["retrieval_config_hash"] = settings.retrieval_config_hash()
    with suppress(Exception):
        from ingestion.manifest import load_manifest

        environment["corpus_manifest_sha"] = load_manifest().digest()
    with suppress(Exception):
        environment["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    return environment


# --- process-wide access ------------------------------------------------------
#
# The cassette is a process singleton keyed by its settings, because the seam
# that uses it is a `ChatAnthropic` subclass and adding a field to a pydantic
# model that langchain owns is a fight with `model_config` for no gain. The
# subclass instead asks this module for the cassette the process opened.

_active: tuple[str, str, str] | None = None
_active_lock = threading.Lock()
_opened: list[Cassette] = []


@cache
def get_cassette(mode: str, directory: str, name: str) -> Cassette | None:
    """The process's cassette for these settings, or None when recording is off."""
    if mode == "off":
        return None
    if mode not in ("record", "replay", "auto"):
        raise ValueError(f"unknown cassette mode: {mode!r}")

    cassette = Cassette(cast(Mode, mode), Path(directory), name)
    atexit.register(cassette.close)
    _opened.append(cassette)
    _banner(cassette)
    return cassette


def open_cassette(mode: str, directory: str, name: str) -> Cassette | None:
    """`get_cassette`, and remember the arguments as the process's active set."""
    cassette = get_cassette(mode, directory, name)
    if cassette is not None:
        global _active
        with _active_lock:
            _active = (mode, directory, name)
    return cassette


def active_cassette() -> Cassette | None:
    with _active_lock:
        arguments = _active
    return get_cassette(*arguments) if arguments is not None else None


def reset_cassette_cache() -> None:
    """Drop the singleton. For tests, which need one cassette per temp directory."""
    global _active
    while _opened:
        _opened.pop().close()
    get_cassette.cache_clear()
    _environment.cache_clear()
    with _active_lock:
        _active = None


def _banner(cassette: Cassette) -> None:
    """Loud, on stderr, every time. A replayed run must never look like a run."""
    recorded = cassette.recorded_at or "never"
    print(
        f"=== CASSETTE {cassette.mode} ({cassette.name}, {cassette.n_entries} entries, "
        f"recorded {recorded}) ===",
        file=sys.stderr,
    )
    if cassette.mode != "record":
        print(
            "=== These numbers are NOT a measurement and cannot be recorded. ===",
            file=sys.stderr,
        )
