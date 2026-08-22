from __future__ import annotations

from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from graphs.qa_flow import Citation, QAState
from kontrakt_guard import __version__
from kontrakt_guard.api import main
from kontrakt_guard.api.main import app
from kontrakt_guard.config import Settings
from retrieval.search import Hit

client = TestClient(app)


def test_health_reports_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_health_does_not_require_a_database():
    """Liveness must not depend on Postgres, or a DB blip reads as a dead process."""
    r = client.get("/health")
    assert r.status_code == 200


def test_root_carries_the_disclaimer_in_both_languages():
    body = client.get("/").json()
    assert "not legal advice" in body["disclaimer_en"]
    assert "porada prawna" in body["disclaimer_pl"]


def test_openapi_schema_is_generated():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["version"] == __version__


# --- POST /ask ----------------------------------------------------------------


def _hit(article: str) -> Hit:
    return Hit(
        chunk_id=int(article),
        act="kp",
        article=article,
        article_display=f"Art. {article}",
        content="treść",
    )


def _stub_ask(state):
    """Replace the graph with a fixed state, so these test the HTTP layer only."""

    def fake(question, conn, embedder, settings, usage=None):
        if usage is not None:
            usage.cost_usd = 0.0123456
        return state

    return fake


@pytest.fixture
def grounded(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(anthropic_api_key="sk-test", postgres_password="x"),
    )
    monkeypatch.setattr(main, "connect", lambda settings: nullcontext(object()))
    monkeypatch.setattr(main, "_embedder", lambda: object())
    return monkeypatch


def test_ask_returns_the_answer_with_citations(grounded):
    grounded.setattr(
        main,
        "ask",
        _stub_ask(
            QAState(
                question="q",
                search_query="okres próbny",
                hits=[_hit("25"), _hit("26")],
                graded=[_hit("25")],
                answer="Okres próbny nie może przekraczać 3 miesięcy.",
                citations=[Citation(act="kp", article="25")],
                refused=False,
            )
        ),
    )
    body = client.post("/ask", json={"question": "czy 6 miesięcy próbnego jest legalne?"}).json()

    assert body["citations"] == [{"act": "kp", "article": "25"}]
    assert body["refused"] is False
    assert body["considered"] == 2 and body["grounded_in"] == 1
    assert "porada prawna" in body["disclaimer"]
    assert body["cost_usd"] == pytest.approx(0.012346)


def test_a_refusal_is_a_200_not_an_error(grounded):
    """ "I cannot ground this" is a successful outcome of the graph.

    Returning 4xx/5xx would push callers toward retrying rather than reading it.
    """
    grounded.setattr(
        main,
        "ask",
        _stub_ask(
            QAState(
                question="q",
                search_query="q",
                hits=[],
                graded=[],
                answer="Nie znalazłem…",
                citations=[],
                refused=True,
                refusal_reason="no chunk survived grading",
            )
        ),
    )
    r = client.post("/ask", json={"question": "jaka jest stolica Francji?"})

    assert r.status_code == 200
    assert r.json()["refused"] is True
    assert r.json()["refusal_reason"] == "no chunk survived grading"


def test_ask_without_an_api_key_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: Settings(anthropic_api_key=""))
    r = client.post("/ask", json={"question": "pytanie o urlop"})

    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_an_empty_question_is_rejected_before_any_billable_call():
    assert client.post("/ask", json={"question": "x"}).status_code == 422
