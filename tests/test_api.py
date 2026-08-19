from __future__ import annotations

from fastapi.testclient import TestClient

from kontrakt_guard import __version__
from kontrakt_guard.api.main import app

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
