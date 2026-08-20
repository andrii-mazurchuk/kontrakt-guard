from __future__ import annotations

import json

import pytest

from ingestion.fetch import ChecksumMismatchError, fetch_act
from ingestion.manifest import load_manifest, save_manifest, sha256_file


def test_shipped_manifest_parses():
    manifest = load_manifest()
    assert [a.slug for a in manifest.acts] == ["kp"]


def test_kodeks_pracy_points_at_the_consolidated_text_not_the_1974_original():
    """DU/1974/141 serves clean HTML of the ORIGINAL 1974 text, which is superseded.

    Art. 1 there still refers to 'socjalistycznych stosunków pracy'. Pointing the
    corpus at it would ground every answer in socialist-era labour law, so the
    manifest must name the consolidated position instead.
    """
    kp = load_manifest().by_slug("kp")
    assert kp.eli == "DU/2025/277"
    assert kp.variant == "U"
    assert "1974/141" not in kp.url
    assert kp.url.endswith(".pdf")


def test_digest_changes_when_a_pin_changes(tmp_path):
    manifest = load_manifest()
    before = manifest.digest()
    manifest.acts[0].sha256 = "0" * 64
    assert manifest.digest() != before


def test_digest_ignores_fields_that_do_not_define_the_corpus():
    """Re-fetching identical bytes must not read as a corpus change."""
    manifest = load_manifest()
    before = manifest.digest()
    manifest.acts[0].fetched_at = "2099-01-01T00:00:00+00:00"
    manifest.acts[0].pages = 999
    assert manifest.digest() == before


def test_by_slug_names_what_it_knows():
    with pytest.raises(KeyError, match="known: kp"):
        load_manifest().by_slug("nope")


def test_save_round_trips_and_keeps_the_comment_block(tmp_path):
    path = tmp_path / "m.json"
    original = json.loads(
        load_manifest().model_dump_json()  # models only; rebuild the file with a comment block
    )
    path.write_text(
        json.dumps({"$comment": ["keep me"], "acts": original["acts"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = load_manifest(path)
    manifest.acts[0].pages = 190
    save_manifest(manifest, path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["$comment"] == ["keep me"]
    assert written["acts"][0]["pages"] == 190


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"kodeks")
    assert sha256_file(f) == sha256_file(f)
    assert len(sha256_file(f)) == 64


def test_checksum_mismatch_is_refused_without_repin(tmp_path, monkeypatch):
    manifest = load_manifest()
    act = manifest.acts[0]
    act.sha256 = "a" * 64  # a pin the download will not match

    payload = tmp_path / act.file_name
    monkeypatch.setattr("ingestion.fetch.download", lambda a, d=None: _write(payload, b"different"))

    with pytest.raises(ChecksumMismatchError, match="law changed"):
        fetch_act(act, repin=False, dest_dir=tmp_path)


def test_repin_accepts_new_bytes_and_records_them(tmp_path, monkeypatch):
    manifest = load_manifest()
    act = manifest.acts[0]
    act.sha256 = "a" * 64

    payload = tmp_path / act.file_name
    monkeypatch.setattr("ingestion.fetch.download", lambda a, d=None: _write(payload, b"different"))

    lines = fetch_act(act, repin=True, dest_dir=tmp_path)

    assert act.sha256 == sha256_file(payload)
    assert act.fetched_at is not None
    assert any("RE-PINNED" in line for line in lines)


def test_first_fetch_pins_without_needing_repin(tmp_path, monkeypatch):
    # model_copy rather than direct assignment: mypy narrows `act.sha256` to None
    # after `act.sha256 = None` and cannot see that fetch_act mutates it, which
    # makes the later comparison look statically unreachable.
    act = load_manifest().acts[0].model_copy(update={"sha256": None})

    payload = tmp_path / act.file_name
    monkeypatch.setattr("ingestion.fetch.download", lambda a, d=None: _write(payload, b"first"))

    lines = fetch_act(act, repin=False, dest_dir=tmp_path)

    assert act.sha256 == sha256_file(payload)
    assert any("first fetch" in line for line in lines)


def _write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
