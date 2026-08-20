"""Corpus acquisition and article-aware parsing.

The corpus itself is never committed. `corpus_manifest.json` pins each act by
checksum; `fetch.py` reproduces the files from it. That keeps the repository
small while still making a corpus change visible as a diff.
"""
