"""Evaluation harness.

Two layers, evaluated independently so that a moving number is attributable to
retrieval or to judgement rather than to an undifferentiated blob:

- Layer 1 (``retrieval``): recall@k on ground-truth article IDs. Deterministic
  and near-free, so it runs as a per-pull-request gate.
- Layer 2 (``audit``): precision / recall / F1 on violation detection over the
  synthetic labeled contract set. LLM-heavy and billable, so it runs nightly or
  on explicit request.
"""
