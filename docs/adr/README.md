# Architecture Decision Records

One file per decision that would otherwise be re-litigated six weeks later, or that a reader
would reasonably question. Each records what was decided, what was rejected, and what would
justify revisiting it.

Format: context → decision → consequences → what would change our mind. Short on purpose; an ADR
nobody reads is worth nothing.

| # | Decision | Status |
|---|---|---|
| [0001](0001-pgvector-over-hosted-vector-db.md) | Postgres + pgvector as the only vector store | Accepted |
| [0002](0002-article-aware-chunking.md) | Chunk by legal article, not by token window | Accepted |
| [0003](0003-hybrid-retrieval-without-reranker.md) | Hybrid BM25 + dense, no reranker in v1 | Accepted |
| [0004](0004-file-based-metrics-tracking.md) | Metrics in a committed JSONL, not W&B or MLflow | Accepted |
| [0005](0005-artifact-pipeline-cloud-deploy-deferred.md) | Publish images now, defer the cloud target | Accepted |
| [0006](0006-trunk-based-branching.md) | Trunk-based branching with mandatory PRs | Accepted |
| [0007](0007-pending-amendments.md) | Ingest the law in force, exclude future-dated amendments | Accepted |
