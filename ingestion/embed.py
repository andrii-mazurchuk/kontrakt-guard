"""Dense embeddings for the corpus and for queries.

Two things here are easy to get wrong and fail silently rather than loudly, which
is why both are enforced in one place:

1. **e5 requires asymmetric prefixes.** Passages are encoded with ``passage: ``
   and questions with ``query: ``. Omitting them, or using the same prefix for
   both, produces embeddings that still work — just worse. There is no error, only
   a lower recall@k that looks like a retrieval design problem.

2. **The model revision must be pinned.** ``retrieval_config_hash`` includes it,
   so an unpinned model silently invalidates comparisons the moment the upstream
   repository is re-uploaded.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from kontrakt_guard.config import Settings, get_settings

# e5 is trained for cosine similarity over L2-normalised vectors, which is also
# what the HNSW index is built for.
NORMALIZE = True


class Embedder:
    """Wraps the sentence-transformers model, loading it lazily.

    Lazy because importing this module must stay cheap: the chunker needs only
    the tokenizer, and the API needs neither until a query arrives.
    """

    def __init__(self, settings: Settings | None = None, batch_size: int = 16) -> None:
        self.settings = settings or get_settings()
        self.batch_size = batch_size

    @cached_property
    def _model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        revision = self.settings.embedding_revision or None
        return SentenceTransformer(
            self.settings.embedding_model,
            revision=revision,
            device=self.settings.embedding_device,
        )

    @cached_property
    def _tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.settings.embedding_model,
            revision=self.settings.embedding_revision or None,
        )

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 6; the old name still works but warns.
        getter = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        return int(getter())

    def count_tokens(self, text: str) -> int:
        """Exact token count for the chunker's budget, using the model's own tokenizer.

        `verbose=False` because measuring an over-long string is the normal way the
        chunker discovers it must split. Without it the tokenizer warns about
        exceeding the maximum sequence length on every such measurement, which
        reads as an error when it is the check working as intended.
        """
        return len(self._tokenizer.encode(text, add_special_tokens=True, verbose=False))

    def resolved_revision(self) -> str:
        """The commit actually loaded, for recording alongside metrics."""
        from huggingface_hub import model_info

        if self.settings.embedding_revision:
            return self.settings.embedding_revision
        return str(model_info(self.settings.embedding_model).sha)

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, question: str) -> list[float]:
        vector = self._model.encode(
            [question],
            batch_size=1,
            normalize_embeddings=NORMALIZE,
            convert_to_numpy=True,
        )[0]
        return list(vector.tolist())
