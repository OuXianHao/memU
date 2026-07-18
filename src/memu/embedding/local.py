"""Local SentenceTransformer embedding client."""

from __future__ import annotations

import asyncio
from typing import Any


class LocalEmbeddingClient:
    """Embed text with a locally loaded SentenceTransformer model.

    This client matches the shared embedding-client interface while avoiding
    external API calls. Vectors are L2-normalized by SentenceTransformer.
    """

    def __init__(self, *, embed_model: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.embed_model = embed_model
        self.batch_size = batch_size
        self._model = SentenceTransformer(embed_model)

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], Any]:
        vectors = await asyncio.to_thread(
            self._model.encode,
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        if hasattr(vectors, "tolist"):
            return vectors.tolist(), None
        return [list(vector) for vector in vectors], None
