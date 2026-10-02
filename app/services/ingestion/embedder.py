"""
Dense Embedding Engine using FastEmbed (ONNX Runtime).
Produces normalized 384-dimensional embeddings for BAAI/bge-small-en-v1.5.
"""

from typing import List
from fastembed import TextEmbedding

from app.core.config import settings


class Embedder:
    """Manages the FastEmbed model lifecycle for high-throughput CPU inference."""

    def __init__(self) -> None:
        self.model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of document chunk texts into dense vectors.
        Returns a list of 384-dimensional float vectors.
        """
        if not texts:
            return []
        embeddings_generator = self.model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query."""
        results = self.embed_documents([query])
        return results[0] if results else []


# Singleton embedder instance
embedder = Embedder()
