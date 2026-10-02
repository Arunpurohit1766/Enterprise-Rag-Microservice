"""
Qdrant Vector Database Infrastructure Layer.
Manages collection lifecycle, HNSW indexing, and payload index optimization for tenant isolation.
"""

from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings


class QdrantManager:
    """Manages the lifecycle and connection to the Qdrant vector engine."""

    def __init__(self) -> None:
        if settings.QDRANT_USE_IN_MEMORY:
            # Embedded in-memory Qdrant instance (fastest for development, CI/CD, and tests)
            self.client = QdrantClient(location=":memory:")
        else:
            # Production remote cluster / container connection
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
            )
        self.init_collection()

    def init_collection(self) -> None:
        """
        Creates the target collection if it does not exist,
        configures 384-dimensional cosine metric for FastEmbed (bge-small-en-v1.5),
        and creates payload indexes on critical filtering attributes.
        """
        collections_response = self.client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]

        if settings.QDRANT_COLLECTION_NAME not in existing_collections:
            # 1. Create collection with HNSW index tuned for cosine similarity
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                ),
                # Optimize HNSW for high precision and sub-second recall
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            )

            # 2. CRITICAL PRODUCTION STEP: Create payload indexes
            # Without payload indexes, filtered searches perform a slow full-collection scan.
            self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="allowed_roles",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="allowed_groups",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="classification",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def get_client(self) -> QdrantClient:
        """Returns the underlying initialized QdrantClient instance."""
        return self.client


# Singleton manager instance
qdrant_manager = QdrantManager()
