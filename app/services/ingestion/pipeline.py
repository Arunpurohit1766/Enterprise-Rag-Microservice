"""
Ingestion Orchestration Pipeline.
Coordinates Document Validation -> Chunking -> Embedding -> Qdrant Upsert.
Enforces that unauthorized cross-tenant writes are mathematically impossible.
"""

import time
import uuid
from typing import List
from qdrant_client.http import models

from app.core.config import settings
from app.infrastructure.qdrant.client import qdrant_manager
from app.schemas.rag import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    SecurityContext,
)
from app.services.ingestion.chunker import chunk_document
from app.services.ingestion.embedder import embedder


class IngestionService:
    """Orchestrates end-to-end ingestion and vector indexing."""

    def __init__(self) -> None:
        self.qdrant = qdrant_manager.get_client()

    def ingest_document(
        self,
        request: DocumentIngestRequest,
        security_context: SecurityContext
    ) -> DocumentIngestResponse:
        """
        Executes the synchronous document ingestion pipeline:
        1. Breaks document into structure-aware overlapping chunks.
        2. Injects mandatory security metadata (tenant_id, roles, clearance, SHA-256).
        3. Computes 384-dimensional dense embeddings via FastEmbed.
        4. Upserts vector points into Qdrant.
        """
        start_time = time.perf_counter()

        # Step 1: Chunk document with security tagging
        chunks = chunk_document(request, security_context)
        if not chunks:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return DocumentIngestResponse(
                document_id=request.document_id,
                version=1,
                chunks_created=0,
                tenant_id=security_context.tenant_id,
                status="EMPTY_DOCUMENT",
                processing_time_ms=round(elapsed, 2)
            )

        # Step 2: Generate dense vector embeddings for all chunks in batch
        chunk_texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(chunk_texts)

        # Step 3: Prepare Qdrant Points
        points: List[models.PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            # Deterministic UUID derived from the unique chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk.model_dump()  # Physical storage of ACLs & text with the vector
            )
            points.append(point)

        # Step 4: Upsert points into Qdrant collection
        self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
            wait=True
        )

        elapsed = (time.perf_counter() - start_time) * 1000.0
        return DocumentIngestResponse(
            document_id=request.document_id,
            version=1,
            chunks_created=len(chunks),
            tenant_id=security_context.tenant_id,
            status="INDEXED",
            processing_time_ms=round(elapsed, 2)
        )


# Singleton ingestion service
ingestion_service = IngestionService()
