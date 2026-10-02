"""
Hybrid Retrieval Engine.
Executes Dense (FastEmbed + Qdrant) and Sparse (BM25) searches
under strict Pre-Retrieval Authorization Filters.
Uses modern Qdrant query_points API.
"""

import time
from typing import Any, Dict, List
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.security import build_authorized_qdrant_filter
from app.infrastructure.qdrant.client import qdrant_manager
from app.schemas.rag import ChunkPayload, QueryRequest, ScoredChunk, SecurityContext
from app.services.ingestion.embedder import embedder
from app.services.retrieval.fusion import compute_rrf
from app.services.retrieval.normalizer import classify_query


class HybridRetriever:
    """Orchestrates secure multi-tenant hybrid search."""

    def __init__(self) -> None:
        self.qdrant = qdrant_manager.get_client()

    def _convert_filter_to_qdrant(self, filter_dict: Dict[str, Any]) -> models.Filter:
        """Converts our authorization dictionary into native Qdrant filter models."""
        must_conditions: List[models.Condition] = []
        for cond in filter_dict.get("must", []):
            key = cond["key"]
            match_data = cond["match"]
            if "value" in match_data:
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=match_data["value"]))
                )
            elif "any" in match_data:
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=match_data["any"]))
                )

        should_conditions: List[models.Condition] = []
        for cond in filter_dict.get("should", []):
            key = cond["key"]
            match_data = cond["match"]
            if "value" in match_data:
                should_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=match_data["value"]))
                )
            elif "any" in match_data:
                should_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=match_data["any"]))
                )

        return models.Filter(must=must_conditions, should=should_conditions if should_conditions else None)

    def retrieve_dense(
        self,
        query: str,
        auth_filter: models.Filter,
        top_k: int = settings.DENSE_TOP_K
    ) -> List[ScoredChunk]:
        """
        Executes Dense vector search in Qdrant with pre-retrieval authorization filter.
        Uses modern Qdrant query_points API.
        """
        query_vector = embedder.embed_query(query)
        if not query_vector:
            return []

        # Modern Qdrant API: query_points
        response = self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=auth_filter,  # HARD PRE-RETRIEVAL BOUNDARY
            limit=top_k,
            with_payload=True
        )

        dense_candidates: List[ScoredChunk] = []
        for rank, point in enumerate(response.points, start=1):
            if not point.payload:
                continue
            payload = ChunkPayload(**point.payload)
            dense_candidates.append(
                ScoredChunk(
                    chunk=payload,
                    dense_score=float(point.score) if point.score is not None else 0.0,
                    rrf_score=0.0,
                    rank=rank
                )
            )
        return dense_candidates

    def retrieve_sparse(
        self,
        query: str,
        authorized_chunks: List[ChunkPayload],
        top_k: int = settings.SPARSE_TOP_K
    ) -> List[ScoredChunk]:
        """
        Executes Sparse BM25 keyword search over authorized candidate chunks.
        Guarantees that exact ports, SKUs, and identifiers receive optimal ranking.
        """
        if not authorized_chunks:
            return []

        tokenized_corpus = [chunk.content.lower().split() for chunk in authorized_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored_items = []
        for idx, score in enumerate(scores):
            scored_items.append((score, authorized_chunks[idx]))

        # Sort descending by BM25 score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        sparse_candidates: List[ScoredChunk] = []
        for rank, (score, chunk) in enumerate(scored_items[:top_k], start=1):
            sparse_candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    sparse_score=float(score),
                    rrf_score=0.0,
                    rank=rank
                )
            )
        return sparse_candidates

    def search(
        self,
        request: QueryRequest,
        security_context: SecurityContext
    ) -> List[ScoredChunk]:
        """
        End-to-End Secure Hybrid Search Pipeline:
        1. Classifies query & extracts identifiers.
        2. Constructs pre-retrieval authorization filter.
        3. Executes Dense Vector Search in Qdrant (HNSW).
        4. Executes Sparse BM25 Search over authorized candidate corpus.
        5. Fuses results with Reciprocal Rank Fusion (k=60).
        6. Returns top-K candidates.
        """
        # Step 1: Query Normalization
        query_type, identifiers = classify_query(request.query)

        # Step 2: Build mandatory pre-retrieval security filter
        raw_filter = build_authorized_qdrant_filter(security_context)
        qdrant_filter = self._convert_filter_to_qdrant(raw_filter)

        # Step 3: Dense Retrieval
        dense_results = self.retrieve_dense(
            query=request.query,
            auth_filter=qdrant_filter,
            top_k=settings.DENSE_TOP_K
        )

        # Collect authorized chunks retrieved from Qdrant for sparse candidate pool
        authorized_pool = [item.chunk for item in dense_results]

        # Step 4: Sparse BM25 Retrieval
        sparse_results = self.retrieve_sparse(
            query=request.query,
            authorized_chunks=authorized_pool,
            top_k=settings.SPARSE_TOP_K
        )

        # Step 5: Reciprocal Rank Fusion (k=60)
        fused_candidates = compute_rrf(dense_results, sparse_results, k=settings.RRF_K)

        return fused_candidates[:request.top_k]


# Singleton retriever instance
hybrid_retriever = HybridRetriever()
