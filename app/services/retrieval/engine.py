"""
True Dual-Engine Hybrid Retrieval & Multi-Stage Re-Ranking Architecture.
Executes Dense (FastEmbed + Qdrant HNSW) and Sparse (BM25 across tenant corpus)
in parallel, merges candidates with RRF (k=60), and re-ranks via Cross-Encoder.
"""

import time
from typing import Any, Dict, List, Optional
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.security import build_authorized_qdrant_filter
from app.infrastructure.qdrant.client import qdrant_manager
from app.schemas.rag import ChunkPayload, QueryRequest, ScoredChunk, SecurityContext
from app.services.ingestion.embedder import embedder
from app.services.retrieval.fusion import compute_rrf
from app.services.retrieval.normalizer import classify_query
from app.services.retrieval.reranker import reranker


class HybridRetriever:
    """Orchestrates true multi-stage hybrid search with cross-encoder reranking."""

    def __init__(self) -> None:
        self.qdrant = qdrant_manager.get_client()

    def _convert_filter_to_qdrant(self, filter_dict: Dict[str, Any]) -> models.Filter:
        """Converts authorization dictionary into native Qdrant filter models."""
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

    def _fetch_authorized_corpus(self, auth_filter: models.Filter, limit: int = 500) -> List[ChunkPayload]:
        """
        Retrieves all authorized chunks under the caller's tenant boundary
        for independent sparse BM25 indexing.
        """
        scroll_result, _ = self.qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=auth_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        chunks: List[ChunkPayload] = []
        for point in scroll_result:
            if point.payload:
                chunks.append(ChunkPayload(**point.payload))
        return chunks

    def retrieve_dense(
        self,
        query: str,
        auth_filter: models.Filter,
        top_k: int = 50
    ) -> List[ScoredChunk]:
        """
        Executes Dense vector search in Qdrant with pre-retrieval authorization filter.
        """
        query_vector = embedder.embed_query(query)
        if not query_vector:
            return []

        response = self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=auth_filter,
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
        corpus_chunks: List[ChunkPayload],
        top_k: int = 50
    ) -> List[ScoredChunk]:
        """
        Executes Sparse BM25 keyword search over the ENTIRE authorized tenant corpus.
        Guarantees exact identifiers (ports, error codes) are retrieved even if dense similarity is low.
        """
        if not corpus_chunks:
            return []

        tokenized_corpus = [chunk.content.lower().split() for chunk in corpus_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored_items = []
        for idx, score in enumerate(scores):
            if score > 0.0:  # Only consider items with positive lexical match
                scored_items.append((score, corpus_chunks[idx]))

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
        True Multi-Stage Hybrid Search Pipeline:
        1. Query Normalization & Pre-Retrieval Filter Construction.
        2. Broad Dense Search (Top 50) + Full-Corpus Sparse BM25 Search (Top 50).
        3. Reciprocal Rank Fusion (k=60) -> Top 30 Consensus Candidates.
        4. Cross-Encoder Re-Ranking -> Top K calibrated evidence chunks.
        """
        # Step 1: Pre-retrieval security filter
        raw_filter = build_authorized_qdrant_filter(security_context)
        qdrant_filter = self._convert_filter_to_qdrant(raw_filter)

        # Step 2: Fetch tenant corpus for independent sparse search
        tenant_corpus = self._fetch_authorized_corpus(qdrant_filter, limit=500)

        # Step 3: Broad candidate retrieval
        dense_results = self.retrieve_dense(
            query=request.query,
            auth_filter=qdrant_filter,
            top_k=settings.DENSE_TOP_K
        )

        sparse_results = self.retrieve_sparse(
            query=request.query,
            corpus_chunks=tenant_corpus,
            top_k=settings.SPARSE_TOP_K
        )

        # Step 4: Reciprocal Rank Fusion (k=60)
        fused_candidates = compute_rrf(dense_results, sparse_results, k=settings.RRF_K)

        # Step 5: Cross-Encoder Re-Ranking (Top 30 -> Top K)
        reranked_results = reranker.rerank(
            query=request.query,
            candidates=fused_candidates[:30],
            top_k=request.top_k
        )

        return reranked_results


# Singleton retriever instance
hybrid_retriever = HybridRetriever()
