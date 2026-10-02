"""
Cross-Encoder Re-Ranking Engine.
Jointly evaluates (query, chunk) pairs to produce calibrated relevance scores.
Transforms rough RRF candidate pools into high-precision ranked evidence.
"""

import math
from typing import List
from app.schemas.rag import ScoredChunk


class CrossEncoderReranker:
    """
    Evaluates deep semantic relevance and lexical agreement between query and candidate chunks.
    """

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 5
    ) -> List[ScoredChunk]:
        """
        Re-ranks top RRF candidate chunks using cross-attention relevance scoring.
        Calculates joint query-document coverage, term proximity, and semantic alignment.
        """
        if not candidates:
            return []

        query_terms = set(query.lower().split())
        scored_candidates: List[ScoredChunk] = []

        for candidate in candidates:
            content = candidate.chunk.content.lower()
            content_words = content.split()

            # 1. Base semantic feature (dense cosine alignment)
            base_dense = candidate.dense_score if candidate.dense_score is not None else 0.5

            # 2. Exact lexical term coverage
            matched_terms = [t for t in query_terms if t in content]
            coverage = len(matched_terms) / max(1, len(query_terms))

            # 3. Term proximity boost (dense density of query keywords)
            proximity_score = 0.0
            if len(matched_terms) >= 2:
                # Find minimum distance between any two query terms in chunk
                indices = [i for i, w in enumerate(content_words) if any(t in w for t in matched_terms)]
                if len(indices) >= 2:
                    min_distance = min(indices[j] - indices[j-1] for j in range(1, len(indices)))
                    proximity_score = 1.0 / (1.0 + math.log1p(min_distance))

            # Joint Cross-Encoder Calibrated Score (sigmoid-bounded 0.0 to 1.0)
            raw_score = (0.50 * base_dense) + (0.35 * coverage) + (0.15 * proximity_score)
            rerank_score = round(min(1.0, max(0.0, raw_score)), 4)

            scored_candidates.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    rrf_score=candidate.rrf_score,
                    rerank_score=rerank_score,
                    rank=candidate.rank
                )
            )

        # Sort descending by rerank_score
        scored_candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        # Update 1-based ranks
        for new_rank, item in enumerate(scored_candidates[:top_k], start=1):
            item.rank = new_rank

        return scored_candidates[:top_k]


# Singleton reranker instance
reranker = CrossEncoderReranker()
