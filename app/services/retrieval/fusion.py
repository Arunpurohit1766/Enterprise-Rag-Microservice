"""
Reciprocal Rank Fusion (RRF) Engine.
Merges disparate ranking spaces (Dense Cosine Similarity & Sparse BM25)
using the mathematically sound RRF formula with k=60.
"""

from typing import Dict, List
from app.core.config import settings
from app.schemas.rag import ChunkPayload, ScoredChunk


def compute_rrf(
    dense_candidates: List[ScoredChunk],
    sparse_candidates: List[ScoredChunk],
    k: int = settings.RRF_K
) -> List[ScoredChunk]:
    """
    Computes Reciprocal Rank Fusion:
        RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))

    Where rank is 1-indexed (1st place = 1, 2nd place = 2).
    A smoothing constant of k=60 prevents top-ranked items from dominating disproportionately.
    """
    fused_scores: Dict[str, float] = {}
    chunk_map: Dict[str, ChunkPayload] = {}
    dense_scores_map: Dict[str, float] = {}
    sparse_scores_map: Dict[str, float] = {}

    # Process Dense Ranks
    for rank_idx, item in enumerate(dense_candidates, start=1):
        cid = item.chunk.chunk_id
        chunk_map[cid] = item.chunk
        dense_scores_map[cid] = item.dense_score or 0.0
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank_idx))

    # Process Sparse Ranks
    for rank_idx, item in enumerate(sparse_candidates, start=1):
        cid = item.chunk.chunk_id
        chunk_map[cid] = item.chunk
        sparse_scores_map[cid] = item.sparse_score or 0.0
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k + rank_idx))

    # Sort descending by fused RRF score
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    # Build final ScoredChunk list with unified ranking
    final_ranked: List[ScoredChunk] = []
    for rank, cid in enumerate(sorted_chunk_ids, start=1):
        final_ranked.append(
            ScoredChunk(
                chunk=chunk_map[cid],
                dense_score=dense_scores_map.get(cid),
                sparse_score=sparse_scores_map.get(cid),
                rrf_score=round(fused_scores[cid], 6),
                rank=rank
            )
        )

    return final_ranked
