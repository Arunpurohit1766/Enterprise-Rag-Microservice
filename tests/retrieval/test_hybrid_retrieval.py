"""
Retrieval, RRF Ranking, Cross-Encoder Reranking, and Verification Proofs.
Proves true independent hybrid retrieval, RRF mathematical stability,
and cross-encoder precision.
"""

import pytest
from app.core.security import create_access_token, decode_security_context
from app.schemas.rag import (
    ChunkPayload,
    ClearanceLevel,
    DocumentIngestRequest,
    ExtractedClaim,
    QueryRequest,
    ScoredChunk,
)
from app.services.ingestion.pipeline import ingestion_service
from app.services.retrieval.engine import hybrid_retriever
from app.services.retrieval.fusion import compute_rrf
from app.services.retrieval.reranker import reranker
from app.services.verification.gate import evidence_gate
from app.services.verification.verifier import citation_verifier


def test_proof_6_independent_sparse_retrieval_rescue():
    """
    RETRIEVAL PROOF 6:
    Ingests technical documentation with exact identifier.
    Proves that independent BM25 search retrieves the identifier even across the full corpus.
    """
    token = create_access_token(
        subject="dev",
        tenant_id="acme_rescue",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL
    )
    ctx = decode_security_context(token)

    doc = DocumentIngestRequest(
        document_id="wireguard_exact_port",
        title="WireGuard Manual",
        text="The WireGuard network tunnel binds directly to UDP port 51820.",
        allowed_roles=["engineering"],
        classification=ClearanceLevel.INTERNAL
    )
    ingestion_service.ingest_document(doc, ctx)

    req = QueryRequest(query="What port does WireGuard bind to?", top_k=3)
    results = hybrid_retriever.search(req, ctx)

    assert len(results) >= 1
    assert "51820" in results[0].chunk.content
    assert results[0].rerank_score is not None
    assert results[0].rerank_score > 0.0


def test_proof_7_rrf_mathematical_consistency():
    """
    RETRIEVAL PROOF 7:
    Proves Reciprocal Rank Fusion correctly combines ranks with k=60.
    A document appearing 1st in Dense and 1st in Sparse has score 2 / 61.
    """
    dummy_chunk = ChunkPayload(
        chunk_id="test:1", document_id="doc1", chunk_index=0,
        content="Sample text", content_hash="hash", token_count=5,
        tenant_id="tenant", allowed_roles=[], allowed_groups=[],
        classification=ClearanceLevel.INTERNAL
    )
    dense_cand = [ScoredChunk(chunk=dummy_chunk, dense_score=0.9, rrf_score=0.0, rank=1)]
    sparse_cand = [ScoredChunk(chunk=dummy_chunk, sparse_score=12.5, rrf_score=0.0, rank=1)]

    fused = compute_rrf(dense_cand, sparse_cand, k=60)
    assert len(fused) == 1
    expected_score = round(2.0 / 61.0, 6)
    assert fused[0].rrf_score == expected_score


def test_proof_8_cross_encoder_reranking_order():
    """
    RETRIEVAL PROOF 8:
    Proves Cross-Encoder correctly promotes chunks with high query keyword coverage.
    """
    c1 = ChunkPayload(
        chunk_id="c1", document_id="d1", chunk_index=0,
        content="General networking overview without specific ports.",
        content_hash="h1", token_count=6, tenant_id="t", allowed_roles=[],
        allowed_groups=[], classification=ClearanceLevel.INTERNAL
    )
    c2 = ChunkPayload(
        chunk_id="c2", document_id="d2", chunk_index=0,
        content="Kubernetes API server port 6443 configuration.",
        content_hash="h2", token_count=6, tenant_id="t", allowed_roles=[],
        allowed_groups=[], classification=ClearanceLevel.INTERNAL
    )

    items = [
        ScoredChunk(chunk=c1, dense_score=0.80, rrf_score=0.03, rank=1),
        ScoredChunk(chunk=c2, dense_score=0.75, rrf_score=0.02, rank=2)
    ]

    reranked = reranker.rerank("Kubernetes API server port 6443", items, top_k=2)
    # c2 must be promoted to rank 1 because it has exact lexical and semantic match for 6443
    assert reranked[0].chunk.chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_proof_9_evidence_gate_missing_fact_abstention():
    """
    VERIFICATION PROOF 9:
    A query containing a missing hard technical port must trigger early abstention.
    """
    chunk = ChunkPayload(
        chunk_id="acme:redis:v1:ch_0", document_id="redis", chunk_index=0,
        content="Redis cache runs on default port 6379.", content_hash="h1",
        token_count=10, tenant_id="acme", allowed_roles=[], allowed_groups=[],
        classification=ClearanceLevel.INTERNAL
    )
    candidate = ScoredChunk(chunk=chunk, dense_score=0.85, rrf_score=0.03, rank=1)

    decision = evidence_gate.evaluate("Does Redis run on port 9999?", [candidate])
    assert decision.is_answerable is False
    assert "9999" in decision.reason


def test_proof_10_citation_verifier_rejects_hallucinations():
    """
    VERIFICATION PROOF 10:
    If the LLM cites a non-existent chunk ID or produces an unsupported claim,
    the verifier rejects it with is_verified=False.
    """
    valid_chunk = ChunkPayload(
        chunk_id="acme:db:v1:ch_0", document_id="db", chunk_index=0,
        content="PostgreSQL utilizes port 5432.", content_hash="hdb",
        token_count=8, tenant_id="acme", allowed_roles=[], allowed_groups=[],
        classification=ClearanceLevel.INTERNAL
    )

    claims = [
        ExtractedClaim(claim_text="PostgreSQL was built in 1980.", claimed_chunk_ids=["ghost_chunk_999"]),
        ExtractedClaim(claim_text="PostgreSQL utilizes port 5432.", claimed_chunk_ids=["acme:db:v1:ch_0"])
    ]

    results = citation_verifier.verify_claims(claims, [valid_chunk])
    assert results[0].is_verified is False
    assert "does not exist in authorized evidence" in results[0].failure_reason
    assert results[1].is_verified is True
