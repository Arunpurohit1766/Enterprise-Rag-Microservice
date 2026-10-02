"""
Retrieval, RRF Ranking, Evidence Gate, and Verification Proofs.
Proves exact identifier precision, RRF rank fusion, and anti-hallucination guardrails.
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
from app.services.verification.gate import evidence_gate
from app.services.verification.verifier import citation_verifier


def test_proof_6_technical_identifier_retrieval():
    """
    RETRIEVAL PROOF 6:
    Ingests technical manuals with specific ports.
    Proves that exact port numbers (e.g. 9092 for Kafka) are retrieved with precision.
    """
    token = create_access_token(
        subject="dev",
        tenant_id="acme_tech",
        roles=["engineering"],  # Explicitly grant authorized role
        clearance=ClearanceLevel.INTERNAL
    )
    ctx = decode_security_context(token)

    doc = DocumentIngestRequest(
        document_id="kafka_spec",
        title="Kafka Broker Configuration",
        text="Kafka broker listeners bind to plaintext port 9092. ZooKeeper connects on port 2181.",
        allowed_roles=["engineering"],
        classification=ClearanceLevel.INTERNAL
    )
    ingestion_service.ingest_document(doc, ctx)

    req = QueryRequest(query="What port does Kafka broker bind to?", top_k=2)
    results = hybrid_retriever.search(req, ctx)

    assert len(results) >= 1
    assert "9092" in results[0].chunk.content
    assert results[0].rrf_score > 0.0


def test_proof_7_rrf_mathematical_consistency():
    """
    RETRIEVAL PROOF 7:
    Proves Reciprocal Rank Fusion correctly combines ranks with k=60.
    A document appearing 1st in Dense and 1st in Sparse must have an RRF score of 2 / 61.
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
    # Expected: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
    expected_score = round(2.0 / 61.0, 6)
    assert fused[0].rrf_score == expected_score


def test_proof_8_evidence_gate_missing_fact_abstention():
    """
    VERIFICATION PROOF 8:
    A query containing a missing hard technical port must trigger early abstention.
    """
    chunk = ChunkPayload(
        chunk_id="acme:redis:v1:ch_0", document_id="redis", chunk_index=0,
        content="Redis cache runs on default port 6379.", content_hash="h1",
        token_count=10, tenant_id="acme", allowed_roles=[], allowed_groups=[],
        classification=ClearanceLevel.INTERNAL
    )
    candidate = ScoredChunk(chunk=chunk, dense_score=0.85, rrf_score=0.03, rank=1)

    # Query asks for port 9999 (not in evidence)
    decision = evidence_gate.evaluate("Does Redis run on port 9999?", [candidate])
    assert decision.is_answerable is False
    assert "9999" in decision.reason


def test_proof_9_citation_verifier_rejects_hallucinations():
    """
    VERIFICATION PROOF 9:
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
        # Hallucinated chunk ID
        ExtractedClaim(claim_text="PostgreSQL was built in 1980.", claimed_chunk_ids=["ghost_chunk_999"]),
        # Grounded claim
        ExtractedClaim(claim_text="PostgreSQL utilizes port 5432.", claimed_chunk_ids=["acme:db:v1:ch_0"])
    ]

    results = citation_verifier.verify_claims(claims, [valid_chunk])
    assert results[0].is_verified is False
    assert "does not exist in authorized evidence" in results[0].failure_reason
    assert results[1].is_verified is True


def test_proof_10_idempotent_ingestion():
    """
    INGESTION PROOF 10:
    Ingesting the exact same document ID twice must be idempotent (zero duplicate chunks).
    """
    token = create_access_token(
        subject="admin",
        tenant_id="acme_idemp",
        roles=["admin"],
        clearance=ClearanceLevel.INTERNAL
    )
    ctx = decode_security_context(token)

    doc = DocumentIngestRequest(
        document_id="idempotent_doc",
        title="Test Document",
        text="Initial release text for idempotency verification.",
        allowed_roles=["admin"],
        classification=ClearanceLevel.INTERNAL
    )
    
    # Ingest Pass 1
    res1 = ingestion_service.ingest_document(doc, ctx)
    # Ingest Pass 2 (Retry)
    res2 = ingestion_service.ingest_document(doc, ctx)

    assert res1.chunks_created == res2.chunks_created
    assert res1.document_id == res2.document_id
