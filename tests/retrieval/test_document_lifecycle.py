"""
Document Lifecycle & Versioning Proofs.
Proves that re-ingesting an updated document purges prior version chunks,
preventing ghost chunk retrieval when a document shrinks or updates.
"""
import pytest
from app.core.security import create_access_token, decode_security_context
from app.schemas.rag import ClearanceLevel, DocumentIngestRequest, QueryRequest
from app.services.ingestion.pipeline import ingestion_service
from app.services.retrieval.engine import hybrid_retriever


def test_proof_11_document_update_purges_ghost_chunks():
    """
    PROOF 11:
    Re-ingesting a document with smaller content must purge older chunks.
    A query for obsolete content in v1 must return ZERO chunks.
    """
    token = create_access_token(
        subject="ops_user",
        tenant_id="tenant_lifecycle_corp",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    ctx = decode_security_context(token)

    content_v1 = (
        "Paragraph 1: Service alpha runs on port 1111.\n\n"
        "Paragraph 2: Detailed architecture of service alpha.\n\n"
        "Paragraph 3: Network topologies and routing rules for alpha.\n\n"
        "Paragraph 4: Diagnostic metrics and legacy monitoring for alpha."
    )

    req_ingest_v1 = DocumentIngestRequest(
        document_id="service_alpha_config",
        title="Service Alpha Configuration v1",
        text=content_v1,
        classification=ClearanceLevel.INTERNAL,
        allowed_roles=["engineering"],
        allowed_groups=[]
    )
    ingestion_service.ingest_document(req_ingest_v1, ctx)

    # Verify v1 data is searchable
    req_v1 = QueryRequest(query="What is port 1111?", top_k=5, similarity_threshold=0.3)
    results_v1 = hybrid_retriever.search(req_v1, ctx)
    assert any("1111" in c.chunk.content for c in results_v1), "v1 content should be retrievable initially"

    # Ingest v2: Document shrinks to 1 concise paragraph with new port 2222
    content_v2 = "Service alpha has been migrated. The new official port is 2222."

    req_ingest_v2 = DocumentIngestRequest(
        document_id="service_alpha_config",
        title="Service Alpha Configuration v2",
        text=content_v2,
        classification=ClearanceLevel.INTERNAL,
        allowed_roles=["engineering"],
        allowed_groups=[]
    )
    ingestion_service.ingest_document(req_ingest_v2, ctx)

    # Query for new port: must be found
    req_v2 = QueryRequest(query="What is port 2222?", top_k=5, similarity_threshold=0.3)
    results_v2 = hybrid_retriever.search(req_v2, ctx)
    assert any("2222" in c.chunk.content for c in results_v2), "v2 content must be retrievable"

    # Query for obsolete port 1111: must NOT return any chunks from this document
    req_ghost = QueryRequest(query="service alpha port 1111", top_k=5, similarity_threshold=0.3)
    results_ghost = hybrid_retriever.search(req_ghost, ctx)
    for res in results_ghost:
        assert "1111" not in res.chunk.content, "Ghost chunk from v1 was retrieved after document update!"


def test_proof_12_tenant_document_isolation_during_purge():
    """
    PROOF 12:
    Purging document A during re-ingestion must not touch document B in the same tenant.
    """
    token = create_access_token(
        subject="ops_user_2",
        tenant_id="tenant_lifecycle_corp",
        roles=["engineering"],
        clearance=ClearanceLevel.INTERNAL,
        scopes=["knowledge:read", "knowledge:write"]
    )
    ctx = decode_security_context(token)

    req_b = DocumentIngestRequest(
        document_id="doc_beta_static",
        title="Document Beta",
        text="Beta static cluster host is beta-cluster.internal.net",
        classification=ClearanceLevel.INTERNAL,
        allowed_roles=["engineering"],
        allowed_groups=[]
    )
    ingestion_service.ingest_document(req_b, ctx)

    req_a1 = DocumentIngestRequest(
        document_id="doc_alpha_dynamic",
        title="Document Alpha",
        text="Alpha revision 1 content.",
        classification=ClearanceLevel.INTERNAL,
        allowed_roles=["engineering"],
        allowed_groups=[]
    )
    ingestion_service.ingest_document(req_a1, ctx)

    req_a2 = DocumentIngestRequest(
        document_id="doc_alpha_dynamic",
        title="Document Alpha",
        text="Alpha revision 2 content.",
        classification=ClearanceLevel.INTERNAL,
        allowed_roles=["engineering"],
        allowed_groups=[]
    )
    ingestion_service.ingest_document(req_a2, ctx)

    req_query_b = QueryRequest(query="beta-cluster.internal.net", top_k=3, similarity_threshold=0.3)
    results_b = hybrid_retriever.search(req_query_b, ctx)
    assert any("beta-cluster" in c.chunk.content for c in results_b), "Document B was corrupted during Document A update"
