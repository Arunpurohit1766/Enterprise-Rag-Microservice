"""
FastAPI REST Endpoints for Knowledge Ingestion and Querying.
Enforces Ingress Security Perimeter:
- Scope checks ('knowledge:read', 'knowledge:write')
- Write authorization policies (clearance bounds, wildcard protection)
- Zero public token minting endpoints in production.
"""

from fastapi import APIRouter, Depends, status

from app.core.security import (
    get_current_security_context,
    require_scope,
    validate_write_authorization,
)
from app.schemas.rag import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    QueryRequest,
    RAGResponse,
    SecurityContext,
)
from app.services.ingestion.pipeline import ingestion_service
from app.services.rag_service import rag_service


router = APIRouter()


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and index a document under the authenticated tenant"
)
async def ingest_document(
    request: DocumentIngestRequest,
    ctx: SecurityContext = Depends(require_scope("knowledge:write"))
) -> DocumentIngestResponse:
    """
    Ingests, chunks, embeds, and indexes a document.
    Security Invariants:
    1. Caller must have 'knowledge:write' scope.
    2. Caller cannot ingest documents with classifications higher than their clearance.
    3. Caller cannot set wildcard '*' access without 'admin' role.
    4. Tenant identity is strictly derived from the authenticated JWT.
    """
    validate_write_authorization(request, ctx)
    return ingestion_service.ingest_document(request, ctx)


@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Execute secure hybrid retrieval and citation-verified RAG"
)
async def query_knowledge_base(
    request: QueryRequest,
    ctx: SecurityContext = Depends(require_scope("knowledge:read"))
) -> RAGResponse:
    """
    Executes the full enterprise RAG pipeline:
    1. Enforces 'knowledge:read' scope.
    2. Pre-retrieval authorization filter.
    3. Hybrid Search (Dense HNSW + Sparse BM25 + RRF).
    4. Evidence Gate (Answerability check).
    5. Structured Groq Generation.
    6. Post-generation citation verification.
    """
    return rag_service.execute_query(request, ctx)
