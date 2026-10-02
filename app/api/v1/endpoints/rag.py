"""
FastAPI REST Endpoints for Knowledge Ingestion, Querying, and Auth Simulation.
Enforces the Ingress Security Boundary:
Caller identity is strictly extracted from Bearer tokens via Dependency Injection.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.core.security import create_access_token, get_current_security_context
from app.schemas.rag import (
    ClearanceLevel,
    DocumentIngestRequest,
    DocumentIngestResponse,
    QueryRequest,
    RAGResponse,
    SecurityContext,
)
from app.services.ingestion.pipeline import ingestion_service
from app.services.rag_service import rag_service


router = APIRouter()


class TokenMintRequest(BaseModel):
    """Payload for minting test JWTs with custom tenant contexts."""
    subject: str = Field(default="user_test_1")
    tenant_id: str = Field(default="acme_corp")
    roles: List[str] = Field(default_factory=lambda: ["engineering"])
    groups: List[str] = Field(default_factory=lambda: ["core-team"])
    clearance: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)


class TokenMintResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    clearance: str


@router.post(
    "/auth/token",
    response_model=TokenMintResponse,
    summary="Mint test JWT with specific tenant and ABAC context"
)
async def mint_test_token(req: TokenMintRequest) -> TokenMintResponse:
    """
    Utility endpoint to issue cryptographically signed JWTs
    for integration testing and API consumer onboarding.
    """
    token = create_access_token(
        subject=req.subject,
        tenant_id=req.tenant_id,
        roles=req.roles,
        groups=req.groups,
        clearance=req.clearance
    )
    return TokenMintResponse(
        access_token=token,
        tenant_id=req.tenant_id,
        clearance=req.clearance.value
    )


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and index a document under the authenticated tenant"
)
async def ingest_document(
    request: DocumentIngestRequest,
    ctx: SecurityContext = Depends(get_current_security_context)
) -> DocumentIngestResponse:
    """
    Ingests, chunks, embeds, and indexes a document.
    Security Invariant: Tenant identity is stamped directly from the authenticated JWT.
    """
    return ingestion_service.ingest_document(request, ctx)


@router.post(
    "/query",
    response_model=RAGResponse,
    summary="Execute secure hybrid retrieval and citation-verified RAG"
)
async def query_knowledge_base(
    request: QueryRequest,
    ctx: SecurityContext = Depends(get_current_security_context)
) -> RAGResponse:
    """
    Executes the full enterprise RAG pipeline:
    1. Pre-retrieval authorization filter.
    2. Hybrid Search (Dense HNSW + Sparse BM25 + RRF).
    3. Evidence Gate (Answerability check).
    4. Structured Groq Generation.
    5. Post-generation citation verification.
    """
    return rag_service.execute_query(request, ctx)
