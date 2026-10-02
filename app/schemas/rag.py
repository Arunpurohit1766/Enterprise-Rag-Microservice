from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# 1. Enums
class ClearanceLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class RAGResponseStatus(str, Enum):
    ANSWERED = "ANSWERED"
    PARTIALLY_ANSWERED = "PARTIALLY_ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    POLICY_DENIED = "POLICY_DENIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class QueryType(str, Enum):
    SEMANTIC = "SEMANTIC"
    IDENTIFIER = "IDENTIFIER"
    MIXED = "MIXED"


# 2. Security Context (from JWT)
class SecurityContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(..., description="User/Service ID")
    tenant_id: str = Field(..., min_length=1, max_length=64)
    roles: List[str] = Field(default_factory=list)
    groups: List[str] = Field(default_factory=list)
    clearance: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)
    scopes: List[str] = Field(default_factory=list)
    policy_version: str = Field(default="1.0.0")


# 3. Document Ingestion Schemas
class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=10, max_length=500_000)
    allowed_roles: List[str] = Field(default_factory=list)
    allowed_groups: List[str] = Field(default_factory=list)
    classification: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    version: int = Field(default=1)
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    content_hash: str
    token_count: int = Field(..., ge=1)
    tenant_id: str
    allowed_roles: List[str] = Field(default_factory=list)
    allowed_groups: List[str] = Field(default_factory=list)
    classification: ClearanceLevel = Field(default=ClearanceLevel.INTERNAL)
    security_policy_version: str = Field(default="1.0.0")


class DocumentIngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    version: int
    chunks_created: int
    tenant_id: str
    status: str = "INDEXED"
    processing_time_ms: float


# 4. Retrieval & Evidence Schemas
class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class ScoredChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: ChunkPayload
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: float
    rerank_score: Optional[float] = None
    rank: int = Field(..., ge=1)


class EvidenceGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_answerable: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reason: str


# 5. Generation & Verification Schemas
class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_text: str
    claimed_chunk_ids: List[str]


class ClaimVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: ExtractedClaim
    is_verified: bool
    failure_reason: Optional[str] = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    document_id: str
    quote: str


class TimingReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authentication_ms: float = 0.0
    authorization_filter_ms: float = 0.0
    query_classification_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    evidence_gate_ms: float = 0.0
    llm_generation_ms: float = 0.0
    citation_verification_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RAGResponseStatus
    query: str
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    reason_code: Optional[str] = None
    timings: TimingReceipt
    tenant_id: str
