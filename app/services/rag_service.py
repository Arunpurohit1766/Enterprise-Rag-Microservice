"""
End-to-End Enterprise RAG Orchestration Service.
Coordinates the full lifecycle under strict architectural invariants:
Authorization -> Retrieval -> Evidence Gate -> LLM Generation -> Verification -> Response.
"""

import time
from typing import List

from app.schemas.rag import (
    QueryRequest,
    RAGResponse,
    RAGResponseStatus,
    SecurityContext,
    TimingReceipt,
)
from app.services.generation.generator import generator_service
from app.services.retrieval.engine import hybrid_retriever
from app.services.verification.gate import evidence_gate
from app.services.verification.verifier import citation_verifier


class RAGOrchestrationService:
    """Coordinates multi-tenant hybrid retrieval and verified grounded generation."""

    def execute_query(
        self,
        request: QueryRequest,
        security_context: SecurityContext
    ) -> RAGResponse:
        """
        Executes the full enterprise RAG pipeline with millisecond telemetry.
        """
        timings = TimingReceipt()
        total_start = time.perf_counter()

        # -----------------------------------------------------------------
        # STAGE 1: HYBRID RETRIEVAL (Dense + Sparse BM25 + RRF)
        # Pre-retrieval authorization filter enforced inside Qdrant
        # -----------------------------------------------------------------
        retrieval_start = time.perf_counter()
        fused_candidates = hybrid_retriever.search(request, security_context)
        timings.rrf_fusion_ms = round((time.perf_counter() - retrieval_start) * 1000.0, 2)

        # -----------------------------------------------------------------
        # STAGE 2: PRE-GENERATION EVIDENCE & ANSWERABILITY GATE
        # Circuit breaker: Evaluates if evidence is sufficient to answer
        # -----------------------------------------------------------------
        gate_start = time.perf_counter()
        gate_decision = evidence_gate.evaluate(request.query, fused_candidates)
        timings.evidence_gate_ms = round((time.perf_counter() - gate_start) * 1000.0, 2)

        # DETERMINISTIC ABSTENTION: If evidence is insufficient, halt here!
        if not gate_decision.is_answerable:
            timings.total_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            return RAGResponse(
                status=RAGResponseStatus.INSUFFICIENT_EVIDENCE,
                query=request.query,
                answer=None,
                citations=[],
                reason_code=gate_decision.reason,
                timings=timings,
                tenant_id=security_context.tenant_id,
            )

        # -----------------------------------------------------------------
        # STAGE 3: STRUCTURED LLM GENERATION (Groq LPU)
        # -----------------------------------------------------------------
        gen_start = time.perf_counter()
        authorized_evidence = [item.chunk for item in fused_candidates]
        answer_text, extracted_claims = generator_service.generate_grounded_response(
            query=request.query,
            evidence=authorized_evidence
        )
        timings.llm_generation_ms = round((time.perf_counter() - gen_start) * 1000.0, 2)

        # -----------------------------------------------------------------
        # STAGE 4: POST-GENERATION CLAIM & CITATION VERIFICATION
        # Audits every claim against immutable retrieved evidence chunks
        # -----------------------------------------------------------------
        verify_start = time.perf_counter()
        verification_results = citation_verifier.verify_claims(
            claims=extracted_claims,
            authorized_evidence=authorized_evidence
        )
        verified_citations = citation_verifier.build_verified_citations(
            verification_results=verification_results,
            authorized_evidence=authorized_evidence
        )
        timings.citation_verification_ms = round((time.perf_counter() - verify_start) * 1000.0, 2)

        # Calculate total end-to-end latency
        timings.total_ms = round((time.perf_counter() - total_start) * 1000.0, 2)

        # If claims were produced but ALL failed verification -> REJECT!
        if extracted_claims and not verified_citations:
            return RAGResponse(
                status=RAGResponseStatus.VERIFICATION_FAILED,
                query=request.query,
                answer="Generated claims could not be verified against authorized evidence.",
                citations=[],
                reason_code="CITATION_VERIFICATION_FAILED",
                timings=timings,
                tenant_id=security_context.tenant_id,
            )

        # Determine terminal status
        all_passed = all(r.is_verified for r in verification_results) if verification_results else True
        terminal_status = RAGResponseStatus.ANSWERED if all_passed else RAGResponseStatus.PARTIALLY_ANSWERED

        return RAGResponse(
            status=terminal_status,
            query=request.query,
            answer=answer_text,
            citations=verified_citations,
            reason_code=None,
            timings=timings,
            tenant_id=security_context.tenant_id,
        )


# Singleton orchestrator
rag_service = RAGOrchestrationService()
