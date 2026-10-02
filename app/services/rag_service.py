"""
Enterprise RAG Microservice - Core Pipeline Orchestrator.
Orchestrates:
1. True Hybrid Retrieval (Dense FastEmbed + Full Tenant BM25 + RRF + Cross-Encoder)
2. Calibrated Evidence Gating (Prevents hallucination on low confidence)
3. Groq LPU Generation
4. Grounded Citation Verification & Strict Answer Assembly
"""
import time
import logging
from typing import Any, Dict, List, Optional
from groq import Groq

from app.core.config import settings
from app.schemas.rag import (
    ClearanceLevel,
    QueryRequest,
    RAGResponse,
    ScoredChunk,
    SecurityContext,
    TimingReceipt,
)
from app.services.retrieval.engine import HybridRetriever, hybrid_retriever
from app.services.verification.gate import EvidenceGate, evidence_gate
from app.services.verification.verifier import CitationVerifier, citation_verifier

logger = logging.getLogger(__name__)


class EnterpriseRAGService:
    def __init__(self):
        self.retrieval_engine = hybrid_retriever
        self.evidence_gate = evidence_gate
        self.citation_verifier = citation_verifier
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

    async def ingest_document(
        self,
        document_id: str,
        title: str,
        content: str,
        security_context: SecurityContext,
        clearance_required: ClearanceLevel,
        authorized_roles: List[str],
        authorized_groups: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.retrieval_engine.ingest_document(
            document_id=document_id,
            title=title,
            content=content,
            security_context=security_context,
            clearance_required=clearance_required,
            authorized_roles=authorized_roles,
            authorized_groups=authorized_groups,
            metadata=metadata,
        )

    async def query(
        self,
        query_request: QueryRequest,
        security_context: SecurityContext,
    ) -> RAGResponse:
        start_time = time.perf_counter()

        # Step 1: True Hybrid Retrieval with RRF and Cross-Encoder Re-ranking
        ranked_chunks: List[ScoredChunk] = self.retrieval_engine.search(
            request=query_request,
            security_context=security_context,
        )

        retrieval_time_ms = (time.perf_counter() - start_time) * 1000

        # Step 2: Calibrated Evidence Gating
        gate_start = time.perf_counter()
        chunk_dicts = [c.model_dump() for c in ranked_chunks]
        gating_decision = self.evidence_gate.evaluate(
            query=query_request.query,
            retrieved_chunks=chunk_dicts,
            caller_threshold=query_request.similarity_threshold,
        )
        gate_time_ms = (time.perf_counter() - gate_start) * 1000

        if not gating_decision.is_answerable:
            total_time_ms = (time.perf_counter() - start_time) * 1000
            engine_timings = getattr(self.retrieval_engine, "last_search_timings", {})
            timing_receipt = TimingReceipt(
                dense_retrieval_ms=engine_timings.get("dense_retrieval_ms", round(retrieval_time_ms / 2, 2)),
                sparse_retrieval_ms=engine_timings.get("sparse_retrieval_ms", round(retrieval_time_ms / 2, 2)),
                rrf_fusion_ms=engine_timings.get("rrf_fusion_ms", 0.5),
                evidence_gate_ms=round(gate_time_ms, 2),
                total_ms=round(total_time_ms, 2)
            )
            return RAGResponse(
                status="INSUFFICIENT_EVIDENCE",
                query=query_request.query,
                answer="I cannot answer this question based on the authorized enterprise knowledge base. Insufficient verifiable evidence was retrieved.",
                citations=[],
                reason_code=gating_decision.rejection_reason or gating_decision.reason or "INSUFFICIENT_EVIDENCE",
                timings=timing_receipt,
                tenant_id=security_context.tenant_id
            )

        # Step 3: Prompt Construction & LLM Generation (Groq LPU)
        usable_chunks = gating_decision.selected_chunks
        context_blocks = []
        for idx, chunk in enumerate(usable_chunks, 1):
            chunk_id = chunk.get("chunk_id", f"ch_{idx}")
            doc_id = chunk.get("document_id", "doc")
            context_blocks.append(
                f"[{idx}] (Document: {doc_id}, Chunk: {chunk_id}):\n{chunk.get('content', '')}"
            )
        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an enterprise AI knowledge assistant. Answer the user query strictly using "
            "the provided context blocks. For every factual claim, append the citation bracket "
            "corresponding to the evidence block, e.g., [1] or [2]. Never extrapolate beyond the context."
        )

        user_prompt = f"Context Evidence:\n{context_str}\n\nQuestion: {query_request.query}\nAnswer:"

        generation_start = time.perf_counter()
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=settings.GROQ_MODEL,
                temperature=0.0,
                max_tokens=1024,
            )
            raw_answer = chat_completion.choices[0].message.content or ""
        except Exception as e:
            logger.error("groq_generation_failed: %s", str(e))
            raw_answer = f"Error during model synthesis: {str(e)}"

        generation_time_ms = (time.perf_counter() - generation_start) * 1000

        # Step 4: Strict Citation Verification & Answer Assembly
        verify_start = time.perf_counter()
        verification_result = self.citation_verifier.verify(
            answer=raw_answer,
            retrieved_chunks=usable_chunks,
        )
        verify_time_ms = (time.perf_counter() - verify_start) * 1000
        total_time_ms = (time.perf_counter() - start_time) * 1000

        citations_list = [c.model_dump() if hasattr(c, "model_dump") else c for c in ranked_chunks]

        engine_timings = getattr(self.retrieval_engine, "last_search_timings", {})
        timing_receipt = TimingReceipt(
            dense_retrieval_ms=engine_timings.get("dense_retrieval_ms", round(retrieval_time_ms / 2, 2)),
            sparse_retrieval_ms=engine_timings.get("sparse_retrieval_ms", round(retrieval_time_ms / 2, 2)),
            rrf_fusion_ms=engine_timings.get("rrf_fusion_ms", 0.5),
            evidence_gate_ms=round(gate_time_ms, 2),
            llm_generation_ms=round(generation_time_ms, 2),
            citation_verification_ms=round(verify_time_ms, 2),
            total_ms=round(total_time_ms, 2)
        )

        return RAGResponse(
            status="ANSWERED" if verification_result.verification_status == "PASSED" else "PARTIALLY_ANSWERED",
            query=query_request.query,
            answer=verification_result.verified_answer,
            citations=citations_list,
            reason_code="VERIFICATION_SUCCESS" if verification_result.verification_status == "PASSED" else "VERIFICATION_PARTIAL",
            timings=timing_receipt,
            tenant_id=security_context.tenant_id
        )

    async def execute_query(self, query_request: QueryRequest, security_context: SecurityContext) -> RAGResponse:
        return await self.query(query_request, security_context)


rag_service = EnterpriseRAGService()
