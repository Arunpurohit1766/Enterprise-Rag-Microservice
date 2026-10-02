"""
Enterprise RAG Microservice - Calibrated Evidence Gate Service
"""
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.core.config import settings

logger = logging.getLogger(__name__)

class GatingDecision(BaseModel):
    is_answerable: bool
    confidence_score: float
    selected_chunks: List[Any] = Field(default_factory=list)
    reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    telemetry: Dict[str, Any] = Field(default_factory=dict)

class EvidenceGate:
    MIN_ABSOLUTE_CONFIDENCE: float = 0.40

    def __init__(self, default_threshold: Optional[float] = None):
        self.default_threshold = default_threshold or getattr(settings, "SIMILARITY_THRESHOLD", 0.5)

    def _extract_chunk_text(self, c: Any) -> str:
        if hasattr(c, "chunk") and hasattr(c.chunk, "content"):
            return str(c.chunk.content)
        if hasattr(c, "content"):
            return str(c.content)
        if isinstance(c, dict):
            if "content" in c:
                return str(c["content"])
            if "chunk" in c and isinstance(c["chunk"], dict):
                return str(c["chunk"].get("content", ""))
        return ""

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        caller_threshold: Optional[float] = None
    ) -> GatingDecision:
        if not retrieved_chunks:
            return GatingDecision(
                is_answerable=False,
                confidence_score=0.0,
                selected_chunks=[],
                rejection_reason="NO_CHUNKS_RETRIEVED",
                telemetry={"retrieved_count": 0}
            )

        threshold = max(self.default_threshold, self.MIN_ABSOLUTE_CONFIDENCE)
        if caller_threshold is not None:
            threshold = max(threshold, caller_threshold)

        def _get_score(c: Any) -> float:
            if hasattr(c, "rerank_score") and c.rerank_score is not None:
                return float(c.rerank_score)
            if hasattr(c, "dense_score") and c.dense_score is not None:
                return float(c.dense_score)
            if hasattr(c, "score") and c.score is not None:
                return float(c.score)
            if isinstance(c, dict):
                return float(c.get("rerank_score") or c.get("dense_score") or c.get("score") or 0.0)
            return 0.0

        # Check for specific hard factual numbers/ports mentioned in query missing from retrieved evidence
        query_numbers = set(re.findall(r"[0-9]+", query))
        corpus_text = " ".join(self._extract_chunk_text(c) for c in retrieved_chunks)
        corpus_numbers = set(re.findall(r"[0-9]+", corpus_text))
        missing_numbers = query_numbers - corpus_numbers

        scored_chunks = sorted(retrieved_chunks, key=_get_score, reverse=True)
        top_score = _get_score(scored_chunks[0])
        qualified_chunks = [c for c in scored_chunks if _get_score(c) >= threshold]

        if missing_numbers:
            logger.info("evidence_gate_abstained: query contains numbers missing from evidence: %s", missing_numbers)
            return GatingDecision(
                is_answerable=False,
                confidence_score=0.0,
                selected_chunks=[],
                reason=f"Missing factual keywords: {missing_numbers}",
                rejection_reason=f"Missing factual keywords: {missing_numbers}",
                telemetry={"missing_numbers": list(missing_numbers)}
            )

        if top_score < threshold:
            return GatingDecision(
                is_answerable=False,
                confidence_score=top_score,
                selected_chunks=[],
                rejection_reason=f"Top chunk score ({top_score:.3f}) below threshold ({threshold:.3f})",
                telemetry={"top_score": top_score, "threshold": threshold}
            )

        return GatingDecision(
            is_answerable=True,
            confidence_score=top_score,
            selected_chunks=qualified_chunks,
            rejection_reason=None,
            telemetry={"top_score": top_score, "qualified_count": len(qualified_chunks)}
        )

evidence_gate = EvidenceGate()
