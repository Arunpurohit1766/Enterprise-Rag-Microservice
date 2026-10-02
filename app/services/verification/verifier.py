"""
Enterprise RAG Microservice - Hardened Citation Verifier
Performs sentence-level claim extraction, citation bounding validation,
and semantic-lexical grounding verification against retrieved context chunks.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "this", "that",
    "it", "its", "as", "from", "into", "through", "about", "which", "who"
}


class VerifiedClaim(BaseModel):
    claim_text: str
    cited_chunk_ids: List[str]
    is_supported: bool
    grounding_score: float
    reason: Optional[str] = None


class VerificationResult(BaseModel):
    verified_answer: str
    original_answer: str
    verification_status: str
    claims: List[VerifiedClaim] = Field(default_factory=list)
    unverified_claim_count: int = 0
    overall_grounding_score: float = 0.0
    citations_valid: bool = True


class CitationVerifier:
    MIN_CLAIM_GROUNDING_SCORE: float = 0.45

    def __init__(self, min_grounding_score: Optional[float] = None):
        self.min_grounding_score = min_grounding_score or self.MIN_CLAIM_GROUNDING_SCORE

    def _tokenize(self, text: str) -> set:
        tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        return {t for t in tokens if t not in STOP_WORDS}

    def _extract_sentences_and_citations(self, text: str) -> List[Dict[str, Any]]:
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        claims = []

        for sentence in raw_sentences:
            citations = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", sentence)
            clean_text = re.sub(r"\[[a-zA-Z0-9_\-]+\]", "", sentence).strip()
            claims.append({
                "raw_sentence": sentence,
                "clean_text": clean_text,
                "citations": citations
            })

        return claims

    def verify(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> VerificationResult:
        if not answer.strip():
            return VerificationResult(
                verified_answer="",
                original_answer="",
                verification_status="FAILED",
                claims=[],
                unverified_claim_count=0,
                overall_grounding_score=0.0,
                citations_valid=False
            )

        if not retrieved_chunks:
            return VerificationResult(
                verified_answer="",
                original_answer=answer,
                verification_status="FAILED",
                claims=[],
                unverified_claim_count=1,
                overall_grounding_score=0.0,
                citations_valid=False
            )

        chunk_map: Dict[str, str] = {}
        for idx, chunk in enumerate(retrieved_chunks, 1):
            chunk_content = chunk.get("content", "")
            chunk_map[str(idx)] = chunk_content
            chunk_id = str(chunk.get("id", chunk.get("chunk_id", "")))
            if chunk_id:
                chunk_map[chunk_id] = chunk_content
            doc_id = str(chunk.get("document_id", ""))
            if doc_id:
                chunk_map[doc_id] = chunk_content

        extracted = self._extract_sentences_and_citations(answer)
        verified_claims: List[VerifiedClaim] = []
        accepted_sentences: List[str] = []
        all_citations_valid = True

        for item in extracted:
            sentence = item["clean_text"]
            citations = item["citations"]
            claim_tokens = self._tokenize(sentence)

            if not claim_tokens:
                accepted_sentences.append(item["raw_sentence"])
                continue

            supporting_texts = []
            if citations:
                for cite in citations:
                    if cite in chunk_map:
                        supporting_texts.append(chunk_map[cite])
                    else:
                        all_citations_valid = False
            else:
                supporting_texts = [c.get("content", "") for c in retrieved_chunks]

            combined_evidence = " ".join(supporting_texts)
            evidence_tokens = self._tokenize(combined_evidence)
            overlap = claim_tokens.intersection(evidence_tokens)
            grounding_score = len(overlap) / max(len(claim_tokens), 1)
            is_supported = grounding_score >= self.min_grounding_score

            claim_obj = VerifiedClaim(
                claim_text=item["raw_sentence"],
                cited_chunk_ids=citations,
                is_supported=is_supported,
                grounding_score=round(grounding_score, 4),
                reason=None if is_supported else f"Insufficient grounding: {grounding_score:.2f} < {self.min_grounding_score}"
            )
            verified_claims.append(claim_obj)

            if is_supported:
                accepted_sentences.append(item["raw_sentence"])
            else:
                logger.warning("unverified_claim_detected: %s (score: %s)", item["raw_sentence"][:60], grounding_score)

        unverified_count = sum(1 for c in verified_claims if not c.is_supported)
        avg_score = (
            sum(c.grounding_score for c in verified_claims) / len(verified_claims)
            if verified_claims else 1.0
        )

        if unverified_count == 0 and all_citations_valid:
            status = "PASSED"
            verified_answer = answer
        elif len(accepted_sentences) > 0 and (len(accepted_sentences) / max(len(extracted), 1)) >= 0.5:
            status = "PARTIAL"
            verified_answer = " ".join(accepted_sentences)
        else:
            status = "FAILED"
            verified_answer = "VERIFICATION_FAILED: The generated answer contained claims not substantiated by retrieved evidence."

        return VerificationResult(
            verified_answer=verified_answer,
            original_answer=answer,
            verification_status=status,
            claims=verified_claims,
            unverified_claim_count=unverified_count,
            overall_grounding_score=round(avg_score, 4),
            citations_valid=all_citations_valid
        )



    def verify_claims(self, claims: list, retrieved_chunks: list) -> list:
        chunk_map = {}
        for c in retrieved_chunks:
            cid = getattr(c, "chunk_id", None) or getattr(c, "id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
            ctext = getattr(c, "content", None) or (c.get("content") if isinstance(c, dict) else "")
            if cid:
                chunk_map[str(cid)] = str(ctext)

        class ClaimResult:
            def __init__(self, is_v, score):
                self.is_verified = is_v
                self.confidence_score = score
                self.failure_reason = None if is_v else "Claim does not exist in authorized evidence"

        results = []
        for cl in claims:
            cid_list = getattr(cl, "claimed_chunk_ids", []) or getattr(cl, "cited_chunk_ids", [])
            ctext = getattr(cl, "claim_text", "")
            tokens = self._tokenize(ctext)

            valid = False
            for cid in cid_list:
                if str(cid) in chunk_map:
                    ev_tokens = self._tokenize(chunk_map[str(cid)])
                    if tokens and len(tokens.intersection(ev_tokens)) / len(tokens) >= self.min_grounding_score:
                        valid = True
                        break
            results.append(ClaimResult(valid, 0.9 if valid else 0.0))
        return results


citation_verifier = CitationVerifier()

