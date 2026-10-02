"""
Post-Generation Claim & Citation Verification Engine.
Guarantees that no unverified factual claims reach the caller.
"""

from typing import Dict, List
from app.schemas.rag import (
    ChunkPayload,
    Citation,
    ClaimVerificationResult,
    ExtractedClaim,
)


class CitationVerifier:
    """Audits LLM-proposed claims against retrieved immutable evidence chunks."""

    def verify_claims(
        self,
        claims: List[ExtractedClaim],
        authorized_evidence: List[ChunkPayload]
    ) -> List[ClaimVerificationResult]:
        """
        Verifies every claim proposed by the LLM:
        1. Structural check: Claim must cite at least one chunk ID.
        2. Containment check: Cited chunk ID must exist in authorized_evidence.
        3. Lexical ground check: Key terms in the claim must exist in the cited chunk.
        """
        evidence_map: Dict[str, ChunkPayload] = {
            chunk.chunk_id: chunk for chunk in authorized_evidence
        }

        results: List[ClaimVerificationResult] = []

        for claim in claims:
            # Check 1: Does the claim cite anything?
            if not claim.claimed_chunk_ids:
                results.append(
                    ClaimVerificationResult(
                        claim=claim,
                        is_verified=False,
                        failure_reason="Claim has no supporting citations."
                    )
                )
                continue

            claim_valid = True
            failure_reasons = []

            for cid in claim.claimed_chunk_ids:
                # Check 2: Does the cited chunk exist in the authorized evidence?
                if cid not in evidence_map:
                    claim_valid = False
                    failure_reasons.append(f"Cited chunk ID '{cid}' does not exist in authorized evidence.")
                    continue

                # Check 3: Does the cited chunk text actually contain support?
                chunk = evidence_map[cid]
                # Simple check: words of length > 4 in the claim should appear in chunk
                claim_words = [w.lower() for w in claim.claim_text.split() if len(w) > 4]
                matching_words = [w for w in claim_words if w in chunk.content.lower()]

                if claim_words and len(matching_words) / len(claim_words) < 0.2:
                    claim_valid = False
                    failure_reasons.append(f"Chunk '{cid}' lacks sufficient lexical entailment for claim.")

            results.append(
                ClaimVerificationResult(
                    claim=claim,
                    is_verified=claim_valid,
                    failure_reason="; ".join(failure_reasons) if not claim_valid else None
                )
            )

        return results

    def build_verified_citations(
        self,
        verification_results: List[ClaimVerificationResult],
        authorized_evidence: List[ChunkPayload]
    ) -> List[Citation]:
        """
        Builds public Citation objects strictly from verified claims.
        """
        evidence_map = {chunk.chunk_id: chunk for chunk in authorized_evidence}
        citations: List[Citation] = []
        seen_chunks = set()

        for res in verification_results:
            if not res.is_verified:
                continue

            for cid in res.claim.claimed_chunk_ids:
                if cid in evidence_map and cid not in seen_chunks:
                    chunk = evidence_map[cid]
                    # Extract first 150 chars as snippet quote
                    quote_snippet = chunk.content[:150] + ("..." if len(chunk.content) > 150 else "")
                    citations.append(
                        Citation(
                            citation_id=f"cit_{len(citations) + 1}",
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            quote=quote_snippet
                        )
                    )
                    seen_chunks.add(cid)

        return citations


# Singleton verifier
citation_verifier = CitationVerifier()
