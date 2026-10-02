"""
Pre-Generation Evidence & Answerability Gate.
Evaluates whether retrieved evidence is sufficient to answer the user query
before invoking the LLM.
Features calibrated routing: hard identifiers (ports/error codes) vs soft identifiers (years).
"""

import re
from typing import List, Set
from app.core.config import settings
from app.schemas.rag import EvidenceGateDecision, QueryType, ScoredChunk
from app.services.retrieval.normalizer import classify_query


# Hard technical identifiers that must strictly be present
HARD_IDENTIFIER_PATTERNS = [
    re.compile(r"\b\d{4,5}\b"),                         # Ports (e.g. 51820, 6443)
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), # IPv4 Addresses
    re.compile(r"\b[A-Z0-9_\-]{4,}(?:_[A-Z0-9]+)+\b"),    # Error Codes (ERR_...)
]

# Soft identifiers (e.g. years) that can be overridden if semantic relevance is exceptionally high
YEAR_PATTERN = re.compile(r"\b(19\d\d|20\d\d)\b")


class EvidenceGate:
    """Evaluates answerability of candidate evidence."""

    def evaluate(
        self,
        query: str,
        candidates: List[ScoredChunk]
    ) -> EvidenceGateDecision:
        """
        Evaluates answerability:
        1. Emptiness check (zero candidates -> abstain).
        2. Relevance score cutoff.
        3. Differentiated identifier coverage (hard vs soft).
        """
        if not candidates:
            return EvidenceGateDecision(
                is_answerable=False,
                confidence_score=0.0,
                reason="No authorized evidence chunks retrieved for query."
            )

        top_candidate = candidates[0]
        top_score = top_candidate.dense_score if top_candidate.dense_score is not None else 0.0

        # Check 1: Minimum relevance cutoff
        if top_score < settings.ANSWERABILITY_THRESHOLD:
            return EvidenceGateDecision(
                is_answerable=False,
                confidence_score=round(top_score, 3),
                reason=f"Top candidate relevance score ({top_score:.3f}) below answerability threshold ({settings.ANSWERABILITY_THRESHOLD})."
            )

        all_evidence_text = " ".join([c.chunk.content for c in candidates]).lower()

        # Check 2: Hard Technical Identifiers (Ports, IPs, Error codes)
        for pattern in HARD_IDENTIFIER_PATTERNS:
            # Exclude 4-digit years from being treated as strictly hard ports
            matches = [m for m in pattern.findall(query) if not YEAR_PATTERN.match(m)]
            for match in matches:
                if match.lower() not in all_evidence_text:
                    return EvidenceGateDecision(
                        is_answerable=False,
                        confidence_score=0.2,
                        reason=f"Missing required technical identifier '{match}' in retrieved evidence."
                    )

        # Check 3: Soft Identifiers (Years)
        years = YEAR_PATTERN.findall(query)
        if years:
            missing_years = [y for y in years if y not in all_evidence_text]
            # If a year is missing, but semantic score is exceptionally high (>= 0.85), allow with warning
            if missing_years and top_score < 0.85:
                return EvidenceGateDecision(
                    is_answerable=False,
                    confidence_score=0.35,
                    reason=f"Query specifies temporal constraint {missing_years} not found in evidence."
                )

        # Evidence is sufficient to answer
        return EvidenceGateDecision(
            is_answerable=True,
            confidence_score=round(max(top_score, 0.75), 3),
            reason="Retrieved evidence satisfies relevance and identifier criteria."
        )


# Singleton evidence gate
evidence_gate = EvidenceGate()
