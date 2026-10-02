"""
Query Normalization, Identifier Extraction, and Intent Classification Engine.
"""

import re
from typing import List, Set, Tuple
from app.schemas.rag import QueryType


# Regex patterns for high-value technical identifiers
IDENTIFIER_PATTERNS = [
    re.compile(r"\b\d{4,5}\b"),                         # Ports / 4-5 digit numbers (e.g. 51820, 6443)
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), # IPv4 Addresses
    re.compile(r"\b[A-Z0-9_\-]{4,}(?:_[A-Z0-9]+)+\b"),    # CONSTANTS / ERROR_CODES (e.g. ERR_CONNECTION_RESET)
    re.compile(r"\b[a-z0-9_\-]+\.[a-z]{2,4}\b"),         # File names / domains (e.g. config.yaml, server.py)
    re.compile(r"\b(?:v\d+|\d{4})\b", re.IGNORECASE),    # Versions or years (e.g. v2, 2026, 2027)
]


def extract_identifiers(query: str) -> Set[str]:
    """Extracts exact alphanumeric tokens, ports, IP addresses, and error codes."""
    extracted: Set[str] = set()
    for pattern in IDENTIFIER_PATTERNS:
        matches = pattern.findall(query)
        for match in matches:
            extracted.add(str(match).strip())
    return extracted


def classify_query(query: str) -> Tuple[QueryType, Set[str]]:
    """
    Classifies a query to tune retrieval weighting deterministically:
    - IDENTIFIER: Exact numbers/codes/ports present with short length
    - SEMANTIC: Broad conceptual / architectural queries
    - MIXED: Concepts combined with exact identifiers
    """
    identifiers = extract_identifiers(query)
    words = query.strip().split()

    if identifiers and len(words) <= 5:
        return QueryType.IDENTIFIER, identifiers
    elif identifiers:
        return QueryType.MIXED, identifiers
    else:
        return QueryType.SEMANTIC, identifiers
