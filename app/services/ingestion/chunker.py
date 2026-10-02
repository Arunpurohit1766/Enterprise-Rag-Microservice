"""
Structure-Aware Sliding Window Chunking Engine.
Preserves sentences, code blocks, and cross-chunk context via sliding overlap.
Computes SHA-256 hashes and tags mandatory ACL security metadata.
"""

import hashlib
import re
from typing import List

from app.schemas.rag import ChunkPayload, DocumentIngestRequest, SecurityContext


def compute_sha256(text: str) -> str:
    """Computes a deterministic SHA-256 hex digest for chunk content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_tokens_approx(text: str) -> int:
    """Fast approximation of token count based on whitespace and punctuation."""
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, len(words))


def split_into_semantic_units(text: str) -> List[str]:
    """
    Splits raw text into indivisible atomic units:
    1. Code blocks (```...```) are kept completely intact.
    2. Regular text is split into complete sentences without breaking punctuation.
    """
    # Regex to extract code blocks vs regular text
    code_block_pattern = re.compile(r"(```[\s\S]*?```)")
    parts = code_block_pattern.split(text)

    atomic_units: List[str] = []
    sentence_end_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n\s*\n")

    for part in parts:
        if not part.strip():
            continue
        # If it's a code block, preserve it as an indivisible unit
        if part.strip().startswith("```") and part.strip().endswith("```"):
            atomic_units.append(part.strip())
        else:
            # Split regular text along sentence boundaries
            sentences = [s.strip() for s in sentence_end_pattern.split(part) if s.strip()]
            atomic_units.extend(sentences)

    return atomic_units


def chunk_document(
    request: DocumentIngestRequest,
    security_context: SecurityContext,
    target_chunk_size: int = 250,  # Target words/tokens per chunk
    overlap_size: int = 50          # Overlap words to preserve cross-chunk context
) -> List[ChunkPayload]:
    """
    Chunks a document using a Structure-Aware Sliding Sentence Window.
    
    Guarantees:
    1. Sentences and code blocks are never cut in half.
    2. Chunks maintain a sliding overlap so antecedents (e.g., 'It', 'This config') retain their context.
    3. Every chunk carries the mandatory tenant_id, security clearance, and SHA-256 content hash.
    """
    raw_text = request.text.strip()
    if not raw_text:
        return []

    units = split_into_semantic_units(raw_text)
    if not units:
        return []

    chunks_text: List[str] = []
    unit_lengths = [count_tokens_approx(u) for u in units]

    start_idx = 0
    total_units = len(units)

    while start_idx < total_units:
        current_chunk_units: List[str] = []
        current_tokens = 0
        end_idx = start_idx

        # Accumulate units until we hit the target chunk size
        while end_idx < total_units:
            unit_tokens = unit_lengths[end_idx]
            if current_tokens + unit_tokens > target_chunk_size and current_chunk_units:
                break
            current_chunk_units.append(units[end_idx])
            current_tokens += unit_tokens
            end_idx += 1

        chunk_str = "\n\n".join(current_chunk_units).strip()
        if chunk_str:
            chunks_text.append(chunk_str)

        # If we have reached the end of the document, break out
        if end_idx >= total_units:
            break

        # Calculate sliding window step back for overlap
        overlap_tokens = 0
        step_back_idx = end_idx - 1

        while step_back_idx > start_idx and overlap_tokens < overlap_size:
            overlap_tokens += unit_lengths[step_back_idx]
            step_back_idx -= 1

        # Move start_idx forward for the next window
        # Guarantees forward progress to prevent infinite loops
        next_start = max(start_idx + 1, step_back_idx + 1)
        start_idx = next_start

    # Build the final ChunkPayload objects
    result_chunks: List[ChunkPayload] = []
    for idx, text in enumerate(chunks_text):
        chunk_id = f"{security_context.tenant_id}:{request.document_id}:v1:ch_{idx}"
        content_hash = compute_sha256(text)
        token_count = count_tokens_approx(text)

        payload = ChunkPayload(
            chunk_id=chunk_id,
            document_id=request.document_id,
            version=1,
            chunk_index=idx,
            content=text,
            content_hash=content_hash,
            token_count=token_count,
            tenant_id=security_context.tenant_id,
            allowed_roles=request.allowed_roles,
            allowed_groups=request.allowed_groups,
            classification=request.classification,
            security_policy_version=security_context.policy_version,
        )
        result_chunks.append(payload)

    return result_chunks
