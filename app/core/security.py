"""
Cryptographic Security, JWT Token Processing, and ABAC/RBAC Policy Engine.
Implements the core invariant: Authorization happens BEFORE retrieval.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.schemas.rag import ClearanceLevel, SecurityContext


# HTTP Bearer scheme for Authorization: Bearer <token>
security_scheme = HTTPBearer(auto_error=True)


# Clearance hierarchy definition: lower index has fewer privileges
CLEARANCE_HIERARCHY = {
    ClearanceLevel.PUBLIC: 0,
    ClearanceLevel.INTERNAL: 1,
    ClearanceLevel.CONFIDENTIAL: 2,
    ClearanceLevel.RESTRICTED: 3,
}


def create_access_token(
    subject: str,
    tenant_id: str,
    roles: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    clearance: ClearanceLevel = ClearanceLevel.INTERNAL,
    scopes: Optional[List[str]] = None,
    policy_version: str = "1.0.0",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Creates a cryptographically signed HMAC-SHA256 JWT carrying the full SecurityContext.
    Used by test suites and auth gateways to issue enterprise identity tokens.
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode: Dict[str, Any] = {
        "sub": subject,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "groups": groups or [],
        "clearance": clearance.value if isinstance(clearance, ClearanceLevel) else clearance,
        "scopes": scopes or [],
        "policy_version": policy_version,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_security_context(token: str) -> SecurityContext:
    """
    Decodes and cryptographically validates a JWT token.
    Derives the immutable SecurityContext.
    Fails closed if the token is tampered, expired, or missing mandatory claims.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        subject: Optional[str] = payload.get("sub")
        tenant_id: Optional[str] = payload.get("tenant_id")
        
        if not subject or not tenant_id:
            raise credentials_exception
            
        raw_clearance = payload.get("clearance", ClearanceLevel.INTERNAL.value)
        try:
            clearance = ClearanceLevel(raw_clearance)
        except ValueError:
            clearance = ClearanceLevel.INTERNAL

        return SecurityContext(
            subject=subject,
            tenant_id=tenant_id,
            roles=payload.get("roles", []),
            groups=payload.get("groups", []),
            clearance=clearance,
            scopes=payload.get("scopes", []),
            policy_version=payload.get("policy_version", "1.0.0")
        )
        
    except JWTError:
        raise credentials_exception


async def get_current_security_context(
    auth_header: HTTPAuthorizationCredentials = Depends(security_scheme)
) -> SecurityContext:
    """
    FastAPI dependency that extracts the Bearer token, validates it,
    and injects the immutable SecurityContext into request routes.
    """
    return decode_security_context(auth_header.credentials)


def is_clearance_sufficient(user_clearance: ClearanceLevel, required_clearance: ClearanceLevel) -> bool:
    """
    Evaluates whether a user's clearance level satisfies the document's classification.
    E.g., CONFIDENTIAL (2) can access INTERNAL (1) and PUBLIC (0), but not RESTRICTED (3).
    """
    return CLEARANCE_HIERARCHY.get(user_clearance, 0) >= CLEARANCE_HIERARCHY.get(required_clearance, 0)


def build_authorized_qdrant_filter(security_context: SecurityContext) -> Dict[str, Any]:
    """
    Constructs the pre-retrieval Qdrant filter condition based on RBAC + ABAC.
    Enforces the core architectural invariant:
        RetrievedChunks ⊆ AuthorizedCorpus(tenant, roles, groups, clearance)
    
    This filter is compiled by the application and handed to Qdrant BEFORE retrieval.
    The LLM never touches this filter.
    """
    allowed_clearances = [
        level.value
        for level, weight in CLEARANCE_HIERARCHY.items()
        if weight <= CLEARANCE_HIERARCHY.get(security_context.clearance, 0)
    ]

    # Qdrant boolean filter dictionary structure:
    # 1. Must match tenant_id (HARD MULTI-TENANT ISOLATION)
    # 2. Must match allowed classification levels (ABAC)
    # 3. Must match user's roles OR user's groups OR public document (RBAC)
    return {
        "must": [
            {"key": "tenant_id", "match": {"value": security_context.tenant_id}},
            {"key": "classification", "match": {"any": allowed_clearances}}
        ],
        "should": [
            {"key": "allowed_roles", "match": {"any": security_context.roles}},
            {"key": "allowed_groups", "match": {"any": security_context.groups}},
            {"key": "allowed_roles", "match": {"value": "*"}},  # Wildcard for globally accessible chunks
        ]
    }
