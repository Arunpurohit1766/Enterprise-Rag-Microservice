"""
Cryptographic Security, OIDC Claims Validation, and ABAC/RBAC Policy Engine.
Enforces that authorization, scope verification, and write policies happen BEFORE execution.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer

from app.core.config import settings
from app.schemas.rag import ClearanceLevel, DocumentIngestRequest, SecurityContext


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
http_bearer_scheme = HTTPBearer(auto_error=False)

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
    Creates a cryptographically signed HMAC-SHA256 JWT carrying full claims,
    strictly validated with issuer and audience metadata.
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))

    to_encode: Dict[str, Any] = {
        "sub": subject,
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "groups": groups or [],
        "clearance": clearance.value if isinstance(clearance, ClearanceLevel) else clearance,
        "scopes": scopes or ["knowledge:read", "knowledge:write"],
        "policy_version": policy_version,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_security_context(token: str) -> SecurityContext:
    """
    Decodes and cryptographically validates a JWT token.
    Enforces signature, expiration, issuer (iss), and audience (aud).
    Fails closed if any claim is invalid or tampered.
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
            algorithms=[settings.JWT_ALGORITHM],
            issuer=settings.JWT_ISSUER,
            audience=settings.JWT_AUDIENCE,
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
    oauth2_token: Optional[str] = Depends(oauth2_scheme),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer_scheme),
) -> SecurityContext:
    """FastAPI dependency: extracts and cryptographically validates caller context from OAuth2 or Bearer header."""
    token = oauth2_token or (bearer_creds.credentials if bearer_creds else None)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required. Use OAuth2 login or Bearer authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_security_context(token)


def require_scope(required_scope: str) -> Callable:
    """
    Dependency factory: Enforces OAuth2 scope compliance.
    E.g. require_scope('knowledge:read') or require_scope('knowledge:write')
    """
    async def scope_dependency(
        ctx: SecurityContext = Depends(get_current_security_context)
    ) -> SecurityContext:
        if required_scope not in ctx.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Principal lacks required scope '{required_scope}'"
            )
        return ctx
    return scope_dependency


def validate_write_authorization(
    request: DocumentIngestRequest,
    ctx: SecurityContext
) -> None:
    """
    Evaluates Ingestion Write Authorization Policy:
    1. Principal cannot classify a document higher than their own clearance.
    2. Principal cannot set wildcard allowed_roles (['*']) unless they possess the 'admin' role.
    """
    user_weight = CLEARANCE_HIERARCHY.get(ctx.clearance, 0)
    doc_weight = CLEARANCE_HIERARCHY.get(request.classification, 0)

    if doc_weight > user_weight:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Write policy violation: Cannot ingest document with classification '{request.classification.value}' "
                f"exceeding principal clearance '{ctx.clearance.value}'."
            )
        )

    if "*" in request.allowed_roles and "admin" not in ctx.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write policy violation: Only principals with 'admin' role can assign wildcard '*' access."
        )


def build_authorized_qdrant_filter(security_context: SecurityContext) -> Dict[str, Any]:
    """
    Constructs the pre-retrieval Qdrant filter condition based on RBAC + ABAC.
    Enforces: RetrievedChunks ⊆ AuthorizedCorpus(tenant, roles, groups, clearance)
    """
    allowed_clearances = [
        level.value
        for level, weight in CLEARANCE_HIERARCHY.items()
        if weight <= CLEARANCE_HIERARCHY.get(security_context.clearance, 0)
    ]

    return {
        "must": [
            {"key": "tenant_id", "match": {"value": security_context.tenant_id}},
            {"key": "classification", "match": {"any": allowed_clearances}}
        ],
        "should": [
            {"key": "allowed_roles", "match": {"any": security_context.roles}},
            {"key": "allowed_groups", "match": {"any": security_context.groups}},
            {"key": "allowed_roles", "match": {"value": "*"}},
        ]
    }
