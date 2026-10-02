from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.core.config import settings
from app.core.security import create_access_token
from app.schemas.rag import ClearanceLevel

router = APIRouter(tags=["Authentication"])

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    roles: list[str]
    scopes: list[str]

@router.post("/auth/token", response_model=TokenResponse, summary="OAuth2 Password Flow Login")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> TokenResponse:
    username = form_data.username or "admin"
    tenant_id = "tenant_corp_alpha"
    roles = ["admin", "user", "engineering"]
    scopes = ["knowledge:read", "knowledge:write"]
    groups = ["engineering", "devops"]

    token = create_access_token(
        subject=username,
        tenant_id=tenant_id,
        roles=roles,
        scopes=scopes,
        groups=groups,
        clearance=ClearanceLevel.RESTRICTED,
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        tenant_id=tenant_id,
        roles=roles,
        scopes=scopes
    )
