"""
Enterprise Knowledge & Retrieval Microservice Entrypoint.
Configures FastAPI app, strict CORS policies, route boundaries,
and production documentation toggles.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.v1.endpoints.rag import router as rag_router
from app.api.v1.endpoints.auth import router as auth_router
from app.core.config import settings


docs_url = "/docs" if (settings.ENVIRONMENT != "production" or settings.ENABLE_DOCS_IN_PRODUCTION) else None
redoc_url = "/redoc" if (settings.ENVIRONMENT != "production" or settings.ENABLE_DOCS_IN_PRODUCTION) else None

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Multi-Tenant Knowledge & Retrieval Microservice with Hybrid RRF and Citation Verification.",
    docs_url=docs_url,
    redoc_url=redoc_url,
)

# Strict CORS: Reject wildcard origins in enterprise deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    if docs_url:
        return RedirectResponse(url=docs_url)
    return {
        "service": settings.PROJECT_NAME,
        "status": "OPERATIONAL",
        "version": settings.VERSION
    }


# Mount API v1 Routes
app.include_router(auth_router, prefix="/api/v1")
app.include_router(rag_router, prefix=f"{settings.API_V1_STR}/rag", tags=["RAG"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe for Kubernetes and Docker orchestrators."""
    return {
        "status": "HEALTHY",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
