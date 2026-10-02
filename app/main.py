"""
Enterprise Knowledge & Retrieval Microservice Entrypoint.
Configures FastAPI app, CORS middleware, API v1 router, health probes,
and seamless root redirection to Swagger documentation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.v1.endpoints.rag import router as rag_router
from app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Multi-Tenant Knowledge & Retrieval Microservice with Hybrid RRF and Citation Verification.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware (Enterprise standard: configure allowed origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint: Seamlessly redirects reviewers directly to Swagger UI
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root traffic directly to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


# Mount API v1 Routes
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
