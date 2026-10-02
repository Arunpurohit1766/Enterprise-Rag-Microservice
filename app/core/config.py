"""
Core Application Settings and Environment Configuration.
Strictly validated using Pydantic Settings v2.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Service Metadata
    PROJECT_NAME: str = "Enterprise-Knowledge-Retrieval-Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")
    
    # Security & Cryptography
    JWT_SECRET_KEY: str = Field(
        default="09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
        description="256-bit secret key for HMAC-SHA256 JWT signing"
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Qdrant Vector Engine Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "enterprise_knowledge_base"
    QDRANT_USE_IN_MEMORY: bool = True  # True enables ultra-fast embedded HNSW for development/testing
    
    # Dense Embedding Engine (FastEmbed)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    
    # Retrieval Tuning Constants
    RRF_K: int = 60  # Smoothing constant for Reciprocal Rank Fusion
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    FINAL_TOP_K: int = 5
    
    # Answerability Evidence Gate Thresholds
    ANSWERABILITY_THRESHOLD: float = 0.35
    SCORE_MARGIN_THRESHOLD: float = 0.02
    
    # LLM Inference (Groq)
    GROQ_API_KEY: str = Field(default="gsk_mock_development_key", description="Groq LPU API key")
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_TEMPERATURE: float = 0.0  # Zero temperature for deterministic factual extraction
    GROQ_MAX_TOKENS: int = 1024


# Singleton settings instance
settings = Settings()
