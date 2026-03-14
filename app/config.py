"""
config.py — Global Configuration Module
========================================

This module is the SINGLE source of truth for every setting in the application.
Nothing in the codebase hard-codes API keys, URLs, or tuning parameters.
Instead, all modules import `settings` from here and read what they need.

How it works:
    pydantic-settings reads values in this priority order:
        1. Environment variables (e.g. OPENAI_API_KEY=... in the shell)
        2. .env file in the project root (never committed to git)
        3. Default values defined in the class (safe fallbacks only)

Why pydantic-settings and not plain os.getenv()?
    - Type coercion: chunk_size="600" becomes int(600) automatically.
    - Validation: bad values raise a clear error at startup, not mid-request.
    - Self-documenting: Field(...) with description acts as inline docs.
    - The model_validator lets us export keys to os.environ so libraries like
      LangChain and Groq that read environment variables still work seamlessly.
"""

import os
from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global configuration for the Compliance RAG system.

    Loads values automatically from:
      - .env file (local development)
      - environment variables (Docker / cloud deployment)

    Design note:
      Fields marked Field(...) are REQUIRED — the app will not start without them.
      Fields with a default value are OPTIONAL — safe to omit in development.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Pydantic-Settings Behaviour
    # ─────────────────────────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        # Look for a .env file in the working directory
        env_file=".env",
        # Ignore unexpected variables in .env (e.g. editor artefacts)
        extra="ignore",
        # OPENAI_API_KEY and openai_api_key are treated the same
        case_sensitive=False,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # API KEYS
    # ─────────────────────────────────────────────────────────────────────────

    # Groq: used for LLM inference on the multi-query rewriting step
    groq_api_key: str = Field(..., description="Groq API key for LLM inference")

    # OpenAI: used for embeddings AND as the main structured-output LLM
    openai_api_key: str = Field(
        ..., description="OpenAI API key for the LLM and embeddings"
    )

    # Cohere: used for reranking retrieved chunks (API-based, no local model needed)
    # Free tier: 1000 calls/month. Get key at https://cohere.com
    cohere_api_key: str = Field(..., description="Cohere API key for reranking")

    # LangSmith: optional observability / tracing. Leave unset to disable.
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "compliance-rag-2026"

    # ─────────────────────────────────────────────────────────────────────────
    # JWT / AUTH SETTINGS
    # ─────────────────────────────────────────────────────────────────────────

    # The secret used to sign JWT tokens. Generate with: openssl rand -hex 32
    jwt_secret_key: str = Field(..., description="Secret key for signing JWT tokens")

    # HS256 is the standard HMAC-SHA256 algorithm — fast and widely supported
    jwt_algorithm: str = "HS256"

    # How long a token stays valid (minutes). 480 = 8 hours.
    jwt_expire_minutes: int = 480

    # ─────────────────────────────────────────────────────────────────────────
    # LLM SETTINGS
    # ─────────────────────────────────────────────────────────────────────────

    # The Groq model used for query rewriting (fast, cheap)
    groq_model: str = Field(..., description="Groq model name for query rewriting")

    # temperature=0.0 means deterministic output — critical for compliance use
    temperature: float = 0.0

    # The OpenAI model used for the main compliance report generation
    openai_model: str = "gpt-4o-mini"

    # ─────────────────────────────────────────────────────────────────────────
    # EMBEDDING SETTINGS
    # ─────────────────────────────────────────────────────────────────────────

    # Provider switch: only "openai" is implemented in this codebase
    embedding_provider: str = "openai"

    # text-embedding-3-small: 1536-dimensional, fast, cost-efficient
    embedding_model: str = "text-embedding-3-small"

    # ─────────────────────────────────────────────────────────────────────────
    # VECTOR DATABASE (Qdrant)
    # ─────────────────────────────────────────────────────────────────────────

    qdrant_url: str = Field(..., description="Qdrant Vector DB URL")
    qdrant_api_key: str = Field(..., description="Qdrant Vector DB API key")
    qdrant_collection_name: str = Field(..., description="Qdrant collection name")

    # ─────────────────────────────────────────────────────────────────────────
    # DATABASE (Supabase)
    # ─────────────────────────────────────────────────────────────────────────

    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_anon_key: str = Field(..., description="Supabase anon/public API key")

    # Full Postgres connection string for direct DB access (SQLAlchemy / asyncpg)
    database_url: str = Field(..., description="PostgreSQL connection string")

    # ─────────────────────────────────────────────────────────────────────────
    # DOCUMENT PATHS
    # ─────────────────────────────────────────────────────────────────────────

    # Folder where compliance PDFs are stored. Relative to the working directory.
    compliance_docs_dir: str = "./documents/"

    # Folder where the embedding cache is stored on disk
    embedding_cache_dir: str = "./.embedding_cache"

    # ─────────────────────────────────────────────────────────────────────────
    # RAG HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────────────────

    # How many characters per chunk (not words, not tokens — characters)
    chunk_size: int = 600

    # How many characters of overlap between consecutive chunks.
    # Overlap prevents a sentence split at a boundary from losing context.
    chunk_overlap: int = 80

    # How many candidate chunks to retrieve BEFORE reranking
    top_k_retrieval: int = 20

    # How many top chunks to keep AFTER cross-encoder reranking
    top_k_rerank: int = 5

    # Maximum tokens to pass to the LLM in the context window
    max_context_tokens: int = 7000

    # RAGAS faithfulness threshold: raise a warning if below this
    min_faithfulness: float = 0.80

    # ─────────────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ─────────────────────────────────────────────────────────────────────────

    # Max query requests per user per minute (protects OpenAI spend)
    rate_limit_per_minute: int = 10

    # ─────────────────────────────────────────────────────────────────────────
    # CORS SETTINGS
    # ─────────────────────────────────────────────────────────────────────────

    # Comma-separated list of allowed frontend origins.
    # In production, replace * with your actual domain.
    cors_origins: str = "*"

    # ─────────────────────────────────────────────────────────────────────────
    # ENVIRONMENT CONFIGURATION (post-load side-effects)
    # ─────────────────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def configure_environment(self) -> "Settings":
        """
        What does this validator do?
        After all settings are loaded, it exports the API keys to os.environ.

        Why export to os.environ?
        Libraries like LangChain, OpenAI SDK, and Groq SDK automatically
        read environment variables. By exporting here, we avoid having to
        pass api_key= explicitly to every client instantiation in the codebase.
        One export here → works everywhere.
        """
        os.environ["GROQ_API_KEY"] = self.groq_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # Only enable LangSmith tracing if a key was provided
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        """
        What does this validator enforce?
        chunk_overlap must be smaller than chunk_size.

        Why does this matter?
        If overlap >= size, the splitter would produce infinite loops or
        nonsensical chunks. This catches the misconfiguration at startup.
        """
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be smaller than chunk_size ({chunk_size})"
            )
        return v

    def cors_origins_list(self) -> list[str]:
        """
        Parses the comma-separated CORS_ORIGINS string into a Python list.
        Used by FastAPI's CORSMiddleware which requires a list, not a string.

        Example: "http://localhost:3000,https://myapp.com"
            → ["http://localhost:3000", "https://myapp.com"]
        """
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
# All other modules do: from app.config import settings
# This runs once at import time. If any required field is missing, the app
# fails fast here with a clear error message — not later during a request.

settings = Settings()
