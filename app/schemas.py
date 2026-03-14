"""
schemas.py — Data Models for the Compliance RAG System
=======================================================

This module defines EVERY Pydantic model used across the application.
Keeping all models in one place means:
  - One import to find any data shape.
  - Shared models (e.g. ComplianceReport) used by both the RAG engine
    and the API endpoint without circular imports.
  - A single source of truth for API documentation (FastAPI reads these
    to generate the OpenAPI / Swagger docs automatically).

Rule of thumb:
  - Models that arrive FROM the outside (HTTP requests, file uploads)
    go in the "API Inputs" section.
  - Models that go OUT to the outside (HTTP responses, Supabase rows)
    go in the "API Outputs" and "Database Models" sections.
  - Models that stay internal to the RAG pipeline go in
    "RAG Pipeline Models".
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# AUTH MODELS
# These models define what the login endpoint accepts and returns.
# ─────────────────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    """
    Data required to register a new user.

    Why full_name and username separately?
    - username is the login identifier: short, lowercase, no spaces (e.g. "john.doe")
    - full_name is what gets displayed in the UI and stored in Supabase
      so we see "John Doe" in audit logs, not a random UUID.
    """

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique login identifier (e.g. 'john.doe')",
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Display name stored in Supabase (e.g. 'John Doe')",
    )
    password: str = Field(..., min_length=8, description="Plain-text password (hashed before storage)")
    role: str = Field(
        default="analyst",
        description="User role: 'analyst', 'auditor', or 'admin'",
    )


class UserInDB(BaseModel):
    """
    User record as stored in Supabase.

    Why store user_id separately from username?
    - user_id (UUID) is immutable — it never changes even if a user renames.
    - username can change without breaking audit log references.
    - Foreign keys in compliance_messages and compliance_audit_log
      reference user_id, not username.
    """

    user_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Immutable UUID primary key",
    )
    username: str
    full_name: str
    hashed_password: str
    role: str = "analyst"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_active: bool = True


class TokenResponse(BaseModel):
    """
    What the /auth/token endpoint returns after successful login.

    The access_token is a signed JWT the client must send in every
    subsequent request as:  Authorization: Bearer <access_token>
    """

    access_token: str
    token_type: str = "bearer"
    username: str
    full_name: str
    role: str


class TokenData(BaseModel):
    """
    Payload decoded FROM a JWT token.

    This is the internal representation inside auth.py.
    It is NOT returned to the client — it's used to identify the
    current user during request processing.
    """

    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# API INPUT MODELS (what clients send to the API)
# ─────────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """
    Payload for the POST /query endpoint.

    question:        The natural-language compliance question.
    doc_type_filter: Optional filter — only retrieve chunks from documents
                     of this type (e.g. "regulation", "legislation").
                     Passed to Qdrant as a payload filter.
    """

    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural-language compliance question",
    )
    doc_type_filter: Optional[str] = Field(
        default=None,
        description="Optional filter: 'regulation', 'legislation', 'policy', etc.",
    )


class IngestRequest(BaseModel):
    """
    Optional payload for POST /ingest.

    If no body is sent, the endpoint scans the configured documents/ folder.
    If a specific subdirectory is provided, only that folder is scanned.
    This allows targeted re-ingestion without rebuilding the entire index.
    """

    subdirectory: Optional[str] = Field(
        default=None,
        description="Sub-folder inside compliance_docs_dir to scan (optional)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE MODELS (internal to the engine and ingestion pipeline)
# ─────────────────────────────────────────────────────────────────────────────


class DocMetadata(BaseModel):
    """
    Validated metadata attached to every PDF page BEFORE chunking.

    Why Pydantic here instead of a plain dict?
    1. page is always int — never the string "3" from a parser quirk.
    2. filename cannot be empty (min_length=1 catches it at parse time).
    3. .model_dump() gives a clean dict for Qdrant payload storage.
    4. The schema is self-documenting — Field descriptions become docs.

    These fields become the Qdrant payload for each stored vector point,
    which enables filtered retrieval (e.g. "only from year >= 2023").
    """

    filename: str = Field(..., min_length=1, description="Original PDF filename")
    file_path: str = Field(..., description="Absolute path to the PDF on disk")
    doc_id: str = Field(
        ..., description="16-char sha256 of filepath+mtime — unique per file version"
    )
    page: int = Field(..., ge=1, description="1-indexed page number in the PDF")
    total_pages: int = Field(..., ge=1, description="Total pages in the source PDF")
    doc_type: str = Field(
        default="regulatory_document",
        description="Inferred doc type: regulation, legislation, policy, etc.",
    )
    category: str = Field(default="compliance", description="Top-level category tag")
    jurisdiction: str = Field(
        default="unknown", description="Legal jurisdiction (e.g. 'Nigeria', 'EU')"
    )
    year: Optional[int] = Field(default=None, description="Year extracted from filename or content")
    reference_number: Optional[str] = Field(
        default=None, description="Official reference number (e.g. CBN/DIR/GEN/2024/001)"
    )
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of when this page was ingested",
    )


class QueryVariants(BaseModel):
    """
    Structured output for the multi-query expansion step.

    The LLM is asked to rewrite the original question into 3 alternative
    formulations. By retrieving against all 4 (original + 3 variants),
    we get a much larger, more diverse candidate pool for reranking.

    Why exactly 3?
    3 variants + 1 original = 4 total. Empirically, 4 retrievals give
    ~2x recall improvement over a single query on legal/compliance text.
    More than 4 gives diminishing returns and increases latency.
    """

    variants: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly 3 rewritten versions of the original question",
    )


class ComplianceReport(BaseModel):
    """
    The structured output of every compliance query.

    This is what llm.with_structured_output(ComplianceReport) returns.
    The Pydantic schema is sent to OpenAI as a JSON schema via tool-calling,
    so the model CANNOT return free-form text — it must match this shape.

    Why typed output for compliance?
    - verdict is a Literal: the LLM CANNOT return "Probably fine" or "maybe"
    - confidence_score has ge=0 le=1: impossible to return 1.5 or -0.2
    - Downstream systems receive a typed object, never raw LLM text
    - Serialization with .model_dump() gives clean JSON for Supabase/API
    """

    verdict: Literal["COMPLIANT", "NON_COMPLIANT", "INSUFFICIENT_DATA", "NEEDS_REVIEW"]
    summary: str = Field(..., description="One-sentence plain-English answer")
    detailed_analysis: str = Field(..., description="Full compliance reasoning")
    relevant_rules: List[str] = Field(
        default_factory=list, description="Specific rules or articles cited"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    citations: List[str] = Field(
        default_factory=list, description="Source document name + page number"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model's confidence from 0.0 to 1.0"
    )
    recommended_action: str = Field(
        default="", description="What the auditor recommends doing next"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Disclaimers, edge cases, or conflicting regulations",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE MODELS (rows written to Supabase)
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """
    One message in a user conversation. Written to the compliance_messages table.

    Why store both user and assistant messages in one table?
    A single table makes it trivial to reconstruct the full conversation
    timeline by ordering on created_at. A two-table design (user_messages
    + assistant_messages) would require a UNION at query time.
    """

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID primary key for this message",
    )
    session_id: str = Field(..., description="Which session this message belongs to")
    user_id: str = Field(..., description="Who sent or received this message")
    role: Literal["user", "assistant"] = Field(
        ..., description="'user' = question, 'assistant' = compliance report"
    )
    content: str = Field(..., description="Message text")
    verdict: Optional[str] = Field(
        default=None,
        description="Compliance verdict (only set for assistant messages)",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AuditLogEntry(BaseModel):
    """
    Immutable compliance audit record. Written to compliance_audit_log.

    Why immutable?
    This table is compliance evidence. In a real Big 4 scenario, altering
    an audit log after the fact could constitute tampering. INSERT-only,
    never UPDATE or DELETE.

    Why store latency_ms?
    Compliance SLAs often require responses within N seconds. Tracking
    latency allows proactive alerting if the system slows down.
    """

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID primary key",
    )
    user_id: str = Field(..., description="Which user submitted this query")
    session_id: str = Field(..., description="Which session this belongs to")
    query: str = Field(..., description="The original compliance question")
    verdict: str = Field(..., description="COMPLIANT / NON_COMPLIANT / etc.")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    confidence: float = Field(..., description="Model confidence score (0.0–1.0)")
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of source citations included in the answer",
    )
    latency_ms: float = Field(..., description="Total query latency in milliseconds")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# API OUTPUT MODELS (what the API returns to clients)
# ─────────────────────────────────────────────────────────────────────────────


class ChatResponse(BaseModel):
    """
    The full response from the POST /query endpoint.

    Contains the ComplianceReport plus the session and latency metadata
    the frontend needs to display the result correctly.
    """

    session_id: str
    query: str
    report: ComplianceReport
    latency_ms: float
    sources_count: int


class IngestResponse(BaseModel):
    """
    Response from the POST /ingest endpoint.

    Tells the caller how many documents were found, how many were new,
    and how many chunks were successfully indexed into Qdrant.
    """

    documents_found: int
    documents_new: int
    documents_skipped: int
    chunks_indexed: int
    latency_seconds: float
    message: str


class HealthResponse(BaseModel):
    """
    Response from the GET /health endpoint.

    Used by Docker health checks, load balancers, and monitoring dashboards.
    Each dependency has its own status field so operators can quickly
    identify which component is degraded.
    """

    status: str = "ok"
    version: str = "1.0.0"
    qdrant: str = "unknown"
    supabase: str = "unknown"
    openai: str = "unknown"
    collection_points: Optional[int] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
