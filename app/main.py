"""
main.py — FastAPI Application Entry Point
==========================================

This is the HTTP API layer that sits on top of the RAG engine.
It handles: routing, authentication, request validation, rate limiting,
CORS, file uploads, and response formatting.

It does NOT contain any RAG logic — that lives in engine.py.
It does NOT contain any database logic — that lives in database.py.
It does NOT contain auth logic — that lives in auth.py.

Principle: main.py should only be concerned with HTTP concerns.
Every piece of business logic should be delegable to another module.

Startup sequence:
    1. Load settings (config.py) — fails fast if .env is incomplete
    2. ensure_collection() — create Qdrant collection if needed
    3. get_query_store() — connect to Qdrant for query-time use
    4. get_reranker() — download and load the cross-encoder model
    5. Start serving requests
"""

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from app.auth import (
    create_access_token,
    get_admin_user,
    get_current_user,
    hash_password,
    verify_password,
)
from app.config import settings
from app.database import (
    create_user,
    get_indexed_documents,
    get_or_create_session,
    get_user_by_username,
    log_document_ingested,
)
from app.engine import run_compliance_query
from app.ingestion import chunk_documents, extract_all_documents, scan_compliance_docs
from app.schemas import (
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    TokenResponse,
    UserCreate,
    UserInDB,
)
from app.vector_store import build_qdrant_store, ensure_collection, get_qdrant_client


# ─────────────────────────────────────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────────────────────────────────────

# We store the Qdrant store as application state so it is initialised
# once at startup and reused across all requests.
# app.state.qdrant_store is set in the lifespan context manager below.


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting (in-memory, per user)
# ─────────────────────────────────────────────────────────────────────────────

# Why in-memory rate limiting instead of Redis?
# For a single-instance deployment, in-memory is simpler and has zero
# additional infrastructure. For multi-instance deployments, swap this
# dict for a Redis-backed counter.

_request_counts: Dict[str, list] = defaultdict(list)


def check_rate_limit(user_id: str) -> None:
    """
    What does this function do?
    Enforces a sliding window rate limit of N requests per minute per user.
    If the user has made too many requests recently, raises HTTP 429.

    Why protect the /query endpoint specifically?
    Each query makes multiple OpenAI API calls:
      - 1 call for multi-query expansion (query rewrite)
      - 1 call for the main compliance report
    Without rate limiting, a single runaway client could generate
    hundreds of dollars of API spend in minutes.

    Implementation:
    We keep a list of timestamps for each user.
    On each request, we: remove old timestamps (outside the window),
    check if the count exceeds the limit, and append the current timestamp.
    """
    now = time.time()
    window = 60  # seconds

    # Remove timestamps that are older than the window
    _request_counts[user_id] = [
        ts for ts in _request_counts[user_id] if now - ts < window
    ]

    # Check if the user has exceeded the limit
    if len(_request_counts[user_id]) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: max {settings.rate_limit_per_minute} "
                f"requests per minute. Please wait and try again."
            ),
        )

    # Record this request
    _request_counts[user_id].append(now)


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    What does this context manager do?
    Runs startup tasks when the application boots and cleanup on shutdown.

    Startup tasks:
        1. ensure_collection() — idempotent: creates Qdrant collection if needed
        2. build_qdrant_store([]) — connect to the existing collection
        3. get_reranker() — download the cross-encoder model (once, at startup)

    Why initialise at startup and not per-request?
    - Qdrant connection: creating a client per request adds ~50ms overhead
    - Reranker model: downloading a HuggingFace model takes 5–30 seconds
    Loading once at startup means the first request is fast, not slow.

    The app.state dictionary is shared across all requests and threads.
    """
    # ── STARTUP ────────────────────────────────────────────────────────────────
    print("[startup] Compliance RAG API starting...")

    # Ensure the vector store collection exists
    ensure_collection()

    # Connect to the existing collection for query-time use
    qdrant_store = build_qdrant_store(chunks=[])
    app.state.qdrant_store = qdrant_store
    print("[startup] Qdrant store ready.")

    # Reranker is Cohere API — no model to preload, no RAM cost at startup
    print("[startup] Reranker: Cohere API (ready)")

    print("[startup] Compliance RAG API ready to serve requests.")

    yield  # Application is now running and serving requests

    # ── SHUTDOWN ────────────────────────────────────────────────────────────────
    print("[shutdown] Compliance RAG API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Compliance RAG API",
    description=(
        "Production-grade Retrieval Augmented Generation for Nigerian compliance documents. "
        "Powered by OpenAI, Qdrant hybrid search, and cross-encoder reranking."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORS Middleware
# ─────────────────────────────────────────────────────────────────────────────

# CORS (Cross-Origin Resource Sharing) is required for the frontend
# running on a different port (e.g. localhost:3000) to call the API.
# Without CORS, the browser will block the API call.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,           # Allow cookies and Auth headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],              # Allow all headers including Authorization
)

# ── Serve the frontend UI ─────────────────────────────────────────────────────
# Mount the ui/ folder so FastAPI serves index.html at the root URL.
# This means your deployed Render URL (e.g. https://lexai.onrender.com) opens
# the chat UI directly — no separate hosting needed for the frontend.
#
# IMPORTANT: This mount must come AFTER all API routes are registered,
# otherwise the wildcard catch-all would intercept API requests.
# We register it here (after middleware, before routes) using a path mount,
# and add a root redirect below after all routes are defined.

import os as _os
_ui_dir = _os.path.join(_os.path.dirname(__file__), "..", "ui")
if _os.path.isdir(_ui_dir):
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    GET /health

    Returns the operational status of each system component.
    Used by Docker health checks, load balancers, and monitoring dashboards.

    No authentication required — health checks must always be accessible.
    """
    qdrant_status = "error"
    collection_points = None

    openai_status = "configured" if settings.openai_api_key else "not configured"
    supabase_status = "configured" if settings.supabase_url else "not configured"

    # Test Qdrant connectivity
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection_name)
        collection_points = info.points_count
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)[:50]}"

    overall = "ok" if qdrant_status == "connected" else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        supabase=supabase_status,
        openai=openai_status,
        collection_points=collection_points,
    )


# ── Auth: Register ─────────────────────────────────────────────────────────────

@app.post(
    "/auth/register",
    response_model=dict,
    summary="Register a new user",
    tags=["Authentication"],
    status_code=status.HTTP_201_CREATED,
)
async def register(user_data: UserCreate) -> dict:
    """
    POST /auth/register

    Creates a new user account.

    The username is stored as a human-readable login identifier
    (e.g. "john.doe"). The full_name is what appears in the UI and audit logs
    (e.g. "John Doe").

    Passwords are bcrypt-hashed before storage — plaintext is never saved.
    """
    # Check if username already taken
    existing = await get_user_by_username(user_data.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{user_data.username}' is already taken.",
        )

    # Create the user object with a hashed password
    new_user = UserInDB(
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=hash_password(user_data.password),
        role=user_data.role,
    )

    created = await create_user(new_user)

    if not created:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed. Check Supabase connection.",
        )

    return {
        "message": f"User '{user_data.username}' ({user_data.full_name}) registered successfully.",
        "user_id": created.user_id,
        "username": created.username,
        "full_name": created.full_name,
    }


# ── Auth: Login ───────────────────────────────────────────────────────────────

@app.post(
    "/auth/token",
    response_model=TokenResponse,
    summary="Login and obtain JWT token",
    tags=["Authentication"],
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """
    POST /auth/token

    Standard OAuth2 Password Flow login.
    Accepts form data: username + password.
    Returns a JWT access token.

    The frontend sends this token in every subsequent request:
        Authorization: Bearer <access_token>

    Why OAuth2PasswordRequestForm instead of a JSON body?
    This is the FastAPI standard for the token endpoint.
    It generates the correct OpenAPI schema so the /docs Swagger UI
    has a working "Authorize" button.
    """
    # Look up the user by username
    user = await get_user_by_username(form_data.username)

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been disabled.",
        )

    # Create a JWT containing: username (as 'sub'), user_id, and role
    access_token = create_access_token(
        data={
            "sub": user.username,      # 'sub' = subject — standard JWT claim
            "user_id": user.user_id,
            "role": user.role,
        }
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        username=user.username,
        full_name=user.full_name,
        role=user.role,
    )


# ── Ingest Documents ──────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger the document ingestion pipeline",
    tags=["Documents"],
)
async def ingest(
    request: Request,
    body: Optional[IngestRequest] = None,
    current_user: UserInDB = Depends(get_admin_user),
) -> IngestResponse:
    """
    POST /ingest

    Triggers the full ingestion pipeline:
        1. Scan the documents/ folder for PDFs
        2. Skip already-indexed documents (idempotent)
        3. Extract text from new PDFs with pymupdf4llm
        4. Split pages into overlapping chunks
        5. Embed and index chunks into Qdrant (hybrid: dense + sparse)
        6. Log indexed documents to Supabase

    Protected: requires 'admin' or 'auditor' role.

    After indexing, the new chunks are immediately available for queries
    because we rebuild app.state.qdrant_store with the updated collection.
    """
    start = time.time()

    subdirectory = body.subdirectory if body else None

    # ── Step 1: Scan ──────────────────────────────────────────────────────────
    records = scan_compliance_docs(docs_dir=subdirectory)

    if not records:
        return IngestResponse(
            documents_found=0,
            documents_new=0,
            documents_skipped=0,
            chunks_indexed=0,
            latency_seconds=round(time.time() - start, 2),
            message="No PDF documents found in the documents folder.",
        )

    new_records = [r for r in records if not r.already_indexed]
    skipped_count = len(records) - len(new_records)

    # ── Step 2: Extract ───────────────────────────────────────────────────────
    raw_pages = extract_all_documents(records)

    if not raw_pages:
        return IngestResponse(
            documents_found=len(records),
            documents_new=0,
            documents_skipped=skipped_count,
            chunks_indexed=0,
            latency_seconds=round(time.time() - start, 2),
            message="All documents are already indexed. No new content to process.",
        )

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    chunks = chunk_documents(raw_pages)

    # ── Step 4: Index into Qdrant ──────────────────────────────────────────────
    updated_store = build_qdrant_store(chunks)

    # Update app state so subsequent queries use the new store immediately
    request.app.state.qdrant_store = updated_store

    # ── Step 5: Log to Supabase ────────────────────────────────────────────────
    for record in new_records:
        log_document_ingested(
            doc_id=record.doc_id,
            filename=record.filename,
            doc_type=record.doc_type,
            chunk_count=record.chunk_count,
            file_size_kb=record.file_size_kb,
        )

    elapsed = round(time.time() - start, 2)

    return IngestResponse(
        documents_found=len(records),
        documents_new=len(new_records),
        documents_skipped=skipped_count,
        chunks_indexed=len(chunks),
        latency_seconds=elapsed,
        message=(
            f"Successfully indexed {len(chunks)} chunks from "
            f"{len(new_records)} new document(s) in {elapsed}s."
        ),
    )


@app.post(
    "/ingest/upload",
    summary="Upload and ingest a PDF document",
    tags=["Documents"],
)
async def upload_and_ingest(
    request: Request,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_admin_user),
) -> dict:
    """
    POST /ingest/upload

    Accepts a PDF file upload, saves it to the documents/ folder,
    and immediately triggers ingestion for that file.

    This is the endpoint called by the "Upload Document" button in the UI.
    The file is saved to the configured compliance_docs_dir folder and
    then processed by the standard ingestion pipeline.

    Protected: requires 'admin' or 'auditor' role.
    """
    import shutil
    from pathlib import Path

    # Validate that the uploaded file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Received: {file.filename}",
        )

    # Save the file to the documents folder
    docs_path = Path(settings.compliance_docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    dest = docs_path / file.filename

    try:
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File save failed: {e}",
        )
    finally:
        file.file.close()

    # Run the ingestion pipeline for this specific file
    start = time.time()
    records = scan_compliance_docs()

    # Find the record that matches the uploaded file
    target_records = [r for r in records if r.filename == file.filename and not r.already_indexed]

    if not target_records:
        return {
            "message": f"'{file.filename}' was saved but is already indexed or empty.",
            "filename": file.filename,
            "chunks_indexed": 0,
        }

    raw_pages = extract_all_documents(target_records)
    chunks = chunk_documents(raw_pages)

    if chunks:
        updated_store = build_qdrant_store(chunks)
        request.app.state.qdrant_store = updated_store

        for record in target_records:
            log_document_ingested(
                doc_id=record.doc_id,
                filename=record.filename,
                doc_type=record.doc_type,
                chunk_count=len(chunks),
                file_size_kb=record.file_size_kb,
            )

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "filename": file.filename,
        "pages_extracted": len(raw_pages),
        "chunks_indexed": len(chunks),
        "latency_seconds": round(time.time() - start, 2),
    }


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=ChatResponse,
    summary="Submit a compliance question",
    tags=["Query"],
)
async def query(
    request: Request,
    body: QueryRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> ChatResponse:
    """
    POST /query

    The main endpoint. Accepts a compliance question and returns a
    structured ComplianceReport with verdict, analysis, and citations.

    Protected: any authenticated user can query.
    Rate limited: max N requests per minute (from settings.rate_limit_per_minute).

    The qdrant_store is read from app.state so it benefits from the
    startup initialisation and any updates from /ingest calls.
    """
    # Enforce rate limit per user
    check_rate_limit(current_user.user_id)

    # Get the Qdrant store from application state
    qdrant_store = request.app.state.qdrant_store

    if qdrant_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialised. Check startup logs.",
        )

    start = time.time()

    # Run the complete RAG pipeline
    report = run_compliance_query(
        user_id=current_user.user_id,
        query=body.question,
        qdrant_store=qdrant_store,
        doc_type_filter=body.doc_type_filter,
    )

    if report is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG chain failed. Check engine logs for details.",
        )

    latency_ms = (time.time() - start) * 1000
    session_id = get_or_create_session(current_user.user_id)

    return ChatResponse(
        session_id=session_id,
        query=body.question,
        report=report,
        latency_ms=latency_ms,
        sources_count=len(report.citations),
    )


# ── Document Library ──────────────────────────────────────────────────────────

@app.get(
    "/documents",
    summary="List all indexed documents",
    tags=["Documents"],
)
async def list_documents(
    current_user: UserInDB = Depends(get_current_user),
) -> dict:
    """
    GET /documents

    Returns the list of all compliance documents currently indexed.
    Used by the frontend to display the "Document Library" panel.

    Protected: any authenticated user can view the library.
    """
    docs = get_indexed_documents()
    return {"documents": docs, "total": len(docs)}


# ── Session History ───────────────────────────────────────────────────────────

@app.get(
    "/history",
    summary="Get conversation history for current user",
    tags=["Query"],
)
async def get_history(
    current_user: UserInDB = Depends(get_current_user),
    limit: int = 20,
) -> dict:
    """
    GET /history

    Returns the most recent N messages for the authenticated user.
    Used by the frontend to restore the conversation view on page load.

    Protected: each user can only see their own history.
    """
    from app.database import load_session_history

    messages = load_session_history(user_id=current_user.user_id, limit=limit)
    return {
        "messages": [m.model_dump() for m in messages],
        "count": len(messages),
    }


# ── Root redirect → UI ────────────────────────────────────────────────────────
# Visiting the bare domain (e.g. https://lexai.onrender.com) opens the chat UI.
# We use FileResponse directly so it works whether ui/ is mounted or not.

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend UI at the root URL."""
    import os as _os
    ui_path = _os.path.join(_os.path.dirname(__file__), "..", "ui", "index.html")
    if _os.path.isfile(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {"message": "LexAI Compliance API", "docs": "/docs"}
