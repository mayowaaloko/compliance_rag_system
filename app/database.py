"""
database.py — Supabase Database Layer
======================================

This module owns ALL communication with Supabase (PostgreSQL).

Why a separate database module?
    Separation of concerns. The RAG engine (engine.py) should not know
    how sessions are stored. The auth module (auth.py) should not know
    SQL. This module knows about tables and rows — others don't.

Tables used (create these in your Supabase SQL editor):
    compliance_users    — registered users with hashed passwords
    compliance_sessions — active chat sessions per user
    compliance_messages — every user question and assistant answer
    compliance_audit_log — immutable compliance decision records
    compliance_documents — index of every PDF that has been ingested

SQL to create these tables is in: supabase_schema.sql
"""

import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app.config import settings
from app.schemas import AuditLogEntry, ChatMessage, UserInDB

# ─────────────────────────────────────────────────────────────────────────────
# Client Initialisation
# ─────────────────────────────────────────────────────────────────────────────

# We attempt to initialise the Supabase client once at import time.
# If credentials are missing or the server is unreachable, we set
# supabase_client to None and degrade gracefully (logs go to console).

try:
    from supabase import create_client, Client as SupabaseClient

    _SUPABASE_AVAILABLE = bool(settings.supabase_url and settings.supabase_anon_key)
except ImportError:
    _SUPABASE_AVAILABLE = False

supabase_client: Optional[object] = None

if _SUPABASE_AVAILABLE:
    try:
        supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )
    except Exception as e:
        print(f"[database] Supabase client initialisation failed: {e}")
        supabase_client = None


# ─────────────────────────────────────────────────────────────────────────────
# User Operations
# ─────────────────────────────────────────────────────────────────────────────


async def get_user_by_username(username: str) -> Optional[UserInDB]:
    """
    What does this function do?
    Looks up a user by their username in Supabase.
    Returns a UserInDB object or None if not found.

    Called by auth.py's get_current_user dependency on every protected request.
    The username is extracted from the JWT token and then verified here.
    """
    if not supabase_client:
        return None

    try:
        result = (
            supabase_client.table("compliance_users")
            .select("*")
            .eq("username", username)
            .limit(1)
            .execute()
        )

        if result.data:
            return UserInDB(**result.data[0])

        return None

    except Exception as e:
        print(f"[database] get_user_by_username failed: {e}")
        return None


async def create_user(user: UserInDB) -> Optional[UserInDB]:
    """
    What does this function do?
    Inserts a new user record into Supabase.

    The caller (main.py's /auth/register endpoint) is responsible for
    hashing the password BEFORE passing the UserInDB object here.
    This function stores whatever it receives.

    Returns the created UserInDB if successful, None if it failed.
    """
    if not supabase_client:
        print("[database] Supabase not available. User not persisted.")
        return user  # Return the object anyway for local development

    try:
        result = (
            supabase_client.table("compliance_users")
            .insert(user.model_dump())
            .execute()
        )

        if result.data:
            return UserInDB(**result.data[0])

        return None

    except Exception as e:
        print(f"[database] create_user failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Session Operations
# ─────────────────────────────────────────────────────────────────────────────


def get_or_create_session(user_id: str) -> str:
    """
    What does this function do?
    Looks up the most recent active session for this user_id.
    If one exists and is recent (within 4 hours): reuses it.
    Otherwise: creates a new session_id and inserts it into Supabase.

    Why reuse sessions?
    This is what makes the chat feel persistent — the user can close
    the browser and return within 4 hours to find their conversation intact.
    After 4 hours, a clean new session begins.

    Why UUID and not an auto-increment integer?
    UUIDs are safe to expose in URLs and logs. Auto-increment integers
    reveal how many sessions exist (an information leak).
    """
    import uuid

    new_session_id = str(uuid.uuid4())

    if not supabase_client:
        return new_session_id

    try:
        # Only reuse sessions that have been active in the last 4 hours
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()

        result = (
            supabase_client.table("compliance_sessions")
            .select("session_id")
            .eq("user_id", user_id)
            .gte("last_active", cutoff)
            .order("last_active", desc=True)
            .limit(1)
            .execute()
        )

        if result.data:
            # Active session found — reuse it
            return result.data[0]["session_id"]

        # No recent session — create a new one
        supabase_client.table("compliance_sessions").insert(
            {
                "session_id": new_session_id,
                "user_id": user_id,
                "last_active": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()

        return new_session_id

    except Exception as e:
        print(f"[database] get_or_create_session failed: {e}")
        return new_session_id


def update_session_activity(session_id: str) -> None:
    """
    What does this function do?
    Bumps the last_active timestamp on a session to "now".
    This is called after every successful query so the 4-hour
    idle timeout restarts from the most recent interaction.
    """
    if not supabase_client:
        return

    try:
        supabase_client.table("compliance_sessions").update(
            {"last_active": datetime.now(timezone.utc).isoformat()}
        ).eq("session_id", session_id).execute()
    except Exception:
        pass  # Non-critical: losing the timestamp update is acceptable


# ─────────────────────────────────────────────────────────────────────────────
# Message Logging
# ─────────────────────────────────────────────────────────────────────────────


def log_chat_message(message: ChatMessage) -> None:
    """
    What does this function do?
    Persists one chat message (user question OR assistant answer) to Supabase.
    This is what enables conversation history to survive server restarts.

    Failure is logged but not re-raised — a logging failure should never
    prevent the user from receiving their compliance answer.
    """
    if not supabase_client:
        return

    try:
        supabase_client.table("compliance_messages").insert(
            message.model_dump()
        ).execute()
    except Exception as e:
        print(f"[database] log_chat_message failed: {e}")


def load_session_history(user_id: str, limit: int = 20) -> List[ChatMessage]:
    """
    What does this function do?
    Fetches the most recent N messages for this user from Supabase.
    Returns them as ChatMessage objects in chronological (oldest-first) order.

    Used to provide conversation context when the LLM generates its answer.
    Without history, each question is answered in isolation.
    """
    if not supabase_client:
        return []

    try:
        result = (
            supabase_client.table("compliance_messages")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        # reversed() restores chronological order (oldest first)
        messages = [ChatMessage(**row) for row in reversed(result.data)]
        return messages

    except Exception as e:
        print(f"[database] load_session_history failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Audit Logging
# ─────────────────────────────────────────────────────────────────────────────


def log_audit_entry(entry: AuditLogEntry) -> None:
    """
    What does this function do?
    Writes one immutable compliance decision to the audit log table.
    This record is insert-only — it is NEVER updated or deleted.
    It is compliance evidence that can be produced in an audit.

    Why store sources_used as JSON?
    Supabase's Postgres stores it as JSONB (binary JSON).
    This allows efficient querying like: "all cases that cited NDPR p.5"
    """
    if not supabase_client:
        # Graceful degradation: print to console if Supabase is unavailable
        row = entry.model_dump()
        row["sources_used"] = json.dumps(row["sources_used"])
        print("[AUDIT LOG FALLBACK]", json.dumps(row, indent=2))
        return

    try:
        row = entry.model_dump()
        row["sources_used"] = json.dumps(row["sources_used"])
        supabase_client.table("compliance_audit_log").insert(row).execute()
    except Exception as e:
        print(f"[database] log_audit_entry failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Document Registry
# ─────────────────────────────────────────────────────────────────────────────


def log_document_ingested(
    doc_id: str,
    filename: str,
    doc_type: str,
    chunk_count: int,
    file_size_kb: float,
) -> None:
    """
    What does this function do?
    Records that a PDF has been successfully indexed into Qdrant.
    This lets us display a "library" of indexed documents in the UI
    and track ingestion history.

    Why a separate table for this and not just Qdrant metadata?
    Qdrant stores vectors and payloads — it is not a document registry.
    This table gives us a SQL-queryable index of what's been ingested,
    when, and how many chunks each document produced.
    """
    if not supabase_client:
        return

    try:
        supabase_client.table("compliance_documents").upsert(
            {
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": doc_type,
                "chunk_count": chunk_count,
                "file_size_kb": file_size_kb,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="doc_id",  # Update if this doc_id already exists
        ).execute()
    except Exception as e:
        print(f"[database] log_document_ingested failed: {e}")


def get_indexed_documents() -> List[dict]:
    """
    What does this function do?
    Returns the list of all documents currently in the compliance library.
    Used by the GET /documents endpoint to show users what's available.
    """
    if not supabase_client:
        return []

    try:
        result = (
            supabase_client.table("compliance_documents")
            .select("*")
            .order("ingested_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[database] get_indexed_documents failed: {e}")
        return []
