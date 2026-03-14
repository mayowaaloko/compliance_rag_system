-- ─────────────────────────────────────────────────────────────────────────────
-- supabase_schema.sql — Compliance RAG Database Schema
-- ─────────────────────────────────────────────────────────────────────────────
--
-- HOW TO USE:
--   1. Open your Supabase project dashboard
--   2. Go to: SQL Editor → New Query
--   3. Paste this entire file and click Run
--   4. All tables will be created (safe to run again — uses IF NOT EXISTS)
--
-- TABLE OVERVIEW:
--   compliance_users        — registered users (auth)
--   compliance_sessions     — active chat sessions
--   compliance_messages     — every question and answer
--   compliance_audit_log    — immutable compliance decision records
--   compliance_documents    — registry of indexed PDFs
-- ─────────────────────────────────────────────────────────────────────────────


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. USERS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
-- Stores registered users with their bcrypt-hashed passwords.
-- This is separate from Supabase Auth — we manage authentication ourselves
-- so we have full control over the JWT payload and role system.
--
-- Why store full_name?
-- The audit log references user_id (UUID), which is opaque.
-- full_name gives human-readable context in audit reports
-- (e.g. "John Doe submitted this query" not "3f2c1a9b... submitted this query").

CREATE TABLE IF NOT EXISTS compliance_users (
    user_id         TEXT PRIMARY KEY,           -- UUID generated in Python
    username        TEXT UNIQUE NOT NULL,       -- Human-readable login (e.g. "john.doe")
    full_name       TEXT NOT NULL,              -- Display name (e.g. "John Doe")
    hashed_password TEXT NOT NULL,              -- bcrypt hash — NEVER plaintext
    role            TEXT NOT NULL DEFAULT 'analyst',  -- 'analyst', 'auditor', 'admin'
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast login lookups (called on every authenticated request)
CREATE INDEX IF NOT EXISTS idx_users_username ON compliance_users (username);

-- Comment explaining the table for future DBAs
COMMENT ON TABLE compliance_users IS
    'Application users. Passwords are bcrypt-hashed by the API before storage.';


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. SESSIONS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
-- Tracks active chat sessions. A session groups related messages into a
-- conversation. Sessions expire after 4 hours of inactivity (enforced in code).
--
-- Why a sessions table and not just timestamps on messages?
-- Sessions allow the app to restore conversation context efficiently:
-- "give me all messages for this session" is one indexed query.
-- Without sessions, restoring context requires scanning all user messages
-- and inferring conversation boundaries from timestamps.

CREATE TABLE IF NOT EXISTS compliance_sessions (
    session_id      TEXT PRIMARY KEY,           -- UUID generated in Python
    user_id         TEXT NOT NULL REFERENCES compliance_users(user_id) ON DELETE CASCADE,
    last_active     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for finding recent sessions by user (called on every query to get/create session)
CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON compliance_sessions (user_id, last_active DESC);

COMMENT ON TABLE compliance_sessions IS
    'Chat sessions. One session per user per 4-hour window. Enables conversation continuity.';


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. MESSAGES TABLE
-- ─────────────────────────────────────────────────────────────────────────────
-- Every user question and assistant answer, in order.
-- This is the conversation history that enables contextual follow-up questions.
--
-- Why store both user and assistant messages in one table?
-- A single table with a 'role' column makes timeline reconstruction trivial:
--   SELECT * FROM compliance_messages WHERE user_id = $1 ORDER BY created_at
-- A two-table design requires a UNION — more complex, no benefit.

CREATE TABLE IF NOT EXISTS compliance_messages (
    message_id      TEXT PRIMARY KEY,           -- UUID
    session_id      TEXT NOT NULL REFERENCES compliance_sessions(session_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES compliance_users(user_id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,              -- The question or answer text
    verdict         TEXT,                       -- Only set for assistant messages (COMPLIANT etc.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for loading conversation history (called after every query)
CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON compliance_messages (user_id, created_at DESC);

-- Index for session-level conversation loading
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON compliance_messages (session_id, created_at ASC);

COMMENT ON TABLE compliance_messages IS
    'All chat messages. role=user for questions, role=assistant for compliance reports.';


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. AUDIT LOG TABLE
-- ─────────────────────────────────────────────────────────────────────────────
-- Immutable compliance decision records. INSERT-ONLY.
-- This table is compliance evidence — it must NEVER be updated or deleted.
-- In a real Big 4 scenario, altering these records could constitute tampering.
--
-- Why store latency_ms?
-- Compliance SLAs may require responses within N seconds.
-- Storing latency allows proactive alerting and trend analysis.
--
-- Why JSONB for sources_used?
-- JSONB allows rich querying:
--   "Show all decisions that cited NDPR.pdf page 12"
-- A TEXT column would require LIKE queries (slow and imprecise).

CREATE TABLE IF NOT EXISTS compliance_audit_log (
    log_id          TEXT PRIMARY KEY,           -- UUID
    user_id         TEXT NOT NULL,              -- Who submitted the query
    session_id      TEXT NOT NULL,              -- Which session
    query           TEXT NOT NULL,              -- The original compliance question
    verdict         TEXT NOT NULL,              -- COMPLIANT / NON_COMPLIANT / etc.
    risk_level      TEXT NOT NULL,              -- LOW / MEDIUM / HIGH / CRITICAL
    confidence      FLOAT NOT NULL,             -- Model confidence 0.0–1.0
    sources_used    JSONB NOT NULL DEFAULT '[]'::jsonb,  -- List of citation strings
    latency_ms      FLOAT NOT NULL,             -- End-to-end query latency
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for per-user audit history reporting
CREATE INDEX IF NOT EXISTS idx_audit_user_time
    ON compliance_audit_log (user_id, created_at DESC);

-- Index for verdict-based reporting (e.g. "all NON_COMPLIANT decisions this month")
CREATE INDEX IF NOT EXISTS idx_audit_verdict
    ON compliance_audit_log (verdict, created_at DESC);

-- Index for risk-level reporting (e.g. "all CRITICAL decisions")
CREATE INDEX IF NOT EXISTS idx_audit_risk
    ON compliance_audit_log (risk_level, created_at DESC);

-- Prevent any UPDATE or DELETE on this table (Row-Level Security)
-- This enforces immutability at the database level, not just the application level
ALTER TABLE compliance_audit_log ENABLE ROW LEVEL SECURITY;

-- Allow INSERT from the service role (our API)
CREATE POLICY audit_insert_policy ON compliance_audit_log
    FOR INSERT
    WITH CHECK (true);  -- The API handles auth — Supabase anon key only inserts

-- Block all UPDATE and DELETE (no policy = blocked)
-- Note: Supabase's service role bypasses RLS — restrict at the API level too

COMMENT ON TABLE compliance_audit_log IS
    'Immutable compliance decisions. INSERT-ONLY. Never UPDATE or DELETE — this is audit evidence.';


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. DOCUMENTS TABLE
-- ─────────────────────────────────────────────────────────────────────────────
-- Registry of every PDF that has been successfully indexed into Qdrant.
-- Used by the GET /documents endpoint to show the document library in the UI.
--
-- Why a SQL table for this and not just Qdrant metadata?
-- Qdrant stores vectors — it is not a document registry.
-- This table gives us a SQL-queryable, human-readable index with:
--   - When each document was ingested
--   - How many chunks each document produced
--   - File size for storage budgeting
-- UPSERT on doc_id: re-ingesting the same file updates the row rather than duplicating.

CREATE TABLE IF NOT EXISTS compliance_documents (
    doc_id          TEXT PRIMARY KEY,           -- 16-char sha256(filepath+mtime)
    filename        TEXT NOT NULL,              -- Original PDF filename
    doc_type        TEXT NOT NULL,              -- regulation, legislation, policy, etc.
    chunk_count     INTEGER NOT NULL DEFAULT 0, -- How many Qdrant points this doc produced
    file_size_kb    FLOAT NOT NULL DEFAULT 0.0, -- File size at ingestion time
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingested_by     TEXT                        -- Optional: who triggered the ingestion
);

-- Index for chronological listing (most recently ingested first in the UI)
CREATE INDEX IF NOT EXISTS idx_documents_ingested
    ON compliance_documents (ingested_at DESC);

-- Index for filtering by document type
CREATE INDEX IF NOT EXISTS idx_documents_type
    ON compliance_documents (doc_type);

COMMENT ON TABLE compliance_documents IS
    'Registry of all PDFs indexed into Qdrant. Used to populate the document library UI.';


-- ─────────────────────────────────────────────────────────────────────────────
-- VERIFICATION QUERY
-- Run this after creating the tables to confirm everything was created:
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns c
     WHERE c.table_name = t.table_name
     AND c.table_schema = 'public') AS column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
AND table_name LIKE 'compliance_%'
ORDER BY table_name;
