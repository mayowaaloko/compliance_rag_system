# LexAI — Compliance Intelligence Platform

> Production-grade Retrieval-Augmented Generation for Nigerian regulatory documents.  
> Built for compliance analysts, auditors, and legal professionals at Big 4 and enterprise firms.

---

## Overview

LexAI is a self-hosted, domain-specific RAG system purpose-built for Nigerian compliance work. It ingests regulatory PDFs — statutes, circulars, guidelines, audit templates — and answers natural-language compliance questions with structured, auditable responses.

Every answer returns a typed verdict (`COMPLIANT`, `NON_COMPLIANT`, `INSUFFICIENT_DATA`, `NEEDS_REVIEW`), a risk level, a confidence score, cited source pages, and a recommended action. Every query is logged to an immutable audit trail. This is not a generic chatbot — it is a compliance decision support tool.

**Current document coverage includes:**
- Nigeria Data Protection Regulation (NDPR) 2019
- Nigeria Data Protection Act (NDP Act) 2023
- Companies and Allied Matters Act (CAMA) 2020
- Companies Regulations 2021
- Labour Act
- National Tax Administration Act (NTAA) 2025
- Nigeria Revenue Service Act
- Persons with Significant Control (PSC) Guidelines
- National Digital Economy Policy and Strategy
- NDPC Journal 2026

---

## Architecture

```
Browser
  │
  ▼
FastAPI (LexAI API)
  │
  ├── Auth layer       JWT tokens, bcrypt password hashing, role-based access
  ├── RAG engine       Multi-query expansion → Hybrid retrieval → Cross-encoder reranking → LLM generation
  ├── Vector store     Qdrant Cloud — dense (OpenAI) + sparse (BM25) hybrid search
  ├── Database         Supabase (PostgreSQL) — users, sessions, messages, audit log
  └── Evaluation       RAGAS faithfulness + answer relevancy scoring
```

**Retrieval pipeline per query:**
1. Query rewriting — the original question is expanded into 3 alternative formulations using an LLM to maximise vocabulary coverage across the document corpus
2. Hybrid retrieval — all 4 query variants are searched against Qdrant using both dense semantic vectors (OpenAI `text-embedding-3-small`, 1536-dim) and sparse BM25 keyword vectors, fused with Reciprocal Rank Fusion
3. Cross-encoder reranking — the candidate pool (up to 80 chunks) is reranked by a `ms-marco-MiniLM-L-6-v2` cross-encoder; only the top 5 survive
4. Structured generation — the reranked context is injected into a compliance auditor prompt; OpenAI `gpt-4o-mini` returns a Pydantic-validated `ComplianceReport` object (no free-form text)

---

## Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| LLM (generation) | OpenAI GPT-4o-mini |
| LLM (query rewriting) | Groq (Llama 3.3 70B) |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Sparse embeddings | FastEmbed BM25 (local, no API) |
| Vector database | Qdrant Cloud (hybrid search) |
| Reranker | HuggingFace cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF extraction | pymupdf4llm (Markdown-preserving) |
| Database | Supabase (PostgreSQL) |
| Auth | OAuth2 Password Flow + JWT (HS256) + bcrypt |
| Evaluation | RAGAS (faithfulness, answer relevancy, factual correctness) |
| Deployment | Docker + Render |
| Observability | LangSmith (optional) |

---

## Project Structure

```
compliance_rag/
├── app/
│   ├── config.py        Settings — pydantic-settings, .env loading, env export
│   ├── schemas.py       All Pydantic models — API I/O, DB rows, RAG pipeline types
│   ├── auth.py          JWT creation/verification, password hashing, FastAPI dependencies
│   ├── database.py      All Supabase operations — users, sessions, messages, audit log
│   ├── ingestion.py     PDF scan → extraction → chunking pipeline
│   ├── vector_store.py  Qdrant collection management, hybrid indexing
│   ├── engine.py        Multi-query retrieval, reranking, LCEL chain, RAGAS evaluation
│   └── main.py          FastAPI application — routes, middleware, static file serving
├── ui/
│   └── index.html       Single-file frontend (no build step required)
├── Dockerfile           Multi-stage build (builder + slim runtime)
├── docker-compose.yml   Local development stack
├── render.yaml          Render deployment blueprint
├── requirements.txt     Python dependencies
└── supabase_schema.sql  Database schema — run once in Supabase SQL editor
```

---

## Local Development

**Prerequisites:** Python 3.11, a Qdrant Cloud account, a Supabase project, OpenAI and Groq API keys.

```bash
# Clone and enter the project
git clone https://github.com/your-username/compliance_rag_system.git
cd compliance_rag_system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in .env with your API keys (see Configuration section below)

# Run the Supabase schema
# Paste supabase_schema.sql into your Supabase SQL editor and execute

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the Swagger API reference.

---

## Configuration

All configuration is through environment variables. Copy `.env.example` to `.env` and populate:

```env
# LLM & Embeddings
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=text-embedding-3-small

# Authentication
JWT_SECRET_KEY=<output of: openssl rand -hex 32>
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=compliance_docs

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# RAG Hyperparameters
CHUNK_SIZE=600
CHUNK_OVERLAP=80
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
MAX_CONTEXT_TOKENS=7000

# Rate limiting (per user per minute)
RATE_LIMIT_PER_MINUTE=10

# Optional: LangSmith observability
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=compliance-rag-2026
```

---

## Document Ingestion

Place PDFs in the `documents/` folder and call the ingest endpoint, or use the upload button in the UI.

```bash
# Trigger ingestion via API (requires admin or auditor role)
curl -X POST https://your-domain.com/ingest \
  -H "Authorization: Bearer <your-token>"
```

The ingestion pipeline is idempotent — re-running it on a folder that has already been processed skips existing documents. Document identity is determined by `sha256(filepath + mtime)[:16]`, so updating a file triggers automatic re-indexing on the next ingest call.

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | System status and collection point count |
| `POST` | `/auth/register` | None | Create a new user account |
| `POST` | `/auth/token` | None | Login, returns JWT access token |
| `POST` | `/query` | Required | Submit a compliance question |
| `POST` | `/ingest` | Admin/Auditor | Trigger document ingestion pipeline |
| `POST` | `/ingest/upload` | Admin/Auditor | Upload and immediately ingest a PDF |
| `GET` | `/documents` | Required | List all indexed documents |
| `GET` | `/history` | Required | Fetch conversation history |

Full interactive documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Deployment (Render)

The repository includes a `render.yaml` blueprint. Deployment is:

1. Push the repository to GitHub
2. Connect the repo in the Render dashboard → **New → Web Service**
3. Add the secret environment variables in Render's **Environment** tab (paste from your `.env`)
4. Click **Deploy**

Render builds the Docker image from the `Dockerfile` and serves the API + UI from a single container. Qdrant Cloud and Supabase are external services — nothing changes for them at deployment time.

The Render free tier sleeps after 15 minutes of inactivity. For a compliance tool in active use, the **Starter plan ($7/month)** is recommended to keep the service always-on.

---

## Database Schema

Five tables are required in Supabase. Run `supabase_schema.sql` once in the SQL editor:

| Table | Purpose |
|---|---|
| `compliance_users` | Registered users with bcrypt-hashed passwords |
| `compliance_sessions` | Active chat sessions, 4-hour idle expiry |
| `compliance_messages` | Every user question and assistant answer |
| `compliance_audit_log` | Immutable compliance decision records (INSERT-only) |
| `compliance_documents` | Registry of indexed PDFs with chunk counts |

The `compliance_audit_log` table has Row Level Security configured to allow INSERT but block UPDATE and DELETE. These records are compliance evidence and must not be altered after the fact.

---

## Evaluation

The system includes a RAGAS evaluation suite measuring:

- **Faithfulness** — fraction of the answer supported by retrieved context (target ≥ 0.85). This directly measures hallucination risk, the most critical failure mode for a compliance tool.
- **Answer Relevancy** — how well the answer addresses the question (target ≥ 0.80)
- **Factual Correctness** — agreement between stated facts and retrieved context

To run an evaluation:

```python
from app.engine import run_faithfulness_evaluation
from app.vector_store import get_query_store

store = get_query_store()
results = run_faithfulness_evaluation(your_question_list, qdrant_store=store)
```

Budget approximately $0.01 per question across three metrics.

---

## Security

- Passwords are bcrypt-hashed before storage. Plaintext passwords never touch the database.
- JWT tokens are signed with HS256. Token expiry defaults to 8 hours.
- The API is the only service that communicates with Supabase. The anon key is scoped to INSERT on audit tables only.
- The Docker container runs as a non-root user (`uid=1001`).
- Rate limiting is enforced per user on the `/query` endpoint to prevent LLM cost exposure.
- The `.env` file is excluded from version control via `.gitignore`. Never commit it.

---

## User Roles

| Role | Permissions |
|---|---|
| `analyst` | Query documents, view history |
| `auditor` | Query documents, view history, trigger ingestion |
| `admin` | All of the above |

---

## Licence

MIT — see `LICENSE`.

---

*Built with LangChain 0.3 LCEL · Qdrant hybrid search · OpenAI · Groq · Supabase · FastAPI*
