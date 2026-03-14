"""
vector_store.py — Qdrant Vector Store Management
==================================================

This module owns ALL communication with Qdrant.
No other module calls QdrantClient directly — they use the functions here.

Responsibilities:
    1. ensure_collection()     — create the collection if it doesn't exist
    2. build_qdrant_store()    — index new chunks into the vector store
    3. get_qdrant_store()      — return a connected store for query-time use

Why hybrid search (dense + sparse)?
    Legal and compliance text requires BOTH retrieval modes:
    - Dense (semantic): finds "data controller obligations" even if the
      document says "obligations of a processor" — semantic similarity.
    - Sparse (BM25): finds "Article 4.1(7) NDPR" exactly — keyword precision.
    Hybrid fuses both scores using Reciprocal Rank Fusion (RRF) at query time.

Why Qdrant over FAISS or Pinecone?
    - Native hybrid search (FAISS has no sparse vectors)
    - Persistent storage with Docker volume (FAISS loses data on restart)
    - Payload filtering (Pinecone charges per filter)
    - Production-grade HNSW index (tunable m and ef_construct)
    - Payload indexes for O(log n) metadata filtering
"""

import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Models
# ─────────────────────────────────────────────────────────────────────────────


def _build_dense_embeddings() -> "OpenAIEmbeddings":
    """
    What does this function return?
    An OpenAI embeddings client configured for text-embedding-3-small.

    Why wrap in a cache?
    The CacheBackedEmbeddings layer stores computed embeddings on disk.
    If the same text is embedded twice (e.g. re-ingesting after a crash),
    the second call returns the cached value instantly — no OpenAI API call.
    This saves money and speeds up re-ingestion dramatically.

    Why text-embedding-3-small (1536 dims) over text-embedding-3-large (3072)?
    3072-dim gives ~3% better retrieval quality but:
    - Doubles Qdrant storage requirements
    - Doubles embedding API cost
    - Doubles vector comparison time at retrieval
    For compliance RAG with high-quality reranking, 1536 dims is sufficient.
    """
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

    # Create the cache directory if it doesn't exist
    cache_dir = Path(settings.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The underlying OpenAI client — makes real API calls
    raw_embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        dimensions=1536,
    )

    # The file-backed cache layer — wraps the raw client
    # namespace="openai_3_small" ensures cache files don't conflict if
    # we ever switch models (different namespace = different cache folder)
    lc_store = LocalFileStore(str(cache_dir / "lc_cache"))
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=raw_embeddings,
        document_embedding_cache=lc_store,
        namespace="openai_3_small",
    )

    return cached_embeddings


def _build_sparse_embeddings() -> FastEmbedSparse:
    """
    What does this function return?
    A BM25 sparse embedding model that runs locally (no API call needed).

    Why BM25 for sparse embeddings?
    BM25 (Best Match 25) is the gold standard for keyword relevance scoring.
    It weights term frequency and inverse document frequency — the same
    algorithm used by Elasticsearch and early Google.

    Why "Qdrant/bm25"?
    FastEmbed's Qdrant/bm25 model runs fully locally via ONNX.
    Zero latency, zero cost, no external dependency for the sparse vectors.
    """
    return FastEmbedSparse(model_name="Qdrant/bm25")


# ─────────────────────────────────────────────────────────────────────────────
# Collection Management
# ─────────────────────────────────────────────────────────────────────────────


def get_qdrant_client() -> QdrantClient:
    """
    What does this function return?
    A connected Qdrant client ready to make API calls.

    Why timeout=40?
    Qdrant collection creation (especially with index building) can take
    longer than the default 10 second timeout on large datasets.
    40 seconds gives headroom without being indefinitely blocking.
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=40,
    )


def ensure_collection() -> bool:
    """
    What does this function do?
    Creates the Qdrant collection with production-tuned parameters IF it
    does not already exist. Safe to call on every application startup.

    Why 'ensure' instead of 'create'?
    'ensure' signals idempotency: calling it 10 times is the same as calling
    it once. 'create' implies failure if the collection exists.
    This is the correct behaviour for a startup check.

    Returns:
        True  — collection was just created
        False — collection already existed (no action taken)

    Collection design:
    ┌─────────────────────────────────────────────────────────────────┐
    │ DENSE VECTOR  ('dense')                                         │
    │   Size: 1536 floats (text-embedding-3-small)                    │
    │   Distance: COSINE                                              │
    │   HNSW m=16, ef_construct=100                                   │
    │   Purpose: semantic similarity search                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ SPARSE VECTOR ('sparse')                                        │
    │   Type: BM25 integer weights                                    │
    │   Storage: RAM (on_disk=False)                                  │
    │   Purpose: keyword precision search                             │
    └─────────────────────────────────────────────────────────────────┘
    Both are stored per point. Qdrant fuses scores with RRF at query time.

    HNSW parameters explained:
        m=16:            number of bidirectional links per graph node.
                         Higher = more RAM, better recall. 16 is production standard.
        ef_construct=100: candidates considered during index build.
                         Higher = better index quality, slower build time.
                         100 is a safe default for compliance documents.
    """
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    # ── Already exists ────────────────────────────────────────────────────────
    if settings.qdrant_collection_name in existing:
        info = client.get_collection(settings.qdrant_collection_name)
        print(
            f"[vector_store] Collection '{settings.qdrant_collection_name}' "
            f"already exists. Points: {info.points_count}"
        )
        return False

    # ── Create new collection ────────────────────────────────────────────────
    print(f"[vector_store] Creating collection '{settings.qdrant_collection_name}'...")

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        # ── Dense vector config ──────────────────────────────────────────────
        vectors_config={
            "dense": qm.VectorParams(
                size=1536,  # Must match embedding model dimension
                distance=qm.Distance.COSINE,  # Normalised cosine: best for text
                hnsw_config=qm.HnswConfigDiff(
                    m=16,  # Bidirectional links per node in the graph
                    ef_construct=100,  # Build quality vs. speed tradeoff
                ),
            )
        },
        # ── Sparse vector config ─────────────────────────────────────────────
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(
                index=qm.SparseIndexParams(
                    on_disk=False  # Keep in RAM for query-time speed
                )
            )
        },
        # ── Payload (metadata) config ────────────────────────────────────────
        # on_disk_payload=False: keep all metadata in RAM.
        # For 50k–100k compliance document chunks, the payload is small
        # enough to fit in RAM and this makes filter+retrieval very fast.
        on_disk_payload=False,
    )

    # ── Payload indexes ───────────────────────────────────────────────────────
    # Without indexes: Qdrant must scan ALL points for a filter (O(n))
    # With indexes:    Qdrant uses an inverted index for the filter (O(log n))
    # These fields are used in retrieval filters — indexing them is critical.
    payload_indexes = [
        ("doc_id", qm.PayloadSchemaType.KEYWORD),
        ("filename", qm.PayloadSchemaType.KEYWORD),
        ("category", qm.PayloadSchemaType.KEYWORD),
        ("doc_type", qm.PayloadSchemaType.KEYWORD),
        ("year", qm.PayloadSchemaType.INTEGER),
    ]

    for field_name, schema_type in payload_indexes:
        client.create_payload_index(
            collection_name=settings.qdrant_collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )

    print(
        f"[vector_store] Collection created with payload indexes: "
        + ", ".join(f[0] for f in payload_indexes)
    )

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Store Building & Connection
# ─────────────────────────────────────────────────────────────────────────────


def build_qdrant_store(chunks: List[Document]) -> QdrantVectorStore:
    """
    What does this function do?
    If chunks is non-empty: indexes them into Qdrant and returns the store.
    If chunks is empty: connects to the existing collection and returns the store.

    This function handles BOTH cases — initial indexing and subsequent
    query-time connections — with a single clean interface.

    Parameters:
        chunks: List of chunk-level LangChain Documents with metadata.
                Pass [] to connect to an existing collection without indexing.

    Why batch_size=64?
    Qdrant processes vector insertions in batches. 64 balances:
    - Memory per batch (embedding 64 chunks at once)
    - API call overhead (fewer calls than batch_size=1)
    - Reliability (smaller than batch_size=512 → easier to retry on error)

    Why timeout=120 for indexing?
    Large ingestion jobs (10,000+ chunks) can take over a minute.
    A short timeout would interrupt a valid long-running operation.
    """
    dense_embeddings = _build_dense_embeddings()
    sparse_embeddings = _build_sparse_embeddings()

    # ── Index new chunks ─────────────────────────────────────────────────────
    if chunks:
        print(f"[vector_store] Indexing {len(chunks)} chunks into Qdrant...")
        start = time.time()

        store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=settings.qdrant_collection_name,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
            timeout=120,
            batch_size=64,
        )

        # Fetch updated stats after indexing
        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection_name)
        elapsed = time.time() - start

        print(f"[vector_store] Indexed {len(chunks)} chunks in {elapsed:.1f}s")
        print(f"[vector_store] Collection total: {info.points_count} points")

        # Show a breakdown of how many chunks came from each document
        doc_counts = Counter(c.metadata.get("doc_id") for c in chunks)
        print(f"[vector_store] Documents represented: {len(doc_counts)}")
        for doc_id, count in list(doc_counts.items())[:3]:
            print(f"[vector_store]   {doc_id}: {count} chunks")

    # ── Connect to existing collection ───────────────────────────────────────
    else:
        print("[vector_store] No new chunks. Connecting to existing collection...")

        store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=settings.qdrant_collection_name,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
            timeout=60,
        )

        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection_name)
        print(f"[vector_store] Connected. {info.points_count} points in collection.")

    return store


# ─────────────────────────────────────────────────────────────────────────────
# Application Startup Store
# ─────────────────────────────────────────────────────────────────────────────


def get_query_store() -> QdrantVectorStore:
    """
    What does this function do?
    Returns a QdrantVectorStore connected to the existing collection.
    Used at application startup to prepare for query handling.

    This is a thin wrapper that makes the intent explicit:
    "We are NOT indexing — we are connecting for query purposes."
    The underlying build_qdrant_store(chunks=[]) handles the rest.
    """
    return build_qdrant_store(chunks=[])
