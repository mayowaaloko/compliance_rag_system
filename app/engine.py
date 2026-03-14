"""
engine.py — RAG Execution Core
================================

This module contains the entire query-time intelligence of the system.
It is the "brain" that runs when a user asks a compliance question.

Pipeline (in order of execution):
    1. generate_query_variants()    — rewrite the question 3 ways (multi-query)
    2. multi_query_retrieve()       — search Qdrant with all 4 variants
    3. retrieve_and_rerank()        — compress the pool with a cross-encoder
    4. format_context()             — format top chunks into a labelled string
    5. auditor_chain.invoke()       — LLM generates a ComplianceReport
    6. run_compliance_query()       — orchestrates all of the above + logging

Why multi-query?
    A single query may miss relevant chunks if the vocabulary doesn't match.
    Example: "data subject consent" may miss chunks that say "lawful basis
    for processing" — the same concept in different words.
    Multi-query generates 3 reformulations to close this vocabulary gap.

Why cross-encoder reranking?
    The bi-encoder (Qdrant's hybrid search) retrieves 20 candidates fast
    but ranks them with lower accuracy.
    The cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores all 20 candidates
    by reading the (question, chunk) pair together — much higher accuracy.
    We keep only the top 5 after reranking.
    Trade-off: slower (O(n) cross-encoder calls) but much better precision.

Why LCEL (LangChain Expression Language) for the chain?
    LCEL chains are:
    - Composable: each | step is a standalone unit you can test independently
    - Async-ready: ainvoke() and astream() work without changes
    - Observable: LangSmith traces every step automatically when enabled
"""

import time
from typing import Any, Dict, List, Optional

import tiktoken
import cohere
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models as qm

from app.config import settings
from app.database import (
    get_or_create_session,
    load_session_history,
    log_audit_entry,
    log_chat_message,
    update_session_activity,
)
from app.schemas import (
    AuditLogEntry,
    ChatMessage,
    ComplianceReport,
    QueryVariants,
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Clients
# ─────────────────────────────────────────────────────────────────────────────

# Primary LLM: OpenAI — used for the final compliance report generation.
# We use structured output (JSON schema enforcement) — only OpenAI supports
# this reliably with Pydantic models.
llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
)

# Query rewriting LLM: OpenAI (used for multi-query expansion as in the notebook)
# Note: ChatGroq could be used here for lower cost if preferred.
query_llm = ChatGroq(
    model=settings.groq_model,
    temperature=settings.temperature,
    api_key=settings.groq_api_key,
)

# Tokenizer for the context window budget calculation
# We use gpt-4o's tokenizer — it works for all GPT-4 family models
TOKENIZER = tiktoken.encoding_for_model("gpt-4o")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Query Expansion
# ─────────────────────────────────────────────────────────────────────────────

MULTI_QUERY_SYSTEM = (
    "You are a compliance analyst at a Big 4 consulting firm.\n"
    "Rewrite the user question into exactly 3 alternative formulations to maximise\n"
    "retrieval coverage across a regulatory document database.\n\n"
    "Version 1: precise regulatory and legal terminology\n"
    "Version 2: general business and operational language\n"
    "Version 3: from the document perspective — "
    "what does the policy say about X, rather than does X violate Y\n\n"
    "Return ONLY a JSON array of 3 strings. No explanation or preamble."
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", MULTI_QUERY_SYSTEM),
        ("human", "Original question: {question}"),
    ]
)

# LCEL chain: prompt | llm with structured output → QueryVariants object
query_rewrite_chain = MULTI_QUERY_PROMPT | query_llm.with_structured_output(
    QueryVariants
)


def generate_query_variants(question: str) -> List[str]:
    """
    What does this function do?
    Sends the original question to the LLM and gets back 3 alternative
    reformulations. Returns all 4 (original + 3) deduplicated.

    Why deduplicate?
    The LLM occasionally returns a variant identical to the original.
    Deduplication prevents redundant Qdrant queries.

    Failure handling:
    If the LLM call fails (API error, timeout), we fall back to just the
    original question. This ensures query execution continues even if
    multi-query fails — graceful degradation.
    """
    try:
        result: QueryVariants = query_rewrite_chain.invoke({"question": question})
        all_questions = [question] + result.variants

        # Deduplicate while preserving order (set() would shuffle)
        seen = set()
        deduped = [q for q in all_questions if not (q in seen or seen.add(q))]

        return deduped[:4]  # Cap at 4 total (1 original + 3 variants)

    except Exception as e:
        print(f"[engine] Query rewriting fell back to original: {e}")
        return [question]


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Retrieval
# ─────────────────────────────────────────────────────────────────────────────


def multi_query_retrieve(
    question: str,
    qdrant_store: QdrantVectorStore,
    k_per_query: Optional[int] = None,
    doc_type_filter: Optional[str] = None,
) -> List[Document]:
    """
    What does this function do?
    Runs up to 4 hybrid Qdrant searches (one per query variant) and
    returns the deduplicated union of all results.

    Why run each variant separately instead of one big query?
    Qdrant's hybrid search returns the best matches for ONE query.
    Running 4 separate searches and merging gives us a much larger
    and more diverse candidate pool — dramatically improving recall.

    Parameters:
        question:        The original user question.
        qdrant_store:    A connected QdrantVectorStore (injected dependency).
        k_per_query:     How many chunks to fetch per variant.
                         Defaults to settings.top_k_retrieval.
        doc_type_filter: Optional Qdrant payload filter to restrict retrieval
                         to a specific document type (e.g. "regulation").

    Deduplication:
        We track chunk_ids we've already seen. If the same chunk is retrieved
        by multiple variants (which is a good sign — high confidence), we
        include it only once.
    """
    k = k_per_query or settings.top_k_retrieval
    variants = generate_query_variants(question)

    print(f"[engine] Query variants ({len(variants)}):")
    for i, q in enumerate(variants):
        print(f"  [{i}] {q[:80]}..." if len(q) > 80 else f"  [{i}] {q}")

    # Build Qdrant filter if a doc_type is specified
    qdrant_filter = None
    if doc_type_filter:
        qdrant_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="doc_type", match=qm.MatchValue(value=doc_type_filter)
                )
            ]
        )

    seen_ids: set = set()
    candidates: List[Document] = []

    for variant in variants:
        try:
            results = qdrant_store.similarity_search(
                query=variant,
                k=k,
                filter=qdrant_filter,
            )

            for doc in results:
                # Use chunk_id if available, else first 50 chars as a proxy
                chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
                if chunk_id not in seen_ids:
                    candidates.append(doc)
                    seen_ids.add(chunk_id)

        except Exception as e:
            print(f"[engine]  Retrieval error for variant '{variant[:40]}...': {e}")

    print(
        f"\n[engine] Candidate pool: {len(candidates)} unique chunks from {len(variants)} searches"
    )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder Reranking
# ─────────────────────────────────────────────────────────────────────────────


def cohere_rerank(
    question: str,
    candidates: List[Document],
) -> List[Document]:
    """
    What does this function do?
    Sends the candidate chunks and the question to the Cohere Rerank API.
    Cohere scores each (question, chunk) pair together and returns them
    sorted from most to least relevant. We keep the top settings.top_k_rerank.

    Why Cohere Rerank instead of a local cross-encoder?
    The local HuggingFace cross-encoder (ms-marco-MiniLM) requires loading
    PyTorch (~800MB RAM) into the container. Cohere Rerank is an API call —
    no model weights, no GPU, no RAM spike. The quality is comparable or
    better because Cohere's rerank-english-v3.0 was trained on a much larger
    dataset than ms-marco-MiniLM.

    Why rerank at all?
    Qdrant's hybrid search retrieves 20+ candidates fast but ranks them with
    a simple fusion score. The reranker reads each (question, chunk) pair
    together with full cross-attention — much more accurate relevance scoring.
    We narrow 20 candidates to 5 with the reranker.

    Failure handling:
    If the Cohere API call fails (network error, quota exhausted), we fall
    back to returning the top-K candidates in their original retrieval order.
    The system degrades gracefully rather than crashing.
    """
    if not candidates:
        return []

    try:
        co = cohere.Client(api_key=settings.cohere_api_key)

        # Cohere expects a list of plain strings — extract the text from Documents
        docs_text = [doc.page_content for doc in candidates]

        response = co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=docs_text,
            top_n=settings.top_k_rerank,
        )

        # response.results contains RerankResponseResultsItem objects.
        # Each has: .index (position in original list) and .relevance_score (0-1).
        # We rebuild the Document list in reranked order, injecting the score
        # into metadata so it appears in the compliance report citations.
        reranked = []
        for result in response.results:
            doc = candidates[result.index]
            # Attach the Cohere relevance score to the chunk metadata
            doc.metadata["relevance_score"] = round(result.relevance_score, 4)
            reranked.append(doc)

        return reranked

    except Exception as e:
        print(f"[engine] Cohere rerank failed, falling back to retrieval order: {e}")
        # Graceful degradation: return top-K in original retrieval order
        return candidates[: settings.top_k_rerank]


def get_reranker():
    """
    Kept for backwards compatibility with main.py lifespan.
    Cohere rerank is stateless (API call) so there is nothing to initialise.
    This function is now a no-op.
    """
    print("[engine] Reranker: Cohere API (stateless, no preload needed)")


def retrieve_and_rerank(
    question: str,
    qdrant_store: QdrantVectorStore,
    doc_type_filter: Optional[str] = None,
) -> List[Document]:
    """
    What does this function do?
    Combines multi-query retrieval and cross-encoder reranking into a
    single function that returns the top-K most relevant chunks.

    This is the complete retrieval stage of the RAG pipeline:
        multi_query_retrieve() → candidate pool (20+ chunks)
        reranker.compress_documents() → top K chunks (5 by default)

    The reranker also adds 'relevance_score' to each chunk's metadata,
    which is shown in the compliance report as [Rerank score: X.XXXX].
    """
    # Step 1: Multi-query retrieval gives us a large, diverse candidate pool
    candidates = multi_query_retrieve(
        question=question,
        qdrant_store=qdrant_store,
        doc_type_filter=doc_type_filter,
    )

    if not candidates:
        return []

    # Step 2: Cohere reranks the candidate pool via API call.
    # This replaces the local HuggingFace cross-encoder — same quality,
    # no torch dependency, no RAM spike on startup.
    reranked = cohere_rerank(question=question, candidates=candidates)

    # Deduplicate by (filename, page) — same chunk can appear under
    # both 'filename' and 'source' metadata keys from different retrieval paths.
    seen_pages: set = set()
    deduped_reranked: List[Document] = []
    for doc in reranked:
        fname = doc.metadata.get("filename") or doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        key = (fname, page)
        if key not in seen_pages:
            deduped_reranked.append(doc)
            seen_pages.add(key)

    print(
        f"\n[engine] Reranked: {len(candidates)} candidates → top {len(deduped_reranked)}"
    )

    for i, doc in enumerate(deduped_reranked, 1):
        import os as _os

        score = doc.metadata.get("relevance_score", 0.0)
        fname = doc.metadata.get("filename") or doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page", "N/A")
        fname = _os.path.basename(fname) if fname != "N/A" else "N/A"
        print(f"  [{i}] score={score:.4f}  {fname} p.{page}")

    return deduped_reranked


# ─────────────────────────────────────────────────────────────────────────────
# Context Formatting
# ─────────────────────────────────────────────────────────────────────────────


def format_context(docs: List[Document]) -> str:
    """
    What does this function do?
    Converts retrieved Document objects into a numbered, labelled string
    ready to be injected into the compliance prompt as {context}.

    Each source gets a [Source N] label. The LLM uses these labels in
    its citations field: "[Source 1: NDPR.pdf | Page 12 | Type: regulation]"

    Why count tokens?
    LLMs have a context window limit. If we pass 20 long chunks, we may
    exceed the limit, causing an API error or silent truncation.
    tiktoken lets us count tokens before making the API call and truncate
    if necessary.

    Why truncate rather than filter?
    We want to pass AS MUCH context as possible within the budget.
    Truncating at the token limit is more efficient than filtering to
    a fixed number of chunks.
    """
    parts = []
    token_count = 0

    for i, doc in enumerate(docs, 1):
        m = doc.metadata

        # Resolve the display filename.
        # Some chunks from the notebook stored the path under 'source' instead
        # of 'filename' (a LangChain convention). We check both keys and then
        # strip any leading directory path so the label shows just the filename.
        import os as _os

        raw_name = m.get("filename") or m.get("source") or "N/A"
        display_name = _os.path.basename(raw_name) if raw_name != "N/A" else "N/A"

        # Format: [Source 1: filename | Page N | Type: doc_type | Rerank score: X]
        label = (
            f"[Source {i}: {display_name} | "
            f'Page {m.get("page", "N/A")} | '
            f'Type: {m.get("doc_type", "N/A")} | '
            f'Rerank score: {m.get("relevance_score", m.get("rerank_score", "N/A"))}]'
        )

        chunk_text = f"{label}\n{doc.page_content}"
        chunk_tokens = len(TOKENIZER.encode(chunk_text))

        # Stop adding chunks once we approach the context window budget
        if token_count + chunk_tokens > settings.max_context_tokens:
            parts.append(f"[Source {i}+: truncated — context window limit reached]")
            break

        parts.append(chunk_text)
        token_count += chunk_tokens

    # Join with a visible separator so the LLM can clearly tell sources apart
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Compliance Prompt
# ─────────────────────────────────────────────────────────────────────────────

COMPLIANCE_SYSTEM = (
    "You are a senior compliance auditor at a Big 4 firm (Deloitte / KPMG / PwC / EY).\n"
    "Your role is to analyse whether business practices, contracts, or decisions\n"
    "comply with the regulatory documents provided as context.\n\n"
    "CRITICAL RULES:\n"
    "1. Base your analysis ONLY on the retrieved context below. Never use outside knowledge.\n"
    "2. Always cite sources precisely: [Source N: document_name, page X].\n"
    "3. If the context is insufficient, return verdict=INSUFFICIENT_DATA.\n"
    "4. Use exact regulatory language when quoting rules — do not paraphrase legal text.\n"
    "5. If multiple regulations apply, address each one separately.\n"
    "6. Note any conflicting regulations explicitly.\n\n"
    "Retrieved Compliance Context:\n"
    "{context}"
)

COMPLIANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", COMPLIANCE_SYSTEM),
        ("human", "{question}"),
    ]
)

# with_structured_output sends the ComplianceReport schema to OpenAI as a
# tool definition. The model MUST return JSON matching this schema —
# no post-processing or regex parsing needed.
structured_llm = llm.with_structured_output(ComplianceReport)


# ─────────────────────────────────────────────────────────────────────────────
# LCEL Auditor Chain Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_auditor_chain(
    qdrant_store: QdrantVectorStore, doc_type_filter: Optional[str] = None
):
    """
    What does this function do?
    Builds and returns the LCEL auditor chain configured with the given
    qdrant_store and optional filter.

    Why build dynamically instead of a module-level chain?
    The qdrant_store is initialised at application startup (not import time).
    Building the chain here avoids circular imports and allows the store
    to be injected cleanly.

    Data flow through the chain:
        {'question': '...'}
              |
              v  RunnableParallel  (runs both branches in parallel)
              |-- 'context':  retrieve_and_rerank → format_context → string
              |-- 'question': passthrough → original question string
              |
              v  COMPLIANCE_PROMPT (fills {context} and {question})
              |
              v  structured_llm → ComplianceReport Pydantic object

    Why RunnableParallel?
    The context branch (retrieval + formatting) and the question passthrough
    can run simultaneously. In practice, the question passthrough is
    instantaneous — but the pattern is correct and future-proof.
    """

    def retrieve_for_chain(question: str) -> List[Document]:
        """Helper that captures qdrant_store and filter for use in the chain."""
        return retrieve_and_rerank(
            question=question,
            qdrant_store=qdrant_store,
            doc_type_filter=doc_type_filter,
        )

    chain = (
        RunnableParallel(
            {
                "context": (
                    RunnableLambda(lambda x: x["question"])
                    | RunnableLambda(retrieve_for_chain)
                    | RunnableLambda(format_context)
                ),
                "question": RunnablePassthrough()
                | RunnableLambda(lambda x: x["question"]),
            }
        )
        | COMPLIANCE_PROMPT
        | structured_llm
    )

    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration Function
# ─────────────────────────────────────────────────────────────────────────────


def run_compliance_query(
    user_id: str,
    query: str,
    qdrant_store: QdrantVectorStore,
    doc_type_filter: Optional[str] = None,
) -> Optional[ComplianceReport]:
    """
    What does this function do?
    The complete end-to-end compliance query function.
    Wraps the auditor_chain with session management, logging, and error handling.

    Steps executed in order:
        1. Get or restore the user's session from Supabase
        2. Log the user's question to compliance_messages
        3. Build and invoke the auditor_chain
        4. Log the assistant's answer to compliance_messages
        5. Update the session's last_active timestamp
        6. Write an immutable audit record to compliance_audit_log
        7. Return the ComplianceReport for the API response

    Parameters:
        user_id:         The authenticated user's ID (from JWT token)
        query:           The compliance question
        qdrant_store:    Connected vector store (injected from app state)
        doc_type_filter: Optional filter for retrieval

    Returns:
        ComplianceReport on success, None if chain invocation fails.
    """
    start = time.time()

    # ── Session management ────────────────────────────────────────────────────
    session_id = get_or_create_session(user_id)

    # ── Log user's question ───────────────────────────────────────────────────
    user_msg = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=query,
    )
    log_chat_message(user_msg)

    # ── Run the RAG chain ─────────────────────────────────────────────────────
    try:
        chain = build_auditor_chain(qdrant_store, doc_type_filter)
        report: ComplianceReport = chain.invoke({"question": query})
    except Exception as e:
        print(f"[engine] Chain error: {e}")
        return None

    latency_ms = (time.time() - start) * 1000

    # ── Log assistant answer ──────────────────────────────────────────────────
    assistant_msg = ChatMessage(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=report.summary,
        verdict=report.verdict,
    )
    log_chat_message(assistant_msg)
    update_session_activity(session_id)

    # ── Write immutable audit record ──────────────────────────────────────────
    # This record is compliance evidence — never updated or deleted
    audit = AuditLogEntry(
        user_id=user_id,
        session_id=session_id,
        query=query,
        verdict=report.verdict,
        risk_level=report.risk_level,
        confidence=report.confidence_score,
        sources_used=report.citations,
        latency_ms=latency_ms,
    )
    log_audit_entry(audit)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Evaluation (run offline, not on the request hot path)
# ─────────────────────────────────────────────────────────────────────────────


def build_ragas_dataset(
    questions: List[str],
    qdrant_store: QdrantVectorStore,
) -> "Dataset":
    """
    What does this function do?
    Runs each evaluation question through the full RAG pipeline and
    collects: question, generated answer, retrieved contexts.
    Returns a HuggingFace Dataset in the exact format RAGAS expects.

    RAGAS column names (must not be renamed):
        question:  the user question
        answer:    the LLM-generated answer string
        contexts:  List[str] of retrieved chunks used to generate the answer

    Why run RAGAS offline?
    Evaluation makes many LLM calls (one per question per metric).
    This is expensive (~$0.01 per question for 3 metrics) and slow.
    We run it as a separate script, not on every user query.
    """
    from datasets import Dataset

    rows = []

    for i, question in enumerate(questions, 1):
        label = (
            f"[{i}/{len(questions)}] {question[:60]}..."
            if len(question) > 60
            else f"[{i}/{len(questions)}] {question}"
        )
        print(label)

        try:
            # Retrieve the contexts we would use to answer this question
            top_docs = retrieve_and_rerank(question, qdrant_store=qdrant_store)
            contexts = [doc.page_content for doc in top_docs]

            # Generate the answer using the full auditor chain
            chain = build_auditor_chain(qdrant_store)
            report: ComplianceReport = chain.invoke({"question": question})

            # Flatten the structured report into a single answer string
            answer = f"{report.verdict}: {report.summary} {report.detailed_analysis}"

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                }
            )

        except Exception as e:
            print(f"  Error: {e}")
            rows.append(
                {
                    "question": question,
                    "answer": "Error during generation",
                    "contexts": [],
                }
            )

    return Dataset.from_list(rows)


def run_faithfulness_evaluation(
    questions: List[str],
    qdrant_store: QdrantVectorStore,
) -> Dict[str, Any]:
    """
    What does this function do?
    Runs a subset of RAGAS metrics on the evaluation dataset and returns
    a summary dict with per-question scores and aggregate averages.

    Metrics used:
        Faithfulness:      Fraction of the answer supported by the retrieved context.
                           This directly measures hallucination risk.
        AnswerRelevancy:   How well the answer addresses the question.
        FactualCorrectness: Whether stated facts match the context.

    Why faithfulness first?
    For compliance use, hallucination is the HIGHEST-RISK failure mode.
    A system that invents regulations is worse than one that says "I don't know."
    Faithfulness directly measures whether the LLM is grounded in sources.

    Target thresholds (from notebook):
        Faithfulness ≥ 0.85: acceptable for compliance use
        AnswerRelevancy ≥ 0.80: acceptable for compliance use

    Cost estimate:
        ~$0.01 per question per metric
        15 questions × 3 metrics ≈ $0.45 total
    """
    from langchain_openai import ChatOpenAI as LCOpenAI
    from langchain_openai import OpenAIEmbeddings as LCEmbed
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import AnswerRelevancy, FactualCorrectness, Faithfulness

    # Wrap LangChain clients for RAGAS
    ragas_llm = LangchainLLMWrapper(LCOpenAI(model=settings.openai_model))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        LCEmbed(model=settings.embedding_model)
    )

    print("[engine] Building evaluation dataset...")
    dataset = build_ragas_dataset(questions, qdrant_store)

    print("\n[engine] Running RAGAS evaluation...")
    print(
        "[engine] (This calls OpenAI API for each question — budget ~$0.01 per question)"
    )

    results = ragas_evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            FactualCorrectness(llm=ragas_llm),
        ],
    )

    results_df = results.to_pandas()
    avg_faith = results_df["faithfulness"].mean()
    avg_relev = results_df["answer_relevancy"].mean()

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Questions evaluated:  {len(questions)}")
    print(f"  Faithfulness avg:     {avg_faith:.3f}  (target >= 0.85)")
    print(f"  Answer relevancy avg: {avg_relev:.3f}  (target >= 0.80)")

    status_f = "PASS" if avg_faith >= 0.85 else "FAIL — investigate retrieval"
    status_r = "PASS" if avg_relev >= 0.80 else "FAIL — check prompt or chunking"
    print(f"  Faithfulness status:  {status_f}")
    print(f"  Relevancy status:     {status_r}")
    print("=" * 60)

    if avg_faith < 0.85:
        print("\nLow faithfulness — investigation checklist:")
        print("  1. Are the right documents indexed? Run /ingest and verify.")
        print("  2. Is chunk_size too large? Reduce to 400 in .env and re-index.")
        print("  3. Is top_k_rerank too low? Increase from 5 to 8 in .env.")
        print(
            "  4. Check the compliance prompt for instructions that encourage extrapolation."
        )

    return {
        "faithfulness": avg_faith,
        "answer_relevancy": avg_relev,
        "details": results_df,
    }
