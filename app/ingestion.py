"""
ingestion.py — Document Ingestion Pipeline
===========================================

This module handles the complete journey from a PDF file on disk to a
set of chunks indexed in Qdrant. It runs only when new documents need
to be indexed — it is NOT on the hot path of user queries.

Pipeline stages:
    1. scan_compliance_docs()         → find PDF files, build DocumentRecords
    2. extract_all_documents()        → PDF → Markdown pages (pymupdf4llm)
    3. chunk_documents()              → pages → overlapping text chunks
    (The vector_store module handles stage 4: chunks → Qdrant)

Key design decisions from the notebook:
    - DocumentRecord is a @dataclass (not Pydantic) because it is built
      entirely by our own code from the file system. No external validation
      needed. Pydantic would add overhead without benefit here.
    - DocMetadata (in schemas.py) IS Pydantic because it validates the
      metadata we attach to every Qdrant vector — wrong types there would
      corrupt the vector store silently.
    - make_doc_id() uses sha256(filepath + mtime) so an updated file
      automatically gets a new ID and is re-indexed.
"""

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.schemas import DocMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Data Container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DocumentRecord:
    """
    What is this class for?
    Represents one PDF file discovered during the folder scan.
    Holds everything we know about the file BEFORE we extract its content.
    Fields are populated progressively:
        - scanner    fills: file_path, filename, doc_id, doc_type, year, file_size_kb
        - extractor  fills: page_count
        - indexer    fills: chunk_count

    Why @dataclass and not Pydantic BaseModel here?
    This object is created entirely by our own code from the filesystem.
    No external input validation is needed. @dataclass is lighter and
    sufficient when we control all construction paths.
    """

    file_path: Path
    filename: str
    doc_id: str
    doc_type: str = "regulatory_document"
    category: str = "compliance"
    jurisdiction: str = "unknown"
    year: Optional[int] = None
    file_size_kb: float = 0.0
    page_count: int = 0
    chunk_count: int = 0
    already_indexed: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ID Generation
# ─────────────────────────────────────────────────────────────────────────────


def make_doc_id(path: Path) -> str:
    """
    What does this function do?
    Creates a deterministic, collision-resistant 16-character ID for a file.
    Includes the modification time so that an updated file gets a new ID
    and is automatically re-indexed without manual intervention.

    Why 16 chars of sha256?
    sha256 produces 64 hex chars = 256 bits of entropy.
    We take the first 16 chars = 64 bits of entropy.
    Probability of a collision across 1 million documents: ~1 in 10^15.
    Effectively zero for a compliance document library.

    Why filepath + mtime?
    - filepath alone: renaming a file would not trigger re-indexing.
    - mtime alone: two files with the same mtime would collide.
    - Both together: unique per file AND per version of that file.
    """
    stat = path.stat()
    # We hash the absolute path so relative path changes don't matter
    raw = f"{path.absolute()}:{stat.st_mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Inference
# ─────────────────────────────────────────────────────────────────────────────


def infer_doc_type(filename: str) -> str:
    """
    What does this function do?
    Maps filename keywords to a standardised document type label.
    Used in Qdrant payload for filtered retrieval and in Supabase for reporting.

    Why infer from filename rather than content?
    Fast: no need to open the PDF. Good enough for ~90% of cases.
    The enrich_metadata_from_content() function can refine this
    after the PDF is extracted using actual page text.

    Why lowercase and replace - and _ with spaces?
    Filenames like "COMPANIES-REGULATIONS-2021" become "companies regulations 2021"
    which matches the keyword "regulation" cleanly without a regex.
    """
    name = filename.lower().replace("-", " ").replace("_", " ")

    type_map = {
        "circular": ["circular"],
        "guideline": ["guideline", "guidance"],
        "framework": ["framework"],
        "regulation": ["regulation", "regulatory", "rule"],
        "legislation": ["act", "decree", "gazette", "statute"],
        "policy": ["policy"],
        "directive": ["directive"],
        "standard": ["standard", "ifrs", "gaap", "ias"],
        "contract": ["contract", "agreement", "vendor", "nda"],
        "report": ["report", "assessment", "audit"],
        "notice": ["notice", "bulletin", "advisory"],
    }

    for dtype, keywords in type_map.items():
        if any(kw in name for kw in keywords):
            return dtype

    return "regulatory_document"  # Safe default for anything unrecognised


def infer_year(filename: str) -> Optional[int]:
    """
    What does this function do?
    Extracts a 4-digit year (2000-2039) from the filename string.
    Returns the LAST year found if multiple are present
    (e.g. "CBN-Circular-2019-2024.pdf" → 2024, the more recent one).

    Why regex and not split?
    Filenames are inconsistent. Some use hyphens, some spaces, some underscores.
    \\b(20[0-3][0-9])\\b matches a standalone 4-digit year between word boundaries.
    """
    hits = re.findall(r"\b(20[0-3][0-9])\b", filename)
    return int(hits[-1]) if hits else None


def enrich_metadata_from_content(text: str, record: "DocumentRecord") -> dict:
    """
    What does this function do?
    Extracts additional metadata from the first ~1000 characters of a
    document's content and merges it with what we inferred from the filename.

    Why enrich from content rather than just using the filename?
    Filenames are user-controlled and inconsistent.
    The document's first page is much more reliable — it usually contains
    the official title, reference number, and issue date.

    What does this function return?
    A dict of metadata fields ready to be passed to DocMetadata().
    Fields: doc_id, filename, file_path, doc_type, category, jurisdiction, year
    Optional: reference_number (if found in first 1000 chars)

    Why only first 1000 chars?
    Metadata almost always appears on the first page.
    Processing the entire document for metadata wastes significant time.
    """
    meta = {
        "doc_id": record.doc_id,
        "filename": record.filename,
        "file_path": str(record.file_path),
        "doc_type": record.doc_type,
        "category": record.category,
        "jurisdiction": record.jurisdiction,
        "year": record.year,
    }

    # Work with the first 1000 characters, lowercase, for keyword matching
    first_1k = text[:1000].lower()

    # Content-based doc_type refinement — more reliable than filename keywords
    # because the document itself uses the official terminology
    type_refinements = {
        "circular": ["hereby circular", "this circular", "is hereby"],
        "guideline": ["guidelines for", "guidance on", "guidance note"],
        "framework": ["this framework", "the framework for"],
        "policy": ["this policy", "policy statement"],
        "standard": ["this standard", "international standard", "ifrs", "ias"],
        "directive": ["directive", "hereby directs"],
        "regulation": ["pursuant to", "in accordance with", "section", "article"],
        "contract": ["whereas", "in consideration of", "the parties agree"],
    }

    for dtype, patterns in type_refinements.items():
        if any(p in first_1k for p in patterns):
            meta["doc_type"] = dtype
            break  # Stop at first match — most specific match wins

    # Extract official reference numbers like CBN/DIR/GEN/2024/001
    # The regex catches common Nigerian regulatory reference formats
    ref_pattern = re.search(
        r"(?:ref(?:erence)?[:\s]+|no\.?\s*)([A-Z]{2,8}[/-][\w/-]{3,30})",
        text[:1000],
        re.IGNORECASE,
    )
    if ref_pattern:
        meta["reference_number"] = ref_pattern.group(1)

    # Refine year from content if filename did not yield one
    # We look in the first 2000 chars because some documents have a long
    # title page before the issue date appears
    if not meta["year"]:
        year_match = re.search(r"\b(20[0-3][0-9])\b", text[:2000])
        if year_match:
            meta["year"] = int(year_match.group(1))

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency Check
# ─────────────────────────────────────────────────────────────────────────────


def is_already_indexed(doc_id: str) -> bool:
    """
    What does this function do?
    Checks Qdrant for any existing vector point whose payload contains
    this exact doc_id.
    Returns True if found — meaning this document version is already indexed.

    Why check Qdrant and not a local file?
    The vector store IS the source of truth. If Qdrant is reset or the
    collection is deleted, this function correctly returns False and
    triggers re-indexing automatically.

    Why O(log n) and not O(n)?
    We created a keyword payload index on 'doc_id' in vector_store.py.
    Qdrant uses that index for this filter — no full collection scan needed.
    At 50,000+ vectors this matters significantly.
    """
    # Avoid circular import: import here since vector_store imports ingestion
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

        results, _ = client.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="doc_id", match=qm.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=1,             # We only need to know IF it exists, not how many
            with_payload=False,  # Don't fetch the full payload — faster
            with_vectors=False,  # Don't fetch the vector values — faster
        )

        return len(results) > 0

    except Exception:
        # If we can't check, assume it's NOT indexed so we don't skip it
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Directory Scanner
# ─────────────────────────────────────────────────────────────────────────────


def scan_compliance_docs(docs_dir: Optional[str] = None) -> List[DocumentRecord]:
    """
    What does this function do?
    Scans the compliance documents directory recursively, finds all PDFs,
    and returns one DocumentRecord per file.

    Production features:
        1. rglob('*.pdf'): recursive — finds PDFs in subfolders too
        2. Case-insensitive: finds .pdf and .PDF
        3. set() deduplication: handles symlinked duplicates
        4. Idempotency check: is_already_indexed() prevents double-indexing
        5. Metadata inference: doc_type and year from filename keywords
        6. Auto-creates folder: if the folder doesn't exist, creates it
           and returns an empty list (clean error message, not crash)

    Parameters:
        docs_dir: Override the default from settings.
                  Used when the /ingest endpoint scans an uploaded file's
                  temporary directory.
    """
    scan_path = Path(docs_dir or settings.compliance_docs_dir)

    # Create the folder if it doesn't exist yet
    if not scan_path.exists():
        scan_path.mkdir(parents=True, exist_ok=True)
        print(f"[ingestion] Created: {scan_path.absolute()}")
        print("[ingestion] Add your compliance PDFs to this folder then re-run.")
        return []

    # Find all PDFs, both .pdf and .PDF (Windows files often use uppercase)
    # set() removes duplicates that could arise from symlinks
    pdf_files = sorted(
        set(list(scan_path.rglob("*.pdf")) + list(scan_path.rglob("*.PDF")))
    )

    if not pdf_files:
        print(f"[ingestion] No PDFs found in {scan_path.absolute()}")
        return []

    records = []
    for path in pdf_files:
        stat = path.stat()
        doc_id = make_doc_id(path)

        records.append(
            DocumentRecord(
                file_path=path,
                filename=path.name,
                doc_id=doc_id,
                doc_type=infer_doc_type(path.name),
                year=infer_year(path.name),
                file_size_kb=round(stat.st_size / 1024, 1),
                # Idempotency: skip files we've already indexed
                already_indexed=is_already_indexed(doc_id),
            )
        )

    return records


# ─────────────────────────────────────────────────────────────────────────────
# PDF Extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_pdf(record: DocumentRecord) -> List[Document]:
    """
    What does this function do?
    Converts one PDF file into a list of LangChain Document objects —
    one per non-empty page — with validated DocMetadata attached.

    Why LangChain Document?
    Document(page_content, metadata) is the standard unit throughout LangChain.
    The text splitter accepts List[Document].
    The vector store accepts List[Document].
    Using this type keeps the pipeline consistent end-to-end.

    Why pymupdf4llm instead of pypdf or pdfplumber?
    pymupdf4llm converts PDF pages to clean Markdown.
    This preserves:
        - Bold headings   → **heading**
        - Tables          → Markdown table syntax
        - Bullet points   → - item
        - Numbered lists  → 1. item
    Clean Markdown dramatically improves chunk quality for legal text
    because it keeps table rows together and preserves heading hierarchy.

    Why skip pages with < 80 chars?
    Table-of-contents pages, blank pages, and copyright-only pages add
    noise to the vector store without contributing searchable content.
    80 chars is approximately 10–15 words — the minimum for a useful chunk.
    """
    if record.already_indexed:
        print(f"[ingestion] Skipping (already indexed): {record.filename}")
        return []

    print(f"[ingestion] Extracting: {record.filename}...")
    start = time.time()

    try:
        import pymupdf4llm

        # page_chunks=True returns a list of dicts, one per page.
        # Each dict has: 'text' (Markdown string), 'metadata' (page info)
        # show_progress=False keeps output clean in production logs
        page_data = pymupdf4llm.to_markdown(
            str(record.file_path),
            page_chunks=True,
            show_progress=False,
        )

    except Exception as e:
        print(f"[ingestion] ERROR extracting {record.filename}: {e}")
        return []

    # Update the record's page count now that we've loaded the PDF
    record.page_count = len(page_data)

    # Extract metadata from the first page's content (more reliable than filename)
    first_page_text = page_data[0].get("text", "") if page_data else ""
    base_meta = enrich_metadata_from_content(first_page_text, record)

    documents = []
    for idx, page in enumerate(page_data):
        text = page.get("text", "").strip()

        # Skip near-empty pages: covers, blank pages, ToC with only page numbers
        if len(text) < 80:
            continue

        # Validate the metadata with Pydantic before storing it.
        # This catches type mismatches (e.g. page="3" instead of page=3)
        # at ingestion time rather than silently corrupting Qdrant.
        doc_meta = DocMetadata(
            filename=record.filename,
            file_path=str(record.file_path),
            doc_id=record.doc_id,
            page=idx + 1,              # 1-indexed (not 0-indexed)
            total_pages=record.page_count,
            # Spread the enriched metadata fields, excluding the three
            # we've already set explicitly above to avoid key conflicts
            **{
                k: v
                for k, v in base_meta.items()
                if k not in ("filename", "file_path", "doc_id")
            },
        )

        documents.append(
            Document(
                page_content=text,
                # .model_dump() converts the Pydantic object to a plain dict.
                # LangChain requires metadata to be a dict, not a Pydantic model.
                metadata=doc_meta.model_dump(),
            )
        )

    elapsed = time.time() - start
    print(f"[ingestion]   {len(documents)} pages extracted in {elapsed:.1f}s")
    return documents


def extract_all_documents(records: List[DocumentRecord]) -> List[Document]:
    """
    What does this function do?
    Runs extract_pdf() on every NEW (not-yet-indexed) DocumentRecord.
    Skips already-indexed files automatically.
    Returns the combined flat list of all pages as LangChain Documents.

    This is the entry point called by the /ingest endpoint in main.py.
    It orchestrates extraction over all pending documents and reports progress.
    """
    all_pages: List[Document] = []
    new_records = [r for r in records if not r.already_indexed]

    if not new_records:
        print("[ingestion] No new documents to extract. All files already indexed.")
        return []

    print(f"[ingestion] Extracting {len(new_records)} document(s):")

    for record in new_records:
        pages = extract_pdf(record)
        all_pages.extend(pages)

    print(
        f"\n[ingestion] Total: {len(all_pages)} pages from {len(new_records)} document(s)"
    )

    return all_pages


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────


def create_splitter() -> RecursiveCharacterTextSplitter:
    """
    What does this function return?
    A RecursiveCharacterTextSplitter configured for compliance legal text.

    Why RecursiveCharacterTextSplitter?
    It tries separators in order of strength, falling back to finer splits
    only when necessary. This keeps semantically cohesive units together.

    Separator hierarchy (strongest to weakest):
        '\\n\\n'  paragraph break  — strongest semantic boundary in legal text
        '\\n'    line break
        '. '     sentence end      (note trailing space: avoids splitting 'e.g.')
        '; '     semicolon         — common in legal enumerations
        ', '     comma             — phrase boundary
        ' '      word boundary     — last natural boundary
        ''       character         — absolute last resort (splits mid-word)

    Why add_start_index=True?
    Adds 'start_index' to each chunk's metadata: the character offset
    of the chunk within its parent document. Useful for debugging and
    for implementing "highlight in source" features in the UI.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )


def chunk_documents(pages: List[Document]) -> List[Document]:
    """
    What does this function do?
    Splits a list of page-level Documents into chunk-level Documents.

    Each resulting chunk:
        1. Inherits ALL metadata from its parent page
           (doc_id, filename, page, doc_type, jurisdiction, year…)
        2. Gets a unique chunk_id: "{doc_id}_{global_index}"
        3. Gets a chunk_index: sequential integer (for ordering)
        4. Gets a char_count: useful for debugging oversized chunks

    Why inherit metadata?
    At query time, when a chunk is retrieved from Qdrant, we need to know
    WHICH document and WHICH page it came from to generate the citation
    in the compliance report (e.g. "[Source 1: NDPR.pdf | Page 12]").
    That information must travel with the chunk through the entire pipeline.

    Why global_idx instead of per-document?
    chunk_id must be unique across the ENTIRE vector store, not just per document.
    "{doc_id}_{global_idx}" guarantees uniqueness even when two documents
    produce the same number of chunks.

    Why skip chunks with < 40 chars?
    These are fragments from badly split sentences — typically a stray
    punctuation mark or a partial word. They add noise without signal.
    40 chars ≈ 6 words — the minimum for a semantically useful chunk.
    """
    if not pages:
        return []

    splitter = create_splitter()
    all_chunks: List[Document] = []
    global_idx = 0

    for page_doc in pages:
        # split_documents returns a new list of Documents with inherited metadata
        page_chunks = splitter.split_documents([page_doc])

        for chunk in page_chunks:
            text = chunk.page_content.strip()

            # Skip fragments too short to be useful
            if len(text) < 40:
                continue

            # Add chunk-specific fields to the inherited metadata
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata.get('doc_id', 'unknown')}_{global_idx}"
            )
            chunk.metadata["chunk_index"] = global_idx
            chunk.metadata["char_count"] = len(text)

            all_chunks.append(chunk)
            global_idx += 1

    return all_chunks
