"""
Local RAG store: PDF ingestion → chunking → embedding → ChromaDB retrieval.

Design constraints (M4B):
- No cloud vector DBs; ChromaDB PersistentClient only.
- No OCR; pypdf text layer only.
- No LangChain, no async pipelines.
- Retrieval is read-only and isolated from rendering.
- Deduplication via SHA-256 file hash — re-ingesting the same PDF is a no-op.
- Chunk IDs are deterministic: f"{file_hash[:16]}_{chunk_index:06d}"
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.config import CHROMA_DIR, EMBEDDING_MODEL
from core.models import RAGChunk, RetrievalResult

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1500    # characters (~512 tokens at ~3 chars/token)
_CHUNK_OVERLAP = 200  # characters (~64 tokens)
_COLLECTION_NAME = "report_gen_docs"

# ---------------------------------------------------------------------------
# Pure helpers (no I/O, fully testable)
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks of *chunk_size* characters."""
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _extract_pages(pdf_bytes: bytes) -> list[str]:
    """Return per-page text strings from *pdf_bytes* using pypdf."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [page.extract_text() or "" for page in reader.pages]


def _build_chunks(
    pdf_bytes: bytes,
    source: str,
    citation_key: str = "",
) -> list[RAGChunk]:
    """Extract text from PDF and produce a deterministic RAGChunk list."""
    file_hash = _sha256(pdf_bytes)
    pages = _extract_pages(pdf_bytes)

    all_chunks: list[RAGChunk] = []
    chunk_index = 0

    for page_idx, page_text in enumerate(pages):
        for piece in _chunk_text(page_text):
            if not piece.strip():
                continue
            chunk_id = f"{file_hash[:16]}_{chunk_index:06d}"
            all_chunks.append(RAGChunk(
                chunk_id=chunk_id,
                text=piece,
                source=source,
                page=page_idx,
                chunk_index=chunk_index,
                file_hash=file_hash,
                citation_key=citation_key,
            ))
            chunk_index += 1

    return all_chunks


# ---------------------------------------------------------------------------
# RAGStore
# ---------------------------------------------------------------------------


class RAGStore:
    """
    Wraps a ChromaDB collection + SentenceTransformer embeddings.

    Instantiation is deferred until first use via get_store().
    All public methods are synchronous.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR, model_name: str = EMBEDDING_MODEL) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
        )
        logger.info(
            "rag_store: initialized — persist_dir=%s model=%s indexed_chunks=%d",
            persist_dir,
            model_name,
            self.document_count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def is_ingested(self, file_hash: str) -> bool:
        """Return True if any chunk from this file hash is already stored."""
        results = self._col.get(where={"file_hash": file_hash}, limit=1)
        return bool(results["ids"])

    def ingest(
        self,
        pdf_bytes: bytes,
        source: str,
        citation_key: str = "",
    ) -> dict:
        """
        Ingest a PDF into the store.

        Returns a dict: source, file_hash, chunks_added, skipped.
        If already ingested (same SHA-256), returns skipped=True immediately.
        """
        file_hash = _sha256(pdf_bytes)

        if self.is_ingested(file_hash):
            logger.info(
                "rag_store: skipping already-ingested file %r (hash %s)",
                source,
                file_hash[:16],
            )
            return {"source": source, "file_hash": file_hash, "chunks_added": 0, "skipped": True}

        chunks = _build_chunks(pdf_bytes, source, citation_key)
        if not chunks:
            logger.warning("rag_store: no text extracted from %r", source)
            return {"source": source, "file_hash": file_hash, "chunks_added": 0, "skipped": False}

        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "file_hash": c.file_hash,
                    "citation_key": c.citation_key,
                }
                for c in chunks
            ],
        )
        logger.info("rag_store: ingested %d chunks from %r", len(chunks), source)
        return {"source": source, "file_hash": file_hash, "chunks_added": len(chunks), "skipped": False}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Return the top-*k* chunks most relevant to *query*."""
        if self._col.count() == 0:
            return RetrievalResult(chunks=(), query=query, k=k)

        k_actual = min(k, self._col.count())
        results = self._col.query(query_texts=[query], n_results=k_actual)

        chunks: list[RAGChunk] = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            chunk_id = results["ids"][0][i]
            chunks.append(RAGChunk(
                chunk_id=chunk_id,
                text=doc,
                source=meta.get("source", ""),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                file_hash=meta.get("file_hash", ""),
                citation_key=meta.get("citation_key", ""),
            ))

        return RetrievalResult(chunks=tuple(chunks), query=query, k=k)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def document_count(self) -> int:
        """Return total number of indexed chunks."""
        return self._col.count()


# ---------------------------------------------------------------------------
# Lazy module-level singleton
# ---------------------------------------------------------------------------

_store: RAGStore | None = None


def get_store() -> RAGStore:
    """Return the module-level RAGStore singleton, initializing on first call."""
    global _store
    if _store is None:
        _store = RAGStore()
    return _store


def rag_startup_info() -> dict:
    """
    Return diagnostic info suitable for logging at application startup.

    Does NOT initialize the embedding model — just reads config constants.
    """
    return {
        "chroma_dir": str(CHROMA_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "collection": _COLLECTION_NAME,
    }
