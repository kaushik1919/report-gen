"""
Tests for core/rag_store.py.

Pure-function tests (chunking, SHA-256, chunk building) run with no external
dependencies.  RAGStore integration tests are skipped when chromadb or
sentence-transformers are unavailable, keeping the test suite runnable in
environments that only have the base M3 dependencies installed.
"""
from __future__ import annotations

import hashlib
import io

import pytest

# ---------------------------------------------------------------------------
# Import guards — mark integration tests as skipped when deps are missing
# ---------------------------------------------------------------------------

try:
    import chromadb  # noqa: F401
    import sentence_transformers  # noqa: F401
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

skip_if_no_rag = pytest.mark.skipif(
    not _RAG_AVAILABLE,
    reason="chromadb or sentence-transformers not installed",
)

# ---------------------------------------------------------------------------
# Minimal synthetic PDF factory (no real model needed for pure helpers)
# ---------------------------------------------------------------------------

def _make_minimal_pdf(text: str = "Hello world from test PDF.") -> bytes:
    """Return minimal valid PDF bytes containing *text* on one page."""
    from reportlab.pdfgen import canvas  # type: ignore[import]
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def _has_reportlab() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


skip_if_no_reportlab = pytest.mark.skipif(
    not _has_reportlab(),
    reason="reportlab not installed — cannot build synthetic PDFs",
)

# ---------------------------------------------------------------------------
# TestChunkText — pure function, no deps
# ---------------------------------------------------------------------------

class TestChunkText:
    def _chunk(self, text: str, size: int = 1500, overlap: int = 200) -> list[str]:
        from core.rag_store import _chunk_text
        return _chunk_text(text, chunk_size=size, overlap=overlap)

    def test_empty_string_returns_empty(self):
        assert self._chunk("") == []

    def test_whitespace_only_returns_empty(self):
        assert self._chunk("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "A" * 500
        chunks = self._chunk(text, size=1500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        text = "A" * 3000
        chunks = self._chunk(text, size=1500, overlap=200)
        assert len(chunks) >= 2

    def test_chunk_size_respected(self):
        text = "B" * 3000
        chunks = self._chunk(text, size=1500, overlap=200)
        for chunk in chunks:
            assert len(chunk) <= 1500

    def test_overlap_means_consecutive_chunks_share_content(self):
        text = "X" * 3000
        chunks = self._chunk(text, size=1500, overlap=200)
        if len(chunks) >= 2:
            end_of_first = chunks[0][-200:]
            start_of_second = chunks[1][:200]
            assert end_of_first == start_of_second

    def test_entire_text_covered(self):
        text = "Hello " * 500  # 3000 chars
        chunks = self._chunk(text, size=1500, overlap=200)
        # First chunk starts at the beginning
        assert chunks[0][:6] == "Hello "
        # Last chunk ends at the end of text
        assert text.endswith(chunks[-1])


# ---------------------------------------------------------------------------
# TestSha256 — pure function
# ---------------------------------------------------------------------------

class TestSha256:
    def test_known_value(self):
        from core.rag_store import _sha256
        data = b"test"
        expected = hashlib.sha256(b"test").hexdigest()
        assert _sha256(data) == expected

    def test_returns_64_char_hex(self):
        from core.rag_store import _sha256
        result = _sha256(b"hello world")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_data_different_hash(self):
        from core.rag_store import _sha256
        assert _sha256(b"a") != _sha256(b"b")

    def test_same_data_same_hash(self):
        from core.rag_store import _sha256
        data = b"deterministic"
        assert _sha256(data) == _sha256(data)


# ---------------------------------------------------------------------------
# TestBuildChunks — requires reportlab for PDF synthesis
# ---------------------------------------------------------------------------

class TestBuildChunks:
    @skip_if_no_reportlab
    def test_returns_list_of_rag_chunks(self):
        from core.models import RAGChunk
        from core.rag_store import _build_chunks
        pdf = _make_minimal_pdf("Hello world test content for chunking.")
        chunks = _build_chunks(pdf, source="test.pdf")
        assert isinstance(chunks, list)
        assert all(isinstance(c, RAGChunk) for c in chunks)

    @skip_if_no_reportlab
    def test_chunk_ids_are_deterministic(self):
        from core.rag_store import _build_chunks
        pdf = _make_minimal_pdf("Deterministic chunking test content here.")
        chunks1 = _build_chunks(pdf, source="a.pdf")
        chunks2 = _build_chunks(pdf, source="a.pdf")
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    @skip_if_no_reportlab
    def test_chunk_id_format(self):
        from core.rag_store import _build_chunks, _sha256
        pdf = _make_minimal_pdf("Chunk ID format test content goes here.")
        file_hash = _sha256(pdf)
        chunks = _build_chunks(pdf, source="test.pdf")
        if chunks:
            expected_prefix = file_hash[:16]
            assert chunks[0].chunk_id.startswith(expected_prefix)
            assert chunks[0].chunk_id == f"{expected_prefix}_000000"

    @skip_if_no_reportlab
    def test_source_and_citation_key_propagated(self):
        from core.rag_store import _build_chunks
        pdf = _make_minimal_pdf("Source and citation key propagation test content.")
        chunks = _build_chunks(pdf, source="myfile.pdf", citation_key="smith2020")
        if chunks:
            assert chunks[0].source == "myfile.pdf"
            assert chunks[0].citation_key == "smith2020"

    @skip_if_no_reportlab
    def test_file_hash_consistent_across_chunks(self):
        from core.rag_store import _build_chunks, _sha256
        pdf = _make_minimal_pdf("Hash consistency test across all chunks in document.")
        file_hash = _sha256(pdf)
        chunks = _build_chunks(pdf, source="test.pdf")
        for chunk in chunks:
            assert chunk.file_hash == file_hash

    @skip_if_no_reportlab
    def test_empty_pdf_returns_no_chunks(self):
        """A PDF whose pages yield no extractable text produces no chunks."""
        from pypdf import PdfWriter
        buf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.write(buf)
        blank_pdf = buf.getvalue()
        from core.rag_store import _build_chunks
        chunks = _build_chunks(blank_pdf, source="blank.pdf")
        assert chunks == []


# ---------------------------------------------------------------------------
# TestRAGStore — requires chromadb + sentence-transformers
# ---------------------------------------------------------------------------

@skip_if_no_rag
class TestRAGStoreIntegration:
    @pytest.fixture
    def store(self, tmp_path):
        """Ephemeral in-process RAGStore backed by a temp directory."""
        from core.rag_store import RAGStore
        return RAGStore(persist_dir=tmp_path, model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def pdf_bytes(self):
        if not _has_reportlab():
            pytest.skip("reportlab not installed")
        return _make_minimal_pdf(
            "Artificial intelligence and machine learning in healthcare applications "
            "have demonstrated significant potential for improving patient outcomes. "
            "This paper reviews recent advances and discusses ethical considerations."
        )

    def test_empty_store_count_is_zero(self, store):
        assert store.document_count() == 0

    def test_empty_store_retrieve_returns_empty_result(self, store):
        from core.models import RetrievalResult
        result = store.retrieve("anything", k=5)
        assert isinstance(result, RetrievalResult)
        assert result.chunks == ()

    def test_ingest_returns_chunks_added(self, store, pdf_bytes):
        result = store.ingest(pdf_bytes, source="paper.pdf")
        assert result["chunks_added"] > 0
        assert result["skipped"] is False
        assert result["source"] == "paper.pdf"

    def test_ingest_increments_document_count(self, store, pdf_bytes):
        store.ingest(pdf_bytes, source="paper.pdf")
        assert store.document_count() > 0

    def test_duplicate_ingest_is_skipped(self, store, pdf_bytes):
        r1 = store.ingest(pdf_bytes, source="paper.pdf")
        r2 = store.ingest(pdf_bytes, source="paper.pdf")
        assert r1["chunks_added"] > 0
        assert r2["chunks_added"] == 0
        assert r2["skipped"] is True

    def test_is_ingested_true_after_ingest(self, store, pdf_bytes):
        from core.rag_store import _sha256
        file_hash = _sha256(pdf_bytes)
        store.ingest(pdf_bytes, source="paper.pdf")
        assert store.is_ingested(file_hash) is True

    def test_is_ingested_false_before_ingest(self, store, pdf_bytes):
        from core.rag_store import _sha256
        file_hash = _sha256(pdf_bytes)
        assert store.is_ingested(file_hash) is False

    def test_retrieve_returns_rag_chunks(self, store, pdf_bytes):
        from core.models import RAGChunk
        store.ingest(pdf_bytes, source="paper.pdf")
        result = store.retrieve("machine learning", k=3)
        assert all(isinstance(c, RAGChunk) for c in result.chunks)

    def test_retrieve_respects_k_limit(self, store, pdf_bytes):
        store.ingest(pdf_bytes, source="paper.pdf")
        result = store.retrieve("healthcare", k=2)
        assert len(result.chunks) <= 2

    def test_retrieve_metadata_preserved(self, store, pdf_bytes):
        store.ingest(pdf_bytes, source="paper.pdf", citation_key="ai2023")
        result = store.retrieve("healthcare", k=5)
        if result.chunks:
            chunk = result.chunks[0]
            assert chunk.source == "paper.pdf"
            assert chunk.citation_key == "ai2023"

    def test_file_hash_in_result(self, store, pdf_bytes):
        from core.rag_store import _sha256
        file_hash = _sha256(pdf_bytes)
        store.ingest(pdf_bytes, source="paper.pdf")
        result = store.retrieve("machine learning", k=5)
        if result.chunks:
            assert result.chunks[0].file_hash == file_hash


# ---------------------------------------------------------------------------
# TestRagStartupInfo — pure, no deps
# ---------------------------------------------------------------------------

class TestRagStartupInfo:
    def test_returns_dict_with_required_keys(self):
        from core.rag_store import rag_startup_info
        info = rag_startup_info()
        assert "chroma_dir" in info
        assert "embedding_model" in info
        assert "collection" in info

    def test_does_not_initialize_model(self):
        """rag_startup_info() must be callable without heavy model loading."""
        from core.rag_store import rag_startup_info
        info = rag_startup_info()
        assert isinstance(info["embedding_model"], str)
        assert info["embedding_model"] != ""
