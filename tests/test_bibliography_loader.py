"""
Tests for core/bibliography_loader.py.
All parsing is done in-memory via tmp_path — no network access required.
"""

import pytest

from core.bibliography_loader import BibliographyLoadError, load_bibtex
from core.models import BibliographyStore, CitationEntry

# ---------------------------------------------------------------------------
# Inline BibTeX fixtures
# ---------------------------------------------------------------------------

SAMPLE_BIB = """
@article{smith2020,
  author  = {Smith, John and Jones, Alice},
  title   = {A Study of Machine Learning in Healthcare},
  journal = {Journal of Medical AI},
  year    = {2020},
  volume  = {15},
  number  = {2},
  pages   = {34--45},
}

@book{doe2019,
  author    = {Doe, Jane},
  title     = {Foundations of Data Science},
  year      = {2019},
  publisher = {Academic Press},
}

@misc{brown2021,
  author = {Brown, Robert},
  title  = {Introduction to Neural Networks},
  year   = {2021},
}
"""

MALFORMED_BIB = """
@article{valid_entry,
  author = {Author, Valid},
  title  = {A Valid Entry After Junk},
  year   = {2022},
}

not valid bibtex content here >>>{{{{{{
"""


# ---------------------------------------------------------------------------
# TestLoadBibtex — happy path
# ---------------------------------------------------------------------------

class TestLoadBibtex:
    def test_returns_bibliography_store(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert isinstance(store, BibliographyStore)

    def test_loads_all_entries(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert len(store.entries) == 3

    def test_returns_correct_keys(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert "smith2020" in store.keys()
        assert "doe2019" in store.keys()
        assert "brown2021" in store.keys()

    def test_entry_type_preserved(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert store.get("smith2020").entry_type == "article"
        assert store.get("doe2019").entry_type == "book"
        assert store.get("brown2021").entry_type == "misc"

    def test_entry_fields_populated(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        entry = store.get("smith2020")
        assert "title" in entry.fields
        assert "author" in entry.fields
        assert "year" in entry.fields

    def test_source_path_recorded(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert str(bib_file) in store.source_path

    def test_entry_key_matches_bibtex_id(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        entry = store.get("smith2020")
        assert isinstance(entry, CitationEntry)
        assert entry.key == "smith2020"


# ---------------------------------------------------------------------------
# TestLoadBibtex — empty file
# ---------------------------------------------------------------------------

class TestEmptyBibtex:
    def test_empty_file_returns_empty_store(self, tmp_path):
        bib_file = tmp_path / "empty.bib"
        bib_file.write_text("", encoding="utf-8")
        store = load_bibtex(bib_file)
        assert store.keys() == []

    def test_empty_store_source_path_set(self, tmp_path):
        bib_file = tmp_path / "empty.bib"
        bib_file.write_text("", encoding="utf-8")
        store = load_bibtex(bib_file)
        assert str(bib_file) in store.source_path


# ---------------------------------------------------------------------------
# TestLoadBibtex — error paths
# ---------------------------------------------------------------------------

class TestLoadBibtexErrors:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(BibliographyLoadError, match="not found"):
            load_bibtex(tmp_path / "nonexistent.bib")

    def test_wrong_extension_raises(self, tmp_path):
        txt_file = tmp_path / "refs.txt"
        txt_file.write_text("not a bib file")
        with pytest.raises(BibliographyLoadError, match="Expected .bib"):
            load_bibtex(txt_file)

    def test_pdf_extension_raises(self, tmp_path):
        pdf_file = tmp_path / "refs.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(BibliographyLoadError):
            load_bibtex(pdf_file)


# ---------------------------------------------------------------------------
# TestLoadBibtex — malformed content
# ---------------------------------------------------------------------------

class TestMalformedBibtex:
    def test_partial_parse_still_returns_valid_entries(self, tmp_path):
        bib_file = tmp_path / "malformed.bib"
        bib_file.write_text(MALFORMED_BIB, encoding="utf-8")
        # Should not raise — valid entries are still returned
        store = load_bibtex(bib_file)
        assert store.has("valid_entry")

    def test_malformed_does_not_raise(self, tmp_path):
        bib_file = tmp_path / "malformed.bib"
        bib_file.write_text(MALFORMED_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert isinstance(store, BibliographyStore)


# ---------------------------------------------------------------------------
# TestBibliographyStore — helper methods
# ---------------------------------------------------------------------------

class TestBibliographyStore:
    def test_has_returns_true_for_known_key(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert store.has("smith2020") is True

    def test_has_returns_false_for_missing_key(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert store.has("nonexistent_key") is False

    def test_get_returns_none_for_missing_key(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        assert store.get("nonexistent_key") is None

    def test_keys_returns_list(self, tmp_path):
        bib_file = tmp_path / "sample.bib"
        bib_file.write_text(SAMPLE_BIB, encoding="utf-8")
        store = load_bibtex(bib_file)
        keys = store.keys()
        assert isinstance(keys, list)
        assert len(keys) == 3
