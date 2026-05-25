"""
Tests for template_loader and style_extractor.
All .docx fixtures are created programmatically in conftest.py — no binaries in git.
"""
from pathlib import Path

import pytest

from core.models import StyleSpec, TemplateProfile
from core.style_extractor import extract
from core.template_loader import TemplateLoadError, load

# ===========================================================================
# TemplateLoader tests
# ===========================================================================

class TestTemplateLoader:
    def test_load_returns_loaded_template(self, simple_template: Path):
        loaded = load(simple_template)
        assert loaded.path == str(simple_template)
        assert loaded.document is not None
        assert len(loaded.content_hash) == 64          # SHA-256 is 64 hex chars

    def test_content_hash_is_stable(self, simple_template: Path):
        hash1 = load(simple_template).content_hash
        hash2 = load(simple_template).content_hash
        assert hash1 == hash2

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(TemplateLoadError, match="File not found"):
            load(tmp_path / "ghost.docx")

    def test_wrong_extension_raises(self, tmp_path: Path):
        fake = tmp_path / "report.pdf"
        fake.write_text("not a docx")
        with pytest.raises(TemplateLoadError, match=".docx"):
            load(fake)

    def test_empty_file_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.docx"
        empty.write_bytes(b"")
        with pytest.raises(TemplateLoadError, match="empty"):
            load(empty)

    def test_corrupt_file_raises(self, tmp_path: Path):
        bad = tmp_path / "corrupt.docx"
        bad.write_bytes(b"PK\x03\x04this is not a real zip")
        with pytest.raises(TemplateLoadError):
            load(bad)


# ===========================================================================
# StyleExtractor — margins
# ===========================================================================

class TestMarginExtraction:
    def test_margins_match_template(self, simple_template: Path):
        profile = extract(load(simple_template))
        m = profile.margins_in
        assert m["top"] == pytest.approx(1.0, abs=0.01)
        assert m["bottom"] == pytest.approx(1.0, abs=0.01)
        assert m["left"] == pytest.approx(1.25, abs=0.01)
        assert m["right"] == pytest.approx(1.25, abs=0.01)

    def test_margins_present_in_minimal_docx(self, minimal_template: Path):
        profile = extract(load(minimal_template))
        assert set(profile.margins_in.keys()) == {"top", "bottom", "left", "right"}
        for v in profile.margins_in.values():
            assert isinstance(v, float)
            assert v > 0


# ===========================================================================
# StyleExtractor — page geometry
# ===========================================================================

class TestPageGeometry:
    def test_us_letter_size(self, simple_template: Path):
        profile = extract(load(simple_template))
        w, h = profile.page_size
        assert w == pytest.approx(8.5, abs=0.1)
        assert h == pytest.approx(11.0, abs=0.1)

    def test_page_size_is_tuple_of_floats(self, minimal_template: Path):
        profile = extract(load(minimal_template))
        assert isinstance(profile.page_size, tuple)
        assert len(profile.page_size) == 2
        assert all(isinstance(v, float) for v in profile.page_size)


# ===========================================================================
# StyleExtractor — heading hierarchy
# ===========================================================================

class TestHeadingHierarchy:
    def test_heading_levels_detected(self, simple_template: Path):
        profile = extract(load(simple_template))
        assert "Heading 1" in profile.heading_hierarchy
        assert "Heading 2" in profile.heading_hierarchy

    def test_heading_order_preserved(self, simple_template: Path):
        profile = extract(load(simple_template))
        h1_idx = profile.heading_hierarchy.index("Heading 1")
        h2_idx = profile.heading_hierarchy.index("Heading 2")
        assert h1_idx < h2_idx

    def test_no_headings_in_minimal(self, minimal_template: Path):
        profile = extract(load(minimal_template))
        assert profile.heading_hierarchy == []

    def test_no_duplicate_headings(self, simple_template: Path):
        profile = extract(load(simple_template))
        assert len(profile.heading_hierarchy) == len(set(profile.heading_hierarchy))


# ===========================================================================
# StyleExtractor — placeholder detection
# ===========================================================================

class TestPlaceholderDetection:
    def test_detects_all_placeholders(self, simple_template: Path):
        profile = extract(load(simple_template))
        assert "{{author}}" in profile.placeholders
        assert "{{date}}" in profile.placeholders
        assert "{{topic}}" in profile.placeholders
        assert "{{experiment}}" in profile.placeholders

    def test_no_placeholders_in_minimal(self, minimal_template: Path):
        profile = extract(load(minimal_template))
        assert profile.placeholders == []

    def test_placeholder_only_template(self, placeholder_only_template: Path):
        profile = extract(load(placeholder_only_template))
        assert "{{title}}" in profile.placeholders
        assert "{{author}}" in profile.placeholders
        assert "{{institution}}" in profile.placeholders

    def test_no_duplicate_placeholders(self, simple_template: Path):
        profile = extract(load(simple_template))
        assert len(profile.placeholders) == len(set(profile.placeholders))


# ===========================================================================
# StyleExtractor — section skeleton
# ===========================================================================

class TestSectionSkeleton:
    def test_skeleton_titles_present(self, simple_template: Path):
        profile = extract(load(simple_template))
        titles = [s["title"] for s in profile.section_skeleton]
        assert "Introduction" in titles
        assert "Background" in titles
        assert "Methodology" in titles
        assert "Results" in titles

    def test_skeleton_levels_correct(self, simple_template: Path):
        profile = extract(load(simple_template))
        by_title = {s["title"]: s for s in profile.section_skeleton}
        assert by_title["Introduction"]["level"] == 1
        assert by_title["Background"]["level"] == 2
        assert by_title["Methodology"]["level"] == 1
        assert by_title["Results"]["level"] == 2

    def test_empty_skeleton_for_minimal(self, minimal_template: Path):
        profile = extract(load(minimal_template))
        assert profile.section_skeleton == []


# ===========================================================================
# StyleExtractor — styles dict
# ===========================================================================

class TestStylesDict:
    def test_styles_is_dict_of_stylespecs(self, simple_template: Path):
        profile = extract(load(simple_template))
        assert isinstance(profile.styles, dict)
        assert len(profile.styles) > 0
        for key, spec in profile.styles.items():
            assert isinstance(key, str)
            assert isinstance(spec, StyleSpec)

    def test_style_fields_have_correct_types(self, simple_template: Path):
        profile = extract(load(simple_template))
        for spec in profile.styles.values():
            assert isinstance(spec.font_name, str)
            assert isinstance(spec.font_size_pt, float)
            assert isinstance(spec.bold, bool)
            assert isinstance(spec.italic, bool)
            assert spec.alignment in ("left", "center", "right", "justify")
            assert isinstance(spec.line_spacing, float)

    def test_returns_profile_dataclass(self, simple_template: Path):
        profile = extract(load(simple_template))
        assert isinstance(profile, TemplateProfile)
