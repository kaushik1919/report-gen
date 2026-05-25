"""
Tests for app/api.py endpoints.
All LLM calls, assembler, and exporter are mocked.
No Ollama daemon required.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.api import app
from core.content_generator import ContentGeneratorError
from core.exporter import ExporterError
from core.models import ReportPlan, SectionContent, SectionSpec, StyleSpec, TemplateProfile
from core.outline_planner import OutlinePlannerError
from core.template_loader import TemplateLoadError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_profile() -> TemplateProfile:
    style = StyleSpec(
        name="Normal", font_name="Calibri", font_size_pt=11.0,
        bold=False, italic=False, color_hex=None, alignment="left", line_spacing=1.0,
    )
    return TemplateProfile(
        styles={"Normal": style},
        margins_in={"top": 1.0, "bottom": 1.0, "left": 1.25, "right": 1.25},
        page_size=(8.5, 11.0),
        heading_hierarchy=["Heading 1", "Heading 2"],
        placeholders=["{{title}}"],
        section_skeleton=[],
    )


@pytest.fixture
def mock_plan() -> ReportPlan:
    return ReportPlan(
        title="Test Report",
        author="Jane Smith",
        sections=[
            SectionSpec(
                id="s1", title="Introduction", level=1,
                target_words=300, instructions="Write an intro.",
            )
        ],
    )


@pytest.fixture
def mock_section() -> SectionContent:
    return SectionContent(
        section_id="s1",
        title="Introduction",
        level=1,
        blocks=[{"type": "paragraph", "text": "Hello world."}],
        citations=[],
    )


@pytest.fixture
def docx_bytes(tmp_path) -> bytes:
    """Minimal valid DOCX bytes for upload tests."""
    p = tmp_path / "upload.docx"
    Document().save(str(p))
    return p.read_bytes()


@pytest.fixture
def uploads_dir(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def outputs_dir(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# POST /template/extract
# ---------------------------------------------------------------------------

class TestExtractTemplate:
    def test_happy_path(self, client, docx_bytes, mock_profile, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)

        with patch("app.api.template_loader.load", return_value=MagicMock()), \
             patch("app.api.style_extractor.extract", return_value=mock_profile):
            resp = client.post(
                "/template/extract",
                files={"file": ("report.docx", docx_bytes, "application/octet-stream")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "template_id" in body
        assert "profile" in body
        assert "styles" in body["profile"]

    def test_non_docx_rejected(self, client):
        resp = client.post(
            "/template/extract",
            files={"file": ("report.pdf", b"fake", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "docx" in resp.json()["detail"].lower()

    def test_load_error_returns_400(self, client, docx_bytes, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)

        with patch("app.api.template_loader.load", side_effect=TemplateLoadError("corrupt file")):
            resp = client.post(
                "/template/extract",
                files={"file": ("report.docx", docx_bytes, "application/octet-stream")},
            )

        assert resp.status_code == 400
        assert "corrupt file" in resp.json()["detail"]

    def test_saved_file_cleaned_up_on_load_error(self, client, docx_bytes, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)

        with patch("app.api.template_loader.load", side_effect=TemplateLoadError("bad")):
            client.post(
                "/template/extract",
                files={"file": ("report.docx", docx_bytes, "application/octet-stream")},
            )

        assert list(uploads_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# POST /report/plan
# ---------------------------------------------------------------------------

class TestGeneratePlan:
    def _make_template(self, uploads_dir: Path) -> str:
        template_id = "test-template-id"
        (uploads_dir / f"{template_id}.docx").write_bytes(b"stub")
        return template_id

    def test_happy_path(self, client, mock_profile, mock_plan, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)
        template_id = self._make_template(uploads_dir)

        with patch("app.api.template_loader.load", return_value=MagicMock()), \
             patch("app.api.style_extractor.extract", return_value=mock_profile), \
             patch("app.api.outline_planner.plan", return_value=mock_plan):
            resp = client.post("/report/plan", json={
                "template_id": template_id,
                "brief": "Write about climate.",
                "topic": "Climate Change",
                "academic_level": "undergraduate",
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["plan"]["title"] == "Test Report"
        assert len(body["plan"]["sections"]) == 1

    def test_template_not_found_returns_404(self, client, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)
        resp = client.post("/report/plan", json={
            "template_id": "nonexistent",
            "brief": "x", "topic": "x", "academic_level": "x",
        })
        assert resp.status_code == 404

    def test_planner_error_returns_502(self, client, mock_profile, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)
        template_id = self._make_template(uploads_dir)

        with patch("app.api.template_loader.load", return_value=MagicMock()), \
             patch("app.api.style_extractor.extract", return_value=mock_profile), \
             patch("app.api.outline_planner.plan",
                   side_effect=OutlinePlannerError("LLM timeout")):
            resp = client.post("/report/plan", json={
                "template_id": template_id,
                "brief": "x", "topic": "x", "academic_level": "x",
            })

        assert resp.status_code == 502
        assert "LLM timeout" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /report/content
# ---------------------------------------------------------------------------

class TestGenerateContent:
    def _plan_dict(self) -> dict:
        return {
            "title": "Test Report",
            "author": "Jane Smith",
            "sections": [
                {"id": "s1", "title": "Intro", "level": 1,
                 "target_words": 300, "instructions": "Write intro."},
            ],
        }

    def test_happy_path(self, client, mock_section):
        with patch("app.api.content_generator.write_section", return_value=mock_section), \
             patch("app.api.content_generator.summarize_section", return_value="Summary text"):
            resp = client.post("/report/content", json={
                "plan": self._plan_dict(),
                "topic": "Climate Change",
            })

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["sections"]) == 1
        assert body["sections"][0]["title"] == "Introduction"

    def test_invalid_plan_returns_422(self, client):
        resp = client.post("/report/content", json={
            "plan": {"missing_title": True},
            "topic": "x",
        })
        assert resp.status_code == 422

    def test_content_error_returns_502(self, client):
        with patch("app.api.content_generator.write_section",
                   side_effect=ContentGeneratorError("Ollama down")):
            resp = client.post("/report/content", json={
                "plan": self._plan_dict(),
                "topic": "x",
            })

        assert resp.status_code == 502
        assert "Ollama down" in resp.json()["detail"]

    def test_multiple_sections_all_generated(self, client, mock_section):
        plan = {
            "title": "T", "author": "A",
            "sections": [
                {"id": "s1", "title": "Intro", "level": 1,
                 "target_words": 200, "instructions": "x"},
                {"id": "s2", "title": "Body", "level": 1,
                 "target_words": 400, "instructions": "y"},
            ],
        }
        with patch("app.api.content_generator.write_section", return_value=mock_section), \
             patch("app.api.content_generator.summarize_section", return_value=""):
            resp = client.post("/report/content", json={"plan": plan, "topic": "x"})

        assert resp.status_code == 200
        assert len(resp.json()["sections"]) == 2


# ---------------------------------------------------------------------------
# POST /report/build
# ---------------------------------------------------------------------------

class TestBuildReport:
    def _payload(self, template_id: str) -> dict:
        return {
            "template_id": template_id,
            "plan": {
                "title": "Test Report", "author": "Jane",
                "sections": [
                    {"id": "s1", "title": "Intro", "level": 1,
                     "target_words": 200, "instructions": "x"},
                ],
            },
            "sections": [
                {"section_id": "s1", "title": "Intro", "level": 1,
                 "blocks": [{"type": "paragraph", "text": "Hello."}], "citations": []},
            ],
        }

    def test_happy_path_returns_docx(
        self, client, mock_profile, uploads_dir, outputs_dir, monkeypatch
    ):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)
        monkeypatch.setattr("app.api.OUTPUTS_DIR", outputs_dir)
        template_id = "tpl-001"
        (uploads_dir / f"{template_id}.docx").write_bytes(b"stub")
        out_docx = outputs_dir / "test_report.docx"
        out_docx.write_bytes(b"docx-content")

        with patch("app.api.template_loader.load", return_value=MagicMock()), \
             patch("app.api.style_extractor.extract", return_value=mock_profile), \
             patch("app.api.document_assembler.build", return_value=out_docx):
            resp = client.post("/report/build", json=self._payload(template_id))

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_template_not_found_returns_404(self, client, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)
        resp = client.post("/report/build", json=self._payload("missing-id"))
        assert resp.status_code == 404

    def test_invalid_sections_returns_422(self, client, mock_profile, uploads_dir, monkeypatch):
        monkeypatch.setattr("app.api.UPLOADS_DIR", uploads_dir)
        template_id = "tpl-002"
        (uploads_dir / f"{template_id}.docx").write_bytes(b"stub")
        payload = self._payload(template_id)
        payload["sections"] = [{"bad": "data"}]  # missing required keys

        with patch("app.api.template_loader.load", return_value=MagicMock()), \
             patch("app.api.style_extractor.extract", return_value=mock_profile):
            resp = client.post("/report/build", json=payload)

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /report/export-pdf
# ---------------------------------------------------------------------------

class TestExportPdf:
    def test_happy_path_returns_pdf(self, client, outputs_dir, monkeypatch):
        monkeypatch.setattr("app.api.OUTPUTS_DIR", outputs_dir)
        docx_file = outputs_dir / "report.docx"
        docx_file.write_bytes(b"stub")
        pdf_file = outputs_dir / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 stub")

        with patch("app.api.exporter.to_pdf", return_value=pdf_file):
            resp = client.post("/report/export-pdf", json={"docx_filename": "report.docx"})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"

    def test_docx_not_found_returns_404(self, client, outputs_dir, monkeypatch):
        monkeypatch.setattr("app.api.OUTPUTS_DIR", outputs_dir)
        resp = client.post("/report/export-pdf", json={"docx_filename": "missing.docx"})
        assert resp.status_code == 404

    def test_exporter_error_returns_502(self, client, outputs_dir, monkeypatch):
        monkeypatch.setattr("app.api.OUTPUTS_DIR", outputs_dir)
        docx_file = outputs_dir / "report.docx"
        docx_file.write_bytes(b"stub")

        with patch("app.api.exporter.to_pdf",
                   side_effect=ExporterError("LibreOffice not found")):
            resp = client.post("/report/export-pdf", json={"docx_filename": "report.docx"})

        assert resp.status_code == 502
        assert "LibreOffice not found" in resp.json()["detail"]

    def test_custom_libreoffice_bin_forwarded(self, client, outputs_dir, monkeypatch):
        monkeypatch.setattr("app.api.OUTPUTS_DIR", outputs_dir)
        docx_file = outputs_dir / "report.docx"
        docx_file.write_bytes(b"stub")
        pdf_file = outputs_dir / "report.pdf"
        pdf_file.write_bytes(b"%PDF stub")

        with patch("app.api.exporter.to_pdf", return_value=pdf_file) as mock_pdf:
            client.post("/report/export-pdf", json={
                "docx_filename": "report.docx",
                "libreoffice_bin": "/opt/lo/soffice",
            })

        _, kwargs = mock_pdf.call_args
        assert kwargs.get("libreoffice_bin") == "/opt/lo/soffice"
