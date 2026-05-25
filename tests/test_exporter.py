"""
Tests for core/exporter.to_pdf().
LibreOffice is never invoked — subprocess.run is mocked throughout.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.exporter import ExporterError, to_pdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestCommandConstruction:
    def test_explicit_bin_used(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        pdf = tmp_path / "report.pdf"
        pdf.touch()

        with patch("subprocess.run", return_value=_make_result()) as mock_run:
            to_pdf(docx, libreoffice_bin="/custom/soffice")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/custom/soffice"

    def test_headless_flag_present(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        (tmp_path / "report.pdf").touch()

        with patch("subprocess.run", return_value=_make_result()) as mock_run:
            to_pdf(docx, libreoffice_bin="soffice")

        cmd = mock_run.call_args[0][0]
        assert "--headless" in cmd

    def test_convert_to_pdf_flags(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        (tmp_path / "report.pdf").touch()

        with patch("subprocess.run", return_value=_make_result()) as mock_run:
            to_pdf(docx, libreoffice_bin="soffice")

        cmd = mock_run.call_args[0][0]
        assert "--convert-to" in cmd
        assert "pdf" in cmd

    def test_outdir_is_docx_parent(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        (tmp_path / "report.pdf").touch()

        with patch("subprocess.run", return_value=_make_result()) as mock_run:
            to_pdf(docx, libreoffice_bin="soffice")

        cmd = mock_run.call_args[0][0]
        outdir_idx = cmd.index("--outdir") + 1
        assert cmd[outdir_idx] == str(tmp_path)

    def test_docx_path_is_last_arg(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        (tmp_path / "report.pdf").touch()

        with patch("subprocess.run", return_value=_make_result()) as mock_run:
            to_pdf(docx, libreoffice_bin="soffice")

        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == str(docx)


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    def test_returns_pdf_path(self, tmp_path: Path):
        docx = tmp_path / "my_report.docx"
        docx.touch()
        pdf = tmp_path / "my_report.pdf"
        pdf.touch()

        with patch("subprocess.run", return_value=_make_result()):
            result = to_pdf(docx, libreoffice_bin="soffice")

        assert result == pdf

    def test_returns_path_type(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        (tmp_path / "report.pdf").touch()

        with patch("subprocess.run", return_value=_make_result()):
            result = to_pdf(docx, libreoffice_bin="soffice")

        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_nonzero_exit_raises(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()

        with patch("subprocess.run", return_value=_make_result(returncode=1, stderr="oops")):
            with pytest.raises(ExporterError, match="exited with code 1"):
                to_pdf(docx, libreoffice_bin="soffice")

    def test_missing_executable_raises(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ExporterError, match="not found"):
                to_pdf(docx, libreoffice_bin="/nonexistent/soffice")

    def test_timeout_raises(self, tmp_path: Path):
        import subprocess as sp

        docx = tmp_path / "report.docx"
        docx.touch()

        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="soffice", timeout=120)):
            with pytest.raises(ExporterError, match="timed out"):
                to_pdf(docx, libreoffice_bin="soffice")

    def test_missing_pdf_after_success_raises(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()
        # PDF is NOT created — subprocess returns 0 but file is absent

        with patch("subprocess.run", return_value=_make_result(returncode=0)):
            with pytest.raises(ExporterError, match="output PDF not found"):
                to_pdf(docx, libreoffice_bin="soffice")

    def test_stderr_included_in_error_message(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        docx.touch()

        with patch("subprocess.run", return_value=_make_result(returncode=2, stderr="fatal: bad format")):
            with pytest.raises(ExporterError, match="fatal: bad format"):
                to_pdf(docx, libreoffice_bin="soffice")
