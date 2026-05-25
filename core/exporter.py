import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_EXECUTABLES = (
    "libreoffice",
    "soffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    "/usr/bin/libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)


class ExporterError(Exception):
    """Raised when PDF export fails."""


def to_pdf(docx_path: Path, *, libreoffice_bin: str | None = None) -> Path:
    """
    Convert a DOCX file to PDF using LibreOffice headless.

    DOCX is the authoritative artifact; PDF is best-effort convenience output.

    Args:
        docx_path: Path to the source .docx file.
        libreoffice_bin: Optional explicit path to the LibreOffice executable.

    Returns:
        Path to the generated .pdf file (sibling of docx_path).

    Raises:
        ExporterError: If LibreOffice is not found or conversion fails.
    """
    exe = _resolve_executable(libreoffice_bin)
    out_dir = docx_path.parent
    expected_pdf = out_dir / (docx_path.stem + ".pdf")

    cmd = [exe, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(docx_path)]
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise ExporterError(f"LibreOffice executable not found: {exe!r}")
    except subprocess.TimeoutExpired:
        raise ExporterError("LibreOffice conversion timed out after 120 s")

    if result.returncode != 0:
        raise ExporterError(
            f"LibreOffice exited with code {result.returncode}. "
            f"stderr: {result.stderr.strip()}"
        )

    if not expected_pdf.exists():
        raise ExporterError(
            f"Conversion appeared to succeed but output PDF not found: {expected_pdf}"
        )

    logger.info("PDF written to %s", expected_pdf)
    return expected_pdf


def _resolve_executable(override: str | None) -> str:
    if override:
        return override
    for candidate in _DEFAULT_EXECUTABLES:
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    return _DEFAULT_EXECUTABLES[0]
