import hashlib
import logging
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from core.models import LoadedTemplate

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class TemplateLoadError(Exception):
    """Raised when a template cannot be loaded or validated."""


def load(path: str | Path) -> LoadedTemplate:
    """Load and validate a .docx template file, returning a LoadedTemplate."""
    path = Path(path)

    if not path.exists():
        raise TemplateLoadError(f"File not found: {path}")

    if path.suffix.lower() != ".docx":
        raise TemplateLoadError(f"Expected a .docx file, got: {path.suffix!r}")

    size = path.stat().st_size
    if size == 0:
        raise TemplateLoadError(f"File is empty: {path.name}")

    if size > MAX_FILE_SIZE_BYTES:
        raise TemplateLoadError(
            f"File too large: {size / 1_048_576:.1f} MB (max 20 MB)"
        )

    try:
        document = Document(str(path))
    except PackageNotFoundError as exc:
        raise TemplateLoadError(f"Not a valid DOCX file: {path.name}") from exc
    except Exception as exc:
        raise TemplateLoadError(f"Failed to open {path.name}: {exc}") from exc

    content_hash = _sha256(path)
    logger.info("Loaded template %r (hash=%s)", path.name, content_hash[:12])

    return LoadedTemplate(
        path=str(path),
        document=document,
        content_hash=content_hash,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
