import logging
import re
import shutil
from pathlib import Path

from docx import Document

from app.config import OUTPUTS_DIR
from core.models import ReportPlan, SectionContent, TemplateProfile

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class DocumentAssemblerError(Exception):
    """Raised when document assembly fails."""


def build(
    template_path: Path,
    profile: TemplateProfile,
    plan: ReportPlan,
    sections: list[SectionContent],
) -> Path:
    """
    Merge TemplateProfile + ReportPlan + [SectionContent] into a styled DOCX.
    Returns the path to the assembled output file.
    """
    out_path = OUTPUTS_DIR / f"{_slug(plan.title)}.docx"
    shutil.copy2(template_path, out_path)
    doc = Document(str(out_path))

    _replace_placeholders(doc, {"title": plan.title, "author": plan.author})

    for section in sections:
        _insert_section(doc, section, profile)

    doc.save(str(out_path))
    logger.info("Assembled document saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Placeholder replacement
# ---------------------------------------------------------------------------

def _replace_placeholders(doc: Document, subs: dict[str, str]) -> None:
    """Replace {{key}} tokens in template paragraphs. Handles cross-run splits."""
    for para in doc.paragraphs:
        if not _PLACEHOLDER_RE.search(para.text):
            continue
        replaced = _PLACEHOLDER_RE.sub(
            lambda m: subs.get(m.group(1), m.group(0)), para.text
        )
        for run in para.runs:
            run.text = ""
        if para.runs:
            para.runs[0].text = replaced
        else:
            para.add_run(replaced)


# ---------------------------------------------------------------------------
# Section insertion
# ---------------------------------------------------------------------------

def _insert_section(
    doc: Document, section: SectionContent, profile: TemplateProfile
) -> None:
    heading_style = _resolve_heading_style(section.level, profile)
    doc.add_paragraph(section.title, style=heading_style)

    for block in section.blocks:
        block_type = block.get("type")
        if block_type == "paragraph":
            _add_paragraph(doc, block.get("text", ""), profile)
        elif block_type == "bullets":
            _add_bullets(doc, block.get("items", []), profile)
        elif block_type == "table":
            _add_table(doc, block.get("headers", []), block.get("rows", []))
        else:
            logger.debug("Skipping unsupported block type %r", block_type)


def _add_paragraph(doc: Document, text: str, profile: TemplateProfile) -> None:
    doc.add_paragraph(text, style=_resolve_body_style(profile))


def _add_bullets(doc: Document, items: list[str], profile: TemplateProfile) -> None:
    bullet_style = "List Bullet" if "List Bullet" in profile.styles else "List Paragraph"
    for item in items:
        doc.add_paragraph(item, style=bullet_style)


def _add_table(
    doc: Document, headers: list[str], rows: list[list[str]]
) -> None:
    if not headers:
        return
    col_count = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=col_count)
    try:
        table.style = "Table Grid"
    except Exception:
        pass  # style may not exist in minimalist templates

    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header

    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row[:col_count]):
            table.rows[r_idx + 1].cells[c_idx].text = cell_text


# ---------------------------------------------------------------------------
# Style resolution helpers
# ---------------------------------------------------------------------------

def _resolve_heading_style(level: int, profile: TemplateProfile) -> str:
    name = f"Heading {level}"
    if name in profile.styles:
        return name
    for lvl in range(level - 1, 0, -1):
        candidate = f"Heading {lvl}"
        if candidate in profile.styles:
            logger.warning("Heading %d absent from profile, falling back to %s", level, candidate)
            return candidate
    return name  # python-docx has Heading 1–9 built-in


def _resolve_body_style(profile: TemplateProfile) -> str:
    for candidate in ("Body Text", "Normal"):
        if candidate in profile.styles:
            return candidate
    return "Normal"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text.lower()).strip("_")[:60]
