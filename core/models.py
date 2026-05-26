from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docx import Document

# ---------------------------------------------------------------------------
# Block type constants
# ---------------------------------------------------------------------------

VALID_BLOCK_TYPES: frozenset[str] = frozenset({
    "paragraph",
    "bullet_list",
    "bullets",           # legacy alias for bullet_list
    "numbered_list",
    "table",
    "heading",
    "figure_placeholder",
    "citation_placeholder",
})


@dataclass
class LoadedTemplate:
    """Raw DOCX file loaded into memory. Passed only within the loader pipeline."""
    path: str
    document: Document
    content_hash: str  # SHA-256 hex of the source file


@dataclass
class StyleSpec:
    """Formatting metadata for a single named paragraph/character style."""
    name: str
    font_name: str
    font_size_pt: float
    bold: bool
    italic: bool
    color_hex: str | None       # e.g. "1F2D3E", None if unset/inherited
    alignment: str              # "left" | "center" | "right" | "justify"
    line_spacing: float         # raw value (1.0 = single, 2.0 = double, etc.)


@dataclass
class TemplateProfile:
    """
    Fully deterministic representation of a DOCX template's formatting.
    This is the boundary between the deterministic world and the AI world.
    The LLM never receives this as layout instructions.
    """
    styles: dict[str, StyleSpec]        # keyed by style name
    margins_in: dict[str, float]        # keys: top, bottom, left, right (inches)
    page_size: tuple[float, float]      # (width_inches, height_inches)
    heading_hierarchy: list[str]        # e.g. ["Heading 1", "Heading 2"] in doc order
    placeholders: list[str]             # e.g. ["{{title}}", "{{author}}"]
    section_skeleton: list[dict]        # ordered headings found in the template
    numbering_scheme: dict | None = None


@dataclass
class SectionSpec:
    """
    Describes one section the LLM must write.
    Produced by outline_planner; consumed by content_generator.
    """
    id: str
    title: str
    level: int              # 1 = H1, 2 = H2
    target_words: int
    instructions: str       # natural-language prompt for the LLM
    needs_table: bool = False
    needs_figure: bool = False
    needs_citations: bool = False


@dataclass
class ReportPlan:
    """Structured outline produced by outline_planner from a user brief."""
    title: str
    author: str
    sections: list[SectionSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------

@dataclass
class CitationEntry:
    """A single bibliography entry parsed from a BibTeX file."""
    key: str
    entry_type: str             # bibtex entry type: article, book, misc, etc.
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class BibliographyStore:
    """Keyed collection of citation entries loaded from a bibliography file."""
    entries: dict[str, CitationEntry] = field(default_factory=dict)
    source_path: str = ""

    def keys(self) -> list[str]:
        return list(self.entries.keys())

    def get(self, key: str) -> CitationEntry | None:
        return self.entries.get(key)

    def has(self, key: str) -> bool:
        return key in self.entries


@dataclass
class SectionContent:
    """
    AI-generated content for one section, expressed as typed blocks.
    The LLM produces these blocks as JSON; the assembler renders them.

    Canonical block types (assembler dispatches on ``type``):
        {"type": "paragraph",           "text": "..."}
        {"type": "bullet_list",         "items": ["...", "..."]}
        {"type": "numbered_list",       "items": ["...", "..."]}
        {"type": "table",               "headers": [...], "rows": [[...]]}
        {"type": "heading",             "text": "...", "level": 2}
        {"type": "figure_placeholder",  "caption": "..."}
        {"type": "citation_placeholder","key": "smith2020"}

    Legacy alias: ``"bullets"`` is accepted as a synonym for ``"bullet_list"``.
    Unknown types are logged as a WARNING and skipped by the assembler.
    """
    section_id: str
    title: str
    level: int
    blocks: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
