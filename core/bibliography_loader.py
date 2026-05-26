"""
Load and validate BibTeX bibliography files.

All parsing is deterministic — no LLM calls, no network access.
Malformed individual entries are skipped with a logged warning;
the rest of the file continues to load.
"""
import logging
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser

from core.models import BibliographyStore, CitationEntry

logger = logging.getLogger(__name__)


class BibliographyLoadError(Exception):
    """Raised when a bibliography file cannot be loaded or parsed."""


def load_bibtex(path: Path) -> BibliographyStore:
    """
    Load a BibTeX .bib file and return a BibliographyStore.

    Raises BibliographyLoadError for:
    - file not found
    - wrong extension
    - unreadable file
    - catastrophic parse failure (entire file unparseable)

    Partial failures (individual malformed entries) are logged as warnings
    and skipped; the function still returns valid entries.
    """
    if not path.exists():
        raise BibliographyLoadError(f"File not found: {path}")
    if path.suffix.lower() != ".bib":
        raise BibliographyLoadError(
            f"Expected .bib file, got: {path.suffix!r}"
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BibliographyLoadError(f"Cannot read file {path}: {exc}") from exc

    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False

    try:
        db = bibtexparser.loads(text, parser=parser)
    except Exception as exc:
        raise BibliographyLoadError(
            f"BibTeX parse error in {path}: {exc}"
        ) from exc

    entries: dict[str, CitationEntry] = {}
    for raw in db.entries:
        key = raw.get("ID", "").strip()
        if not key:
            logger.warning(
                "bibliography_loader: skipping entry with missing citation key"
            )
            continue
        entry_type = raw.get("ENTRYTYPE", "misc").lower()
        fields = {k: v for k, v in raw.items() if k not in ("ID", "ENTRYTYPE")}
        entries[key] = CitationEntry(key=key, entry_type=entry_type, fields=fields)

    logger.info(
        "bibliography_loader: loaded %d citation entries from %s",
        len(entries),
        path,
    )
    return BibliographyStore(entries=entries, source_path=str(path))
