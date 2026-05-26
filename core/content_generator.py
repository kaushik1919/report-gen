from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError, model_validator

import core.ollama_client as ollama_client
from app.config import DEFAULT_MODEL
from core.models import ReportPlan, SectionContent, SectionSpec
from core.ollama_client import OllamaClientError

if TYPE_CHECKING:
    from core.rag_store import RAGStore

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_MAX_SUMMARY_CHARS = 600

_VALID_BLOCK_TYPE = Literal[
    "paragraph",
    "bullet_list",
    "bullets",           # legacy alias
    "numbered_list",
    "table",
    "heading",
    "figure_placeholder",
    "citation_placeholder",
]


class _BlockModel(BaseModel):
    type: _VALID_BLOCK_TYPE
    text: str | None = None
    items: list[str] | None = None
    headers: list[str] | None = None
    rows: list[list[str]] | None = None
    caption: str | None = None
    image_ref: str | None = None
    key: str | None = None

    @model_validator(mode="after")
    def _validate_citation_key(self) -> _BlockModel:
        if self.type == "citation_placeholder" and not (self.key or "").strip():
            raise ValueError("citation_placeholder block must have a non-empty 'key'")
        return self


class _SectionContentModel(BaseModel):
    section_id: str
    title: str
    level: int
    blocks: list[_BlockModel]
    citations: list[str] = []


class ContentGeneratorError(Exception):
    """Raised when section generation fails after retries."""


def write_section(
    plan: ReportPlan,
    section: SectionSpec,
    topic: str,
    previous_summaries: list[str],
    model: str = DEFAULT_MODEL,
    citation_keys: list[str] | None = None,
    rag_store: RAGStore | None = None,
) -> SectionContent:
    """Generate structured JSON content for one section via Ollama."""
    prompt = _build_prompt(section, topic, previous_summaries, citation_keys, rag_store)
    raw = _call_llm(prompt, model)
    try:
        content = _parse_content(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Section %r attempt 1 failed: %s — retrying", section.id, exc)
        retry_prompt = (
            prompt
            + f"\n\nPrevious response was invalid JSON: {exc}\n"
            "Return valid JSON only, no other text."
        )
        raw = _call_llm(retry_prompt, model)
        try:
            content = _parse_content(raw)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ContentGeneratorError(
                f"Section {section.id!r} generation failed after retry: {exc2}"
            ) from exc2

    if citation_keys is not None:
        _warn_invalid_citation_keys(content, citation_keys)
    return content


def summarize_section(content: SectionContent) -> str:
    """Extract a concise rolling summary from generated content for context management."""
    texts: list[str] = []
    for block in content.blocks:
        if block.get("type") == "paragraph" and block.get("text"):
            texts.append(block["text"])
            if sum(len(t) for t in texts) >= _MAX_SUMMARY_CHARS:
                break
    raw = " ".join(texts)
    if len(raw) > _MAX_SUMMARY_CHARS:
        return raw[:_MAX_SUMMARY_CHARS] + "…"
    return raw


def _build_prompt(
    section: SectionSpec,
    topic: str,
    previous_summaries: list[str],
    citation_keys: list[str] | None = None,
    rag_store: RAGStore | None = None,
) -> str:
    template_text = (_PROMPTS_DIR / "section_prompt.txt").read_text(encoding="utf-8")
    context_hint = ""
    if previous_summaries:
        recent = " | ".join(previous_summaries[-3:])
        context_hint = f"Previous sections covered: {recent}"

    if citation_keys:
        keys_str = ", ".join(citation_keys)
        citation_keys_hint = f"Available citation keys (use ONLY these): {keys_str}"
        citation_rules_hint = (
            "- Only use citation_placeholder blocks for references\n"
            "- Cite ONLY from the provided keys above — NEVER invent or hallucinate citation keys\n"
            '- Use the format: {"type": "citation_placeholder", "key": "author2020"}\n'
            "- Leave the citations array empty"
        )
    else:
        citation_keys_hint = ""
        citation_rules_hint = (
            "- Leave the citations array empty — do not hallucinate sources\n"
            "- Prefer block types: paragraph, bullet_list, table"
        )

    reference_material = _build_reference_material(section, rag_store)

    return Template(template_text).safe_substitute(
        topic=topic,
        title=section.title,
        level=section.level,
        target_words=section.target_words,
        instructions=section.instructions,
        section_id=section.id,
        context_hint=context_hint,
        citation_keys_hint=citation_keys_hint,
        citation_rules_hint=citation_rules_hint,
        reference_material=reference_material,
    )


def _build_reference_material(section: SectionSpec, rag_store: RAGStore | None) -> str:
    """Retrieve top-k passages and format as a prompt block; empty string if RAG disabled."""
    if rag_store is None:
        return ""
    try:
        result = rag_store.retrieve(section.title, k=5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("content_generator: RAG retrieval failed for %r: %s", section.title, exc)
        return ""
    if not result.chunks:
        return ""
    lines = ["Reference Material (use as supporting evidence, do not copy verbatim):"]
    for i, chunk in enumerate(result.chunks, 1):
        src = f"{chunk.source} p.{chunk.page + 1}"
        lines.append(f"[{i}] ({src}) {chunk.text[:400]}")
    return "\n".join(lines)


def _warn_invalid_citation_keys(content: SectionContent, valid_keys: list[str]) -> None:
    """Warn if citation_placeholder blocks use keys not in the provided valid set."""
    valid_set = set(valid_keys)
    for block in content.blocks:
        if block.get("type") == "citation_placeholder":
            key = block.get("key", "")
            if key not in valid_set:
                logger.warning(
                    "content_generator: citation_placeholder key %r not in provided set "
                    "— possible hallucination; will surface as unknown during post-processing",
                    key,
                )


def _call_llm(prompt: str, model: str) -> str:
    try:
        return ollama_client.generate(
            prompt,
            model=model,
            temperature=0.4,
            prompt_type="section",
        )
    except OllamaClientError as exc:
        raise ContentGeneratorError(f"Ollama request failed: {exc}") from exc


def _parse_content(raw: str) -> SectionContent:
    data = json.loads(raw)
    validated = _SectionContentModel.model_validate(data)
    return SectionContent(
        section_id=validated.section_id,
        title=validated.title,
        level=validated.level,
        blocks=[b.model_dump(exclude_none=True) for b in validated.blocks],
        citations=validated.citations,
    )
