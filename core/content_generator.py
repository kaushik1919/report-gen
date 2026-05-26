import json
import logging
from pathlib import Path
from string import Template

from pydantic import BaseModel, ValidationError

import core.ollama_client as ollama_client
from app.config import DEFAULT_MODEL
from core.models import ReportPlan, SectionContent, SectionSpec
from core.ollama_client import OllamaClientError

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_MAX_SUMMARY_CHARS = 600


class _BlockModel(BaseModel):
    type: str
    text: str | None = None
    items: list[str] | None = None
    headers: list[str] | None = None
    rows: list[list[str]] | None = None
    caption: str | None = None
    image_ref: str | None = None
    key: str | None = None


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
) -> SectionContent:
    """Generate structured JSON content for one section via Ollama."""
    prompt = _build_prompt(section, topic, previous_summaries)
    raw = _call_llm(prompt, model)
    try:
        return _parse_content(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Section %r attempt 1 failed: %s — retrying", section.id, exc)
        retry_prompt = (
            prompt
            + f"\n\nPrevious response was invalid JSON: {exc}\n"
            "Return valid JSON only, no other text."
        )
        raw = _call_llm(retry_prompt, model)
        try:
            return _parse_content(raw)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ContentGeneratorError(
                f"Section {section.id!r} generation failed after retry: {exc2}"
            ) from exc2


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
) -> str:
    template_text = (_PROMPTS_DIR / "section_prompt.txt").read_text(encoding="utf-8")
    context_hint = ""
    if previous_summaries:
        recent = " | ".join(previous_summaries[-3:])
        context_hint = f"Previous sections covered: {recent}"
    return Template(template_text).safe_substitute(
        topic=topic,
        title=section.title,
        level=section.level,
        target_words=section.target_words,
        instructions=section.instructions,
        section_id=section.id,
        context_hint=context_hint,
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
