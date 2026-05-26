import json
import logging
from pathlib import Path
from string import Template

from pydantic import BaseModel, ValidationError

import core.ollama_client as ollama_client
from app.config import DEFAULT_MODEL
from core.models import ReportPlan, SectionSpec
from core.ollama_client import OllamaClientError

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class _SectionSpecModel(BaseModel):
    id: str
    title: str
    level: int
    target_words: int
    instructions: str
    needs_table: bool = False
    needs_figure: bool = False
    needs_citations: bool = False


class _ReportPlanModel(BaseModel):
    title: str
    author: str
    sections: list[_SectionSpecModel]


class OutlinePlannerError(Exception):
    """Raised when outline generation fails after retries."""


def plan(
    brief: str,
    topic: str,
    academic_level: str,
    heading_hierarchy: list[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> ReportPlan:
    """Generate a structured ReportPlan from a user brief via Ollama."""
    prompt = _build_prompt(brief, topic, academic_level, heading_hierarchy)
    raw = _call_llm(prompt, model)
    try:
        return _parse_plan(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Outline attempt 1 failed: %s — retrying", exc)
        retry_prompt = (
            prompt
            + f"\n\nPrevious response was invalid JSON: {exc}\n"
            "Return valid JSON only, no other text."
        )
        raw = _call_llm(retry_prompt, model)
        try:
            return _parse_plan(raw)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise OutlinePlannerError(
                f"Outline generation failed after retry: {exc2}"
            ) from exc2


def _build_prompt(
    brief: str,
    topic: str,
    academic_level: str,
    heading_hierarchy: list[str] | None,
) -> str:
    template_text = (_PROMPTS_DIR / "outline_prompt.txt").read_text(encoding="utf-8")
    heading_hint = (
        f"Template headings to follow: {', '.join(heading_hierarchy)}"
        if heading_hierarchy
        else ""
    )
    return Template(template_text).safe_substitute(
        topic=topic,
        academic_level=academic_level,
        brief=brief,
        heading_hint=heading_hint,
    )


def _call_llm(prompt: str, model: str) -> str:
    try:
        return ollama_client.generate(
            prompt,
            model=model,
            temperature=0.3,
            prompt_type="outline",
        )
    except OllamaClientError as exc:
        raise OutlinePlannerError(f"Ollama request failed: {exc}") from exc


def _parse_plan(raw: str) -> ReportPlan:
    data = json.loads(raw)
    validated = _ReportPlanModel.model_validate(data)
    sections = [
        SectionSpec(
            id=s.id,
            title=s.title,
            level=s.level,
            target_words=s.target_words,
            instructions=s.instructions,
            needs_table=s.needs_table,
            needs_figure=s.needs_figure,
            needs_citations=s.needs_citations,
        )
        for s in validated.sections
    ]
    return ReportPlan(title=validated.title, author=validated.author, sections=sections)
