import dataclasses
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import core.content_generator as content_generator
import core.document_assembler as document_assembler
import core.exporter as exporter
import core.ollama_client as ollama_client
import core.ollama_detector as ollama_detector
import core.outline_planner as outline_planner
import core.style_extractor as style_extractor
import core.template_loader as template_loader
from app.config import DEFAULT_MODEL, OLLAMA_BASE_URL, OUTPUTS_DIR, UPLOADS_DIR
from core.content_generator import ContentGeneratorError
from core.exporter import ExporterError
from core.models import ReportPlan, SectionContent, SectionSpec
from core.outline_planner import OutlinePlannerError
from core.template_loader import TemplateLoadError

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(application: FastAPI):
    ollama_detector.startup_diagnostics()
    yield


app = FastAPI(title="Report Generator API", version="0.3", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    template_id: str
    brief: str
    topic: str
    academic_level: str
    model: str = DEFAULT_MODEL


class ContentRequest(BaseModel):
    plan: dict
    topic: str
    model: str = DEFAULT_MODEL


class BuildRequest(BaseModel):
    template_id: str
    plan: dict
    sections: list[dict]


class ExportRequest(BaseModel):
    docx_filename: str
    libreoffice_bin: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plan_from_dict(d: dict) -> ReportPlan:
    sections = [
        SectionSpec(
            id=s["id"],
            title=s["title"],
            level=s["level"],
            target_words=s["target_words"],
            instructions=s["instructions"],
            needs_table=s.get("needs_table", False),
            needs_figure=s.get("needs_figure", False),
            needs_citations=s.get("needs_citations", False),
        )
        for s in d.get("sections", [])
    ]
    return ReportPlan(title=d["title"], author=d["author"], sections=sections)


def _section_from_dict(d: dict) -> SectionContent:
    return SectionContent(
        section_id=d["section_id"],
        title=d["title"],
        level=d["level"],
        blocks=d.get("blocks", []),
        citations=d.get("citations", []),
    )


def _require_template(template_id: str) -> Path:
    p = UPLOADS_DIR / f"{template_id}.docx"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id!r}")
    return p


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/template/extract")
async def extract_template(file: UploadFile = File(...)):
    """Upload a DOCX template and return its TemplateProfile."""
    if not (file.filename or "").lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are accepted")

    content = await file.read()
    template_id = str(uuid.uuid4())
    saved_path = UPLOADS_DIR / f"{template_id}.docx"
    saved_path.write_bytes(content)

    try:
        loaded = template_loader.load(saved_path)
        profile = style_extractor.extract(loaded)
    except TemplateLoadError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))

    return {"template_id": template_id, "profile": dataclasses.asdict(profile)}


@app.post("/report/plan")
def generate_plan(req: PlanRequest):
    """Generate a structured ReportPlan from a brief via the LLM."""
    tpl_path = _require_template(req.template_id)
    try:
        loaded = template_loader.load(tpl_path)
        profile = style_extractor.extract(loaded)
        report_plan = outline_planner.plan(
            brief=req.brief,
            topic=req.topic,
            academic_level=req.academic_level,
            heading_hierarchy=profile.heading_hierarchy,
            model=req.model,
        )
    except TemplateLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OutlinePlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {"plan": dataclasses.asdict(report_plan)}


@app.post("/report/content")
def generate_content(req: ContentRequest):
    """Generate SectionContent for every section in the supplied plan."""
    try:
        report_plan = _plan_from_dict(req.plan)
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid plan structure: {exc}")

    sections: list[dict] = []
    previous_summaries: list[str] = []

    for spec in report_plan.sections:
        try:
            section_content = content_generator.write_section(
                plan=report_plan,
                section=spec,
                topic=req.topic,
                previous_summaries=previous_summaries,
                model=req.model,
            )
        except ContentGeneratorError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        previous_summaries.append(content_generator.summarize_section(section_content))
        sections.append(dataclasses.asdict(section_content))

    return {"sections": sections}


@app.post("/report/build")
def build_report(req: BuildRequest):
    """Assemble a styled DOCX from template + plan + sections."""
    tpl_path = _require_template(req.template_id)
    try:
        loaded = template_loader.load(tpl_path)
        profile = style_extractor.extract(loaded)
        report_plan = _plan_from_dict(req.plan)
        section_contents = [_section_from_dict(s) for s in req.sections]
    except TemplateLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request data: {exc}")

    out_path = document_assembler.build(tpl_path, profile, report_plan, section_contents)
    return FileResponse(
        path=str(out_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=out_path.name,
    )


@app.get("/health/ollama")
def health_ollama():
    """Return Ollama server status, installed models, and selected model."""
    reachable = ollama_client.check_connectivity()
    installed = ollama_client.list_models() if reachable else []
    selected = ollama_detector.select_default_model(installed)
    return {
        "server_reachable": reachable,
        "installed_models": installed,
        "selected_model": selected,
        "server_url": OLLAMA_BASE_URL,
    }


@app.post("/report/export-pdf")
def export_pdf(req: ExportRequest):
    """Convert an assembled DOCX to PDF via LibreOffice. PDF is best-effort."""
    docx_path = OUTPUTS_DIR / req.docx_filename
    if not docx_path.exists():
        raise HTTPException(status_code=404, detail=f"DOCX not found: {req.docx_filename!r}")

    try:
        pdf_path = exporter.to_pdf(docx_path, libreoffice_bin=req.libreoffice_bin)
    except ExporterError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
