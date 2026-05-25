import dataclasses
import logging

import streamlit as st

from app.config import UPLOADS_DIR
from core.content_generator import ContentGeneratorError, summarize_section, write_section
from core.document_assembler import DocumentAssemblerError, build
from core.exporter import ExporterError, to_pdf
from core.outline_planner import OutlinePlannerError
from core.outline_planner import plan as plan_report
from core.style_extractor import extract
from core.template_loader import TemplateLoadError, load

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Report Generator – M3", layout="wide")
st.title("AI Academic Report Generator")

tab_extract, tab_generate = st.tabs(["Template Analysis", "Generate Report"])


# ── M1: Template Analysis ─────────────────────────────────────────────────────
with tab_extract:
    st.caption("Upload a DOCX template to extract its formatting profile.")
    uploaded = st.file_uploader("Upload your university DOCX template", type=["docx"])

    if uploaded is not None:
        save_path = UPLOADS_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        st.session_state["template_path"] = save_path

        with st.spinner("Extracting template profile…"):
            try:
                loaded = load(save_path)
                profile = extract(loaded)
                st.session_state["template_profile"] = profile
            except TemplateLoadError as exc:
                st.error(f"Could not load template: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error during extraction: {exc}")
                st.exception(exc)
                st.stop()

        col_summary, col_json = st.columns([1, 2])

        with col_summary:
            st.subheader("Summary")
            st.metric("Styles found", len(profile.styles))
            st.metric("Heading levels", len(profile.heading_hierarchy))
            st.metric("Placeholders", len(profile.placeholders))
            st.metric(
                "Page size",
                f"{profile.page_size[0]:.2f}\" × {profile.page_size[1]:.2f}\"",
            )
            m = profile.margins_in
            st.markdown(
                f"**Margins (in):** T {m['top']} · B {m['bottom']} "
                f"· L {m['left']} · R {m['right']}"
            )
            if profile.heading_hierarchy:
                st.subheader("Heading Hierarchy")
                for h in profile.heading_hierarchy:
                    st.text(f"  {h}")
            if profile.placeholders:
                st.subheader("Placeholders")
                for p in profile.placeholders:
                    st.code(p, language=None)
            if profile.section_skeleton:
                st.subheader("Section Skeleton")
                for sec in profile.section_skeleton:
                    indent = "  " * (sec["level"] - 1)
                    st.text(f"{indent}▸ {sec['title']}")

        with col_json:
            st.subheader("Full TemplateProfile JSON")
            st.json(dataclasses.asdict(profile))

        st.success(f"Extraction complete — hash: `{loaded.content_hash[:16]}…`")


# ── M2 + M3: Report Generation ────────────────────────────────────────────────
with tab_generate:
    st.caption(
        "Enter a brief to generate a structured outline, per-section content, "
        "and a styled DOCX ready for download."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        topic = st.text_input("Topic", placeholder="e.g. Machine Learning in Healthcare")
        academic_level = st.selectbox(
            "Academic Level", ["Undergraduate", "Postgraduate", "PhD"]
        )
    with col_right:
        brief = st.text_area(
            "Report Brief",
            height=160,
            placeholder="Describe what the report should cover…",
        )

    heading_hint: list[str] | None = None
    if "template_profile" in st.session_state:
        heading_hint = st.session_state["template_profile"].heading_hierarchy
        if heading_hint:
            st.info(f"Using {len(heading_hint)} template headings from Template Analysis tab.")

    if st.button("Generate Outline", disabled=not (topic and brief)):
        with st.spinner("Generating report outline…"):
            try:
                report_plan = plan_report(
                    brief=brief,
                    topic=topic,
                    academic_level=academic_level,
                    heading_hierarchy=heading_hint,
                )
                st.session_state["report_plan"] = report_plan
                st.session_state.pop("section_contents", None)
                st.session_state.pop("assembled_docx", None)
                st.session_state.pop("assembled_pdf", None)
            except OutlinePlannerError as exc:
                st.error(f"Outline generation failed: {exc}")

    if "report_plan" in st.session_state:
        rp = st.session_state["report_plan"]
        st.subheader(f"Report Plan: {rp.title}")
        st.json(dataclasses.asdict(rp))

        if st.button("Generate All Sections"):
            summaries: list[str] = []
            contents = []
            progress = st.progress(0)
            total = len(rp.sections)
            failed = False
            for i, section in enumerate(rp.sections):
                with st.spinner(f"Writing: {section.title}…"):
                    try:
                        content = write_section(rp, section, topic, summaries)
                        summaries.append(summarize_section(content))
                        contents.append(content)
                    except ContentGeneratorError as exc:
                        st.error(f"Failed to write '{section.title}': {exc}")
                        failed = True
                        break
                progress.progress((i + 1) / total)
            if not failed:
                st.session_state["section_contents"] = contents
                st.session_state.pop("assembled_docx", None)
                st.session_state.pop("assembled_pdf", None)

    if "section_contents" in st.session_state:
        st.subheader("Generated Section Content")
        for content in st.session_state["section_contents"]:
            with st.expander(f"{'#' * content.level} {content.title}"):
                st.json(dataclasses.asdict(content))

        # ── M3: Assembly + Download ────────────────────────────────────────
        st.divider()
        st.subheader("Assemble & Download")

        has_template = (
            "template_path" in st.session_state
            and "template_profile" in st.session_state
        )
        if not has_template:
            st.warning(
                "Upload a template in the **Template Analysis** tab first to enable assembly."
            )
        else:
            if st.button("Assemble DOCX"):
                with st.spinner("Assembling styled document…"):
                    try:
                        out_path = build(
                            st.session_state["template_path"],
                            st.session_state["template_profile"],
                            st.session_state["report_plan"],
                            st.session_state["section_contents"],
                        )
                        st.session_state["assembled_docx"] = out_path
                        st.session_state.pop("assembled_pdf", None)
                        st.success(f"Document assembled: `{out_path.name}`")
                    except DocumentAssemblerError as exc:
                        st.error(f"Assembly failed: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected assembly error: {exc}")
                        st.exception(exc)

        if "assembled_docx" in st.session_state:
            docx_path = st.session_state["assembled_docx"]
            st.download_button(
                label="Download DOCX",
                data=docx_path.read_bytes(),
                file_name=docx_path.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

            # PDF is best-effort — requires LibreOffice installed locally
            if "assembled_pdf" not in st.session_state:
                if st.button("Export PDF  (requires LibreOffice)"):
                    with st.spinner("Converting to PDF via LibreOffice…"):
                        try:
                            pdf_path = to_pdf(docx_path)
                            st.session_state["assembled_pdf"] = pdf_path
                            st.success(f"PDF exported: `{pdf_path.name}`")
                        except ExporterError as exc:
                            st.warning(
                                f"PDF export failed — LibreOffice may not be installed.\n\n{exc}"
                            )

            if "assembled_pdf" in st.session_state:
                pdf_path = st.session_state["assembled_pdf"]
                st.download_button(
                    label="Download PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )
