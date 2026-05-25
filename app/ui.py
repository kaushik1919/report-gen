import dataclasses
import logging

import streamlit as st

from app.config import UPLOADS_DIR
from core.style_extractor import extract
from core.template_loader import TemplateLoadError, load

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Report Generator – M1", layout="wide")
st.title("AI Academic Report Generator")
st.caption("M1: Upload a DOCX template to extract its formatting profile.")


def _profile_to_dict(profile) -> dict:
    """Convert TemplateProfile (with nested dataclasses) to a plain dict."""
    return dataclasses.asdict(profile)


uploaded = st.file_uploader("Upload your university DOCX template", type=["docx"])

if uploaded is not None:
    save_path = UPLOADS_DIR / uploaded.name
    save_path.write_bytes(uploaded.getvalue())

    with st.spinner("Extracting template profile…"):
        try:
            loaded = load(save_path)
            profile = extract(loaded)
        except TemplateLoadError as exc:
            st.error(f"Could not load template: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error during extraction: {exc}")
            st.exception(exc)
            st.stop()

    # ── Summary column + JSON column ────────────────────────────────────────
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

        margins = profile.margins_in
        st.markdown(
            f"**Margins (in):** T {margins['top']} · B {margins['bottom']} "
            f"· L {margins['left']} · R {margins['right']}"
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
        st.json(_profile_to_dict(profile))

    st.success(
        f"Extraction complete — file hash: `{loaded.content_hash[:16]}…`"
    )
