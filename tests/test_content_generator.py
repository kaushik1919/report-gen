"""
Tests for content_generator.
Ollama is fully mocked — no running daemon required.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from core.content_generator import (
    ContentGeneratorError,
    _parse_content,
    summarize_section,
    write_section,
)
from core.models import ReportPlan, SectionContent, SectionSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SECTION_JSON = json.dumps({
    "section_id": "sec_01",
    "title": "Introduction",
    "level": 1,
    "blocks": [
        {
            "type": "paragraph",
            "text": (
                "This report examines the application of machine learning techniques "
                "within clinical healthcare settings. The increasing availability of "
                "electronic health records has enabled the development of predictive "
                "models that support diagnostic and prognostic decision-making."
            ),
        },
        {"type": "bullets", "items": ["Improved diagnostic accuracy.", "Reduced clinician workload.", "Scalable to large patient cohorts."]},
    ],
    "citations": [],
})

PLAN = ReportPlan(title="ML in Healthcare", author="Author", sections=[])
SECTION = SectionSpec(
    id="sec_01",
    title="Introduction",
    level=1,
    target_words=300,
    instructions="Introduce the role of ML in clinical settings.",
)


def _chat_mock(content: str) -> MagicMock:
    response = MagicMock()
    response.message.content = content
    return response


# ---------------------------------------------------------------------------
# _parse_content — schema validation
# ---------------------------------------------------------------------------

class TestParseContent:
    def test_valid_json_returns_section_content(self):
        result = _parse_content(VALID_SECTION_JSON)
        assert isinstance(result, SectionContent)
        assert result.section_id == "sec_01"
        assert result.title == "Introduction"
        assert result.level == 1

    def test_blocks_are_dicts(self):
        result = _parse_content(VALID_SECTION_JSON)
        assert isinstance(result.blocks, list)
        for block in result.blocks:
            assert isinstance(block, dict)

    def test_paragraph_block_has_type_and_text(self):
        result = _parse_content(VALID_SECTION_JSON)
        para = result.blocks[0]
        assert para["type"] == "paragraph"
        assert "text" in para
        assert isinstance(para["text"], str)

    def test_bullets_block_has_items(self):
        result = _parse_content(VALID_SECTION_JSON)
        bullets = result.blocks[1]
        assert bullets["type"] == "bullets"
        assert "items" in bullets
        assert isinstance(bullets["items"], list)

    def test_none_fields_excluded_from_block_dict(self):
        raw = json.dumps({
            "section_id": "s1",
            "title": "T",
            "level": 1,
            "blocks": [{"type": "paragraph", "text": "Hello.", "items": None, "headers": None}],
            "citations": [],
        })
        result = _parse_content(raw)
        assert "items" not in result.blocks[0]
        assert "headers" not in result.blocks[0]

    def test_citations_default_empty(self):
        raw = json.dumps({
            "section_id": "s1",
            "title": "T",
            "level": 1,
            "blocks": [{"type": "paragraph", "text": "Text."}],
        })
        result = _parse_content(raw)
        assert result.citations == []

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_content("{bad json")

    def test_missing_section_id_raises(self):
        raw = json.dumps({"title": "T", "level": 1, "blocks": [], "citations": []})
        with pytest.raises(Exception):
            _parse_content(raw)


# ---------------------------------------------------------------------------
# summarize_section — context window logic
# ---------------------------------------------------------------------------

class TestSummarizeSection:
    def test_returns_non_empty_string(self):
        content = _parse_content(VALID_SECTION_JSON)
        summary = summarize_section(content)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_only_paragraph_blocks_included(self):
        content = _parse_content(VALID_SECTION_JSON)
        summary = summarize_section(content)
        assert "Improved diagnostic accuracy" not in summary

    def test_paragraph_text_in_summary(self):
        content = _parse_content(VALID_SECTION_JSON)
        summary = summarize_section(content)
        assert "machine learning" in summary.lower()

    def test_long_content_truncated_to_limit(self):
        long_text = "Academic prose. " * 200
        raw = json.dumps({
            "section_id": "s1",
            "title": "T",
            "level": 1,
            "blocks": [{"type": "paragraph", "text": long_text}],
            "citations": [],
        })
        content = _parse_content(raw)
        summary = summarize_section(content)
        assert len(summary) <= 650

    def test_truncated_summary_ends_with_ellipsis(self):
        long_text = "x " * 500
        raw = json.dumps({
            "section_id": "s1",
            "title": "T",
            "level": 1,
            "blocks": [{"type": "paragraph", "text": long_text}],
            "citations": [],
        })
        content = _parse_content(raw)
        summary = summarize_section(content)
        assert summary.endswith("…")

    def test_no_paragraphs_returns_empty_string(self):
        raw = json.dumps({
            "section_id": "s1",
            "title": "T",
            "level": 1,
            "blocks": [{"type": "bullets", "items": ["a", "b"]}],
            "citations": [],
        })
        content = _parse_content(raw)
        assert summarize_section(content) == ""


# ---------------------------------------------------------------------------
# write_section() — retry logic (Ollama mocked)
# ---------------------------------------------------------------------------

class TestWriteSectionRetry:
    @patch("core.content_generator.ollama.Client")
    def test_valid_first_attempt_returns_content(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock(VALID_SECTION_JSON)
        result = write_section(PLAN, SECTION, "ML", [])
        assert isinstance(result, SectionContent)
        assert MockClient.return_value.chat.call_count == 1

    @patch("core.content_generator.ollama.Client")
    def test_invalid_first_valid_second_returns_content(self, MockClient):
        MockClient.return_value.chat.side_effect = [
            _chat_mock("not json"),
            _chat_mock(VALID_SECTION_JSON),
        ]
        result = write_section(PLAN, SECTION, "ML", [])
        assert isinstance(result, SectionContent)
        assert MockClient.return_value.chat.call_count == 2

    @patch("core.content_generator.ollama.Client")
    def test_both_attempts_fail_raises(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock("{{{bad")
        with pytest.raises(ContentGeneratorError, match="failed after retry"):
            write_section(PLAN, SECTION, "ML", [])
        assert MockClient.return_value.chat.call_count == 2

    @patch("core.content_generator.ollama.Client")
    def test_previous_summaries_in_prompt(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock(VALID_SECTION_JSON)
        write_section(PLAN, SECTION, "ML", ["Summary of intro.", "Summary of background."])
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "Summary of intro." in prompt

    @patch("core.content_generator.ollama.Client")
    def test_only_last_three_summaries_used(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock(VALID_SECTION_JSON)
        summaries = ["S1", "S2", "S3", "S4", "S5"]
        write_section(PLAN, SECTION, "ML", summaries)
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "S1" not in prompt
        assert "S2" not in prompt
        assert "S3" in prompt
        assert "S4" in prompt
        assert "S5" in prompt

    @patch("core.content_generator.ollama.Client")
    def test_no_summaries_omits_context_hint(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock(VALID_SECTION_JSON)
        write_section(PLAN, SECTION, "ML", [])
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "Previous sections covered" not in prompt
