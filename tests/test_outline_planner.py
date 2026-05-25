"""
Tests for outline_planner.
Ollama is fully mocked — no running daemon required.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from core.models import ReportPlan, SectionSpec
from core.outline_planner import OutlinePlannerError, _parse_plan, plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PLAN_JSON = json.dumps({
    "title": "Machine Learning in Healthcare",
    "author": "Student Author",
    "sections": [
        {
            "id": "sec_01",
            "title": "Introduction",
            "level": 1,
            "target_words": 300,
            "instructions": "Introduce the topic and state the report's objectives.",
        },
        {
            "id": "sec_02",
            "title": "Background",
            "level": 2,
            "target_words": 400,
            "instructions": "Summarise key prior work in machine learning applied to diagnostics.",
            "needs_table": True,
        },
    ],
})


def _chat_mock(content: str) -> MagicMock:
    response = MagicMock()
    response.message.content = content
    return response


# ---------------------------------------------------------------------------
# _parse_plan — schema validation
# ---------------------------------------------------------------------------

class TestParsePlan:
    def test_valid_json_returns_report_plan(self):
        result = _parse_plan(VALID_PLAN_JSON)
        assert isinstance(result, ReportPlan)
        assert result.title == "Machine Learning in Healthcare"
        assert len(result.sections) == 2

    def test_sections_are_section_spec_instances(self):
        result = _parse_plan(VALID_PLAN_JSON)
        for s in result.sections:
            assert isinstance(s, SectionSpec)

    def test_section_fields_populated(self):
        result = _parse_plan(VALID_PLAN_JSON)
        intro = result.sections[0]
        assert intro.id == "sec_01"
        assert intro.level == 1
        assert intro.target_words == 300

    def test_boolean_defaults_applied(self):
        result = _parse_plan(VALID_PLAN_JSON)
        intro = result.sections[0]
        assert intro.needs_table is False
        assert intro.needs_figure is False
        assert intro.needs_citations is False

    def test_explicit_boolean_preserved(self):
        result = _parse_plan(VALID_PLAN_JSON)
        bg = result.sections[1]
        assert bg.needs_table is True

    def test_missing_author_raises(self):
        raw = json.dumps({"title": "T", "sections": []})
        with pytest.raises((ValidationError, Exception)):
            _parse_plan(raw)

    def test_invalid_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_plan("not json {{{")

    def test_section_missing_id_raises(self):
        raw = json.dumps({
            "title": "T",
            "author": "A",
            "sections": [{"title": "X", "level": 1, "target_words": 100, "instructions": "Y"}],
        })
        with pytest.raises((ValidationError, Exception)):
            _parse_plan(raw)


# ---------------------------------------------------------------------------
# plan() — retry logic (Ollama mocked)
# ---------------------------------------------------------------------------

class TestPlanRetry:
    @patch("core.outline_planner.ollama.Client")
    def test_valid_first_attempt_returns_plan(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock(VALID_PLAN_JSON)
        result = plan("a brief", "ML in Healthcare", "Undergraduate")
        assert isinstance(result, ReportPlan)
        assert MockClient.return_value.chat.call_count == 1

    @patch("core.outline_planner.ollama.Client")
    def test_invalid_first_valid_second_returns_plan(self, MockClient):
        MockClient.return_value.chat.side_effect = [
            _chat_mock("not valid json {{{"),
            _chat_mock(VALID_PLAN_JSON),
        ]
        result = plan("a brief", "ML in Healthcare", "Undergraduate")
        assert isinstance(result, ReportPlan)
        assert MockClient.return_value.chat.call_count == 2

    @patch("core.outline_planner.ollama.Client")
    def test_both_attempts_invalid_raises(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock("{not json}")
        with pytest.raises(OutlinePlannerError, match="failed after retry"):
            plan("a brief", "ML", "Undergraduate")
        assert MockClient.return_value.chat.call_count == 2

    @patch("core.outline_planner.ollama.Client")
    def test_ollama_connection_error_raises(self, MockClient):
        MockClient.return_value.chat.side_effect = ConnectionRefusedError("Ollama not running")
        with pytest.raises(OutlinePlannerError, match="Ollama request failed"):
            plan("brief", "topic", "PhD")

    @patch("core.outline_planner.ollama.Client")
    def test_heading_hierarchy_passed_to_prompt(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_mock(VALID_PLAN_JSON)
        plan("brief", "topic", "Postgraduate", heading_hierarchy=["Heading 1", "Heading 2"])
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "Heading 1" in prompt
        assert "Heading 2" in prompt
