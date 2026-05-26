"""
Tests for core/ollama_detector.py.
subprocess and network calls are mocked — no Ollama daemon required.
"""
import subprocess
from unittest.mock import MagicMock, patch

from core.ollama_detector import (
    detect_executable,
    detect_version,
    select_default_model,
    startup_diagnostics,
)

# ---------------------------------------------------------------------------
# detect_executable()
# ---------------------------------------------------------------------------

class TestDetectExecutable:
    @patch("core.ollama_detector.shutil.which", return_value="/usr/local/bin/ollama")
    def test_returns_path_when_found_in_PATH(self, _):
        result = detect_executable()
        assert result == "/usr/local/bin/ollama"

    @patch("core.ollama_detector.shutil.which", return_value=None)
    @patch("core.ollama_detector.platform.system", return_value="Linux")
    def test_returns_none_when_not_found_on_linux(self, _sys, _which, tmp_path):
        result = detect_executable()
        assert result is None

    @patch("core.ollama_detector.shutil.which", return_value=None)
    @patch("core.ollama_detector.platform.system", return_value="Windows")
    def test_windows_uses_localappdata(self, _sys, _which):
        with patch.dict("os.environ", {"LOCALAPPDATA": "C:\\Users\\test\\AppData\\Local"}):
            result = detect_executable()
            assert result is None or "Ollama" in (result or "")


# ---------------------------------------------------------------------------
# detect_version()
# ---------------------------------------------------------------------------

class TestDetectVersion:
    def test_returns_version_string_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ollama version 0.3.12\n"
        with patch("core.ollama_detector.subprocess.run", return_value=mock_result):
            result = detect_version("/usr/local/bin/ollama")
        assert result == "ollama version 0.3.12"

    def test_returns_none_on_nonzero_returncode(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("core.ollama_detector.subprocess.run", return_value=mock_result):
            result = detect_version("/usr/local/bin/ollama")
        assert result is None

    def test_returns_none_on_file_not_found(self):
        with patch("core.ollama_detector.subprocess.run",
                   side_effect=FileNotFoundError("not found")):
            result = detect_version("/nonexistent/ollama")
        assert result is None

    def test_returns_none_on_timeout(self):
        with patch("core.ollama_detector.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=5)):
            result = detect_version("/usr/local/bin/ollama")
        assert result is None


# ---------------------------------------------------------------------------
# select_default_model()
# ---------------------------------------------------------------------------

class TestSelectDefaultModel:
    def test_returns_none_for_empty_list(self):
        assert select_default_model([]) is None

    def test_returns_highest_priority_model(self):
        installed = ["llama3.1:8b", "qwen2.5:14b", "qwen2.5:7b"]
        assert select_default_model(installed) == "qwen2.5:14b"

    def test_falls_through_priority_list(self):
        installed = ["llama3.1:8b", "qwen2.5:7b"]
        assert select_default_model(installed) == "qwen2.5:7b"

    def test_returns_llama_when_no_qwen(self):
        installed = ["llama3.1:8b", "llama3"]
        assert select_default_model(installed) == "llama3.1:8b"

    def test_falls_back_to_first_installed_when_no_priority_match(self):
        installed = ["custom-model:latest", "another-model:7b"]
        assert select_default_model(installed) == "custom-model:latest"

    def test_single_model_returned(self):
        assert select_default_model(["only-model:3b"]) == "only-model:3b"


# ---------------------------------------------------------------------------
# startup_diagnostics()
# ---------------------------------------------------------------------------

class TestStartupDiagnostics:
    @patch("core.ollama_detector.detect_executable", return_value="/usr/local/bin/ollama")
    @patch("core.ollama_detector.detect_version", return_value="ollama version 0.3.12")
    @patch("core.ollama_client.ollama.Client")
    def test_returns_dict_with_expected_keys(self, MockClient, _ver, _exe):
        result_mock = MagicMock()
        result_mock.models = []
        MockClient.return_value.list.return_value = result_mock
        result = startup_diagnostics()
        assert "executable" in result
        assert "version" in result
        assert "server_reachable" in result
        assert "installed_models" in result
        assert "selected_model" in result
        assert "server_url" in result

    @patch("core.ollama_detector.detect_executable", return_value=None)
    @patch("core.ollama_detector.detect_version", return_value=None)
    @patch("core.ollama_client.ollama.Client")
    def test_handles_missing_executable(self, MockClient, _ver, _exe):
        MockClient.return_value.list.side_effect = ConnectionRefusedError("no server")
        result = startup_diagnostics()
        assert result["executable"] is None
        assert result["server_reachable"] is False
        assert result["installed_models"] == []
        assert result["selected_model"] is None
