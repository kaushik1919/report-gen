"""
Cross-platform Ollama installation and model detection.

Responsibilities
----------------
- Locate the Ollama executable on Windows / macOS / Linux
- Verify the Ollama server is reachable
- Discover installed models and select a default using priority rules
- Emit structured startup diagnostics (called once at API boot)

No inference logic here.  Uses ollama_client for HTTP checks.
"""
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

from app.config import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Priority order for automatic model selection.
# First match in the installed list wins.
MODEL_PRIORITY = [
    "qwen2.5:14b",
    "qwen2.5:7b",
    "llama3.1:8b",
    "llama3",
]


# ---------------------------------------------------------------------------
# Executable detection
# ---------------------------------------------------------------------------

def detect_executable() -> str | None:
    """
    Return the path to the Ollama executable, or None if not found.

    Search order:
      1. PATH lookup (works on all platforms)
      2. Platform-specific well-known install locations
    """
    found = shutil.which("ollama")
    if found:
        return found

    system = platform.system()
    if system == "Windows":
        candidates = [
            # Use LOCALAPPDATA env var so we never hardcode a username.
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
        ]
    elif system == "Darwin":
        candidates = [
            Path("/usr/local/bin/ollama"),
            Path("/opt/homebrew/bin/ollama"),
        ]
    else:  # Linux and others
        candidates = [
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def detect_version(executable: str) -> str | None:
    """Run ``ollama --version`` and return the trimmed output, or None on failure."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_default_model(installed: list[str]) -> str | None:
    """
    Pick the best available model using MODEL_PRIORITY order.

    Falls back to the first installed model if no priority match exists.
    Returns None when *installed* is empty.
    """
    if not installed:
        return None
    for preferred in MODEL_PRIORITY:
        if preferred in installed:
            return preferred
    return installed[0]


# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------

def startup_diagnostics() -> dict:
    """
    Collect and log Ollama environment info.  Always returns a dict —
    never raises.  Call once at application startup for observability.

    Return keys:
        executable, version, server_reachable,
        installed_models, selected_model, server_url
    """
    # Deferred import avoids circular dependency at module load time.
    from core.ollama_client import check_connectivity, list_models

    exe = detect_executable()
    version = detect_version(exe) if exe else None
    reachable = check_connectivity()
    installed = list_models() if reachable else []
    model = select_default_model(installed)

    info: dict = {
        "executable": exe,
        "version": version,
        "server_reachable": reachable,
        "installed_models": installed,
        "selected_model": model,
        "server_url": OLLAMA_BASE_URL,
    }

    if exe:
        logger.info("Ollama executable  : %s", exe)
    else:
        logger.warning(
            "Ollama executable not found in PATH or common install locations. "
            "Download from https://ollama.com"
        )

    if version:
        logger.info("Ollama version     : %s", version)

    if reachable:
        logger.info("Ollama server      : reachable at %s", OLLAMA_BASE_URL)
        logger.info("Installed models   : %s", installed or "(none)")
        if model:
            logger.info("Selected model     : %s", model)
        else:
            logger.warning("No installed models found. Run: ollama pull llama3.1:8b")
    else:
        logger.warning(
            "Ollama server not reachable at %s. Start with: ollama serve",
            OLLAMA_BASE_URL,
        )

    return info
