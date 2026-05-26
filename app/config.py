import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

WORKDIR = ROOT_DIR / "workdir"
UPLOADS_DIR = WORKDIR / "uploads"
OUTPUTS_DIR = WORKDIR / "outputs"
CHROMA_DIR = WORKDIR / "chroma"   # used in M4

for _d in (UPLOADS_DIR, OUTPUTS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama inference settings
# All values are overridable via environment variables for local flexibility.
# ---------------------------------------------------------------------------

# Base URL of the Ollama HTTP server.
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Default model name.  ollama_detector auto-selects from installed models at
# runtime; this fallback is used when the server is unreachable or no model
# is installed.
DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "llama3.1:8b")

# Per-request timeout in seconds (applies to ollama_client.generate calls).
DEFAULT_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# Default sampling temperature.  Individual callers may override per-request.
DEFAULT_TEMPERATURE: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.4"))

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------

MAX_UPLOAD_MB: int = 20
