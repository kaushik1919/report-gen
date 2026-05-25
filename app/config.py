from pathlib import Path

# Project root (two levels up from this file: app/ -> report-gen/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Runtime working directories — created at import time, excluded from git
WORKDIR = ROOT_DIR / "workdir"
UPLOADS_DIR = WORKDIR / "uploads"
OUTPUTS_DIR = WORKDIR / "outputs"
CHROMA_DIR = WORKDIR / "chroma"   # used in M4

for _d in (UPLOADS_DIR, OUTPUTS_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# LLM settings — not used until M2
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"

# Limits
MAX_UPLOAD_MB = 20
