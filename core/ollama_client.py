"""
Thin inference wrapper around the Ollama Python library.

Responsibilities
----------------
- Centralise all Ollama HTTP calls
- Map every failure to OllamaClientError with structured log output
- Expose connectivity checks and model discovery
- Own NO business logic, NO prompt logic, NO retry loops

Callers (outline_planner, content_generator) own retry behaviour.
"""
import logging
import time

import ollama

from app.config import DEFAULT_MODEL, DEFAULT_TIMEOUT, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Raised when any Ollama call fails."""


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    prompt_type: str = "unknown",
) -> str:
    """
    Call Ollama chat in JSON mode and return the raw content string.

    Raises OllamaClientError on any failure.  No retries — callers own that.
    """
    t0 = time.monotonic()
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": temperature},
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.error(
            "Ollama call failed | type=%s model=%s elapsed_ms=%d error=%r",
            prompt_type, model, elapsed_ms, str(exc),
        )
        raise OllamaClientError(f"Ollama request failed: {exc}") from exc

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    content = response.message.content
    logger.info(
        "Ollama response | type=%s model=%s elapsed_ms=%d chars=%d",
        prompt_type, model, elapsed_ms, len(content),
    )
    return content


# ---------------------------------------------------------------------------
# Connectivity and discovery
# ---------------------------------------------------------------------------

def check_connectivity() -> bool:
    """Return True if the Ollama server answers a list request."""
    try:
        ollama.Client(host=OLLAMA_BASE_URL).list()
        return True
    except Exception:
        return False


def list_models() -> list[str]:
    """Return installed model name strings.  Returns [] on any failure."""
    try:
        result = ollama.Client(host=OLLAMA_BASE_URL).list()
        return [m.model for m in (result.models or [])]
    except Exception:
        return []


def model_exists(name: str) -> bool:
    """Return True if *name* appears in the installed model list."""
    return name in list_models()


def server_version() -> str | None:
    """Return server version metadata if the library exposes it, else None."""
    try:
        result = ollama.Client(host=OLLAMA_BASE_URL).list()
        return getattr(result, "version", None)
    except Exception:
        return None


# Expose the configured timeout so callers can reference it without
# re-importing config directly.
TIMEOUT = DEFAULT_TIMEOUT
