"""Unified LLM router with multi-key Groq failover.

Local  (IS_HOSTED=False) : Ollama → Groq fallback (plain requests)
Hosted (IS_HOSTED=True)  : Groq via langchain_groq.ChatGroq (no Ollama check)

Features:
- Backend detection cached for 30s to avoid repeated health-checks.
- Multi-key Groq failover: if one key fails (401/429), rotates to the next.
"""

import logging
import os
import time
from typing import Generator

logger = logging.getLogger(__name__)

# ── Backend cache ─────────────────────────────────────────────────────────────
_cache: dict = {"backend": None, "model": "", "ts": 0.0}
_CACHE_TTL = 30  # seconds


def _get_groq_key_manager():
    """Lazy import to avoid circular imports at module load time."""
    from app.config import groq_key_manager
    return groq_key_manager


def _format_groq_error(exc: Exception) -> str:
    """Return a user-facing Groq error string with auth failures called out."""
    message = str(exc)
    lowered = message.lower()
    if "401" in message or "unauthorized" in lowered or "authentication" in lowered:
        return "[ERROR: Groq rejected the API key. Check GROQ_API_KEY in your environment.]"
    if "429" in message or "rate limit" in lowered:
        return "[ERROR: Groq rate limit exceeded. Try again in a moment.]"
    return f"[ERROR: Groq LLM error — {exc}]"


def _is_hosted() -> bool:
    try:
        from flask import current_app
        return current_app.config.get("IS_HOSTED", False)
    except RuntimeError:
        import os
        return os.getenv("IS_HOSTED", "").lower() in ("1", "true", "yes")


def _ollama_model() -> str:
    try:
        from flask import current_app
        return current_app.config.get("OLLAMA_MODEL", "gemma4-e2b")
    except RuntimeError:
        import os
        return os.getenv("OLLAMA_MODEL", "gemma4-e2b")


def _groq_model() -> str:
    try:
        from flask import current_app
        return current_app.config.get("GROQ_MODEL", "llama-3.1-8b-instant")
    except RuntimeError:
        import os
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _groq_api_key() -> str:
    """Get the best available Groq API key from the KeyManager."""
    km = _get_groq_key_manager()
    if km.has_keys:
        return km.get_key()
    # Fallback to single key from config
    try:
        from flask import current_app
        return current_app.config.get("GROQ_API_KEY", "")
    except RuntimeError:
        import os
        return os.getenv("GROQ_API_KEY", "")


def resolve_backend() -> tuple[str, str]:
    """Return (backend, model). Uses cache to avoid repeated health-checks.

    Hosted mode  → always "groq" (langchain_groq path)
    Local mode   → tries Ollama, falls back to plain Groq client, then "none"

    Returns:
        tuple[str, str]: (backend_name, model_name)
            backend_name: "ollama" | "groq" | "groq_langchain" | "none"
    """
    now = time.time()
    if _cache["backend"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["backend"], _cache["model"]

    if _is_hosted() or os.getenv("FORCE_GROQ", "").lower() in ("1", "true", "yes"):
        result = ("groq_langchain", _groq_model())
        logger.info("LLM backend: Groq (via langchain-groq) [forced/hosted]")
    else:
        # Local: try Ollama first, then fallback to Groq
        from app.rag.ollama_client import check_connection as ollama_up
        if ollama_up():
            result = ("ollama", _ollama_model())
            logger.info("LLM backend: Ollama (%s)", result[1])
        else:
            km = _get_groq_key_manager()
            if km.has_keys:
                result = ("groq_langchain", _groq_model())
                logger.info("LLM backend: Groq (via langchain-groq) — Ollama unavailable")
            else:
                result = ("none", "")
                logger.warning(
                    "LLM backend: none — Ollama offline and no GROQ_API_KEY set"
                )

    _cache["backend"] = result[0]
    _cache["model"]   = result[1]
    _cache["ts"]      = now
    return result


def invalidate_cache() -> None:
    """Force re-detection on the next call (e.g. after config change)."""
    _cache["backend"] = None


def generate_stream(prompt: str) -> Generator[str, None, None]:
    """Route a prompt to the active backend and stream tokens.

    Args:
        prompt: The fully-formatted prompt string.

    Yields:
        str: Individual tokens from the LLM.
    """
    backend, model = resolve_backend()

    if backend == "ollama":
        from app.rag.ollama_client import generate_stream as _stream
        yield from _stream(prompt, model)

    elif backend == "groq_langchain":
        yield from _groq_langchain_stream_with_failover(prompt, model)

    else:
        yield (
            "[ERROR: No LLM available. "
            "Either start Ollama (ollama serve) or add GROQ_API_KEY to .env]"
        )


def _groq_langchain_stream_with_failover(prompt: str, model: str) -> Generator[str, None, None]:
    """Stream tokens from Groq with automatic key failover.

    Tries each available key until one succeeds or all are exhausted.
    """
    km = _get_groq_key_manager()

    if not km.has_keys:
        yield "[ERROR: GROQ_API_KEY is not set. Cannot use hosted LLM.]"
        return

    max_attempts = km.key_count
    last_error = ""

    for attempt in range(max_attempts):
        api_key = km.get_key()
        if not api_key:
            break

        logger.info(
            "Groq stream attempt %d/%d with key ...%s",
            attempt + 1, max_attempts, api_key[-6:],
        )

        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
        except ImportError:
            yield (
                "[ERROR: langchain_groq is not installed. "
                "Run: pip install langchain-groq]"
            )
            return

        try:
            llm = ChatGroq(
                api_key=api_key,
                model_name=model,
                streaming=True,
                temperature=0.7,
            )

            tokens = []
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                if chunk.content:
                    tokens.append(chunk.content)
                    yield chunk.content

            # If we got tokens, the key worked
            if tokens:
                km.mark_success(api_key)
                return

        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Groq key ...%s failed on attempt %d: %s",
                api_key[-6:], attempt + 1, exc,
            )
            km.mark_failed(api_key, reason=last_error)
            continue

    # All keys exhausted
    yield _format_groq_error(Exception(last_error or "All Groq API keys failed"))


def generate(prompt: str) -> str:
    """Route a prompt to the active backend and return the full response.

    Non-streaming variant used for quick internal tasks like query rewriting.

    Args:
        prompt: The fully-formatted prompt string.

    Returns:
        str: The complete response text, or empty string on error.
    """
    backend, model = resolve_backend()

    if backend == "ollama":
        from app.rag.ollama_client import generate as _gen
        return _gen(prompt, model)

    elif backend == "groq_langchain":
        return _groq_langchain_generate_with_failover(prompt, model)

    elif backend == "groq":
        from app.rag.groq_client import generate as _gen
        return _gen(prompt, model)

    else:
        return ""


def _groq_langchain_generate_with_failover(prompt: str, model: str) -> str:
    """Non-streaming response with key failover."""
    km = _get_groq_key_manager()

    if not km.has_keys:
        return ""

    for attempt in range(km.key_count):
        api_key = km.get_key()
        if not api_key:
            break

        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
        except ImportError:
            return ""

        try:
            llm = ChatGroq(
                api_key=api_key,
                model_name=model,
                streaming=False,
                temperature=0.3,
            )
            result = llm.invoke([HumanMessage(content=prompt)])
            km.mark_success(api_key)
            return result.content.strip() if result.content else ""

        except Exception as exc:
            logger.warning(
                "Groq generate key ...%s failed: %s",
                api_key[-6:], exc,
            )
            km.mark_failed(api_key, reason=str(exc))
            continue

    return ""
