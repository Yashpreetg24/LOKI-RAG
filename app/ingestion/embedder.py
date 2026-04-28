"""Hybrid embedding generation with caching and multi-key failover.

Features:
- **Embedding Cache**: Hash-based memoization avoids redundant API calls
  for previously seen text (DP overlapping-subproblems optimisation).
- **HF Token Failover**: If one HuggingFace token fails, automatically
  rotates to the next available token before falling back to local model.
- **Resilient Fallback**: Cloud API → next token → tokenless API → local model.
"""

from __future__ import annotations

import os
import logging
import requests

from app.cache import embedding_cache, texts_hash

logger = logging.getLogger(__name__)

# Module-level singleton for local mode
_model = None

# Standard Hugging Face Inference API endpoint
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def _get_hf_key_manager():
    """Lazy import to avoid circular imports at module load time."""
    from app.config import hf_key_manager
    return hf_key_manager


def _get_local_model():
    """Load the local sentence-transformers model on demand."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local all-MiniLM-L6-v2 model into memory...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed locally. "
                "Run `pip install sentence-transformers` if you want to run this locally without an API key."
            )
    return _model


def _local_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings with the bundled local model."""
    logger.info("Processing embeddings locally with sentence-transformers...")
    model = _get_local_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def _hf_api_embeddings(texts: list[str], token: str) -> list[list[float]] | None:
    """Try to get embeddings from HuggingFace Inference API with a specific token.

    Returns:
        list of vectors on success, or None on failure.
    """
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code in (401, 403):
            logger.warning(
                "HF token ...%s rejected (%s).",
                token[-6:], response.status_code,
            )
            return None

        if response.status_code == 429:
            logger.warning("HF token ...%s rate-limited.", token[-6:])
            return None

        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            logger.warning("HF API error payload: %s", data.get("error"))
            return None

        return data

    except requests.exceptions.RequestException as exc:
        logger.warning("HF API request failed with token ...%s: %s", token[-6:], exc)
        return None


def _hf_api_embeddings_no_auth(texts: list[str]) -> list[list[float]] | None:
    """Try HuggingFace Inference API without any auth token.

    Public models like all-MiniLM-L6-v2 are accessible without a token,
    but with lower rate limits. This is a last-resort fallback for production.

    Returns:
        list of vectors on success, or None on failure.
    """
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    try:
        response = requests.post(API_URL, json=payload, timeout=60)

        if response.status_code == 429:
            logger.warning("Tokenless HF API rate-limited.")
            return None

        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            logger.warning("Tokenless HF API error: %s", data.get("error"))
            return None

        logger.info("Tokenless HF API succeeded for %d text(s).", len(texts))
        return data

    except requests.exceptions.RequestException as exc:
        logger.warning("Tokenless HF API request failed: %s", exc)
        return None


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using caching + multi-key failover.

    Pipeline:
    1. Check the embedding cache (hash of input texts).
    2. If HF tokens exist, try each token in rotation until one succeeds.
    3. Fall back to local sentence-transformers model.

    The result is always cached for future lookups.
    """
    # ── 1. Check cache ────────────────────────────────────────────────────────
    cache_key = texts_hash(texts)
    cached = embedding_cache.get(cache_key)
    if cached is not None:
        logger.info("[CACHE HIT] Embeddings for %d text(s) returned from cache.", len(texts))
        return cached

    # ── 2. Try HuggingFace API with key failover ─────────────────────────────
    km = _get_hf_key_manager()

    if km.has_keys:
        # Try up to key_count times (one attempt per available key)
        for attempt in range(km.key_count):
            token = km.get_key()
            if not token:
                break

            logger.info(
                "HF embedding attempt %d/%d with token ...%s",
                attempt + 1, km.key_count, token[-6:],
            )
            result = _hf_api_embeddings(texts, token)

            if result is not None:
                km.mark_success(token)
                embedding_cache.put(cache_key, result)
                logger.info(
                    "[CACHE STORE] Cached embeddings for %d text(s).", len(texts)
                )
                return result
            else:
                km.mark_failed(token, reason="API error or auth failure")

        logger.warning("All HF tokens exhausted. Trying tokenless fallback...")

    # ── 3. Tokenless HF API fallback (public models work without auth) ────────
    logger.info("Attempting tokenless HF Inference API (public model, lower rate limits)...")
    result = _hf_api_embeddings_no_auth(texts)
    if result is not None:
        embedding_cache.put(cache_key, result)
        logger.info("[CACHE STORE] Cached tokenless embeddings for %d text(s).", len(texts))
        return result

    # ── 4. Local fallback (dev only — disabled in production) ─────────────────
    is_production = os.environ.get("PRODUCTION") == "1" or os.environ.get("RENDER") == "1"

    if is_production:
        logger.error("All embedding methods failed in production.")
        raise RuntimeError(
            "All embedding methods failed (tokens + tokenless API). "
            "Please try again shortly or check your HF_TOKEN(s)."
        )

    result = _local_embeddings(texts)
    embedding_cache.put(cache_key, result)
    logger.info("[CACHE STORE] Cached local embeddings for %d text(s).", len(texts))
    return result
