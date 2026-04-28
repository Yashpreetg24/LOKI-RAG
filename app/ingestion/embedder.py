"""Hybrid embedding generation with resilient cloud/local fallback."""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# Module-level singleton for local mode
_model = None

# Standard Hugging Face Inference API endpoint
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using a resilient hybrid approach.

    Preference order:
    1. If HF_TOKEN is present, try Hugging Face's hosted inference API.
    2. If the token is missing or the API call fails/auths badly, fall back to
       the local sentence-transformers model.
    """
    hf_token = (os.environ.get("HF_TOKEN") or "").strip()

    if hf_token:
        logger.info("HF_TOKEN found. Attempting Hugging Face Inference API embeddings...")
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code in (401, 403):
                logger.warning(
                    "Hugging Face embedding request was rejected (%s). Falling back to local embeddings.",
                    response.status_code,
                )
                return _local_embeddings(texts)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                logger.warning(
                    "Hugging Face embedding API returned an error payload: %s. Falling back to local embeddings.",
                    data.get("error"),
                )
                return _local_embeddings(texts)
            return data

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Hugging Face embedding API request failed: %s. Falling back to local embeddings.",
                exc,
            )

    return _local_embeddings(texts)
