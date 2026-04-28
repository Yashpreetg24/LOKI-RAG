"""Hybrid Embedding generation (Hugging Face API for Cloud, Local for Dev)."""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# Module-level singleton for local mode
_model = None

# Standard Hugging Face Inference API endpoint
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def _get_local_model():
    """Load the local PyTorch model only if we aren't using the cloud API."""
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

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using a smart Hybrid approach:
    1. If HF_TOKEN is in environment -> Route to Hugging Face API (Production Mode)
    2. If no token AND not in production -> Load model into local memory (Local Dev Mode)
    """
    hf_token = os.environ.get("HF_TOKEN")
    is_production = os.environ.get("PRODUCTION") == "1" or os.environ.get("RENDER") == "1"

    if hf_token:
        # ==========================================
        # CLOUD MODE (Render / Production)
        # ==========================================
        logger.info("HF_TOKEN found. Routing embeddings to Hugging Face API...")
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error("Hugging Face API Failed: %s", e)
            raise RuntimeError(f"Cloud embedding generation failed: {e}")

    elif is_production:
        # ==========================================
        # PRODUCTION FAILSAFE
        # ==========================================
        # Never try to load local models on Render/Free tiers.
        logger.error("No HF_TOKEN found in Production environment!")
        raise RuntimeError(
            "HF_TOKEN is missing. Local model loading is disabled in production to prevent crashes. "
            "Please add HF_TOKEN to your environment variables."
        )

    else:
        # ==========================================
        # LOCAL MODE (Development)
        # ==========================================
        logger.info("No HF_TOKEN found. Processing embeddings locally...")
        model = _get_local_model()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]