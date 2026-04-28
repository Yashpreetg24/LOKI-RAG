"""Configuration loaded from environment variables via python-dotenv."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration with defaults for every value."""

    # Environment mode
    IS_HOSTED: bool = os.getenv("LOKI_HOSTED_MODE", "").lower() in ("1", "true", "yes")

    # Ollama (local LLM)
    OLLAMA_BASE_URL: str = os.getenv("LOKI_LOCAL_NODE", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("LOKI_LOCAL_BRAIN", "gemma4-e2b")

    # Groq (hosted LLM)
    GROQ_API_KEY: str = os.getenv("LOKI_VAULT_TOKEN", "")
    GROQ_MODEL: str = os.getenv("LOKI_CLOUD_BRAIN", "llama-3.1-8b-instant")

    # Pinecone (hosted vector store)
    PINECONE_API_KEY: str = os.getenv("LOKI_VECTOR_KEY", "")
    PINECONE_INDEX: str = os.getenv("LOKI_VECTOR_INDEX", "rag-terminal")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "LOKI_EMBED_CORE", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("LOKI_LOCAL_VAULT", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("LOKI_CELL_ID", "documents")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("LOKI_SLICE_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("LOKI_SLICE_OVERLAP", 50))

    # Flask
    FLASK_HOST: str = os.getenv("LOKI_NODE_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("LOKI_NODE_PORT", 5001))
    FLASK_DEBUG: bool = os.getenv("LOKI_DEBUG_MODE", "true").lower() == "true"

    # Upload
    UPLOAD_FOLDER: str = os.getenv("LOKI_DATA_STREAM", "./uploads")
    MAX_CONTENT_LENGTH: int = int(os.getenv("LOKI_MAX_PAYLOAD", 16777216))
    ALLOWED_EXTENSIONS: set = {
        ext.strip().lower()
        for ext in os.getenv("LOKI_ALLOWED_TYPES", "pdf,txt,md").split(",")
        if ext.strip()
    }
