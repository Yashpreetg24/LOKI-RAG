"""Intelligent caching layer for the RAG pipeline.

Implements two memoization caches inspired by dynamic programming's
overlapping-subproblems optimisation:

1. **Embedding Cache** — Avoids re-computing embeddings for text that has
   already been seen (same document re-uploaded, same question asked twice).
   TTL: 1 hour.  Max entries: 500.

2. **Query Result Cache** — Caches vector-search results keyed by the
   embedding hash.  If the same question is asked within the TTL window,
   the Pinecone / ChromaDB round-trip is skipped entirely.
   TTL: 5 minutes (documents can change).  Max entries: 200.

Both caches use an LRU eviction policy to bound memory on free-tier hosts.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with per-entry TTL expiration.

    Args:
        max_size: Maximum number of entries before the oldest is evicted.
        ttl: Time-to-live in seconds for each entry.
        name: Human-readable name for log messages.
    """

    def __init__(self, max_size: int, ttl: int, name: str = "LRUCache"):
        self._max_size = max_size
        self._ttl = ttl
        self._name = name
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str):
        """Return the cached value or None if missing / expired."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            ts, value = self._store[key]
            if time.time() - ts > self._ttl:
                # Expired — evict
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most-recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("[CACHE HIT] %s — key=%s…", self._name, key[:16])
            return value

    def put(self, key: str, value: object) -> None:
        """Insert or update an entry, evicting the oldest if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), value)

            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("[CACHE EVICT] %s — key=%s…", self._name, evicted_key[:16])

    def invalidate(self, key: str) -> None:
        """Remove a specific entry."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush all entries."""
        with self._lock:
            self._store.clear()
            logger.info("[CACHE CLEAR] %s", self._name)

    @property
    def stats(self) -> dict:
        return {
            "name": self._name,
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1) * 100, 1),
        }


# ── Global cache instances ────────────────────────────────────────────────────

# Embedding cache: text hash → vector list
# TTL 1 hour — embeddings don't change for the same text
embedding_cache = LRUCache(max_size=500, ttl=3600, name="EmbeddingCache")

# Query result cache: embedding hash → search results
# TTL 5 minutes — documents can be added/deleted
query_cache = LRUCache(max_size=200, ttl=300, name="QueryCache")


# ── Helpers ───────────────────────────────────────────────────────────────────

def text_hash(text: str) -> str:
    """Produce a deterministic hash for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def texts_hash(texts: list[str]) -> str:
    """Produce a deterministic hash for a list of text strings."""
    combined = "||".join(texts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def vector_hash(vector: list[float]) -> str:
    """Produce a hash for an embedding vector (for query cache keys)."""
    # Round to 6 decimal places for stability
    rounded = ",".join(f"{v:.6f}" for v in vector)
    return hashlib.sha256(rounded.encode("utf-8")).hexdigest()
