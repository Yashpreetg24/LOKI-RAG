"""API key manager with round-robin failover and cooldown recovery.

Supports multiple API keys per service (Groq, HuggingFace, etc.).
Keys are provided as comma-separated values in environment variables.

On failure (401, 429, timeout), the current key is placed on a cooldown
timer and the manager rotates to the next available key.  After the
cooldown expires, the key is retried automatically.

Usage::

    groq_keys = KeyManager("GROQ", ["key1", "key2", "key3"])
    key = groq_keys.get_key()         # returns the best available key
    groq_keys.mark_failed("key1")     # puts key1 on 60s cooldown
    key = groq_keys.get_key()         # returns key2
"""

import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN = 60  # seconds before retrying a failed key


class KeyManager:
    """Thread-safe API key rotation manager.

    Args:
        service_name: Human-readable name (e.g. "Groq", "HuggingFace").
        keys: List of API keys to rotate through.
        cooldown: Seconds to wait before retrying a failed key.
    """

    def __init__(self, service_name: str, keys: list[str], cooldown: int = DEFAULT_COOLDOWN):
        self._service = service_name
        self._keys = [k.strip() for k in keys if k.strip()]
        self._cooldown = cooldown
        self._lock = Lock()

        # Track failure state: key → timestamp when cooldown expires
        self._failed: dict[str, float] = {}

        # Current index for round-robin
        self._index = 0

        if self._keys:
            logger.info(
                "%s KeyManager initialised with %d key(s).",
                self._service, len(self._keys),
            )
        else:
            logger.warning("%s KeyManager: no keys provided.", self._service)

    @property
    def has_keys(self) -> bool:
        """Return True if at least one key was provided."""
        return bool(self._keys)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get_key(self) -> str:
        """Return the next available (non-cooled-down) key.

        Tries all keys in round-robin order. If all keys are on cooldown,
        returns the one whose cooldown expires soonest (best effort).

        Returns:
            str: An API key, or empty string if no keys exist.
        """
        if not self._keys:
            return ""

        with self._lock:
            now = time.time()

            # Clean up expired cooldowns
            expired = [k for k, t in self._failed.items() if now >= t]
            for k in expired:
                del self._failed[k]
                logger.info(
                    "%s key ...%s recovered from cooldown.",
                    self._service, k[-6:],
                )

            # Try to find a healthy key starting from current index
            for _ in range(len(self._keys)):
                candidate = self._keys[self._index]
                self._index = (self._index + 1) % len(self._keys)

                if candidate not in self._failed:
                    return candidate

            # All keys are on cooldown — return the one expiring soonest
            logger.warning(
                "%s: all %d key(s) are on cooldown. Using least-recently-failed key.",
                self._service, len(self._keys),
            )
            soonest_key = min(self._failed, key=self._failed.get)
            del self._failed[soonest_key]
            return soonest_key

    def mark_failed(self, key: str, reason: str = "") -> None:
        """Place a key on cooldown after a failure.

        Args:
            key: The API key that failed.
            reason: Optional description of the failure.
        """
        with self._lock:
            self._failed[key] = time.time() + self._cooldown
            masked = f"...{key[-6:]}" if len(key) > 6 else "***"
            logger.warning(
                "%s key %s failed%s — on cooldown for %ds. %d key(s) remaining.",
                self._service,
                masked,
                f" ({reason})" if reason else "",
                self._cooldown,
                len(self._keys) - len(self._failed),
            )

    def mark_success(self, key: str) -> None:
        """Remove a key from cooldown if it was previously failed.

        Call this after a successful request to ensure recovered keys
        stay in the rotation.
        """
        with self._lock:
            if key in self._failed:
                del self._failed[key]
                logger.info(
                    "%s key ...%s recovered after successful request.",
                    self._service, key[-6:],
                )

    @property
    def stats(self) -> dict:
        now = time.time()
        return {
            "service": self._service,
            "total_keys": len(self._keys),
            "healthy_keys": len(self._keys) - len(self._failed),
            "cooling_down": {
                f"...{k[-6:]}": round(t - now, 1)
                for k, t in self._failed.items()
                if t > now
            },
        }


# ── Helpers for parsing comma-separated keys from env ─────────────────────────

def parse_keys(env_value: str) -> list[str]:
    """Split a comma-separated env var into a list of trimmed, non-empty keys.

    Args:
        env_value: Raw environment variable value (e.g. "key1,key2,key3").

    Returns:
        list[str]: List of individual API keys.
    """
    if not env_value:
        return []
    return [k.strip() for k in env_value.split(",") if k.strip()]
