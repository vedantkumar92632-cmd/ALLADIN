import time
from typing import Any, Dict, Optional, Tuple

class Cache:
    """
    In-memory state store.
    
    NOTE: Redis is the intended backing store later (per Section 3 Part 5).
    We start with a plain dict-backed version to avoid adding a Redis dependency
    prematurely before data flows through the system.
    """
    def __init__(self):
        # Store is a dict of key -> (value, expiry_timestamp)
        self._store: Dict[str, Tuple[Any, Optional[float]]] = {}

    def set(self, key: str, value: Any, expire_seconds: Optional[int] = None):
        """Set a value in the cache with an optional expiration in seconds."""
        expiry = time.time() + expire_seconds if expire_seconds is not None else None
        self._store[key] = (value, expiry)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache. Returns None if expired or not found."""
        if key not in self._store:
            return None
            
        value, expiry = self._store[key]
        if expiry is not None and time.time() > expiry:
            del self._store[key]
            return None
            
        return value

    def delete(self, key: str) -> bool:
        """Delete a key from cache. Returns True if the key was present and deleted."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache and is not expired."""
        if key not in self._store:
            return False
        _, expiry = self._store[key]
        if expiry is not None and time.time() > expiry:
            return False
        return True

    def clear(self):
        """Clear all entries from the cache."""
        self._store.clear()

# Global cache instance
cache = Cache()
