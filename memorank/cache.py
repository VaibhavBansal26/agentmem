import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("MEMORANK_CACHE", ".cache"))


def key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def cached(namespace: str, parts, fn):
    """Return fn() the first time; afterwards read the stored JSON value."""
    path = CACHE_DIR / namespace / f"{key(*parts)}.json"
    if path.exists():
        return json.loads(path.read_text())
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value
