"""Content-addressed disk cache for model calls.

A full §4 reproduction makes on the order of 10^5 model calls (2,917 scenarios x up to 10 horizons for
extraction alone, plus 117,178 questions for generation and judging). Without a cache, every debugging
pass re-bills the entire run. The key is a hash of the request payload, so the cache is safe to share
across processes and never returns a stale answer for different inputs.

Nothing here is specified by the paper; it is operational infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_CACHE_VERSION = "v1"


def cache_key(
    *,
    kind: str,
    model: str,
    payload: Mapping[str, Any],
    temperature: float,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build a stable hex key for one model request.

    Args:
        kind: Call family, e.g. ``"chat"`` or ``"embed"``.
        model: The model identifier.
        payload: The request body (messages, input texts, schema, ...). Must be JSON-serialisable.
        temperature: Sampling temperature — part of the key so a cached 0.0 answer is never reused for
            a 0.7 request.
        extra: Any additional key-forming state.

    Returns:
        A hex digest. ``sort_keys=True`` makes the key insensitive to dict ordering, which matters
        because the schema dict is built in different orders on different code paths.
    """
    blob = json.dumps(
        {
            "version": _CACHE_VERSION,
            "kind": kind,
            "model": model,
            "temperature": temperature,
            "payload": payload,
            "extra": extra or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """A tiny JSON-file cache, sharded two levels deep to keep directory sizes sane.

    Writes are atomic (write to a temp file in the same directory, then ``os.replace``) so a crashed run
    cannot leave a half-written entry that a later run would read as valid.
    """

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` on a miss or an unreadable entry."""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A corrupt entry is treated as a miss. Deleting it here would race with other readers.
            return None

    def set(self, key: str, value: Any) -> None:
        """Store a value under ``key`` atomically."""
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"value": value}, fh, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def stats(self) -> dict[str, int]:
        """Number of cached entries and total bytes on disk, for run manifests."""
        if not self.enabled or not self.root.exists():
            return {"entries": 0, "bytes": 0}
        entries = 0
        total = 0
        for p in self.root.rglob("*.json"):
            entries += 1
            total += p.stat().st_size
        return {"entries": entries, "bytes": total}


def messages_key(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Normalise a message list into a JSON-stable form for hashing."""
    return [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages]
