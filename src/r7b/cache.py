"""Content-addressed disk cache: every API response is written once and reused forever."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def request_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class DiskCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / key[:2] / f"{key}.json"

    def get(self, kind: str, payload: dict) -> dict | None:
        path = self._path(kind, request_key(payload))
        if not path.exists():
            return None
        return json.loads(path.read_text())["response"]

    def put(self, kind: str, payload: dict, response: dict) -> None:
        path = self._path(kind, request_key(payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": payload, "response": response},
                                  ensure_ascii=False, indent=1))
        tmp.replace(path)

    def count(self, kind: str) -> int:
        base = self.root / kind
        return sum(1 for _ in base.rglob("*.json")) if base.exists() else 0
