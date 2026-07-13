from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCache:
    """Development-time JSON-file cache for LLM calls, keyed by a hash of the
    exact prompt inputs. Avoids repeat API calls (and cost/rate-limit churn)
    while iterating locally or running deterministic tests."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("LLM cache at %s is corrupt; starting fresh", path)
                self._data = {}

    @staticmethod
    def make_key(model: str, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        payload = f"{model}|{json_mode}|{system_prompt}|{user_prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
