from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping


def toolgen_debug_enabled() -> bool:
    value = os.environ.get("TOOLGEN_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ToolgenDebugLogger:
    def __init__(
        self,
        log_path: Path,
        *,
        enabled: bool,
        run_id: str,
        environment: str,
        max_field_len: int = 2000,
    ) -> None:
        self._enabled = bool(enabled)
        self._log_path = Path(log_path)
        self._run_id = run_id
        self._environment = environment
        self._max_field_len = max_field_len

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_registry_event(self, payload: Mapping[str, Any]) -> None:
        self.log_event("registry_" + str(payload.get("event", "unknown")), **payload)

    def log_event(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "event": event,
            "run_id": self._run_id,
            "session_id": self._run_id,
            "environment": self._environment,
        }
        payload.update(fields)
        payload = self._sanitize(payload)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception:
            return

    def _sanitize(self, data: Mapping[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            cleaned.update(self._sanitize_field(key, value))
        return cleaned

    def _sanitize_field(self, key: str, value: Any) -> dict[str, Any]:
        if value is None or isinstance(value, (int, float, bool)):
            return {key: value}
        if isinstance(value, str):
            return {key: self._truncate(key, value)}
        if isinstance(value, Mapping):
            repr_value = repr(value)
            if len(repr_value) <= self._max_field_len:
                return {key: value}
            return {
                f"{key}_preview": repr_value[: self._max_field_len - 3] + "...",
                f"{key}_len": len(repr_value),
                f"{key}_truncated": True,
            }
        if isinstance(value, (list, tuple)):
            repr_value = repr(value)
            if len(repr_value) <= self._max_field_len:
                return {key: list(value)}
            return {
                f"{key}_preview": repr_value[: self._max_field_len - 3] + "...",
                f"{key}_len": len(repr_value),
                f"{key}_truncated": True,
            }
        return {key: self._truncate(key, repr(value))}

    def _truncate(self, key: str, text: str) -> Any:
        if len(text) <= self._max_field_len:
            return text
        return {
            f"{key}_preview": text[: self._max_field_len - 3] + "...",
            f"{key}_len": len(text),
            f"{key}_truncated": True,
        }
