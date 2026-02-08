from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def get_state_path(state_dir: str, run_id: str) -> str:
    safe_run_id = (run_id or "default").replace(os.sep, "_")
    return os.path.join(state_dir, f"{safe_run_id}.json")


def load_state(state_path: str) -> dict[str, Any]:
    if not state_path or not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state_atomic(state_path: str, state_dict: dict[str, Any]) -> None:
    state_dir = os.path.dirname(state_path) or "."
    os.makedirs(state_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="state_", suffix=".json", dir=state_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state_dict, handle, ensure_ascii=True, default=str)
        os.replace(tmp_path, state_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
