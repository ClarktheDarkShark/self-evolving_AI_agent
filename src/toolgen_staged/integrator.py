from __future__ import annotations

from typing import Mapping


_ANCHORS = {
    "phase2": "# === PHASE2_INSERT_TRACE_NORMALIZATION ===",
    "phase3": "# === PHASE3_INSERT_NEXT_ACTION ===",
    "phase4": "# === PHASE4_INSERT_ANSWER_GATES ===",
    "phase5": "# === PHASE5_INSERT_SELF_TESTS ===",
}


def integrate(parts: Mapping[str, str]) -> str:
    base = (parts.get("phase1") or "").strip()
    if not base:
        return ""
    merged = base
    for key, anchor in _ANCHORS.items():
        snippet = (parts.get(key) or "").rstrip()
        if snippet:
            merged = merged.replace(anchor, snippet)
        else:
            merged = merged.replace(anchor, "")
    return merged.strip()
