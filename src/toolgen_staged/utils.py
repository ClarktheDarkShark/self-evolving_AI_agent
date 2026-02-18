from __future__ import annotations

import hashlib
import re
from typing import Any, Optional


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    fence = re.search(r"```(?:python|py|text)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def extract_marked_block(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("###TOOL_START")
    if start < 0:
        return None
    end = text.find("###TOOL_END", start + len("###TOOL_START"))
    if end < 0:
        return None
    return text[start:end + len("###TOOL_END")].strip()


def ensure_markers(code: str) -> str:
    if "###TOOL_START" in code and "###TOOL_END" in code:
        return code.strip()
    return "###TOOL_START\n" + code.strip() + "\n###TOOL_END"


def summarize_text(text: str, max_len: int = 160) -> dict[str, Any]:
    safe = "" if text is None else str(text)
    return {
        "len": len(safe),
        "sha1": hashlib.sha1(safe.encode("utf-8")).hexdigest(),
        "preview": safe if len(safe) <= max_len else safe[: max_len - 3] + "...",
    }
