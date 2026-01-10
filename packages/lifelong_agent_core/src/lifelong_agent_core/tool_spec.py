from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass
class ToolSpec:
    name: str
    description: str
    signature: str
    code_lines: Sequence[str]
    tool_type: Optional[str] = None
    input_schema: Optional[Any] = None
    capabilities: Optional[Any] = None
    examples: Optional[Sequence[str]] = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ToolSpec":
        code_lines = payload.get("code_lines") or []
        if isinstance(code_lines, str):
            code_lines = code_lines.splitlines()
        return cls(
            name=str(payload.get("name") or "tool"),
            description=str(payload.get("description") or ""),
            signature=str(payload.get("signature") or "run()"),
            code_lines=code_lines,
            tool_type=payload.get("tool_type"),
            input_schema=payload.get("input_schema"),
            capabilities=payload.get("capabilities"),
            examples=payload.get("examples"),
        )
