from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ToolCallRecord:
    tool_name: str
    success: bool
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GeneratedToolRecord:
    tool_name: str
    fingerprint: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AgentResult:
    final_answer: str
    termination_reason: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    generated_tools: list[GeneratedToolRecord] = field(default_factory=list)
    steps: int = 0
    raw_agent_output: str | None = None
    metadata: Mapping[str, Any] | None = None
