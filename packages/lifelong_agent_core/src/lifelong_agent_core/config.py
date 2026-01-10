from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class AgentConfig:
    model_adapter: Any | None = None
    tool_registry_path: str = "outputs/tool_library"
    system_prompt: str = ""
    inference_config: Optional[Mapping[str, Any]] = None
    max_generated_tools_per_run: int = 3
    force_tool_generation_if_missing: bool = True
    tool_match_min_score: float = 0.25
    include_registry_in_prompt: bool = True
    environment_label: str = "unknown"
    retrieval_top_k: int = 5
    reuse_top_k: int = 3
    reuse_similarity_threshold: Optional[float] = None
    reuse_min_reliability: float = 0.0
    canonical_tool_naming: bool = True
    max_steps: int = 6
    event_listener: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)
