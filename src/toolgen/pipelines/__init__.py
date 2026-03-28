from __future__ import annotations

from typing import Callable, Optional

from src.typings import ChatHistory
from src.self_evolving_agent.tool_registry import ToolMetadata

from .baseline import BaselineToolgenPipeline
from .base import ToolgenPipeline


def build_toolgen_pipeline(
    *,
    pipeline: str,
    agg_n: int,
    build_user_prompt: Callable[[str, ChatHistory], str],
    compact_task: Callable[[str], str],
    invoke_toolgen: Callable[[str, str, ChatHistory, str], Optional[ToolMetadata]],
    system_prompt_selector: Callable[[str, str | None], str],
    name_prefix: str,
) -> ToolgenPipeline:
    return BaselineToolgenPipeline(
        build_user_prompt=build_user_prompt,
        invoke_toolgen=invoke_toolgen,
        system_prompt_selector=system_prompt_selector,
        name_prefix=name_prefix,
    )


__all__ = [
    "ToolgenPipeline",
    "build_toolgen_pipeline",
]
