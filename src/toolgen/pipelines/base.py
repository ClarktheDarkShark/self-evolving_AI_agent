from __future__ import annotations

from typing import Optional

from src.typings import ChatHistory
from src.self_evolving_agent.tool_registry import ToolMetadata


class ToolgenPipeline:
    def maybe_generate_tools(
        self, env_name: str, task_payload: str, chat_history: ChatHistory
    ) -> Optional[list[ToolMetadata]]:
        raise NotImplementedError

    def should_fallback(self, env_name: str) -> bool:
        return False

    def force_bootstrap(self, env_name: str) -> None:
        return None
