from __future__ import annotations

from typing import Callable, Optional

from src.typings import ChatHistory
from src.self_evolving_agent.tool_registry import ToolMetadata

from .base import ToolgenPipeline


class BaselineToolgenPipeline(ToolgenPipeline):
    def __init__(
        self,
        *,
        build_user_prompt: Callable[[str, ChatHistory], str],
        invoke_toolgen: Callable[[str, str, ChatHistory, str], Optional[ToolMetadata]],
        system_prompt_selector: Callable[[str, str | None], str],
        name_prefix: str,
    ) -> None:
        self._build_user_prompt = build_user_prompt
        self._invoke_toolgen = invoke_toolgen
        self._system_prompt_selector = system_prompt_selector
        self._name_prefix = name_prefix

    def maybe_generate_tools(
        self, env_name: str, task_payload: str, chat_history: ChatHistory
    ) -> Optional[list[ToolMetadata]]:
        user_prompt = self._build_user_prompt(task_payload, chat_history)
        system_prompt = self._system_prompt_selector("baseline", env_name)
        tool = self._invoke_toolgen(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            name_prefix=self._name_prefix,
        )
        return [tool] if tool else None
