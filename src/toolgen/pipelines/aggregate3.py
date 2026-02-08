from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from src.typings import ChatHistory
from src.self_evolving_agent.tool_registry import ToolMetadata
from src.toolgen.prompting.build_task_pack import build_task_pack

from .base import ToolgenPipeline


@dataclass
class _TaskPayload:
    full_text: str
    compact_text: str


class Aggregate3ToolgenPipeline(ToolgenPipeline):
    def __init__(
        self,
        *,
        agg_n: int,
        compact_task: Callable[[str], str],
        invoke_toolgen: Callable[[str, str, ChatHistory, str], Optional[ToolMetadata]],
        system_prompt_selector: Callable[[str, str | None], str],
        name_prefix: str,
    ) -> None:
        self._agg_n = max(1, agg_n)
        self._compact_task = compact_task
        self._invoke_toolgen = invoke_toolgen
        self._system_prompt_selector = system_prompt_selector
        self._name_prefix = name_prefix
        self._buffers: dict[str, list[_TaskPayload]] = {}
        self._bootstrapped_envs: set[str] = set()

    def should_fallback(self, env_name: str) -> bool:
        return env_name in self._bootstrapped_envs

    def force_bootstrap(self, env_name: str) -> None:
        self._bootstrapped_envs.add(env_name)
        self._buffers.pop(env_name, None)

    def maybe_generate_tools(
        self, env_name: str, task_payload: str, chat_history: ChatHistory
    ) -> Optional[list[ToolMetadata]]:
        if env_name in self._bootstrapped_envs:
            return None
        buffer = self._buffers.setdefault(env_name, [])
        buffer.append(
            _TaskPayload(
                full_text=task_payload,
                compact_text=self._compact_task(task_payload),
            )
        )
        if len(buffer) < self._agg_n:
            return None
        tasks = [item.compact_text for item in buffer[: self._agg_n]]
        env_contract_full = buffer[0].full_text
        user_prompt = build_task_pack(env_name, env_contract_full, tasks)
        system_prompt = self._system_prompt_selector("aggregate3", env_name)
        tool = self._invoke_toolgen(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            name_prefix=self._name_prefix,
        )
        if tool:
            self._bootstrapped_envs.add(env_name)
        self._buffers[env_name] = []
        return [tool] if tool else None
