from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.typings import ChatHistory, ChatHistoryItem, Role


def _ensure_packaged_agent_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pkg_src = repo_root / "packages" / "lifelong_agent_core" / "src"
    if pkg_src.exists():
        path_str = str(pkg_src)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


class PackagedSelfEvolvingShim:
    def __init__(
        self,
        *,
        language_model: Any,
        tool_registry_path: str,
        max_generated_tools_per_run: int = 3,
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        bootstrap_tools: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: str = "",
        force_tool_generation_if_missing: bool = True,
        tool_match_min_score: float = 0.25,
        include_registry_in_prompt: bool = True,
        environment_label: str = "unknown",
        retrieval_top_k: int = 5,
        reuse_top_k: int = 3,
        reuse_similarity_threshold: Optional[float] = None,
        reuse_min_reliability: float = 0.0,
        canonical_tool_naming: bool = True,
    ) -> None:
        _ensure_packaged_agent_on_path()
        from lifelong_agent_core.controller import SelfEvolvingController
        from lifelong_agent_core.model import ModelResponse
        from lifelong_agent_core.types import ChatHistory as PackagedChatHistory
        from lifelong_agent_core.types import ChatHistoryItem as PackagedChatHistoryItem
        from lifelong_agent_core.types import Role as PackagedRole

        class _LegacyModelAdapter:
            def __init__(self, base_model: Any):
                self._base_model = base_model

            def complete(self, messages, *, system_prompt: str, inference_config=None):
                chat_history = ChatHistory()
                for msg in messages:
                    role = msg.get("role")
                    if role == "user":
                        chat_history.inject(
                            ChatHistoryItem(role=Role.USER, content=msg.get("content", ""))
                        )
                    elif role in ("agent", "assistant"):
                        chat_history.inject(
                            ChatHistoryItem(role=Role.AGENT, content=msg.get("content", ""))
                        )
                response = self._base_model.inference(
                    [chat_history],
                    inference_config or inference_config_dict,
                    system_prompt,
                )[0]
                return ModelResponse(content=response.content, raw=response)

        self._packaged_chat_history = PackagedChatHistory
        self._packaged_item = PackagedChatHistoryItem
        self._packaged_role = PackagedRole
        self._controller = SelfEvolvingController(
            language_model=_LegacyModelAdapter(language_model),
            tool_registry_path=tool_registry_path,
            max_generated_tools_per_run=max_generated_tools_per_run,
            inference_config_dict=inference_config_dict,
            bootstrap_tools=bootstrap_tools,
            system_prompt=system_prompt,
            force_tool_generation_if_missing=force_tool_generation_if_missing,
            tool_match_min_score=tool_match_min_score,
            include_registry_in_prompt=include_registry_in_prompt,
            environment_label=environment_label,
            retrieval_top_k=retrieval_top_k,
            reuse_top_k=reuse_top_k,
            reuse_similarity_threshold=reuse_similarity_threshold,
            reuse_min_reliability=reuse_min_reliability,
            canonical_tool_naming=canonical_tool_naming,
        )

    def _to_packaged_history(self, history: ChatHistory) -> Any:
        items = []
        for idx in range(history.get_value_length()):
            item = history.get_item_deep_copy(idx)
            role = self._packaged_role.USER if item.role == Role.USER else self._packaged_role.AGENT
            items.append(self._packaged_item(role=role, content=item.content))
        return self._packaged_chat_history(items)

    def inference(self, history: ChatHistory) -> ChatHistoryItem:
        packaged_history = self._to_packaged_history(history)
        response = self._controller._inference(packaged_history)
        role = Role.AGENT if response.role.value == "agent" else Role.USER
        return ChatHistoryItem(role=role, content=response.content)

    def get_role_dict(self) -> Mapping[Role, str]:
        return {Role.USER: "user", Role.AGENT: "assistant"}
