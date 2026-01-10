from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

from .exceptions import (
    AgentContextLimitError,
    AgentOutOfMemoryError,
    AgentUnknownError,
    ModelContextLimitError,
    ModelOutOfMemoryError,
    ModelUnknownError,
)
from .types import ChatHistory, ChatHistoryItem, Role


@dataclass
class ModelResponse:
    content: str
    finish_reason: Optional[str] = None
    raw: Any = None


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system_prompt: str,
        inference_config: Optional[Mapping[str, Any]] = None,
    ) -> ModelResponse:
        ...


class LanguageModelAgent:
    def __init__(
        self,
        model: Optional[ModelAdapter] = None,
        *,
        language_model: Optional[ModelAdapter] = None,
        system_prompt: str = "You are a helpful assistant.",
        inference_config_dict: Optional[Mapping[str, Any]] = None,
    ) -> None:
        resolved = model or language_model
        if resolved is None:
            raise ValueError("LanguageModelAgent requires a ModelAdapter.")
        self._model = resolved
        self._system_prompt = system_prompt
        self._inference_config_dict = inference_config_dict
        self.role_dict = {Role.USER: "user", Role.AGENT: "assistant"}

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        messages = [
            {"role": item.role.value, "content": item.content}
            for item in chat_history.value
        ]
        try:
            response = self._model.complete(
                messages,
                system_prompt=self._system_prompt,
                inference_config=self._inference_config_dict,
            )
        except ModelContextLimitError as exc:
            raise AgentContextLimitError(str(exc)) from exc
        except ModelOutOfMemoryError as exc:
            raise AgentOutOfMemoryError(str(exc)) from exc
        except ModelUnknownError as exc:
            raise AgentUnknownError(str(exc)) from exc
        return ChatHistoryItem(role=Role.AGENT, content=response.content)

    def get_role_dict(self) -> Mapping[Role, str]:
        return self.role_dict
