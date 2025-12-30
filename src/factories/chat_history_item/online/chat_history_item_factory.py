import json
from typing import Optional

from src.typings import Role, ChatHistoryItemDict, ChatHistoryItem
from src.utils import SafeLogger
from abc import ABC, abstractmethod


class ChatHistoryItemFactoryInterface(ABC):
    @abstractmethod
    def construct(
        self, chat_history_item_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        pass

    @abstractmethod
    def get_chat_history_item_dict_deep_copy(self) -> ChatHistoryItemDict:
        pass

    @abstractmethod
    def set(self, prompt_index: int, role: Role, content: str) -> None:
        pass


class ChatHistoryItemFactory(ChatHistoryItemFactoryInterface):
    def __init__(self, chat_history_item_dict_path: str):
        super().__init__()
        self._chat_history_item_dict = ChatHistoryItemDict.model_validate(
            json.load(open(chat_history_item_dict_path))
        )

    def construct(
        self, chat_history_item_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        try:
            result = self._chat_history_item_dict.value[str(chat_history_item_index)]
        except KeyError:
            if expected_role is None:
                raise
            SafeLogger.warning(
                "ChatHistoryItemFactory missing index %s; using fallback role %s.",
                chat_history_item_index,
                expected_role,
            )
            fallback_content = "OK." if expected_role == Role.AGENT else ""
            return ChatHistoryItem(role=expected_role, content=fallback_content)
        if expected_role is not None and result.role != expected_role:
            SafeLogger.warning(
                "ChatHistoryItemFactory role mismatch at index %s: expected %s, got %s. "
                "Using fallback role.",
                chat_history_item_index,
                expected_role,
                result.role,
            )
            fallback_content = "OK." if expected_role == Role.AGENT else result.content
            return ChatHistoryItem(role=expected_role, content=fallback_content)
        return result

    def get_chat_history_item_dict_deep_copy(self) -> ChatHistoryItemDict:
        return self._chat_history_item_dict.model_copy(deep=True)

    def set(self, prompt_index: int, role: Role, content: str) -> None:
        self._chat_history_item_dict.set_chat_history_item(prompt_index, role, content)
