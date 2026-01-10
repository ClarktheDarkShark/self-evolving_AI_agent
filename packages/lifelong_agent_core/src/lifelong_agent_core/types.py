from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Any, Iterable, Mapping


@unique
class Role(StrEnum):
    USER = "user"
    AGENT = "agent"


def normalize_role(value: object) -> Role:
    if isinstance(value, Role):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "assistant":
            return Role.AGENT
        if normalized in ("agent", "user"):
            return Role(normalized)
    raise ValueError(f"Unsupported role: {value}")


@dataclass
class ChatHistoryItem:
    role: Role
    content: str

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "ChatHistoryItem":
        role = normalize_role(item.get("role"))
        content = str(item.get("content", ""))
        return cls(role=role, content=content)


@dataclass
class Message:
    role: str
    content: str


class ChatHistory:
    def __init__(self, items: Iterable[ChatHistoryItem] | None = None) -> None:
        self.value: list[ChatHistoryItem] = list(items or [])

    def inject(self, item: ChatHistoryItem | Mapping[str, Any]) -> "ChatHistory":
        if isinstance(item, Mapping):
            item = ChatHistoryItem.from_mapping(item)
        if self.value:
            if self.value[-1].role == item.role:
                raise ValueError("ChatHistory roles must alternate")
        self.value.append(item)
        return self

    def set(self, item_index: int, item: ChatHistoryItem | Mapping[str, Any]) -> None:
        if isinstance(item, Mapping):
            item = ChatHistoryItem.from_mapping(item)
        original = self.value[item_index]
        if original.role != item.role:
            raise ValueError("ChatHistory role mismatch on set")
        self.value[item_index] = item

    def pop(self, item_index: int) -> ChatHistoryItem:
        return self.value.pop(item_index)

    def get_item_deep_copy(self, item_index: int) -> ChatHistoryItem:
        item = self.value[item_index]
        return ChatHistoryItem(role=item.role, content=item.content)

    def get_value_length(self) -> int:
        return len(self.value)

    def to_messages(self) -> list[Message]:
        return [Message(role=item.role.value, content=item.content) for item in self.value]
