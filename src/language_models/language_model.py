from abc import ABC, abstractmethod
from typing import Sequence, Mapping, Any, Optional, final

from src.typings import (
    ChatHistory,
    ChatHistoryItem,
    Role,
    ModelException,
    LanguageModelUnknownException,
)


TOOL_CALL_KEYS = {"tools", "tool_choice", "functions", "function_call"}

def _strip_tool_calling(obj: Any) -> Any:
    """
    Recursively remove tool-calling fields from dicts/lists so Ollama never sees them.
    This prevents Ollama's 'error parsing tool call' 500 forever.
    """
    if isinstance(obj, dict):
        # copy so we never mutate caller-owned dicts
        new_d = {}
        for k, v in obj.items():
            if k in TOOL_CALL_KEYS:
                continue
            new_d[k] = _strip_tool_calling(v)
        return new_d
    if isinstance(obj, list):
        return [_strip_tool_calling(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_tool_calling(x) for x in obj)
    return obj


class LanguageModel(ABC):
    def __init__(self, role_dict: Mapping[str, str]) -> None:
        self.role_dict: Mapping[Role, str] = {
            Role(role): role_dict[role] for role in Role
        }

    def _convert_chat_history_to_message_list(
        self, chat_history: ChatHistory
    ) -> list[Mapping[str, str]]:
        message_list: list[Mapping[str, str]] = []
        for item_index in range(chat_history.get_value_length()):
            chat_history_item = chat_history.get_item_deep_copy(item_index)
            message_list.append(
                {
                    "role": self.role_dict[chat_history_item.role],
                    "content": chat_history_item.content,
                }
            )
        return message_list

    @final
    def inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Sequence[ChatHistoryItem]:
        for chat_history in batch_chat_history:
            if chat_history.get_item_deep_copy(-1).role != Role.USER:
                try:
                    chat_history.inject(ChatHistoryItem(role=Role.USER, content=""))
                except Exception:
                    pass
        try:
            if inference_config_dict is None:
                inference_config_dict = {}

            # --- PROMPT TRACING LOGIC ---
            if system_prompt:
                prompt_preview = system_prompt[:50].replace("\n", " ")
                identity = "UNKNOWN"
                if "Combined Orchestrator" in system_prompt:
                    identity = "COMBINED_ORCHESTRATOR"
                elif "TOOLGEN" in system_prompt or "ToolGen" in system_prompt:
                    identity = "TOOLGEN_GENERATOR"
                elif "###TOOL_START" in system_prompt or "Python" in system_prompt:
                    identity = "TOOLGEN_GENERATOR"
                elif "intelligent agent tasked with answering questions" in system_prompt:
                    identity = "SOLVER_ACTOR"
                elif "Actor" in system_prompt or "Solver" in system_prompt:
                    identity = "SOLVER_ACTOR"
                print(f"\n[PROMPT_TRACE] Agent Instance: {self.__class__.__name__}")
                print(f"[PROMPT_TRACE] Detected Identity: {identity}")
                print(f"[PROMPT_TRACE] System Prompt Preview: {prompt_preview}...")

            # ---- GLOBAL KILL SWITCH (prevents Ollama tool-call parsing 500) ----
            # Convert Mapping -> dict then recursively strip tool calling keys.
            cleaned = _strip_tool_calling(dict(inference_config_dict))
            if any(k in cleaned for k in TOOL_CALL_KEYS):
                raise RuntimeError(
                    f"Tool calling leaked after sanitize: {set(cleaned.keys())}"
                )
            inference_result = self._inference(batch_chat_history, cleaned, system_prompt)
        except ModelException as e:
            raise e
        except Exception as e:
            raise LanguageModelUnknownException(str(e)) from e
        return inference_result

    @abstractmethod
    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        pass
