from typing import Any, Optional, Mapping
from typing_extensions import override

from src.agents.agent import Agent
from src.typings import (
    ChatHistoryItem,
    ChatHistory,
    LanguageModelContextLimitException,
    AgentContextLimitException,
    LanguageModelOutOfMemoryException,
    AgentOutOfMemoryException,
    LanguageModelUnknownException,
    AgentUnknownException,
    Role,
)
from src.language_models import LanguageModel
from src.language_models.inference_context import get_inference_context, set_inference_context


class LanguageModelAgent(Agent):
    def __init__(
        self,
        language_model: LanguageModel,
        system_prompt: str = "You are a helpful assistant.",
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        agent_name: Optional[str] = None,
    ):
        """
        The name of the parameter `language_model` is referenced in `src.run_experiment.py` by string.
            So do not change it.
        """
        self._language_model = language_model
        self._system_prompt = system_prompt
        self._inference_config_dict = inference_config_dict
        self._agent_name = agent_name

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        prev_context = get_inference_context()
        try:
            if self._agent_name:
                set_inference_context(
                    task_name=prev_context.task_name,
                    sample_index=prev_context.sample_index,
                    chat_history_len=prev_context.chat_history_len,
                    agent_name=self._agent_name,
                )
            batch = [chat_history]
            result = self._language_model.inference(
                batch, self._inference_config_dict, self._system_prompt
            )
            item0 = result[0]
            return item0
        except LanguageModelContextLimitException as e:
            raise AgentContextLimitException(str(e)) from e
        except LanguageModelOutOfMemoryException as e:
            raise AgentOutOfMemoryException(str(e)) from e
        except LanguageModelUnknownException as e:
            raise AgentUnknownException(str(e)) from e
        finally:
            if self._agent_name:
                set_inference_context(
                    task_name=prev_context.task_name,
                    sample_index=prev_context.sample_index,
                    chat_history_len=prev_context.chat_history_len,
                    agent_name=prev_context.agent_name,
                )

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        return self._language_model.role_dict
