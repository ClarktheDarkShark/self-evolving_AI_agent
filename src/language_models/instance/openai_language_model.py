from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import openai
import os
import logging
from typing import Any, Optional, Sequence, Mapping, TypeGuard
import json

from src.language_models.language_model import LanguageModel
from src.typings import (
    Role,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    ChatHistory,
)
from src.utils import RetryHandler, ExponentialBackoffStrategy


class OpenaiLanguageModel(LanguageModel):
    """
    To keep the name of the class consistent with the name of file, use OpenaiAgent instead of OpenAIAgent.
    """

    def __init__(
        self,
        model_name: str,
        role_dict: Mapping[str, str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        maximum_prompt_token_count: Optional[int] = None,
    ):
        """
        max_prompt_tokens: The maximum number of tokens that can be used in the prompt. It can be used to set the
            context limit manually. If it is set to None, the context limit will be the same as the context length of
            the model selected.
        """
        super().__init__(role_dict)
        self.model_name = model_name
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.maximum_prompt_token_count = maximum_prompt_token_count

    @staticmethod
    def _is_valid_message_list(
        message_list: list[Mapping[str, str]],
    ) -> TypeGuard[list[ChatCompletionMessageParam]]:
        for message_dict in message_list:
            if (
                "role" not in message_dict.keys()
                or "content" not in message_dict.keys()
            ):
                return False
        return True

    @RetryHandler.handle(
        max_retries=3,
        retry_on=(openai.BadRequestError,),
        waiting_strategy=ExponentialBackoffStrategy(interval=(None, 60), multiplier=2),
    )
    def _get_completion_content(
        self,
        message_list: Sequence[ChatCompletionMessageParam],
        inference_config_dict: Mapping[str, Any],
    ) -> Sequence[str]:
        """
        I do not know what will happen when the context limit is reached. According to OpenAI documents, there is no
        type of error for the context limit. So I guess the model will return an empty response when the context limit
        is reached. This may be a potential bug and I apologize in advance.
        There are also some issues on GitHub state that the model will raise openai.BadRequestError in this situation.
        So I also handle this error in the code.
        Reference:
        https://platform.openai.com/docs/guides/error-codes#python-library-error-types
        https://github.com/run-llama/llama_index/discussions/11889
        """
        logger = logging.getLogger(__name__)
        stripped_fields: dict[str, Any] = {}
        sanitized_config = dict(inference_config_dict)
        for key in ("tools", "tool_choice", "parallel_tool_calls", "response_format"):
            if key in sanitized_config:
                stripped_fields[key] = sanitized_config.pop(key)
        if stripped_fields:
            logger.warning(
                "Stripped unsupported tool fields from inference_config: %s",
                sorted(stripped_fields.keys()),
            )
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                **sanitized_config,
            )
            # right after create(...)
            # print("=== RAW COMPLETION OBJECT ===")
            # print(completion.model_dump_json(indent=2))

            # # optional: also dump what you sent (careful: this prints your whole prompt)
            # print("=== REQUEST PAYLOAD (model/messages/config) ===")
            # print(
            #     json.dumps(
            #         {"model": self.model_name, "messages": message_list, **inference_config_dict},
            #         indent=2,
            #         default=str,
            #     )
            # )

        except openai.BadRequestError as e:
            if "context length" in str(e):
                # Raise LanguageModelContextLimitException to skip retrying.
                raise LanguageModelContextLimitException(
                    f"Model {self.model_name} reaches the context limit. "
                )
            else:
                # Raise the original exception to retry.
                raise e
        except openai.InternalServerError as e:
            try:
                message_size = len(
                    json.dumps(message_list, ensure_ascii=True, default=str)
                )
            except Exception:
                message_size = -1
            logger.error(
                "OpenAI server error. model=%s message_count=%d message_json_len=%s "
                "inference_config=%s stripped=%s error=%s",
                self.model_name,
                len(message_list),
                message_size,
                sanitized_config,
                sorted(stripped_fields.keys()),
                e,
            )
            raise

        if (
            completion.usage is not None
            and self.maximum_prompt_token_count is not None
            and completion.usage.prompt_tokens > self.maximum_prompt_token_count
        ):
            raise LanguageModelContextLimitException(
                f"Model {self.model_name} reaches the context limit. "
                f"Current prompt tokens: {completion.usage.prompt_tokens}. "
                f"Max prompt tokens: {self.maximum_prompt_token_count}."
            )
        content_list: list[str] = []
        content_all_invalid_flag: bool = True
        logger = logging.getLogger(__name__)
        for choice in completion.choices:
            message = choice.message
            content = message.content
            if not content:
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    first = tool_calls[0]
                    func = getattr(first, "function", None)
                    name = getattr(func, "name", None)
                    args = getattr(func, "arguments", None)
                    if name:
                        if args is None:
                            args = "{}"
                        if not isinstance(args, str):
                            args = json.dumps(args, ensure_ascii=True, default=str)
                        content = f'<internal_tool name="{name}">{args}</internal_tool>'
                        if len(tool_calls) > 1:
                            logger.warning(
                                "Multiple tool_calls returned; using first only. count=%d",
                                len(tool_calls),
                            )
            if content is not None and len(content) > 0:
                content_all_invalid_flag = False
            content_list.append(content or "")
        if content_all_invalid_flag:
            logger = logging.getLogger(__name__)
            try:
                message_size = len(
                    json.dumps(message_list, ensure_ascii=True, default=str)
                )
            except Exception:
                message_size = -1
            choices_info = []
            for choice in completion.choices:
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None)
                choices_info.append(
                    {
                        "finish_reason": choice.finish_reason,
                        "content_len": len(message.content or ""),
                        "tool_calls": bool(tool_calls),
                    }
                )
            logger.error(
                "OpenAI completion returned empty content. model=%s message_count=%d "
                "message_json_len=%s inference_config=%s usage=%s choices=%s",
                self.model_name,
                len(message_list),
                message_size,
                inference_config_dict,
                getattr(completion, "usage", None),
                choices_info,
            )
            raise LanguageModelContextLimitException(
                f"Model {self.model_name} returns empty response. The context limit may be reached."
            )
        return content_list

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        """
        system_prompt: It is usually called as system_prompt. But in OpenAI documents, it is called as developer_prompt.
            But in practice, using `message_list = [{"role": "developer", "content": self.system_prompt}]` will raise an
            error. So all after all, I call it as system_prompt.
            Reference:
            https://platform.openai.com/docs/guides/text-generation#messages-and-roles
            https://platform.openai.com/docs/api-reference/chat/create
        inference_config_dict: Other config for OpenAI().chat.completions.create.
            e.g.:
            max_completion_tokens: The maximum number of tokens that can be generated in the chat completion. Notice
                that max_tokens is deprecated.
            Reference:
            https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens
        """
        # region Construct batch_message_list
        message_list_prefix: list[ChatCompletionMessageParam]
        if len(system_prompt) > 0:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []
        batch_message_list: list[Sequence[ChatCompletionMessageParam]] = []
        for chat_history in batch_chat_history:
            conversion_result = self._convert_chat_history_to_message_list(chat_history)
            assert OpenaiLanguageModel._is_valid_message_list(conversion_result)
            batch_message_list.append(message_list_prefix + conversion_result)
        # endregion
        # region Generate output
        output_str_list: list[str] = []
        for message_list in batch_message_list:
            output_str_list.extend(
                self._get_completion_content(message_list, inference_config_dict)
            )
        # endregion
        # region Convert output to ChatHistoryItem
        return [
            ChatHistoryItem(role=Role.AGENT, content=output_str)
            for output_str in output_str_list
        ]
        # endregion
