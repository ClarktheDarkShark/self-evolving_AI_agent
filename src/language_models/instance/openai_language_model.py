from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import openai
import os
import logging
import json
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Mapping, TypeGuard

from src.language_models.language_model import LanguageModel
from src.language_models.inference_context import get_inference_context
from src.typings import Role, ChatHistoryItem, ChatHistory
from src.utils import RetryHandler, ExponentialBackoffStrategy


OLLAMA_ALLOWED_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "n",
    "stream",
    # NOTE: intentionally NOT allowing "stop" to avoid truncating multi-line formats
}

OLLAMA_ALLOWED_MESSAGE_KEYS = {"role", "content", "name"}


class OpenaiLanguageModel(LanguageModel):
    """
    Thin chat wrapper that does NOT rewrite or parse model content.

    Key behaviors:
    - Supports a "benchmark-safe" mode where API tool calling is disabled, but
      the model can still be instructed (in plain text) how to use your benchmark
      action formats.
    - If API tool calls occur anyway (or output is empty / contains <internal_tool>),
      a repair pass forces a final benchmark-compliant output.
    """

    def __init__(
        self,
        model_name: str,
        role_dict: Mapping[str, str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        maximum_prompt_token_count: Optional[int] = None,
        disable_api_tools: bool = False,
    ):
        super().__init__(role_dict)
        self.model_name = model_name

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.maximum_prompt_token_count = maximum_prompt_token_count

        # If True: prevents OpenAI/ChatCompletions tool calls from being used,
        # while still allowing plain-text tool instructions in the prompt.
        self.disable_api_tools = disable_api_tools

    @staticmethod
    def _is_valid_message_list(
        message_list: list[Mapping[str, str]],
    ) -> TypeGuard[list[ChatCompletionMessageParam]]:
        for message_dict in message_list:
            if "role" not in message_dict or "content" not in message_dict:
                return False
        return True

    def _is_ollama_endpoint(self) -> bool:
        return bool(self.base_url) and "11434" in self.base_url

    @staticmethod
    def _sanitize_for_ollama(cfg: Mapping[str, Any]) -> dict[str, Any]:
        # Only forward keys Ollama supports. "stop" is intentionally excluded.
        return {k: cfg[k] for k in OLLAMA_ALLOWED_KEYS if k in cfg}

    def _strip_tool_messages(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        For endpoints (like many Ollama setups) that don't support tool roles,
        remove tool/function messages and keep only simple message keys.
        """
        cleaned: list[ChatCompletionMessageParam] = []
        for message in messages:
            if message.get("role") in ("tool", "function"):
                continue
            sanitized = {k: message[k] for k in OLLAMA_ALLOWED_MESSAGE_KEYS if k in message}
            cleaned.append(sanitized)
        return cleaned

    def _is_os_interaction_prompt(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> bool:
        for message in messages:
            content = message.get("content", "")
            if "Act: bash" in content or "Act: finish" in content:
                return True
        return False

    def _needs_bash_block_repair(self, content: str) -> bool:
        if not content:
            return False
        first_line = content.strip().splitlines()[0].strip()
        if not (first_line.startswith("Act: bash") or first_line.startswith("Action: bash")):
            return False
        return "```bash" not in content

    def _bash_block_repair_instruction(self) -> str:
        return (
            "Output exactly one action with a bash code block.\n"
            "Format:\n"
            "Act: bash\n"
            "```bash\n"
            "<commands>\n"
            "```"
        )

    def _needs_final_repair(self, content: str) -> bool:
        """
        Generic benchmark-safe checks:
        - empty/whitespace output
        - leaked internal tool protocol blocks
        """
        if not content or not content.strip():
            return True
        if "<internal_tool" in content:
            return True
        return False

    def _is_internal_tool_block(self, content: str) -> bool:
        text = (content or "").strip()
        return text.startswith("<internal_tool") and text.endswith("</internal_tool>")

    def _final_output_repair_instruction(self) -> str:
        return (
            "You MUST follow the task instructions EXACTLY.\n"
            "Do NOT call tools/functions via API tool calls.\n"
            "Do NOT output any <internal_tool ...> blocks.\n"
            "Output ONLY the final required format specified by the task (e.g., Action/Act format or final answer), "
            "with no extra commentary.\n"
        )

    def _log_inference_payload(
        self,
        *,
        messages: Sequence[ChatCompletionMessageParam],
        config: Mapping[str, Any],
        completion: Any,
        allow_tools: bool,
        is_ollama: bool,
    ) -> None:
        path = os.environ.get("LIFELONG_MODEL_OUTPUT_PATH")
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        context = get_inference_context()
        payload = completion.model_dump() if hasattr(completion, "model_dump") else completion
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": timestamp,
            "model": self.model_name,
            "base_url": self.base_url,
            "allow_tools": allow_tools,
            "is_ollama": is_ollama,
            "context": {
                "task_name": context.task_name,
                "sample_index": context.sample_index,
                "chat_history_len": context.chat_history_len,
            },
            "request": {
                "messages": list(messages),
                "config": dict(config),
            },
            "completion": payload,
        }
        with open(path, "a") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=str))
            handle.write("\n")

    def _create_completion_with_optional_tool_choice_none(
        self,
        client: OpenAI,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        config: dict[str, Any],
        try_tool_choice_none: bool,
    ):
        """
        Some OpenAI-compatible servers accept tool_choice="none" as a hard guard,
        but others may reject it. We try it once and fall back automatically.
        """
        if not try_tool_choice_none:
            return client.chat.completions.create(model=model, messages=messages, **config)

        cfg_try = dict(config)
        # Only set if not already present
        cfg_try.setdefault("tool_choice", "none")
        try:
            return client.chat.completions.create(model=model, messages=messages, **cfg_try)
        except openai.BadRequestError:
            # Fallback: remove tool_choice and retry once
            cfg_fallback = dict(config)
            cfg_fallback.pop("tool_choice", None)
            return client.chat.completions.create(model=model, messages=messages, **cfg_fallback)

    @RetryHandler.handle(
        max_retries=3,
        retry_on=(openai.BadRequestError,),
        waiting_strategy=ExponentialBackoffStrategy(interval=(None, 60), multiplier=2),
    )
    def _get_completion_content(
        self,
        message_list: Sequence[ChatCompletionMessageParam],
        inference_config_dict: Mapping[str, Any],
        allow_tools: bool = True,
    ) -> list[str]:
        logger = logging.getLogger(__name__)

        is_ollama = self._is_ollama_endpoint()
        allow_internal_tool_protocol = bool(
            inference_config_dict.get("allow_internal_tool_protocol")
        )

        # Force-disable API tools on Ollama lane (and/or if configured).
        allow_tools = bool(allow_tools) and (not self.disable_api_tools) and (not is_ollama)

        base_messages = list(message_list)

        # Start from user config, but NEVER allow "stop" to truncate multi-line formats.
        sanitized_config = dict(inference_config_dict)
        sanitized_config.pop("stop", None)

        if is_ollama:
            # Ollama lane: strip tool roles and only pass allowed config keys.
            base_messages = self._strip_tool_messages(base_messages)
            sanitized_config = self._sanitize_for_ollama(sanitized_config)  # stop already removed
        else:
            # Non-ollama lane: if tools are disabled, remove all tool-related config.
            if not allow_tools:
                for key in ("tools", "tool_choice", "parallel_tool_calls", "response_format"):
                    sanitized_config.pop(key, None)

        # Execute request
        client = self.client.with_options(max_retries=0) if is_ollama else self.client

        # Try tool_choice="none" only when not allowing tools and not ollama.
        # If server rejects it, we auto-fallback.
        completion = self._create_completion_with_optional_tool_choice_none(
            client=client,
            model=self.model_name,
            messages=base_messages,
            config=sanitized_config,
            try_tool_choice_none=(not is_ollama and not allow_tools),
        )

        self._log_inference_payload(
            messages=base_messages,
            config=sanitized_config,
            completion=completion,
            allow_tools=allow_tools,
            is_ollama=is_ollama,
        )

        # Optional debug
        # print("=== RAW COMPLETION OBJECT ===")
        # print(completion.model_dump_json(indent=2))
        # print("=== REQUEST PAYLOAD (model/messages/config) ===")
        # print(
        #     json.dumps(
        #         {"model": self.model_name, "messages": base_messages, **sanitized_config},
        #         indent=2,
        #         default=str,
        #     )
        # )

        out: list[str] = []
        is_os_prompt = self._is_os_interaction_prompt(base_messages)

        for choice in completion.choices:
            content = choice.message.content or ""
            tool_calls = getattr(choice.message, "tool_calls", None)

            # If tools are allowed (agent/controller mode) and content is empty but tool_calls exist,
            # preserve the tool call as <internal_tool ...>.
            # If tools are NOT allowed (benchmark-safe mode), DO NOT emit <internal_tool ...>;
            # instead, force a repair pass below.
            if (not content) and tool_calls:
                if allow_tools:
                    first = tool_calls[0]
                    func = getattr(first, "function", None)
                    name = getattr(func, "name", None)
                    args = getattr(func, "arguments", None) or "{}"
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=True, default=str)
                    content = f'<internal_tool name="{name}">{args}</internal_tool>'
                else:
                    content = ""  # trigger final repair

            # OS-interaction bash fence repair (kept as-is)
            if is_os_prompt and self._needs_bash_block_repair(content):
                repair_messages = [
                    {"role": "system", "content": self._bash_block_repair_instruction()},
                    *list(base_messages),
                ]
                repair = client.chat.completions.create(
                    model=self.model_name,
                    messages=repair_messages,
                    **sanitized_config,
                )
                repair_content = repair.choices[0].message.content or ""
                if "```bash" in repair_content:
                    content = repair_content

            # Benchmark-safe: force final output when empty / internal tool blocks show up.
            if (not allow_tools) and self._needs_final_repair(content):
                if allow_internal_tool_protocol and self._is_internal_tool_block(content):
                    out.append(content)
                    continue
                repair_messages: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": self._final_output_repair_instruction()},
                    *list(base_messages),
                    # Provide the bad output so the model can correct itself deterministically.
                    {"role": "assistant", "content": content or "<empty>"},
                    {
                        "role": "user",
                        "content": "Repair your output to match the required final format. Output only the corrected final response.",
                    },
                ]

                # Use same sanitized_config (already tool-free). Try to hard-guard tool_choice none again.
                repair = self._create_completion_with_optional_tool_choice_none(
                    client=client,
                    model=self.model_name,
                    messages=repair_messages,
                    config=sanitized_config,
                    try_tool_choice_none=(not is_ollama),
                )
                repaired = repair.choices[0].message.content or ""
                if repaired.strip():
                    content = repaired

            out.append(content)

        logger.debug("Model outputs (repr): %s", [repr(x) for x in out])
        return out

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        message_list_prefix: list[ChatCompletionMessageParam]
        if system_prompt:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []

        batch_message_list: list[Sequence[ChatCompletionMessageParam]] = []
        for chat_history in batch_chat_history:
            conversion_result = self._convert_chat_history_to_message_list(chat_history)
            assert OpenaiLanguageModel._is_valid_message_list(conversion_result)
            batch_message_list.append(message_list_prefix + conversion_result)

        output_str_list: list[str] = []
        for message_list in batch_message_list:
            # Default policy:
            # - Ollama endpoints: API tools disabled
            # - disable_api_tools=True: API tools disabled
            # - else: allow (controller/agent) tool protocol if you want it
            allow_tools = (not self.disable_api_tools) and (not self._is_ollama_endpoint())

            output_str_list.extend(
                self._get_completion_content(
                    message_list=message_list,
                    inference_config_dict=inference_config_dict,
                    allow_tools=allow_tools,
                )
            )

        return [ChatHistoryItem(role=Role.AGENT, content=s) for s in output_str_list]
