# openai_language_model.py

from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import openai
import os
import logging
import json
import uuid
import httpx
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Mapping, TypeGuard

from src.language_models.language_model import LanguageModel
from src.language_models.inference_context import get_inference_context
from src.typings import Role, ChatHistoryItem, ChatHistory
from src.utils import RetryHandler, ExponentialBackoffStrategy


OLLAMA_RESPOND_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "respond",
            "description": "Return the final assistant output in the 'content' field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    }
]


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
TOOL_REQUEST_KEYS = {
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "functions",
    "function_call",
}

def _dbg_preview(x, n=220):
    s = "" if x is None else str(x)
    s = s.replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")


class OpenaiLanguageModel(LanguageModel):
    """
    Thin chat wrapper that does NOT rewrite or parse model content.
    """

    # Always-long default timeout (seconds). 30 minutes.
    DEFAULT_TIMEOUT_S: float = 1200.0

    def __init__(
        self,
        model_name: str,
        role_dict: Mapping[str, str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        maximum_prompt_token_count: Optional[int] = None,
        disable_api_tools: bool = True,
    ):
        super().__init__(role_dict)
        self.model_name = model_name

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")

        self.api_key = api_key
        self.base_url = base_url
        self.maximum_prompt_token_count = maximum_prompt_token_count
        self.disable_api_tools = disable_api_tools

        # Always-long timeout; no env var required.
        timeout_s = self.DEFAULT_TIMEOUT_S
        timeout = httpx.Timeout(timeout_s, read=timeout_s, write=timeout_s, connect=30.0)

        http_client = self._build_httpx_client()

        if http_client is None:
            if os.getenv("LIFELONG_HTTPX_LOG", "").strip():
                print("[LM] WARNING: httpx logging requested but client not available.")
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            try:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=http_client,
                    timeout=timeout,
                )
            except TypeError:
                # Older SDKs might not support http_client.
                print("[LM] WARNING: OpenAI client does not accept http_client; httpx logging disabled.")
                self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
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

    @staticmethod
    def _strip_tool_request_fields(config: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in config.items() if k not in TOOL_REQUEST_KEYS}

    @staticmethod
    def _assert_no_tool_request_fields(config: Mapping[str, Any], *, is_ollama: bool = False) -> None:
        if is_ollama:
            # Ollama lane: we intentionally use tools/tool_choice to force JSON-safe output.
            return
        leftover = TOOL_REQUEST_KEYS.intersection(config.keys())
        if leftover:
            raise RuntimeError(f"Tool request keys must not be sent: {sorted(leftover)}")


    @staticmethod
    def _minimal_retry_config(config: Mapping[str, Any]) -> dict[str, Any]:
        keep = {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "n",
        }
        return {k: v for k, v in config.items() if k in keep}

    @staticmethod
    def _tool_call_avoidance_prefix() -> str:
        return (
            "IMPORTANT: Do NOT use tool calls or function-call JSON. "
            "Output plain text only."
        )

    @staticmethod
    def _get_debug_payload_dir() -> Optional[str]:
        override = os.getenv("LIFELONG_DEBUG_PAYLOAD_DIR", "").strip()
        if override:
            return override
        model_path = os.getenv("LIFELONG_MODEL_OUTPUT_PATH", "").strip()
        if model_path:
            return os.path.dirname(model_path)
        output_dir = os.getenv("LIFELONG_OUTPUT_DIR", "").strip()
        if output_dir:
            return output_dir
        return None

    def _append_payload_log(self, payload: Mapping[str, Any]) -> None:
        log_dir = self._get_debug_payload_dir()
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "payloads.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=str))
            handle.write("\n")

    def _write_payload_snapshot(self, filename: str, payload: Mapping[str, Any]) -> None:
        log_dir = self._get_debug_payload_dir()
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, default=str, indent=2)

    def _log_httpx_request(self, request: Any) -> None:
        log_dir = self._get_debug_payload_dir()
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        try:
            body = request.content
            if isinstance(body, bytes):
                body_text = body.decode("utf-8", errors="replace")
            else:
                body_text = str(body)
        except Exception as exc:
            body_text = f"<unavailable: {exc}>"
        headers = dict(request.headers)
        if "authorization" in {k.lower() for k in headers}:
            for key in list(headers.keys()):
                if key.lower() == "authorization":
                    headers[key] = "<redacted>"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "url": str(request.url),
            "headers": headers,
            "body": body_text,
        }
        path = os.path.join(log_dir, "httpx_requests.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str))
            handle.write("\n")

    def _log_httpx_response(self, response: Any) -> None:
        log_dir = self._get_debug_payload_dir()
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": getattr(response, "status_code", None),
            "url": str(getattr(response, "url", "")),
        }
        path = os.path.join(log_dir, "httpx_responses.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str))
            handle.write("\n")

    def _build_httpx_client(self) -> Any:
        flag = os.getenv("LIFELONG_HTTPX_LOG", "").strip().lower()
        if flag not in {"1", "true", "yes"}:
            return None
        try:
            import httpx
        except Exception:
            return None
        return httpx.Client(
            event_hooks={
                "request": [self._log_httpx_request],
                "response": [self._log_httpx_response],
            }
        )

    @staticmethod
    def _ollama_tool_response_name() -> str:
        return "respond"

    def _ollama_tool_parser_prefix(self) -> str:
        name = self._ollama_tool_response_name()
        return (
            "You MUST respond with a single JSON object for the tool parser.\n"
            f"Format: {{\"name\":\"{name}\",\"arguments\":{{\"content\":\"<OUTPUT>\"}}}}\n"
            "Where <OUTPUT> is the exact environment-required output string.\n"
            "Use double quotes only. Escape newlines as \\n and quotes as \\\" inside content.\n"
            "Return JSON only, no extra text."
        )

    def _ollama_output_repair_instruction(self) -> str:
        return (
            "Your previous output was invalid.\n"
            "Return ONLY the correct environment-required output inside the JSON tool call."
        )

    @staticmethod
    def _is_tool_parse_error(exc: Exception) -> bool:
        text = str(exc)
        return "error parsing tool call" in text.lower()

    def _prepend_ollama_tool_parser_prefix(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        prefix = self._ollama_tool_parser_prefix()
        if messages:
            first = messages[0]
            if (
                first.get("role") == "system"
                and first.get("content") == prefix
            ):
                return list(messages)
        return [{"role": "system", "content": prefix}, *list(messages)]

    def _prepare_ollama_messages(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        # Ollama lane: just strip tool roles; DO NOT inject any JSON tool-parser system prefix.
        return self._strip_tool_messages(messages)


    def _extract_ollama_tool_content(
        self,
        *,
        content: str,
        tool_calls: Any,
    ) -> Optional[str]:
        target_name = self._ollama_tool_response_name()
        if tool_calls:
            for call in tool_calls:
                func = getattr(call, "function", None)
                if func is None:
                    continue
                name = getattr(func, "name", None)
                raw_args = getattr(func, "arguments", None)
                args: Optional[dict[str, Any]] = None
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = None
                elif isinstance(raw_args, dict):
                    args = raw_args
                if not isinstance(args, dict):
                    continue
                content_value = args.get("content")
                if isinstance(content_value, str) and (
                    name is None or name == target_name
                ):
                    return content_value

        text = (content or "").strip()
        if not (text.startswith("{") and text.endswith("}")):
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            direct_content = payload.get("content")
            if isinstance(direct_content, str):
                return direct_content
            args = payload.get("arguments")
            if isinstance(args, dict):
                nested = args.get("content")
                if isinstance(nested, str):
                    return nested
        return None

    def _send_completion(
        self,
        *,
        client: OpenAI,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        config: Mapping[str, Any],
        is_ollama: bool,
    ) -> ChatCompletion:
        self._assert_no_tool_request_fields(config, is_ollama=is_ollama)
        return client.chat.completions.create(model=model, messages=messages, **config)


    def _create_completion_with_optional_tool_choice_none(
        self,
        client: OpenAI,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        config: dict[str, Any],
        try_tool_choice_none: bool,
    ):
        del try_tool_choice_none
        config = self._strip_tool_request_fields(config)
        self._assert_no_tool_request_fields(config)
        return self._send_completion(
            client=client,
            model=model,
            messages=messages,
            config=config,
        )


    @staticmethod
    def _deep_find_toolish_keys(obj, keys):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys:
                    found.append(k)
                found.extend(OpenaiLanguageModel._deep_find_toolish_keys(v, keys))
        elif isinstance(obj, list):
            for it in obj:
                found.extend(OpenaiLanguageModel._deep_find_toolish_keys(it, keys))
        return found

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
        return_completion: bool = False,
    ) -> list[str] | tuple[list[str], ChatCompletion]:
        logger = logging.getLogger(__name__)

        is_ollama = self._is_ollama_endpoint()

        print(
            "[LM] _get_completion_content | "
            f"is_ollama={is_ollama} | base_url={self.base_url} | model={self.model_name}"
        )
        print(
            "[LM] flags | "
            f"disable_api_tools={self.disable_api_tools} | "
            f"allow_internal_tool_protocol={bool(inference_config_dict.get('allow_internal_tool_protocol'))}"
        )
        print(
            "[LM] incoming cfg keys:",
            sorted(list(inference_config_dict.keys()))
        )

        allow_internal_tool_protocol = bool(
            inference_config_dict.get("allow_internal_tool_protocol")
        )

        # Hard-disable API tool usage; we only allow plain-text <internal_tool> blocks.
        allow_tools = False

        base_messages = list(message_list)
        request_messages = list(base_messages)

        # Start from user config, but NEVER allow "stop" to truncate multi-line formats.
        sanitized_config = dict(inference_config_dict)
        sanitized_config.pop("stop", None)
        sanitized_config.pop("allow_internal_tool_protocol", None)

        print(f"[LM] messages | base_len={len(base_messages)} request_len={len(request_messages)}")
        if request_messages:
            print(f"[LM] first msg role={request_messages[0].get('role')} head={_dbg_preview(request_messages[0].get('content'))}")
            print(f"[LM] last  msg role={request_messages[-1].get('role')} head={_dbg_preview(request_messages[-1].get('content'))}")
        print("[LM] sanitized_config:", {k: sanitized_config.get(k) for k in sorted(sanitized_config.keys())})

        if is_ollama:
            # Ollama lane: enforce JSON tool-parser output + strip tool roles.
            request_messages = self._prepare_ollama_messages(request_messages)
            sanitized_config = self._sanitize_for_ollama(sanitized_config)  # stop already removed

        # Execute request
        client = self.client.with_options(max_retries=0) if is_ollama else self.client

        def _request_completion(
            messages: Sequence[ChatCompletionMessageParam],
            config: Mapping[str, Any],
        ) -> ChatCompletion:
            request_config = dict(config)

            if is_ollama:
                # Force Ollama to always return a tool call with JSON args.
                request_config["tools"] = OLLAMA_RESPOND_TOOL
                request_config["tool_choice"] = {"type": "function", "function": {"name": "respond"}}

            # Still strip any stray tool fields for non-ollama
            if not is_ollama:
                request_config = self._strip_tool_request_fields(request_config)

            self._assert_no_tool_request_fields(request_config, is_ollama=is_ollama)

            return self._send_completion(
                client=client,
                model=self.model_name,
                messages=messages,
                config=request_config,
                is_ollama=is_ollama,
            )


        completion = _request_completion(request_messages, sanitized_config)
        last_completion: Optional[ChatCompletion] = completion

        self._log_inference_payload(
            messages=request_messages,
            config=self._strip_tool_request_fields(sanitized_config),
            completion=completion,
            allow_tools=allow_tools,
            is_ollama=is_ollama,
        )

        # Optional debug
        print("=== RAW COMPLETION OBJECT ===")
        print(completion.model_dump_json(indent=2))
        print("=== REQUEST PAYLOAD (model/messages/config) ===")
        print(
            json.dumps(
                {"model": self.model_name, "messages": base_messages, **sanitized_config},
                indent=2,
                default=str,
            )
        )

        out: list[str] = []
        is_os_prompt = self._is_os_interaction_prompt(base_messages)

        for choice in completion.choices:
            content = choice.message.content or ""
            tool_calls = getattr(choice.message, "tool_calls", None)

            if is_ollama:
                extracted = self._extract_ollama_tool_content(
                    content=content,
                    tool_calls=tool_calls,
                )
                if extracted is not None:
                    content = extracted
                    tool_calls = None

            # If tools are allowed (agent/controller mode) and content is empty but tool_calls exist,
            # preserve the tool call as <internal_tool ...>.
            # If tools are NOT allowed (benchmark-safe mode), DO NOT emit <internal_tool ...>;
            # instead, force a repair pass below.
            if (not content) and tool_calls:
                first = tool_calls[0]
                func = getattr(first, "function", None)
                name = getattr(func, "name", None)
                args = getattr(func, "arguments", None) or "{}"
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=True, default=str)
                content = f'<internal_tool name="{name}">{args}</internal_tool>'


            # OS-interaction bash fence repair (kept as-is)
            if is_os_prompt and self._needs_bash_block_repair(content):
                repair_messages = [
                    {"role": "system", "content": self._bash_block_repair_instruction()},
                    *list(base_messages),
                ]
                if is_ollama:
                    repair_messages = self._prepare_ollama_messages(repair_messages)
                repair = _request_completion(repair_messages, sanitized_config)
                last_completion = repair
                repair_content = repair.choices[0].message.content or ""
                if "```bash" in repair_content:
                    content = repair_content

            # # Benchmark-safe: force final output when empty / internal tool blocks show up.
            # if (not allow_tools) and self._needs_final_repair(content):
            #     if allow_internal_tool_protocol and self._is_internal_tool_block(content):
            #         out.append(content)
            #         continue
            #     repair_system = (
            #         self._ollama_output_repair_instruction()
            #         if is_ollama
            #         else self._final_output_repair_instruction()
            #     )
            #     repair_messages: list[ChatCompletionMessageParam] = [
            #         {"role": "system", "content": repair_system},
            #         *list(base_messages),
            #         # Provide the bad output so the model can correct itself deterministically.
            #         {"role": "assistant", "content": content or "<empty>"},
            #         {
            #             "role": "user",
            #             "content": "Repair your output to match the required final format. Output only the corrected final response.",
            #         },
            #     ]
            #     if is_ollama:
            #         repair_messages = self._prepare_ollama_messages(repair_messages)

            #     repair = _request_completion(repair_messages, sanitized_config)
            #     last_completion = repair
            #     repaired = repair.choices[0].message.content or ""
            #     if repaired.strip():
            #         if is_ollama:
            #             extracted = self._extract_ollama_tool_content(
            #                 content=repaired,
            #                 tool_calls=getattr(repair.choices[0].message, "tool_calls", None),
            #             )
            #             if extracted is not None:
            #                 content = extracted
            #             else:
            #                 content = repaired
            #         else:
            #             content = repaired

            out.append(content)

        logger.debug("Model outputs (repr): %s", [repr(x) for x in out])
        if return_completion:
            if last_completion is None:
                raise RuntimeError("No completion available for return.")
            return out, last_completion
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
            output_str_list.extend(
                self._get_completion_content(
                    message_list=message_list,
                    inference_config_dict=inference_config_dict,
                    allow_tools=False,
                )
            )

        return [ChatHistoryItem(role=Role.AGENT, content=s) for s in output_str_list]
