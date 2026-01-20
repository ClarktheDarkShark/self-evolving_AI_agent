import json
import os
import re
from typing import Any, Mapping, Optional, Sequence

from src.typings import ChatHistory, ChatHistoryItem, Role

from .tool_registry import ToolMetadata
from .tool_spec import ToolSpec
from .tool_validation import validate_tool_code
from .tool_retrieval import retrieve_tools


class ControllerToolgenMixin:
    def _toolgen_output_mode(self) -> str:
        value = os.environ.get("TOOLGEN_OUTPUT_MODE", "markers")
        mode = value.strip().lower()
        return "json" if mode == "json" else "markers"

    def _normalize_toolgen_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                if isinstance(part, Mapping)
                else str(part)
                for part in content
            )
        return ""

    def _extract_marked_python(self, text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("###TOOL_START")
        if start < 0:
            return None
        end = text.find("###TOOL_END", start + len("###TOOL_START"))
        if end < 0:
            return None
        return text[start + len("###TOOL_START"):end].strip() or None

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        if not text:
            return None
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse_creation_payload(self, payload: str) -> Optional[Mapping[str, Any]]:
        try:
            obj = json.loads(payload)
        except Exception:
            return None
        return obj if isinstance(obj, Mapping) else None

    def _extract_tool_call_arguments(self, response_obj: Any) -> Optional[str]:
        if response_obj is None:
            return None
        if isinstance(response_obj, Mapping):
            raw = response_obj.get("raw_output") or response_obj.get("arguments")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            tool_calls = response_obj.get("tool_calls")
            if tool_calls:
                first = tool_calls[0]
                if isinstance(first, Mapping):
                    func = first.get("function")
                    if isinstance(func, Mapping):
                        args = func.get("arguments")
                        if isinstance(args, str) and args.strip():
                            return args.strip()
            func_call = response_obj.get("function_call")
            if isinstance(func_call, Mapping):
                args = func_call.get("arguments")
                if isinstance(args, str) and args.strip():
                    return args.strip()
            return None
        tool_calls = getattr(response_obj, "tool_calls", None)
        if tool_calls:
            first = tool_calls[0]
            func = getattr(first, "function", None)
            args = getattr(func, "arguments", None) if func is not None else None
            if isinstance(args, str) and args.strip():
                return args.strip()
        func_call = getattr(response_obj, "function_call", None)
        if func_call is not None:
            args = getattr(func_call, "arguments", None)
            if isinstance(args, str) and args.strip():
                return args.strip()
        return None

    def extract_tool_spec(self, raw_text: str, response_obj: Any) -> dict[str, Any]:
        raw_text_full = raw_text or ""
        if not isinstance(raw_text_full, str):
            raise ValueError(f"raw_text must be str (got {type(raw_text_full).__name__})")
        obj = self._parse_creation_payload(raw_text_full)
        if obj is not None:
            return dict(obj)
        obj_text = self._extract_first_json_object(raw_text_full)
        if obj_text:
            obj = self._parse_creation_payload(obj_text)
            if obj is not None:
                return dict(obj)
        tool_args = self._extract_tool_call_arguments(response_obj)
        if tool_args:
            obj = self._parse_creation_payload(tool_args)
            if obj is not None:
                return dict(obj)
        preview = raw_text_full[:200]
        raise ValueError(f"tool_spec_parse_failed preview={preview!r}")

    def _toolgen_compact_existing_tools(self) -> list[dict[str, Any]]:
        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        return [
            {
                "name": tool.name,
                "tool_type": tool.tool_type,
                "tool_category": getattr(tool, "tool_category", None),
                "signature": tool.signature,
            }
            for tool in tools[-50:]
        ]

    def _toolgen_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        existing = []
        try:
            existing = self._toolgen_compact_existing_tools()
        except Exception:
            existing = []

        history_items = self._history_items(chat_history)[-6:]
        history_lines = []
        for i, it in enumerate(history_items):
            content = (it.content or "").strip().replace("\n", " ")
            content = self._truncate(content, 240)
            history_lines.append("{}:{}:{}".format(i, it.role.value, content))

        payload = {
            "task": self._truncate((query or "").strip(), 800),
            "history": "\n".join(history_lines),
            "existing_tools": existing,
            "output_mode": self._toolgen_output_mode(),
            "output_instructions": (
                "Output ONLY:\n###TOOL_START\n<python>\n###TOOL_END"
                if self._toolgen_output_mode() == "markers"
                else "Output a single JSON object ToolSpec."
            ),
            "requested_tool_name": getattr(self, "_toolgen_requested_tool_name", None),
            "requested_tool_category": getattr(self, "_toolgen_requested_tool_category", None),
        }
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _normalize_retrieval_query(self, query: str) -> str:
        return (query or "").strip()[:1200]

    def _join_code_lines(self, code_lines: Sequence[Any]) -> Optional[str]:
        normalized_lines: list[str] = []
        for line in code_lines:
            text = str(line)
            if "\r\n" in text:
                text = text.replace("\r\n", "\n")
            if "\n" in text:
                normalized_lines.extend(text.splitlines())
            else:
                normalized_lines.append(text)
        if not normalized_lines:
            return None
        return "\n".join(normalized_lines).rstrip() + "\n"

    def _wrap_marker_tool_spec(self, python_code: str) -> dict[str, Any]:
        name = getattr(self, "_toolgen_requested_tool_name", None) or "generated_tool"
        description = getattr(self, "_toolgen_requested_description", None) or "Generated tool."
        signature = getattr(self, "_toolgen_requested_signature", None) or "run(payload: dict) -> dict"
        tool_type = getattr(self, "_toolgen_requested_tool_type", None) or "utility"
        tool_category = getattr(self, "_toolgen_requested_tool_category", None) or "utility"
        input_schema = getattr(self, "_toolgen_requested_input_schema", None)
        if not isinstance(input_schema, Mapping):
            input_schema = {
                "type": "object",
                "required": ["payload"],
                "properties": {"payload": {"type": "object", "required": [], "properties": {}}},
            }
        capabilities = getattr(self, "_toolgen_requested_capabilities", None) or []
        return {
            "name": name,
            "description": description,
            "signature": signature,
            "tool_type": tool_type,
            "tool_category": tool_category,
            "input_schema": input_schema,
            "capabilities": capabilities,
            "code_lines": python_code.splitlines(),
        }

    def _normalize_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(spec)
        normalized.setdefault("name", "generated_tool")
        normalized.setdefault("description", "Generated tool.")
        normalized.setdefault("signature", "run(payload: dict) -> dict")
        normalized.setdefault("tool_type", "utility")
        normalized.setdefault("tool_category", "utility")
        normalized.setdefault(
            "input_schema",
            {
                "type": "object",
                "required": ["payload"],
                "properties": {"payload": {"type": "object", "required": [], "properties": {}}},
            },
        )
        normalized.setdefault("capabilities", [])
        return normalized

    def _validate_and_register_tool(
        self,
        spec: ToolSpec,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolMetadata]:
        code = self._join_code_lines(spec.code_lines or [])
        if not code:
            self._trace("tool_generation_error", "stage=code_join error=empty_code")
            return None
        try:
            result = validate_tool_code(code)
        except Exception as exc:
            self._trace("tool_generation_error", f"stage=validate_exception error={exc}")
            return None
        if not result or not result.success:
            error = getattr(result, "error", None) or "validate failed"
            self._trace("tool_generation_error", f"stage=validate error={error}")
            return None
        try:
            metadata = self._registry.register_tool(
                name=spec.name,
                code=code,
                signature=spec.signature,
                description=spec.description,
                tool_type=spec.tool_type,
                tool_category=spec.tool_category,
                input_schema=spec.input_schema,
                capabilities=spec.capabilities,
            )
        except Exception as exc:
            self._trace("tool_generation_error", f"stage=register_exception error={exc}")
            return None
        if metadata:
            self._tool_creation_successes += 1
        return metadata

    def _register_tool_from_payload(
        self, creation_request: Mapping[str, Any], chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            return None
        if not isinstance(creation_request, Mapping):
            return None
        tool_spec = self._normalize_tool_spec(dict(creation_request))
        code_lines = tool_spec.get("code_lines") or []
        if isinstance(code_lines, str):
            tool_spec["code_lines"] = code_lines.splitlines()
        spec = ToolSpec.from_payload(tool_spec)
        metadata = self._validate_and_register_tool(spec, chat_history, raw_spec=tool_spec)
        if metadata:
            self._generated_tool_counter += 1
            self._mark_tool_invoked()
        return metadata

    def _consider_tool_generation(
        self, query: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if not query.strip():
            return None
        if not self._force_tool_generation_if_missing:
            return None
        normalized = self._normalize_retrieval_query(query)
        if not normalized or normalized in self._toolgen_attempted_queries:
            return None
        self._toolgen_attempted_queries.add(normalized)
        return self._maybe_generate_tool_for_query(query, chat_history)

    def _maybe_generate_tool_for_query(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        allow_reuse: bool = True,
        force: bool = False,
    ) -> Optional[ToolMetadata]:
        if not query.strip():
            return None
        if allow_reuse:
            reuse_query = self._normalize_retrieval_query(query)
            candidate_output = self._get_candidate_output(chat_history, query)
            reuse = self._reuse_existing_tool(
                reuse_query, candidate_output=candidate_output, needed_archetype=None
            )
            if reuse is not None:
                return reuse

        if not self._force_tool_generation_if_missing and not force:
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run and not force:
            return None
        if getattr(self, "_toolgen_agent", None) is None:
            return None

        prompt = self._toolgen_request_prompt(query, chat_history)
        self._trace("tool_agent_input", prompt)
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response = self._toolgen_agent._inference(tool_history)
        self._trace("tool_agent_result", response.content)
        raw_text_full = self._normalize_toolgen_content(response.content)

        if self._toolgen_output_mode() == "markers":
            extracted = self._extract_marked_python(raw_text_full)
            if not extracted:
                return None
            tool_spec = self._wrap_marker_tool_spec(extracted)
            return self._register_tool_from_payload(tool_spec, chat_history)

        creation_request = self.extract_tool_spec(raw_text_full, response)
        return self._register_tool_from_payload(creation_request, chat_history)
