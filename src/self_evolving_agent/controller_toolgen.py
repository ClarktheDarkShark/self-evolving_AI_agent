#controller_toolgen.py

import datetime
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import ast

from src.typings import ChatHistory, ChatHistoryItem, Role

from .tool_registry import ToolMetadata
from .tool_spec import ToolSpec
from .tool_validation import validate_tool_code
from .tool_retrieval import retrieve_tools


class ControllerToolgenMixin:
    def _toolgen_output_mode(self) -> str:
        return "markers"

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
                "signature": tool.signature,
                "docstring": tool.docstring,
            }
            for tool in tools[-50:]
        ]

    def _toolgen_default_name(self) -> str:
        return "generated_tool"

    def _toolgen_default_description(self) -> str:
        query = getattr(self, "_toolgen_last_query", "") or ""
        summary = query.strip()
        return f"Utility tool for: {summary[:120]}" if summary else "Utility tool for the current task."

    def _extract_tool_name_from_code(self, python_code: str) -> Optional[str]:
        if not python_code:
            return None
        for raw_line in python_code.splitlines():
            line = raw_line.strip()
            if not line.startswith("#"):
                continue
            match = re.match(r"^#\s*tool_name\s*:\s*([a-zA-Z0-9_]+)\s*$", line)
            if not match:
                continue
            name = match.group(1).strip().lower()
            if name:
                if not name.endswith("_generated_tool"):
                    return f"{name}_generated_tool"
                return name
        return None

    def _toolgen_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        existing = []
        try:
            existing = self._toolgen_compact_existing_tools()
        except Exception:
            existing = []

        def _shorten_history_line(text: str) -> str:
            if not text:
                return ""
            if "Observation:" in text or "executes successfully" in text:
                head, sep, tail = text.partition("Observation:")
                if sep:
                    trimmed_tail = tail.strip()
                    if len(trimmed_tail) > 300:
                        trimmed_tail = trimmed_tail[:300] + "...[truncated_observation]"
                    return (head.strip() + " Observation: " + trimmed_tail).strip()
            if len(text) > 1000:
                return text[:1000] + "...[truncated]"
            return text

        history_items = self._history_items(chat_history)[-8:]
        history_lines = []
        for i, it in enumerate(history_items):
            content = (it.content or "").strip()
            content = _shorten_history_line(content)
            history_lines.append("{}:{}:{}".format(i, it.role.value, content))

        payload = {
            "task": (query or "").strip(),
            "task_requirement": (
                "Tool must directly help solve the current task; generic tools are invalid."
            ),
            "history": "\n".join(history_lines),
            "existing_tools": existing,
        }
        solver_recommendation = (
            getattr(self, "_toolgen_last_recommendation", "") or ""
        ).strip()
        if solver_recommendation:
            # if len(solver_recommendation) > 1200:
            #     solver_recommendation = solver_recommendation[:1200] + "...[truncated]"
            payload["solver_recommendation"] = solver_recommendation
            payload["recommendation_note"] = (
                "Solver provided a draft response. Use it to design a tool that "
                "validates or strengthens the draft for this task."
            )

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
        name = self._extract_tool_name_from_code(python_code) or self._toolgen_default_name()
        description = self._toolgen_default_description()
        signature = "run(payload: dict) -> dict"
        tool_type = "utility"
        tool_category = "utility"
        input_schema = {
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {
                "type": "object",
                "required": ["task_text", "asked_for", "trace", "actions_spec"],
                "properties": {
                    "task_text": {"type": "string"},
                    "asked_for": {"type": "string"},
                    "trace": {"type": "array"},
                    "actions_spec": {"type": "object"},
                    "constraints": {"type": "array"},
                    "output_contract": {"type": "object"},
                    "draft_response": {"type": ["string","null"]},
                    "candidate_output": {},
                    "env_observation": {},},
                }
            }
        }


        capabilities = []
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
        normalized.setdefault("name", self._toolgen_default_name())
        name = str(normalized.get("name") or "").strip().lower()
        if name and not name.endswith("_generated_tool"):
            normalized["name"] = f"{name}_generated_tool"
        normalized.setdefault("description", self._toolgen_default_description())
        normalized.setdefault("signature", "run(payload: dict) -> dict")
        normalized.setdefault("tool_type", "utility")
        normalized.setdefault("tool_category", "utility")
        normalized.setdefault(
            "input_schema",
            {
                "type": "object",
                "required": ["payload"],
                "properties": {
                    "payload": {
                    "type": "object",
                    "required": ["task_text", "asked_for", "trace", "actions_spec"],
                    "properties": {
                        "task_text": {"type": "string"},
                        "asked_for": {"type": "string"},
                        "trace": {"type": "array"},
                        "actions_spec": {"type": "object"},
                        "constraints": {"type": "array"},
                        "output_contract": {"type": "object"},
                        "draft_response": {"type": ["string","null"]},
                        "candidate_output": {},
                        "env_observation": {},}
                    },
                }
            },
        )
        normalized.setdefault("capabilities", [])
        schema = normalized.get("input_schema")
        if isinstance(schema, Mapping):
            props = schema.get("properties") or {}
            payload_schema = props.get("payload")
            if isinstance(payload_schema, Mapping):
                required = payload_schema.get("required") or []
                if not isinstance(required, list):
                    required = []
                required_keys = ["task_text", "asked_for", "trace", "actions_spec"]
                for key in required_keys:
                    if key not in required:
                        required.append(key)
                payload_schema["required"] = required
                properties = payload_schema.get("properties") or {}
                if "task_text" not in properties:
                    properties["task_text"] = {"type": "string"}
                if "asked_for" not in properties:
                    properties["asked_for"] = {"type": "string"}
                if "trace" not in properties:
                    properties["trace"] = {"type": "array"}
                if "actions_spec" not in properties:
                    properties["actions_spec"] = {"type": "object"}
                payload_schema["properties"] = properties
                props["payload"] = payload_schema
                schema["properties"] = props
                normalized["input_schema"] = schema
        return normalized

    def _validate_spec_alignment(self, spec: ToolSpec) -> Optional[str]:
        if spec.signature.strip() != "run(payload: dict) -> dict":
            return "signature_mismatch"
        schema = spec.input_schema
        if not isinstance(schema, Mapping):
            return "missing_input_schema"
        if schema.get("type") != "object":
            return "input_schema_type_mismatch"
        required = schema.get("required") or []
        if required != ["payload"]:
            return "input_schema_required_mismatch"
        payload_schema = (schema.get("properties") or {}).get("payload")
        if not isinstance(payload_schema, Mapping):
            return "payload_schema_missing"
        if payload_schema.get("type") != "object":
            return "payload_schema_type_mismatch"
        if "required" not in payload_schema:
            return "payload_required_missing"
        return None

    def _validate_run_ast(self, code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
        except Exception as exc:
            return f"ast_parse_failed: {exc}"
        run_fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_fn = node
                break
        if run_fn is None:
            return "run_not_found"
        args = run_fn.args
        total_args = list(args.posonlyargs) + list(args.args)
        if len(total_args) != 1 or total_args[0].arg != "payload":
            return "run_signature_mismatch"
        if args.vararg or args.kwarg or args.kwonlyargs:
            return "run_signature_mismatch"
        return None

    def _failed_tool_log_dir(self) -> Path:
        base_path = None
        log_path = getattr(self, "_generated_tools_log_path", None)
        if log_path is not None:
            base_path = Path(log_path).parent
        if base_path is None:
            base_path = Path("outputs")
        return base_path / "callback_state" / "callback_generated_tool_logging"

    def _write_failed_tool_artifact(
        self,
        *,
        stage: str,
        error: str,
        spec: Optional[ToolSpec] = None,
        code: Optional[str] = None,
        raw_spec: Optional[Mapping[str, Any]] = None,
        raw_output: Optional[str] = None,
    ) -> None:
        try:
            tool_name = (spec.name if spec else None) or "unknown_tool"
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
            suffix = "py" if code else "txt"
            filename = f"{tool_name}__{stage}__{ts}.{suffix}"
            out_dir = self._failed_tool_log_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            header = (
                f"# stage: {stage}\n"
                f"# tool_name: {tool_name}\n"
                f"# signature: {(spec.signature if spec else '')}\n"
                f"# error: {error}\n"
            )
            if code:
                content = header + "\n" + code
            else:
                meta = {
                    "stage": stage,
                    "tool_name": tool_name,
                    "signature": spec.signature if spec else None,
                    "error": error,
                    "raw_spec": raw_spec,
                    "raw_output": raw_output,
                }
                content = header + "\n" + json.dumps(meta, ensure_ascii=True, default=str, indent=2)
            (out_dir / filename).write_text(content, encoding="utf-8")
        except Exception:
            return

    def _validate_and_register_tool(
        self,
        spec: ToolSpec,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolMetadata]:
        alignment_error = self._validate_spec_alignment(spec)
        if alignment_error:
            self._trace("tool_generation_error", f"stage=spec_alignment error={alignment_error}")
            self._write_failed_tool_artifact(
                stage="spec_alignment",
                error=alignment_error,
                spec=spec,
                raw_spec=raw_spec,
            )
            return None
        code = self._join_code_lines(spec.code_lines or [])
        if not code:
            self._trace("tool_generation_error", "stage=code_join error=empty_code")
            self._write_failed_tool_artifact(
                stage="code_join",
                error="empty_code",
                spec=spec,
                raw_spec=raw_spec,
            )
            return None
        ast_error = self._validate_run_ast(code)
        if ast_error:
            self._trace("tool_generation_error", f"stage=run_signature error={ast_error}")
            self._write_failed_tool_artifact(
                stage="run_signature",
                error=ast_error,
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        try:
            result = validate_tool_code(code)
        except Exception as exc:
            self._trace("tool_generation_error", f"stage=validate_exception error={exc}")
            self._write_failed_tool_artifact(
                stage="validate_exception",
                error=str(exc),
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        if not result or not result.success:
            error = getattr(result, "error", None) or "validate failed"
            self._trace("tool_generation_error", f"stage=validate error={error}")
            self._write_failed_tool_artifact(
                stage="validate",
                error=error,
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
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
            self._write_failed_tool_artifact(
                stage="register_exception",
                error=str(exc),
                spec=spec,
                code=code,
                raw_spec=raw_spec,
            )
            return None
        if metadata:
            self._tool_creation_successes += 1
            try:
                payload = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "event": "create",
                    "tool_name": metadata.name,
                    "signature": metadata.signature,
                    "description": metadata.description,
                    "tool_type": metadata.tool_type,
                    "tool_category": metadata.tool_category,
                    "input_schema": metadata.input_schema,
                    "capabilities": metadata.capabilities,
                    "path": getattr(self._registry, "_get_tool_path", lambda n: None)(metadata.name),
                }
                payload.update(self._get_run_task_metadata())
                self._append_generated_tools_log(payload)
            except Exception:
                pass
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
            if reuse is not None and self._reuse_matches_request(reuse):
                return reuse

        if not self._force_tool_generation_if_missing and not force:
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run and not force:
            return None
        if getattr(self, "_toolgen_agent", None) is None:
            return None

        self._toolgen_last_query = query
        prompt = self._toolgen_request_prompt(query, chat_history)
        self._write_agent_system_prompt(
            "toolgen",
            getattr(self._toolgen_agent, "_system_prompt", "") or "",
        )
        self._trace("tool_agent_input", prompt)
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )

        response = self._toolgen_agent._inference(tool_history)

        self._trace("tool_agent_result", response.content)
        # print()
        # print("*************************************************************")
        # print('Tool agent response', response.content)
        # print("*************************************************************")
        # print()
        raw_text_full = self._normalize_toolgen_content(response.content)

        extracted = self._extract_marked_python(raw_text_full)
        if not extracted:
            self._write_failed_tool_artifact(
                stage="toolgen_markers_missing",
                error="marker_block_not_found",
                raw_output=raw_text_full,
            )
            return None
        tool_spec = self._wrap_marker_tool_spec(extracted)
        return self._register_tool_from_payload(tool_spec, chat_history)

    def _reuse_matches_request(self, tool: ToolMetadata) -> bool:
        return True
