import ast
import hashlib
import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import yaml  # add pyyaml dependency if not already present
except Exception:
    yaml = None

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

    def _toolgen_test_mode(self) -> bool:
        value = os.environ.get("TOOLGEN_TEST_MODE") or os.environ.get("TEST", "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _toolgen_env_int(self, key: str, default: int) -> int:
        raw = os.environ.get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    def _broken_tools_dir(self) -> Path:
        base = None
        if getattr(self, "_generated_tools_log_path", None) is not None:
            base = Path(self._generated_tools_log_path).parent
        elif getattr(self, "_tool_invocation_log_path", None) is not None:
            base = Path(self._tool_invocation_log_path).parent
        else:
            base = Path("outputs")
        return base / "broken_tools"

    def _toolgen_failure_dir(self, tool_name: Optional[str]) -> Path:
        out_dir = self._broken_tools_dir()
        safe_name = self._sanitize_tool_filename(tool_name or "toolgen")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{ts}_{safe_name}"
        counter = 1
        while path.exists():
            path = out_dir / f"{ts}_{safe_name}_{counter}"
            counter += 1
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sanitize_tool_filename(self, name: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name or "tool")
        return text[:80] if len(text) > 80 else text

    def _persist_broken_tool(
        self,
        *,
        stage: str,
        error: str,
        spec: ToolSpec,
        raw_spec: Optional[Mapping[str, Any]],
        code: str,
    ) -> None:
        if not code:
            return
        try:
            out_dir = self._broken_tools_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = self._sanitize_tool_filename(spec.name or "tool")
            safe_stage = self._sanitize_tool_filename(stage or "unknown")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            filename = f"{safe_name}__{safe_stage}__{ts}.py"
            path = out_dir / filename
            counter = 1
            while path.exists():
                path = out_dir / f"{safe_name}__{safe_stage}__{ts}_{counter}.py"
                counter += 1
            raw_dump = json.dumps(
                raw_spec if raw_spec is not None else spec.__dict__,
                ensure_ascii=True,
                default=str,
            )
            header = (
                "# BROKEN TOOL\n"
                f"# stage: {stage}\n"
                f"# error: {error}\n"
                f"# name: {spec.name}\n"
                f"# signature: {spec.signature}\n"
                f"# raw_spec: {raw_dump}\n\n"
            )
            path.write_text(header + code, encoding="utf-8")
        except Exception:
            return

    def _persist_broken_tool_output(
        self,
        *,
        stage: str,
        error: str,
        raw_output: str,
        tool_name: Optional[str] = None,
    ) -> None:
        try:
            out_dir = self._broken_tools_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = self._sanitize_tool_filename(tool_name or "toolgen")
            safe_stage = self._sanitize_tool_filename(stage or "unknown")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            filename = f"{safe_name}__{safe_stage}__{ts}.txt"
            path = out_dir / filename
            counter = 1
            while path.exists():
                path = out_dir / f"{safe_name}__{safe_stage}__{ts}_{counter}.txt"
                counter += 1
            header = (
                "# BROKEN TOOL OUTPUT\n"
                f"# stage: {stage}\n"
                f"# error: {error}\n"
                f"# tool_name: {tool_name}\n\n"
            )
            body = raw_output or "<empty>"
            path.write_text(header + body, encoding="utf-8")
        except Exception:
            return

    def _persist_toolgen_failure_artifacts(
        self,
        *,
        stage: str,
        error: str,
        tool_name: Optional[str] = None,
        prompt: Optional[str] = None,
        raw_output: Optional[str] = None,
        extracted_python: Optional[str] = None,
    ) -> None:
        try:
            tool_name = tool_name or getattr(self, "_toolgen_requested_tool_name", None)
            prompt = prompt if prompt is not None else getattr(self, "_toolgen_last_prompt", "")
            raw_output = raw_output if raw_output is not None else getattr(self, "_toolgen_last_raw_output", "")
            extracted_python = (
                extracted_python
                if extracted_python is not None
                else getattr(self, "_toolgen_last_extracted_python", "")
            )
            out_dir = self._toolgen_failure_dir(tool_name)
            (out_dir / "error.txt").write_text(
                f"stage: {stage}\nerror: {error}\n", encoding="utf-8"
            )
            (out_dir / "prompt.txt").write_text(prompt or "<empty>", encoding="utf-8")
            (out_dir / "raw_output.txt").write_text(raw_output or "<empty>", encoding="utf-8")
            if extracted_python:
                (out_dir / "extracted.py").write_text(extracted_python, encoding="utf-8")
        except Exception:
            return

    def _extract_marked_python(self, text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("###TOOL_START")
        if start < 0:
            return None
        end = text.find("###TOOL_END", start + len("###TOOL_START"))
        if end < 0:
            return None
        body = text[start + len("###TOOL_START"):end]
        body = body.strip()
        return body or None

    def _normalize_toolgen_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return ""

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """
        Return the first top-level JSON object substring from text.
        Handles leading prose, code fences, and trailing junk.
        """
        if not text:
            return None

        s = text.strip()

        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()

        start = s.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]

        return None

    def _escape_invalid_json_backslashes(self, text: str) -> str:
        if not text:
            return text
        out: list[str] = []
        in_str = False
        esc = False
        i = 0
        while i < len(text):
            ch = text[i]
            if not in_str:
                if ch == '"':
                    in_str = True
                out.append(ch)
                i += 1
                continue

            if esc:
                out.append(ch)
                esc = False
                i += 1
                continue

            if ch == "\\":
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if nxt in ('"', "\\", "/", "b", "f", "n", "r", "t", "u"):
                    out.append(ch)
                    esc = True
                else:
                    out.append("\\\\")
                i += 1
                continue

            if ch == '"':
                in_str = False
            out.append(ch)
            i += 1

        return "".join(out)

    def _escape_control_chars_in_json_strings(self, text: str) -> str:
        if not text:
            return text
        out: list[str] = []
        in_str = False
        esc = False
        for ch in text:
            if not in_str:
                if ch == '"':
                    in_str = True
                out.append(ch)
                continue

            if esc:
                out.append(ch)
                esc = False
                continue

            if ch == "\\":
                out.append(ch)
                esc = True
                continue

            if ch == '"':
                in_str = False
                out.append(ch)
                continue

            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                continue

            out.append(ch)

        return "".join(out)

    def _close_truncated_json(self, text: str) -> Optional[str]:
        if not text:
            return None
        s = text.strip()
        if not s.startswith("{"):
            return None
        stack: list[str] = []
        in_str = False
        esc = False
        out: list[str] = []
        for ch in s:
            out.append(ch)
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
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]"):
                if stack and ch == stack[-1]:
                    stack.pop()
        if esc:
            out.append("\\")
        if in_str:
            out.append('"')
        while stack:
            out.append(stack.pop())
        return "".join(out)

    def _parse_creation_payload(self, payload: str) -> Optional[Mapping[str, Any]]:
        try:
            obj = json.loads(payload)
            if isinstance(obj, Mapping):
                return obj
        except Exception:
            pass

        if yaml is not None:
            try:
                obj = yaml.safe_load(payload)
                if isinstance(obj, Mapping):
                    return obj
            except Exception:
                pass

        try:
            obj = ast.literal_eval(payload)
            if isinstance(obj, Mapping):
                return obj
        except Exception:
            pass

        return None

    def _required_tool_spec_keys(self) -> list[str]:
        return [
            "name",
            "description",
            "signature",
            "tool_type",
            "tool_category",
            "input_schema",
            "capabilities",
            "code_lines",
        ]

    def _tool_spec_has_required_keys(self, obj: Any) -> bool:
        if not isinstance(obj, Mapping):
            return False
        required = self._required_tool_spec_keys()
        return all(key in obj for key in required)

    def _repair_misnested_tool_spec(self, obj: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(obj, Mapping):
            return obj
        input_schema = obj.get("input_schema")
        if not isinstance(input_schema, Mapping):
            return obj
        required = self._required_tool_spec_keys()
        moved: dict[str, Any] = {}
        for key in required:
            if key == "input_schema" or key in obj:
                continue
            if key in input_schema:
                moved[key] = input_schema[key]
        if not moved:
            return obj
        updated = dict(obj)
        updated_schema = dict(input_schema)
        for key in moved:
            updated_schema.pop(key, None)
        updated.update(moved)
        updated["input_schema"] = updated_schema
        self._toolgen_debug_event(
            "toolgen_parse_repaired",
            reason="hoist_keys_from_input_schema",
            moved_keys=list(moved.keys()),
        )
        return updated

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
                func = None
                if isinstance(first, Mapping):
                    func = first.get("function")
                    if isinstance(func, Mapping):
                        args = func.get("arguments")
                        if isinstance(args, str) and args.strip():
                            return args.strip()
                return None

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
        if '"output_preview"' in raw_text_full and '"output_truncated"' in raw_text_full:
            raise ValueError("raw_text appears to be a truncated preview, not full output")

        parsed_from: Optional[str] = None

        def _parse_json_text(text: str, source_label: str) -> Optional[Mapping[str, Any]]:
            nonlocal parsed_from
            try:
                obj = json.loads(text)
            except Exception:
                sanitized = self._escape_invalid_json_backslashes(text)
                sanitized = self._escape_control_chars_in_json_strings(sanitized)
                if sanitized != text:
                    try:
                        obj = json.loads(sanitized)
                        parsed_from = source_label + "_sanitized"
                    except Exception:
                        obj = None
                else:
                    obj = None
            if obj is None:
                fallback = self._parse_creation_payload(text)
                if fallback is None and "sanitized" in locals():
                    fallback = self._parse_creation_payload(sanitized)
                if isinstance(fallback, Mapping):
                    fallback = self._repair_misnested_tool_spec(fallback)
                    if self._tool_spec_has_required_keys(fallback):
                        parsed_from = source_label + "_fallback_parse"
                        self._toolgen_debug_event(
                            "toolgen_parse_fallback_used",
                            source=source_label,
                            parsed_from=parsed_from,
                        )
                        return fallback
                return None
            if isinstance(obj, Mapping):
                obj = self._repair_misnested_tool_spec(obj)
                if self._tool_spec_has_required_keys(obj):
                    if parsed_from is None:
                        parsed_from = source_label
                    return obj
                wrapped = obj.get("content")
                if isinstance(wrapped, str) and wrapped.strip():
                    inner = wrapped.strip()
                    inner_obj = _parse_json_text(inner, "wrapper_content_unwrap")
                    if inner_obj is not None:
                        parsed_from = "wrapper_content_unwrap"
                        return inner_obj
            return None

        obj = _parse_json_text(raw_text_full, "full_message_content")
        if obj is not None:
            self._last_toolgen_parse_source = parsed_from
            return dict(obj)

        sanitized_full = self._escape_invalid_json_backslashes(raw_text_full)
        sanitized_full = self._escape_control_chars_in_json_strings(sanitized_full)

        obj_text = self._extract_first_json_object(sanitized_full)
        if obj_text:
            obj = _parse_json_text(obj_text, "balanced_json_extraction")
            if obj is not None:
                self._last_toolgen_parse_source = parsed_from or "balanced_json_extraction"
                return dict(obj)

        repaired = self._close_truncated_json(sanitized_full)
        if repaired:
            obj = _parse_json_text(repaired, "closed_truncated_json")
            if obj is not None:
                self._last_toolgen_parse_source = parsed_from or "closed_truncated_json"
                return dict(obj)

        tool_args = self._extract_tool_call_arguments(response_obj)
        if tool_args:
            obj = _parse_json_text(tool_args, "tool_calls_args")
            if obj is not None:
                self._last_toolgen_parse_source = parsed_from or "tool_calls_args"
                return dict(obj)

        preview = raw_text_full[:200]
        truncated = raw_text_full.endswith("...[truncated]") or "[truncated]" in raw_text_full
        raise ValueError(
            "tool_spec_parse_failed "
            f"preview={preview!r} truncated={truncated} parsed_from={parsed_from!r}"
        )

    def _toolgen_compact_existing_tools(self) -> list[dict[str, Any]]:
        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )

        compact: list[dict[str, Any]] = []
        for t in tools:
            compact.append(
                {
                    "name": t.name,
                    "tool_type": t.tool_type,
                    "tool_category": getattr(t, "tool_category", None),
                    "signature": t.signature,
                }
            )

        return compact[-50:]

    def _toolgen_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        existing = []
        try:
            existing = self._toolgen_compact_existing_tools()
        except Exception:
            existing = []

        compact_query = self._toolgen_compact_query(query)
        output_mode = self._toolgen_output_mode()

        test_mode = self._toolgen_test_mode()
        history_k = self._toolgen_env_int("TOOLGEN_HISTORY_K", 4 if test_mode else 8)
        tools_k = self._toolgen_env_int("TOOLGEN_TOOLS_K", 5 if test_mode else 20)
        task_chars = self._toolgen_env_int("TOOLGEN_TASK_CHARS", 400 if test_mode else 900)
        max_lines = self._toolgen_env_int("TOOLGEN_MAX_LINES", 60 if test_mode else 90)

        requested_tool_name = getattr(self, "_toolgen_requested_tool_name", None)
        requested_tool_name_for_prompt = None
        if requested_tool_name:
            requested_tool_name_for_prompt = requested_tool_name.split("__", 1)[0]
        requested_tool_category = getattr(self, "_toolgen_requested_tool_category", None) or "validator"
        requested_capabilities = getattr(self, "_toolgen_requested_capabilities", None) or [
            "validate_sql_select",
            "check_columns",
            "check_tables",
            "check_limit",
        ]

        design_goal = (
            "Design a HIGH-ROI reusable tool to reduce repeated failures. "
            "Prefer validators/normalizers/formatters that work across many tasks."
        )
        if requested_tool_category == "validator":
            name_hint = requested_tool_name_for_prompt or requested_tool_name
            if name_hint and "select" in name_hint.lower():
                design_goal += " Output a SELECT VALIDATOR; do not output a builder/parser."
            else:
                design_goal += " Output a VALIDATOR; do not output a builder/parser."
        if output_mode == "markers":
            design_goal += (
                " OUTPUT FORMAT: first line ###TOOL_START, then raw Python source, "
                "then last line ###TOOL_END. Output ONLY those lines."
            )

        history_items = self._history_items(chat_history)[-history_k:]
        mini_lines = []
        for i, it in enumerate(history_items):
            content = (it.content or "").strip().replace("\n", " ")
            content = self._truncate(content, 280)
            mini_lines.append("{}:{}:{}".format(i, it.role.value, content))
        mini_history = "\n".join(mini_lines)

        payload = {
            "existing_tools": existing[-tools_k:],
            "user_task": self._truncate(compact_query, task_chars),
            "history_hint": mini_history,
            "design_goal": design_goal,
            "output_mode": output_mode,
            "output_instructions": (
                "Output ONLY:\n###TOOL_START\n<python>\n###TOOL_END\nNo JSON."
                if output_mode == "markers"
                else "Output a single JSON object ToolSpec."
            ),
            "requested_tool_name": requested_tool_name_for_prompt,
            "requested_tool_category": requested_tool_category,
            "requested_capabilities": requested_capabilities,
            "hard_constraints": {
                "stdlib_only": True,
                "max_lines": max_lines,
                "deterministic": True,
                "self_test_required": False if test_mode else True,
            },
        }
        if output_mode == "json":
            payload["hard_constraints"]["output_keys"] = [
                "name","description","signature","tool_type","tool_category",
                "input_schema","capabilities","code_lines"
            ]
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _normalize_retrieval_query(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""

        lines = q.splitlines()
        tail = lines[-40:]
        drop_prefixes = (
            "i will ask you a question",
            "you have to explain",
            "after thinking",
            "to do operation",
            "you must put sql",
            "every time you",
            "if the sql",
            "if you have obtain",
            "we note that",
            "once you commit",
            "now, i will give you",
        )

        kept = []
        for line in tail:
            l = line.strip().lower()
            if any(l.startswith(p) for p in drop_prefixes):
                continue
            kept.append(line.strip())

        compact = "\n".join([x for x in kept if x])[:1200]
        return compact

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
        normalized_lines = self._ensure_tool_header(normalized_lines)
        if not normalized_lines:
            return None
        return "\n".join(normalized_lines).rstrip() + "\n"

    def _explode_code_lines(self, code_lines: Sequence[Any]) -> list[str]:
        exploded: list[str] = []
        for line in code_lines or []:
            text = str(line)
            if "\r\n" in text:
                text = text.replace("\r\n", "\n")
            if "\n" in text:
                exploded.extend(text.splitlines())
            else:
                exploded.append(text)
        return exploded

    def _is_placeholder_line(self, line: str) -> bool:
        text = (line or "").strip()
        if not text:
            return False
        if "<generated>" in text or "[truncated]" in text:
            return True
        return text in {"...", "…"}

    def _normalize_module_docstring_lines(
        self, lines: Sequence[str], description: Optional[str] = None
    ) -> list[str]:
        normalized = list(lines)
        idx = 0
        while idx < len(normalized) and normalized[idx].strip() == "":
            idx += 1
        normalized = normalized[idx:]

        desc = (description or "").strip()
        if not desc:
            desc = "Reusable tool."
        desc_line = desc.splitlines()[0].strip()
        if not desc_line:
            desc_line = "Reusable tool."
        desc_line = desc_line.replace('"""', '"').replace("'''", "'")
        if len(desc_line) > 120:
            desc_line = desc_line[:117] + "..."

        nonempty_idx = [i for i, line in enumerate(normalized) if line.strip()]
        matches = False
        if len(nonempty_idx) >= 3:
            first = normalized[nonempty_idx[0]].strip()
            second = normalized[nonempty_idx[1]].strip()
            third = normalized[nonempty_idx[2]].strip()
            if (
                first == '"""'
                and third == '"""'
                and second
                and '"""' not in second
                and "'''" not in second
            ):
                matches = True

        if not matches:
            normalized = ['"""', desc_line, '"""', ""] + normalized
            nonempty_idx = [i for i, line in enumerate(normalized) if line.strip()]

        if len(nonempty_idx) < 3:
            raise ValueError("docstring_invariant_failed: missing 3-line docstring")

        allowed = {nonempty_idx[0], nonempty_idx[2]}
        for i, line in enumerate(normalized):
            if "'''" in line:
                raise ValueError("docstring_invariant_failed: found '''")
            if '"""' in line and i not in allowed:
                raise ValueError("docstring_invariant_failed: extra triple quotes")

        return normalized

    def _ensure_tool_header(self, lines: list[str]) -> list[str]:
        # IMPORTANT:
        # Do NOT inject any `from __future__` imports.
        # They have strict placement rules and your post-processing can reorder lines
        # (e.g., inserting `import re`) and cause compile failures.
        if not lines:
            return lines

        header = "from __future__ import annotations"

        # Strip it if the model emitted it anywhere.
        lines = [ln for ln in lines if str(ln).strip() != header]

        return self._ensure_tool_imports(lines)

    def _normalize_code_lines(self, code_lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in code_lines or []:
            if not isinstance(line, str):
                line = str(line)
            s = line.strip("\n")
            if s.startswith("```"):
                continue
            if s.startswith('"""') or s.startswith("'''"):
                out.append(s)
                continue
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                inner = s[1:-1]
                inner = inner.replace(r"\\", "\\")
                inner = inner.replace(r"\'", "'").replace(r"\"", '"')
                s = inner
            out.append(s)

        while out and not out[0].strip():
            out.pop(0)
        while out and not out[-1].strip():
            out.pop()
        return out

    def _ensure_self_test(self, code_lines: list[str], input_schema: dict) -> list[str]:
        if any("def self_test" in (ln or "") for ln in code_lines):
            return code_lines

        payload_props = (
            (input_schema or {})
            .get("properties", {})
            .get("payload", {})
            .get("properties", {})
        )
        payload_required = (
            (input_schema or {})
            .get("properties", {})
            .get("payload", {})
            .get("required", [])
        )

        def _placeholder(v: dict) -> object:
            t = (v or {}).get("type")
            if t == "string":
                return "x"
            if t == "integer":
                return 1
            if t == "number":
                return 1
            if t == "boolean":
                return True
            if t == "array":
                return []
            if t == "object":
                return {}
            return "x"

        smoke: dict[str, Any] = {}
        for k in payload_required:
            smoke[k] = _placeholder(payload_props.get(k, {}))

        code_lines = list(code_lines)
        code_lines += [
            "",
            "def self_test() -> bool:",
            "    try:",
            f"        out = run({repr(smoke)})",
            "        return isinstance(out, dict)",
            "    except Exception:",
            "        return False",
        ]
        return code_lines

    def _normalize_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        required = [
            "name",
            "description",
            "signature",
            "tool_type",
            "tool_category",
            "input_schema",
            "capabilities",
            "code_lines",
        ]
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(f"missing required fields: {missing}")

        code_lines_before = list(spec.get("code_lines") or [])
        spec["code_lines"] = self._normalize_code_lines(code_lines_before)
        if self._toolgen_test_mode():
            spec["code_lines"] = self._ensure_self_test(
                spec["code_lines"], spec.get("input_schema") or {}
            )
        else:
            category = str(spec.get("tool_category") or "").lower()
            if category not in {"validator", "linter"}:
                spec["code_lines"] = self._ensure_self_test(
                    spec["code_lines"], spec.get("input_schema") or {}
                )
        return spec

    def _wrap_marker_tool_spec(self, python_code: str) -> dict[str, Any]:
        name = getattr(self, "_toolgen_requested_tool_name", None) or "generated_tool"
        description = (
            getattr(self, "_toolgen_requested_description", None)
            or "Generated tool."
        )
        signature = (
            getattr(self, "_toolgen_requested_signature", None)
            or "run(payload: dict) -> dict"
        )
        tool_type = (
            getattr(self, "_toolgen_requested_tool_type", None)
            or "utility"
        )
        tool_category = (
            getattr(self, "_toolgen_requested_tool_category", None)
            or "utility"
        )
        input_schema = getattr(self, "_toolgen_requested_input_schema", None)
        if not isinstance(input_schema, Mapping):
            input_schema = {
                "type": "object",
                "required": ["payload"],
                "properties": {
                    "payload": {"type": "object", "required": [], "properties": {}}
                },
            }
        capabilities = getattr(self, "_toolgen_requested_capabilities", None) or []
        code_lines = python_code.splitlines()
        return {
            "name": name,
            "description": description,
            "signature": signature,
            "tool_type": tool_type,
            "tool_category": tool_category,
            "input_schema": input_schema,
            "capabilities": capabilities,
            "code_lines": code_lines,
        }

    def _ensure_tool_imports(self, lines: list[str]) -> list[str]:
        needs_re = any("re." in line for line in lines)
        has_re = any(
            line.strip().startswith("import re")
            or line.strip().startswith("from re import")
            for line in lines
        )
        if not (needs_re and not has_re):
            return lines

        def _first_nonempty_index(ls: list[str]) -> int:
            i = 0
            while i < len(ls) and str(ls[i]).strip() == "":
                i += 1
            return i

        def _docstring_end_index(ls: list[str], start: int) -> Optional[int]:
            # Detect a top-of-file module docstring block:
            #   """
            #   ...
            #   """
            if start >= len(ls):
                return None
            first = str(ls[start]).strip()
            if first not in {'"""', "'''"}:
                return None
            quote = first
            j = start + 1
            while j < len(ls):
                if str(ls[j]).strip() == quote:
                    return j
                j += 1
            return None

        insert_at = _first_nonempty_index(lines)

        # If a module docstring exists at the top, insert AFTER it (and following blank lines).
        end = _docstring_end_index(lines, insert_at)
        if end is not None:
            insert_at = end + 1
            while insert_at < len(lines) and str(lines[insert_at]).strip() == "":
                insert_at += 1

        # Final guard: never insert above any future import (if one somehow survived).
        while insert_at < len(lines) and str(lines[insert_at]).strip().startswith("from __future__ import"):
            insert_at += 1
        while insert_at < len(lines) and str(lines[insert_at]).strip() == "":
            insert_at += 1

        lines.insert(insert_at, "import re")
        return lines

    def _validate_spec_alignment(self, spec: ToolSpec) -> Optional[str]:
        if not spec.signature:
            return "missing signature"

        if spec.signature.strip() != "run(payload: dict) -> dict":
            return f"signature must be exactly run(payload: dict) -> dict (got {spec.signature!r})"

        schema = spec.input_schema if isinstance(spec.input_schema, Mapping) else None
        if schema is None:
            return "missing input_schema"

        if schema.get("type") != "object":
            return "input_schema.type must be 'object'"

        required = schema.get("required") or []
        if required != ["payload"]:
            return f"input_schema.required must be ['payload'] (got {required})"

        props = schema.get("properties") or {}
        payload_schema = props.get("payload")
        if not isinstance(payload_schema, Mapping):
            return "input_schema.properties.payload must exist and be an object schema"

        if payload_schema.get("type") != "object":
            return "input_schema.properties.payload.type must be 'object'"

        if "properties" not in payload_schema:
            return "input_schema.properties.payload.properties is required"
        if "required" not in payload_schema:
            return "input_schema.properties.payload.required is required"

        try:
            dumped = json.dumps(schema, ensure_ascii=True, default=str)
            if '"set"' in dumped or "'set'" in dumped:
                return "schema contains forbidden key name 'set' (use 'set_values')"
        except Exception:
            pass

        return None

    def _validate_and_register_tool(
        self,
        spec: ToolSpec,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolMetadata]:
        test_mode = self._toolgen_test_mode()
        max_attempts = 3
        max_lines = self._toolgen_env_int(
            "TOOLGEN_MAX_LINES", 60 if test_mode else 90
        )
        last_error: Optional[str] = None
        current_raw_spec = raw_spec if raw_spec is not None else spec.__dict__

        for attempt in range(1, max_attempts + 1):
            self._tool_creation_attempts += 1

            alignment_error = self._validate_spec_alignment(spec)
            if alignment_error:
                last_error = str(alignment_error)
                self._trace("tool_generation_error", f"stage=spec_alignment error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="spec_alignment",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                raw_code = self._join_code_lines(spec.code_lines or [])
                if raw_code:
                    self._persist_broken_tool(
                        stage="spec_alignment",
                        error=last_error,
                        spec=spec,
                        raw_spec=current_raw_spec,
                        code=raw_code,
                    )
                self._persist_toolgen_failure_artifacts(
                    stage="spec_alignment",
                    error=last_error,
                    tool_name=spec.name,
                )

                if attempt == max_attempts:
                    break

                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            try:
                raw_lines = self._explode_code_lines(list(spec.code_lines or []))
                enforced_lines = self._normalize_module_docstring_lines(
                    raw_lines,
                    description=spec.description,
                )
            except Exception as exc:
                last_error = f"docstring_invariant_failed: {exc}"
                self._trace("tool_generation_error", f"stage=docstring error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="docstring_invariant",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                raw_code = self._join_code_lines(spec.code_lines or [])
                if raw_code:
                    self._persist_broken_tool(
                        stage="docstring_invariant",
                        error=last_error,
                        spec=spec,
                        raw_spec=current_raw_spec,
                        code=raw_code,
                    )
                self._persist_toolgen_failure_artifacts(
                    stage="docstring_invariant",
                    error=last_error,
                    tool_name=spec.name,
                )
                if attempt == max_attempts:
                    break
                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue
            spec.code_lines = enforced_lines
            line_count = 0
            for line in spec.code_lines or []:
                text = str(line)
                if "\r\n" in text:
                    text = text.replace("\r\n", "\n")
                if "\n" in text:
                    line_count += len(text.splitlines())
                else:
                    line_count += 1
            # if line_count > max_lines:
            #     last_error = f"line_limit_exceeded: code_lines > {max_lines}"
            #     self._trace("tool_generation_error", f"stage=line_limit error={last_error}")
            #     self._toolgen_debug_event(
            #         "toolgen_validate_failed",
            #         stage="line_limit",
            #         tool_name=spec.name,
            #         signature=spec.signature,
            #         error=last_error,
            #     )
            #     raw_code = self._join_code_lines(spec.code_lines or [])
            #     if raw_code:
            #         self._persist_broken_tool(
            #             stage="line_limit",
            #             error=last_error,
            #             spec=spec,
            #             raw_spec=current_raw_spec,
            #             code=raw_code,
            #         )
            #     if attempt == max_attempts:
            #         break
            #     repair_spec = self._repair_tool_spec(
            #         spec, last_error, chat_history, raw_spec=current_raw_spec
            #     )
            #     if repair_spec is None:
            #         break
            #     spec = repair_spec
            #     current_raw_spec = spec.__dict__
            #     continue

            placeholder_lines = [
                line
                for line in (spec.code_lines or [])
                if isinstance(line, str)
                and self._is_placeholder_line(line)
            ]
            if placeholder_lines:
                last_error = "placeholder tokens in code_lines"
                self._trace("tool_generation_error", f"stage=placeholder error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="placeholder_tokens",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                raw_code = self._join_code_lines(spec.code_lines or [])
                if raw_code:
                    self._persist_broken_tool(
                        stage="placeholder_tokens",
                        error=last_error,
                        spec=spec,
                        raw_spec=current_raw_spec,
                        code=raw_code,
                    )
                self._persist_toolgen_failure_artifacts(
                    stage="placeholder_tokens",
                    error=last_error,
                    tool_name=spec.name,
                )
                if attempt == max_attempts:
                    break
                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            code = self._join_code_lines(spec.code_lines)
            if not code:
                last_error = "empty code after joining code_lines"
                self._trace("tool_generation_error", f"stage=code_join error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="code_join",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                self._persist_toolgen_failure_artifacts(
                    stage="code_join",
                    error=last_error,
                    tool_name=spec.name,
                )

                if attempt == max_attempts:
                    break

                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            self._toolgen_debug_event(
                "toolgen_validate_start",
                tool_name=spec.name,
                signature=spec.signature,
            )

            result = None
            try:
                result = validate_tool_code(code)
            except Exception as exc:
                last_error = f"validate exception: {exc}"
                self._trace("tool_generation_error", f"stage=validate_exception error={last_error}")
                self._trace("tool_generation_traceback", traceback.format_exc())
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="validate_exception",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )

            if result and result.success:
                self._toolgen_debug_event(
                    "toolgen_validate_success",
                    tool_name=spec.name,
                    signature=spec.signature,
                    self_test_passed=result.self_test_passed,
                )

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
                    last_error = f"register exception: {exc}"
                    self._trace("tool_generation_error", f"stage=register_exception error={last_error}")
                    self._trace("tool_generation_traceback", traceback.format_exc())
                    self._toolgen_debug_event(
                        "toolgen_registry_add_failed",
                        tool_name=spec.name,
                        signature=spec.signature,
                        error=last_error,
                    )
                    metadata = None

                if metadata:
                    tool_path = (
                        self._registry._get_tool_path(metadata.name)
                        if hasattr(self._registry, "_get_tool_path")
                        else None
                    )
                    if hasattr(self._registry, "_load_metadata"):
                        try:
                            self._registry._load_metadata()
                        except Exception:
                            pass
                    self._toolgen_debug_event(
                        "toolgen_registry_add_success",
                        tool_name=metadata.name,
                        signature=metadata.signature,
                        path=tool_path,
                    )
                    self._toolgen_debug_registered_tools[metadata.name] = 0
                    self._registry.record_validation_result(
                        metadata.name, True, self_test_passed=result.self_test_passed
                    )
                    self._tool_creation_successes += 1
                    return metadata

                last_error = last_error or "registration failed (metadata is None)"
                self._trace("tool_generation_error", f"stage=register error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_registry_add_failed",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                self._toolgen_debug_event(
                    "INVARIANT_SAVE_NO_REGISTER",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                self._persist_broken_tool(
                    stage="register",
                    error=last_error,
                    spec=spec,
                    raw_spec=current_raw_spec,
                    code=code,
                )
                self._persist_toolgen_failure_artifacts(
                    stage="register",
                    error=last_error,
                    tool_name=spec.name,
                )

            else:
                last_error = getattr(result, "error", None) or last_error or "unknown validate failure"
                self._trace("tool_generation_error", f"stage=validate error={last_error}")
                self._trace(
                    "tool_generation_debug",
                    f"validate_failed name={spec.name} signature={spec.signature} tool_type={spec.tool_type}",
                )

                stage = "unknown"
                if isinstance(last_error, str):
                    if last_error.startswith("compile failed"):
                        stage = "compile"
                    elif last_error.startswith("exec failed"):
                        stage = "exec"
                    elif last_error.startswith("smoke test failed"):
                        stage = "smoke"
                    elif last_error.startswith("self_test failed"):
                        stage = "self_test"
                    elif "run() does not reference" in last_error:
                        stage = "static"

                code_preview = "\n".join(code.splitlines()[:40])
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage=stage,
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                    code_preview=code_preview,
                )
                self._persist_broken_tool(
                    stage=stage,
                    error=last_error,
                    spec=spec,
                    raw_spec=current_raw_spec,
                    code=code,
                )
                self._persist_toolgen_failure_artifacts(
                    stage=stage,
                    error=last_error,
                    tool_name=spec.name,
                )

            if attempt == max_attempts:
                break

            repair_spec = self._repair_tool_spec(spec, last_error or "", chat_history)
            if repair_spec is None:
                break
            spec = repair_spec

        return None

    def _repair_tool_spec(
        self,
        spec: ToolSpec,
        error: str,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolSpec]:
        prompt = (
            "The previous tool failed validation.\n"
            f"Error: {error}\n"
            f"Previous spec: {json.dumps(raw_spec if raw_spec is not None else spec.__dict__, ensure_ascii=True, default=str)}\n"
            "Repair rules:\n"
            "- signature must be exactly run(payload: dict) -> dict\n"
            "- input_schema.required must be exactly ['payload']\n"
            "- input_schema.properties.payload must include properties + required\n"
            "- include all required keys: name, description, signature, tool_type, tool_category, input_schema, capabilities, code_lines\n"
            "- rename any field named 'set' to 'set_values' (schema + code)\n"
            "- first 3 lines must be exactly: \"\"\", <short line>, \"\"\"\n"
            "- no other triple quotes anywhere; do NOT use '''\n"
            "- do NOT add a run() docstring; use # Example: comments instead\n"
            "- run() must return a dict; self_test() must return True\n"
            "- self_test() must include good+bad cases; bad case must assert errors contains 'missing columns'\n"
            "Return the full corrected JSON tool spec.\n"
        )
        self._toolgen_debug_event(
            "toolgen_repair_requested",
            error=error,
            prompt=prompt,
        )
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response = self._toolgen_agent._inference(tool_history)
        self._toolgen_debug_event(
            "toolgen_repair_output",
            output=response.content or "",
            output_len=len(response.content or ""),
            finish_reason=getattr(response, "finish_reason", None),
        )
        raw_text_full = self._normalize_toolgen_content(response.content)
        if response.content and not raw_text_full.strip():
            raise RuntimeError(
                "BUG: toolgen repair content empty after normalization (parsing wrong source)"
            )
        if "[truncated]" in raw_text_full:
            raise RuntimeError("BUG: truncation marker found in toolgen repair raw_text")
        self._last_toolgen_parse_source = None
        fallback_info = self._toolgen_fallback_info()
        self._toolgen_debug_event(
            "toolgen_repair_parse_input",
            content_type=type(response.content).__name__,
            raw_text_len=len(raw_text_full),
            raw_text_head=raw_text_full[:120],
            raw_text_tail=raw_text_full[-120:] if len(raw_text_full) > 120 else raw_text_full,
        )
        try:
            if not raw_text_full.strip() and fallback_info and fallback_info.get("raw_output"):
                raw_text_full = str(fallback_info["raw_output"])
            payload = self.extract_tool_spec(raw_text_full, response)
        except Exception as exc:
            self._toolgen_debug_event(
                "toolgen_repair_parse_failed",
                raw_output=raw_text_full,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                error=str(exc),
            )
            return None
        raw_spec = dict(payload)
        before_lines = list(raw_spec.get("code_lines") or [])
        try:
            raw_spec = self._normalize_tool_spec(raw_spec)
        except Exception as exc:
            self._toolgen_debug_event(
                "toolgen_repair_parse_failed",
                raw_output=raw_text_full,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                error=str(exc),
            )
            return None
        if before_lines != raw_spec.get("code_lines"):
            self._toolgen_debug_event(
                "toolgen_code_lines_normalized", tool_name=raw_spec.get("name")
            )
        return ToolSpec.from_payload(raw_spec)

    def _register_tool_from_payload(
        self, creation_request: Mapping[str, Any], chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            return None

        if not isinstance(creation_request, Mapping):
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="invalid_creation_request_type",
                request_type=type(creation_request).__name__,
            )
            self._persist_toolgen_failure_artifacts(
                stage="invalid_creation_request_type",
                error="creation_request not mapping",
                tool_name=getattr(self, "_toolgen_requested_tool_name", None),
            )
            return None

        tool_spec = dict(creation_request)
        before_lines = list(tool_spec.get("code_lines") or [])
        try:
            tool_spec = self._normalize_tool_spec(tool_spec)
            if before_lines != tool_spec.get("code_lines"):
                self._toolgen_debug_event(
                    "toolgen_code_lines_normalized", tool_name=tool_spec.get("name")
                )
        except Exception as exc:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="normalization_failed",
                error=str(exc),
            )
            self._persist_toolgen_failure_artifacts(
                stage="normalization_failed",
                error=str(exc),
                tool_name=tool_spec.get("name"),
            )
            return None
        self._toolgen_debug_event(
            "toolgen_parse_start",
            payload_keys=list(tool_spec.keys()),
        )

        requested_tool_category = getattr(self, "_toolgen_requested_tool_category", None)
        if requested_tool_category and tool_spec.get("tool_category") != requested_tool_category:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="tool_category_mismatch",
                tool_category=tool_spec.get("tool_category"),
                expected=requested_tool_category,
            )
            self._persist_toolgen_failure_artifacts(
                stage="tool_category_mismatch",
                error="tool_category_mismatch",
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "tool_category_mismatch",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        requested_name = getattr(self, "_toolgen_requested_tool_name", None)
        if requested_name:
            expected_prefix = requested_name.split("__", 1)[0]
            suffix = requested_name.split("__", 1)[1] if "__" in requested_name else ""
            if suffix in {"<generated>", "generated"}:
                digest = hashlib.sha1(
                    json.dumps(tool_spec, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:8]
                requested_name = f"{expected_prefix}__{digest}"
            if expected_prefix and not str(tool_spec.get("name") or "").startswith(expected_prefix):
                self._toolgen_debug_event(
                    "toolgen_register_failed",
                    reason="tool_name_mismatch",
                    tool_name=tool_spec.get("name"),
                    expected_prefix=expected_prefix,
                )
                self._persist_toolgen_failure_artifacts(
                    stage="tool_name_mismatch",
                    error="tool_name_mismatch",
                    tool_name=tool_spec.get("name"),
                )
                repair_spec = self._repair_tool_spec(
                    ToolSpec.from_payload(tool_spec),
                    "tool_name_mismatch",
                    chat_history,
                    raw_spec=tool_spec,
                )
                if repair_spec is None:
                    return None
                return self._validate_and_register_tool(
                    repair_spec, chat_history, raw_spec=tool_spec
                )
            if expected_prefix and requested_name != expected_prefix:
                if str(tool_spec.get("name") or "") == expected_prefix:
                    tool_spec["name"] = requested_name

        required_keys = [
            "name",
            "description",
            "signature",
            "tool_type",
            "tool_category",
            "input_schema",
            "capabilities",
            "code_lines",
        ]
        missing = [
            key for key in required_keys if tool_spec.get(key) in (None, "", [], {})
        ]
        if missing:
            self._trace(
                "tool_generation_error",
                f"stage=register error=missing required fields ({', '.join(missing)})",
            )
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="missing_required_fields",
                missing_keys=missing,
                tool_name=tool_spec.get("name"),
                signature=tool_spec.get("signature"),
            )
            self._toolgen_debug_event(
                "INVARIANT_BROKEN",
                reason="parsed_missing_required_fields",
                payload_keys=list(tool_spec.keys()),
            )
            self._persist_toolgen_failure_artifacts(
                stage="missing_required_fields",
                error=f"missing required fields: {', '.join(missing)}",
                tool_name=tool_spec.get("name"),
            )
            return None

        preflight_error = None
        if tool_spec.get("signature") != "run(payload: dict) -> dict":
            preflight_error = "signature_mismatch"
        else:
            schema_required = (
                (tool_spec.get("input_schema") or {}).get("required") or []
            )
            if schema_required != ["payload"]:
                preflight_error = "input_schema_required_mismatch"
        if preflight_error:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason=preflight_error,
                tool_name=tool_spec.get("name"),
            )
            self._persist_toolgen_failure_artifacts(
                stage=preflight_error,
                error=preflight_error,
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                preflight_error,
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        tool_category = str(tool_spec.get("tool_category") or "").lower()
        if tool_category in {"validator", "linter"}:
            code_lines = tool_spec.get("code_lines") or []
            if not any("def self_test" in (line or "") for line in code_lines):
                self._toolgen_debug_event(
                    "toolgen_register_failed",
                    reason="missing_self_test",
                    tool_name=tool_spec.get("name"),
                )
                self._persist_toolgen_failure_artifacts(
                    stage="missing_self_test",
                    error="missing self_test",
                    tool_name=tool_spec.get("name"),
                )
                repair_spec = self._repair_tool_spec(
                    ToolSpec.from_payload(tool_spec),
                    "missing self_test",
                    chat_history,
                    raw_spec=tool_spec,
                )
                if repair_spec is None:
                    return None
                return self._validate_and_register_tool(
                    repair_spec, chat_history, raw_spec=tool_spec
                )

        code_lines = tool_spec.get("code_lines") or []
        placeholder_lines = [
            line
            for line in code_lines
            if isinstance(line, str)
            and self._is_placeholder_line(line)
        ]
        if placeholder_lines:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="placeholder_tokens_in_code",
                tool_name=tool_spec.get("name"),
                offending_lines=placeholder_lines[:3],
            )
            self._persist_toolgen_failure_artifacts(
                stage="placeholder_tokens_in_code",
                error="placeholder tokens in code_lines",
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "placeholder tokens in code_lines",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )
        inline_compounds = [
            line
            for line in code_lines
            if isinstance(line, str) and "for " in line and ": if " in line
        ]
        if inline_compounds:
            self._trace(
                "tool_generation_error",
                "stage=validate error=inline_compound_statement_detected",
            )
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="code_style_inline_if",
                offending_lines=inline_compounds[:3],
            )
            self._persist_toolgen_failure_artifacts(
                stage="code_style_inline_if",
                error="inline compound statements",
                tool_name=tool_spec.get("name"),
            )
            return None

        tool_type = str(tool_spec.get("tool_type"))
        tool_category = tool_spec.get("tool_category")
        allowed_categories = {"parser", "normalizer", "planner", "validator", "linter", "formatter"}
        if tool_category not in allowed_categories:
            self._trace(
                "tool_generation_error",
                f"stage=validate error=invalid tool_category {tool_category!r}",
            )
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="invalid_tool_category",
                tool_category=tool_category,
                tool_name=tool_spec.get("name"),
            )
            self._persist_toolgen_failure_artifacts(
                stage="invalid_tool_category",
                error=f"invalid tool_category: {tool_category}",
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                f"invalid tool_category: {tool_category}",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        input_schema = tool_spec.get("input_schema")
        if not isinstance(input_schema, Mapping):
            self._trace("tool_generation_error", "stage=validate error=missing input_schema")
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="missing_input_schema",
                tool_type=tool_type,
                tool_name=tool_spec.get("name"),
            )
            self._persist_toolgen_failure_artifacts(
                stage="missing_input_schema",
                error="missing input_schema",
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "missing input_schema",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        capabilities = tool_spec.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            self._trace("tool_generation_error", "stage=validate error=missing capabilities")
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="missing_capabilities",
                tool_name=tool_spec.get("name"),
            )
            self._persist_toolgen_failure_artifacts(
                stage="missing_capabilities",
                error="missing capabilities",
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "missing capabilities",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        spec = ToolSpec.from_payload(tool_spec)
        metadata = self._validate_and_register_tool(
            spec, chat_history, raw_spec=tool_spec
        )
        if metadata:
            self._generated_tool_counter += 1
            self._mark_tool_invoked()
        return metadata

    def _consider_tool_generation(
        self, query: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if not self._is_structured_task(query) and self._resolved_environment_label() not in {"db_bench", "mysql"}:
            return None

        if not query.strip():
            return None
        if not self._force_tool_generation_if_missing:
            return None
        normalized = (
            self._normalize_retrieval_query(query)
            if hasattr(self, "_normalize_retrieval_query")
            else query.strip()
        )
        if not normalized or normalized in self._toolgen_attempted_queries:
            return None
        self._toolgen_attempted_queries.add(normalized)

        last_user = self._get_last_user_item(chat_history)
        last_obs = (last_user.content or "") if last_user else ""
        if not self._is_wrapper_parse_error_prompt(last_obs) and not self._detect_repeated_env_action(chat_history):
            return None

        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        retrieved = retrieve_tools(
            normalized,
            list(tools),
            top_k=1,
            min_reliability=self._reuse_min_reliability,
        )
        if retrieved:
            best = retrieved[0]
            if best.score >= self._reuse_similarity_threshold:
                return best.tool

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
            self._toolgen_debug_event("toolgen_skipped", reason="empty_query")
            return None

        if allow_reuse:
            reuse_query = (
                self._normalize_retrieval_query(query)
                if hasattr(self, "_normalize_retrieval_query")
                else query
            )
            candidate_output = self._get_candidate_output(chat_history, query)
            reuse = self._reuse_existing_tool(
                reuse_query, candidate_output=candidate_output, needed_archetype=None
            )
            if reuse is not None:
                self._toolgen_debug_event(
                    "toolgen_reuse_hit",
                    tool_name=reuse.name,
                    reason="reuse_existing_tool",
                )
                return reuse

        if not self._force_tool_generation_if_missing and not force:
            self._toolgen_debug_event("toolgen_skipped", reason="force_disabled")
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run and not force:
            self._toolgen_debug_event("toolgen_skipped", reason="max_generated_tools_per_run")
            return None
        if getattr(self, "_toolgen_agent", None) is None:
            self._toolgen_debug_event("toolgen_skipped", reason="missing_toolgen_agent")
            return None

        prompt = self._toolgen_request_prompt(query, chat_history)
        self._trace("tool_agent_input", prompt)
        self._toolgen_debug_event(
            "toolgen_input_built",
            prompt=prompt,
            prompt_len=len(prompt),
        )
        self._toolgen_last_prompt = prompt
        tool_history = ChatHistory()
        tool_history = self._safe_inject(tool_history, ChatHistoryItem(role=Role.USER, content=prompt))

        model_info = self._toolgen_model_info()
        self._toolgen_debug_event("toolgen_model_called", model_info=model_info)
        response = self._toolgen_agent._inference(tool_history)
        self._trace("tool_agent_result", response.content)
        fallback_info = self._toolgen_fallback_info()
        if fallback_info:
            self._toolgen_debug_event("toolgen_output_extracted", **fallback_info)
        self._toolgen_debug_event(
            "toolgen_raw_output",
            output=response.content or "",
            output_len=len(response.content or ""),
            finish_reason=getattr(response, "finish_reason", None),
        )
        raw_text_full = self._normalize_toolgen_content(response.content)
        if response.content and not raw_text_full.strip():
            raise RuntimeError(
                "BUG: toolgen content empty after normalization (parsing wrong source)"
            )
        if "[truncated]" in raw_text_full:
            raise RuntimeError("BUG: truncation marker found in toolgen raw_text")
        self._toolgen_last_raw_output = raw_text_full
        self._toolgen_last_extracted_python = ""
        self._toolgen_debug_event(
            "toolgen_parse_input",
            content_type=type(response.content).__name__,
            raw_text_len=len(raw_text_full),
            raw_text_head=raw_text_full[:120],
            raw_text_tail=raw_text_full[-120:] if len(raw_text_full) > 120 else raw_text_full,
        )
        self._last_toolgen_parse_source = None

        output_mode = self._toolgen_output_mode()
        if output_mode == "markers":
            extracted = self._extract_marked_python(raw_text_full)
            self._toolgen_last_extracted_python = extracted or ""
            if not extracted:
                if raw_text_full.lstrip().startswith("{"):
                    self._toolgen_debug_event(
                        "toolgen_markers_missing_fallback_json",
                        raw_output_len=len(raw_text_full),
                    )
                    try:
                        creation_request = self.extract_tool_spec(raw_text_full, response)
                        created = self._register_tool_from_payload(creation_request, chat_history)
                        if created is None:
                            self._persist_toolgen_failure_artifacts(
                                stage="register_returned_none",
                                error="register_returned_none",
                                tool_name=getattr(self, "_toolgen_requested_tool_name", None),
                                prompt=prompt,
                                raw_output=raw_text_full,
                                extracted_python=None,
                            )
                        return created
                    except Exception as exc:
                        note = "invalid_json"
                        if '"code_lines"' in raw_text_full:
                            note = "invalid_json_unescaped_quote_in_code_lines"
                        error = f"markers_missing_fallback_json_failed: {note}: {exc}"
                        self._toolgen_debug_event(
                            "toolgen_parse_failed",
                            raw_output=raw_text_full,
                            fallback_source=fallback_info.get("fallback_source") if fallback_info else None,
                            parsed_from="markers_missing_fallback_json",
                            parsed_source_len=len(raw_text_full),
                            error=error,
                        )
                        self._persist_toolgen_failure_artifacts(
                            stage="markers_missing_fallback_json_failed",
                            error=error,
                            tool_name=getattr(self, "_toolgen_requested_tool_name", None),
                            prompt=prompt,
                            raw_output=raw_text_full,
                            extracted_python=None,
                        )
                        return None
                error = "toolgen_markers_missing"
                self._toolgen_debug_event(
                    "toolgen_parse_failed",
                    raw_output=raw_text_full,
                    fallback_source=fallback_info.get("fallback_source") if fallback_info else None,
                    parsed_from="markers_missing",
                    parsed_source_len=len(raw_text_full),
                    error=error,
                )
                self._persist_toolgen_failure_artifacts(
                    stage="markers_missing",
                    error=error,
                    tool_name=getattr(self, "_toolgen_requested_tool_name", None),
                    prompt=prompt,
                    raw_output=raw_text_full,
                    extracted_python=None,
                )
                return None
            tool_spec = self._wrap_marker_tool_spec(extracted)
            created = self._register_tool_from_payload(tool_spec, chat_history)
            if created is None:
                self._persist_toolgen_failure_artifacts(
                    stage="register_failed",
                    error="register_returned_none",
                    tool_name=getattr(self, "_toolgen_requested_tool_name", None),
                    prompt=prompt,
                    raw_output=raw_text_full,
                    extracted_python=extracted,
                )
            return created

        try:
            if not raw_text_full.strip() and fallback_info and fallback_info.get("raw_output"):
                raw_text_full = str(fallback_info["raw_output"])
            creation_request = self.extract_tool_spec(raw_text_full, response)
        except Exception as exc:
            preview = (response.content or "")[:800]
            self._trace("tool_generation_debug", f"parse_failed content_preview={preview!r}")
            self._toolgen_debug_event(
                "toolgen_parse_failed",
                raw_output=raw_text_full,
                fallback_source=fallback_info.get("fallback_source") if fallback_info else None,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                parsed_source_len=len(raw_text_full),
                error=str(exc),
            )
            self._persist_broken_tool_output(
                stage="toolgen_parse_failed",
                error=str(exc),
                raw_output=raw_text_full,
                tool_name=getattr(self, "_toolgen_requested_tool_name", None),
            )
            self._persist_toolgen_failure_artifacts(
                stage="toolgen_parse_failed",
                error=str(exc),
                tool_name=getattr(self, "_toolgen_requested_tool_name", None),
                prompt=getattr(self, "_toolgen_last_prompt", None),
                raw_output=raw_text_full,
                extracted_python=getattr(self, "_toolgen_last_extracted_python", None),
            )
            self._toolgen_debug_event(
                "INVARIANT_PARSE_FAIL",
                raw_output=raw_text_full,
                fallback_source=fallback_info.get("fallback_source") if fallback_info else None,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                parsed_source_len=len(raw_text_full),
            )
            return None

        self._toolgen_debug_event(
            "toolgen_parse_success",
            parsed_from=getattr(self, "_last_toolgen_parse_source", None),
            parsed_source_len=len(raw_text_full),
        )
        created = self._register_tool_from_payload(creation_request, chat_history)
        if created is None:
            self._toolgen_debug_event("toolgen_register_failed", reason="register_returned_none")
            self._persist_toolgen_failure_artifacts(
                stage="register_returned_none",
                error="register_returned_none",
                tool_name=getattr(self, "_toolgen_requested_tool_name", None),
                prompt=prompt,
                raw_output=raw_text_full,
                extracted_python=getattr(self, "_toolgen_last_extracted_python", None),
            )
        return created
