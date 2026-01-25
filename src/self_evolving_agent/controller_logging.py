import datetime
import html
import hashlib
import json
import re
from typing import Any, Mapping, Optional

from src.typings import ChatHistory
from src.typings.config import get_predefined_timestamp_structure
from src.utils.output_paths import prefix_filename

from .tool_registry import ToolResult


class ControllerLoggingMixin:
    def _truncate(self, s: str, n: int) -> str:
        s = (s or "")
        return s if len(s) <= n else (s[:n] + "...[truncated]")

    def _trace(self, label: str, content: Any) -> None:
        text = "" if content is None else str(content)
        print(f"[TRACE] {label}:\n{text}")

    def _preview_for_log(self, obj: Any, max_len: int = 300) -> str:
        try:
            text = repr(obj)
        except Exception:
            text = f"<unreprable {type(obj).__name__}>"
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _append_generated_tools_log(self, payload: Mapping[str, Any]) -> None:
        if not self._generated_tools_log_path:
            return
        try:
            self._generated_tools_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._generated_tools_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception:
            return

    def _toolgen_debug_event(self, event: str, **fields: Any) -> None:
        logger = self._toolgen_debug_logger
        if not logger or not logger.enabled:
            return
        meta = self._get_run_task_metadata()
        logger.log_event(
            event,
            task_name=meta.get("task_name"),
            sample_index=meta.get("sample_index"),
            environment=self._resolved_environment_label(),
            **fields,
        )

    def _append_tool_invocation_log(self, text: str) -> None:
        if not self._tool_invocation_log_path:
            return
        try:
            self._tool_invocation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._tool_invocation_log_path.open("a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            return

    def _serialize_chat_history(self, chat_history: Optional[ChatHistory]) -> list[dict[str, Any]]:
        if chat_history is None:
            return []
        items = self._history_items(chat_history)
        return [{"role": item.role.value, "content": item.content} for item in items]

    def _load_current_session_payload(self) -> Optional[dict[str, Any]]:
        if not self._generated_tools_log_path:
            return None
        session_path = self._generated_tools_log_path.parent / prefix_filename(
            "current_session.json"
        )
        if not session_path.exists():
            return None
        try:
            raw = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return raw if isinstance(raw, dict) else None

    def _build_flow_log_base(self, *, chat_history: Optional[ChatHistory]) -> dict[str, Any]:
        base: dict[str, Any] = {}
        session = self._load_current_session_payload()
        if session:
            for key in ("task_name", "sample_index", "sample_status"):
                if key in session:
                    base[key] = session.get(key)
        else:
            base.update(self._get_run_task_metadata())
        if chat_history is not None:
            base["chat_history"] = {"value": self._serialize_chat_history(chat_history)}
        elif session and isinstance(session.get("chat_history"), dict):
            base["chat_history"] = session.get("chat_history")
        else:
            base["chat_history"] = {"value": []}
        base["events"] = []
        return base

    def _append_flow_log(self, payload: Mapping[str, Any], *, scope: str, chat_history: Optional[ChatHistory]) -> None:
        path = None
        if scope == "session":
            path = getattr(self, "_flow_session_log_path", None)
        elif scope == "full":
            path = getattr(self, "_flow_full_log_path", None)
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    existing = None
            else:
                existing = None
            if not isinstance(existing, dict):
                existing = self._build_flow_log_base(chat_history=chat_history)
            events = existing.get("events")
            if not isinstance(events, list):
                events = []
            events.append(dict(payload))
            existing["events"] = events
            if chat_history is not None:
                existing["chat_history"] = {"value": self._serialize_chat_history(chat_history)}
            path.write_text(json.dumps(existing, ensure_ascii=True, default=str), encoding="utf-8")
            self._append_flow_view(path, dict(payload))
        except Exception:
            return

    def _append_flow_view(self, json_path, payload: Mapping[str, Any]) -> None:
        try:
            view_path = json_path.with_suffix(".view.md")
            index = self._next_view_index(view_path)
            prev_event = self._load_previous_flow_event(json_path)
            lines = self._format_flow_view_event(payload, prev_event, index, view_path)
            if not lines:
                return
            with view_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n\n---\n")
        except Exception:
            return

    def _next_view_index(self, view_path) -> int:
        if not view_path.exists():
            return 1
        try:
            text = view_path.read_text(encoding="utf-8")
        except Exception:
            return 1
        matches = re.findall(r"^\\d+\\.", text, flags=re.MULTILINE)
        return len(matches) + 1

    def _load_previous_flow_event(self, json_path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        events = data.get("events") if isinstance(data, dict) else None
        if isinstance(events, list) and len(events) >= 2 and isinstance(events[-2], dict):
            return events[-2]
        return None

    def _write_view_header_if_missing(self, view_path, payload: Mapping[str, Any]) -> list[str]:
        if view_path.exists() and view_path.stat().st_size > 0:
            return []
        task_name = payload.get("task_name")
        sample_index = payload.get("sample_index")
        if task_name is None and sample_index is None:
            return []
        task = f"Task: {task_name}" if task_name is not None else "Task: unknown"
        sample = f"Sample: {sample_index}" if sample_index is not None else "Sample: unknown"
        return [f"{task} | {sample}", ""]

    def _format_flow_view_event(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
        index: int,
        view_path,
    ) -> list[str]:
        event = payload.get("event") or "event"
        lines: list[str] = []
        lines.extend(self._write_view_header_if_missing(view_path, payload))
        lines.append(f"{index}. {event}")
        formatter = {
            "orchestrator_input": self._view_orchestrator_input,
            "orchestrator_output": self._view_orchestrator_output,
            "tool_agent_input": self._view_tool_agent_input,
            "tool_agent_output": self._view_tool_agent_output,
            "solver_input": self._view_solver_input,
            "final_response": self._view_final_response,
        }.get(event)
        if formatter:
            lines.extend(formatter(payload, prev_event))
        else:
            lines.extend(self._view_generic_event(payload, prev_event))
        return lines

    def _view_orchestrator_input(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        prompt = payload.get("prompt") or ""
        parsed = self._parse_json_maybe(prompt)
        prev_parsed = self._parse_json_maybe(prev_event.get("prompt") if prev_event else "")
        task_text = parsed.get("task_text") if isinstance(parsed, dict) else None
        history = parsed.get("history") if isinstance(parsed, dict) else None
        existing = parsed.get("existing_tools") if isinstance(parsed, dict) else None
        lines: list[str] = []
        lines.extend(self._view_text_field("task", task_text, prev_parsed.get("task_text") if isinstance(prev_parsed, dict) else None, same_as_above=True))
        lines.extend(self._view_text_field("history", history, prev_parsed.get("history") if isinstance(prev_parsed, dict) else None, collapsed=True))
        lines.extend(self._view_tools_field(existing, prev_parsed.get("existing_tools") if isinstance(prev_parsed, dict) else None))
        return lines

    def _view_orchestrator_output(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        output = payload.get("output") or ""
        parsed = self._parse_json_maybe(output)
        prev_parsed = self._parse_json_maybe(prev_event.get("output") if prev_event else "")
        lines: list[str] = []
        if isinstance(parsed, dict):
            lines.extend(self._view_simple_field("action", parsed.get("action"), prev_parsed.get("action") if isinstance(prev_parsed, dict) else None))
            lines.extend(self._view_simple_field("tool", parsed.get("tool_name"), prev_parsed.get("tool_name") if isinstance(prev_parsed, dict) else None))
            reason = parsed.get("reason")
            lines.extend(self._view_text_field("reason", reason, prev_parsed.get("reason") if isinstance(prev_parsed, dict) else None))
            lines.extend(self._view_json_details("output_json", parsed))
        else:
            lines.extend(self._view_text_field("output", output, prev_event.get("output") if prev_event else None))
        return lines

    def _view_tool_agent_input(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        tool_name = payload.get("tool_name")
        prev_tool = prev_event.get("tool_name") if prev_event else None
        args = payload.get("tool_args")
        if args is None:
            args = {"args": payload.get("args"), "kwargs": payload.get("kwargs")}
        prev_args = prev_event.get("tool_args") if prev_event else None
        sql = None
        if isinstance(args, Mapping):
            if isinstance(args.get("sql"), str):
                sql = args.get("sql")
            elif isinstance(args.get("query"), str):
                sql = args.get("query")
            elif isinstance(args.get("args"), list) and args["args"]:
                first = args["args"][0]
                if isinstance(first, Mapping):
                    sql = first.get("sql") or first.get("query")
        lines: list[str] = []
        lines.extend(self._view_simple_field("tool", tool_name, prev_tool))
        lines.extend(self._view_text_field("sql", sql, None))
        lines.extend(self._view_json_details("args", args, prev_args))
        return lines

    def _view_tool_agent_output(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        success = payload.get("success")
        error = payload.get("error")
        output = payload.get("output")
        lines: list[str] = []
        lines.extend(self._view_simple_field("success", success, prev_event.get("success") if prev_event else None))
        summary = self._summarize_tool_output(error, output)
        lines.extend(self._view_text_field("result", summary, None))
        lines.extend(self._view_json_details("output", output, prev_event.get("output") if prev_event else None))
        return lines

    def _view_solver_input(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        solver_payload = payload.get("payload") or {}
        history = solver_payload.get("history") if isinstance(solver_payload, Mapping) else ""
        extracted = self._extract_solver_context(history)
        prev_extracted = self._extract_solver_context((prev_event.get("payload") or {}).get("history") if isinstance(prev_event, Mapping) else "")
        lines: list[str] = []
        lines.extend(self._view_text_field("LAST_OBSERVATION", extracted.get("LAST_OBSERVATION"), prev_extracted.get("LAST_OBSERVATION")))
        lines.extend(self._view_text_field("TASK_INTENT", extracted.get("TASK_INTENT"), prev_extracted.get("TASK_INTENT")))
        lines.extend(self._view_text_field("ORIGINAL_TASK", extracted.get("ORIGINAL_TASK"), prev_extracted.get("ORIGINAL_TASK")))
        return lines

    def _view_final_response(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        content = payload.get("content") or ""
        return self._view_text_field("response", content, prev_event.get("content") if prev_event else None)

    def _view_generic_event(
        self,
        payload: Mapping[str, Any],
        prev_event: Optional[Mapping[str, Any]],
    ) -> list[str]:
        lines: list[str] = []
        for key, value in payload.items():
            if key in {"event", "task_name", "sample_index", "timestamp"}:
                continue
            prev_val = prev_event.get(key) if prev_event else None
            lines.extend(self._view_text_field(key, value, prev_val))
        return lines

    def _view_simple_field(self, label: str, value: Any, prev_value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if prev_value is not None and value == prev_value:
            return []
        return [f"- {label}: {value}"]

    def _view_text_field(
        self,
        label: str,
        value: Any,
        prev_value: Any,
        *,
        collapsed: bool = False,
        same_as_above: bool = False,
    ) -> list[str]:
        if value is None or value == "":
            return []
        if prev_value is not None and value == prev_value:
            return [f"- {label}: same as above"] if same_as_above or collapsed else []
        if self._needs_details(value):
            preview = self._preview_words(value, 50)
            full = self._format_detail_content(value)
            return [
                f"- {label}: {preview}",
                f"  <details><summary>{preview}</summary><pre>{full}</pre></details>",
            ]
        return [f"- {label}: {value}"]

    def _view_tools_field(self, tools: Any, prev_tools: Any) -> list[str]:
        if tools is None:
            return []
        if prev_tools is not None and tools == prev_tools:
            return ["- tools: same as above"]
        if not isinstance(tools, list):
            return self._view_text_field("tools", tools, prev_tools)
        names = []
        for tool in tools:
            if isinstance(tool, Mapping) and tool.get("name"):
                names.append(str(tool.get("name")))
        preview = f"{len(tools)}" if tools else "0"
        detail = json.dumps(tools, ensure_ascii=True, default=str, indent=2)
        return [
            f"- tools: {preview}",
            f"  <details><summary>{preview}</summary><pre>{html.escape(detail)}</pre></details>",
        ]

    def _view_json_details(self, label: str, value: Any, prev_value: Any = None) -> list[str]:
        if value is None:
            return []
        if prev_value is not None and value == prev_value:
            return []
        if isinstance(value, (dict, list)):
            keys = []
            if isinstance(value, dict):
                keys = list(value.keys())[:3]
            summary = f"{{…}} (keys: {', '.join(keys)})" if keys else "{…}"
            pretty = json.dumps(value, ensure_ascii=True, default=str, indent=2)
        else:
            summary = self._preview_words(value, 50)
            pretty = self._pretty_json_if_possible(value) or str(value)
        return [
            f"- {label}: {summary}",
            f"  <details><summary>{summary}</summary><pre>{html.escape(pretty)}</pre></details>",
        ]

    def _summarize_tool_output(self, error: Any, output: Any) -> str:
        if error:
            return self._preview_words(str(error), 20)
        if isinstance(output, Mapping):
            errors = output.get("errors")
            warnings = output.get("warnings")
            if isinstance(errors, list) and errors:
                return self._preview_words(str(errors[0]), 20)
            if isinstance(warnings, list) and warnings:
                return self._preview_words(str(warnings[0]), 20)
        if output is None:
            return ""
        return self._preview_words(str(output), 20)

    def _extract_solver_context(self, history: Any) -> dict[str, str]:
        if not isinstance(history, str):
            return {}
        lines = history.splitlines()
        extracted: dict[str, str] = {}
        for line in reversed(lines):
            for key in ("LAST_OBSERVATION:", "TASK_INTENT:", "ORIGINAL_TASK:"):
                if line.strip().startswith(key):
                    extracted[key[:-1]] = line.split(":", 1)[1].strip()
            if len(extracted) == 3:
                break
        return extracted

    def _parse_json_maybe(self, text: Any) -> Optional[dict[str, Any]]:
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            obj = json.loads(text)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _format_detail_content(self, value: Any) -> str:
        pretty = self._pretty_json_if_possible(value)
        if pretty is None:
            pretty = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, default=str, indent=2)
        return html.escape(pretty)

    def _format_view_field(self, key: str, value: Any) -> list[str]:
        if self._needs_details(value):
            preview, full = self._format_view_details(value)
            return [
                f"{key}: {preview}",
                f"<details><summary>{preview}</summary><pre>{full}</pre></details>",
            ]
        return [f"{key}: {self._format_view_scalar(value)}"]

    def _needs_details(self, value: Any) -> bool:
        if isinstance(value, (dict, list)):
            return True
        if not isinstance(value, str):
            return False
        words = value.split()
        if len(words) > 50:
            return True
        stripped = value.lstrip()
        return stripped.startswith("{") or stripped.startswith("[")

    def _format_view_details(self, value: Any) -> tuple[str, str]:
        preview = self._preview_words(value, 50)
        pretty = self._pretty_json_if_possible(value)
        if pretty is None:
            pretty = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, default=str, indent=2)
        return preview, html.escape(pretty)

    def _preview_words(self, value: Any, max_words: int) -> str:
        if isinstance(value, (dict, list)):
            raw = json.dumps(value, ensure_ascii=True, default=str)
        else:
            raw = str(value)
        words = raw.split()
        truncated = len(words) > max_words
        preview = " ".join(words[:max_words])
        return (preview + " …") if truncated else preview

    def _pretty_json_if_possible(self, value: Any) -> Optional[str]:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True, default=str, indent=2)
        if not isinstance(value, str):
            return None
        stripped = value.lstrip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return None
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=True, default=str, indent=2)
        return None

    def _format_view_scalar(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return self._preview_words(value, 50)
        return self._preview_words(str(value), 50)

    def _log_flow_event(
        self,
        event: str,
        *,
        chat_history: Optional[ChatHistory] = None,
        scope: str = "both",
        **fields: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
        }
        payload.update(self._get_run_task_metadata())
        payload.update(fields)
        if scope in {"both", "session"}:
            self._append_flow_log(payload, scope="session", chat_history=chat_history)
        if scope in {"both", "full"}:
            self._append_flow_log(payload, scope="full", chat_history=chat_history)

    def _get_run_task_label(self) -> str:
        if self._run_task_label is not None:
            return self._run_task_label
        if not self._tool_invocation_log_path:
            self._run_task_label = "Task: unknown"
            return self._run_task_label
        session_path = self._tool_invocation_log_path.parent / prefix_filename(
            "current_session.json"
        )
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            task_name = payload.get("task_name")
            sample_index = payload.get("sample_index")
            parts = []
            if task_name is not None:
                parts.append(f"Task: {task_name}")
            if sample_index is not None:
                parts.append(f"Sample: {sample_index}")
            self._run_task_label = " | ".join(parts) if parts else "Task: unknown"
        except Exception:
            self._run_task_label = "Task: unknown"
        return self._run_task_label

    def _solver_context_key(self, chat_history: ChatHistory) -> str:
        last_user = self._get_last_user_item(chat_history)
        content = last_user.content if last_user else ""
        if not content:
            return ""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_run_task_metadata(self) -> dict[str, Any]:
        if self._run_task_metadata is not None:
            return dict(self._run_task_metadata)
        if not self._generated_tools_log_path:
            return {}
        session_path = self._generated_tools_log_path.parent / prefix_filename(
            "current_session.json"
        )
        if not session_path.exists():
            return {}
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        task_name = payload.get("task_name")
        sample_index = payload.get("sample_index")
        meta: dict[str, Any] = {}
        if task_name is not None:
            meta["task_name"] = task_name
        if sample_index is not None:
            meta["sample_index"] = sample_index
        if meta:
            self._run_task_metadata = meta
        return dict(meta)

    def _resolved_environment_label(self) -> str:
        meta = self._get_run_task_metadata()
        task_name = meta.get("task_name")
        if task_name in {"db_bench", "mysql"}:
            return "db_bench"
        return self._environment_label

    def _is_db_bench_env(self) -> bool:
        return self._resolved_environment_label() in {"db_bench", "mysql"}

    def _is_structured_task(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        score = 0
        if any(token in q for token in ("format", "exact", "exactly", "must", "return", "provide")):
            score += 1
        if any(token in q for token in ("json", "csv", "table", "columns", "schema")):
            score += 1
        if any(token in q for token in ("group by", "having", "order by", "limit", "offset")):
            score += 1
        if any(token in q for token in ("sum", "average", "avg", "count", "total", "aggregate")):
            score += 1
        if re.search(r"(less than|greater than|at least|at most|>=|<=|!=|=)", q):
            score += 1
        if q.count(" and ") + q.count(" or ") >= 1:
            score += 1
        return score >= 2

    def _strip_supplemental_sections(self, text: str) -> str:
        if not text:
            return ""
        base = text
        for marker in ("\n\nTOOL RESULTS:", "\n\nCONTEXT:"):
            if marker in base:
                base = base.split(marker, 1)[0]
        return base.strip()

    def _log_tool_invocation_event(
        self,
        *,
        tool_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        result: ToolResult,
        reason: str,
        args_auto_built: bool = False,
        decision_action: Optional[str] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": "invoke",
            "tool_name": tool_name,
            "args_preview": self._preview_for_log(args),
            "kwargs_preview": self._preview_for_log(kwargs),
            "success": result.success,
            "error": result.error,
            "reason": reason,
            "environment_label": self._resolved_environment_label(),
            "source": "controller",
            "args_auto_built": args_auto_built,
        }
        if decision_action:
            payload["decision_action"] = decision_action
        payload.update(self._get_run_task_metadata())
        self._append_generated_tools_log(payload)

    def _mark_tool_invoked(self) -> None:
        self._tool_invoked_in_last_inference = True

    def _format_tool_response_text(self, result: ToolResult) -> str:
        lines = [f"Success: {result.success}"]
        if result.output is not None:
            lines.append(f"Output: {self._registry._preview(result.output)}")
        if result.error:
            lines.append(f"Error: {result.error}")
        return "\n".join(lines)

    def _write_agent_system_prompt(self, agent_name: str, prompt: str) -> None:
        path = getattr(self, "_agent_system_prompt_path", None)
        if not path or not agent_name:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}
            data[agent_name] = prompt or ""
            data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            path.write_text(json.dumps(data, ensure_ascii=True, default=str), encoding="utf-8")
        except Exception:
            return

    def _build_tool_trace(
        self,
        *,
        summary: str,
        tool_name: str,
        args: list[Any],
        kwargs: Mapping[str, Any],
        result: ToolResult,
    ) -> dict[str, Any]:
        return {
            "summary": summary,
            "tool_name": tool_name,
            "args": list(args),
            "kwargs": dict(kwargs),
            "result": result,
            "model_use": None,
        }

    def _flush_tool_traces(
        self, traces: list[dict[str, Any]], final_response: str
    ) -> None:
        if not traces:
            return
        task_label = self._get_run_task_label()
        blocks: list[str] = []
        for trace in traces:
            model_use = trace.get("model_use") or "(no model response captured yet)"
            result = trace.get("result")
            if not isinstance(result, ToolResult):
                continue
            blocks.append(
                "\n".join(
                    [
                        "=== Tool Invocation ===",
                        task_label,
                        f"Summary: {trace.get('summary')}",
                        f"Tool: {trace.get('tool_name')}",
                        "Content sent to tool:",
                        f"  args: {trace.get('args')}",
                        f"  kwargs: {trace.get('kwargs')}",
                        "Tool response:",
                        self._format_tool_response_text(result),
                        "How the model uses the tool response:",
                        str(model_use),
                        "Model final response after considering tool content:",
                        str(final_response),
                    ]
                )
            )
        if blocks:
            self._append_tool_invocation_log("\n\n".join(blocks) + "\n")
