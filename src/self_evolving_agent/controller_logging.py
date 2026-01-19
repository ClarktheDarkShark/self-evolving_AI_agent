import datetime
import hashlib
import json
import os
import re
from pathlib import Path
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

    def _toolgen_model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        toolgen_agent = getattr(self, "_toolgen_agent", None)
        if toolgen_agent is None:
            return info
        model = getattr(toolgen_agent, "_language_model", None)
        if model is None:
            return info
        info["model_type"] = type(model).__name__
        for attr in ("model", "model_name", "name", "base_url"):
            if hasattr(model, attr):
                try:
                    info[attr] = getattr(model, attr)
                except Exception:
                    info[attr] = None
        return info

    def _toolgen_fallback_info(self) -> Optional[dict[str, Any]]:
        toolgen_agent = getattr(self, "_toolgen_agent", None)
        if toolgen_agent is None:
            return None
        model = getattr(toolgen_agent, "_language_model", None)
        if model is None:
            return None
        getter = getattr(model, "get_last_tool_call_fallback", None)
        if callable(getter):
            try:
                return getter(clear=True)
            except Exception:
                return getter()
        return None

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

    def _is_trivial_task(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return True
        if self._is_structured_task(q):
            return False
        words = q.split()
        if len(words) <= 8:
            return True
        if len(q) <= 60 and not re.search(r"\d", q):
            return True
        return False

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
