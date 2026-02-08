import json
import re
from typing import Any, Mapping, Optional, Sequence

from src.typings import ChatHistory, ChatHistoryItem, Role

from .tool_registry import ToolMetadata, ToolResult
from .tool_retrieval import retrieve_tools


class ControllerToolsMixin:
    def _normalize_trace(self, trace: Any) -> list[dict[str, Any]]:
        if not isinstance(trace, list):
            return []
        if not trace:
            return []
        if all(isinstance(item, dict) for item in trace):
            return list(trace)

        def _parse_action_name(text: str) -> str:
            if not text:
                return ""
            raw = text.strip()
            if raw.lower().startswith("action:"):
                raw = raw.split(":", 1)[1].strip()
            if "(" in raw:
                return raw.split("(", 1)[0].strip()
            return raw.split()[0].strip() if raw.split() else ""

        normalized: list[dict[str, Any]] = []
        for item in trace:
            if isinstance(item, str):
                action = _parse_action_name(item)
                normalized.append({"action": action, "raw": item, "args": {}})
        return normalized

    def _extract_actions_spec_from_text(self, text: str) -> dict[str, Any]:
        if not text:
            return {}

        heading_markers = (
            "available actions",
            "actions",
            "action space",
            "you can call",
            "available tools",
        )
        end_markers = (
            "output rules",
            "final answer",
            "task completion",
            "output contract",
            "input guidelines",
        )
        blacklist = {
            "action",
            "final",
            "answer",
            "example",
            "output",
            "input",
            "format",
            "return",
            "use",
            "call",
        }

        def _extract_from_line(line: str) -> list[str]:
            clean = line.replace("`", "").replace("*", "").strip()
            found: list[str] = []
            bullet = re.match(r"^\s*(?:[-*]|\d+\.)\s*([a-zA-Z_][a-zA-Z0-9_]*)\b", clean)
            if bullet:
                found.append(bullet.group(1))
            for name in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", clean):
                found.append(name)
            return found

        actions: set[str] = set()
        capture = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                if capture:
                    capture = False
                continue
            low = line.lower()
            if any(marker in low for marker in heading_markers):
                capture = True
                continue
            if capture and any(marker in low for marker in end_markers):
                capture = False
                continue

            if capture or "action:" in low:
                for name in _extract_from_line(line):
                    if name and name.lower() not in blacklist:
                        actions.add(name.lower())

        if not actions:
            return {}
        return {name: {} for name in sorted(actions)}



    def _extract_asked_for(self, task_text: str) -> str:
        """Extract the question/asked_for from task text."""
        if not task_text:
            return ""
        # Pattern: "Question: <question>, Entities: [...]"
        match = re.match(r"Question:\s*(.+?),\s*Entities:", task_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Pattern: "Question: <question>" (no entities)
        match = re.match(r"Question:\s*(.+?)(?:\n|$)", task_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: first line or first sentence
        first_line = task_text.split("\n")[0].strip()
        if "?" in first_line:
            return first_line
        return ""

    def _extract_trace_from_history(self, chat_history: Optional[ChatHistory]) -> list[dict[str, Any]]:
        """Extract action trace from chat history."""
        if chat_history is None:
            return []
        trace = []
        for item in self._history_items(chat_history):
            content = (item.content or "").strip()
            # Match "Action: operation(args)" pattern
            action_match = re.search(r"Action:\s*(\w+)\s*\(([^)]*)\)", content)
            if action_match:
                action_name = action_match.group(1).lower()
                trace.append({
                    "action": action_name,
                    "ok": None,  # We don't know success status
                    "output": None,
                    "args": {},
                    "error": None
                })
            # Check for observation (indicates prior action succeeded)
            if trace and "Observation:" in content:
                trace[-1]["ok"] = True
                # Extract observation content
                obs_match = re.search(r"Observation:\s*(.+)", content, re.DOTALL)
                if obs_match:
                    trace[-1]["output"] = obs_match.group(1).strip()[:500]
        return trace



    def _normalize_analyzer_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace = payload.get("trace")
        normalized_trace = self._normalize_trace(trace)
        if normalized_trace:
            payload["trace"] = normalized_trace
        actions_spec = payload.get("actions_spec")
        if not actions_spec:
            extracted = self._extract_actions_spec_from_text(
                str(payload.get("task_text") or "")
            )
            if extracted:
                payload["actions_spec"] = extracted
            elif normalized_trace:
                fallback_spec: dict[str, Any] = {}
                for step in normalized_trace:
                    action = step.get("action")
                    if isinstance(action, str) and action:
                        fallback_spec[action] = {}
                if fallback_spec:
                    payload["actions_spec"] = fallback_spec
        return payload

    def _task_text_from_history(self, chat_history: ChatHistory) -> str:
        user_items = [
            item
            for item in self._history_items(chat_history)
            if item.role == Role.USER and (item.content or "").strip()
        ]
        if not user_items:
            return ""
        first_user = (user_items[0].content or "").strip()
        last_user = (user_items[-1].content or "").strip()
        if first_user and last_user and last_user not in first_user:
            return f"{first_user}\n\n{last_user}"
        return first_user or last_user
    def _bootstrap_tools(self, bootstrap_tools: Sequence[Mapping[str, Any]]) -> None:
        if not bootstrap_tools:
            return
        for index, tool in enumerate(bootstrap_tools):
            if not isinstance(tool, Mapping):
                continue
            name = str(tool.get("name") or f"bootstrap_tool_{index}")
            description = str(tool.get("description") or "")
            signature = str(tool.get("signature") or "run(task_text: str) -> str")
            code = str(tool.get("code") or "")
            tool_type = tool.get("tool_type")
            tool_category = tool.get("tool_category")
            input_schema = tool.get("input_schema")
            capabilities = tool.get("capabilities")
            metadata = self._registry.register_tool(
                name=name,
                code=code,
                signature=signature,
                description=description,
                tool_type=str(tool_type) if tool_type is not None else None,
                tool_category=str(tool_category) if tool_category is not None else None,
                input_schema=input_schema,
                capabilities=capabilities,
            )
            if metadata:
                self._generated_tool_counter += 1

    def _get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        for tool in self._registry.list_tools():
            if tool.name == name:
                return tool
        return None

    def _fail_payload_invoke(self, tool_name: str, error: str) -> ToolResult:
        self._trace("tool_generation_error", error)
        self._log_failed_invoke_event(
            tool_name=tool_name,
            reason=error,
        )
        self._log_flow_event(
            "tool_agent_output",
            tool_name=tool_name,
            success=False,
            error=error,
            output=None,
        )
        return ToolResult.failure(error)


    def _invoke_tool_by_payload(
        self,
        tool_name: str,
        tool_args: Any,
        *,
        reason: str,
        chat_history: Optional[ChatHistory] = None,
        args_auto_built: bool = False,
        decision_action: Optional[str] = None,
    ) -> ToolResult:
        resolved_name = (
            self._registry.resolve_name(tool_name)
            if hasattr(self._registry, "resolve_name")
            else None
        )
        resolved_name = resolved_name or tool_name

        # Look up signature so we can enforce "payload tools always get 1 dict arg"
        tool_meta = self._get_tool_metadata(resolved_name)

        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        # ---- Normalize tool_args into args/kwargs ----
        if isinstance(tool_args, Mapping):
            if "args" in tool_args or "kwargs" in tool_args:
                args = list(tool_args.get("args") or [])
                kwargs = dict(tool_args.get("kwargs") or {})
            else:
                args = [tool_args]
                kwargs = {}
        elif isinstance(tool_args, (list, tuple)):
            args = list(tool_args)
            kwargs = {}
        else:
            args = [tool_args] if tool_args is not None else []
            kwargs = {}

        if tool_meta and self._signature_prefers_payload(tool_meta.signature):
            payload = None
            if len(args) == 1 and isinstance(args[0], Mapping):
                if "payload" in args[0]:
                    return self._fail_payload_invoke(
                        resolved_name, "payload_wrapper_not_allowed"
                    )
                payload = dict(args[0])
            if payload is None:
                fallback_text = str(args[0]) if args else ""
                payload = self._default_payload_dict(
                    task_text=fallback_text,
                    candidate_output=None,
                    chat_history=chat_history,
                )
                args_auto_built = True
            if kwargs:
                payload.update(kwargs)
                kwargs = {}
            payload = self._normalize_analyzer_payload(payload)
            args = [payload]
            # Inject trace if missing (LLM-provided args bypass _default_payload_dict)
            if not args[0].get("trace"):
                args[0]["trace"] = self._extract_trace_from_history(chat_history)
            if len(args) != 1 or not isinstance(args[0], Mapping) or "payload" in args[0]:
                return self._fail_payload_invoke(
                    resolved_name, "payload_tool_requires_single_dict_arg"
                )

        if tool_meta and tool_meta.environment_usage:
            current_env = self._resolved_environment_label()
            if current_env not in tool_meta.environment_usage:
                return self._fail_payload_invoke(
                    resolved_name, f"tool_environment_mismatch: {current_env}"
                )

        # ---- Logging / tracing ----
        self._trace(
            "tool_agent_input",
            json.dumps(
                {
                    "tool_name": resolved_name,
                    "args": args,
                    "kwargs": kwargs,
                    "reason": reason,
                },
                ensure_ascii=True,
                default=str,
            ),
        )
        self._log_flow_event(
            "tool_agent_input",
            tool_name=resolved_name,
            args=args,
            kwargs=kwargs,
            reason=reason,
        )
        self._toolgen_debug_event(
            "tool_invoke_start",
            tool_name=resolved_name,
            args_preview=self._preview_for_log(args),
            kwargs_preview=self._preview_for_log(kwargs),
            reason=reason,
        )

        self._mark_tool_invoked()
        self._tool_invocation_attempts += 1

        result = self._registry.invoke_tool(
            resolved_name,
            *args,
            invocation_context={
                "source": "orchestrator",
                "reason": reason,
                "environment": self._resolved_environment_label(),
            },
            **kwargs,
        )

        if result.success:
            self._tool_invocation_successes += 1

        self._toolgen_debug_event(
            "tool_invoke_result",
            tool_name=resolved_name,
            success=result.success,
            error=result.error,
        )
        self._log_tool_invocation_event(
            tool_name=resolved_name,
            args=args,
            kwargs=kwargs,
            result=result,
            reason=reason,
            args_auto_built=args_auto_built,
            decision_action=decision_action,
        )
        self._log_flow_event(
            "tool_agent_output",
            tool_name=resolved_name,
            success=result.success,
            error=result.error,
            output=result.output,
        )
        self._trace("tool_agent_result", self._format_tool_result(resolved_name, result))
        return result


    def _invoke_tool_for_query(
        self,
        tool: ToolMetadata,
        query: str,
        *,
        chat_history: Optional[ChatHistory] = None,
        candidate_output: Optional[str] = None,
        reason: str = "auto_invoke",
        decision_action: Optional[str] = None,
    ) -> Optional[tuple[ToolResult, list[Any], dict[str, Any]]]:
        payload = self._build_tool_invocation(
            tool, query=query, candidate_output=candidate_output, chat_history=chat_history
        )
        if payload is None:
            return None
        args, kwargs = payload
        try:
            self._trace(
                "tool_agent_input",
                json.dumps(
                    {
                        "tool_name": tool.name,
                        "args": args,
                        "kwargs": kwargs,
                        "reason": reason,
                    },
                    ensure_ascii=True,
                    default=str,
                ),
            )
        except Exception:
            self._trace("tool_agent_input", f"{tool.name}({args}, {kwargs})")
        self._log_flow_event(
            "tool_agent_input",
            tool_name=tool.name,
            args=args,
            kwargs=kwargs,
            reason=reason,
        )
        self._toolgen_debug_event(
            "tool_invoke_start",
            tool_name=tool.name,
            args_preview=self._preview_for_log(args),
            kwargs_preview=self._preview_for_log(kwargs),
            reason=reason,
        )
        self._mark_tool_invoked()
        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            tool.name,
            *args,
            invocation_context={
                "source": "self_evolving_controller",
                "reason": reason,
                "query_preview": query[:200],
                "environment": self._resolved_environment_label(),
            },
            **kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        self._toolgen_debug_event(
            "tool_invoke_result",
            tool_name=tool.name,
            success=result.success,
            error=result.error,
        )
        self._log_tool_invocation_event(
            tool_name=tool.name,
            args=args,
            kwargs=kwargs,
            result=result,
            reason=reason,
            decision_action=decision_action,
        )
        self._log_flow_event(
            "tool_agent_output",
            tool_name=tool.name,
            success=result.success,
            error=result.error,
            output=result.output,
        )
        self._trace("tool_agent_result", self._format_tool_result(tool.name, result))
        return result, args, kwargs

    def _signature_prefers_payload(self, signature: str) -> bool:
        if not signature:
            return False
        match = re.search(r"run\s*\(([^)]*)\)", signature)
        if not match:
            return False
        params = [p.strip() for p in match.group(1).split(",") if p.strip()]
        if len(params) != 1:
            return False
        name = params[0].split(":")[0].split("=", 1)[0].strip().lower()
        return name == "payload"

    def _build_tool_invocation(
        self,
        tool: ToolMetadata,
        *,
        query: str,
        candidate_output: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> Optional[tuple[list[Any], dict[str, Any]]]:
        if not query and candidate_output is None:
            return None
        if self._signature_prefers_payload(tool.signature):
            if isinstance(candidate_output, str) and candidate_output.strip().lower().startswith("action:"):
                candidate_output = None
            payload = self._default_payload_dict(
                task_text=query or "",
                candidate_output=candidate_output,
                chat_history=chat_history,
            )
            required = []
            schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else None
            if schema:
                req = schema.get("required") or []
                if isinstance(req, list):
                    required = [str(x) for x in req if x]
            if required and any(key not in payload for key in required):
                return None
            return [payload], {}
        return [query or candidate_output or ""], {}

    def _auto_build_tool_args(
        self,
        tool: ToolMetadata,
        *,
        query: str,
        chat_history: ChatHistory,
    ) -> Optional[dict[str, Any]]:
        candidate_output = self._get_candidate_output(chat_history, query)
        
        # Get cached actions_spec
        actions_spec = getattr(self, "_cached_actions_spec", None) or {}
        if not actions_spec:
            actions_spec = self._extract_actions_spec_from_text(
                self._task_text_from_history(chat_history)
            )
        
        payload = self._build_tool_invocation(
            tool,
            query=query,
            candidate_output=candidate_output,
            chat_history=chat_history,
        )
        if payload:
            args, kwargs = payload
            # Ensure actions_spec is populated in the payload
            if args and isinstance(args[0], dict):
                if not args[0].get("actions_spec"):
                    args[0]["actions_spec"] = actions_spec
            packed: dict[str, Any] = {}
            if args:
                packed["args"] = list(args)
            if kwargs:
                packed["kwargs"] = dict(kwargs)
            if packed:
                return packed
        if query:
            return {"args": [{"task_text": query, "actions_spec": actions_spec, "candidate_output": candidate_output}]}
        return {"args": [{"task_text": query, "actions_spec": actions_spec, "candidate_output": candidate_output}]}


    def _reuse_existing_tool(
        self,
        query: str,
        *,
        candidate_output: Optional[str] = None,
        needed_archetype: Optional[str] = None,
    ) -> Optional[ToolMetadata]:
        # Filter tools by current environment
        current_env = self._resolved_environment_label()
        tools = (
            self._registry.list_latest_tools(environment=current_env)
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools(environment=current_env)
        )
        print(f"[REUSE_TOOL] Found {len(tools)} tools for environment '{current_env}'")
        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=self._reuse_top_k,
            min_reliability=self._reuse_min_reliability,
        )
        if not retrieved:
            return None
        best = retrieved[0]
        if best.score < self._reuse_similarity_threshold:
            return None
        return best.tool

    def _select_tool_for_query(
        self,
        query: str,
        *,
        categories: Optional[set[str]] = None,
        candidate_output: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> Optional[ToolMetadata]:
        # Filter tools by current environment
        current_env = self._resolved_environment_label()
        tools = (
            self._registry.list_latest_tools(environment=current_env)
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools(environment=current_env)
        )
        print(f"[SELECT_TOOL] Found {len(tools)} tools for environment '{current_env}'")
        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=self._retrieval_top_k,
            min_reliability=self._min_reliability,
        )
        if not retrieved:
            return None
        score_threshold = self._tool_match_min_score if min_score is None else min_score
        for item in retrieved:
            if item.score < score_threshold:
                continue
            tool = item.tool
            if categories and getattr(tool, "tool_category", None) not in categories:
                continue
            return tool
        return None

    def _handle_internal_tool_call(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        chat_history: ChatHistory,
    ) -> tuple[ToolResult, list[Any], dict[str, Any], str]:
        self._mark_tool_invoked()
        if "_parse_error" in payload:
            return (
                ToolResult.failure(str(payload.get("_parse_error"))),
                [],
                {},
                tool_name,
            )

        if tool_name == "create_tool":
            metadata = self._register_tool_from_payload(payload, chat_history)
            if metadata:
                return (
                    ToolResult.success_result(
                        {"created_tool": metadata.name, "description": metadata.description}
                    ),
                    [],
                    {},
                    tool_name,
                )
            return (
                ToolResult.failure("Tool creation failed validation."),
                [],
                {},
                tool_name,
            )

        resolved_name = (
            self._registry.resolve_name(tool_name)
            if hasattr(self._registry, "resolve_name")
            else None
        )

        if resolved_name is None and not self._registry.has_tool(tool_name):
            last_user = self._get_last_user_item(chat_history)
            query = last_user.content if last_user else ""
            if query and self._force_tool_generation_if_missing:
                created = self._maybe_generate_tool_for_query(query, chat_history)
                if created:
                    resolved_name = created.name

        if resolved_name is None and not self._registry.has_tool(tool_name):
            return (
                ToolResult.failure(
                    f"Tool '{tool_name}' not found. Use create_tool or answer directly."
                ),
                [],
                {},
                tool_name,
            )

        tool_args = payload.get("args", [])
        tool_kwargs = payload.get("kwargs", {})
        args_auto_built = False
        tool_meta = self._get_tool_metadata(resolved_name or tool_name)

        if not isinstance(tool_args, list):
            tool_args = []
        if not isinstance(tool_kwargs, dict):
            tool_kwargs = {}

        if not tool_args and not tool_kwargs:
            last_user = self._get_last_user_item(chat_history)
            query = (last_user.content or "").strip() if last_user else ""
            if tool_meta and self._signature_prefers_payload(tool_meta.signature):
                candidate_output = self._get_candidate_output(chat_history, query)
                if isinstance(candidate_output, str) and candidate_output.strip().lower().startswith("action:"):
                    candidate_output = None
                tool_args = [self._default_payload_dict(
                    task_text=query,
                    candidate_output=candidate_output,
                    chat_history=chat_history,  # or None if you don't want it
                )]
                # Inject trace if missing (in case payload was provided by LLM)
                if isinstance(tool_args[0], dict) and not tool_args[0].get("trace"):
                    tool_args[0]["trace"] = self._extract_trace_from_history(chat_history)

            elif query:
                tool_args = [query]
            else:
                tool_args = [""]
            args_auto_built = True

        # Final trace injection for any remaining cases (LLM-provided args)
        if tool_args and isinstance(tool_args[0], dict) and not tool_args[0].get("trace"):
            tool_args[0]["trace"] = self._extract_trace_from_history(chat_history)

        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            resolved_name or tool_name,
            *tool_args,
            invocation_context={"environment": self._resolved_environment_label()},
            **tool_kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        self._log_tool_invocation_event(
            tool_name=resolved_name or tool_name,
            args=tool_args,
            kwargs=tool_kwargs,
            result=result,
            reason="internal_tool",
            args_auto_built=args_auto_built,
            decision_action="use_tool",
        )
        return result, tool_args, tool_kwargs, (resolved_name or tool_name)
