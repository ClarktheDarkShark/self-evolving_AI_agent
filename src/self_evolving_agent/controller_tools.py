import json
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.typings import ChatHistory, ChatHistoryItem

from .tool_registry import ToolMetadata, ToolResult
from .tool_retrieval import retrieve_tools


class ControllerToolsMixin:
    def _build_payload_from_payload_schema(
        self,
        schema: Mapping[str, Any],
        key_map: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        props = schema.get("properties") or {}
        payload_schema = props.get("payload")
        if not isinstance(payload_schema, Mapping):
            return None

        inner_props = payload_schema.get("properties") or {}
        inner_required = payload_schema.get("required") or []

        payload: dict[str, Any] = {}
        for k in inner_props.keys():
            if k in key_map and key_map[k] is not None:
                payload[k] = key_map[k]

        for k in inner_required:
            if k not in payload:
                return None

        return payload

    def _get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        for tool in self._registry.list_tools():
            if tool.name == name:
                return tool
        return None

    def _invoke_tool_by_payload(
        self,
        tool_name: str,
        tool_args: Any,
        *,
        reason: str,
        args_auto_built: bool = False,
        decision_action: Optional[str] = None,
    ) -> ToolResult:
        resolved_name = (
            self._registry.resolve_name(tool_name)
            if hasattr(self._registry, "resolve_name")
            else None
        )
        resolved_name = resolved_name or tool_name
        tool_meta = self._get_tool_metadata(resolved_name)
        if isinstance(tool_args, Mapping) and "args" not in tool_args and "kwargs" not in tool_args:
            if tool_meta and self._signature_prefers_payload(tool_meta.signature):
                if "payload" in tool_args:
                    tool_args = {"args": [tool_args.get("payload")]}
                else:
                    tool_args = {"args": [dict(tool_args)]}
            elif tool_meta and isinstance(tool_meta.input_schema, Mapping):
                properties = tool_meta.input_schema.get("properties") or {}
                required = tool_meta.input_schema.get("required") or []
                if "payload" in properties and "payload" in required and "payload" not in tool_args:
                    tool_args = {"kwargs": {"payload": dict(tool_args)}}
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        if isinstance(tool_args, Mapping):
            if "args" in tool_args or "kwargs" in tool_args:
                args = list(tool_args.get("args") or [])
                kwargs = dict(tool_args.get("kwargs") or {})
            else:
                kwargs = dict(tool_args)
        elif isinstance(tool_args, (list, tuple)):
            args = list(tool_args)
        else:
            args = [tool_args] if tool_args is not None else []
        if tool_meta and self._signature_prefers_payload(tool_meta.signature):
            if kwargs:
                if "payload" in kwargs and len(kwargs) == 1:
                    args = [kwargs.get("payload")]
                    kwargs = {}
                elif not args:
                    args = [dict(kwargs)]
                    kwargs = {}
            if len(args) == 1 and isinstance(args[0], dict) and "payload" in args[0] and not kwargs:
                args = [args[0].get("payload")]
            if not (len(args) == 1 and isinstance(args[0], dict)):
                if args:
                    args = [{"task_text": str(args[0])}]
                else:
                    args = [{}]
                kwargs = {}
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
        self._trace("tool_agent_result", self._format_tool_result(resolved_name, result))
        return result

    def _invoke_tool_for_query(
        self,
        tool: ToolMetadata,
        query: str,
        *,
        candidate_output: Optional[str] = None,
        reason: str = "auto_invoke",
        decision_action: Optional[str] = None,
    ) -> Optional[tuple[ToolResult, list[Any], dict[str, Any]]]:
        payload = self._build_tool_invocation(
            tool, query=query, candidate_output=candidate_output
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
    ) -> Optional[tuple[list[Any], dict[str, Any]]]:
        if not query and candidate_output is None:
            return None

        schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else None
        values: dict[str, Any] = {}

        key_map = {
            "task_text": query,
            "query": query,
            "text": query,
            "instruction": query,
            "candidate_output": candidate_output,
            "candidate_artifact": candidate_output,
            "output": candidate_output,
            "response": candidate_output,
            "history": None,
            "environment": self._resolved_environment_label(),
        }

        if schema is not None:
            if "payload" in (schema.get("properties") or {}) and "payload" in (schema.get("required") or []):
                payload = self._build_payload_from_payload_schema(schema, key_map)
                if payload is None:
                    return None
                if self._signature_prefers_payload(tool.signature):
                    return [payload], {}
                return [], {"payload": payload}

            properties = schema.get("properties") or {}
            required = schema.get("required") or []
            for key in properties.keys():
                if key in key_map and key_map[key] is not None:
                    values[key] = key_map[key]
            for key in required:
                if key not in values:
                    return None
            if required:
                return [values[k] for k in required], {}
            if values:
                return [values], {}

        if query:
            return [query], {}
        if candidate_output is not None:
            return [candidate_output], {}
        return None

    @staticmethod
    def _pack_tool_args(
        args: Sequence[Any], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        packed: dict[str, Any] = {}
        if args:
            packed["args"] = list(args)
        if kwargs:
            packed["kwargs"] = dict(kwargs)
        return packed

    def _auto_build_tool_args(
        self,
        tool: ToolMetadata,
        *,
        query: str,
        chat_history: ChatHistory,
    ) -> Optional[dict[str, Any]]:
        last_user = self._get_last_user_item(chat_history)
        candidate_output = self._get_candidate_output(chat_history, query)
        history_text = self._toolgen_render_history(chat_history, max_chars_per_item=300)

        payload: Optional[tuple[list[Any], dict[str, Any]]] = None
        try:
            payload = self._build_tool_invocation(tool, query=query, candidate_output=candidate_output)
        except Exception:
            payload = None

        if payload:
            args, kwargs = payload
            packed = self._pack_tool_args(args, kwargs)
            if packed:
                return packed

        schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else None

        key_map: dict[str, Any] = {
            "task_text": query,
            "query": query,
            "text": query,
            "instruction": query,
            "candidate_output": candidate_output,
            "candidate_artifact": candidate_output,
            "output": candidate_output,
            "response": candidate_output,
            "history": history_text,
            "history_compact": history_text,
            "environment": self._resolved_environment_label(),
        }

        if schema is not None:
            properties = schema.get("properties") or {}
            required = schema.get("required") or []

            if "payload" in properties and "payload" in required:
                payload_schema = properties.get("payload")
                if isinstance(payload_schema, Mapping):
                    inner_props = payload_schema.get("properties") or {}
                    inner_required = payload_schema.get("required") or []

                    built_payload: dict[str, Any] = {}
                    for k in inner_props.keys():
                        if k in key_map and key_map[k] is not None:
                            built_payload[k] = key_map[k]

                    if inner_required and not all(k in built_payload for k in inner_required):
                        return None

                    if self._signature_prefers_payload(tool.signature):
                        return {"args": [built_payload]}

                    return {"kwargs": {"payload": built_payload}}

                return None

            values: dict[str, Any] = {}
            for k in properties.keys():
                if k in key_map and key_map[k] is not None:
                    values[k] = key_map[k]

            if required and all(k in values for k in required):
                if self._signature_prefers_payload(tool.signature):
                    return {"args": [values]}
                return {"args": [values[k] for k in required]}

            if values:
                return {"args": [values]}

        fallback_payload = {
            "task_text": query,
            "candidate_output": candidate_output,
            "history": history_text,
            "environment": self._resolved_environment_label(),
        }

        if self._signature_prefers_payload(tool.signature) or isinstance(tool.input_schema, Mapping):
            return {"args": [fallback_payload]}

        if query:
            return {"args": [query]}

        return {"args": [fallback_payload]}

    def _expected_tool_type_for_archetype(self, archetype: str) -> Optional[str]:
        return {
            "answer_guard": "validator",
            "sql_static_check": "linter",
            "final_output_formatter": "formatter",
            "general_helper": "utility",
        }.get(archetype)

    def _should_use_tool(
        self,
        tool: ToolMetadata,
        *,
        candidate_output: Optional[str],
        query: Optional[str] = None,
    ) -> bool:
        category = (tool.tool_category or "").lower()
        if category in {"validator", "linter", "formatter"} and not candidate_output:
            return False
        if query and self._is_structured_task(query):
            if category in {"utility"} and not candidate_output:
                return False
        return True

    def _reuse_existing_tool(
        self,
        query: str,
        *,
        candidate_output: Optional[str] = None,
        needed_archetype: Optional[str] = None,
    ) -> Optional[ToolMetadata]:
        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=self._reuse_top_k,
            min_reliability=self._reuse_min_reliability,
        )
        if not retrieved:
            return None
        best = retrieved[0]
        if needed_archetype:
            expected_type = self._expected_tool_type_for_archetype(needed_archetype)
            if expected_type and (best.tool.tool_type or "") != expected_type:
                return None

        if not self._should_use_tool(
            best.tool, candidate_output=candidate_output, query=query
        ):
            return None
        if getattr(best.tool, "self_test_passed", True) is False:
            return None
        if getattr(best.tool, "reliability_score", 0.0) < 0.4:
            return None

        if best.score < self._reuse_similarity_threshold:
            return None
        if "run(" not in (best.tool.signature or ""):
            return None
        return best.tool

    @staticmethod
    def _capability_overlap(target: Iterable[str], candidate: Iterable[str]) -> float:
        target_set = {str(x) for x in (target or []) if x}
        candidate_set = {str(x) for x in (candidate or []) if x}
        if not target_set or not candidate_set:
            return 0.0
        return len(target_set & candidate_set) / max(len(target_set), len(candidate_set))

    def _select_tool_for_query(
        self,
        query: str,
        *,
        categories: Optional[set[str]] = None,
        candidate_output: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> Optional[ToolMetadata]:
        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=self._retrieval_top_k,
            min_reliability=self._min_reliability,
        )
        if retrieved:
            self._toolgen_debug_event(
                "tool_retrieval",
                query_preview=(query or "")[:300],
                results=[
                    {"name": item.tool.name, "score": item.score}
                    for item in retrieved
                ],
            )
            if self._toolgen_debug_registered_tools:
                retrieved_names = {item.tool.name for item in retrieved}
                missing = [
                    name
                    for name in self._toolgen_debug_registered_tools.keys()
                    if name not in retrieved_names
                ]
                if missing:
                    self._toolgen_debug_event(
                        "INVARIANT_REGISTER_NO_REUSE",
                        query_preview=(query or "")[:300],
                        missing_registered_tools=missing,
                    )
        if not retrieved:
            if self._toolgen_debug_registered_tools:
                self._toolgen_debug_event(
                    "INVARIANT_REGISTER_NO_REUSE",
                    query_preview=(query or "")[:300],
                    missing_registered_tools=list(self._toolgen_debug_registered_tools.keys()),
                )
            return None
        score_threshold = self._tool_match_min_score if min_score is None else min_score
        for item in retrieved:
            if item.score < score_threshold:
                continue
            tool = item.tool
            if categories and getattr(tool, "tool_category", None) not in categories:
                continue
            if not self._should_use_tool(
                tool, candidate_output=candidate_output, query=query
            ):
                continue
            self._toolgen_debug_event(
                "tool_selected",
                tool_name=tool.name,
                score=item.score,
                query_preview=(query or "")[:300],
            )
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

        if len(tool_args) == 1 and isinstance(tool_args[0], dict) and not tool_kwargs:
            if not (tool_meta and self._signature_prefers_payload(tool_meta.signature)):
                tool_kwargs = dict(tool_args[0])
                tool_args = []

        if not isinstance(tool_args, list):
            tool_args = []
        if not isinstance(tool_kwargs, dict):
            tool_kwargs = {}
        if tool_meta and self._signature_prefers_payload(tool_meta.signature):
            if tool_kwargs:
                if "payload" in tool_kwargs and len(tool_kwargs) == 1 and not tool_args:
                    tool_args = [tool_kwargs.get("payload")]
                    tool_kwargs = {}
                elif not tool_args:
                    tool_args = [dict(tool_kwargs)]
                    tool_kwargs = {}
            if len(tool_args) == 1 and isinstance(tool_args[0], dict) and "payload" in tool_args[0] and not tool_kwargs:
                tool_args = [tool_args[0].get("payload")]
            if not (len(tool_args) == 1 and isinstance(tool_args[0], dict)):
                if tool_args:
                    tool_args = [{"value": tool_args[0]}]
                else:
                    tool_args = [{}]
                tool_kwargs = {}

        if not tool_args and not tool_kwargs:
            last_user = self._get_last_user_item(chat_history)
            query = (last_user.content or "").strip() if last_user else ""
            if tool_meta and self._signature_prefers_payload(tool_meta.signature):
                tool_args = [self._default_payload_dict(query=query, chat_history=chat_history)]
            elif query:
                tool_args = [query]
            else:
                tool_args = [""]
            args_auto_built = True

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
