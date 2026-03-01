import json
import os
import re
from typing import Any, Mapping, Optional, Sequence

from src.typings import ChatHistory, Role
from src.typings import Session, ChatHistoryItem

from .tool_registry import ToolMetadata, ToolResult
from .tool_retrieval import retrieve_tools


class ControllerToolsMixin:
    # ------------------------------------------------------------------
    # Macro routing: execute Macro tools on the server via task.interact
    # ------------------------------------------------------------------
    def _route_macro_to_server(
        self, tool_name: str, args: list[Any]
    ) -> ToolResult:
        """Route a Macro tool execution to the server-side _execute_macro engine.

        Instead of running the Macro locally (which lacks live KG context),
        this method sends a lightweight action string containing ONLY the tool
        name and entity list.  The server's ``_execute_macro`` constructs the
        full payload (task_text, trace, live actions_spec callables, etc.).

        This avoids serializing the entire trace/payload into the chat history
        which was causing exponential context bloat and preventing the macro
        from receiving real callable functions.
        """
        task_ref = getattr(self, "_kg_task_ref", None)
        session: Optional[Session] = getattr(self, "_current_session", None)

        if task_ref is None or session is None:
            return ToolResult.failure(
                "macro_route_failed:missing_task_ref_or_session"
            )

        # Extract entities from the payload if present; fall back to
        # regex extraction from task_text.  Do NOT serialize the full
        # payload — the server builds its own with live callables, etc.
        entities: list[str] = []
        if args and isinstance(args[0], Mapping):
            # Handle the {"args": [payload], "kwargs": {}} wrapper that
            # _invoke_tool_by_payload passes.  Drill down to the actual
            # payload dict when the wrapper structure is detected.
            raw_arg = args[0]
            if "args" in raw_arg and isinstance(raw_arg.get("args"), (list, tuple)):
                inner_args = raw_arg["args"]
                payload_dict = inner_args[0] if inner_args and isinstance(inner_args[0], Mapping) else raw_arg
            else:
                payload_dict = raw_arg

            raw_entities = payload_dict.get("entities") or []
            if isinstance(raw_entities, (list, tuple)):
                entities = [str(e) for e in raw_entities if e]
            # Fallback: extract from task_text when LLM omits entities
            if not entities:
                task_text = str(payload_dict.get("task_text") or "")
                ent_match = re.search(
                    r"Entities\s*:\s*\[([^\]]+)\]", task_text, flags=re.IGNORECASE
                )
                if ent_match:
                    entities = [
                        e.strip().strip("'\"")
                        for e in ent_match.group(1).split(",")
                        if e.strip()
                    ]
                if not entities:
                    ent_match2 = re.search(
                        r"Entities\s*:\s*([^\n\r;]+)", task_text, flags=re.IGNORECASE
                    )
                    if ent_match2:
                        entities = [
                            e.strip().strip("'\"")
                            for e in ent_match2.group(1).split(",")
                            if e.strip()
                        ]
        try:
            tool_name_json = json.dumps(str(tool_name), ensure_ascii=True)
            # BYPASS: Join with pipe to avoid environment comma-splitting
            entities_str = "|".join([str(e) for e in entities])
            entities_json = json.dumps(entities_str, ensure_ascii=True)
        except Exception as exc:
            return ToolResult.failure(f"macro_route_failed:serialize:{exc}")
        action_str = f"Action: execute_macro({tool_name_json}, {entities_json})"

        # Inject the lightweight macro action into the live session and let
        # the server handle payload construction + execution.
        session.chat_history.inject(
            ChatHistoryItem(role=Role.AGENT, content=action_str)
        )

        try:
            task_ref.interact(session)
        except Exception as exc:
            return ToolResult.failure(f"macro_execution_failed:{exc}")

        # The server appends the observation as the last USER message.
        try:
            obs_item = session.chat_history.get_item_deep_copy(-1)
            observation = obs_item.content or ""
        except Exception:
            observation = ""

        self._log_flow_event(
            "macro_routed",
            tool_name=tool_name,
            observation_preview=observation[:200],
        )

        return ToolResult.success_result(
            {"macro_observation": observation, "tool_name": tool_name, "tool_type": "macro"}
        )

    def _read_tool_source(self, tool_name: str) -> Optional[str]:
        resolved_name = (
            self._registry.resolve_name(tool_name)
            if hasattr(self._registry, "resolve_name")
            else None
        )
        resolved_name = resolved_name or tool_name
        meta = self._get_tool_metadata(resolved_name)
        environment = meta.environment if meta else None
        tool_path = None
        if hasattr(self._registry, "_get_tool_path"):
            tool_path = self._registry._get_tool_path(resolved_name, environment=environment)
        candidate_paths: list[str] = []
        if tool_path:
            candidate_paths.append(tool_path)
        if not environment:
            current_env = self._resolved_environment_label()
            if hasattr(self._registry, "_get_tool_path") and current_env:
                candidate_paths.append(
                    self._registry._get_tool_path(resolved_name, environment=current_env)
                )
        if hasattr(self._registry, "_get_tool_path") and environment:
            candidate_paths.append(self._registry._get_tool_path(resolved_name, environment=None))

        seen: set[str] = set()
        for path in candidate_paths:
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return handle.read()
            except Exception:
                pass

        tools_dir = getattr(self._registry, "tools_dir", None)
        if tools_dir:
            filename = f"{resolved_name}.py"
            for root, _, files in os.walk(tools_dir):
                if filename in files:
                    try:
                        with open(os.path.join(root, filename), "r", encoding="utf-8") as handle:
                            return handle.read()
                    except Exception:
                        break
        return None

    def _parse_tool_invoke_contract(self, tool_name: str) -> Optional[dict[str, Any]]:
        source = self._read_tool_source(tool_name)
        if not source:
            return None

        def _find_line(label: str) -> Optional[str]:
            match = re.search(
                rf"{re.escape(label)}\\s*(.+)",
                source,
                flags=re.IGNORECASE,
            )
            if not match:
                return None
            return match.group(1).strip()

        def _parse_list(raw: Optional[str]) -> list[str]:
            if not raw:
                return []
            raw = raw.strip()
            bracket = re.search(r"(\\[[^\\]]*\\])", raw)
            candidate = bracket.group(1) if bracket else raw
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x]
            except Exception:
                pass
            cleaned = candidate.strip().strip("[]")
            if not cleaned:
                return []
            parts = []
            for chunk in cleaned.split(","):
                val = chunk.strip().strip("'").strip('"')
                if val:
                    parts.append(val)
            return parts

        invoke_with = _find_line("INVOKE_WITH:")
        required = _parse_list(_find_line("RUN_PAYLOAD_REQUIRED:"))
        optional = _parse_list(_find_line("RUN_PAYLOAD_OPTIONAL:"))
        if not invoke_with:
            req_line = _find_line("input_schema_required:")
            opt_line = _find_line("input_schema_optional:")
            if req_line:
                required = [v.strip() for v in req_line.split(",") if v.strip()]
                optional = [v.strip() for v in (opt_line or "").split(",") if v.strip()]
            if not required:
                doc_match = re.search(
                    r"INPUT_SCHEMA:.*?required=([^;]+);\\s*optional=([^\\n]+)",
                    source,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if doc_match:
                    required = [v.strip() for v in doc_match.group(1).split(",") if v.strip()]
                    optional = [v.strip() for v in doc_match.group(2).split(",") if v.strip()]
            if not required:
                return None
            invoke_with = '{"args":[<RUN_PAYLOAD>], "kwargs":{}}'
        return {
            "invoke_with": invoke_with,
            "required": required,
            "optional": optional,
        }

    def _wrap_payload_with_contract(
        self, invoke_with: str, payload: Mapping[str, Any]
    ) -> Optional[tuple[list[Any], dict[str, Any]]]:
        if not invoke_with:
            return None
        placeholder_pattern = re.compile(r"<RUN_PAYLOAD>|<PAYLOAD>", flags=re.IGNORECASE)
        token = "__RUN_PAYLOAD__"
        template = placeholder_pattern.sub(f"\"{token}\"", invoke_with)
        try:
            wrapper = json.loads(template)
        except Exception:
            return None

        def _replace_token(value: Any) -> Any:
            if value == token:
                return dict(payload)
            if isinstance(value, list):
                return [_replace_token(item) for item in value]
            if isinstance(value, dict):
                return {k: _replace_token(v) for k, v in value.items()}
            return value

        wrapper = _replace_token(wrapper)
        if isinstance(wrapper, dict) and ("args" in wrapper or "kwargs" in wrapper):
            args = list(wrapper.get("args") or [])
            kwargs = dict(wrapper.get("kwargs") or {})
            return args, kwargs
        return [dict(payload)], {}
    # DEPRECATED: Removed per zero-mutation policy
    # This function violated the "no helpful mutations" requirement by:
    # 1. Overwriting trace with normalized version
    # 2. Injecting actions_spec from text extraction
    # 3. Creating fallback actions_spec from trace
    # Tools must now handle missing/invalid fields themselves.
    def _normalize_analyzer_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        # NO-OP: Just return payload as-is
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

    def _validate_payload_shape(
        self, payload: Mapping[str, Any], tool_meta: Optional[ToolMetadata]
    ) -> list[str]:
        errors: list[str] = []
        if tool_meta is None:
            return errors
        contract = self._parse_tool_invoke_contract(tool_meta.name)
        required_keys: list[str] = []
        if contract and contract.get("invoke_with"):
            required_keys = list(contract.get("required") or [])
        elif isinstance(tool_meta.required_keys, list):
            required_keys = list(tool_meta.required_keys)
        if required_keys == ["payload"] and isinstance(tool_meta.input_schema, Mapping):
            props = tool_meta.input_schema.get("properties") or {}
            payload_schema = props.get("payload") if isinstance(props, Mapping) else None
            nested_required = []
            if isinstance(payload_schema, Mapping):
                nested_required = list(payload_schema.get("required") or [])
            if nested_required:
                required_keys = [str(k) for k in nested_required if str(k)]
        for key in required_keys:
            if key not in payload:
                errors.append(f"missing:{key}")
        return errors

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

        # ---- Macro routing: bypass local execution, send to server ----
        _is_macro = (
            (tool_meta and tool_meta.tool_type and "macro" in tool_meta.tool_type.lower())
            or "macro" in resolved_name.lower()
        )
        if _is_macro and self._resolved_environment_label() == "knowledge_graph":
            self._log_flow_event(
                "macro_route_detected",
                tool_name=resolved_name,
                reason=reason,
            )
            self._mark_tool_invoked(resolved_name)
            self._tool_invocation_attempts += 1
            result = self._route_macro_to_server(resolved_name, [tool_args])
            if result.success:
                self._tool_invocation_successes += 1
            return result

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
                if set(args[0].keys()) == {"payload"} and isinstance(args[0].get("payload"), Mapping):
                    payload = dict(args[0].get("payload") or {})
                else:
                    payload = dict(args[0])
            if payload is None:
                return self._fail_payload_invoke(
                    resolved_name, "missing_payload_dict_for_payload_tool"
                )
            payload_errors = self._validate_payload_shape(payload, tool_meta)
            if payload_errors:
                missing_keys = sorted(
                    {err.split("missing:", 1)[1] for err in payload_errors if err.startswith("missing:")}
                )
                run_id = payload.get("run_id") if isinstance(payload, Mapping) else None
                if missing_keys:
                    error = (
                        "missing_required_keys:"
                        + ",".join(missing_keys)
                        + f"|tool={resolved_name}|run_id={run_id or ''}"
                    )
                else:
                    error = (
                        "invalid_payload:"
                        + ",".join(payload_errors)
                        + f"|tool={resolved_name}|run_id={run_id or ''}"
                    )
                return self._fail_payload_invoke(resolved_name, error)
            if kwargs:
                return self._fail_payload_invoke(
                    resolved_name, "payload_kwargs_not_allowed"
                )
            if isinstance(payload, Mapping) and set(payload.keys()) == {"payload"} and isinstance(payload.get("payload"), Mapping):
                payload = dict(payload.get("payload") or {})
            raw_actions_spec = payload.get("actions_spec")
            # NO MUTATIONS: Pass payload as-is, let tool handle missing fields
            args = [payload]
            trace_source = "payload"  # Always payload now, no injection
            if len(args) != 1 or not isinstance(args[0], Mapping) or "payload" in args[0]:
                return self._fail_payload_invoke(
                    resolved_name, "payload_tool_requires_single_dict_arg"
                )
            actions_spec_source = "missing"
            if isinstance(raw_actions_spec, Mapping):
                actions_spec_source = "payload" if raw_actions_spec else "payload_empty"
            actions_spec_keys = []
            if isinstance(args[0].get("actions_spec"), Mapping):
                actions_spec_keys = sorted(str(k) for k in args[0]["actions_spec"].keys())
            trace_len = 0
            last_actions: list[str] = []
            if isinstance(args[0].get("trace"), list):
                trace_list = args[0]["trace"]
                trace_len = len(trace_list)
                for step in trace_list[-3:]:
                    if isinstance(step, Mapping):
                        action = step.get("action")
                        if isinstance(action, str):
                            last_actions.append(action)
                    else:
                        last_actions.append(str(step))
            self._log_flow_event(
                "tool_trace_meta",
                tool_name=resolved_name,
                trace_source=trace_source,
                trace_len=trace_len,
                last_actions=last_actions,
            )
            self._log_flow_event(
                "tool_actions_spec_meta",
                tool_name=resolved_name,
                actions_spec_source=actions_spec_source,
                actions_spec_keys=actions_spec_keys,
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

        self._mark_tool_invoked(resolved_name or tool_name)
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
        self._mark_tool_invoked(tool.name)
        self._tool_invocation_attempts += 1

        # ---- Macro routing: bypass local execution, send to server ----
        _is_macro = (
            (tool.tool_type and "macro" in tool.tool_type.lower())
            or "macro" in tool.name.lower()
        )
        if _is_macro and self._resolved_environment_label() == "knowledge_graph":
            self._log_flow_event(
                "macro_route_detected",
                tool_name=tool.name,
                reason=reason,
            )
            result = self._route_macro_to_server(tool.name, list(args))
            if result.success:
                self._tool_invocation_successes += 1
            self._log_flow_event(
                "tool_agent_output",
                tool_name=tool.name,
                success=result.success,
                error=result.error,
                output=result.output,
            )
            self._trace("tool_agent_result", self._format_tool_result(tool.name, result))
            return result, args, kwargs

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
        return None

    def _auto_build_tool_args(
        self,
        tool: ToolMetadata,
        *,
        query: str,
        chat_history: ChatHistory,
    ) -> Optional[dict[str, Any]]:
        return None


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
        self._mark_tool_invoked(tool_name)
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
            return (
                ToolResult.failure("missing_tool_args"),
                [],
                {},
                tool_name,
            )

        # NO MUTATION POLICY: Payload passed as-is, tool handles missing fields

        effective_name = resolved_name or tool_name

        # ---- Macro routing: bypass local execution, send to server ----
        _is_macro = (
            (tool_meta and tool_meta.tool_type and "macro" in tool_meta.tool_type.lower())
            or "macro" in effective_name.lower()
        )
        if _is_macro and self._resolved_environment_label() == "knowledge_graph":
            self._log_flow_event(
                "macro_route_detected",
                tool_name=effective_name,
                reason="internal_tool",
            )
            self._tool_invocation_attempts += 1
            result = self._route_macro_to_server(effective_name, list(tool_args))
            if result.success:
                self._tool_invocation_successes += 1
            return result, tool_args, tool_kwargs, effective_name

        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            effective_name,
            *tool_args,
            invocation_context={"environment": self._resolved_environment_label()},
            **tool_kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        self._log_tool_invocation_event(
            tool_name=effective_name,
            args=tool_args,
            kwargs=tool_kwargs,
            result=result,
            reason="internal_tool",
            args_auto_built=args_auto_built,
            decision_action="use_tool",
        )
        return result, tool_args, tool_kwargs, effective_name
