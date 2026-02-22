import hashlib
import json
import os
import re
from typing import Any, Mapping, Optional

from src.typings import ChatHistory, ChatHistoryItem, Role


class ControllerOrchestratorMixin:
    _INTERNAL_TOOL_RE = re.compile(
        r"<internal_tool\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</internal_tool>"
    )
    _RELATIONS_OBS_PREFIX = "Observation: ["

    def _extract_internal_tool_body(self, text: str) -> Optional[str]:
        match = self._INTERNAL_TOOL_RE.search(text or "")
        if not match:
            return None
        return match.group("body")

    def _truncate_relations_observations(
        self, history_text: str, max_list_chars: int = 400
    ) -> str:
        if not history_text:
            return history_text
        lines = history_text.splitlines()
        for idx, line in enumerate(lines):
            if "get_relations(" not in line:
                continue
            obs_idx = line.find(self._RELATIONS_OBS_PREFIX)
            if obs_idx < 0:
                continue
            start = obs_idx + len(self._RELATIONS_OBS_PREFIX)
            end = line.find("]", start)
            if end < 0:
                continue
            body = line[start:end]
            if len(body) <= max_list_chars:
                continue
            trimmed = body[:max_list_chars].rstrip()
            removed = len(body) - len(trimmed)
            lines[idx] = (
                line[:start]
                + trimmed
                + f" ... [truncated {removed} chars]"
                + line[end:]
            )
        return "\n".join(lines)

    def _parse_json_lenient(self, text: str) -> Optional[Mapping[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None

        parsed = self._parse_creation_payload(raw)
        if isinstance(parsed, Mapping):
            return parsed

        obj_text = self._extract_first_json_object(raw)
        if obj_text:
            parsed = self._parse_creation_payload(obj_text)
            if isinstance(parsed, Mapping):
                return parsed

        repaired = raw
        repaired = repaired.replace("}}},\"reason\"", "}},\"reason\"", 1)
        repaired = repaired.replace("}}}, \"reason\"", "}}, \"reason\"", 1)
        repaired = repaired.replace("}}},\"reason\":", "}},\"reason\":", 1)
        repaired = repaired.replace("}}}, \"reason\":", "}}, \"reason\":", 1)

        def _brace_delta(value: str) -> int:
            depth = 0
            in_str = False
            esc = False
            for ch in value:
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
            return depth

        delta = _brace_delta(repaired)
        if delta > 0:
            repaired = repaired + ("}" * delta)

        parsed = self._parse_creation_payload(repaired)
        if isinstance(parsed, Mapping):
            return parsed

        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(repaired)
            if isinstance(obj, Mapping):
                return obj
        except Exception:
            return None
        return None
    def _format_orchestrator_docstring(self, tool) -> str:
        base_doc = (tool.docstring or "").strip()
        if base_doc:
            return base_doc
        description = (tool.description or "").strip()
        return description or tool.name

    def _orchestrator_compact_existing_tools(self) -> list[dict[str, Any]]:
        # Filter tools by current environment
        current_env = self._resolved_environment_label()
        tools = (
            self._registry.list_latest_tools(environment=current_env)
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools(environment=current_env)
        )
        print(f"[ORCHESTRATOR] Found {len(tools)} tools for environment '{current_env}'")
        compact: list[dict[str, Any]] = []
        for t in tools:
            contract = self._parse_tool_invoke_contract(t.name)
            invoke_with = contract.get("invoke_with") if contract else None
            run_payload_required = list(contract.get("required") or []) if contract else []
            run_payload_optional = list(contract.get("optional") or []) if contract else []
            if not invoke_with:
                invoke_with = '{"args":[<RUN_PAYLOAD>], "kwargs":{}}'
            if not run_payload_required:
                run_payload_required = list(t.required_keys or [])
            if not run_payload_optional:
                run_payload_optional = list(t.optional_keys or [])
            compact.append(
                {
                    "name": t.name,
                    "signature": t.signature,
                    "docstring": self._format_orchestrator_docstring(t),
                    "input_schema": t.input_schema,
                    "required_keys": t.required_keys,
                    "optional_keys": t.optional_keys,
                    "property_types": t.property_types,
                    "invoke_with": invoke_with,
                    "run_payload_required": run_payload_required,
                    "run_payload_optional": run_payload_optional,
                    # usage_count removed - redundant with success+failure counts
                    "success": t.success_count,
                    "failure": t.failure_count,
                    "reliability_score": t.reliability_score,
                    "negative_marks": t.negative_marks,
                }
            )
        # Limit to 15 most recent tools to reduce token usage
        return compact[-15:]

    def _orchestrator_request_prompt(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        solver_recommendation: Optional[str] = None,
    ) -> str:
        history_text_full = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=1200,
            preserve_first_user_n=2,
        )
        history_lines = history_text_full.splitlines()[-6:]
        history_text = "\n".join(history_lines)
        cleaned_query = self._truncate((query or ""), 1200)

        output_schema: dict[str, Any] = {
            "action": "use_tool|create_tool|no_tool",
            "tool_name": "only if use_tool",
            "reason": "short reason",
        }
        if getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3":
            output_schema["insufficiency"] = "why existing tools fail the gate"
            output_schema["needed_capabilities"] = "what the new tool must provide"
            output_schema["evidence"] = "specific symptoms from inputs/trace/tool metadata"
            output_schema["must_differ_from_existing"] = "delta vs existing tools"
            output_schema["self_test_cases"] = "minimal tests"
        payload = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "output_schema": output_schema,
        }
        if solver_recommendation:
            payload["solver_recommendation"] = solver_recommendation
            payload["recommendation_note"] = (
                "Solver provided a draft response. Use it to decide whether a tool "
                "can validate or strengthen the draft before returning it."
            )
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _tool_orchestrator_request_prompt(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        solver_recommendation: Optional[str] = None,
    ) -> str:
        history_text = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=None,
            preserve_first_user_n=2,
        )
        history_text = self._truncate_relations_observations(history_text)
        cleaned_query = (query or "").strip()

        output_schema: dict[str, Any] = {
            "action": "use_tool|create_tool",
            "tool_name": "only if use_tool",
            "reason": "short reason",
        }
        if getattr(self, "_toolgen_pipeline_name", "baseline") == "aggregate3":
            output_schema["insufficiency"] = "why existing tools fail the gate"
            output_schema["needed_capabilities"] = "what the new tool must provide"
        payload = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "output_schema": output_schema,
        }
        if solver_recommendation:
            payload["solver_recommendation"] = solver_recommendation
            payload["recommendation_note"] = (
                "Solver provided a draft response. Use it to decide whether a tool "
                "can validate or strengthen the draft before returning it."
            )
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _tool_invoker_request_prompt(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        suggestion: Optional[Mapping[str, Any]] = None,
        actions_spec: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        state_dir: Optional[str] = None,
        repair_note: Optional[str] = None,
    ) -> str:
        if actions_spec is None:
            actions_spec = self._available_actions_spec()
        history_text = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=None,
            preserve_first_user_n=2,
        )
        cleaned_query = (query or "").strip()
        if run_id is None or state_dir is None:
            meta = self._get_run_task_metadata()
            run_id, state_dir, _ = self._scoped_run_id_state_dir(
                task_text_full=cleaned_query,
                asked_for="",
                sample_index=meta.get("sample_index"),
                task_name=meta.get("task_name") or self._environment_label,
            )

        payload = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "AVAILABLE_ACTIONS_SPEC": actions_spec,
            "run_id": run_id,
            "state_dir": state_dir,
            "suggestion": suggestion or {},
            "tools_summary": self._tool_invoker_tools_summary(),
            "output_schema": {
                "tool_name": "required",
                "payload": "object with required tool keys",
                "reason": "short reason",
            },
        }
        if repair_note:
            payload["invoker_error"] = repair_note
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _tool_invoker_tools_summary(self) -> list[dict[str, Any]]:
        current_env = self._resolved_environment_label()
        tools = (
            self._registry.list_latest_tools(environment=current_env)
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools(environment=current_env)
        )
        summary: list[dict[str, Any]] = []
        for tool in tools:
            summary.append(
                {
                    "name": tool.name,
                    "reliability_score": tool.reliability_score,
                    "negative_marks": tool.negative_marks,
                }
            )
        return summary

    def _available_actions_spec(self) -> dict[str, Any]:
        env = self._resolved_environment_label()
        actions: list[str] = []
        if env == "knowledge_graph":
            try:
                from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI

                actions = KnowledgeGraphAPI.get_valid_api_name_list()
            except Exception:
                actions = []
        elif env in {"db_bench", "mysql"}:
            actions = ["operation", "answer"]
        elif env == "os_interaction":
            actions = ["bash", "finish"]
        return {str(name): {} for name in actions}

    def _validate_tool_invoker_payload(
        self, tool_name: Optional[str], payload: Any
    ) -> list[str]:
        errors: list[str] = []
        if not isinstance(payload, Mapping):
            return ["payload_not_mapping"]
        if not tool_name:
            errors.append("missing_tool_name")
            return errors

        tool_meta = self._get_tool_metadata(str(tool_name))
        if tool_meta is None:
            errors.append("unknown_tool_name")
            return errors

        contract = self._parse_tool_invoke_contract(str(tool_name))
        required_keys = []
        if contract and contract.get("invoke_with"):
            required_keys = list(contract.get("required") or [])
        else:
            required_keys = list(tool_meta.required_keys or [])
        for key in required_keys:
            if key not in payload:
                errors.append(f"missing:{key}")
        return errors



    def _parse_orchestrator_payload(self, content: str) -> Optional[Mapping[str, Any]]:
        text = (content or "").strip()
        if not text:
            return None

        parsed = self._parse_json_lenient(text)
        if isinstance(parsed, Mapping):
            return parsed
        return None

    def _orchestrate_decision(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        solver_recommendation: Optional[str] = None,
        observation_triggers: Optional[list[dict[str, Any]]] = None,
        last_observation: Optional[dict[str, Any]] = None,  # reserved for LLM prompt enrichment
    ) -> dict[str, Any]:
        _ = last_observation  # reserved for future orchestrator prompt enrichment
        if not self._orchestrator_agent:
            return {"action": "no_tool"}

        # Trigger-based fast path: auto-decide based on observation characteristics
        if observation_triggers:
            try:
                tools = self._orchestrator_compact_existing_tools()
            except Exception:
                tools = []
            for trigger in observation_triggers:
                trigger_type = trigger.get("type")
                if trigger_type in {"size_trigger", "error_trigger"}:
                    # Find an existing generated tool for this environment
                    filter_tool = next(
                        (
                            t
                            for t in tools
                            if isinstance(t.get("name"), str)
                            and t["name"].endswith("_generated_tool")
                        ),
                        None,
                    )
                    if filter_tool:
                        return {
                            "action": "use_tool",
                            "tool_name": filter_tool["name"],
                            "reason": f"{trigger_type}: {trigger.get('reason', '')}",
                        }
                    else:
                        return {
                            "action": "create_tool",
                            "reason": f"{trigger_type}_no_tool: {trigger.get('reason', '')}",
                        }

        prompt = self._orchestrator_request_prompt(
            query, chat_history, solver_recommendation=solver_recommendation
        )
        try:
            tools = self._orchestrator_compact_existing_tools()
        except Exception:
            tools = []
        tool_list_text = json.dumps(tools, ensure_ascii=True, default=str)
        original_prompt = getattr(self._orchestrator_agent, "_system_prompt", "") or ""
        self._orchestrator_agent._system_prompt = (
            original_prompt
            + "\n\nThere are environment tools that you must NOT consider. Only use tools in the following list to make your decision. if the list is empty, you must generate a tool:\n"
            + tool_list_text
        ).strip()
        self._write_agent_system_prompt("top_orchestrator", self._orchestrator_agent._system_prompt)
        self._trace("orchestrator_input", prompt)
        self._log_flow_event(
            "orchestrator_input",
            chat_history=chat_history,
            prompt=prompt,
        )
        orchestration_history = ChatHistory()
        orchestration_history = self._safe_inject(
            orchestration_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        try:
            response = self._orchestrator_agent._inference(orchestration_history)
        finally:
            self._orchestrator_agent._system_prompt = original_prompt
        self._trace("orchestrator_result", response.content)
        self._log_flow_event(
            "orchestrator_output",
            chat_history=chat_history,
            output=response.content or "",
        )
        payload = self._parse_orchestrator_payload(response.content)
        if not isinstance(payload, Mapping):
            return {"action": "no_tool"}
        action = str(payload.get("action") or "no_tool").strip().lower()
        if action not in {"use_tool", "no_tool", "create_tool"}:
            action = "no_tool"
        tool_name = payload.get("tool_name")
        if action == "use_tool" and tool_name and not str(tool_name).endswith("_generated_tool"):
            tool_name = None
        if tools:
            selected_name = str(tool_name).strip() if tool_name else None
            if not selected_name or action != "use_tool":
                fallback_name = None
                for tool in tools:
                    candidate = tool.get("name")
                    if isinstance(candidate, str) and candidate.endswith("_generated_tool"):
                        fallback_name = candidate
                        break
                if fallback_name:
                    action = "use_tool"
                    tool_name = fallback_name
        return {
            "action": action,
            "tool_name": str(tool_name).strip() if tool_name else None,
            "reason": payload.get("reason"),
            "insufficiency": payload.get("insufficiency"),
            "needed_capabilities": payload.get("needed_capabilities"),
            "evidence": payload.get("evidence"),
            "must_differ_from_existing": payload.get("must_differ_from_existing"),
            "self_test_cases": payload.get("self_test_cases"),
        }

    def _tool_orchestrate_decision(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        solver_recommendation: Optional[str] = None,
    ) -> dict[str, Any]:
        agent = getattr(self, "_tool_orchestrator_agent", None)
        if agent is None:
            return {"action": "create_tool", "reason": "missing_tool_orchestrator"}
        prompt = self._tool_orchestrator_request_prompt(
            query, chat_history, solver_recommendation=solver_recommendation
        )
        try:
            tools = self._orchestrator_compact_existing_tools()
        except Exception:
            tools = []
        tool_list_text = json.dumps(tools, ensure_ascii=True, default=str)
        original_prompt = getattr(agent, "_system_prompt", "") or ""
        agent._system_prompt = (
            original_prompt
            + "\n\nThere are environment tools that you must NOT consider. Only use tools in the following list to make your decision. if the list is empty, you must generate a tool:\n"
            + tool_list_text
        ).strip()
        self._write_agent_system_prompt("tool_orchestrator", agent._system_prompt)
        self._trace("tool_orchestrator_input", prompt)
        self._log_flow_event(
            "tool_orchestrator_input",
            chat_history=chat_history,
            prompt=prompt,
        )
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        try:
            response = agent._inference(tool_history)
        finally:
            agent._system_prompt = original_prompt
        self._trace("tool_orchestrator_result", response.content)
        self._log_flow_event(
            "tool_orchestrator_output",
            chat_history=chat_history,
            output=response.content or "",
        )
        payload = self._parse_orchestrator_payload(response.content)
        if not isinstance(payload, Mapping):
            return {"action": "create_tool", "reason": "parse_failed"}
        action = str(payload.get("action") or "create_tool").strip().lower()
        if action not in {"use_tool", "create_tool"}:
            action = "create_tool"
        tool_name = payload.get("tool_name")
        if tool_name and not str(tool_name).endswith("_generated_tool"):
            tool_name = None
            action = "create_tool"
        return {
            "action": action,
            "tool_name": str(tool_name).strip() if tool_name else None,
            "reason": payload.get("reason"),
            "insufficiency": payload.get("insufficiency"),
            "needed_capabilities": payload.get("needed_capabilities"),
        }

    def _tool_invoker_decision(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        suggestion: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        def _summarize_text(text: str) -> dict[str, Any]:
            safe = text or ""
            return {
                "len": len(safe),
                "sha1": hashlib.sha1(safe.encode("utf-8")).hexdigest(),
                "preview": self._truncate(safe, 220),
                "tail": self._truncate(safe[-220:], 220) if safe else "",
            }

        agent = getattr(self, "_tool_invoker_agent", None)
        if agent is None:
            return {"tool_name": None, "payload": None, "reason": "missing_tool_invoker"}
        actions_spec = self._available_actions_spec()
        meta = self._get_run_task_metadata()
        run_id, state_dir, _ = self._scoped_run_id_state_dir(
            task_text_full=(query or "").strip(),
            asked_for="",
            sample_index=meta.get("sample_index"),
            task_name=meta.get("task_name") or self._environment_label,
        )
        prompt = self._tool_invoker_request_prompt(
            query,
            chat_history,
            suggestion=suggestion,
            actions_spec=actions_spec,
            run_id=run_id,
            state_dir=state_dir,
        )
        try:
            tools = self._orchestrator_compact_existing_tools()
        except Exception:
            tools = []
        tool_list_text = json.dumps(tools, ensure_ascii=True, default=str)
        original_prompt = getattr(agent, "_system_prompt", "") or ""
        agent._system_prompt = (
            original_prompt
            + "\n\nYou must ONLY select tools from the following list:\n"
            + tool_list_text
        ).strip()
        try:
            self._append_generated_tools_log(
                {
                    "event": "tool_invoker_request",
                    "tool_name_suggestion": (suggestion or {}).get("tool_name"),
                    "existing_tools_count": len(tools),
                    "run_id": run_id,
                    "state_dir_basename": os.path.basename(state_dir) if state_dir else None,
                    "environment_label": self._resolved_environment_label(),
                }
            )
        except Exception:
            pass
        self._write_agent_system_prompt("tool_invoker", agent._system_prompt)
        self._trace("tool_invoker_input", prompt)
        self._log_flow_event(
            "tool_invoker_input",
            chat_history=chat_history,
            prompt=prompt,
        )
        self._append_tool_invoker_io_log(
            {
                "event": "tool_invoker_input",
                "system_prompt": agent._system_prompt,
                "user_prompt": prompt,
                "suggestion": suggestion,
            }
        )
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response_content = ""
        output_summary: dict[str, Any] = {}
        parse_error: Optional[str] = None
        payload: Optional[Mapping[str, Any]] = None
        wrapper_stripped = False
        try:
            response = agent._inference(tool_history)
            raw_response_content = response.content or ""
            response_content = raw_response_content
            output_summary = _summarize_text(response_content)
            wrapper_body = self._extract_internal_tool_body(response_content)
            if wrapper_body is not None:
                wrapper_stripped = True
                response_content = wrapper_body.strip()
            if "<" in response_content or ">" in response_content:
                parse_error = "forbidden_wrapper_chars"
                payload = None
            else:
                try:
                    json.loads(response_content)
                except Exception as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"
            self._trace("tool_invoker_result", response_content)
            self._log_flow_event(
                "tool_invoker_output",
                chat_history=chat_history,
                output=response_content,
            )
            self._append_tool_invoker_io_log(
                {
                    "event": "tool_invoker_output",
                    "content": raw_response_content,
                }
            )
            if payload is None and "<" not in response_content and ">" not in response_content:
                payload = self._parse_orchestrator_payload(response_content)
        finally:
            agent._system_prompt = original_prompt
        def _validate_top_level(obj: Any) -> tuple[list[str], Optional[str], Optional[Mapping[str, Any]], Optional[str]]:
            errs: list[str] = []
            if not isinstance(obj, Mapping):
                return ["parse_failed"], None, None, None
            tool_name_val = obj.get("tool_name")
            payload_val = obj.get("payload")
            reason_val = obj.get("reason")
            if not isinstance(tool_name_val, str) or not tool_name_val.strip():
                errs.append("missing_tool_name")
            if not isinstance(payload_val, Mapping):
                errs.append("payload_not_mapping")
            if not isinstance(reason_val, str) or not reason_val.strip():
                errs.append("missing_reason")
            return errs, tool_name_val, payload_val if isinstance(payload_val, Mapping) else None, reason_val

        errors, tool_name, payload_dict, reason_text = _validate_top_level(payload)
        if errors:
            repair_note = "Invalid JSON. Output ONE JSON object with keys tool_name,payload,reason."
            repair_prompt = self._tool_invoker_request_prompt(
                query,
                chat_history,
                suggestion=suggestion,
                actions_spec=actions_spec,
                run_id=run_id,
                state_dir=state_dir,
                repair_note=repair_note,
            )
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_request",
                        "tool_name_suggestion": (suggestion or {}).get("tool_name"),
                        "existing_tools_count": len(tools),
                        "run_id": run_id,
                        "state_dir_basename": os.path.basename(state_dir) if state_dir else None,
                        "environment_label": self._resolved_environment_label(),
                        "repair": True,
                    }
                )
            except Exception:
                pass
            self._append_tool_invoker_io_log(
                {
                    "event": "tool_invoker_input_repair",
                    "system_prompt": (
                        original_prompt
                        + "\n\nYou must ONLY select tools from the following list:\n"
                        + tool_list_text
                    ).strip(),
                    "user_prompt": repair_prompt,
                }
            )
            repair_history = ChatHistory()
            repair_history = self._safe_inject(
                repair_history, ChatHistoryItem(role=Role.USER, content=repair_prompt)
            )
            try:
                agent._system_prompt = (
                    original_prompt
                    + "\n\nYou must ONLY select tools from the following list:\n"
                    + tool_list_text
                ).strip()
                repair_response = agent._inference(repair_history)
                repair_content = repair_response.content or ""
            finally:
                agent._system_prompt = original_prompt
            self._append_tool_invoker_io_log(
                {
                    "event": "tool_invoker_output_repair",
                    "content": repair_content,
                }
            )
            output_summary = _summarize_text(repair_content)
            parse_error = None
            wrapper_body = self._extract_internal_tool_body(repair_content)
            response_content = wrapper_body.strip() if wrapper_body is not None else repair_content
            if wrapper_body is not None:
                wrapper_stripped = True
            if "<" in response_content or ">" in response_content:
                parse_error = "forbidden_wrapper_chars"
                payload = None
            else:
                try:
                    json.loads(response_content)
                except Exception as exc:
                    parse_error = f"{type(exc).__name__}: {exc}"
                payload = self._parse_orchestrator_payload(response_content)
            errors, tool_name, payload_dict, reason_text = _validate_top_level(payload)
        if errors:
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_result",
                        "parse_ok": False,
                        "reason": "parse_failed",
                        "tool_name": None,
                        "errors": errors,
                        "payload_keys_count": 0,
                        "output": output_summary,
                        "parse_error": parse_error,
                        "wrapper_stripped": wrapper_stripped,
                        "environment_label": self._resolved_environment_label(),
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "parse_failed"}

        payload_errors = self._validate_tool_invoker_payload(tool_name, payload_dict)
        if payload_errors:
            payload_keys = sorted(str(k) for k in payload_dict.keys()) if payload_dict else []
            contract = self._parse_tool_invoke_contract(str(tool_name)) if tool_name else None
            required_keys = list(contract.get("required") or []) if contract else []
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_result",
                        "parse_ok": True,
                        "reason": "invalid_payload",
                        "tool_name": tool_name,
                        "errors": payload_errors,
                        "payload_keys": payload_keys,
                        "payload_keys_count": len(payload_keys),
                        "required_keys": required_keys,
                        "output": output_summary,
                        "wrapper_stripped": wrapper_stripped,
                        "environment_label": self._resolved_environment_label(),
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "invalid_payload"}
        if tool_name and not str(tool_name).endswith("_generated_tool"):
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_result",
                        "parse_ok": True,
                        "reason": "invalid_tool_name",
                        "tool_name": tool_name,
                        "payload_keys_count": len(payload_dict or {}),
                        "output": output_summary,
                        "wrapper_stripped": wrapper_stripped,
                        "environment_label": self._resolved_environment_label(),
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "invalid_tool_name"}
        try:
            self._append_generated_tools_log(
                {
                    "event": "tool_invoker_result",
                    "parse_ok": True,
                    "reason": "ok",
                    "tool_name": tool_name,
                    "payload_keys_count": len(payload_dict or {}),
                    "wrapper_stripped": wrapper_stripped,
                    "environment_label": self._resolved_environment_label(),
                }
            )
        except Exception:
            pass
        return {
            "tool_name": str(tool_name).strip() if tool_name else None,
            "payload": payload_dict,
            "reason": reason_text,
        }
