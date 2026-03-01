import hashlib
import json
import os
import re
import sys
from typing import Any, Mapping, Optional

from src.typings import ChatHistory, ChatHistoryItem, Role


class ControllerOrchestratorMixin:
    _INTERNAL_TOOL_RE = re.compile(
        r"<internal_tool\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</internal_tool>"
    )
    _RELATIONS_OBS_PREFIX = "Observation: ["

    def set_registry_dir(self, registry_dir: str) -> None:
        """Align orchestrator registry path with the controller."""
        if not registry_dir:
            return
        try:
            from .tool_registry import get_registry
        except Exception:
            return
        self._registry_dir = os.path.abspath(registry_dir)
        try:
            self._registry = get_registry(self._registry_dir)
        except Exception:
            pass

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
    # ------------------------------------------------------------------
    # Escape hatch baseline tool (Phase 4)
    # ------------------------------------------------------------------
    _REQUEST_NEW_TOOL_ENTRY = {
        "name": "request_new_tool",
        "signature": "request_new_tool(tool_type, reason)",
        "docstring": (
            "CRITICAL: Trigger this ONLY when the Actor Agent is stuck in a loop, "
            "experiences a database timeout (node explosion), or lacks an existing "
            "tool in the catalog to perform the required logic. Do NOT use this if "
            "an existing tool can accomplish the goal. This pauses the Actor and "
            "commands the ToolGen pipeline to write a new Python tool. "
            "Set tool_type='advisory' for read-only recommendation tools, or "
            "'macro' for tools that execute live multi-step KG operations directly."
        ),
        "input_schema": {
            "type": "object",
            "required": ["tool_type", "reason"],
            "properties": {
                "tool_type": {
                    "type": "string",
                    "enum": ["advisory", "macro"],
                    "description": "Whether to generate an advisory (read-only recommender) or macro (autonomous executor) tool.",
                },
                "reason": {
                    "type": "string",
                    "description": "Must follow the template: INPUT: [raw data/variables in trace]. GOAL: [exact transformation or execution needed].",
                },
            },
        },
        "required_keys": ["tool_type", "reason"],
        "optional_keys": [],
        "property_types": {
            "tool_type": "string",
            "reason": "string",
        },
        "invoke_with": None,
        "run_payload_required": [],
        "run_payload_optional": [],
        "success": 0,
        "failure": 0,
        "reliability_score": 1.0,
        "negative_marks": 0,
    }

    # ------------------------------------------------------------------
    # Dynamic registry loader (Phase 3)
    # ------------------------------------------------------------------
    def _load_dynamic_registry(self):
        """
        Load the tool catalog from the persisted registry (metadata.json)
        and filter out inactive / broken tools.

        A tool is considered **inactive** when it has been invoked at least
        once yet has zero successes (i.e. every invocation failed).  All
        other tools — including brand-new tools that have never been invoked
        — are treated as active.

        Returns a list of ToolMetadata objects.  Gracefully returns an empty
        list when the registry is empty or unreadable (first-run safe).
        """
        try:
            current_env = self._resolved_environment_label()
            tools = (
                self._registry.list_latest_tools(environment=current_env)
                if hasattr(self._registry, "list_latest_tools")
                else self._registry.list_tools(environment=current_env)
            )
        except Exception:
            return []

        active_tools = []
        for t in tools:
            # Filter out tools where every invocation has failed
            if t.usage_count > 0 and t.success_count == 0 and t.failure_count > 0:
                continue
            active_tools.append(t)
        return active_tools

    # ------------------------------------------------------------------
    # Escape hatch handler (Phase 4)
    # ------------------------------------------------------------------
    def _handle_escape_hatch(
        self,
        decision: dict,
        query: str,
        chat_history,
    ) -> dict:
        """
        Handle the ``request_new_tool`` escape hatch.

        Pauses the Actor, triggers ToolGen with enriched context from the
        Orchestrator's escape-hatch arguments, and returns an observation
        dict for the Orchestrator to consume on its next turn.

        Returns ``{"success": bool, "tool_name": str|None, "observation": str}``.
        """
        print(
            "[!] ESCAPE HATCH TRIGGERED. "
            "Pausing Actor Agent and spinning up ToolGen.",
            file=sys.stderr,
            flush=True,
        )

        # Extract escape hatch arguments from the orchestrator decision
        reason = str(decision.get("reason") or "")
        tool_type = str(decision.get("tool_type") or "advisory")

        # Build an enriched query for ToolGen from the escape hatch context
        parts = [query]
        parts.append(f"TOOL_TYPE: {tool_type}")
        if reason:
            parts.append(f"FORGE CONTEXT: {reason}")
        toolgen_query = "\n".join(parts)

        self._trace("escape_hatch_trigger", toolgen_query)

        # Trigger ToolGen (force=True, no reuse — existing tools are insufficient)
        # NOTE: force_strict=True is passed inside _run_escape_hatch_toolgen so
        # all escape-hatch tools go through the validation loop (2 rounds for advisory).
        runner = getattr(self, "_run_escape_hatch_toolgen", None)
        if callable(runner):
            new_tool = runner(decision, query, chat_history)
        else:
            new_tool = None

        print(
            f"\n[FORGE_DEBUG] Pipeline returned result type: {type(new_tool)}",
            file=sys.stderr,
            flush=True,
        )
        if new_tool:
            if isinstance(new_tool, Mapping):
                tool_name = new_tool.get("tool_name") or new_tool.get("name")
                code = new_tool.get("code")
                print(
                    f"[FORGE_DEBUG] Tool Name: {tool_name or 'MISSING'}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"[FORGE_DEBUG] Code Found: {'YES' if code else 'NO'}",
                    file=sys.stderr,
                    flush=True,
                )
                if code:
                    print(
                        f"[FORGE_DEBUG] Code Length: {len(code)} chars",
                        file=sys.stderr,
                        flush=True,
                    )
                if new_tool.get("error"):
                    print(
                        f"[FORGE_DEBUG] PIPELINE ERROR: {new_tool.get('error')}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                tool_name = getattr(new_tool, "name", None)
                print(
                    f"[FORGE_DEBUG] Tool Name: {tool_name or 'MISSING'}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    "[FORGE_DEBUG] Code Found: N/A (ToolMetadata)",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            print(
                "[FORGE_DEBUG] FATAL: Pipeline returned None or empty result.",
                file=sys.stderr,
                flush=True,
            )

        if new_tool is not None:
            if isinstance(new_tool, Mapping):
                if (
                    new_tool.get("reason") == "aborted_duplicate"
                    or new_tool.get("error") == "aborted_duplicate"
                ):
                    self._trace("escape_hatch_failed", "aborted_duplicate")
                    return {
                        "success": False,
                        "tool_name": None,
                        "observation": (
                            "Observation: Tool generation FAILED. You attempted to create a tool that is a duplicate "
                            "of an existing tool in the catalog. You MUST either USE an existing tool from the catalog, "
                            "or change your strategy. DO NOT request this specific tool again."
                        ),
                    }
                tool_spec = new_tool.get("tool_spec")
                tool_code = new_tool.get("tool_code") or new_tool.get("code")
                metadata = None
                if tool_spec and tool_code:
                    try:
                        metadata = self._register_tool_from_payload_relaxed(
                            tool_spec, tool_code, chat_history
                        )
                    except Exception:
                        metadata = None
                if metadata is None:
                    self._trace("escape_hatch_failed", "toolgen_returned_mapping")
                    return {
                        "success": False,
                        "tool_name": None,
                        "observation": (
                            "Observation: ToolGen returned code payload but "
                            "registration failed. Check ToolGen logs."
                        ),
                    }
                new_tool = metadata
            # Hot-reload the registry so the Orchestrator sees the new tool
            self._load_dynamic_registry()
            self._trace("escape_hatch_success", new_tool.name)
            setattr(self, "_just_generated_tool", new_tool.name)
            return {
                "success": True,
                "tool_name": new_tool.name,
                "observation": (
                    "Observation: ToolGen successfully created and registered "
                    "the new tool. Review your updated catalog and invoke the "
                    "new tool now to advise the Solver."
                ),
            }
        else:
            self._trace("escape_hatch_failed", "toolgen_returned_none")
            return {
                "success": False,
                "tool_name": None,
                "observation": (
                    "Observation: ToolGen failed to create a stable tool due "
                    "to strict validation constraints. You must re-evaluate "
                    "the problem and attempt to guide the Actor using existing "
                    "tools."
                ),
            }

    def _format_orchestrator_docstring(self, tool) -> str:
        base_doc = (tool.docstring or "").strip()
        if base_doc:
            return base_doc
        description = (tool.description or "").strip()
        return description or tool.name

    def _orchestrator_compact_existing_tools(
        self, *, query_text: Optional[str] = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        # Use dynamic registry loader (filters inactive/broken tools)
        tools = self._load_dynamic_registry()
        latest_tool = None
        if tools:
            try:
                latest_tool = max(
                    tools, key=lambda t: getattr(t, "creation_time", "") or ""
                )
            except Exception:
                latest_tool = None
        current_env = self._resolved_environment_label()
        if query_text:
            try:
                if hasattr(self._registry, "retrieve_similar_tools"):
                    retrieved = self._registry.retrieve_similar_tools(
                        query_text, top_k=top_k, environment=current_env
                    )
                    if retrieved:
                        tools = retrieved
                        if (
                            latest_tool
                            and all(
                                getattr(t, "name", None) != latest_tool.name
                                for t in tools
                            )
                        ):
                            tools = list(tools) + [latest_tool]
            except Exception:
                pass
        print(f"[ORCHESTRATOR] Found {len(tools)} active tools for environment '{current_env}'")
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
        compact = compact[-15:]
        # Append the escape hatch baseline tool (Phase 4)
        compact.append(dict(self._REQUEST_NEW_TOOL_ENTRY))
        return compact

    def _orchestrator_request_prompt(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        solver_recommendation: Optional[str] = None,
        stagnation_count: Optional[int] = None,
        forced_tool_name: Optional[str] = None,
    ) -> str:
        history_text_full = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=1200,
            preserve_first_user_n=2,
        )
        history_lines = history_text_full.splitlines()
        if history_lines:
            history_lines = history_lines[1:]
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
        payload: dict[str, Any] = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "output_schema": output_schema,
        }
        if stagnation_count is not None:
            payload["SYSTEM STATUS"] = {
                "Stagnation Count": int(stagnation_count),
            }
        if solver_recommendation:
            payload["solver_recommendation"] = solver_recommendation
            payload["recommendation_note"] = (
                "Solver provided a draft response. Use it to decide whether a tool "
                "can validate or strengthen the draft before returning it."
            )
        prompt = json.dumps(payload, ensure_ascii=True, default=str)
        if forced_tool_name:
            prompt += (
                f"\n\n[CRITICAL OVERRIDE]: The Forge just successfully generated a new tool "
                f"named '{forced_tool_name}' specifically to solve your current roadblock. "
                f"You MUST output action='use_tool' and tool_name='{forced_tool_name}' on this exact turn."
            )
        return prompt

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
        stagnation_count: Optional[int] = None,
    ) -> dict[str, Any]:
        _ = last_observation  # reserved for future orchestrator prompt enrichment
        if not self._orchestrator_agent:
            return {"action": "no_tool"}

        obs_text = ""
        try:
            for item in reversed(list(self._history_items(chat_history))):
                if item.role == Role.USER and "Observation:" in (item.content or ""):
                    obs_text = item.content or ""
                    break
        except Exception:
            obs_text = ""
        retrieval_query = "\n".join([query or "", obs_text]).strip()

        # Trigger-based fast path: auto-decide based on observation characteristics
        if observation_triggers:
            try:
                tools = self._orchestrator_compact_existing_tools(
                    query_text=retrieval_query
                )
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
                            "action": "request_new_tool",
                            "reason": f"{trigger_type}_no_tool: {trigger.get('reason', '')}",
                        }
                if trigger_type == "derailment_trigger":
                    severity = trigger.get("severity", "medium")
                    filter_tool = next(
                        (
                            t
                            for t in tools
                            if isinstance(t.get("name"), str)
                            and t["name"].endswith("_generated_tool")
                        ),
                        None,
                    )
                    if severity == "high":
                        # High severity: skip straight to new tool generation
                        return {
                            "action": "request_new_tool",
                            "reason": f"derailment_high: {trigger.get('reason', '')}",
                        }
                    else:
                        # Medium severity: prefer existing tool, else request new
                        if filter_tool:
                            return {
                                "action": "use_tool",
                                "tool_name": filter_tool["name"],
                                "reason": f"derailment_medium: {trigger.get('reason', '')}",
                            }
                        else:
                            return {
                                "action": "request_new_tool",
                                "reason": f"derailment_medium_no_tool: {trigger.get('reason', '')}",
                            }

        forced_tool_name = getattr(self, "_just_generated_tool", None)
        prompt = self._orchestrator_request_prompt(
            query,
            chat_history,
            solver_recommendation=solver_recommendation,
            stagnation_count=stagnation_count,
            forced_tool_name=forced_tool_name,
        )
        if forced_tool_name:
            setattr(self, "_just_generated_tool", None)
        try:
            tools = self._orchestrator_compact_existing_tools(
                query_text=retrieval_query
            )
        except Exception:
            tools = []
        tool_list_text = json.dumps(tools, ensure_ascii=True, default=str)
        original_prompt = getattr(self._orchestrator_agent, "_system_prompt", "") or ""
        base_prompt = (getattr(self, "_orchestrator_system_prompt", "") or original_prompt).strip()
        final_prompt = (
            base_prompt
            + "\n\nThere are environment tools that you must NOT consider. Only use tools in the following list to make your decision. if the list is empty, you must generate a tool:\n"
            + tool_list_text
        ).strip()
        self._write_agent_system_prompt("top_orchestrator", final_prompt)
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
        response = None
        try:
            lm = getattr(self._orchestrator_agent, "_language_model", None)
            cfg = getattr(self._orchestrator_agent, "_inference_config_dict", None) or {}
            if lm is not None:
                response = lm.inference(
                    [orchestration_history],
                    cfg,
                    final_prompt,
                )[0]
            else:
                self._orchestrator_agent._system_prompt = final_prompt
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
        if action not in {"use_tool", "no_tool", "create_tool", "request_new_tool"}:
            action = "no_tool"
        tool_name = payload.get("tool_name")
        if (
            action == "use_tool"
            and tool_name
            and str(tool_name) != "request_new_tool"
            and not str(tool_name).endswith("_generated_tool")
        ):
            tool_name = None
        has_agent_action = False
        try:
            for item in self._history_items(chat_history):
                if item.role != Role.AGENT:
                    continue
                content = (item.content or "").strip()
                if content.startswith("Action:") or content.startswith("Final Answer:"):
                    has_agent_action = True
                    break
        except Exception:
            has_agent_action = False
        if action == "request_new_tool" and not has_agent_action:
            candidates = [
                t
                for t in tools
                if isinstance(t.get("name"), str)
                and t["name"].endswith("_generated_tool")
            ]
            if candidates:
                best = max(
                    candidates,
                    key=lambda t: (
                        float(t.get("reliability_score") or 0.0),
                        int(t.get("success") or 0),
                        -int(t.get("failure") or 0),
                    ),
                )
                action = "use_tool"
                tool_name = best.get("name")
                payload = dict(payload)
                payload["reason"] = "bootstrap_tool_available_use_tool"
        if tools and action != "request_new_tool":
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
            "tool_type": payload.get("tool_type"),
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
        if (
            tool_name
            and str(tool_name) != "request_new_tool"
            and not str(tool_name).endswith("_generated_tool")
        ):
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

        # ── Inject controller-known mandatory keys as defaults ──────────
        # The tool_invoker LLM may not echo back every required key.
        # Inject them here so _validate_tool_invoker_payload sees a
        # complete payload.  Using setdefault preserves any value the
        # LLM explicitly provided.
        if isinstance(payload_dict, dict):
            payload_dict.setdefault("task_text", (query or "").strip())
            payload_dict.setdefault("asked_for", (query or "").strip())
            payload_dict.setdefault("run_id", run_id)
            payload_dict.setdefault("state_dir", state_dir)
            payload_dict.setdefault("trace", [])
            payload_dict.setdefault("actions_spec", actions_spec or self._available_actions_spec())
            payload_dict.setdefault("entities", [])
            payload_dict.setdefault("env_observation", "")
            entities_val = payload_dict.get("entities")
            if not isinstance(entities_val, list) or not any(
                isinstance(item, str) and item.strip() for item in entities_val
            ):
                text = (query or "").strip()
                extracted: list[str] = []
                match = re.search(r"Entities\s*:\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
                if match:
                    raw = match.group(1)
                    parts = re.findall(r"'([^']+)'|\"([^\"]+)\"|([^,]+)", raw)
                    for part in parts:
                        token = next((x for x in part if x and x.strip()), None)
                        if token:
                            extracted.append(token.strip())
                if extracted:
                    payload_dict["entities"] = extracted
        elif payload_dict is None:
            pass  # will be caught by validation below

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
