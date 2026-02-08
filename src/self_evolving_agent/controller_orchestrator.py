import json
from typing import Any, Mapping, Optional

from src.typings import ChatHistory, ChatHistoryItem, Role


class ControllerOrchestratorMixin:
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
            compact.append(
                {
                    "name": t.name,
                    "signature": t.signature,
                    "docstring": self._format_orchestrator_docstring(t),
                    "input_schema": t.input_schema,
                    "usage_count": t.usage_count,
                    "success_count": t.success_count,
                    "failure_count": t.failure_count,
                }
            )
        return compact[-50:]

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
        history_text_full = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=1200,
            preserve_first_user_n=2,
        )
        history_lines = history_text_full.splitlines()
        history_text = "\n".join(history_lines)
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
    ) -> str:
        history_text_full = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=1200,
            preserve_first_user_n=2,
        )
        history_lines = history_text_full.splitlines()
        history_text = "\n".join(history_lines)
        cleaned_query = (query or "").strip()

        payload = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "suggestion": suggestion or {},
            "output_schema": {
                "tool_name": "required",
                "tool_args": "object with args/kwargs",
                "reason": "short reason",
            },
        }
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _parse_orchestrator_payload(self, content: str) -> Optional[Mapping[str, Any]]:
        text = (content or "").strip()
        if not text:
            return None

        parsed = self._parse_creation_payload(text)
        if isinstance(parsed, Mapping):
            return parsed

        obj_text = self._extract_first_json_object(text)
        if obj_text:
            parsed = self._parse_creation_payload(obj_text)
            if isinstance(parsed, Mapping):
                return parsed

        return None

    def _orchestrate_decision(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        solver_recommendation: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self._orchestrator_agent:
            return {"action": "no_tool"}
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
        agent = getattr(self, "_tool_invoker_agent", None)
        if agent is None:
            return {"tool_name": None, "tool_args": None, "reason": "missing_tool_invoker"}
        prompt = self._tool_invoker_request_prompt(query, chat_history, suggestion=suggestion)
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
        self._write_agent_system_prompt("tool_invoker", agent._system_prompt)
        self._trace("tool_invoker_input", prompt)
        self._log_flow_event(
            "tool_invoker_input",
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
        self._trace("tool_invoker_result", response.content)
        self._log_flow_event(
            "tool_invoker_output",
            chat_history=chat_history,
            output=response.content or "",
        )
        payload = self._parse_orchestrator_payload(response.content)
        if not isinstance(payload, Mapping):
            return {"tool_name": None, "tool_args": None, "reason": "parse_failed"}
        tool_name = payload.get("tool_name")
        tool_args = payload.get("tool_args")
        if tool_name and not str(tool_name).endswith("_generated_tool"):
            return {"tool_name": None, "tool_args": None, "reason": "invalid_tool_name"}
        return {
            "tool_name": str(tool_name).strip() if tool_name else None,
            "tool_args": tool_args,
            "reason": payload.get("reason"),
        }
