import json
from typing import Any, Mapping, Optional

from src.typings import ChatHistory, ChatHistoryItem, Role


class ControllerOrchestratorMixin:
    def _format_orchestrator_docstring(self, tool) -> str:
        description = (tool.description or "").strip()
        base_doc = (tool.docstring or "").strip()
        summary = description or (base_doc.splitlines()[0].strip() if base_doc else tool.name)
        return summary.strip()

    def _orchestrator_compact_existing_tools(self) -> list[dict[str, Any]]:
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
                    "signature": t.signature,
                    "summary": self._format_orchestrator_docstring(t),
                }
            )
        return compact[-50:]

    def _orchestrator_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        try:
            existing = self._orchestrator_compact_existing_tools()
        except Exception:
            existing = []

        recent = self._history_items(chat_history)[-6:]
        history_lines = []
        for i, it in enumerate(recent):
            content = (it.content or "").strip().replace("\n", " ")
            content = self._truncate(content, 240)
            history_lines.append("{}:{}:{}".format(i, it.role.value, content))
        history_text = "\n".join(history_lines)
        cleaned_query = self._truncate((query or ""), 1200)

        payload = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "existing_tools": existing,
            "output_schema": {
                "action": "use_tool|create_tool|no_tool",
                "tool_name": "required if use_tool",
                "tool_args": "optional",
                "tool_request": {
                    "tool_category": "validator|linter|formatter|parser|normalizer|planner|utility",
                    "capabilities": "list[str]",
                    "signature": "run(payload: dict) -> dict",
                    "input_schema": "payload-style schema object",
                    "description": "short description",
                    "example_payload": "optional payload dict for auto-invoke",
                },
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
        self, query: str, chat_history: ChatHistory
    ) -> dict[str, Any]:
        if not self._orchestrator_agent:
            return {"action": "no_tool"}
        prompt = self._orchestrator_request_prompt(query, chat_history)
        self._write_agent_system_prompt(
            "orchestrator",
            getattr(self._orchestrator_agent, "_system_prompt", "") or "",
        )
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
        response = self._orchestrator_agent._inference(orchestration_history)
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
        tool_name = payload.get("tool_name")
        tool_args = payload.get("tool_args")
        if action not in {"use_tool", "create_tool", "no_tool"}:
            action = "no_tool"
        return {
            "action": action,
            "tool_name": str(tool_name).strip() if tool_name else None,
            "tool_args": tool_args,
            "reason": payload.get("reason"),
        }
