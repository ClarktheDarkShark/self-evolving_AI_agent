import json
from typing import Any, Mapping, Optional

from src.typings import ChatHistory, ChatHistoryItem, Role


class ControllerOrchestratorMixin:
    def _format_orchestrator_docstring(self, tool) -> str:
        description = (tool.description or "").strip()
        base_doc = (tool.docstring or "").strip()
        input_req = "unknown"
        if tool.input_schema is not None:
            input_req = json.dumps(tool.input_schema, ensure_ascii=True, default=str)
        elif tool.signature:
            input_req = f"See signature: {tool.signature}"
        output = "unknown"
        if tool.signature and "->" in tool.signature:
            output = tool.signature.split("->", 1)[-1].strip()
        does = description or (base_doc.splitlines()[0].strip() if base_doc else tool.name)
        lines = [
            f"Input Requirements: {input_req}",
            f"Does: {does}",
            f"Output: {output}",
        ]
        if base_doc and base_doc not in does:
            lines.append(f"Notes: {base_doc}")
        return "\n".join(lines).strip()

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
                    "tool_type": t.tool_type,
                    "tool_category": getattr(t, "tool_category", None),
                    "signature": t.signature,
                    "docstring": self._format_orchestrator_docstring(t),
                }
            )
        return compact[-50:]

    def _orchestrator_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        existing = []
        try:
            existing = self._orchestrator_compact_existing_tools()
        except Exception:
            existing = []

        recent = self._history_items(chat_history)[-10:]
        history_lines = []
        for i, it in enumerate(recent):
            content = (it.content or "").strip().replace("\n", " ")
            content = self._truncate(content, 320)
            history_lines.append("{}:{}:{}".format(i, it.role.value, content))
        history_text = "\n".join(history_lines)
        cleaned_query = self._truncate((query or ""), 1200)

        last_observation = self._get_last_env_observation(chat_history)

        last_agent = None
        for msg in reversed(self._history_items(chat_history)):
            if msg.role == Role.AGENT and (msg.content or "").strip():
                last_agent = (msg.content or "").strip()
                break

        existing_names = [tool.get("name") for tool in existing if tool.get("name")]
        existing_categories = sorted(
            {
                tool.get("tool_category")
                for tool in existing
                if tool.get("tool_category")
            }
        )
        existing_types = sorted(
            {tool.get("tool_type") for tool in existing if tool.get("tool_type")}
        )

        payload = {
            "environment": self._resolved_environment_label(),
            "task_text": cleaned_query,
            "history": history_text,
            "last_observation": last_observation,
            "candidate_output": last_agent,
            "existing_tools": existing,
            "existing_tool_summary": {
                "count": len(existing),
                "names": existing_names,
                "categories": existing_categories,
                "types": existing_types,
            },
            "structured_task": self._is_structured_task(cleaned_query),
            "trivial_task": self._is_trivial_task(cleaned_query),
            "candidate_output_present": bool(self._get_candidate_output(chat_history, cleaned_query)),
            "decision_policy": (
                "You MUST decide if a tool is needed by looking at task_text + history + last_observation.\n"
                "Choose use_tool when: (a) task has multiple constraints, (b) strict formatting, "
                "(c) any earlier step could be wrong, (d) you must be exact, or (e) the last_observation suggests mismatch.\n"
                "Choose create_tool when no existing tool can help validate/repair/plan for this pattern.\n"
                "Use existing_tool_summary to judge coverage; if tools are insufficient for end-to-end checking, create_tool.\n"
                "If structured_task is true and no tool obviously fits, default to create_tool.\n"
                "Choose no_tool only for trivial single-constraint, low-risk responses."
            ),
            "output_schema": {
                "action": "use_tool|create_tool|no_tool",
                "tool_name": "required if use_tool",
                "tool_args": "optional; if omitted controller will auto-build from schema/history",
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
        self._trace("orchestrator_input", prompt)
        try:
            parsed_prompt = json.loads(prompt)
        except Exception:
            parsed_prompt = None
        self._toolgen_debug_event(
            "orchestrator_input",
            prompt=prompt,
            task_text=(parsed_prompt or {}).get("task_text") if parsed_prompt else None,
            existing_tool_summary=(parsed_prompt or {}).get("existing_tool_summary"),
        )
        orchestration_history = ChatHistory()
        orchestration_history = self._safe_inject(
            orchestration_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response = self._orchestrator_agent._inference(orchestration_history)
        self._trace("orchestrator_result", response.content)
        payload = self._parse_orchestrator_payload(response.content)
        if not isinstance(payload, Mapping):
            self._toolgen_debug_event(
                "orchestrator_parse_failed",
                raw_output=response.content or "",
            )
            return {"action": "no_tool"}
        action = str(payload.get("action") or "no_tool").strip().lower()
        tool_name = payload.get("tool_name")
        tool_args = payload.get("tool_args")
        if action not in {"use_tool", "create_tool", "no_tool"}:
            action = "no_tool"
        self._toolgen_debug_event(
            "orchestrator_result",
            action=action,
            tool_name=str(tool_name).strip() if tool_name else None,
            has_tool_args=tool_args is not None,
            reason=payload.get("reason"),
        )
        if action == "create_tool":
            self._toolgen_debug_event(
                "toolgen_requested",
                reason=payload.get("reason"),
                tool_name=str(tool_name).strip() if tool_name else None,
            )
        return {
            "action": action,
            "tool_name": str(tool_name).strip() if tool_name else None,
            "tool_args": tool_args,
            "reason": payload.get("reason"),
        }
