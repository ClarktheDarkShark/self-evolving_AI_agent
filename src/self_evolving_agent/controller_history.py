import json
import re
from typing import Any, Mapping, Optional

from src.typings import ChatHistory, ChatHistoryItem, Role

from .tool_registry import ToolResult


class ControllerHistoryMixin:
    # ---------- helpers to deal with ChatHistory shapes ----------
    def _history_items(self, chat_history: ChatHistory) -> list[ChatHistoryItem]:
        # Preferred path: use ChatHistory's explicit API (works in client/proxy mode)
        if hasattr(chat_history, "get_value_length") and hasattr(chat_history, "get_item_deep_copy"):
            n = chat_history.get_value_length()
            return [chat_history.get_item_deep_copy(i) for i in range(n)]

        # Fallbacks for non-proxy / older shapes
        if isinstance(chat_history, (list, tuple)):
            return list(chat_history)

        v = getattr(chat_history, "value", None)
        if isinstance(v, list):
            return list(v)

        return []

    def _clone_history(self, chat_history: ChatHistory) -> ChatHistory:
        cloned = ChatHistory()
        for item in self._history_items(chat_history):
            cloned = self._safe_inject(
                cloned, ChatHistoryItem(role=item.role, content=item.content)
            )
        return cloned

    def _prune_for_current_task(self, chat_history: ChatHistory) -> ChatHistory:
        return chat_history

    def _get_last_env_observation(self, chat_history: ChatHistory) -> str:
        for msg in reversed(self._history_items(chat_history)):
            if msg.role != Role.USER:
                continue
            t = (msg.content or "").strip()
            if not t:
                continue
            if t.startswith("TOOL_RESULT "):
                continue
            if "TOOL RESULTS:" in t:
                t = t.split("\n\nTOOL RESULTS:", 1)[0].strip()
            if "CONTEXT:" in t:
                t = t.split("\n\nCONTEXT:", 1)[0].strip()
            return t
        return ""

    def _toolgen_render_history(
        self,
        chat_history: ChatHistory,
        *,
        max_chars_per_item: Optional[int] = 900,
        preserve_first_user_n: int = 0,
    ) -> str:
        items = self._history_items(chat_history)
        rendered: list[str] = []
        preserved_users = 0
        for idx, item in enumerate(items):
            role = item.role.value
            content = (item.content or "").strip()
            if (
                preserve_first_user_n > 0
                and item.role == Role.USER
                and preserved_users < preserve_first_user_n
            ):
                preserved_users += 1
            elif max_chars_per_item is not None and len(content) > max_chars_per_item:
                content = content[:max_chars_per_item] + "...[truncated]"
            rendered.append(f"{idx}:{role}: {content}")
        return "\n".join(rendered)

    def _toolgen_compact_query(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""

        lines = q.splitlines()[-30:]
        drop_substrings = (
            "### interaction rules",
            "### available actions",
            "### output contract",
            "absolute rules:",
            "do not output any other text",
            "now, i will give you the question",
            "you are an intelligent agent tasked with",
            "final answer format",
            "task completion",
            "input guidelines",
            "output rules",
        )

        kept: list[str] = []
        for line in lines:
            l = line.strip()
            low = l.lower()
            if not l:
                continue
            if any(s in low for s in drop_substrings):
                continue
            kept.append(l)

        text = "\n".join(kept).strip()
        if len(text) > 600:
            text = text[-600:]
        return text

    def _default_payload_dict(self, *, query: str, chat_history: ChatHistory) -> dict[str, Any]:
        candidate_output = self._get_candidate_output(chat_history, query)
        history_text = self._toolgen_render_history(
            chat_history,
            max_chars_per_item=1200,
            preserve_first_user_n=2,
        )
        last_user = self._get_last_user_item(chat_history)
        last_observation = (last_user.content or "").strip() if last_user else ""
        last_observation = self._strip_supplemental_sections(last_observation)

        return {
            "task_text": query,
            "candidate_output": candidate_output,
            "history": history_text,
            "last_observation": last_observation,
            "environment": self._resolved_environment_label(),
        }

    def _get_last_user_item(self, chat_history: ChatHistory) -> Optional[ChatHistoryItem]:
        items = self._history_items(chat_history)
        for msg in reversed(items):
            if msg.role == Role.USER:
                return msg
        return None

    def _get_candidate_output(
        self, chat_history: ChatHistory, task_query: str
    ) -> Optional[str]:
        """
        Candidate output should be the last *agent* draft, not the last user observation.
        This is used by the orchestrator + tool reuse gating.
        """
        items = self._history_items(chat_history)

        for msg in reversed(items):
            if msg.role != Role.AGENT:
                continue
            content = (msg.content or "").strip()
            if not content:
                continue

            if self._contains_internal_tool(content):
                continue

            if content.strip() == "":
                continue

            if task_query and content.strip() == task_query.strip():
                continue

            return content

        return None

    def _format_tool_result(self, name: str, result: ToolResult) -> str:
        payload = {
            "tool_name": name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }
        # Extract next_action recommendation if present in tool output
        if isinstance(result.output, Mapping):
            next_action = result.output.get("next_action")
            if isinstance(next_action, Mapping):
                action_name = next_action.get("name")
                if not isinstance(action_name, str):
                    action_name = next_action.get("action")
                args = next_action.get("args")
                if action_name:
                    payload["recommended_next_action"] = action_name
                    if args is not None:
                        payload["recommended_args"] = args
            # Also include status to help solver understand tool state
            status = result.output.get("status")
            if isinstance(status, str):
                payload["status"] = status
        return "TOOL_RESULT: " + json.dumps(
            payload, ensure_ascii=True, default=str
        )

    def _format_tool_result_payload(
        self, tool_name: str, result: ToolResult, max_output_len: int = 1200
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "success": result.success,
        }
        if result.error:
            payload["error"] = result.error
        if result.output is not None:
            if isinstance(result.output, str):
                if len(result.output) <= max_output_len:
                    payload["output"] = result.output
                else:
                    payload["output_preview"] = result.output[: max_output_len - 3] + "..."
                    payload["output_truncated"] = True
            else:
                raw = repr(result.output)
                if len(raw) <= max_output_len:
                    payload["output"] = result.output
                else:
                    payload["output_preview"] = raw[: max_output_len - 3] + "..."
                    payload["output_truncated"] = True
        return payload

    def _append_tool_result_to_last_task(
        self, chat_history: ChatHistory, tool_name: str, result: ToolResult
    ) -> Optional[str]:
        items = self._history_items(chat_history)
        tool_text = self._format_tool_result(tool_name, result)
        for idx in range(len(items) - 1, -1, -1):
            if items[idx].role != Role.USER:
                continue
            current = items[idx].content or ""
            updated = (
                current
                + "\n\nTOOL RESULTS:\n"
                + f"{tool_text}"
            )
            try:
                if hasattr(chat_history, "set"):
                    chat_history.set(
                        idx, ChatHistoryItem(role=Role.USER, content=updated)
                    )
                else:
                    items[idx] = ChatHistoryItem(role=Role.USER, content=updated)
            except Exception:
                return None
            return updated
        return None

    def _inject_tool_result_message(
        self, chat_history: ChatHistory, tool_name: str, result: ToolResult
    ) -> ChatHistory:
        payload = self._format_tool_result_payload(tool_name, result)
        content = "TOOL_RESULT " + json.dumps(payload, ensure_ascii=True, default=str)
        return self._safe_inject(
            chat_history, ChatHistoryItem(role=Role.AGENT, content=content)
        )

    def _infer_task_intent(self, task_query: str) -> dict[str, str]:
        q = (task_query or "").lower()
        if any(token in q for token in ("insert", "update", "delete", "add ", "create ", "remove ", "modify ")):
            return {"intent": "modify", "reason": "modification_keyword"}
        if any(token in q for token in ("select", "list", "which", "show", "return", "count", "sum", "average", "avg")):
            return {"intent": "query", "reason": "query_keyword"}
        return {"intent": "unknown", "reason": "no_keyword"}

    def _infer_sql_verb(self, sql: str) -> str:
        if not sql:
            return "unknown"
        token = sql.strip().split(None, 1)[0].lower()
        if token in {"select", "insert", "update", "delete"}:
            return token
        return "other"

    def _looks_like_success_observation(self, observation: str) -> bool:
        return (observation or "").strip() == "[]"

    def _get_last_action_info(self, chat_history: ChatHistory) -> Optional[dict[str, str]]:
        last_agent = None
        for msg in reversed(self._history_items(chat_history)):
            if msg.role == Role.AGENT and (msg.content or "").strip():
                last_agent = msg.content or ""
                break
        if not last_agent:
            return None
        action, payload = self._extract_action_type_payload(last_agent)
        info: dict[str, str] = {}
        if action:
            info["action"] = action
        if payload:
            info["payload"] = payload
        if action == "operation" and payload:
            info["sql_verb"] = self._infer_sql_verb(payload)
        last_user = self._get_last_user_item(chat_history)
        if last_user and last_user.content is not None:
            observation = self._get_last_env_observation(chat_history)
            if observation:
                info["observation"] = observation
        return info or None

    def _extract_action_type_payload(self, content: str) -> tuple[Optional[str], str]:
        text = (content or "").strip()
        match = re.search(
            r"^Action:\s*(Operation|Answer)\b",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not match:
            return None, ""
        action = match.group(1).lower()
        payload = ""
        if action == "operation":
            fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
            if fence_match:
                payload = fence_match.group(1).strip()
            else:
                payload = text[match.end():].strip()
        else:
            answer_match = re.search(r"Final Answer:\s*([\s\S]+)$", text, flags=re.IGNORECASE)
            if answer_match:
                payload = answer_match.group(1).strip()
            else:
                payload = text[match.end():].strip()
        payload = re.sub(r"\s+", " ", payload.strip())
        if action == "operation":
            payload = payload.rstrip(";").strip().lower()
        return action, payload

    def _detect_repeated_env_action(
        self, chat_history: ChatHistory
    ) -> Optional[dict[str, str]]:
        items = self._history_items(chat_history)
        last_user_idx = None
        for idx in range(len(items) - 1, -1, -1):
            if items[idx].role == Role.USER:
                last_user_idx = idx
                break
        if last_user_idx is None:
            return None
        last_agent_idx = None
        for idx in range(last_user_idx - 1, -1, -1):
            if items[idx].role == Role.AGENT:
                last_agent_idx = idx
                break
        if last_agent_idx is None:
            return None
        prev_user_idx = None
        for idx in range(last_agent_idx - 1, -1, -1):
            if items[idx].role == Role.USER:
                prev_user_idx = idx
                break
        if prev_user_idx is None:
            return None
        prev_agent_idx = None
        for idx in range(prev_user_idx - 1, -1, -1):
            if items[idx].role == Role.AGENT:
                prev_agent_idx = idx
                break
        if prev_agent_idx is None:
            return None
        last_action_type, last_payload = self._extract_action_type_payload(
            items[last_agent_idx].content or ""
        )
        prev_action_type, prev_payload = self._extract_action_type_payload(
            items[prev_agent_idx].content or ""
        )
        if last_action_type != "operation" or prev_action_type != "operation":
            return None
        last_observation = (items[last_user_idx].content or "").strip()
        prev_observation = (items[prev_user_idx].content or "").strip()
        if not last_payload or last_payload != prev_payload:
            return None
        if not last_observation or last_observation != prev_observation:
            return None
        return {
            "action": last_action_type,
            "payload": last_payload,
            "observation": last_observation,
        }

    def _build_solver_context_message(
        self,
        *,
        task_query: str,
        tool_result_injected: bool,
        last_action_info: Optional[Mapping[str, str]] = None,
        task_intent: Optional[Mapping[str, str]] = None,
        repeat_info: Optional[Mapping[str, str]] = None,
    ) -> str:
        lines = []
        if repeat_info:
            lines.append(
                "LOOP_GUARD: Detected the same Operation repeated with the same observation. "
                "Do NOT repeat the same action."
            )
            lines.append(f"Last operation: {repeat_info.get('payload', '')}")
            lines.append(f"Last observation: {repeat_info.get('observation', '')}")
        # if tool_result_injected:
        #     lines.append(
        #         "NOTE: TOOL_RESULT messages are supplemental evidence, not new tasks."
        #     )
        if last_action_info:
            action = last_action_info.get("action")
            payload = last_action_info.get("payload")
            observation = last_action_info.get("observation")
            sql_verb = last_action_info.get("sql_verb")
            if action:
                lines.append(f"LAST_ACTION: {action}")
            if sql_verb:
                lines.append(f"LAST_SQL_TYPE: {sql_verb}")
            if payload:
                payload_preview = payload if len(payload) <= 200 else payload[:197] + "..."
                lines.append(f"LAST_ACTION_PAYLOAD: {payload_preview}")
            if observation:
                obs_preview = observation if len(observation) <= 200 else observation[:197] + "..."
                lines.append(f"LAST_OBSERVATION: {obs_preview}")
        if task_intent:
            intent = task_intent.get("intent") or "unknown"
            lines.append(f"TASK_INTENT: {intent}")
            if (
                intent == "modify"
                and last_action_info
                and last_action_info.get("action") == "operation"
                and last_action_info.get("sql_verb") in {"insert", "update", "delete"}
            ):
                observation = last_action_info.get("observation", "")
                if self._looks_like_success_observation(observation):
                    lines.append(
                        "HINT: The modification appears successful; you can respond with Action: Answer."
                    )
        if task_query:
            lines.append(f"ORIGINAL_TASK: {task_query}")
        return "\n".join([ln for ln in lines if ln.strip()]).strip()

    def _inject_solver_context_message(
        self,
        chat_history: ChatHistory,
        *,
        task_query: str,
        tool_result_injected: bool,
        last_action_info: Optional[Mapping[str, str]] = None,
        task_intent: Optional[Mapping[str, str]] = None,
        repeat_info: Optional[Mapping[str, str]] = None,
    ) -> ChatHistory:
        context_msg = self._build_solver_context_message(
            task_query=task_query,
            tool_result_injected=tool_result_injected,
            last_action_info=last_action_info,
            task_intent=task_intent,
            repeat_info=repeat_info,
        )
        if not context_msg:
            return chat_history
        items = self._history_items(chat_history)
        for msg in reversed(items):
            if msg.role == Role.AGENT and "CONTEXT:" in (msg.content or ""):
                return chat_history
        return self._safe_inject(
            chat_history, ChatHistoryItem(role=Role.AGENT, content="CONTEXT:\n" + context_msg)
        )

    def _extract_internal_tool_calls(self, content: str) -> list[tuple[str, Mapping[str, Any]]]:
        """
        Extract ALL <internal_tool ...> blocks from content (even if surrounded by other text).
        Returns list[(tool_name, payload_mapping)].
        If payload can't be parsed, payload includes _parse_error + raw.
        """
        text = (content or "")
        calls: list[tuple[str, Mapping[str, Any]]] = []
        for m in self._INTERNAL_TOOL_PATTERN.finditer(text):
            name = (m.group("name") or "").strip()
            body = (m.group("body") or "").strip()
            payload = self._parse_creation_payload(body)
            if payload is None:
                payload = {"_parse_error": "Unable to parse internal tool payload.", "raw": body}
            calls.append((name, payload))
        return calls

    def _contains_internal_tool(self, content: str) -> bool:
        return "<internal_tool" in (content or "")

    def _safe_inject(self, h: ChatHistory, item: ChatHistoryItem) -> ChatHistory:
        """Injects item while respecting ChatHistory rule: roles must alternate."""
        try:
            n = h.get_value_length()
            if n > 0:
                last = h.get_item_deep_copy(n - 1)
                if last.role == item.role:
                    filler_role = Role.AGENT if item.role == Role.USER else Role.USER
                    h = h.inject(ChatHistoryItem(role=filler_role, content="")) or h
        except Exception:
            pass

        return h.inject(item) or h
