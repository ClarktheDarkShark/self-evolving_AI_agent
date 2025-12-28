import ast
import json
import re
from typing import Any, Mapping, Optional, Sequence, Iterable
import yaml  # add pyyaml dependency if not already present

from typing_extensions import override

from src.agents.agent import Agent
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.language_models import LanguageModel
from src.typings import (
    AgentContextLimitException,
    AgentOutOfMemoryException,
    AgentUnknownException,
    ChatHistory,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    LanguageModelOutOfMemoryException,
    LanguageModelUnknownException,
    Role,
)

from .tool_registry import ToolMetadata, ToolResult, get_registry


class SelfEvolvingController(Agent):
    _ACTION_PATTERN = re.compile(
        r"<action\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</action>",
        re.MULTILINE,
    )


    def __init__(
        self,
        language_model: LanguageModel,
        tool_registry_path: str,
        max_generated_tools_per_run: int = 3,
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        bootstrap_tools: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: str = (
            "You are a helpful assistant that can emit <action> blocks to either "
            "call previously generated tools or request new ones. Always keep task-"
            "specific output formats (e.g., Action: Operation) intact."
        ),
        # NEW:
        force_tool_generation_if_missing: bool = True,
        tool_match_min_score: float = 0.25,
        include_registry_in_prompt: bool = True,
    ):
        self._max_empty_result_retries = 1
        self._empty_result_retries_used = 0

        self._language_model_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=system_prompt,
            inference_config_dict=inference_config_dict,
        )

        # A SECOND agent whose *only job* is to design tools.
        # A SECOND agent whose *only job* is to design tools.
        self._toolgen_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=(
                "Reasoning: low\n"
                "You are a tool generator. Your job is to output EXACTLY ONE tool creation block.\n"
                "Output format must be a JSON (preferred) or YAML mapping with keys:\n"
                "name, description, signature, code_lines.\n"
                "name must be lowercase_snake_case.\n"
                "signature must be run(task_text: str) -> str\n"
                "code_lines must be a JSON/YAML list of strings, each string is ONE LINE of Python.\n"
                "The joined code must define a function run(task_text: str) -> str and return a string.\n"
                "Do NOT include a single multiline string for code. Use code_lines only.\n"
                "Do NOT include any other text or XML.\n"
                "Code must be <= 100 lines.\n"
            ),
            # CHANGE: toolgen gets a smaller cap + stop sequence
            inference_config_dict={
                **(dict(inference_config_dict) if inference_config_dict else {}),
                # "max_tokens": 800,
                # "stop": ["</action>"],
                "temperature": 0.0,
            },
        )


        self._registry = get_registry(tool_registry_path)
        self._max_generated_tools_per_run = max_generated_tools_per_run
        self._generated_tool_counter = 0

        # NEW:
        self._force_tool_generation_if_missing = force_tool_generation_if_missing
        self._tool_match_min_score = tool_match_min_score
        self._include_registry_in_prompt = include_registry_in_prompt

        self._bootstrap_tools(bootstrap_tools or [])
        self._solver_system_prompt = ("""
                Reasoning: low\n
                You are an agent solving DB_BENCH tasks.

                HARD OUTPUT RULES (follow exactly):
                - Output EXACTLY ONE action per response.
                - Your response MUST start with exactly ONE of these lines:
                    Action: Operation
                    Action: Answer
                - Do NOT output any text before the Action line.

                IF Action: Operation:
                - Output EXACTLY two lines and NOTHING else:
                    Action: Operation
                    ```sql
                    <SQL HERE>
                    ```
                - Use a single SQL statement unless the task explicitly requires multiple.
                - End the SQL with a semicolon.

                IF Action: Answer:
                - Output EXACTLY two lines and NOTHING else:
                    Action: Answer
                    Final Answer: <answer>
                - The answer must be the final result required by the task/environment (e.g., an MD5 string or direct output).
                - The answer can be in SQL, MD5 string, or direct output (if direct output, ensure the formatting is a list of tuples)
                - Ensure the output format is exactly as expected.

                IMPORTANT:
                - Never include explanations, reasoning, or multiple actions in one response.
                - Use the MOST RECENT user message as the source of truth (it contains the table/schema).
                - If the previous user message contains an SQL error, correct the SQL and respond with Action: Operation.
                """
        )


    def _is_empty_result(self, txt: str) -> bool:
        return (txt or "").strip() == "[]"

    def _previous_agent_was_operation(self, chat_history: ChatHistory) -> bool:
        items = self._history_items(chat_history)
        # Find the message right before the last USER message
        for i in range(len(items) - 2, -1, -1):
            if items[i].role == Role.AGENT:
                return (items[i].content or "").startswith("Action: Operation")
        return False

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

        print("[SelfEvolvingController] Unable to read ChatHistory via supported methods; skipping.")
        return []


    def _is_dbbench_observation(self, txt: str) -> bool:
        t = (txt or "").strip()
        if not t:
            return True
        if t.startswith("[") or t.startswith("("):
            return True
        if t[:4].isdigit() and "(" in t and ":" in t:  # e.g., "1146 (42S02): ..."
            return True
        return False

    def _prune_for_current_task(self, chat_history: ChatHistory) -> ChatHistory:
        items = self._history_items(chat_history)
        if not items:
            return chat_history

        # Find the most recent USER "task prompt" (not an observation/result/error dump)
        start_idx = 0
        for i in range(len(items) - 1, -1, -1):
            it = items[i]
            if it.role == Role.USER and not self._is_dbbench_observation(it.content or ""):
                start_idx = i
                break

        # Keep from that task prompt onward
        kept = items[start_idx:]

        from src.typings.session import ChatHistory as CH
        h = CH()
        for it in kept:
            h = self._safe_inject(h, it)
        return h



    def _with_added_item(self, chat_history: ChatHistory, item: ChatHistoryItem) -> ChatHistory:
        items = self._history_items(chat_history)
        items.append(item)
        if hasattr(chat_history, "value"):
            # Rebuild same type if your ChatHistory is a pydantic model/dataclass wrapper
            # If this fails, just return items and adjust LanguageModelAgent to accept a list.
            chat_history.value = items  # type: ignore[attr-defined]
            return chat_history
        return items  # type: ignore[return-value]

    # ---------- registry prompt context ----------
    def _registry_prompt_context(self) -> str:
        # You may need to adapt these method names to your ToolRegistry implementation.
        tools: Iterable[ToolMetadata] = []
        if hasattr(self._registry, "list_tools"):
            tools = self._registry.list_tools()
        elif hasattr(self._registry, "tools"):
            tools = self._registry.tools.values()  # type: ignore[attr-defined]

        tool_lines = []
        for t in tools:
            tool_lines.append(f"- {t.name}: {t.description} (signature: {t.signature})")
        if not tool_lines:
            return "GENERATED TOOL REGISTRY: (empty)"
        return "GENERATED TOOL REGISTRY:\n" + "\n".join(tool_lines)

    # ---------- tool matching ----------
    def _score_tool_match(self, query: str, tool: ToolMetadata) -> float:
        # Dumb-but-effective baseline: token overlap with name+description
        q = set(re.findall(r"[a-zA-Z_]+", query.lower()))
        text = f"{tool.name} {tool.description} {tool.signature}".lower()
        t = set(re.findall(r"[a-zA-Z_]+", text))
        if not q or not t:
            return 0.0
        return len(q & t) / max(1, len(q))

    def _best_tool_for(self, query: str) -> Optional[ToolMetadata]:
        tools: Iterable[ToolMetadata] = []
        if hasattr(self._registry, "list_tools"):
            tools = self._registry.list_tools()
        elif hasattr(self._registry, "tools"):
            tools = self._registry.tools.values()  # type: ignore[attr-defined]

        best: Optional[ToolMetadata] = None
        best_score = 0.0
        for t in tools:
            score = self._score_tool_match(query, t)
            if score > best_score:
                best_score = score
                best = t
        if best and best_score >= self._tool_match_min_score:
            return best
        return None

    # ---------- parsing creation payload robustly ----------
    def _parse_creation_payload(self, payload: str) -> Optional[Mapping[str, Any]]:
        # 1) JSON
        try:
            obj = json.loads(payload)
            if isinstance(obj, Mapping):
                return obj
        except Exception:
            pass

        # 2) YAML (accepts JSON too)
        try:
            obj = yaml.safe_load(payload)
            if isinstance(obj, Mapping):
                return obj
        except Exception:
            pass

        # 3) Python-literal fallback (last resort)
        try:
            obj = ast.literal_eval(payload)
            if isinstance(obj, Mapping):
                return obj
        except Exception:
            pass

        return None


    def _get_last_user_item(self, chat_history: ChatHistory) -> Optional[ChatHistoryItem]:
        items = self._history_items(chat_history)
        for msg in reversed(items):
            if msg.role == Role.USER:
                return msg
        return None

    def _extract_toolgen_payload(self, content: str) -> Optional[Mapping[str, Any]]:
        text = (content or "").strip()
        if not text:
            return None
        if match := self._ACTION_PATTERN.search(text):
            text = match.group("body").strip()
        return self._parse_creation_payload(text)

    def _join_code_lines(self, code_lines: Sequence[Any]) -> Optional[str]:
        normalized_lines: list[str] = []
        for line in code_lines:
            text = str(line)
            if "\\n" in text:
                text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
            if "\n" in text:
                normalized_lines.extend(text.splitlines())
            else:
                normalized_lines.append(text)
        if not normalized_lines:
            return None
        if len(normalized_lines) > 100:
            print("[SelfEvolvingController] Tool code exceeds 100 lines; skipping.")
            return None
        return "\n".join(normalized_lines).rstrip() + "\n"

    def _register_tool_from_payload(
        self, creation_request: Mapping[str, Any], chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            print("[SelfEvolvingController] Reached generated tool limit; skipping.")
            return None
        code_lines = creation_request.get("code_lines", None)
        if not isinstance(code_lines, list) or not code_lines:
            print("[SelfEvolvingController] Missing code_lines for tool creation; skipping.")
            return None
        code = self._join_code_lines(code_lines)
        if not code:
            print("[SelfEvolvingController] Empty tool code after normalization; skipping.")
            return None

        tool_name = str(creation_request.get("name") or "generated_tool")
        description = str(creation_request.get("description") or "")
        signature = str(creation_request.get("signature") or "run(task_text: str) -> str")
        print(f"[SelfEvolvingController] Tool creation request: {tool_name} :: {signature}")
        metadata = self.plan_and_generate_tool(
            tool_name=tool_name,
            description=description,
            signature=signature,
            code=code,
            chat_history=chat_history,
        )
        if metadata:
            self._generated_tool_counter += 1
            print(f"[SelfEvolvingController] Registered tool '{metadata.name}'.")
        return metadata

    def _maybe_generate_tool_for_query(
        self, query: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if not query.strip():
            return None
        if not self._force_tool_generation_if_missing:
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            return None

        from src.typings.session import ChatHistory

        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=query)
        )
        response = self._toolgen_agent._inference(tool_history)
        creation_request = self._extract_toolgen_payload(response.content)
        if not creation_request:
            print("[SelfEvolvingController] No tool creation payload; ignoring.")
            return None
        return self._register_tool_from_payload(creation_request, chat_history)

    def _auto_invoke_tool_for_query(
        self, tool: ToolMetadata, query: str
    ) -> Optional[ToolResult]:
        if not query.strip():
            return None
        return self._registry.invoke_tool(
            tool.name,
            query,
            invocation_context={
                "source": "self_evolving_controller",
                "reason": "auto_invoke_for_query",
                "query_preview": query[:200],
            },
        )

    def _is_valid_sql(self, sql: str) -> bool:
        sql_text = (sql or "").strip()
        if not sql_text:
            return False
        if "Action:" in sql_text or "<action" in sql_text:
            return False
        return sql_text.endswith(";")

    def _format_operation_response(self, sql: str) -> str:
        sql_text = sql.strip()
        return f"Action: Operation\n```sql\n{sql_text}\n```"

    def _format_answer_response(self, result_text: str) -> str:
        return f"Action: Answer\nFinal Answer: {result_text}"

    def _validate_solver_output(self, content: str) -> bool:
        if re.fullmatch(r"Action: Answer\nFinal Answer: [^\n]*\n?", content or ""):
            return True
        match = re.fullmatch(
            r"Action: Operation\n```sql\n([\s\S]+)\n```\n?", content or ""
        )
        if not match:
            return False
        sql = match.group(1).strip()
        return self._is_valid_sql(sql)

    def _solver_inference_with_retry(
        self, chat_history: ChatHistory
    ) -> ChatHistoryItem:
        invalid_notice = (
            "Invalid format. Output again EXACTLY in required format. No other text."
        )
        last_response: Optional[ChatHistoryItem] = None
        for attempt in range(3):
            if attempt == 0:
                system_prompt = self._solver_system_prompt
            else:
                system_prompt = f"{self._solver_system_prompt}\n\n{invalid_notice}"
            last_response = self._infer_with_system_prompt(
                chat_history, system_prompt
            )
            if self._validate_solver_output(last_response.content):
                return last_response
        fallback_content = self._format_answer_response("")
        return ChatHistoryItem(role=Role.AGENT, content=fallback_content)

    def _infer_with_system_prompt(
        self, chat_history: ChatHistory, system_prompt: Optional[str] = None
    ) -> ChatHistoryItem:
        if system_prompt is None:
            return self._language_model_agent._inference(chat_history)
        original_prompt = self._language_model_agent._system_prompt
        if system_prompt == original_prompt:
            return self._language_model_agent._inference(chat_history)
        self._language_model_agent._system_prompt = system_prompt
        try:
            return self._language_model_agent._inference(chat_history)
        finally:
            self._language_model_agent._system_prompt = original_prompt




    def plan_and_generate_tool(
        self,
        *,
        tool_name: str,
        description: str,
        signature: str,
        code: str,
        chat_history: ChatHistory,
    ) -> Optional[ToolMetadata]:
        print(f"[SelfEvolvingController] Persisting tool '{tool_name}'...")
        return self._registry.register_tool(
            name=tool_name,
            code=code,
            signature=signature,
            description=description,
        )

    def _bootstrap_tools(self, bootstrap_tools: Sequence[Mapping[str, Any]]) -> None:
        if not bootstrap_tools:
            return
        print(f"[SelfEvolvingController] Bootstrapping {len(bootstrap_tools)} tool(s).")
        for index, tool in enumerate(bootstrap_tools):
            if not isinstance(tool, Mapping):
                continue
            name = str(tool.get("name") or f"bootstrap_tool_{index}")
            description = str(tool.get("description") or "")
            signature = str(tool.get("signature") or "run(*args, **kwargs)")
            code = str(tool.get("code") or "")
            metadata = self._registry.register_tool(
                name=name,
                code=code,
                signature=signature,
                description=description,
            )
            self._generated_tool_counter += 1
            print(f"[SelfEvolvingController] Bootstrapped '{metadata.name}'.")


    from src.typings import Role
    from src.typings.general import ChatHistoryItem  # or wherever it is
    from src.typings.session import ChatHistory

    def _safe_inject(self, h: ChatHistory, item: ChatHistoryItem) -> ChatHistory:
        """Injects item while respecting ChatHistory rule: roles must alternate."""
        try:
            n = h.get_value_length()
            if n > 0:
                last = h.get_item_deep_copy(n - 1)
                if last.role == item.role:
                    # Insert a minimal opposite-role "continuation marker"
                    # so we can inject the desired item next.
                    filler_role = Role.AGENT if item.role == Role.USER else Role.USER
                    h = h.inject(ChatHistoryItem(role=filler_role, content="")) or h
        except Exception:
            # If we can't read last role for some reason, just try inject and let it fail loudly.
            pass

        return h.inject(item) or h

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        last_user = self._get_last_user_item(chat_history)
        if last_user is None:
            return ChatHistoryItem(role=Role.AGENT, content=self._format_answer_response(""))

        query = last_user.content or ""
        if not self._is_dbbench_observation(query):
            self._empty_result_retries_used = 0

        if self._is_dbbench_observation(query):
            # If result is empty AND it came from our last SQL, try a repair operation first
            if self._is_empty_result(query) and self._previous_agent_was_operation(chat_history):
                if self._empty_result_retries_used < self._max_empty_result_retries:
                    self._empty_result_retries_used += 1

                    # Ask solver to repair SQL (general, not task-specific)
                    repair_hint = (
                        "Previous SQL execution returned an empty result ([]). "
                        "Assume the SQL may be too restrictive or incorrect. "
                        "Re-check the task requirements, joins, grouping, filters, HAVING vs WHERE, "
                        "and LIMIT/OFFSET. Produce a corrected SQL statement."
                    )

                    effective_history = self._prune_for_current_task(chat_history)
                    effective_history = self._safe_inject(
                        effective_history,
                        ChatHistoryItem(role=Role.USER, content=repair_hint),
                    )
                    solver_response = self._solver_inference_with_retry(effective_history)
                    return ChatHistoryItem(role=Role.AGENT, content=solver_response.content)

            # Otherwise, accept the observation as final answer
            return ChatHistoryItem(role=Role.AGENT, content=self._format_answer_response(query))


        tool = self._best_tool_for(query)
        if tool is None:
            tool = self._maybe_generate_tool_for_query(query, chat_history)

        if tool is not None:
            tool_result = self._auto_invoke_tool_for_query(tool, query)
            if (
                tool_result is not None
                and tool_result.success
                and isinstance(tool_result.output, str)
                and self._is_valid_sql(tool_result.output)
            ):
                return ChatHistoryItem(
                    role=Role.AGENT,
                    content=self._format_operation_response(tool_result.output),
                )

        effective_history = self._prune_for_current_task(chat_history)
        try:
            solver_response = self._solver_inference_with_retry(effective_history)
        except LanguageModelContextLimitException as e:
            raise AgentContextLimitException(str(e)) from e
        except LanguageModelOutOfMemoryException as e:
            raise AgentOutOfMemoryException(str(e)) from e
        except LanguageModelUnknownException as e:
            raise AgentUnknownException(str(e)) from e
        return ChatHistoryItem(role=solver_response.role, content=solver_response.content)

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        return self._language_model_agent.get_role_dict()
