import ast
import json
import re
from typing import Any, Mapping, Optional, Sequence, Iterable
import yaml  # add pyyaml dependency if not already present
from src.typings.config import get_predefined_timestamp_structure

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
from .tool_spec import ToolSpec
from .tool_validation import validate_tool_code
from .tool_retrieval import retrieve_tools


class SelfEvolvingController(Agent):
    _INTERNAL_TOOL_PATTERN = re.compile(
        r"<internal_tool\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</internal_tool>",
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
        environment_label: str = "unknown",
        retrieval_top_k: int = 5,
        reuse_top_k: int = 3,
        reuse_similarity_threshold: Optional[float] = None,
        reuse_min_reliability: float = 0.0,
        canonical_tool_naming: bool = True,
    ):
        self._language_model_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=system_prompt,
            inference_config_dict=inference_config_dict,
        )

        # A SECOND agent whose *only job* is to design tools.
        self._toolgen_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=(
                "Reasoning: low\n"
                "You are a tool generator that creates reusable tools to help the main agent solve tasks.\n"
                "Output EXACTLY ONE JSON (preferred) or YAML mapping with keys:\n"
                "name, description, signature, code_lines.\n"
                "\n"
                "HARD RULES:\n"
                "- Output ONLY the mapping. No prose, no markdown.\n"
                "- name MUST be lowercase_snake_case.\n"
                "- description MUST be a short human-readable summary.\n"
                "- signature MUST describe the run() entrypoint.\n"
                "- code_lines MUST be a JSON/YAML list of strings (1 line per item).\n"
                "- Joining code_lines with '\\n' MUST produce valid Python.\n"
                "- Tools must be reusable across multiple tasks; do NOT hard-code a single answer.\n"
                "- Avoid external dependencies; use only the standard library.\n"
                "- Include clear MODULE and run() docstrings with usage examples.\n"
                "- You may optionally include input_schema, tool_type, capabilities, examples.\n"
                "- Code must be <= 150 lines.\n"
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
        self._registry.set_run_snapshot(
            get_predefined_timestamp_structure()["TIMESTAMP"]
        )
        self._max_generated_tools_per_run = max_generated_tools_per_run
        self._generated_tool_counter = 0

        # NEW:
        self._force_tool_generation_if_missing = force_tool_generation_if_missing
        self._tool_match_min_score = tool_match_min_score
        self._include_registry_in_prompt = include_registry_in_prompt
        self._retrieval_top_k = retrieval_top_k
        self._reuse_top_k = reuse_top_k
        self._reuse_similarity_threshold = (
            tool_match_min_score
            if reuse_similarity_threshold is None
            else reuse_similarity_threshold
        )
        self._reuse_min_reliability = reuse_min_reliability
        self._canonical_tool_naming = canonical_tool_naming
        self._min_reliability = 0.2
        self._internal_tool_max_steps = 3
        self._tool_creation_attempts = 0
        self._tool_creation_successes = 0
        self._tool_invocation_attempts = 0
        self._tool_invocation_successes = 0
        self._environment_label = environment_label
        if hasattr(self._registry, "set_canonical_naming"):
            self._registry.set_canonical_naming(self._canonical_tool_naming)

        self._bootstrap_tools(bootstrap_tools or [])
        self._solver_system_prompt = system_prompt



    def _infer_tool_archetype(self, query: str) -> str:
        q = (query or "").lower()
        if self._environment_label in {"db_bench", "mysql"}:
            # DB_Bench pain points
            if "final answer" in q or "[]" in q or "answer" in q:
                return "answer_guard"
            if "sql" in q and ("wrong" in q or "error" in q or "column" in q or "table" in q):
                return "sql_static_check"
            return "final_output_formatter"
        # Generic fallback
        return "general_helper"


    def _toolgen_request_prompt(self, archetype: str, query: str) -> str:
        # Keep this very structured for small models.
        existing = []
        try:
            existing = [t.name for t in self._registry.list_tools()]
        except Exception:
            pass

        # Core IO contracts – these are what make tools reusable + callable.
        archetype_specs = {
            "answer_guard": {
                "name_hint": "answer_guard",
                "tool_type": "validator",
                "signature": "run(task_text: str, last_sql: str | None = None, last_db_response: object | None = None) -> dict",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_text": {"type": "string"},
                        "last_sql": {"type": ["string", "null"]},
                        "last_db_response": {},
                    },
                    "required": ["task_text"],
                },
                "capabilities": ["classify_task_intent", "can_submit_check", "final_answer_exact_format"],
                "requirements": [
                    "Detect mutation vs query tasks from task_text (INSERT/UPDATE/DELETE vs SELECT intent).",
                    "If last_sql begins with SELECT, final_answer MUST be exactly last_db_response (verbatim repr).",
                    "If mutation task and last_db_response is [], allow submission with final_answer = [].",
                    "Return dict with keys: can_submit(bool), final_answer(str), reason(str), intent(str).",
                    "No external deps; stdlib only."
                ],
            },
            "sql_static_check": {
                "name_hint": "sql_static_check",
                "tool_type": "linter",
                "signature": "run(table_name: str, headers: list[str], sql: str) -> dict",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "headers": {"type": "array", "items": {"type": "string"}},
                        "sql": {"type": "string"},
                    },
                    "required": ["table_name", "headers", "sql"],
                },
                "capabilities": ["schema_sanity_checks", "common_sql_mistake_detection"],
                "requirements": [
                    "Check referenced table_name matches.",
                    "Check column names used in INSERT/SELECT are subset of headers.",
                    "Check SQL is single-line (no newline chars).",
                    "Return dict with keys: valid(bool), errors(list[str]), fixed_sql(optional str)."
                ],
            },
            "final_output_formatter": {
                "name_hint": "final_output_formatter",
                "tool_type": "formatter",
                "signature": "run(mode: str, sql: str | None = None, final_answer: object | None = None) -> str",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["Operation", "Answer"]},
                        "sql": {"type": ["string", "null"]},
                        "final_answer": {},
                    },
                    "required": ["mode"],
                },
                "capabilities": ["exact_action_block_formatting"],
                "requirements": [
                    "For Operation: output exactly:\nAction: Operation\\n```sql\\n<ONE LINE SQL>\\n```",
                    "For Answer: output exactly:\nAction: Answer\\nFinal Answer: <repr(final_answer)>",
                    "Never add extra text.",
                ],
            },
            "general_helper": {
                "name_hint": "general_helper",
                "tool_type": "utility",
                "signature": "run(task_text: str) -> str",
                "input_schema": {"type": "object", "properties": {"task_text": {"type": "string"}}, "required": ["task_text"]},
                "capabilities": ["general_parsing"],
                "requirements": ["Keep it reusable; stdlib only."],
            },
        }

        spec = archetype_specs.get(archetype, archetype_specs["general_helper"])

        # IMPORTANT: toolgen needs a “do not duplicate” instruction.
        return (
            "You are generating ONE reusable internal tool.\n"
            f"ENVIRONMENT: {self._environment_label}\n"
            f"ARCHETYPE: {archetype}\n"
            f"EXISTING_TOOL_NAMES: {json.dumps(existing)}\n"
            "\n"
            "USER_QUERY (context only):\n"
            f"{query}\n"
            "\n"
            "You MUST output a single JSON or YAML mapping ONLY with keys:\n"
            "name, description, signature, code_lines, tool_type, input_schema, capabilities.\n"
            "\n"
            f"REQUIRED tool_type: {spec['tool_type']}\n"
            f"REQUIRED signature: {spec['signature']}\n"
            f"REQUIRED input_schema: {json.dumps(spec['input_schema'], ensure_ascii=True)}\n"
            f"REQUIRED capabilities: {json.dumps(spec['capabilities'], ensure_ascii=True)}\n"
            "\n"
            "HARD CONSTRAINTS:\n"
            "- name must be lowercase_snake_case and NOT in EXISTING_TOOL_NAMES.\n"
            "- code_lines must join into valid Python. stdlib only.\n"
            "- Include module docstring + run() docstring.\n"
            "- Include a small self_test() function that returns True/False (no I/O).\n"
            "- Keep code <= 150 lines.\n"
            "\n"
            "FUNCTIONAL REQUIREMENTS:\n"
            + "\n".join(f"- {r}" for r in spec["requirements"])
        )

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


    def _prune_for_current_task(self, chat_history: ChatHistory) -> ChatHistory:
        return chat_history



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
        tools: Iterable[ToolMetadata] = []
        if hasattr(self._registry, "list_latest_tools"):
            tools = self._registry.list_latest_tools()
        else:
            tools = self._registry.list_tools()

        blocks = []
        for t in tools:
            usage = (t.docstring or "").strip()
            block = (
                f"TOOL: {t.name}\n"
                f"Signature: {t.signature}\n"
                f"Description: {t.description}\n"
                f"Reliability: {t.reliability_score:.2f}\n"
            )
            if usage:
                block += f"Docstring:\n{usage}\n"
            if t.tool_type:
                block += f"Tool Type: {t.tool_type}\n"
            if t.input_schema is not None:
                block += f"Input Schema: {json.dumps(t.input_schema, ensure_ascii=True, default=str)}\n"
            if t.capabilities is not None:
                block += f"Capabilities: {json.dumps(t.capabilities, ensure_ascii=True, default=str)}\n"
            blocks.append(block.strip())

        if not blocks:
            return "TOOL_REGISTRY: (empty)"

        return "TOOL_REGISTRY (how to use each tool):\n\n" + "\n\n---\n\n".join(blocks)


    def _build_toolbelt(self, query: str) -> str:
        tools = []
        if hasattr(self._registry, "list_latest_tools"):
            tools = self._registry.list_latest_tools()
        else:
            tools = self._registry.list_tools()
        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=self._retrieval_top_k,
            min_reliability=self._min_reliability,
        )
        if not retrieved:
            return ""
        blocks = []
        for item in retrieved:
            t = item.tool
            block = (
                f"TOOL: {t.name}\n"
                f"Signature: {t.signature}\n"
                f"Description: {t.description}\n"
                f"Reliability: {t.reliability_score:.2f}\n"
            )
            if t.docstring:
                block += f"Docstring:\n{t.docstring}\n"
            if t.tool_type:
                block += f"Tool Type: {t.tool_type}\n"
            if t.input_schema is not None:
                block += f"Input Schema: {json.dumps(t.input_schema, ensure_ascii=True, default=str)}\n"
            if t.capabilities is not None:
                block += f"Capabilities: {json.dumps(t.capabilities, ensure_ascii=True, default=str)}\n"
            blocks.append(block.strip())
        return "TOOLBELT (retrieved tools):\n\n" + "\n\n---\n\n".join(blocks)

    def _augment_system_prompt(self, toolbelt: str) -> str:
        tool_rules = (
            "\n\nTOOL USE RULES:\n"
            "- Prefer existing tools from TOOLBELT when they help.\n"
            "- Only emit <internal_tool> blocks when you truly need a tool.\n"
            "- Internal tool calls are NOT environment actions.\n"
            "- Keep the environment-required action format unchanged.\n"
            "\nINTERNAL TOOL CALL FORMAT:\n"
            "<internal_tool name=\"tool_name\">{ \"args\": [...], \"kwargs\": { ... } }</internal_tool>\n"
            "<internal_tool name=\"create_tool\">{ \"name\": \"...\", \"description\": \"...\", \"signature\": \"...\", \"code_lines\": [...] }</internal_tool>\n"
        )
        if self._environment_label == "db_bench":
            tool_rules += (
                "\nDB_BENCH POLICY:\n"
                "- Before emitting 'Action: Answer', prefer calling internal tool 'answer_guard' if available.\n"
                "- If your last SQL was SELECT, Final Answer MUST equal the raw DB response exactly.\n"
                "- If task is INSERT/UPDATE/DELETE and DB response is [], you may submit Final Answer: [].\n"
            )

        sections: list[str] = []
        if self._include_registry_in_prompt:
            registry_context = self._registry_prompt_context()
            if registry_context and registry_context != "TOOL_REGISTRY: (empty)":
                sections.append(registry_context)
        if toolbelt:
            sections.append(toolbelt)
        sections.append(tool_rules.strip())
        return (self._solver_system_prompt or "") + "\n\n" + "\n\n".join(sections)

    def _parse_internal_tool_call(
        self, content: str
    ) -> Optional[tuple[str, Mapping[str, Any]]]:
        text = (content or "").strip()
        if not text:
            return None
        match = self._INTERNAL_TOOL_PATTERN.fullmatch(text)
        if not match:
            return None
        name = match.group("name").strip()
        body = match.group("body").strip()
        payload = self._parse_creation_payload(body)
        if payload is None:
            return name, {
                "_parse_error": "Unable to parse internal tool payload.",
                "raw": body,
            }
        return name, payload

    def _format_tool_result(self, name: str, result: ToolResult) -> str:
        payload = {
            "tool_name": name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }
        return "ToolResult: " + json.dumps(payload, ensure_ascii=True, default=str)

    def _validate_and_register_tool(
        self, spec: ToolSpec, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        attempts = 0
        last_error = None
        while attempts <= 2:
            attempts += 1
            self._tool_creation_attempts += 1
            code = self._join_code_lines(spec.code_lines)
            if not code:
                last_error = "empty code"
            else:
                result = validate_tool_code(code)
                if result.success:
                    metadata = self._registry.register_tool(
                        name=spec.name,
                        code=code,
                        signature=spec.signature,
                        description=spec.description,
                        tool_type=spec.tool_type,
                        input_schema=spec.input_schema,
                        capabilities=spec.capabilities,
                    )
                    if metadata:
                        self._registry.record_validation_result(
                            metadata.name, True, self_test_passed=result.self_test_passed
                        )
                        self._tool_creation_successes += 1
                        return metadata
                    last_error = "registration failed"
                else:
                    last_error = result.error
            if attempts > 2:
                break
            repair_spec = self._repair_tool_spec(spec, last_error or "", chat_history)
            if repair_spec is None:
                break
            spec = repair_spec
        return None

    def _repair_tool_spec(
        self, spec: ToolSpec, error: str, chat_history: ChatHistory
    ) -> Optional[ToolSpec]:
        prompt = (
            "The previous tool failed validation.\n"
            f"Error: {error}\n"
            f"Previous spec: {json.dumps(spec.__dict__, ensure_ascii=True, default=str)}\n"
            "Output a corrected tool spec (JSON/YAML mapping only)."
        )
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response = self._toolgen_agent._inference(tool_history)
        payload = self._extract_toolgen_payload(response.content)
        if not isinstance(payload, Mapping):
            return None
        return ToolSpec.from_payload(dict(payload))

    def _handle_internal_tool_call(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        chat_history: ChatHistory,
    ) -> ToolResult:
        if "_parse_error" in payload:
            return ToolResult.failure(str(payload.get("_parse_error")))
        if tool_name == "create_tool":
            spec = ToolSpec.from_payload(dict(payload))
            metadata = self._validate_and_register_tool(spec, chat_history)
            if metadata:
                return ToolResult.success_result(
                    {"created_tool": metadata.name, "description": metadata.description}
                )
            return ToolResult.failure("Tool creation failed validation.")

        resolved_name = (
            self._registry.resolve_name(tool_name) if hasattr(self._registry, "resolve_name") else None
        )
        if resolved_name is None and not self._registry.has_tool(tool_name):
            last_user = self._get_last_user_item(chat_history)
            query = last_user.content if last_user else ""
            if query and self._force_tool_generation_if_missing:
                created = self._maybe_generate_tool_for_query(query, chat_history)
                if created:
                    resolved_name = created.name
        if resolved_name is None and not self._registry.has_tool(tool_name):
            return ToolResult.failure(
                f"Tool '{tool_name}' not found. Use create_tool or answer directly."
            )

        tool_args = payload.get("args", [])
        tool_kwargs = payload.get("kwargs", {})
        # If caller passed a single dict as args[0], treat it as kwargs for typed tools.
        if len(tool_args) == 1 and isinstance(tool_args[0], dict) and not tool_kwargs:
            tool_kwargs = dict(tool_args[0])
            tool_args = []

        if not isinstance(tool_args, list):
            tool_args = []
        if not isinstance(tool_kwargs, dict):
            tool_kwargs = {}
        if not tool_args and not tool_kwargs:
            last_user = self._get_last_user_item(chat_history)
            if last_user and last_user.content:
                tool_args = [last_user.content]
        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            resolved_name or tool_name,
            *tool_args,
            invocation_context={"environment": self._environment_label},
            **tool_kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        return result

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
        return self._parse_creation_payload(text)

    def _parse_tool_use_payload(self, content: str) -> Optional[Mapping[str, Any]]:
        text = (content or "").strip()
        if not text:
            return None
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        return self._parse_creation_payload(text)

    def _get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        for tool in self._registry.list_tools():
            if tool.name == name:
                return tool
        return None

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
        normalized_lines = self._normalize_module_docstring_lines(normalized_lines)
        normalized_lines = self._ensure_tool_header(normalized_lines)
        if not normalized_lines:
            return None
        if len(normalized_lines) > 150:
            print("[SelfEvolvingController] Tool code exceeds 150 lines; skipping.")
            return None
        return "\n".join(normalized_lines).rstrip() + "\n"

    def _normalize_module_docstring_lines(self, lines: Sequence[str]) -> list[str]:
        # Normalize module docstring to a single contiguous triple-quoted block.
        normalized = list(lines)
        doc_quote = None

        def _find_first_code_idx(items: list[str]) -> int:
            markers = ("def ", "class ", "import ", "from ")
            return next(
                (
                    i
                    for i, line in enumerate(items)
                    if line.lstrip().startswith(markers)
                ),
                len(items),
            )

        def _strip_empty_prefix(items: list[str]) -> list[str]:
            idx = 0
            while idx < len(items) and items[idx].strip() == "":
                idx += 1
            return items[idx:]

        normalized = _strip_empty_prefix(normalized)
        if not normalized:
            return normalized

        first_line = normalized[0].strip()
        if first_line not in {'"""', "'''"}:
            return normalized
        doc_quote = first_line

        # Find the first non-empty line after the opening docstring line
        idx = 1
        while idx < len(normalized) and normalized[idx].strip() == "":
            idx += 1

        # If the next non-empty line is the same quote, drop it (empty docstring opener/closer)
        if idx < len(normalized) and normalized[idx].strip() == doc_quote:
            normalized.pop(idx)

        # If there is another standalone triple-quote before the first def, treat it as stray and remove it.
        first_def_idx = _find_first_code_idx(normalized)
        stray_idx = next(
            (
                i
                for i in range(1, first_def_idx)
                if normalized[i].strip() in {'"""', "'''"}
            ),
            None,
        )
        if stray_idx is not None:
            normalized.pop(stray_idx)
            first_def_idx = _find_first_code_idx(normalized)

        # Ensure the module docstring closes immediately before the first def.
        if first_def_idx > 0 and normalized[first_def_idx - 1].strip() != doc_quote:
            normalized.insert(first_def_idx, doc_quote)
        return normalized

    def _ensure_tool_header(self, lines: list[str]) -> list[str]:
        if not lines:
            return lines
        header = "from __future__ import annotations"
        header_present = any(line.strip() == header for line in lines[:5])
        if lines and lines[0].strip() in {'"""', "'''"}:
            closing_idx = next(
                (i for i in range(1, len(lines)) if lines[i].strip() in {'"""', "'''"}),
                None,
            )
            if closing_idx is not None:
                insert_at = closing_idx + 1
                while insert_at < len(lines) and lines[insert_at].strip() == "":
                    insert_at += 1
                if not header_present:
                    lines.insert(insert_at, "")
                    lines.insert(insert_at, header)
                return self._ensure_tool_imports(lines)
        if not header_present:
            lines.insert(0, header)
            lines.insert(1, "")
        return self._ensure_tool_imports(lines)

    def _ensure_tool_imports(self, lines: list[str]) -> list[str]:
        needs_re = any("re." in line for line in lines)
        has_re = any(
            line.strip().startswith("import re")
            or line.strip().startswith("from re import")
            for line in lines
        )
        if not (needs_re and not has_re):
            return lines
        insert_at = 0
        while insert_at < len(lines) and lines[insert_at].strip().startswith(
            "from __future__ import"
        ):
            insert_at += 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        lines.insert(insert_at, "import re")
        return lines

    def _register_tool_from_payload(
        self, creation_request: Mapping[str, Any], chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            print("[SelfEvolvingController] Reached generated tool limit; skipping.")
            return None

        # Enforce schema-bearing tools for reliability
        tool_type = creation_request.get("tool_type")
        input_schema = creation_request.get("input_schema")
        if tool_type in {"validator", "linter", "formatter"} and input_schema is None:
            print("[SelfEvolvingController] Missing input_schema for typed tool; skipping.")
            return None

        spec = ToolSpec.from_payload(dict(creation_request))
        metadata = self._validate_and_register_tool(spec, chat_history)
        if metadata:
            self._generated_tool_counter += 1
            # print(f"[SelfEvolvingController] Registered tool '{metadata.name}'.")
        return metadata

    def _reuse_existing_tool(self, query: str) -> Optional[ToolMetadata]:
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
        if best.score < self._reuse_similarity_threshold:
            print(
                "[SelfEvolvingController] Reuse gate skipped: "
                f"best_score={best.score:.3f} < threshold={self._reuse_similarity_threshold:.3f}"
            )
            return None
        if "run(" not in (best.tool.signature or ""):
            print(
                "[SelfEvolvingController] Reuse gate skipped: "
                f"tool '{best.tool.name}' missing run() signature."
            )
            return None
        print(
            "[SelfEvolvingController] Reuse gate selected "
            f"tool='{best.tool.name}' score={best.score:.3f}."
        )
        return best.tool


    def _maybe_generate_tool_for_query(
        self, query: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if not query.strip():
            return None
        reuse = self._reuse_existing_tool(query)
        if reuse is not None:
            return reuse
        if not self._force_tool_generation_if_missing:
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            return None

        archetype = self._infer_tool_archetype(query)
        prompt = self._toolgen_request_prompt(archetype, query)

        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response = self._toolgen_agent._inference(tool_history)

        creation_request = self._extract_toolgen_payload(response.content)
        if not creation_request:
            # print("[SelfEvolvingController] No tool creation payload; ignoring.")
            return None
        return self._register_tool_from_payload(creation_request, chat_history)

    def _select_tool_for_query(self, query: str) -> Optional[ToolMetadata]:
        tools = []
        if hasattr(self._registry, "list_latest_tools"):
            tools = self._registry.list_latest_tools()
        else:
            tools = self._registry.list_tools()
        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=1,
            min_reliability=self._min_reliability,
        )
        if not retrieved:
            return None
        return retrieved[0].tool

    def _invoke_tool_for_query(
        self, tool: ToolMetadata, query: str
    ) -> Optional[ToolResult]:
        if not query.strip():
            return None
        return self._registry.invoke_tool(
            tool.name,
            query,
            invocation_context={
                "source": "self_evolving_controller",
                "reason": "model_requested",
                "query_preview": query[:200],
            },
        )

    def _tool_metrics(self) -> dict[str, Any]:
        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        reuse_count = sum(1 for t in tools if (t.usage_count or 0) > 1)
        env_usage: dict[str, int] = {}
        for tool in tools:
            for env, count in (tool.environment_usage or {}).items():
                env_usage[env] = env_usage.get(env, 0) + int(count)
        creation_rate = (
            self._tool_creation_successes / self._tool_creation_attempts
            if self._tool_creation_attempts
            else 0.0
        )
        invocation_rate = (
            self._tool_invocation_successes / self._tool_invocation_attempts
            if self._tool_invocation_attempts
            else 0.0
        )
        return {
            "tool_creation_attempts": self._tool_creation_attempts,
            "tool_creation_successes": self._tool_creation_successes,
            "tool_creation_pass_rate": round(creation_rate, 3),
            "tool_invocation_attempts": self._tool_invocation_attempts,
            "tool_invocation_successes": self._tool_invocation_successes,
            "tool_invocation_success_rate": round(invocation_rate, 3),
            "tool_reuse_count": reuse_count,
            "per_environment_usage": env_usage,
        }

    def _solver_inference_with_retry(
        self, chat_history: ChatHistory, system_prompt: Optional[str] = None
    ) -> ChatHistoryItem:
        return self._infer_with_system_prompt(chat_history, system_prompt)

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
        tool_type: Optional[Any] = None,
        input_schema: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        code: str,
        chat_history: ChatHistory,
    ) -> Optional[ToolMetadata]:
        # print(f"[SelfEvolvingController] Persisting tool '{tool_name}'...")
        return self._registry.register_tool(
            name=tool_name,
            code=code,
            signature=signature,
            description=description,
            tool_type=str(tool_type) if tool_type is not None else None,
            input_schema=input_schema,
            capabilities=capabilities,
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
            signature = str(tool.get("signature") or "run(task_text: str) -> str")
            code = str(tool.get("code") or "")
            tool_type = tool.get("tool_type")
            input_schema = tool.get("input_schema")
            capabilities = tool.get("capabilities")
            metadata = self._registry.register_tool(
                name=name,
                code=code,
                signature=signature,
                description=description,
                tool_type=str(tool_type) if tool_type is not None else None,
                input_schema=input_schema,
                capabilities=capabilities,
            )
            if metadata:
                self._generated_tool_counter += 1
                print(f"[SelfEvolvingController] Bootstrapped '{metadata.name}'.")

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
        working_history = self._prune_for_current_task(chat_history)
        for _ in range(self._internal_tool_max_steps):
            last_user = self._get_last_user_item(working_history)
            query = last_user.content if last_user else ""
            toolbelt = self._build_toolbelt(query or "")
            system_prompt = self._augment_system_prompt(toolbelt)
            try:
                solver_response = self._solver_inference_with_retry(
                    working_history, system_prompt=system_prompt
                )
            except LanguageModelContextLimitException as e:
                raise AgentContextLimitException(str(e)) from e
            except LanguageModelOutOfMemoryException as e:
                raise AgentOutOfMemoryException(str(e)) from e
            except LanguageModelUnknownException as e:
                raise AgentUnknownException(str(e)) from e

            internal = self._parse_internal_tool_call(solver_response.content)
            if internal is None:
                return ChatHistoryItem(
                    role=solver_response.role, content=solver_response.content
                )

            tool_name, payload = internal
            working_history = self._safe_inject(working_history, solver_response)
            tool_result = self._handle_internal_tool_call(
                tool_name, payload, working_history
            )
            working_history = self._safe_inject(
                working_history,
                ChatHistoryItem(
                    role=Role.USER,
                    content=self._format_tool_result(tool_name, tool_result),
                ),
            )

        return ChatHistoryItem(
            role=Role.AGENT,
            content="Action: Answer\nFinal Answer: Unable to complete due to internal tool loop.",
        )

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        return self._language_model_agent.get_role_dict()
