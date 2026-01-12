#controller.py 

import ast
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Iterable
import yaml  # add pyyaml dependency if not already present
from src.typings.config import get_predefined_timestamp_structure

from typing_extensions import override

from src.agents.agent import Agent
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.language_models import LanguageModel
from src.utils.output_paths import get_output_dir_override, prefix_filename
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


USE_PACKAGED_AGENT = False
# Put this near the top of controller.py (or wherever you build prompts)
import textwrap


TOOLGEN_SYSTEM_PROMPT = textwrap.dedent("""
Reasoning: low
You are ToolGen, an internal tool generator. You create HIGH-ROI, COMPOSABLE utilities (not task solvers) that the main agent can reuse across many tasks in the same environment.

OUTPUT FORMAT (HARD)
- Output EXACTLY ONE JSON object. No prose. No markdown. No code fences.
- The JSON object MUST contain ONLY these keys:
  name, description, signature, tool_type, tool_category, input_schema, capabilities, code_lines
- code_lines MUST be a JSON array of strings. Joining them with "\n" MUST produce valid Python source.
- name MUST be lowercase_snake_case.

HARD CONSTRAINTS
- Use ONLY the Python standard library.
- Total code produced by joining code_lines MUST be <= 150 lines.
- Deterministic behavior: no randomness unless explicitly required and documented.

PURPOSE / ROI
- Primary goal: build a generic transformation/validation/planning primitive that can be reused in 20+ distinct tasks in this environment.
- If you cannot honestly meet the 20+ tasks bar, generate a smaller but still reusable primitive (typically a validator/linter/referee, then a formatter, then a parser).

ABSOLUTELY FORBIDDEN
- Do NOT hard-code a single final answer or constant output.
- Do NOT hard-code a single query/script/artifact as the only supported case.
- Do NOT embed environment-specific action strings (e.g., "Action: ...") unless tool_category is formatter AND the tool’s purpose is to output an exact required string format.
- Outputs MUST vary meaningfully for distinct inputs.

FAILURE-MODE TARGETING (REQUIRED)
- Before implementing, identify the 3 most likely failure-mode TYPES the main agent encounters (e.g., constraint omission, structure mismatch, format noncompliance, schema/interface mismatch, operator misuse).
- Your tool MUST deterministically detect at least 2 failure-mode TYPES via explicit checks.
- Your tool MUST return machine-actionable diagnostics with stable keys and lists (e.g., errors: [..], warnings: [..]).
- When safe, your tool SHOULD provide suggested repairs using fixed_* fields.

DECISION POLICY (MUST FOLLOW)
1) Prefer REFEREE tools (validator/linter) over parsers when possible.
2) Create a PARSER only when:
   - field extraction is repeatedly required, AND
   - constraints are frequently missed, AND
   - validation alone cannot reliably recover/verify needed structure.
3) Prefer “check + optionally repair” over “extract only”.

ALLOWED tool_category (MUST be one of)
- parser, normalizer, planner, validator, linter, formatter

ALLOWED tool_type (MUST be one of)
- utility, validator, linter, formatter

INPUT/OUTPUT CONTRACT
- input_schema MUST be a JSON-schema-like object with required properties clearly listed.
- run() MUST validate required inputs and return structured outputs appropriate to tool_category:
  - parser/normalizer/planner: return a dict with stable keys
  - validator/linter: return a dict with at least: valid (bool), errors (list[str]), warnings (list[str]) and optional fixed_* fields
  - formatter: return a single string exactly in the target format, with no extra text

QUALITY REQUIREMENTS (HARD)
- Include a module docstring explaining purpose and composability.
- Include a run() docstring with at least 2 short usage examples.
- Include self_test() with 3+ tests:
  - two distinct “good” cases, one “bad” case
  - assert meaningful invariants
- self_test() MUST return True only if all assertions pass.
""").strip()


def build_solver_tool_rules() -> str:
    return textwrap.dedent("""
    TOOL USE RULES:
    - First consider whether any tool in the TOOLBELT can help with the task.
    - If no available tool is adequate, consider creating a NEW reusable tool.
    - Emit <internal_tool> blocks when you truly need a tool.
    - Internal tool calls are NOT environment actions.
    - Keep the environment-required action format unchanged.

    INTERNAL TOOL CALL FORMAT:
    <internal_tool name="tool_name">{ "args": [...], "kwargs": { ... } }</internal_tool>

    CREATE_TOOL FORMAT (MUST MATCH TOOLGEN CONTRACT):
    <internal_tool name="create_tool">{
      "name": "lowercase_snake_case",
      "description": "short summary",
      "signature": "run(...) -> ...",
      "tool_type": "utility|validator|linter|formatter",
      "tool_category": "parser|normalizer|planner|validator|linter|formatter",
      "input_schema": {"type":"object","properties":{...},"required":[...]},
      "capabilities": ["capability_a", "capability_b"],
      "code_lines": ["line 1", "line 2", "..."]
    }</internal_tool>

    MANDATORY PREFLIGHT (GENERAL):
    - If you are about to emit a structured artifact (code, query, command, config, plan, or any strict-format output),
      you MUST first call an internal validator/linter tool if one exists that can verify:
      (a) intent alignment with the user task, and (b) format/structure correctness.
    - Prefight means: draft the candidate artifact internally, pass it to the validator/linter as a STRING inside the
      <internal_tool> payload (e.g., kwargs={"task_text": "...", "candidate_artifact": "..."}), then apply any fixes
      returned by the tool before emitting the final artifact to the environment.
    - If no suitable validator/linter exists, create one (tool_type=validator or linter) whose primary job is preflight
      validation of the candidate artifact against task_text, returning:
      {valid/can_proceed, errors, warnings, optional fixed_artifact}.
    - Do NOT create another parser unless validation cannot be performed without it.

    NOTES:
    - Output ONLY the <internal_tool> block when calling tools.
    - code_lines must join into valid Python; stdlib only.
    """).strip()



def _dbg_preview(x, n=220):
    s = "" if x is None else str(x)
    s = s.replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")

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
        tool_match_min_score: float = 0.7,
        include_registry_in_prompt: bool = True,
        environment_label: str = "unknown",
        retrieval_top_k: int = 5,
        reuse_top_k: int = 3,
        reuse_similarity_threshold: Optional[float] = None,
        reuse_min_reliability: float = 0.0,
        canonical_tool_naming: bool = True,
        use_packaged_agent: bool = USE_PACKAGED_AGENT,
    ):
        self._tasks_seen = 0
        self._explore_every_n_tasks = 30   # tune: 20–50

        self._use_packaged_agent = use_packaged_agent
        self._packaged_shim = None
        if self._use_packaged_agent:
            from .packaged_shim import PackagedSelfEvolvingShim

            self._packaged_shim = PackagedSelfEvolvingShim(
                language_model=language_model,
                tool_registry_path=tool_registry_path,
                max_generated_tools_per_run=max_generated_tools_per_run,
                inference_config_dict=inference_config_dict,
                bootstrap_tools=bootstrap_tools,
                system_prompt=system_prompt,
                force_tool_generation_if_missing=force_tool_generation_if_missing,
                tool_match_min_score=tool_match_min_score,
                include_registry_in_prompt=include_registry_in_prompt,
                environment_label=environment_label,
                retrieval_top_k=retrieval_top_k,
                reuse_top_k=reuse_top_k,
                reuse_similarity_threshold=reuse_similarity_threshold,
                reuse_min_reliability=reuse_min_reliability,
                canonical_tool_naming=canonical_tool_naming,
            )
            return
        self._language_model_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=system_prompt,
            inference_config_dict=inference_config_dict,
        )

        # A SECOND agent whose *only job* is to design tools.
        self._toolgen_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=(TOOLGEN_SYSTEM_PROMPT),

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
        self._tool_invocation_log_path: Optional[Path] = None
        self._run_task_label: Optional[str] = None
        try:
            output_dir = get_output_dir_override()
            if output_dir:
                self._tool_invocation_log_path = (
                    Path(output_dir) / prefix_filename("tool_invocations.log")
                )
            else:
                run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
                self._tool_invocation_log_path = (
                    Path("outputs") / run_id / "tool_invocations.log"
                )
        except Exception:
            self._tool_invocation_log_path = None
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
        self._toolgen_attempted_queries: set[str] = set()
        if hasattr(self._registry, "set_canonical_naming"):
            self._registry.set_canonical_naming(self._canonical_tool_naming)

        self._bootstrap_tools(bootstrap_tools or [])
        self._solver_system_prompt = system_prompt


    def _infer_tool_archetype(self, query: str) -> str:
        q = (query or "").lower()
        if self._environment_label in {"os_interaction", "os"}:
            if any(token in q for token in ("final answer", "format", "output", "action:", "act:")):
                return "output_format_checker"
            if any(token in q for token in ("path", "file", "directory", "chmod", "mkdir", "rm ")):
                return "path_safety_linter"
            if re.search(r"(?:^|\\s)/\\S+", q):
                return "path_safety_linter"
            if any(token in q for token in ("command", "bash", "shell", "terminal")):
                return "command_validator"
            return "command_validator"
        if self._environment_label in {"db_bench", "mysql"}:
            # DB_Bench pain points
            if "final answer" in q or "[]" in q or "answer" in q:
                return "answer_guard"
            if "sql" in q and ("wrong" in q or "error" in q or "column" in q or "table" in q):
                return "sql_static_check"
            return "final_output_formatter"
        # Generic fallback
        return "general_helper"


    def _toolgen_compact_existing_tools(self) -> list[dict[str, Any]]:
        """
        Minimal inventory for ToolGen: enough to avoid duplicates and align signatures.
        DO NOT include docstrings, input_schema, capabilities, etc. (ToolGen doesn't need them here).
        """
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
                }
            )

        # Keep it bounded (toolgen only needs to know what already exists, not everything forever)
        # Prefer the most recent tools if list_latest_tools() exists; otherwise keep tail.
        return compact[-50:]


    def _toolgen_compact_query(self, query: str) -> str:
        """
        Strip boilerplate and keep only the most relevant tail / error lines.
        This is *toolgen* context, not retrieval context.
        """
        q = (query or "").strip()
        if not q:
            return ""

        # Keep last ~60 lines; toolgen doesn't need full prompt contracts or long histories.
        lines = q.splitlines()[-60:]

        # Drop known boilerplate patterns commonly present in benchmark wrappers.
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

        # Hard bound so ToolGen doesn't get spammed
        if len(text) > 1200:
            text = text[-1200:]
        return text


    def _toolgen_request_prompt(self, archetype: str, query: str) -> str:
        # Minimal tool inventory (avoid duplicates)
        existing = []
        try:
            existing = self._toolgen_compact_existing_tools()
        except Exception:
            existing = []

        archetype_specs = self._get_archetype_specs()
        spec = archetype_specs.get(archetype, archetype_specs["general_helper"])

        # Compact query context (remove benchmark boilerplate / giant logs)
        compact_query = self._toolgen_compact_query(query)

        # ToolGen already has strict JSON-only requirements in TOOLGEN_SYSTEM_PROMPT (system role),
        # so this user prompt should be strictly the minimum spec + context.
        payload = {
            "environment": self._environment_label,
            "archetype": archetype,
            "existing_tools": existing,  # minimal: name/type/category/signature only
            "user_context": compact_query,  # compact: only relevant tail/error lines
            "required": {
                "tool_type": spec["tool_type"],
                "tool_category": spec["tool_category"],
                "signature": spec["signature"],
                "input_schema": spec["input_schema"],
                "capabilities": spec["capabilities"],
                "requirements": spec["requirements"],
            },
            "constraints": {
                "name": "lowercase_snake_case and not in existing_tools[].name",
                "stdlib_only": True,
                "max_lines": 150,
                "must_include": ["module docstring", "run() docstring", "self_test()"],
            },
            "output_contract": "Return ONE JSON object with keys: name, description, signature, tool_type, tool_category, input_schema, capabilities, code_lines",
        }

        return json.dumps(payload, ensure_ascii=True, default=str)


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

    def _clone_history(self, chat_history: ChatHistory) -> ChatHistory:
        cloned = ChatHistory()
        for item in self._history_items(chat_history):
            cloned = self._safe_inject(
                cloned, ChatHistoryItem(role=item.role, content=item.content)
            )
        return cloned


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
        tool_rules = ("\n\n" + build_solver_tool_rules())

        # if self._environment_label == "db_bench":
        #     tool_rules += (
        #         "\nDB_BENCH POLICY:\n"
        #         "- Before emitting 'Action: Answer', prefer calling internal tool 'answer_guard' if available.\n"
        #         "- If your last SQL was SELECT, Final Answer MUST equal the raw DB response exactly.\n"
        #         "- If task is INSERT/UPDATE/DELETE and DB response is [], you may submit Final Answer: [].\n"
        #     )

        sections: list[str] = []
        if self._include_registry_in_prompt:
            registry_context = self._registry_prompt_context()
            if registry_context and registry_context != "TOOL_REGISTRY: (empty)":
                sections.append(registry_context)
        if toolbelt:
            sections.append(toolbelt)
        sections.append(tool_rules.strip())
        return (self._solver_system_prompt or "") + "\n\n" + "\n\n".join(sections)

    def _consider_tool_generation(
        self, query: str, chat_history: ChatHistory
    ) -> Optional[ToolMetadata]:
        if not query.strip():
            return None
        if not self._force_tool_generation_if_missing:
            return None
        normalized = (
            self._normalize_retrieval_query(query)
            if hasattr(self, "_normalize_retrieval_query")
            else query.strip()
        )
        if not normalized or normalized in self._toolgen_attempted_queries:
            return None
        self._toolgen_attempted_queries.add(normalized)

        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        retrieved = retrieve_tools(
            normalized,
            list(tools),
            top_k=1,
            min_reliability=self._reuse_min_reliability,
        )
        if retrieved:
            best = retrieved[0]
            if best.score >= self._reuse_similarity_threshold:
                return best.tool

        return self._maybe_generate_tool_for_query(query, chat_history)

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

    def _format_tool_response_text(self, result: ToolResult) -> str:
        lines = [f"Success: {result.success}"]
        if result.output is not None:
            lines.append(f"Output: {self._registry._preview(result.output)}")
        if result.error:
            lines.append(f"Error: {result.error}")
        return "\n".join(lines)

    def _append_tool_invocation_log(self, text: str) -> None:
        if not self._tool_invocation_log_path:
            return
        try:
            self._tool_invocation_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._tool_invocation_log_path.open("a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            return

    def _get_run_task_label(self) -> str:
        if self._run_task_label is not None:
            return self._run_task_label
        if not self._tool_invocation_log_path:
            self._run_task_label = "Task: unknown"
            return self._run_task_label
        session_path = self._tool_invocation_log_path.parent / prefix_filename(
            "current_session.json"
        )
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
            task_name = payload.get("task_name")
            sample_index = payload.get("sample_index")
            parts = []
            if task_name is not None:
                parts.append(f"Task: {task_name}")
            if sample_index is not None:
                parts.append(f"Sample: {sample_index}")
            self._run_task_label = " | ".join(parts) if parts else "Task: unknown"
        except Exception:
            self._run_task_label = "Task: unknown"
        return self._run_task_label

    def _build_tool_trace(
        self,
        *,
        summary: str,
        tool_name: str,
        args: list[Any],
        kwargs: Mapping[str, Any],
        result: ToolResult,
    ) -> dict[str, Any]:
        return {
            "summary": summary,
            "tool_name": tool_name,
            "args": list(args),
            "kwargs": dict(kwargs),
            "result": result,
            "model_use": None,
        }

    def _flush_tool_traces(
        self, traces: list[dict[str, Any]], final_response: str
    ) -> None:
        if not traces:
            return
        task_label = self._get_run_task_label()
        blocks: list[str] = []
        for trace in traces:
            model_use = trace.get("model_use") or "(no model response captured yet)"
            result = trace.get("result")
            if not isinstance(result, ToolResult):
                continue
            blocks.append(
                "\n".join(
                    [
                        "=== Tool Invocation ===",
                        task_label,
                        f"Summary: {trace.get('summary')}",
                        f"Tool: {trace.get('tool_name')}",
                        "Content sent to tool:",
                        f"  args: {trace.get('args')}",
                        f"  kwargs: {trace.get('kwargs')}",
                        "Tool response:",
                        self._format_tool_response_text(result),
                        "How the model uses the tool response:",
                        str(model_use),
                        "Model final response after considering tool content:",
                        str(final_response),
                    ]
                )
            )
        if blocks:
            self._append_tool_invocation_log("\n\n".join(blocks) + "\n")

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
                        tool_category=spec.tool_category,   # <-- ADD (after you update register_tool)
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
    ) -> tuple[ToolResult, list[Any], dict[str, Any], str]:
        # Parse error from <internal_tool> payload extraction
        if "_parse_error" in payload:
            return (
                ToolResult.failure(str(payload.get("_parse_error"))),
                [],
                {},
                tool_name,
            )

        # CREATE_TOOL: route through the same validation/registration pipeline
        # so tool_category + input_schema gates always apply.
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

        # Resolve canonical name (if your registry supports it)
        resolved_name = (
            self._registry.resolve_name(tool_name)
            if hasattr(self._registry, "resolve_name")
            else None
        )

        # If tool isn't found, optionally auto-generate a tool for the current query
        if resolved_name is None and not self._registry.has_tool(tool_name):
            last_user = self._get_last_user_item(chat_history)
            query = last_user.content if last_user else ""
            if query and self._force_tool_generation_if_missing:
                created = self._maybe_generate_tool_for_query(query, chat_history)
                if created:
                    resolved_name = created.name

        # Still not found -> fail
        if resolved_name is None and not self._registry.has_tool(tool_name):
            return (
                ToolResult.failure(
                    f"Tool '{tool_name}' not found. Use create_tool or answer directly."
                ),
                [],
                {},
                tool_name,
            )

        # Extract args/kwargs
        tool_args = payload.get("args", [])
        tool_kwargs = payload.get("kwargs", {})

        # If caller passed a single dict as args[0], treat it as kwargs.
        if len(tool_args) == 1 and isinstance(tool_args[0], dict) and not tool_kwargs:
            tool_kwargs = dict(tool_args[0])
            tool_args = []

        # Defensive normalization
        if not isinstance(tool_args, list):
            tool_args = []
        if not isinstance(tool_kwargs, dict):
            tool_kwargs = {}

        # Default behavior: if no args were given, pass last user message content
        if not tool_args and not tool_kwargs:
            last_user = self._get_last_user_item(chat_history)
            if last_user and last_user.content:
                tool_args = [last_user.content]

        # Invoke tool
        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            resolved_name or tool_name,
            *tool_args,
            invocation_context={"environment": self._environment_label},
            **tool_kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        return result, tool_args, tool_kwargs, (resolved_name or tool_name)



    def _normalize_retrieval_query(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""

        # Keep the *actual task* lines; drop long boilerplate instructions
        # Heuristic: keep last ~40 lines and remove lines that look like meta-instructions.
        lines = q.splitlines()
        tail = lines[-40:]  # keep the most recent / relevant
        drop_prefixes = (
            "i will ask you a question",
            "you have to explain",
            "after thinking",
            "to do operation",
            "you must put sql",
            "every time you",
            "if the sql",
            "if you have obtain",
            "we note that",
            "once you commit",
            "now, i will give you",
        )

        kept = []
        for line in tail:
            l = line.strip().lower()
            if any(l.startswith(p) for p in drop_prefixes):
                continue
            kept.append(line.strip())

        # limit length to avoid matching on huge schemas
        compact = "\n".join([x for x in kept if x])[:1200]
        return compact

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

        tool_category = creation_request.get("tool_category")
        allowed_categories = {"parser", "normalizer", "planner", "validator", "linter", "formatter"}
        if tool_category not in allowed_categories:
            print(f"[SelfEvolvingController] Missing/invalid tool_category={tool_category}; skipping.")
            return None

        spec = ToolSpec.from_payload(dict(creation_request))
        metadata = self._validate_and_register_tool(spec, chat_history)
        if metadata:
            self._generated_tool_counter += 1
            # print(f"[SelfEvolvingController] Registered tool '{metadata.name}'.")
        return metadata


    def _expected_tool_type_for_archetype(self, archetype: str) -> Optional[str]:
        return {
            "answer_guard": "validator",
            "sql_static_check": "linter",
            "final_output_formatter": "formatter",
            "general_helper": "utility",
        }.get(archetype)

    def _reuse_existing_tool(self, query: str, needed_archetype: Optional[str] = None) -> Optional[ToolMetadata]:
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
                print(
                    "[SelfEvolvingController] Reuse gate skipped: "
                    f"tool_type='{best.tool.tool_type}' != expected='{expected_type}'"
                )
                return None

        if getattr(best.tool, "self_test_passed", True) is False:
            print("[SelfEvolvingController] Reuse gate skipped: self_test failed.")
            return None
        if getattr(best.tool, "reliability_score", 0.0) < 0.4:
            print("[SelfEvolvingController] Reuse gate skipped: low reliability.")
            return None


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
    
    def _get_archetype_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "command_validator": {
                "name_hint": "command_validator",
                "tool_type": "validator",
                "tool_category": "validator",
                "signature": "run(task_text: str, command_text: str) -> dict",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_text": {"type": "string"},
                        "command_text": {"type": "string"},
                    },
                    "required": ["task_text", "command_text"],
                },
                "capabilities": [
                    "command_intent_check",
                    "risky_flag_detection",
                    "command_format_guard",
                ],
                "requirements": [
                    "Validate that command_text aligns with task_text intent (best-effort heuristics).",
                    "Detect risky patterns (e.g., rm -rf, sudo, destructive redirects) and report them.",
                    "Return dict with keys: valid(bool), errors(list[str]), warnings(list[str]), fixed_command(optional str).",
                    "Never raise; on error return valid=False with a reason in errors.",
                    "Use stdlib only; keep logic general across OS tasks.",
                ],
            },
            "path_safety_linter": {
                "name_hint": "path_safety_linter",
                "tool_type": "linter",
                "tool_category": "linter",
                "signature": "run(command_text: str, allowed_roots: list[str] | None = None) -> dict",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command_text": {"type": "string"},
                        "allowed_roots": {"type": ["array", "null"], "items": {"type": "string"}},
                    },
                    "required": ["command_text"],
                },
                "capabilities": [
                    "path_traversal_check",
                    "destructive_path_guard",
                    "root_scope_check",
                ],
                "requirements": [
                    "Detect path traversal patterns (e.g., ../) and absolute root operations.",
                    "Warn on destructive path usage (rm -rf /, mv to /, chmod -R /).",
                    "Return dict with keys: valid(bool), errors(list[str]), warnings(list[str]), safe_command(optional str).",
                    "If allowed_roots is provided, flag paths outside those roots.",
                    "Use stdlib only; keep logic general.",
                ],
            },
            "output_format_checker": {
                "name_hint": "output_format_checker",
                "tool_type": "validator",
                "tool_category": "validator",
                "signature": "run(task_text: str, candidate_output: str) -> dict",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_text": {"type": "string"},
                        "candidate_output": {"type": "string"},
                    },
                    "required": ["task_text", "candidate_output"],
                },
                "capabilities": [
                    "output_parseability_check",
                    "action_block_check",
                    "final_answer_format_check",
                ],
                "requirements": [
                    "Infer expected output format hints from task_text (Action/Act/Final Answer, code fences).",
                    "Validate candidate_output against those hints (best-effort).",
                    "Return dict with keys: can_proceed(bool), errors(list[str]), warnings(list[str]), fixed_output(optional str).",
                    "Never add extra commentary in fixed_output beyond required format.",
                    "Use stdlib only; keep logic general.",
                ],
            },
            "answer_guard": {
                "name_hint": "answer_guard",
                "tool_type": "validator",
                "tool_category": "validator",
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
                "capabilities": [
                    "classify_task_intent",
                    "final_answer_format_guard",
                    "select_result_passthrough",
                ],
                "requirements": [
                    "Detect whether task intent is SELECT vs mutation (INSERT/UPDATE/DELETE) from task_text.",
                    "If last_sql starts with SELECT, final_answer MUST be exactly repr(last_db_response).",
                    "Return a dict with keys: can_submit(bool), final_answer(str), reason(str), intent(str).",
                    "Never raise; on error return can_submit=False with a reason.",
                    "Use stdlib only; keep logic general across many DB tasks.",
                ],
            },
            "sql_static_check": {
                "name_hint": "sql_static_check",
                "tool_type": "linter",
                "tool_category": "linter",
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
                "capabilities": [
                    "single_line_sql_check",
                    "unknown_column_detection",
                    "table_name_sanity_check",
                ],
                "requirements": [
                    "Validate sql is a single line (no '\\n' characters).",
                    "Check referenced columns in simple SELECT/INSERT patterns are a subset of headers (best-effort heuristics).",
                    "Check table_name appears to match expected table (best-effort).",
                    "Return dict with keys: valid(bool), errors(list[str]), warnings(list[str]), fixed_sql(optional str).",
                    "If you propose fixed_sql, keep it semantically close and still single-line.",
                    "Use stdlib only; keep logic general.",
                ],
            },
            "final_output_formatter": {
                "name_hint": "final_output_formatter",
                "tool_type": "formatter",
                "tool_category": "formatter",
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
                "capabilities": [
                    "emit_exact_action_block",
                    "emit_sql_fence",
                    "emit_final_answer_line",
                ],
                "requirements": [
                    "For Operation: output exactly:\nAction: Operation\\n```sql\\n<ONE LINE SQL>\\n```",
                    "For Answer: output exactly:\nAction: Answer\\nFinal Answer: <repr(final_answer)>",
                    "Never add extra text outside the required format.",
                    "If mode=Operation and sql is missing/empty, return an error string explaining missing sql (still no extra formatting).",
                ],
            },
            "general_helper": {
                "name_hint": "general_helper",
                "tool_type": "utility",
                "tool_category": "parser",
                "signature": "run(task_text: str) -> dict",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_text": {"type": "string"}},
                    "required": ["task_text"],
                },
                "capabilities": [
                    "extract_intent",
                    "extract_constraints",
                    "extract_table_and_headers_when_present",
                ],
                "requirements": [
                    "Parse task_text and extract a small structured dict: intent, table_name (if present), limit/offset (if present), and any mentioned columns (best-effort).",
                    "Return stable keys even when fields are missing (use None).",
                    "Use stdlib only; keep it general across tasks.",
                ],
            },
        }



    @staticmethod
    def _capability_overlap(target: Iterable[str], candidate: Iterable[str]) -> float:
        target_set = {str(x) for x in (target or []) if x}
        candidate_set = {str(x) for x in (candidate or []) if x}
        if not target_set or not candidate_set:
            return 0.0
        return len(target_set & candidate_set) / max(len(target_set), len(candidate_set))

    def _find_existing_singleton(self, query: str, spec: Mapping[str, Any]) -> Optional[ToolMetadata]:
        tools = (
            self._registry.list_latest_tools()
            if hasattr(self._registry, "list_latest_tools")
            else self._registry.list_tools()
        )
        if not tools:
            return None

        spec_type = spec.get("tool_type")
        spec_category = spec.get("tool_category")
        target_caps = spec.get("capabilities") or []

        retrieved = retrieve_tools(
            query,
            list(tools),
            top_k=self._reuse_top_k,
            min_reliability=self._reuse_min_reliability,
        )
        min_score = max(self._reuse_similarity_threshold, self._tool_match_min_score)
        for item in retrieved:
            tool = item.tool
            if spec_type and tool.tool_type and tool.tool_type != spec_type:
                continue
            if spec_category and tool.tool_category and tool.tool_category != spec_category:
                continue
            overlap = self._capability_overlap(target_caps, tool.capabilities or [])
            if overlap >= 0.6 or (overlap >= 0.3 and item.score >= min_score):
                return tool

        # Fallback: reuse by capability overlap even if query similarity is weak.
        for tool in tools:
            if spec_type and tool.tool_type and tool.tool_type != spec_type:
                continue
            if spec_category and tool.tool_category and tool.tool_category != spec_category:
                continue
            overlap = self._capability_overlap(target_caps, tool.capabilities or [])
            if overlap >= 0.8:
                return tool
        return None

    def _maybe_generate_tool_for_query(self, query: str, chat_history: ChatHistory) -> Optional[ToolMetadata]:
        if not query.strip():
            return None

        # Always compute first (used by reuse gating + toolgen)
        archetype = self._infer_tool_archetype(query)
        archetype_specs = self._get_archetype_specs()
        spec = archetype_specs.get(archetype, archetype_specs["general_helper"])
        singleton = self._find_existing_singleton(query, spec)
        if singleton is not None:
            return singleton


        # Optional: exploration budget (if you added it)
        self._tasks_seen = getattr(self, "_tasks_seen", 0) + 1
        explore_every = getattr(self, "_explore_every_n_tasks", 0)  # 0 means disabled
        force_explore = (explore_every and self._tasks_seen % explore_every == 0)

        # Reuse gate (skip if forcing exploration)
        if not force_explore:
            reuse_query = self._normalize_retrieval_query(query) if hasattr(self, "_normalize_retrieval_query") else query
            reuse = self._reuse_existing_tool(reuse_query, needed_archetype=archetype)
            if reuse is not None:
                return reuse

        if not self._force_tool_generation_if_missing:
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            return None

        prompt = self._toolgen_request_prompt(archetype, query)
        tool_history = ChatHistory()
        tool_history = self._safe_inject(tool_history, ChatHistoryItem(role=Role.USER, content=prompt))

        # print("\n=== TOOLGEN PROMPT (tail) ===")
        # print(prompt[-1500:])  # avoid dumping 10k+ chars

        response = self._toolgen_agent._inference(tool_history)

        # print("\n=== TOOLGEN RAW RESPONSE TYPE ===")
        # print(type(response))

        # print("\n=== TOOLGEN RESPONSE.CONTENT (head/tail) ===")
        # content = getattr(response, "content", None)
        # print((content or "")[:1000])
        # print("... [snip] ...")
        # print((content or "")[-1000:])

        creation_request = self._extract_toolgen_payload(response.content)
        if not creation_request:
            return None
        return self._register_tool_from_payload(creation_request, chat_history)


    def _signature_prefers_payload(self, signature: str) -> bool:
        if not signature:
            return False
        match = re.search(r"run\\s*\\(([^)]*)\\)", signature)
        if not match:
            return False
        params = [p.strip() for p in match.group(1).split(",") if p.strip()]
        if len(params) != 1:
            return False
        name = params[0].split(":")[0].split("=")[0].strip().lower()
        return name in {"payload", "data", "inputs", "params", "context", "request", "body"}

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
            "candidate_artifact": candidate_output,
            "candidate_output": candidate_output,
            "output": candidate_output,
            "response": candidate_output,
            "command_text": candidate_output,
            "command": candidate_output,
            "action": candidate_output,
            "environment": self._environment_label,
        }

        if schema is not None:
            properties = schema.get("properties") or {}
            required = schema.get("required") or []
            for key in properties.keys():
                if key in key_map and key_map[key] is not None:
                    values[key] = key_map[key]
            for key in required:
                if key not in values:
                    return None
            if self._signature_prefers_payload(tool.signature):
                payload = dict(values)
                return [payload], {}
            if required:
                return [values[key] for key in required], {}
            if values:
                return [values], {}

        if query:
            return [query], {}
        if candidate_output is not None:
            return [candidate_output], {}
        return None

    def _select_tool_for_query(
        self, query: str, *, categories: Optional[set[str]] = None
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
        if not retrieved:
            return None
        for item in retrieved:
            if item.score < self._tool_match_min_score:
                continue
            tool = item.tool
            if categories and getattr(tool, "tool_category", None) not in categories:
                continue
            return tool
        return None

    def _invoke_tool_for_query(
        self,
        tool: ToolMetadata,
        query: str,
        *,
        candidate_output: Optional[str] = None,
        reason: str = "auto_invoke",
    ) -> Optional[tuple[ToolResult, list[Any], dict[str, Any]]]:
        payload = self._build_tool_invocation(
            tool, query=query, candidate_output=candidate_output
        )
        if payload is None:
            return None
        args, kwargs = payload
        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            tool.name,
            *args,
            invocation_context={
                "source": "self_evolving_controller",
                "reason": reason,
                "query_preview": query[:200],
            },
            **kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        return result, args, kwargs

    @staticmethod
    def _should_retry_after_validation(result: ToolResult) -> bool:
        if not result.success:
            return False
        output = result.output
        if isinstance(output, Mapping):
            if output.get("valid") is False or output.get("can_proceed") is False:
                return True
            errors = output.get("errors")
            if isinstance(errors, list) and errors:
                return True
        return False

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


    # def _is_parseable_env_output(self, text: str) -> bool:
    #     t = (text or "").strip()
    #     if not t:
    #         return False
    #     # if self._parse_internal_tool_call(t):
    #     #     return True
    #     if t.startswith("Action:"):
    #         return True
    #     if t.startswith("Final Answer:"):
    #         return True
    #     return False

    def _is_wrapper_parse_error_prompt(self, text: str) -> bool:
        if not text:
            return False
        return "Your last response was not parseable" in text and "Bad output was:" in text

    def _extract_bad_output(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"Bad output was:\s*([\\s\\S]+)$", text.strip())
        if not match:
            return None
        raw = match.group(1).strip()
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            raw = raw[1:-1]
        try:
            return bytes(raw, "utf-8").decode("unicode_escape")
        except Exception:
            return raw


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

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        """
        Controller inference:
        - Returns the solver model's output EXACTLY as-is (no env-format parsing/repair).
        - BUT: <internal_tool ...> blocks must NEVER be returned to the environment.
        If present, we execute them internally and loop until the solver returns
        a response with no internal_tool blocks.

        Key fixes:
        1) Pin ORIGINAL_TASK once so tool results don't become the "query" driver.
        2) Dedupe internal tool calls so the model can't re-run the same call forever.
        3) Lightly anchor ORIGINAL_TASK into the system prompt each step.
        """
        import json

        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.inference(chat_history)

        working_history = self._clone_history(self._prune_for_current_task(chat_history))

        # Pin the original task text ONCE (before tool-result user messages start dominating)
        orig_last_user = self._get_last_user_item(working_history)
        original_query = orig_last_user.content if orig_last_user else ""

        auto_invoked_tools: set[str] = set()
        tool_traces: list[dict[str, Any]] = []

        # Prevent infinite loops of identical internal calls (tool_name + payload)
        seen_internal_calls: set[str] = set()

        for _ in range(self._internal_tool_max_steps):
            last_user = self._get_last_user_item(working_history)
            query = last_user.content if last_user else ""

            # Use ORIGINAL_TASK as the driver for toolbelt/toolgen/tool-selection
            task_query = (original_query or "").strip() or (query or "").strip()

            # ---------------- Optional debug (uncomment if needed) ----------------
            # print(f"[CTRL] loop | task_query_len={len(task_query)} | task_head={_dbg_preview(task_query)}")
            # print(f"[CTRL] last_user_len={len(query)} | last_user_head={_dbg_preview(query)}")
            # print(f"[CTRL] working_history_len={working_history.get_value_length()}")
            # ---------------------------------------------------------------------

            # Keep your existing toolgen hook, but drive it by the pinned task
            self._consider_tool_generation(task_query, working_history)

            # Optional: auto-preprocess tools (parser/normalizer/planner), also driven by pinned task
            if task_query:
                preprocess_tool = self._select_tool_for_query(
                    task_query, categories={"parser", "normalizer", "planner"}
                )
                if preprocess_tool and preprocess_tool.name not in auto_invoked_tools:
                    preprocess_payload = self._invoke_tool_for_query(
                        preprocess_tool, task_query, reason="auto_preprocess"
                    )
                    if preprocess_payload is not None:
                        preprocess_result, tool_args, tool_kwargs = preprocess_payload
                        auto_invoked_tools.add(preprocess_tool.name)
                        tool_traces.append(
                            self._build_tool_trace(
                                summary="Auto-invoked tool to extract or normalize the task.",
                                tool_name=preprocess_tool.name,
                                args=tool_args,
                                kwargs=tool_kwargs,
                                result=preprocess_result,
                            )
                        )
                        working_history = self._safe_inject(
                            working_history,
                            ChatHistoryItem(
                                role=Role.USER,
                                content=self._format_tool_result(
                                    preprocess_tool.name, preprocess_result
                                ),
                            ),
                        )

            try:
                # print(f"[CTRL] toolbelt_len={len(toolbelt)}")
                # print(f"[CTRL] system_prompt_len={len(system_prompt)} | head={_dbg_preview(system_prompt)}")
                solver_response = self._solver_inference_with_retry(
                    working_history, system_prompt=self._solver_system_prompt
                )
                # print(f"[CTRL] solver_response_len={len(solver_response.content or '')} | head={_dbg_preview(solver_response.content)}")
            except LanguageModelContextLimitException as e:
                raise AgentContextLimitException(str(e)) from e
            except LanguageModelOutOfMemoryException as e:
                raise AgentOutOfMemoryException(str(e)) from e
            except LanguageModelUnknownException as e:
                raise AgentUnknownException(str(e)) from e

            content = getattr(solver_response, "content", "") or ""

            # ---------------- HARD GATE: keep internal tools internal ----------------
            if self._contains_internal_tool(content):
                calls = self._extract_internal_tool_calls(content)

                # Record the raw model output internally (unchanged), but NEVER return it.
                working_history = self._safe_inject(working_history, solver_response)

                # If it contains "<internal_tool" but we couldn't parse any blocks,
                # ask the model to emit ONLY a valid internal_tool block next.
                if not calls:
                    working_history = self._safe_inject(
                        working_history,
                        ChatHistoryItem(
                            role=Role.USER,
                            content=(
                                'If you are calling an internal tool, output ONLY a valid '
                                '<internal_tool name="...">{...}</internal_tool> block and nothing else.'
                            ),
                        ),
                    )
                    continue

                # Execute each internal tool call in order, then loop again.
                for tool_name, payload in calls:
                    # Dedupe identical internal tool calls to avoid burning the loop cap
                    if isinstance(payload, (dict, list)):
                        payload_key = json.dumps(payload, sort_keys=True, default=str)
                    else:
                        payload_key = str(payload)
                    call_key = f"{tool_name}::{payload_key}"

                    if call_key in seen_internal_calls:
                        # Don't re-run; tell the model it already has the result and must proceed.
                        working_history = self._safe_inject(
                            working_history,
                            ChatHistoryItem(
                                role=Role.USER,
                                content=(
                                    f"You already called {tool_name} with the same arguments and received the result above. "
                                    "Do NOT call it again. Continue solving ORIGINAL_TASK using the available results. "
                                    "Return a non-internal response next."
                                ),
                            ),
                        )
                        continue

                    seen_internal_calls.add(call_key)

                    tool_result, tool_args, tool_kwargs, resolved_name = self._handle_internal_tool_call(
                        tool_name, payload, working_history
                    )

                    tool_traces.append(
                        self._build_tool_trace(
                            summary="Model invoked an internal tool (kept internal).",
                            tool_name=resolved_name,
                            args=tool_args,
                            kwargs=tool_kwargs,
                            result=tool_result,
                        )
                    )

                    working_history = self._safe_inject(
                        working_history,
                        ChatHistoryItem(
                            role=Role.USER,
                            content=self._format_tool_result(tool_name, tool_result),
                        ),
                    )

                # After tools run, go back to model for a non-internal final response.
                continue

            # ---------------- NO INTERNAL TOOLS: return EXACT output ----------------
            self._flush_tool_traces(tool_traces, content)
            return ChatHistoryItem(role=Role.AGENT, content=content)

        # If we hit the internal loop cap, fail loudly.
        self._flush_tool_traces(tool_traces, "Exceeded internal tool steps.")
        raise AgentUnknownException(
            "Exceeded internal tool steps without producing a non-internal output."
        )




    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.get_role_dict()
        return self._language_model_agent.get_role_dict()
