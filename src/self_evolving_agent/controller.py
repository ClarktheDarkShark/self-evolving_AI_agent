#controller.py 

import ast
import datetime
import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Iterable
try:
    import yaml  # add pyyaml dependency if not already present
except Exception:
    yaml = None
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
from .toolgen_debug_logger import ToolgenDebugLogger, toolgen_debug_enabled
from .tool_retrieval import retrieve_tools


USE_PACKAGED_AGENT = False
# Put this near the top of controller.py (or wherever you build prompts)
import textwrap


TOOLGEN_SYSTEM_PROMPT = textwrap.dedent('''
Reasoning: low
You are ToolGen, an internal tool generator. You create HIGH-ROI, COMPOSABLE utilities (not task solvers) that the main agent can reuse across many tasks in the same environment.

OUTPUT FORMAT (HARD)
- Output EXACTLY ONE JSON object. No prose. No markdown. No code fences.
- The JSON object MUST contain ONLY these keys:
  name, description, signature, tool_type, tool_category, input_schema, capabilities, code_lines
- code_lines MUST be a JSON array of strings. Joining them with "\\n" MUST produce valid Python source.
- name MUST be lowercase_snake_case.

HARD CONSTRAINTS
- Use ONLY the Python standard library.
- Total code produced by joining code_lines MUST be <= 90 lines.
- Deterministic behavior: no randomness unless explicitly required and documented.
- run() MUST NEVER raise. Wrap logic in try/except and on exception return a dict with errors=[...]. Validate types BEFORE calling methods (e.g., .lower()).
- If returning valid=True then errors MUST be [] and every fixed_* field MUST exist and be a non-empty string (never None).
- capabilities MUST be a JSON array of strings (list[str]), NOT a single string.
- code_lines are RAW Python source lines. Do NOT manually escape quotes for JSON. Never include backslashes before quotes (no \" or \\"). Any JSON escaping will be handled by the JSON serializer, not by you.
- signature MUST be exactly "run(payload: dict) -> dict" (no other parameters allowed).

ESCAPING RULES (HARD)
- In code_lines, DO NOT output any of these substrings anywhere: \" or \\"
- Use SINGLE QUOTES for all Python strings and dict keys, except the required 3-line module docstring.
- Do not use backslashes for quoting. If you need quotes inside text, switch quote type (use ' outside, " inside) or build strings without escapes.
- Every code_lines entry must be directly pasteable into a .py file as-is.


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
- Your tool MUST return machine-actionable diagnostics with stable keys and lists (e.g., errors: [...], warnings: [...]).
- When safe, your tool SHOULD provide suggested repairs using fixed_* fields.

DECISION POLICY
- Choose the tool category that best reduces repeated failures in this environment.
- Validators/linters are useful for strict formats and safety checks, but are NOT required.
- Parsers/normalizers/planners are allowed whenever they improve reuse or reduce solver complexity.
- Prefer “check + optionally repair” when it naturally fits, but don’t force multi-tool pipelines.

INPUT/OUTPUT CONTRACT (HARD - SINGLE SIGNATURE STYLE)
- signature MUST be exactly: "run(payload: dict) -> dict"
- run() MUST accept exactly ONE argument named payload and read all inputs from it.
- input_schema MUST have:
  - type: "object"
  - required: ["payload"]
  - properties.payload: { "type":"object", "properties": {...}, "required":[...] }
- All tool-specific inputs (table, columns, where, etc.) MUST live under input_schema.properties.payload.properties
- NEVER use the key name "set" anywhere (schema or code). Use "set_values" instead.
- run() MUST validate required inputs and return structured outputs appropriate to tool_category:
  - parser/normalizer/planner/formatter: return a dict with stable keys
  - validator/linter: return a dict with at least: valid (bool), errors (list[str]), warnings (list[str]) and optional fixed_* fields
- Invocation standard: callers will pass the INNER payload dict to run(payload) (do NOT expect a wrapper with a "payload" key).


QUALITY REQUIREMENTS (HARD)
- Include EXACTLY ONE docstring: a SHORT module docstring (1–2 lines).
- The module docstring MUST be emitted as THREE separate code_lines entries (exactly these three lines):
  1) """
  2) one short line of text (no quotes, no \\n)
  3) """
                                        
- ABSOLUTELY FORBIDDEN: any other triple-quoted strings anywhere in the file.
  That means: do NOT write a run() docstring. Do NOT use """ again after line 3.
- For usage, include a comment example instead:
  - Add a single comment line near run(): "# Example: run({'key': 'value'})"
- Include self_test() with 2 tests (good + bad), BUT self_test() must use only normal quotes (' or ") and NEVER triple quotes.
- Do NOT write any string literal that starts with """ except the 3-line module docstring at the very top.

SELF_TEST QUOTE RULE (HARD)
- In self_test(), do not use f-strings that contain nested quotes like result["x"] inside the f-string.
- Use intermediate variables + repr() for error messages.

                                        
FINAL SELF-CHECK (REQUIRED)
- Before outputting JSON, scan your own code_lines mentally:
  - If any line contains a backslash followed by a quote (\" or \\") you MUST rewrite it using single quotes and remove the backslashes.
  - Ensure dict returns look like: {'valid': False, 'errors': ['...'], 'warnings': []}

                                        
''').strip()


ORCHESTRATOR_SYSTEM_PROMPT = textwrap.dedent("""
Reasoning: low
You are the Orchestrator. Your job is to decide whether a TOOL is needed.

You will receive JSON containing:
- task_text
- history (includes prior attempts and TOOL_RESULTs)
- last_observation (the latest environment/tool output)
- candidate_output (latest agent attempt)
- existing_tools (with signatures + docstrings)

OUTPUT FORMAT (HARD)
- Output EXACTLY ONE JSON object. No prose. No markdown.
- Keys: action, tool_name, tool_args, reason
- action MUST be one of: use_tool | create_tool | no_tool

DECISION RUBRIC (GENERAL)
- First, extract the task constraints from task_text (mentally).
- Then, compare constraints against the evidence in history + last_observation.
- If there is any meaningful chance a constraint is missed, a format is wrong, or a result is incomplete:
  => use_tool (validator/linter/checker) if one exists.
- If no existing tool can check/repair this failure pattern:
  => create_tool (a reusable checker/validator/normalizer), NOT a task-specific solver.
- If existing tools are too narrow to verify multi-step constraints or strict formats end-to-end:
  => create_tool.
- Choose no_tool ONLY if the task is trivial AND there is no strict format AND nothing to validate.
- In general (all tasks), prefer create_tool over no_tool when unsure.

TOOL ARGUMENTS
- If action=use_tool:
  - Provide tool_args if you can.
  - If unsure, omit tool_args (null) and the controller will auto-build from the tool schema + history.

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
      "signature": "run(payload: dict) -> dict",
      "tool_type": "utility|validator|linter|formatter",
      "tool_category": "parser|normalizer|planner|validator|linter|formatter",
      "input_schema": {
        "type": "object",
        "required": ["payload"],
        "properties": {
          "payload": {
            "type": "object",
            "required": ["field_a", "field_b"],
            "properties": {
              "field_a": {"type": "string"},
              "field_b": {"type": "array", "items": {"type": "string"}}
            }
          }
        }
      },
      "capabilities": ["capability_a", "capability_b"],
      "code_lines": ["line 1", "line 2", "line 3"]
    }</internal_tool>

    PREFLIGHT (CONDITIONAL):
    - If the task specifies a strict output contract (e.g., “output EXACTLY…”, parseable wrappers, Action/Final Answer formats),
      OR if you previously received a parse/format error from the environment,
      THEN run a validator/linter tool if one exists (or create one).
    - Otherwise, preflight is OPTIONAL. Prefer making forward progress over building a preflight stack.
    - If you do preflight, prefer a SINGLE tool that can both validate and propose a fixed_* output (check + repair).

    NOTES:
    - Output ONLY the <internal_tool> block when calling tools.
    - code_lines must join into valid Python; stdlib only.
    - Never use payload keys named "set"; use "set_values".
    - IMPORTANT: Tools must have ONLY the 3-line module docstring at the very top.
    - Do NOT add a run() docstring. Put usage examples in # comments.

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
        use_orchestrator: bool = True,
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

        solver_cfg = dict(inference_config_dict) if inference_config_dict else {}
        for k in ("tools", "tool_choice", "functions", "function_call"):
            solver_cfg.pop(k, None)

        # Force “no tool calling” in a way that won't break servers that ignore it.
        solver_cfg["tool_choice"] = "none"

        self._language_model_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=system_prompt,
            inference_config_dict=solver_cfg,
        )


        # A SECOND agent whose *only job* is to design tools.
        base_cfg = dict(inference_config_dict) if inference_config_dict else {}

        # Remove anything that could trigger function/tool calling or special structured responses
        for k in ("tools", "tool_choice", "functions", "function_call"):
            base_cfg.pop(k, None)

        base_cfg["tool_choice"] = "none"

        # Strongly prefer pure JSON output (supported by OpenAI-compatible backends; safe to ignore if unsupported)
        base_cfg["response_format"] = {"type": "json_object"}
        base_cfg["toolgen_extract_tool_calls"] = True
        base_cfg["ollama_force_tool_calls"] = False


        self._toolgen_agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=TOOLGEN_SYSTEM_PROMPT,
            inference_config_dict={
                **base_cfg,
                "temperature": 0.0,
            },
        )

        self._use_orchestrator = use_orchestrator
        self._orchestrator_agent: Optional[LanguageModelAgent] = None
        if self._use_orchestrator:
            orchestrator_cfg = dict(inference_config_dict) if inference_config_dict else {}
            for k in ("tools", "tool_choice", "functions", "function_call"):
                orchestrator_cfg.pop(k, None)

            orchestrator_cfg["tool_choice"] = "none"
            orchestrator_cfg["response_format"] = {"type": "json_object"}

            self._orchestrator_agent = LanguageModelAgent(
                language_model=language_model,
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                inference_config_dict={
                    **orchestrator_cfg,
                    "temperature": 0.0,
                },
            )


        self._registry = get_registry(tool_registry_path)
        self._registry.set_run_snapshot(
            get_predefined_timestamp_structure()["TIMESTAMP"]
        )
        self._tool_invocation_log_path: Optional[Path] = None
        self._generated_tools_log_path: Optional[Path] = None
        self._toolgen_debug_logger: Optional[ToolgenDebugLogger] = None
        self._run_task_label: Optional[str] = None
        try:
            output_dir = get_output_dir_override()
            if output_dir:
                self._tool_invocation_log_path = (
                    Path(output_dir) / prefix_filename("tool_invocations.log")
                )
                self._generated_tools_log_path = (
                    Path(output_dir) / prefix_filename("generated_tools.log")
                )
            else:
                run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
                self._tool_invocation_log_path = (
                    Path("outputs") / run_id / "tool_invocations.log"
                )
                self._generated_tools_log_path = (
                    Path("outputs") / run_id / "generated_tools.log"
                )
        except Exception:
            self._tool_invocation_log_path = None
            self._generated_tools_log_path = None
        try:
            debug_enabled = toolgen_debug_enabled()
            if debug_enabled:
                if self._generated_tools_log_path is not None:
                    debug_path = self._generated_tools_log_path.parent / "toolgen_debug.log"
                    run_id = self._generated_tools_log_path.parent.name
                else:
                    run_id = get_predefined_timestamp_structure()["TIMESTAMP"]
                    debug_path = Path("outputs") / "tool_library" / "toolgen_debug.log"
                self._toolgen_debug_logger = ToolgenDebugLogger(
                    debug_path,
                    enabled=True,
                    run_id=run_id,
                    environment=environment_label,
                )
                self._registry.add_event_listener(
                    self._toolgen_debug_logger.log_registry_event
                )
        except Exception:
            self._toolgen_debug_logger = None
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
        self._tool_invoked_in_last_inference = False
        self._environment_label = environment_label
        self._toolgen_attempted_queries: set[str] = set()
        self._toolgen_debug_registered_tools: dict[str, int] = {}
        self._last_toolgen_parse_source: Optional[str] = None
        self._run_task_metadata: Optional[dict[str, Any]] = None
        self._last_solver_output: Optional[str] = None
        self._last_solver_context_key: Optional[str] = None
        self._solver_repeat_count = 0
        if hasattr(self._registry, "set_canonical_naming"):
            self._registry.set_canonical_naming(self._canonical_tool_naming)

        self._bootstrap_tools(bootstrap_tools or [])
        self._solver_system_prompt = system_prompt


    # def _infer_tool_archetype(self, query: str) -> str:
    #     q = (query or "").lower()
    #     if self._environment_label in {"os_interaction", "os"}:
    #         if any(token in q for token in ("final answer", "format", "output", "action:", "act:")):
    #             return "output_format_checker"
    #         if any(token in q for token in ("path", "file", "directory", "chmod", "mkdir", "rm ")):
    #             return "path_safety_linter"
    #         if re.search(r"(?:^|\\s)/\\S+", q):
    #             return "path_safety_linter"
    #         if any(token in q for token in ("command", "bash", "shell", "terminal")):
    #             return "command_validator"
    #         return "command_validator"
    #     if self._environment_label in {"db_bench", "mysql"}:
    #         # DB_Bench pain points
    #         if "final answer" in q or "[]" in q or "answer" in q:
    #             return "answer_guard"
    #         if "sql" in q and ("wrong" in q or "error" in q or "column" in q or "table" in q):
    #             return "sql_static_check"
    #         return "final_output_formatter"
    #     # Generic fallback
    #     return "general_helper"



    def _truncate(self, s: str, n: int) -> str:
        s = (s or "")
        return s if len(s) <= n else (s[:n] + "...[truncated]")

    def _normalize_toolgen_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return ""

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """
        Return the first top-level JSON object substring from text.
        Handles leading prose, code fences, and trailing junk.
        """
        if not text:
            return None

        s = text.strip()

        # Strip ```json fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()

        # Find first '{' and then balance braces
        start = s.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]

        return None

    def _escape_invalid_json_backslashes(self, text: str) -> str:
        if not text:
            return text
        out: list[str] = []
        in_str = False
        esc = False
        i = 0
        while i < len(text):
            ch = text[i]
            if not in_str:
                if ch == '"':
                    in_str = True
                out.append(ch)
                i += 1
                continue

            if esc:
                out.append(ch)
                esc = False
                i += 1
                continue

            if ch == "\\":
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if nxt in ('"', "\\", "/", "b", "f", "n", "r", "t", "u"):
                    out.append(ch)
                    esc = True
                else:
                    out.append("\\\\")
                i += 1
                continue

            if ch == '"':
                in_str = False
            out.append(ch)
            i += 1

        return "".join(out)

    def _escape_control_chars_in_json_strings(self, text: str) -> str:
        if not text:
            return text
        out: list[str] = []
        in_str = False
        esc = False
        for ch in text:
            if not in_str:
                if ch == '"':
                    in_str = True
                out.append(ch)
                continue

            if esc:
                out.append(ch)
                esc = False
                continue

            if ch == "\\":
                out.append(ch)
                esc = True
                continue

            if ch == '"':
                in_str = False
                out.append(ch)
                continue

            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                continue

            out.append(ch)

        return "".join(out)

    def _close_truncated_json(self, text: str) -> Optional[str]:
        if not text:
            return None
        s = text.strip()
        if not s.startswith("{"):
            return None
        stack: list[str] = []
        in_str = False
        esc = False
        out: list[str] = []
        for ch in s:
            out.append(ch)
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
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]"):
                if stack and ch == stack[-1]:
                    stack.pop()
        if esc:
            out.append("\\")
        if in_str:
            out.append('"')
        while stack:
            out.append(stack.pop())
        return "".join(out)

    def _build_payload_from_payload_schema(
        self,
        schema: Mapping[str, Any],
        key_map: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        For ToolGen payload-style schemas:
        {"type":"object","required":["payload"],"properties":{"payload":{"type":"object","properties":{...},"required":[...]}}}
        Build the dict that should be passed into run(payload).
        """
        props = schema.get("properties") or {}
        payload_schema = props.get("payload")
        if not isinstance(payload_schema, Mapping):
            return None

        inner_props = payload_schema.get("properties") or {}
        inner_required = payload_schema.get("required") or []

        payload: dict[str, Any] = {}
        for k in inner_props.keys():
            if k in key_map and key_map[k] is not None:
                payload[k] = key_map[k]

        # required fields must exist in the *inner* payload schema
        for k in inner_required:
            if k not in payload:
                return None

        return payload

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

    def _format_orchestrator_docstring(self, tool: ToolMetadata) -> str:
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


    def _toolgen_compact_query(self, query: str) -> str:
        """
        Strip boilerplate and keep only the most relevant tail / error lines.
        This is *toolgen* context, not retrieval context.
        """
        q = (query or "").strip()
        if not q:
            return ""

        # Keep last N lines; toolgen doesn't need full prompt contracts or long histories.
        lines = q.splitlines()[-30:]

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
        if len(text) > 600:
            text = text[-600:]
        return text


    def _toolgen_render_history(
        self, chat_history: ChatHistory, *, max_chars_per_item: Optional[int] = 900
    ) -> str:
        items = self._history_items(chat_history)
        rendered: list[str] = []
        for idx, item in enumerate(items):
            role = item.role.value
            content = (item.content or "").strip()
            if max_chars_per_item is not None and len(content) > max_chars_per_item:
                content = content[:max_chars_per_item] + "...[truncated]"
            rendered.append(f"{idx}:{role}: {content}")
        return "\n".join(rendered)


    def _toolgen_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        existing = []
        try:
            existing = self._toolgen_compact_existing_tools()
        except Exception:
            existing = []

        compact_query = self._toolgen_compact_query(query)

        # MUCH smaller history for toolgen; it only needs the last few turns
        history_items = self._history_items(chat_history)[-8:]
        mini_lines = []
        for i, it in enumerate(history_items):
            content = (it.content or "").strip().replace("\n", " ")
            content = self._truncate(content, 280)
            mini_lines.append("{}:{}:{}".format(i, it.role.value, content))
        mini_history = "\n".join(mini_lines)

        payload = {
            "existing_tools": existing[-20:],      # was -50
            "user_task": self._truncate(compact_query, 900),
            "history_hint": mini_history,          # was history_full
            "design_goal": (
                "Design a HIGH-ROI reusable tool to reduce repeated failures. "
                "Prefer validators/normalizers/formatters that work across many tasks."
            ),
            "hard_constraints": {
                "stdlib_only": True,
                "max_lines": 90,
                "deterministic": True,
                "self_test_required": True,
                "output_keys": [
                    "name","description","signature","tool_type","tool_category",
                    "input_schema","capabilities","code_lines"
                ],
            },
        }
        return json.dumps(payload, ensure_ascii=True, default=str)


    def _get_last_env_observation(self, chat_history: ChatHistory) -> str:
        for msg in reversed(self._history_items(chat_history)):
            if msg.role != Role.USER:
                continue
            t = (msg.content or "").strip()
            if not t:
                continue
            # skip injected tool results / context blobs
            if t.startswith("TOOL_RESULT "):
                continue
            if "TOOL RESULTS:" in t:
                t = t.split("\n\nTOOL RESULTS:", 1)[0].strip()
            if "CONTEXT:" in t:
                t = t.split("\n\nCONTEXT:", 1)[0].strip()
            return t
        return ""


    def _orchestrator_request_prompt(self, query: str, chat_history: ChatHistory) -> str:
        existing = []
        try:
            existing = self._orchestrator_compact_existing_tools()
        except Exception:
            existing = []

        # DO NOT strip tool results; orchestrator needs evidence to decide tool necessity.
        # Orchestrator only needs recent evidence, not full transcript
        recent = self._history_items(chat_history)[-10:]
        history_lines = []
        for i, it in enumerate(recent):
            content = (it.content or "").strip().replace("\n", " ")
            content = self._truncate(content, 320)
            history_lines.append("{}:{}:{}".format(i, it.role.value, content))
        history_text = "\n".join(history_lines)
        cleaned_query = self._truncate((query or ""), 1200)


        # Add a few “state signals” that are useful in ANY environment
        last_observation = self._get_last_env_observation(chat_history)


        # Try to capture last agent message (candidate artifact)
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
            "candidate_output": last_agent,  # optional: what solver last attempted
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

        # print("[SelfEvolvingController] Unable to read ChatHistory via supported methods; skipping.")
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
        # Only toolgen on structured tasks or when we saw an error / loop / parse failure
        if not self._is_structured_task(query) and self._resolved_environment_label() not in {"db_bench", "mysql"}:
            return None

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

        last_user = self._get_last_user_item(chat_history)
        last_obs = (last_user.content or "") if last_user else ""
        if not self._is_wrapper_parse_error_prompt(last_obs) and not self._detect_repeated_env_action(chat_history):
            # no evidence we failed format-wise or are stuck
            return None


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
        return "TOOL_RESULT (supplemental): " + json.dumps(
            payload, ensure_ascii=True, default=str
        )

    def _mark_tool_invoked(self) -> None:
        self._tool_invoked_in_last_inference = True

    def _trace(self, label: str, content: Any) -> None:
        text = "" if content is None else str(content)
        print(f"[TRACE] {label}:\n{text}")

    def _get_run_task_metadata(self) -> dict[str, Any]:
        if self._run_task_metadata is not None:
            return dict(self._run_task_metadata)
        if not self._generated_tools_log_path:
            return {}
        session_path = self._generated_tools_log_path.parent / prefix_filename(
            "current_session.json"
        )
        if not session_path.exists():
            return {}
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        task_name = payload.get("task_name")
        sample_index = payload.get("sample_index")
        meta: dict[str, Any] = {}
        if task_name is not None:
            meta["task_name"] = task_name
        if sample_index is not None:
            meta["sample_index"] = sample_index
        if meta:
            self._run_task_metadata = meta
        return dict(meta)

    def _resolved_environment_label(self) -> str:
        meta = self._get_run_task_metadata()
        task_name = meta.get("task_name")
        if task_name in {"db_bench", "mysql"}:
            return "db_bench"
        return self._environment_label

    def _is_db_bench_env(self) -> bool:
        return self._resolved_environment_label() in {"db_bench", "mysql"}

    def _is_structured_task(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        score = 0
        if any(token in q for token in ("format", "exact", "exactly", "must", "return", "provide")):
            score += 1
        if any(token in q for token in ("json", "csv", "table", "columns", "schema")):
            score += 1
        if any(token in q for token in ("group by", "having", "order by", "limit", "offset")):
            score += 1
        if any(token in q for token in ("sum", "average", "avg", "count", "total", "aggregate")):
            score += 1
        if re.search(r"(less than|greater than|at least|at most|>=|<=|!=|=)", q):
            score += 1
        if q.count(" and ") + q.count(" or ") >= 1:
            score += 1
        return score >= 2

    def _is_trivial_task(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return True
        if self._is_structured_task(q):
            return False
        words = q.split()
        if len(words) <= 8:
            return True
        if len(q) <= 60 and not re.search(r"\d", q):
            return True
        return False

    # def _get_candidate_output(
    #     self, chat_history: ChatHistory, task_query: str
    # ) -> Optional[str]:
    #     last_user = self._get_last_user_item(chat_history)
    #     if not last_user or not last_user.content:
    #         return None
    #     content = last_user.content.strip()
    #     if task_query and content == task_query.strip():
    #         return None
    #     return last_user.content
    
    def _get_candidate_output(
        self, chat_history: ChatHistory, task_query: str
    ) -> Optional[str]:
        """
        Candidate output should be the last *agent* draft, not the last user observation.
        This is used by the orchestrator + tool reuse gating.
        """
        items = self._history_items(chat_history)

        # Walk backward to find the last meaningful AGENT message.
        for msg in reversed(items):
            if msg.role != Role.AGENT:
                continue
            content = (msg.content or "").strip()
            if not content:
                continue

            # Never treat internal tool calls as candidate output
            if self._contains_internal_tool(content):
                continue

            # Optional: ignore "filler" alternation markers if you used them
            if content.strip() == "":
                continue

            # If the agent output exactly repeats the task text, it isn't a draft
            if task_query and content.strip() == task_query.strip():
                continue

            return content

        return None


    def _preview_for_log(self, obj: Any, max_len: int = 300) -> str:
        try:
            text = repr(obj)
        except Exception:
            text = f"<unreprable {type(obj).__name__}>"
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

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

    def _inject_tool_result_message(
        self, chat_history: ChatHistory, tool_name: str, result: ToolResult
    ) -> ChatHistory:
        appended = self._append_tool_result_to_last_task(
            chat_history, tool_name, result
        )
        if appended:
            return chat_history
        payload = self._format_tool_result_payload(tool_name, result)
        content = "TOOL_RESULT " + json.dumps(payload, ensure_ascii=True, default=str)
        return self._safe_inject(
            chat_history, ChatHistoryItem(role=Role.USER, content=content)
        )

    def _append_generated_tools_log(self, payload: Mapping[str, Any]) -> None:
        if not self._generated_tools_log_path:
            return
        try:
            self._generated_tools_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._generated_tools_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception:
            return

    def _toolgen_debug_event(self, event: str, **fields: Any) -> None:
        logger = self._toolgen_debug_logger
        if not logger or not logger.enabled:
            return
        meta = self._get_run_task_metadata()
        logger.log_event(
            event,
            task_name=meta.get("task_name"),
            sample_index=meta.get("sample_index"),
            environment=self._resolved_environment_label(),
            **fields,
        )

    def _toolgen_model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        model = getattr(self._toolgen_agent, "_language_model", None)
        if model is None:
            return info
        info["model_type"] = type(model).__name__
        for attr in ("model", "model_name", "name", "base_url"):
            if hasattr(model, attr):
                try:
                    info[attr] = getattr(model, attr)
                except Exception:
                    info[attr] = None
        return info

    def _toolgen_fallback_info(self) -> Optional[dict[str, Any]]:
        model = getattr(self._toolgen_agent, "_language_model", None)
        if model is None:
            return None
        getter = getattr(model, "get_last_tool_call_fallback", None)
        if callable(getter):
            try:
                return getter(clear=True)
            except Exception:
                return getter()
        return None


    def _default_payload_dict(self, *, query: str, chat_history: ChatHistory) -> dict[str, Any]:
        candidate_output = self._get_candidate_output(chat_history, query)
        history_text = self._toolgen_render_history(chat_history, max_chars_per_item=300)
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


    def _log_tool_invocation_event(
        self,
        *,
        tool_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        result: ToolResult,
        reason: str,
        args_auto_built: bool = False,
        decision_action: Optional[str] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": "invoke",
            "tool_name": tool_name,
            "args_preview": self._preview_for_log(args),
            "kwargs_preview": self._preview_for_log(kwargs),
            "success": result.success,
            "error": result.error,
            "reason": reason,
            "environment_label": self._resolved_environment_label(),
            "source": "controller",
            "args_auto_built": args_auto_built,
        }
        if decision_action:
            payload["decision_action"] = decision_action
        payload.update(self._get_run_task_metadata())
        self._append_generated_tools_log(payload)

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

    def _invoke_tool_by_payload(
        self,
        tool_name: str,
        tool_args: Any,
        *,
        reason: str,
        args_auto_built: bool = False,
        decision_action: Optional[str] = None,
    ) -> ToolResult:
        resolved_name = (
            self._registry.resolve_name(tool_name)
            if hasattr(self._registry, "resolve_name")
            else None
        )
        resolved_name = resolved_name or tool_name
        tool_meta = self._get_tool_metadata(resolved_name)
        if isinstance(tool_args, Mapping) and "args" not in tool_args and "kwargs" not in tool_args:
            if tool_meta and self._signature_prefers_payload(tool_meta.signature):
                if "payload" in tool_args:
                    tool_args = {"args": [tool_args.get("payload")]}
                else:
                    tool_args = {"args": [dict(tool_args)]}
            elif tool_meta and isinstance(tool_meta.input_schema, Mapping):
                properties = tool_meta.input_schema.get("properties") or {}
                required = tool_meta.input_schema.get("required") or []
                if "payload" in properties and "payload" in required and "payload" not in tool_args:
                    tool_args = {"kwargs": {"payload": dict(tool_args)}}
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        if isinstance(tool_args, Mapping):
            if "args" in tool_args or "kwargs" in tool_args:
                args = list(tool_args.get("args") or [])
                kwargs = dict(tool_args.get("kwargs") or {})
            else:
                kwargs = dict(tool_args)
        elif isinstance(tool_args, (list, tuple)):
            args = list(tool_args)
        else:
            args = [tool_args] if tool_args is not None else []
        if tool_meta and self._signature_prefers_payload(tool_meta.signature):
            if kwargs:
                if "payload" in kwargs and len(kwargs) == 1:
                    args = [kwargs.get("payload")]
                    kwargs = {}
                elif not args:
                    args = [dict(kwargs)]
                    kwargs = {}
            if len(args) == 1 and isinstance(args[0], dict) and "payload" in args[0] and not kwargs:
                args = [args[0].get("payload")]
            if not (len(args) == 1 and isinstance(args[0], dict)):
                if args:
                    args = [{"value": args[0]}]
                else:
                    args = [{}]
                kwargs = {}
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
        self._toolgen_debug_event(
            "tool_invoke_start",
            tool_name=resolved_name,
            args_preview=self._preview_for_log(args),
            kwargs_preview=self._preview_for_log(kwargs),
            reason=reason,
        )
        self._mark_tool_invoked()
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
        self._trace("tool_agent_result", self._format_tool_result(resolved_name, result))
        return result

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

    def _solver_context_key(self, chat_history: ChatHistory) -> str:
        last_user = self._get_last_user_item(chat_history)
        content = last_user.content if last_user else ""
        if not content:
            return ""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _strip_supplemental_sections(self, text: str) -> str:
        if not text:
            return ""
        base = text
        for marker in ("\n\nTOOL RESULTS:", "\n\nCONTEXT:"):
            if marker in base:
                base = base.split(marker, 1)[0]
        return base.strip()

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
        if tool_result_injected:
            lines.append(
                "NOTE: TOOL RESULTS appended in the last task entry are supplemental evidence, not new tasks."
            )
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
        for idx in range(len(items) - 1, -1, -1):
            if items[idx].role != Role.USER:
                continue
            current = items[idx].content or ""
            if "CONTEXT:" in current:
                return chat_history
            updated = current + "\n\nCONTEXT:\n" + context_msg
            try:
                if hasattr(chat_history, "set"):
                    chat_history.set(
                        idx, ChatHistoryItem(role=Role.USER, content=updated)
                    )
                else:
                    items[idx] = ChatHistoryItem(role=Role.USER, content=updated)
            except Exception:
                return chat_history
            return chat_history
        return chat_history

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
        self,
        spec: ToolSpec,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolMetadata]:
        max_attempts = 3  # initial + up to 2 repairs
        last_error: Optional[str] = None
        current_raw_spec = raw_spec if raw_spec is not None else spec.__dict__

        for attempt in range(1, max_attempts + 1):
            self._tool_creation_attempts += 1

            # 1) Spec alignment (schema/signature/key contract)
            alignment_error = self._validate_spec_alignment(spec)
            if alignment_error:
                last_error = str(alignment_error)
                self._trace("tool_generation_error", f"stage=spec_alignment error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="spec_alignment",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )

                if attempt == max_attempts:
                    break

                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            try:
                raw_lines = self._explode_code_lines(list(spec.code_lines or []))
                enforced_lines = self._normalize_module_docstring_lines(
                    raw_lines,
                    description=spec.description,
                )
            except Exception as exc:
                last_error = f"docstring_invariant_failed: {exc}"
                self._trace("tool_generation_error", f"stage=docstring error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="docstring_invariant",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                if attempt == max_attempts:
                    break
                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue
            spec.code_lines = enforced_lines
            line_count = 0
            for line in spec.code_lines or []:
                text = str(line)
                if "\r\n" in text:
                    text = text.replace("\r\n", "\n")
                if "\n" in text:
                    line_count += len(text.splitlines())
                else:
                    line_count += 1
            if line_count > 90:
                last_error = "line_limit_exceeded: code_lines > 90"
                self._trace("tool_generation_error", f"stage=line_limit error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="line_limit",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                if attempt == max_attempts:
                    break
                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            placeholder_lines = [
                line
                for line in (spec.code_lines or [])
                if isinstance(line, str)
                and self._is_placeholder_line(line)
            ]
            if placeholder_lines:
                last_error = "placeholder tokens in code_lines"
                self._trace("tool_generation_error", f"stage=placeholder error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="placeholder_tokens",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                if attempt == max_attempts:
                    break
                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            # 2) Join code lines
            code = self._join_code_lines(spec.code_lines)
            if not code:
                last_error = "empty code after joining code_lines"
                self._trace("tool_generation_error", f"stage=code_join error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="code_join",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )

                if attempt == max_attempts:
                    break

                repair_spec = self._repair_tool_spec(
                    spec, last_error, chat_history, raw_spec=current_raw_spec
                )
                if repair_spec is None:
                    break
                spec = repair_spec
                current_raw_spec = spec.__dict__
                continue

            # 3) Validate tool code (compile/exec/smoke/self_test)
            self._toolgen_debug_event(
                "toolgen_validate_start",
                tool_name=spec.name,
                signature=spec.signature,
            )

            result = None
            try:
                result = validate_tool_code(code)
            except Exception as exc:
                last_error = f"validate exception: {exc}"
                self._trace("tool_generation_error", f"stage=validate_exception error={last_error}")
                self._trace("tool_generation_traceback", traceback.format_exc())
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage="validate_exception",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )

            if result and result.success:
                # Validation success
                self._toolgen_debug_event(
                    "toolgen_validate_success",
                    tool_name=spec.name,
                    signature=spec.signature,
                    self_test_passed=result.self_test_passed,
                )

                # 4) Register tool in registry
                try:
                    metadata = self._registry.register_tool(
                        name=spec.name,
                        code=code,
                        signature=spec.signature,
                        description=spec.description,
                        tool_type=spec.tool_type,
                        tool_category=spec.tool_category,  # ensure register_tool accepts this
                        input_schema=spec.input_schema,
                        capabilities=spec.capabilities,
                    )
                except Exception as exc:
                    last_error = f"register exception: {exc}"
                    self._trace("tool_generation_error", f"stage=register_exception error={last_error}")
                    self._trace("tool_generation_traceback", traceback.format_exc())
                    self._toolgen_debug_event(
                        "toolgen_registry_add_failed",
                        tool_name=spec.name,
                        signature=spec.signature,
                        error=last_error,
                    )
                    metadata = None

                if metadata:
                    # registry success
                    tool_path = (
                        self._registry._get_tool_path(metadata.name)
                        if hasattr(self._registry, "_get_tool_path")
                        else None
                    )
                    self._toolgen_debug_event(
                        "toolgen_registry_add_success",
                        tool_name=metadata.name,
                        signature=metadata.signature,
                        path=tool_path,
                    )
                    self._toolgen_debug_registered_tools[metadata.name] = 0
                    self._registry.record_validation_result(
                        metadata.name, True, self_test_passed=result.self_test_passed
                    )
                    self._tool_creation_successes += 1
                    return metadata

                # registration failed (but validation passed)
                last_error = last_error or "registration failed (metadata is None)"
                self._trace("tool_generation_error", f"stage=register error={last_error}")
                self._toolgen_debug_event(
                    "toolgen_registry_add_failed",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )
                self._toolgen_debug_event(
                    "INVARIANT_SAVE_NO_REGISTER",
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                )

            else:
                # Validation failed (normal case)
                last_error = getattr(result, "error", None) or last_error or "unknown validate failure"
                self._trace("tool_generation_error", f"stage=validate error={last_error}")
                self._trace(
                    "tool_generation_debug",
                    f"validate_failed name={spec.name} signature={spec.signature} tool_type={spec.tool_type}",
                )

                # stage classification (helps your logs)
                stage = "unknown"
                if isinstance(last_error, str):
                    if last_error.startswith("compile failed"):
                        stage = "compile"
                    elif last_error.startswith("exec failed"):
                        stage = "exec"
                    elif last_error.startswith("smoke test failed"):
                        stage = "smoke"
                    elif last_error.startswith("self_test failed"):
                        stage = "self_test"
                    elif "run() does not reference" in last_error:
                        stage = "static"

                code_preview = "\n".join(code.splitlines()[:40])
                self._toolgen_debug_event(
                    "toolgen_validate_failed",
                    stage=stage,
                    tool_name=spec.name,
                    signature=spec.signature,
                    error=last_error,
                    code_preview=code_preview,
                )

            # 5) Repair if we have attempts remaining
            if attempt == max_attempts:
                break

            repair_spec = self._repair_tool_spec(spec, last_error or "", chat_history)
            if repair_spec is None:
                break
            spec = repair_spec

        return None


    def _repair_tool_spec(
        self,
        spec: ToolSpec,
        error: str,
        chat_history: ChatHistory,
        *,
        raw_spec: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ToolSpec]:
        prompt = (
            "The previous tool failed validation.\n"
            f"Error: {error}\n"
            f"Previous spec: {json.dumps(raw_spec if raw_spec is not None else spec.__dict__, ensure_ascii=True, default=str)}\n"
            "Repair rules:\n"
            "- signature must be exactly run(payload: dict) -> dict\n"
            "- input_schema.required must be exactly ['payload']\n"
            "- input_schema.properties.payload must include properties + required\n"
            "- include all required keys: name, description, signature, tool_type, tool_category, input_schema, capabilities, code_lines\n"
            "- rename any field named 'set' to 'set_values' (schema + code)\n"
            "- first 3 lines must be exactly: \"\"\", <short line>, \"\"\"\n"
            "- no other triple quotes anywhere; do NOT use '''\n"
            "- do NOT add a run() docstring; use # Example: comments instead\n"
            "- run() must return a dict; self_test() must return True\n"
            "Return the full corrected JSON tool spec.\n"
        )
        self._toolgen_debug_event(
            "toolgen_repair_requested",
            error=error,
            prompt=prompt,
        )
        tool_history = ChatHistory()
        tool_history = self._safe_inject(
            tool_history, ChatHistoryItem(role=Role.USER, content=prompt)
        )
        response = self._toolgen_agent._inference(tool_history)
        self._toolgen_debug_event(
            "toolgen_repair_output",
            output=response.content or "",
            output_len=len(response.content or ""),
            finish_reason=getattr(response, "finish_reason", None),
        )
        raw_text_full = self._normalize_toolgen_content(response.content)
        if response.content and not raw_text_full.strip():
            raise RuntimeError(
                "BUG: toolgen repair content empty after normalization (parsing wrong source)"
            )
        if "[truncated]" in raw_text_full:
            raise RuntimeError("BUG: truncation marker found in toolgen repair raw_text")
        self._last_toolgen_parse_source = None
        fallback_info = self._toolgen_fallback_info()
        self._toolgen_debug_event(
            "toolgen_repair_parse_input",
            content_type=type(response.content).__name__,
            raw_text_len=len(raw_text_full),
            raw_text_head=raw_text_full[:120],
            raw_text_tail=raw_text_full[-120:] if len(raw_text_full) > 120 else raw_text_full,
        )
        try:
            if not raw_text_full.strip() and fallback_info and fallback_info.get("raw_output"):
                raw_text_full = str(fallback_info["raw_output"])
            payload = self.extract_tool_spec(raw_text_full, response)
        except Exception as exc:
            self._toolgen_debug_event(
                "toolgen_repair_parse_failed",
                raw_output=raw_text_full,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                error=str(exc),
            )
            return None
        raw_spec = dict(payload)
        before_lines = list(raw_spec.get("code_lines") or [])
        try:
            raw_spec = self._normalize_tool_spec(raw_spec)
        except Exception as exc:
            self._toolgen_debug_event(
                "toolgen_repair_parse_failed",
                raw_output=raw_text_full,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                error=str(exc),
            )
            return None
        if before_lines != raw_spec.get("code_lines"):
            self._toolgen_debug_event(
                "toolgen_code_lines_normalized", tool_name=raw_spec.get("name")
            )
        return ToolSpec.from_payload(raw_spec)

    def _handle_internal_tool_call(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        chat_history: ChatHistory,
    ) -> tuple[ToolResult, list[Any], dict[str, Any], str]:
        self._mark_tool_invoked()
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
        args_auto_built = False
        tool_meta = self._get_tool_metadata(resolved_name or tool_name)

        # If caller passed a single dict as args[0], treat it as kwargs.
        if len(tool_args) == 1 and isinstance(tool_args[0], dict) and not tool_kwargs:
            if not (tool_meta and self._signature_prefers_payload(tool_meta.signature)):
                tool_kwargs = dict(tool_args[0])
                tool_args = []

        # Defensive normalization
        if not isinstance(tool_args, list):
            tool_args = []
        if not isinstance(tool_kwargs, dict):
            tool_kwargs = {}
        if tool_meta and self._signature_prefers_payload(tool_meta.signature):
            if tool_kwargs:
                if "payload" in tool_kwargs and len(tool_kwargs) == 1 and not tool_args:
                    tool_args = [tool_kwargs.get("payload")]
                    tool_kwargs = {}
                elif not tool_args:
                    tool_args = [dict(tool_kwargs)]
                    tool_kwargs = {}
            if len(tool_args) == 1 and isinstance(tool_args[0], dict) and "payload" in tool_args[0] and not tool_kwargs:
                tool_args = [tool_args[0].get("payload")]
            if not (len(tool_args) == 1 and isinstance(tool_args[0], dict)):
                if tool_args:
                    tool_args = [{"value": tool_args[0]}]
                else:
                    tool_args = [{}]
                tool_kwargs = {}

        # Default behavior: if no args were given, pass last user message content
        if not tool_args and not tool_kwargs:
            last_user = self._get_last_user_item(chat_history)
            query = (last_user.content or "").strip() if last_user else ""
            if tool_meta and self._signature_prefers_payload(tool_meta.signature):
                tool_args = [self._default_payload_dict(query=query, chat_history=chat_history)]
            elif query:
                tool_args = [query]
            else:
                tool_args = [""]
            args_auto_built = True


        # Invoke tool
        self._tool_invocation_attempts += 1
        result = self._registry.invoke_tool(
            resolved_name or tool_name,
            *tool_args,
            invocation_context={"environment": self._resolved_environment_label()},
            **tool_kwargs,
        )
        if result.success:
            self._tool_invocation_successes += 1
        self._log_tool_invocation_event(
            tool_name=resolved_name or tool_name,
            args=tool_args,
            kwargs=tool_kwargs,
            result=result,
            reason="internal_tool",
            args_auto_built=args_auto_built,
            decision_action="use_tool",
        )
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
        if yaml is not None:
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


    def _required_tool_spec_keys(self) -> list[str]:
        return [
            "name",
            "description",
            "signature",
            "tool_type",
            "tool_category",
            "input_schema",
            "capabilities",
            "code_lines",
        ]

    def _tool_spec_has_required_keys(self, obj: Any) -> bool:
        if not isinstance(obj, Mapping):
            return False
        required = self._required_tool_spec_keys()
        return all(key in obj for key in required)

    def _repair_misnested_tool_spec(self, obj: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(obj, Mapping):
            return obj
        input_schema = obj.get("input_schema")
        if not isinstance(input_schema, Mapping):
            return obj
        required = self._required_tool_spec_keys()
        moved: dict[str, Any] = {}
        for key in required:
            if key == "input_schema" or key in obj:
                continue
            if key in input_schema:
                moved[key] = input_schema[key]
        if not moved:
            return obj
        updated = dict(obj)
        updated_schema = dict(input_schema)
        for key in moved:
            updated_schema.pop(key, None)
        updated.update(moved)
        updated["input_schema"] = updated_schema
        self._toolgen_debug_event(
            "toolgen_parse_repaired",
            reason="hoist_keys_from_input_schema",
            moved_keys=list(moved.keys()),
        )
        return updated

    def _extract_tool_call_arguments(self, response_obj: Any) -> Optional[str]:
        if response_obj is None:
            return None

        if isinstance(response_obj, Mapping):
            raw = response_obj.get("raw_output") or response_obj.get("arguments")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()

            tool_calls = response_obj.get("tool_calls")
            if tool_calls:
                first = tool_calls[0]
                func = None
                if isinstance(first, Mapping):
                    func = first.get("function")
                    if isinstance(func, Mapping):
                        args = func.get("arguments")
                        if isinstance(args, str) and args.strip():
                            return args.strip()
                return None

            func_call = response_obj.get("function_call")
            if isinstance(func_call, Mapping):
                args = func_call.get("arguments")
                if isinstance(args, str) and args.strip():
                    return args.strip()
            return None

        tool_calls = getattr(response_obj, "tool_calls", None)
        if tool_calls:
            first = tool_calls[0]
            func = getattr(first, "function", None)
            args = getattr(func, "arguments", None) if func is not None else None
            if isinstance(args, str) and args.strip():
                return args.strip()

        func_call = getattr(response_obj, "function_call", None)
        if func_call is not None:
            args = getattr(func_call, "arguments", None)
            if isinstance(args, str) and args.strip():
                return args.strip()

        return None

    def extract_tool_spec(self, raw_text: str, response_obj: Any) -> dict[str, Any]:
        raw_text_full = raw_text or ""
        if not isinstance(raw_text_full, str):
            raise ValueError(f"raw_text must be str (got {type(raw_text_full).__name__})")
        if '"output_preview"' in raw_text_full and '"output_truncated"' in raw_text_full:
            raise ValueError("raw_text appears to be a truncated preview, not full output")

        required = self._required_tool_spec_keys()
        parsed_from: Optional[str] = None

        def _parse_json_text(text: str, source_label: str) -> Optional[Mapping[str, Any]]:
            nonlocal parsed_from
            try:
                obj = json.loads(text)
            except Exception:
                sanitized = self._escape_invalid_json_backslashes(text)
                sanitized = self._escape_control_chars_in_json_strings(sanitized)
                if sanitized != text:
                    try:
                        obj = json.loads(sanitized)
                        parsed_from = source_label + "_sanitized"
                    except Exception:
                        return None
                else:
                    return None
            if isinstance(obj, Mapping):
                obj = self._repair_misnested_tool_spec(obj)
                if self._tool_spec_has_required_keys(obj):
                    if parsed_from is None:
                        parsed_from = source_label
                    return obj
                wrapped = obj.get("content")
                if isinstance(wrapped, str) and wrapped.strip():
                    inner = wrapped.strip()
                    inner_obj = _parse_json_text(inner, "wrapper_content_unwrap")
                    if inner_obj is not None:
                        parsed_from = "wrapper_content_unwrap"
                        return inner_obj
            return None

        obj = _parse_json_text(raw_text_full, "full_message_content")
        if obj is not None:
            self._last_toolgen_parse_source = parsed_from
            return dict(obj)

        sanitized_full = self._escape_invalid_json_backslashes(raw_text_full)
        sanitized_full = self._escape_control_chars_in_json_strings(sanitized_full)

        obj_text = self._extract_first_json_object(sanitized_full)
        if obj_text:
            obj = _parse_json_text(obj_text, "balanced_json_extraction")
            if obj is not None:
                self._last_toolgen_parse_source = parsed_from or "balanced_json_extraction"
                return dict(obj)

        repaired = self._close_truncated_json(sanitized_full)
        if repaired:
            obj = _parse_json_text(repaired, "closed_truncated_json")
            if obj is not None:
                self._last_toolgen_parse_source = parsed_from or "closed_truncated_json"
                return dict(obj)

        tool_args = self._extract_tool_call_arguments(response_obj)
        if tool_args:
            obj = _parse_json_text(tool_args, "tool_calls_args")
            if obj is not None:
                self._last_toolgen_parse_source = parsed_from or "tool_calls_args"
                return dict(obj)

        preview = raw_text_full[:200]
        truncated = raw_text_full.endswith("...[truncated]") or "[truncated]" in raw_text_full
        raise ValueError(
            "tool_spec_parse_failed "
            f"preview={preview!r} truncated={truncated} parsed_from={parsed_from!r}"
        )

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

        # Try parse as-is
        parsed = self._parse_creation_payload(text)
        if isinstance(parsed, Mapping):
            wrapped = parsed.get("content")
            if isinstance(wrapped, Mapping):
                return wrapped
            if isinstance(wrapped, str):
                inner = wrapped.strip()
                if inner:
                    inner_parsed = self._parse_creation_payload(inner)
                    if isinstance(inner_parsed, Mapping):
                        return inner_parsed
                    obj_text = self._extract_first_json_object(inner)
                    if obj_text:
                        inner_parsed = self._parse_creation_payload(obj_text)
                        if isinstance(inner_parsed, Mapping):
                            return inner_parsed
            return parsed

        # Try first JSON object inside the text
        obj_text = self._extract_first_json_object(text)
        if obj_text:
            parsed = self._parse_creation_payload(obj_text)
            if isinstance(parsed, Mapping):
                return parsed

        return None


    def _parse_tool_use_payload(self, content: str) -> Optional[Mapping[str, Any]]:
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


    def _get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        for tool in self._registry.list_tools():
            if tool.name == name:
                return tool
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

    def _join_code_lines(self, code_lines: Sequence[Any]) -> Optional[str]:
        normalized_lines: list[str] = []
        for line in code_lines:
            text = str(line)
            if "\r\n" in text:
                text = text.replace("\r\n", "\n")
            if "\n" in text:
                normalized_lines.extend(text.splitlines())
            else:
                normalized_lines.append(text)
        normalized_lines = self._ensure_tool_header(normalized_lines)
        if not normalized_lines:
            return None
        return "\n".join(normalized_lines).rstrip() + "\n"

    def _explode_code_lines(self, code_lines: Sequence[Any]) -> list[str]:
        exploded: list[str] = []
        for line in code_lines or []:
            text = str(line)
            if "\r\n" in text:
                text = text.replace("\r\n", "\n")
            if "\n" in text:
                exploded.extend(text.splitlines())
            else:
                exploded.append(text)
        return exploded

    def _is_placeholder_line(self, line: str) -> bool:
        text = (line or "").strip()
        if not text:
            return False
        if "<generated>" in text or "[truncated]" in text:
            return True
        return text in {"...", "…"}

    def _normalize_module_docstring_lines(
        self, lines: Sequence[str], description: Optional[str] = None
    ) -> list[str]:
        normalized = list(lines)
        idx = 0
        while idx < len(normalized) and normalized[idx].strip() == "":
            idx += 1
        normalized = normalized[idx:]

        desc = (description or "").strip()
        if not desc:
            desc = "Reusable tool."
        desc_line = desc.splitlines()[0].strip()
        if not desc_line:
            desc_line = "Reusable tool."
        desc_line = desc_line.replace('"""', '"').replace("'''", "'")
        if len(desc_line) > 120:
            desc_line = desc_line[:117] + "..."

        nonempty_idx = [i for i, line in enumerate(normalized) if line.strip()]
        matches = False
        if len(nonempty_idx) >= 3:
            first = normalized[nonempty_idx[0]].strip()
            second = normalized[nonempty_idx[1]].strip()
            third = normalized[nonempty_idx[2]].strip()
            if (
                first == '"""'
                and third == '"""'
                and second
                and '"""' not in second
                and "'''" not in second
            ):
                matches = True

        if not matches:
            normalized = ['"""', desc_line, '"""', ""] + normalized
            nonempty_idx = [i for i, line in enumerate(normalized) if line.strip()]

        if len(nonempty_idx) < 3:
            raise ValueError("docstring_invariant_failed: missing 3-line docstring")

        allowed = {nonempty_idx[0], nonempty_idx[2]}
        for i, line in enumerate(normalized):
            if "'''" in line:
                raise ValueError("docstring_invariant_failed: found '''")
            if '"""' in line and i not in allowed:
                raise ValueError("docstring_invariant_failed: extra triple quotes")

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

    def _normalize_code_lines(self, code_lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in code_lines or []:
            if not isinstance(line, str):
                line = str(line)
            s = line.strip("\n")
            # Strip markdown fences if the model included them
            if s.startswith("```"):
                continue
            if s.startswith('"""') or s.startswith("'''"):
                out.append(s)
                continue
            # Unwrap quoted lines like "\"def run(...)\"" or "'def run(...)'"
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                inner = s[1:-1]
                inner = inner.replace(r"\\", "\\")
                inner = inner.replace(r"\'", "'").replace(r"\"", '"')
                s = inner
            out.append(s)

        # Trim leading/trailing empties
        while out and not out[0].strip():
            out.pop(0)
        while out and not out[-1].strip():
            out.pop()
        return out

    def _ensure_self_test(self, code_lines: list[str], input_schema: dict) -> list[str]:
        if any("def self_test" in (ln or "") for ln in code_lines):
            return code_lines

        payload_props = (
            (input_schema or {})
            .get("properties", {})
            .get("payload", {})
            .get("properties", {})
        )
        payload_required = (
            (input_schema or {})
            .get("properties", {})
            .get("payload", {})
            .get("required", [])
        )

        def _placeholder(v: dict) -> object:
            t = (v or {}).get("type")
            if t == "string":
                return "x"
            if t == "integer":
                return 1
            if t == "number":
                return 1
            if t == "boolean":
                return True
            if t == "array":
                return []
            if t == "object":
                return {}
            return "x"

        smoke: dict[str, Any] = {}
        for k in payload_required:
            smoke[k] = _placeholder(payload_props.get(k, {}))

        code_lines = list(code_lines)
        code_lines += [
            "",
            "def self_test() -> bool:",
            "    try:",
            f"        out = run({repr(smoke)})",
            "        return isinstance(out, dict)",
            "    except Exception:",
            "        return False",
        ]
        return code_lines

    def _normalize_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        required = [
            "name",
            "description",
            "signature",
            "tool_type",
            "tool_category",
            "input_schema",
            "capabilities",
            "code_lines",
        ]
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(f"missing required fields: {missing}")

        code_lines_before = list(spec.get("code_lines") or [])
        spec["code_lines"] = self._normalize_code_lines(code_lines_before)
        spec["code_lines"] = self._ensure_self_test(
            spec["code_lines"], spec.get("input_schema") or {}
        )
        return spec

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
            # print("[SelfEvolvingController] Reached generated tool limit; skipping.")
            return None

        if not isinstance(creation_request, Mapping):
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="invalid_creation_request_type",
                request_type=type(creation_request).__name__,
            )
            return None

        tool_spec = dict(creation_request)
        before_lines = list(tool_spec.get("code_lines") or [])
        try:
            tool_spec = self._normalize_tool_spec(tool_spec)
            if before_lines != tool_spec.get("code_lines"):
                self._toolgen_debug_event(
                    "toolgen_code_lines_normalized", tool_name=tool_spec.get("name")
                )
        except Exception as exc:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="normalization_failed",
                error=str(exc),
            )
            return None
        self._toolgen_debug_event(
            "toolgen_parse_start",
            payload_keys=list(tool_spec.keys()),
        )

        required_keys = [
            "name",
            "description",
            "signature",
            "tool_type",
            "tool_category",
            "input_schema",
            "capabilities",
            "code_lines",
        ]
        missing = [
            key for key in required_keys if tool_spec.get(key) in (None, "", [], {})
        ]
        if missing:
            self._trace(
                "tool_generation_error",
                f"stage=register error=missing required fields ({', '.join(missing)})",
            )
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="missing_required_fields",
                missing_keys=missing,
                tool_name=tool_spec.get("name"),
                signature=tool_spec.get("signature"),
            )
            self._toolgen_debug_event(
                "INVARIANT_BROKEN",
                reason="parsed_missing_required_fields",
                payload_keys=list(tool_spec.keys()),
            )
            return None

        preflight_error = None
        if tool_spec.get("signature") != "run(payload: dict) -> dict":
            preflight_error = "signature_mismatch"
        else:
            schema_required = (
                (tool_spec.get("input_schema") or {}).get("required") or []
            )
            if schema_required != ["payload"]:
                preflight_error = "input_schema_required_mismatch"
        if preflight_error:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason=preflight_error,
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                preflight_error,
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        code_lines = tool_spec.get("code_lines") or []
        placeholder_lines = [
            line
            for line in code_lines
            if isinstance(line, str)
            and self._is_placeholder_line(line)
        ]
        if placeholder_lines:
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="placeholder_tokens_in_code",
                tool_name=tool_spec.get("name"),
                offending_lines=placeholder_lines[:3],
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "placeholder tokens in code_lines",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )
        inline_compounds = [
            line
            for line in code_lines
            if isinstance(line, str) and "for " in line and ": if " in line
        ]
        if inline_compounds:
            self._trace(
                "tool_generation_error",
                "stage=validate error=inline_compound_statement_detected",
            )
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="code_style_inline_if",
                offending_lines=inline_compounds[:3],
            )
            return None

        tool_type = str(tool_spec.get("tool_type"))
        tool_category = tool_spec.get("tool_category")
        allowed_categories = {"parser", "normalizer", "planner", "validator", "linter", "formatter"}
        if tool_category not in allowed_categories:
            self._trace(
                "tool_generation_error",
                f"stage=validate error=invalid tool_category {tool_category!r}",
            )
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="invalid_tool_category",
                tool_category=tool_category,
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                f"invalid tool_category: {tool_category}",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        input_schema = tool_spec.get("input_schema")
        if not isinstance(input_schema, Mapping):
            self._trace("tool_generation_error", "stage=validate error=missing input_schema")
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="missing_input_schema",
                tool_type=tool_type,
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "missing input_schema",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        capabilities = tool_spec.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            self._trace("tool_generation_error", "stage=validate error=missing capabilities")
            self._toolgen_debug_event(
                "toolgen_register_failed",
                reason="missing_capabilities",
                tool_name=tool_spec.get("name"),
            )
            repair_spec = self._repair_tool_spec(
                ToolSpec.from_payload(tool_spec),
                "missing capabilities",
                chat_history,
                raw_spec=tool_spec,
            )
            if repair_spec is None:
                return None
            return self._validate_and_register_tool(
                repair_spec, chat_history, raw_spec=tool_spec
            )

        spec = ToolSpec.from_payload(tool_spec)
        metadata = self._validate_and_register_tool(
            spec, chat_history, raw_spec=tool_spec
        )
        if metadata:
            self._generated_tool_counter += 1
            self._mark_tool_invoked()
        return metadata

    def _signature_param_names(self, signature: str) -> tuple[list[str], bool]:
        if not signature:
            return [], False
        match = re.search(r"run\s*\(([^)]*)\)", signature)
        if not match:
            return [], False
        params = [p.strip() for p in match.group(1).split(",") if p.strip()]
        names: list[str] = []
        has_var_kwargs = False
        for param in params:
            if param.startswith("**"):
                has_var_kwargs = True
                continue
            if param.startswith("*"):
                continue
            base = param.split(":", 1)[0].split("=", 1)[0].strip()
            if base:
                names.append(base)
        return names, has_var_kwargs

    def _validate_spec_alignment(self, spec: ToolSpec) -> Optional[str]:
        if not spec.signature:
            return "missing signature"

        # Enforce the single signature style you want
        if spec.signature.strip() != "run(payload: dict) -> dict":
            return f"signature must be exactly run(payload: dict) -> dict (got {spec.signature!r})"

        schema = spec.input_schema if isinstance(spec.input_schema, Mapping) else None
        if schema is None:
            return "missing input_schema"

        if schema.get("type") != "object":
            return "input_schema.type must be 'object'"

        required = schema.get("required") or []
        if required != ["payload"]:
            return f"input_schema.required must be ['payload'] (got {required})"

        props = schema.get("properties") or {}
        payload_schema = props.get("payload")
        if not isinstance(payload_schema, Mapping):
            return "input_schema.properties.payload must exist and be an object schema"

        if payload_schema.get("type") != "object":
            return "input_schema.properties.payload.type must be 'object'"

        if "properties" not in payload_schema:
            return "input_schema.properties.payload.properties is required"
        if "required" not in payload_schema:
            return "input_schema.properties.payload.required is required"

        # forbid 'set' anywhere (shallow check + JSON dump check)
        try:
            dumped = json.dumps(schema, ensure_ascii=True, default=str)
            if '"set"' in dumped or "'set'" in dumped:
                return "schema contains forbidden key name 'set' (use 'set_values')"
        except Exception:
            pass

        return None




    def _expected_tool_type_for_archetype(self, archetype: str) -> Optional[str]:
        return {
            "answer_guard": "validator",
            "sql_static_check": "linter",
            "final_output_formatter": "formatter",
            "general_helper": "utility",
        }.get(archetype)

    def _should_use_tool(
        self,
        tool: ToolMetadata,
        *,
        candidate_output: Optional[str],
        query: Optional[str] = None,
    ) -> bool:
        category = (tool.tool_category or "").lower()
        if category in {"validator", "linter", "formatter"} and not candidate_output:
            return False
        if query and self._is_structured_task(query):
            if category in {"utility"} and not candidate_output:
                return False
        return True

    def _reuse_existing_tool(
        self,
        query: str,
        *,
        candidate_output: Optional[str] = None,
        needed_archetype: Optional[str] = None,
    ) -> Optional[ToolMetadata]:
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
                # print(
                #     "[SelfEvolvingController] Reuse gate skipped: "
                #     f"tool_type='{best.tool.tool_type}' != expected='{expected_type}'"
                # )
                return None

        if not self._should_use_tool(
            best.tool, candidate_output=candidate_output, query=query
        ):
            return None
        if getattr(best.tool, "self_test_passed", True) is False:
            # print("[SelfEvolvingController] Reuse gate skipped: self_test failed.")
            return None
        if getattr(best.tool, "reliability_score", 0.0) < 0.4:
            # print("[SelfEvolvingController] Reuse gate skipped: low reliability.")
            return None


        if best.score < self._reuse_similarity_threshold:
            # print(
            #     "[SelfEvolvingController] Reuse gate skipped: "
            #     f"best_score={best.score:.3f} < threshold={self._reuse_similarity_threshold:.3f}"
            # )
            return None
        if "run(" not in (best.tool.signature or ""):
            # print(
            #     "[SelfEvolvingController] Reuse gate skipped: "
            #     f"tool '{best.tool.name}' missing run() signature."
            # )
            return None
        # print(
        #     "[SelfEvolvingController] Reuse gate selected "
        #     f"tool='{best.tool.name}' score={best.score:.3f}."
        # )
        return best.tool
    
    # def _get_archetype_specs(self) -> dict[str, dict[str, Any]]:
    #     return {
    #         "command_validator": {
    #             "name_hint": "command_validator",
    #             "tool_type": "validator",
    #             "tool_category": "validator",
    #             "signature": "run(task_text: str, command_text: str) -> dict",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "task_text": {"type": "string"},
    #                     "command_text": {"type": "string"},
    #                 },
    #                 "required": ["task_text", "command_text"],
    #             },
    #             "capabilities": [
    #                 "command_intent_check",
    #                 "risky_flag_detection",
    #                 "command_format_guard",
    #             ],
    #             "requirements": [
    #                 "Validate that command_text aligns with task_text intent (best-effort heuristics).",
    #                 "Detect risky patterns (e.g., rm -rf, sudo, destructive redirects) and report them.",
    #                 "Return dict with keys: valid(bool), errors(list[str]), warnings(list[str]), fixed_command(optional str).",
    #                 "Never raise; on error return valid=False with a reason in errors.",
    #                 "Use stdlib only; keep logic general across OS tasks.",
    #             ],
    #         },
    #         "path_safety_linter": {
    #             "name_hint": "path_safety_linter",
    #             "tool_type": "linter",
    #             "tool_category": "linter",
    #             "signature": "run(command_text: str, allowed_roots: list[str] | None = None) -> dict",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "command_text": {"type": "string"},
    #                     "allowed_roots": {"type": ["array", "null"], "items": {"type": "string"}},
    #                 },
    #                 "required": ["command_text"],
    #             },
    #             "capabilities": [
    #                 "path_traversal_check",
    #                 "destructive_path_guard",
    #                 "root_scope_check",
    #             ],
    #             "requirements": [
    #                 "Detect path traversal patterns (e.g., ../) and absolute root operations.",
    #                 "Warn on destructive path usage (rm -rf /, mv to /, chmod -R /).",
    #                 "Return dict with keys: valid(bool), errors(list[str]), warnings(list[str]), safe_command(optional str).",
    #                 "If allowed_roots is provided, flag paths outside those roots.",
    #                 "Use stdlib only; keep logic general.",
    #             ],
    #         },
    #         "output_format_checker": {
    #             "name_hint": "output_format_checker",
    #             "tool_type": "validator",
    #             "tool_category": "validator",
    #             "signature": "run(task_text: str, candidate_output: str) -> dict",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "task_text": {"type": "string"},
    #                     "candidate_output": {"type": "string"},
    #                 },
    #                 "required": ["task_text", "candidate_output"],
    #             },
    #             "capabilities": [
    #                 "output_parseability_check",
    #                 "action_block_check",
    #                 "final_answer_format_check",
    #             ],
    #             "requirements": [
    #                 "Infer expected output format hints from task_text (Action/Act/Final Answer, code fences).",
    #                 "Validate candidate_output against those hints (best-effort).",
    #                 "Return dict with keys: can_proceed(bool), errors(list[str]), warnings(list[str]), fixed_output(optional str).",
    #                 "Never add extra commentary in fixed_output beyond required format.",
    #                 "Use stdlib only; keep logic general.",
    #             ],
    #         },
    #         "answer_guard": {
    #             "name_hint": "answer_guard",
    #             "tool_type": "validator",
    #             "tool_category": "validator",
    #             "signature": "run(task_text: str, last_sql: str | None = None, last_db_response: object | None = None) -> dict",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "task_text": {"type": "string"},
    #                     "last_sql": {"type": ["string", "null"]},
    #                     "last_db_response": {},
    #                 },
    #                 "required": ["task_text"],
    #             },
    #             "capabilities": [
    #                 "classify_task_intent",
    #                 "final_answer_format_guard",
    #                 "select_result_passthrough",
    #             ],
    #             "requirements": [
    #                 "Detect whether task intent is SELECT vs mutation (INSERT/UPDATE/DELETE) from task_text.",
    #                 "If last_sql starts with SELECT, final_answer MUST be exactly repr(last_db_response).",
    #                 "Return a dict with keys: can_submit(bool), final_answer(str), reason(str), intent(str).",
    #                 "Never raise; on error return can_submit=False with a reason.",
    #                 "Use stdlib only; keep logic general across many DB tasks.",
    #             ],
    #         },
    #         "sql_static_check": {
    #             "name_hint": "sql_static_check",
    #             "tool_type": "linter",
    #             "tool_category": "linter",
    #             "signature": "run(table_name: str, headers: list[str], sql: str) -> dict",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "table_name": {"type": "string"},
    #                     "headers": {"type": "array", "items": {"type": "string"}},
    #                     "sql": {"type": "string"},
    #                 },
    #                 "required": ["table_name", "headers", "sql"],
    #             },
    #             "capabilities": [
    #                 "single_line_sql_check",
    #                 "unknown_column_detection",
    #                 "table_name_sanity_check",
    #             ],
    #             "requirements": [
    #                 "Validate sql is a single line (no '\\n' characters).",
    #                 "Check referenced columns in simple SELECT/INSERT patterns are a subset of headers (best-effort heuristics).",
    #                 "Check table_name appears to match expected table (best-effort).",
    #                 "Return dict with keys: valid(bool), errors(list[str]), warnings(list[str]), fixed_sql(optional str).",
    #                 "If you propose fixed_sql, keep it semantically close and still single-line.",
    #                 "Use stdlib only; keep logic general.",
    #             ],
    #         },
    #         "final_output_formatter": {
    #             "name_hint": "final_output_formatter",
    #             "tool_type": "formatter",
    #             "tool_category": "formatter",
    #             "signature": "run(mode: str, sql: str | None = None, final_answer: object | None = None) -> str",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "mode": {"type": "string", "enum": ["Operation", "Answer"]},
    #                     "sql": {"type": ["string", "null"]},
    #                     "final_answer": {},
    #                 },
    #                 "required": ["mode"],
    #             },
    #             "capabilities": [
    #                 "emit_exact_action_block",
    #                 "emit_sql_fence",
    #                 "emit_final_answer_line",
    #             ],
    #             "requirements": [
    #                 "For Operation: output exactly:\nAction: Operation\\n```sql\\n<ONE LINE SQL>\\n```",
    #                 "For Answer: output exactly:\nAction: Answer\\nFinal Answer: <repr(final_answer)>",
    #                 "Never add extra text outside the required format.",
    #                 "If mode=Operation and sql is missing/empty, return an error string explaining missing sql (still no extra formatting).",
    #             ],
    #         },
    #         "general_helper": {
    #             "name_hint": "general_helper",
    #             "tool_type": "utility",
    #             "tool_category": "parser",
    #             "signature": "run(task_text: str) -> dict",
    #             "input_schema": {
    #                 "type": "object",
    #                 "properties": {"task_text": {"type": "string"}},
    #                 "required": ["task_text"],
    #             },
    #             "capabilities": [
    #                 "extract_intent",
    #                 "extract_constraints",
    #                 "extract_table_and_headers_when_present",
    #             ],
    #             "requirements": [
    #                 "Parse task_text and extract a small structured dict: intent, table_name (if present), limit/offset (if present), and any mentioned columns (best-effort).",
    #                 "Return stable keys even when fields are missing (use None).",
    #                 "Use stdlib only; keep it general across tasks.",
    #             ],
    #         },
    #     }



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


    def _extract_toolgen_payload_from_response(self, response) -> Optional[Mapping[str, Any]]:
        text = (getattr(response, "content", "") or "")
        try:
            return self.extract_tool_spec(text, response)
        except Exception:
            return None

    def _maybe_generate_tool_for_query(
        self,
        query: str,
        chat_history: ChatHistory,
        *,
        allow_reuse: bool = True,
    ) -> Optional[ToolMetadata]:
        if not query.strip():
            self._toolgen_debug_event("toolgen_skipped", reason="empty_query")
            return None

        # Reuse first (no archetype gating anymore)
        if allow_reuse:
            reuse_query = (
                self._normalize_retrieval_query(query)
                if hasattr(self, "_normalize_retrieval_query")
                else query
            )
            candidate_output = self._get_candidate_output(chat_history, query)
            reuse = self._reuse_existing_tool(
                reuse_query, candidate_output=candidate_output, needed_archetype=None
            )
            if reuse is not None:
                self._toolgen_debug_event(
                    "toolgen_reuse_hit",
                    tool_name=reuse.name,
                    reason="reuse_existing_tool",
                )
                return reuse

        if not self._force_tool_generation_if_missing:
            self._toolgen_debug_event("toolgen_skipped", reason="force_disabled")
            return None
        if self._generated_tool_counter >= self._max_generated_tools_per_run:
            self._toolgen_debug_event("toolgen_skipped", reason="max_generated_tools_per_run")
            return None

        prompt = self._toolgen_request_prompt(query, chat_history)
        self._trace("tool_agent_input", prompt)
        self._toolgen_debug_event(
            "toolgen_input_built",
            prompt=prompt,
            prompt_len=len(prompt),
        )
        tool_history = ChatHistory()
        tool_history = self._safe_inject(tool_history, ChatHistoryItem(role=Role.USER, content=prompt))

        model_info = self._toolgen_model_info()
        self._toolgen_debug_event("toolgen_model_called", model_info=model_info)
        response = self._toolgen_agent._inference(tool_history)
        self._trace("tool_agent_result", response.content)
        fallback_info = self._toolgen_fallback_info()
        if fallback_info:
            self._toolgen_debug_event("toolgen_output_extracted", **fallback_info)
        self._toolgen_debug_event(
            "toolgen_raw_output",
            output=response.content or "",
            output_len=len(response.content or ""),
            finish_reason=getattr(response, "finish_reason", None),
        )
        raw_text_full = self._normalize_toolgen_content(response.content)
        if response.content and not raw_text_full.strip():
            raise RuntimeError(
                "BUG: toolgen content empty after normalization (parsing wrong source)"
            )
        if "[truncated]" in raw_text_full:
            raise RuntimeError("BUG: truncation marker found in toolgen raw_text")
        self._toolgen_debug_event(
            "toolgen_parse_input",
            content_type=type(response.content).__name__,
            raw_text_len=len(raw_text_full),
            raw_text_head=raw_text_full[:120],
            raw_text_tail=raw_text_full[-120:] if len(raw_text_full) > 120 else raw_text_full,
        )
        self._last_toolgen_parse_source = None
        try:
            if not raw_text_full.strip() and fallback_info and fallback_info.get("raw_output"):
                raw_text_full = str(fallback_info["raw_output"])
            creation_request = self.extract_tool_spec(raw_text_full, response)
        except Exception as exc:
            preview = (response.content or "")[:800]
            self._trace("tool_generation_debug", f"parse_failed content_preview={preview!r}")
            self._toolgen_debug_event(
                "toolgen_parse_failed",
                raw_output=raw_text_full,
                fallback_source=fallback_info.get("fallback_source") if fallback_info else None,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                parsed_source_len=len(raw_text_full),
                error=str(exc),
            )
            self._toolgen_debug_event(
                "INVARIANT_PARSE_FAIL",
                raw_output=raw_text_full,
                fallback_source=fallback_info.get("fallback_source") if fallback_info else None,
                parsed_from=getattr(self, "_last_toolgen_parse_source", None),
                parsed_source_len=len(raw_text_full),
            )
            return None

        self._toolgen_debug_event(
            "toolgen_parse_success",
            parsed_from=getattr(self, "_last_toolgen_parse_source", None),
            parsed_source_len=len(raw_text_full),
        )
        created = self._register_tool_from_payload(creation_request, chat_history)
        if created is None:
            self._toolgen_debug_event("toolgen_register_failed", reason="register_returned_none")
        return created



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
            "candidate_output": candidate_output,
            "candidate_artifact": candidate_output,
            "output": candidate_output,
            "response": candidate_output,
            "history": None,
            "environment": self._resolved_environment_label(),
        }

        if schema is not None:
            # Payload-style schema (ToolGen contract)
            if "payload" in (schema.get("properties") or {}) and "payload" in (schema.get("required") or []):
                payload = self._build_payload_from_payload_schema(schema, key_map)
                if payload is None:
                    return None
                # run(payload: dict) expects the inner dict
                if self._signature_prefers_payload(tool.signature):
                    return [payload], {}
                # if someone made signature run(payload=...) but registry wants kwargs
                return [], {"payload": payload}

            # Non-payload schemas (legacy / bootstrap tools)
            properties = schema.get("properties") or {}
            required = schema.get("required") or []
            for key in properties.keys():
                if key in key_map and key_map[key] is not None:
                    values[key] = key_map[key]
            for key in required:
                if key not in values:
                    return None
            if required:
                return [values[k] for k in required], {}
            if values:
                return [values], {}


        if query:
            return [query], {}
        if candidate_output is not None:
            return [candidate_output], {}
        return None

    @staticmethod
    def _pack_tool_args(
        args: Sequence[Any], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        packed: dict[str, Any] = {}
        if args:
            packed["args"] = list(args)
        if kwargs:
            packed["kwargs"] = dict(kwargs)
        return packed

    def _auto_build_tool_args(
        self,
        tool: ToolMetadata,
        *,
        query: str,
        chat_history: ChatHistory,
    ) -> Optional[dict[str, Any]]:
        """
        Auto-build args/kwargs for a tool call.
        Supports BOTH:
        (A) ToolGen contract tools: signature run(payload: dict) -> dict with nested schema under properties.payload.properties
        (B) legacy/simple tools with top-level schema properties
        """
        last_user = self._get_last_user_item(chat_history)
        candidate_output = self._get_candidate_output(chat_history, query)
        history_text = self._toolgen_render_history(chat_history, max_chars_per_item=300)

        # Prefer the main builder (it should already handle payload-style schemas if you patched it)
        payload: Optional[tuple[list[Any], dict[str, Any]]] = None
        try:
            payload = self._build_tool_invocation(tool, query=query, candidate_output=candidate_output)
        except Exception:
            payload = None

        if payload:
            args, kwargs = payload
            packed = self._pack_tool_args(args, kwargs)
            if packed:
                return packed

        schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else None

        # Common values we can map into schemas
        key_map: dict[str, Any] = {
            "task_text": query,
            "query": query,
            "text": query,
            "instruction": query,
            "candidate_output": candidate_output,
            "candidate_artifact": candidate_output,
            "output": candidate_output,
            "response": candidate_output,
            "history": history_text,
            "history_compact": history_text,
            "environment": self._resolved_environment_label(),
        }

        if schema is not None:
            properties = schema.get("properties") or {}
            required = schema.get("required") or []

            # --- (A) ToolGen contract: top-level requires ["payload"], with nested payload schema ---
            if "payload" in properties and "payload" in required:
                # Expect payload schema like:
                # properties.payload: {"type":"object","properties":{...},"required":[...]}
                payload_schema = properties.get("payload")
                if isinstance(payload_schema, Mapping):
                    inner_props = payload_schema.get("properties") or {}
                    inner_required = payload_schema.get("required") or []

                    built_payload: dict[str, Any] = {}
                    for k in inner_props.keys():
                        if k in key_map and key_map[k] is not None:
                            built_payload[k] = key_map[k]

                    # If nested required keys exist, enforce them
                    if inner_required and not all(k in built_payload for k in inner_required):
                        return None

                    # If the tool expects run(payload) we pass the INNER dict as positional arg
                    if self._signature_prefers_payload(tool.signature):
                        return {"args": [built_payload]}

                    # If someone made a tool that actually wants kwargs payload=...
                    return {"kwargs": {"payload": built_payload}}

                return None  # schema says payload required but payload schema is malformed

            # --- (B) Legacy/simple schemas: required are top-level keys ---
            values: dict[str, Any] = {}
            for k in properties.keys():
                if k in key_map and key_map[k] is not None:
                    values[k] = key_map[k]

            if required and all(k in values for k in required):
                if self._signature_prefers_payload(tool.signature):
                    # Some tools may have signature run(payload) but schema isn't nested; still pass dict
                    return {"args": [values]}
                return {"args": [values[k] for k in required]}

            if values:
                # best-effort: pass as one dict
                return {"args": [values]}

        # --- Final fallback: give the tool something useful ---
        fallback_payload = {
            "task_text": query,
            "candidate_output": candidate_output,
            "history": history_text,
            "environment": self._resolved_environment_label(),
        }

        if self._signature_prefers_payload(tool.signature) or isinstance(tool.input_schema, Mapping):
            return {"args": [fallback_payload]}

        if query:
            return {"args": [query]}

        return {"args": [fallback_payload]}


    def _select_tool_for_query(
        self,
        query: str,
        *,
        categories: Optional[set[str]] = None,
        candidate_output: Optional[str] = None,
        min_score: Optional[float] = None,
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
        if retrieved:
            self._toolgen_debug_event(
                "tool_retrieval",
                query_preview=(query or "")[:300],
                results=[
                    {"name": item.tool.name, "score": item.score}
                    for item in retrieved
                ],
            )
            if self._toolgen_debug_registered_tools:
                retrieved_names = {item.tool.name for item in retrieved}
                missing = [
                    name
                    for name in self._toolgen_debug_registered_tools.keys()
                    if name not in retrieved_names
                ]
                if missing:
                    self._toolgen_debug_event(
                        "INVARIANT_REGISTER_NO_REUSE",
                        query_preview=(query or "")[:300],
                        missing_registered_tools=missing,
                    )
        if not retrieved:
            if self._toolgen_debug_registered_tools:
                self._toolgen_debug_event(
                    "INVARIANT_REGISTER_NO_REUSE",
                    query_preview=(query or "")[:300],
                    missing_registered_tools=list(self._toolgen_debug_registered_tools.keys()),
                )
            return None
        score_threshold = self._tool_match_min_score if min_score is None else min_score
        for item in retrieved:
            if item.score < score_threshold:
                continue
            tool = item.tool
            if categories and getattr(tool, "tool_category", None) not in categories:
                continue
            if not self._should_use_tool(
                tool, candidate_output=candidate_output, query=query
            ):
                continue
            self._toolgen_debug_event(
                "tool_selected",
                tool_name=tool.name,
                score=item.score,
                query_preview=(query or "")[:300],
            )
            return tool
        return None

    def _invoke_tool_for_query(
        self,
        tool: ToolMetadata,
        query: str,
        *,
        candidate_output: Optional[str] = None,
        reason: str = "auto_invoke",
        decision_action: Optional[str] = None,
    ) -> Optional[tuple[ToolResult, list[Any], dict[str, Any]]]:
        payload = self._build_tool_invocation(
            tool, query=query, candidate_output=candidate_output
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
        self._toolgen_debug_event(
            "tool_invoke_start",
            tool_name=tool.name,
            args_preview=self._preview_for_log(args),
            kwargs_preview=self._preview_for_log(kwargs),
            reason=reason,
        )
        self._mark_tool_invoked()
        self._tool_invocation_attempts += 1
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
        self._trace("tool_agent_result", self._format_tool_result(tool.name, result))
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
        tool_category: Optional[Any] = None,
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
            tool_category=str(tool_category) if tool_category is not None else None,
            input_schema=input_schema,
            capabilities=capabilities,
        )

    def _bootstrap_tools(self, bootstrap_tools: Sequence[Mapping[str, Any]]) -> None:
        if not bootstrap_tools:
            return
        # print(f"[SelfEvolvingController] Bootstrapping {len(bootstrap_tools)} tool(s).")
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
                # print(f"[SelfEvolvingController] Bootstrapped '{metadata.name}'.")

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
        match = re.search(r"Bad output was:\s*([\s\S]+)$", text.strip())
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

    # def _solver_prompt_no_tools(self) -> str:
    #     base = (self._solver_system_prompt or "").strip()
    #     if not base:
    #         return "You are a helpful assistant."
    #     lines = []
    #     for line in base.splitlines():
    #         if "tool" in line.lower():
    #             continue
    #         lines.append(line)
    #     cleaned = "\n".join([ln for ln in lines if ln.strip()])
    #     return cleaned.strip() or "You are a helpful assistant."

    def _solver_prompt_no_tools(self) -> str:
        return (self._solver_system_prompt or "").strip() or "You are a helpful assistant."


    def _orchestrated_inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.inference(chat_history)

        working_history = self._clone_history(self._prune_for_current_task(chat_history))
        self._tool_invoked_in_last_inference = False

        user_items = [item for item in self._history_items(working_history) if item.role == Role.USER]
        if len(user_items) >= 2:
            original_query = user_items[1].content
        else:
            orig_last_user = self._get_last_user_item(working_history)
            original_query = orig_last_user.content if orig_last_user else ""
        task_query = (original_query or "").strip()

        def _record_tool_error(stage: str, message: str, tb: Optional[str] = None) -> None:
            self._trace("tool_pipeline_error", f"{stage}: {message}")
            if tb:
                self._trace("tool_pipeline_traceback", tb)

        def _requires_action_format() -> bool:
            return self._is_db_bench_env()

        def _fallback_response() -> str:
            if _requires_action_format():
                return "Action: Answer\nFinal Answer: []"
            return "OK."

        def _ensure_solver_output(content: str) -> str:
            text = (content or "").strip()
            if not text:
                return _fallback_response()
            if _requires_action_format() and not re.match(r"^Action:\s*(Operation|Answer)\b", text):
                return _fallback_response()
            return content

        tool_traces: list[dict[str, Any]] = []
        tool_error: Optional[str] = None
        decision: dict[str, Any] = {"action": "no_tool"}
        action = "no_tool"
        selected_tool: Optional[ToolMetadata] = None
        tool_agent_traced = False
        tool_result_injected = False

        try:
            decision = self._orchestrate_decision(task_query, working_history)
            action = decision.get("action", "no_tool")
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args")
            candidate_output = self._get_candidate_output(working_history, task_query)

            if action == "use_tool":
                tool_agent_traced = True
                if not isinstance(tool_name, str) or not tool_name:
                    auto_tool = self._select_tool_for_query(
                        task_query, candidate_output=candidate_output
                    )
                    if auto_tool:
                        tool_name = auto_tool.name
                    else:
                        tool_error = "missing tool_name for use_tool"
                        self._trace("tool_agent_input", "ERROR missing tool_name for use_tool")
                        _record_tool_error("use_tool", tool_error)
                        tool_result = ToolResult.failure(tool_error)
                        working_history = self._inject_tool_result_message(
                            working_history, "use_tool", tool_result
                        )
                        tool_result_injected = True
                        tool_name = None
                if tool_name:
                    args_auto_built = False
                    tool_args_final = tool_args
                    if tool_args is None or tool_args == {} or tool_args == []:
                        resolved_name = (
                            self._registry.resolve_name(tool_name)
                            if hasattr(self._registry, "resolve_name")
                            else None
                        )
                        resolved_name = resolved_name or tool_name
                        tool_meta = self._get_tool_metadata(resolved_name)
                        if tool_meta is not None:
                            tool_args_final = self._auto_build_tool_args(
                                tool_meta, query=task_query, chat_history=working_history
                            )
                            args_auto_built = tool_args_final is not None
                    if tool_args_final is None or tool_args_final == {} or tool_args_final == []:
                        resolved_name = (
                            self._registry.resolve_name(tool_name)
                            if hasattr(self._registry, "resolve_name")
                            else None
                        )
                        resolved_name = resolved_name or tool_name
                        tool_meta = self._get_tool_metadata(resolved_name)

                        if tool_meta and self._signature_prefers_payload(tool_meta.signature):
                            tool_args_final = {"args": [self._default_payload_dict(query=task_query, chat_history=working_history)]}
                        else:
                            tool_args_final = {"args": [task_query]}
                        args_auto_built = True

                    tool_result = self._invoke_tool_by_payload(
                        tool_name,
                        tool_args_final,
                        reason="orchestrator_use_tool",
                        args_auto_built=args_auto_built,
                        decision_action=action,
                    )
                    tool_traces.append(
                        self._build_tool_trace(
                            summary="Orchestrator invoked a tool.",
                            tool_name=tool_name,
                            args=[],
                            kwargs={"tool_args": tool_args_final},
                            result=tool_result,
                        )
                    )
                    working_history = self._inject_tool_result_message(
                        working_history, tool_name, tool_result
                    )
                    tool_result_injected = True
            elif action == "create_tool":
                tool_agent_traced = True
                selected_tool = self._maybe_generate_tool_for_query(
                    task_query, working_history, allow_reuse=False
                )
                if selected_tool is None:
                    tool_error = "tool generation failed"
                    self._trace("tool_agent_result", "ERROR tool generation failed")
                    _record_tool_error("create_tool", tool_error)
                    self._toolgen_debug_event(
                        "INVARIANT_BROKEN",
                        reason="create_tool_no_result",
                        tool_error=tool_error,
                    )
                    tool_result = ToolResult.failure(tool_error)
                    working_history = self._inject_tool_result_message(
                        working_history, "create_tool", tool_result
                    )
                    tool_result_injected = True
                else:
                    if not self._registry.has_tool(selected_tool.name):
                        tool_error = f"registry missing {selected_tool.name}"
                        self._trace("tool_agent_result", f"ERROR registry missing {selected_tool.name}")
                        _record_tool_error("register", tool_error)
                        tool_result = ToolResult.failure(tool_error)
                        working_history = self._inject_tool_result_message(
                            working_history, selected_tool.name, tool_result
                        )
                        tool_result_injected = True
                        selected_tool = None
                    else:
                        tool_path = (
                            self._registry._get_tool_path(selected_tool.name)
                            if hasattr(self._registry, "_get_tool_path")
                            else ""
                        )
                        if tool_path and not os.path.exists(tool_path):
                            tool_error = f"missing file {tool_path}"
                            self._trace("tool_agent_result", f"ERROR missing file {tool_path}")
                            _record_tool_error("persist", tool_error)
                            tool_result = ToolResult.failure(tool_error)
                            working_history = self._inject_tool_result_message(
                                working_history, selected_tool.name, tool_result
                            )
                            tool_result_injected = True
                            selected_tool = None
                        else:
                            if tool_path:
                                self._trace("tool_saved_path", tool_path)
                            self._trace("registry_add", selected_tool.name)
        except Exception as exc:
            tool_error = f"{type(exc).__name__}: {exc}"
            _record_tool_error("tool_pipeline", tool_error, traceback.format_exc())

        candidate_output = self._get_candidate_output(working_history, task_query)

        if selected_tool is not None:
            payload = self._invoke_tool_for_query(
                selected_tool,
                task_query,
                candidate_output=candidate_output,
                reason=f"orchestrator_{action}",
                decision_action=action,
            )
            if payload is not None:
                tool_result, tool_args, tool_kwargs = payload
                tool_traces.append(
                    self._build_tool_trace(
                        summary="Orchestrator invoked a tool.",
                        tool_name=selected_tool.name,
                        args=tool_args,
                        kwargs=tool_kwargs,
                        result=tool_result,
                    )
                )
                working_history = self._inject_tool_result_message(
                    working_history, selected_tool.name, tool_result
                )
                tool_result_injected = True
        if not tool_agent_traced:
            self._trace("tool_agent_input", "none")
            self._trace("tool_agent_result", "none")

        repeat_info = self._detect_repeated_env_action(working_history)
        last_action_info = self._get_last_action_info(working_history)
        task_intent = self._infer_task_intent(task_query)
        working_history = self._inject_solver_context_message(
            working_history,
            task_query=task_query,
            tool_result_injected=tool_result_injected,
            last_action_info=last_action_info,
            task_intent=task_intent,
            repeat_info=repeat_info,
        )

        for _ in range(3):
            solver_payload = {
                "system_prompt": self._solver_prompt_no_tools(),
                "history": self._toolgen_render_history(
                    working_history, max_chars_per_item=None
                ),
                "tool_error": tool_error,
                "tool_result_injected": tool_result_injected,
            }
            self._trace("solver_input", json.dumps(solver_payload, ensure_ascii=True, default=str))
            solver_response = self._solver_inference_with_retry(
                working_history, system_prompt=self._solver_prompt_no_tools()
            )
            content = getattr(solver_response, "content", "") or ""
            if self._contains_internal_tool(content):
                self._trace("solver_result", content)
                working_history = self._safe_inject(working_history, solver_response)
                working_history = self._safe_inject(
                    working_history,
                    ChatHistoryItem(
                        role=Role.USER,
                        content=(
                            "Tool use is handled by the orchestrator. "
                            "Do NOT call tools. Respond with the final answer only."
                        ),
                    ),
                )
                continue
            content = _ensure_solver_output(content)
            context_key = self._solver_context_key(working_history)
            if content == self._last_solver_output and context_key == self._last_solver_context_key:
                self._solver_repeat_count += 1
                if self._solver_repeat_count >= 2:
                    fallback = _fallback_response()
                    self._trace("solver_result", fallback)
                    self._flush_tool_traces(tool_traces, fallback)
                    return ChatHistoryItem(role=Role.AGENT, content=fallback)
                working_history = self._safe_inject(working_history, solver_response)
                working_history = self._safe_inject(
                    working_history,
                    ChatHistoryItem(
                        role=Role.USER,
                        content=(
                            "Stop repeating; use latest TOOL_RESULT/env output and produce final answer now."
                        ),
                    ),
                )
                continue
            self._solver_repeat_count = 0
            self._last_solver_output = content
            self._last_solver_context_key = context_key
            self._trace("solver_result", content)
            self._flush_tool_traces(tool_traces, content)
            return ChatHistoryItem(role=Role.AGENT, content=content)

        fallback = _fallback_response()
        self._trace("solver_result", fallback)
        self._flush_tool_traces(tool_traces, fallback)
        return ChatHistoryItem(role=Role.AGENT, content=fallback)

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

        if self._use_orchestrator:
            return self._orchestrated_inference(chat_history)
        if self._use_packaged_agent and self._packaged_shim is not None:
            return self._packaged_shim.inference(chat_history)

        working_history = self._clone_history(self._prune_for_current_task(chat_history))
        self._tool_invoked_in_last_inference = False

        # Pin the original task text ONCE (before tool-result user messages start dominating)
        user_items = [item for item in self._history_items(working_history) if item.role == Role.USER]
        if len(user_items) >= 2:
            original_query = user_items[1].content
        else:
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
                        working_history = self._inject_tool_result_message(
                            working_history, preprocess_tool.name, preprocess_result
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

                    working_history = self._inject_tool_result_message(
                        working_history, tool_name, tool_result
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
