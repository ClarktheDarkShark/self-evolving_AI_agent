import hashlib
import json
import os
import re
import sys
from typing import Any, Mapping, Optional, Sequence

from src.typings import ChatHistory, ChatHistoryItem, Role
from .controller_prompts import (
    ARCHETYPE_REGISTRY,
    EXECUTION_STYLE_VOCAB,
    PREFERRED_TOOL_MODE_VOCAB,
    STRATEGY_FAMILY_VOCAB,
)


class ControllerOrchestratorMixin:
    _INTERNAL_TOOL_RE = re.compile(
        r"<internal_tool\s+name=\"(?P<name>[^\"]+)\">(?P<body>[\s\S]*?)</internal_tool>"
    )
    _RELATIONS_OBS_PREFIX = "Observation: ["
    _PLAN_STEP_PREFIX_RE = re.compile(r"^(?P<prefix>(?:Step\s*\d+[:.)-]?|\d+[.)]))\s*(?P<body>.*)$", re.IGNORECASE)
    _PLAN_HELPER_RE = re.compile(
        r"\b(?P<helper>kg_utils\.[A-Za-z_]+|actions_spec\.get\([\"']count[\"']\)|count)\b"
    )
    _PLAN_ALIAS_RE = re.compile(r"(?P<alias>\$[A-Z0-9_]+)")
    _SEMANTIC_STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "then",
        "than",
        "same",
        "such",
        "each",
        "only",
        "must",
        "should",
        "would",
        "using",
        "use",
        "used",
        "after",
        "before",
        "through",
        "their",
        "them",
        "they",
        "which",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "while",
        "does",
        "did",
        "have",
        "has",
        "had",
        "been",
        "being",
        "also",
        "very",
        "most",
        "more",
        "less",
        "least",
        "many",
        "much",
        "your",
        "task",
        "tasks",
        "question",
        "entities",
        "entity",
        "input",
        "output",
        "schema",
        "required",
        "optional",
        "property",
        "properties",
        "keys",
        "payload",
        "dict",
        "trace",
        "state",
        "asked",
        "text",
        "tool",
        "tools",
        "macro",
        "generated",
        "run",
        "contract",
        "guard",
        "execution",
        "execute",
        "result",
        "results",
        "return",
        "returns",
        "returned",
        "variable",
        "variables",
        "vars",
        "final",
        "observation",
        "observations",
        "helper",
        "helpers",
        "runtime",
        "server",
        "side",
        "knowledge",
        "graph",
        "kg",
        "utils",
        "actions",
        "spec",
        "resolve",
        "resolved",
        "semantic",
        "filter",
        "filters",
        "cross",
        "intersect",
        "intersection",
        "walk",
        "target",
        "targets",
        "concept",
        "concepts",
        "domain",
        "hints",
        "attribute",
        "attributes",
        "value",
        "values",
        "plan",
        "topological",
        "execution_plan",
        "execution",
        "step",
        "steps",
        "strict",
        "bounded",
        "completion",
        "recovery",
        "policy",
        "compose",
        "composite",
        "topology",
        "single",
        "multi",
        "hop",
        "macro_generated_tool",
        "generated_tool",
        "count",
        "counter",
        "counting",
        "intersector",
        "finder",
        "superlative",
        "anchor",
        "materialize",
        "materialized",
        "ids",
        "id",
        "number",
        "none",
    }

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

    @classmethod
    def _semantic_tokens_from_value(cls, value: Any) -> set[str]:
        tokens: set[str] = set()
        if value is None:
            return tokens
        if isinstance(value, Mapping):
            for inner in value.values():
                tokens.update(cls._semantic_tokens_from_value(inner))
            return tokens
        if isinstance(value, (list, tuple, set)):
            for inner in value:
                tokens.update(cls._semantic_tokens_from_value(inner))
            return tokens
        text = str(value or "").strip().lower()
        if not text:
            return tokens
        for raw in re.findall(r"[a-z0-9_./-]+", text):
            for token in re.split(r"[^a-z0-9]+", raw):
                token = token.strip()
                if len(token) < 3 or token.isdigit():
                    continue
                if token in cls._SEMANTIC_STOPWORDS:
                    continue
                tokens.add(token)
        return tokens

    def _request_semantic_tokens(
        self,
        *,
        query_text: Optional[str] = None,
        tool_plan: Optional[Mapping[str, Any]] = None,
    ) -> set[str]:
        tokens = self._semantic_tokens_from_value(query_text)
        if isinstance(tool_plan, Mapping):
            for key in (
                "target_concept",
                "entity_target_concepts",
                "attribute_target_concept",
                "intermediate_target_concepts",
                "topological_execution_plan",
                "composite_topology",
            ):
                tokens.update(self._semantic_tokens_from_value(tool_plan.get(key)))
        return tokens

    def _tool_specific_semantic_tokens(self, tool: Any) -> set[str]:
        text_parts = [
            getattr(tool, "name", "") or "",
            getattr(tool, "docstring", "") or "",
            getattr(tool, "description", "") or "",
        ]
        return self._semantic_tokens_from_value(" ".join(text_parts))

    def _expected_request_output_form(
        self,
        *,
        query_text: Optional[str] = None,
        tool_plan: Optional[Mapping[str, Any]] = None,
    ) -> str:
        desired_archetype = ""
        if isinstance(tool_plan, Mapping):
            desired_archetype = str(tool_plan.get("target_archetype") or "").strip().upper()
        if desired_archetype in {"COUNTER", "COUNTING_INTERSECTOR"}:
            return "count_variable"
        if desired_archetype:
            return "pointer_variable"
        lowered = str(query_text or "").lower()
        if any(kw in lowered for kw in ("how many", "count", "number of", "total")):
            return "count_variable"
        return "pointer_variable" if lowered else ""

    def _tool_output_form(self, tool: Any, *, archetype_label: str) -> str:
        if archetype_label in {"COUNTER", "COUNTING_INTERSECTOR"}:
            return "count_variable"
        if archetype_label and archetype_label != "UNKNOWN":
            return "pointer_variable"
        text = " ".join(
            [
                getattr(tool, "name", "") or "",
                getattr(tool, "docstring", "") or "",
                getattr(tool, "description", "") or "",
            ]
        ).lower()
        if any(kw in text for kw in ("count variable", "count(", "counter", "number of", "how many")):
            return "count_variable"
        return "pointer_variable" if text else ""

    def _is_tool_output_form_compatible(
        self,
        tool: Any,
        *,
        archetype_label: str,
        query_text: Optional[str] = None,
        tool_plan: Optional[Mapping[str, Any]] = None,
    ) -> tuple[bool, str, str, str]:
        request_output_form = self._expected_request_output_form(
            query_text=query_text, tool_plan=tool_plan
        )
        tool_output_form = self._tool_output_form(tool, archetype_label=archetype_label)
        if request_output_form and tool_output_form and request_output_form != tool_output_form:
            return (
                False,
                f"output_form_mismatch:{tool_output_form}->{request_output_form}",
                request_output_form,
                tool_output_form,
            )
        return True, "ok", request_output_form, tool_output_form

    def _is_tool_semantically_compatible(
        self,
        tool: Any,
        *,
        archetype_label: str,
        query_text: Optional[str] = None,
        tool_plan: Optional[Mapping[str, Any]] = None,
    ) -> tuple[bool, str]:
        desired_archetype = ""
        if isinstance(tool_plan, Mapping):
            desired_archetype = str(tool_plan.get("target_archetype") or "").strip().upper()
        if desired_archetype:
            # Fail closed — UNKNOWN archetype is not reusable when the request
            # specifies a concrete target_archetype.
            if archetype_label in {"", "UNKNOWN"}:
                return False, "archetype_unknown_fail_closed"
            if desired_archetype != archetype_label:
                return False, f"archetype_mismatch:{archetype_label}->{desired_archetype}"
        elif isinstance(tool_plan, Mapping):
            # A tool_plan exists but carries no target_archetype (e.g. use_tool response
            # that omits the field).  An UNKNOWN tool is still unsafe to reuse — we cannot
            # verify fit without both sides of the archetype comparison.
            if archetype_label in {"", "UNKNOWN"}:
                return False, "archetype_unknown_plan_no_archetype"

        request_tokens = self._request_semantic_tokens(query_text=query_text, tool_plan=tool_plan)
        tool_tokens = self._tool_specific_semantic_tokens(tool)
        if request_tokens and tool_tokens and not (request_tokens & tool_tokens):
            return False, f"semantic_mismatch:{sorted(tool_tokens)[:6]}"
        return True, "ok"

    def _normalize_tool_invoker_response(
        self, text: str
    ) -> tuple[str, Optional[Mapping[str, Any]], Optional[str], bool]:
        candidate = (text or "").strip()
        if not candidate:
            return "", None, "empty_response", False

        wrapper_stripped = False
        wrapper_body = self._extract_internal_tool_body(candidate)
        if wrapper_body is not None:
            wrapper_stripped = True
            candidate = wrapper_body.strip()

        extracted_json = self._extract_first_json_object(candidate)
        if extracted_json:
            candidate = extracted_json.strip()

        payload = self._parse_orchestrator_payload(candidate)
        if isinstance(payload, Mapping):
            return candidate, payload, None, wrapper_stripped

        parse_error: Optional[str] = None
        try:
            json.loads(candidate)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        if parse_error is None and not extracted_json and ("<" in candidate or ">" in candidate):
            parse_error = "forbidden_wrapper_chars"
        return candidate, None, parse_error, wrapper_stripped

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

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            return [text]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items: list[str] = []
            for item in value:
                token = str(item).strip()
                if token:
                    items.append(token)
            return items
        token = str(value).strip()
        return [token] if token else []

    def _normalize_plan_steps(self, value: Any) -> list[str]:
        def _normalize_item(item: Any) -> str:
            return self._sanitize_plan_step_text(str(item).strip())

        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_normalize_item(v) for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [_normalize_item(v) for v in parsed if str(v).strip()]
            except Exception:
                pass
        lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return [_normalize_item(line) for line in lines]
        parts = re.split(r"(?=(?:Step\s*\d+[:.)-]?|\d+[.)-]\s+))", text)
        steps = [_normalize_item(part) for part in parts if part and part.strip()]
        return steps or [_normalize_item(text)]

    def _sanitize_plan_step_text(self, text: str) -> str:
        step = str(text or "").strip()
        if not step:
            return ""

        prefix = ""
        body = step
        prefix_match = self._PLAN_STEP_PREFIX_RE.match(step)
        if prefix_match:
            prefix = str(prefix_match.group("prefix") or "").strip()
            body = str(prefix_match.group("body") or "").strip()

        helper_match = self._PLAN_HELPER_RE.search(body)
        helper_name = str(helper_match.group("helper") or "").strip() if helper_match else ""
        alias_source = body.split("->", 1)[1] if "->" in body else ""
        alias_match = self._PLAN_ALIAS_RE.search(alias_source)
        alias = str(alias_match.group("alias") or "").strip() if alias_match else ""
        description = ""
        desc_match = re.search(r"\((?P<desc>[^()]*)\)\s*$", alias_source)
        if desc_match:
            description = str(desc_match.group("desc") or "").strip()

        pythonish = bool(helper_name) and (
            "(" in body
            or "->" in body
            or "actions_spec" in body
            or re.search(
                r"\b(?:domain_hints|max_k|max_calls|attribute|asked_for|max_type_candidates|variable_list|base_var|target_concept)\s*=",
                body,
            )
        )
        if not pythonish:
            return step

        sentence = f"Use {helper_name}"
        if description:
            sentence += f" to {description.rstrip('.')}"
        if alias:
            sentence += f" and store the result as {alias}"
        sentence = sentence.rstrip(".") + "."
        return f"{prefix} {sentence}".strip() if prefix else sentence

    def _build_tool_plan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = payload
        nested_plan = payload.get("tool_plan")
        if isinstance(nested_plan, Mapping):
            source = nested_plan

        def _normalize_choice(value: Any, allowed: tuple[str, ...]) -> str:
            text = str(value or "").strip()
            if text in allowed:
                return text
            return ""

        def _default_execution_style(
            target_archetype: str,
            entity_count: int,
            *,
            has_attribute_target: bool,
            ambiguous_entity_types: bool,
        ) -> str:
            archetype = target_archetype.strip().upper()
            if archetype == "SUPERLATIVE_FINDER":
                return "attribute_mapping_first"
            if archetype in {"ATTRIBUTE_INTERSECTOR", "ATTRIBUTE_EXTRACTOR"}:
                return "attribute_mapping_first"
            if archetype == "SHARED_TRAIT_PIVOT":
                return "probe_then_commit" if ambiguous_entity_types else "diagnostic_first"
            if archetype == "COUNTING_INTERSECTOR":
                return "relation_first" if entity_count >= 2 else "walk_first"
            if archetype in {"INTERSECTOR", "UNION_AGGREGATOR", "EXCLUSION_FILTER"}:
                return "walk_first"
            if has_attribute_target:
                return "attribute_mapping_first"
            return "walk_first"

        def _default_preferred_tool_mode(execution_style: str) -> str:
            if execution_style == "diagnostic_first":
                return "diagnostic_probe"
            if execution_style == "partial_value_first":
                return "progress_tool"
            return "full_solve"

        def _infer_primary_strategy(
            execution_style: str,
            preferred_tool_mode: str,
            target_archetype: str,
        ) -> str:
            archetype = target_archetype.strip().upper()
            if (
                preferred_tool_mode == "diagnostic_probe"
                or execution_style == "diagnostic_first"
            ):
                return "diagnostic_probe"
            if preferred_tool_mode == "progress_tool":
                if execution_style == "attribute_mapping_first":
                    return "attribute_preparer"
                if execution_style == "partial_value_first":
                    return "set_builder"
            if execution_style in {"walk_first", "relation_first", "probe_then_commit"}:
                return execution_style
            if execution_style == "attribute_mapping_first":
                return "attribute_preparer"
            if archetype == "SHARED_TRAIT_PIVOT":
                return "shared_trait_pivot"
            if archetype == "SUPERLATIVE_FINDER":
                return "superlative_finder"
            return "generic_macro"

        def _default_fallback_strategies(
            target_archetype: str,
            execution_style: str,
        ) -> list[str]:
            archetype = target_archetype.strip().upper()
            if archetype == "COUNTING_INTERSECTOR":
                if execution_style == "walk_first":
                    ordered = ["relation_first", "probe_then_commit", "set_builder"]
                elif execution_style == "probe_then_commit":
                    ordered = ["set_builder", "intersector_counter", "relation_first"]
                else:
                    ordered = [
                        "probe_then_commit",
                        "set_builder",
                        "intersector_counter",
                    ]
            elif archetype == "SUPERLATIVE_FINDER":
                ordered = ["attribute_preparer", "diagnostic_probe", "walk_first"]
            elif archetype == "SHARED_TRAIT_PIVOT":
                ordered = ["diagnostic_probe", "probe_then_commit", "walk_first"]
            elif execution_style == "diagnostic_first":
                ordered = ["probe_then_commit", "partial_handoff", "generic_macro"]
            else:
                ordered = ["probe_then_commit", "diagnostic_probe", "generic_macro"]
            unique: list[str] = []
            for item in ordered:
                if item in STRATEGY_FAMILY_VOCAB and item not in unique:
                    unique.append(item)
            return unique[:3]

        def _normalize_fallback_strategies(
            primary_strategy: str,
            fallback_values: Any,
            *,
            default_values: Sequence[str],
        ) -> list[str]:
            values = self._normalize_string_list(fallback_values)
            values = [item for item in values if item in STRATEGY_FAMILY_VOCAB]
            if not values:
                values = list(default_values)
            normalized: list[str] = []
            for item in values:
                if item == primary_strategy:
                    continue
                if item in STRATEGY_FAMILY_VOCAB and item not in normalized:
                    normalized.append(item)
            return normalized[:3]

        def _pick(key: str) -> Any:
            if key in source:
                return source.get(key)
            return payload.get(key)

        raw_plan = _pick("topological_execution_plan")
        plan_steps = self._normalize_plan_steps(raw_plan)
        plan_text = "\n".join(plan_steps) if plan_steps else (
            str(raw_plan).strip() if raw_plan is not None else ""
        )
        tool_plan: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"action", "tool_name", "tool_type", "reason"}:
                continue
            if key == "tool_plan":
                continue
            tool_plan[key] = value
        if isinstance(source, Mapping):
            for key, value in source.items():
                tool_plan[key] = value
        tool_plan.update(
            {
                "target_concept": str(_pick("target_concept") or "").strip(),
                "entity_target_concepts": self._normalize_string_list(
                    _pick("entity_target_concepts")
                ),
                "attribute_target_concept": str(
                    _pick("attribute_target_concept") or ""
                ).strip(),
                "intermediate_target_concepts": self._normalize_string_list(
                    _pick("intermediate_target_concepts")
                ),
                # Preserve the raw value for debugging, but expose the
                # sanitized prose-only plan to downstream ToolGen consumers.
                "topological_execution_plan": plan_steps,
                "topological_execution_plan_raw": raw_plan,
                "topological_execution_plan_steps": plan_steps,
                "topological_execution_plan_text": plan_text,
                "composite_topology": self._normalize_string_list(
                    _pick("composite_topology")
                ),
                "recovery_policy": str(_pick("recovery_policy") or "").strip(),
                "target_archetype": str(_pick("target_archetype") or "").strip(),
            }
        )
        entity_count = len(tool_plan.get("entities") or [])
        has_attribute_target = bool(tool_plan.get("attribute_target_concept"))
        entity_target_concepts = tool_plan.get("entity_target_concepts") or []
        ambiguous_entity_types = not any(
            isinstance(item, str) and item.strip() for item in entity_target_concepts
        )
        execution_style = _normalize_choice(
            _pick("execution_style"), EXECUTION_STYLE_VOCAB
        ) or _default_execution_style(
            tool_plan.get("target_archetype", ""),
            entity_count,
            has_attribute_target=has_attribute_target,
            ambiguous_entity_types=ambiguous_entity_types,
        )
        preferred_tool_mode = _normalize_choice(
            _pick("preferred_tool_mode"), PREFERRED_TOOL_MODE_VOCAB
        ) or _default_preferred_tool_mode(execution_style)
        primary_strategy = _infer_primary_strategy(
            execution_style,
            preferred_tool_mode,
            tool_plan.get("target_archetype", ""),
        )
        fallback_strategies = _normalize_fallback_strategies(
            primary_strategy,
            _pick("fallback_strategies"),
            default_values=_default_fallback_strategies(
                tool_plan.get("target_archetype", ""),
                execution_style,
            ),
        )
        tool_plan["execution_style"] = execution_style
        tool_plan["preferred_tool_mode"] = preferred_tool_mode
        tool_plan["fallback_strategies"] = fallback_strategies[:3]
        # Reinforce count-variable semantics for counting archetypes so ToolGen
        # does not return a numeric answer as final_variable.  The count primitive
        # creates a new Variable ID — that ID is what must be returned.
        _archetype_upper = str(tool_plan.get("target_archetype") or "").strip().upper()
        if _archetype_upper in {"COUNTER", "COUNTING_INTERSECTOR"}:
            tool_plan["final_variable_is_count_variable"] = True
            tool_plan["count_variable_note"] = (
                "CRITICAL: final_variable MUST be the Variable ID returned by the count "
                "primitive (e.g. '#5'), NOT a numeric integer or string number. "
                "Call count(set_var), extract the returned Variable ID with "
                "kg_utils.extract_var_ids(), and return that ID as final_variable."
            )
        domain_hints = _pick("domain_hints")
        if domain_hints is not None:
            tool_plan["domain_hints"] = self._normalize_string_list(domain_hints)
        return tool_plan
    # ------------------------------------------------------------------
    # Escape hatch baseline tool (Phase 4)
    # ------------------------------------------------------------------
    _REQUEST_NEW_TOOL_ENTRY = {
        "name": "request_new_tool",
        "signature": "request_new_tool(tool_type, reason, target_archetype)",
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
            "required": ["tool_type", "reason", "target_archetype"],
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
                "target_archetype": {
                    "type": "string",
                    "enum": list(ARCHETYPE_REGISTRY.keys()),
                },
            },
        },
        "required_keys": ["tool_type", "reason", "target_archetype"],
        "optional_keys": [],
        "property_types": {
            "tool_type": "string",
            "reason": "string",
            "target_archetype": "string",
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
    # Archetype arity rules: minimum entity count required per archetype
    # ------------------------------------------------------------------
    _ARCHETYPE_MIN_ENTITIES: dict[str, int] = {
        "ATTRIBUTE_INTERSECTOR": 3,   # 2 subject entities + 1 attribute literal
        "INTERSECTOR": 2,
        "COUNTING_INTERSECTOR": 2,
        "EXCLUSION_FILTER": 2,
        "UNION_AGGREGATOR": 2,
        "SHARED_TRAIT_PIVOT": 2,
        "COUNTER": 1,
        "SUPERLATIVE_FINDER": 1,
        "MULTI_HOP_CHAIN": 1,
        "ATTRIBUTE_EXTRACTOR": 1,
    }

    @staticmethod
    def _get_archetype_min_entities(archetype_str: str) -> int:
        """Return the minimum entity count required to invoke an archetype tool."""
        key = (archetype_str or "").strip().upper()
        return ControllerOrchestratorMixin._ARCHETYPE_MIN_ENTITIES.get(key, 1)

    @staticmethod
    def _extract_query_entities(query: str) -> list[str]:
        """Parse the entity list from a task query string."""
        text = (query or "").strip()
        match = re.search(r"Entities\s*:\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1)
        parts = re.findall(r"'([^']+)'|\"([^\"]+)\"|([^,\s][^,]*)", raw)
        entities = [
            token.strip()
            for part in parts
            for token in [next((x for x in part if x and x.strip()), None)]
            if token
        ]
        return entities

    @staticmethod
    def _extract_query_entity_count(query: str) -> Optional[int]:
        """Parse the entity list from a task query string and return its length.

        Returns None when no entity list is detected (meaning: do not filter).
        """
        entities = ControllerOrchestratorMixin._extract_query_entities(query)
        if entities:
            return len(entities)
        return None

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
                if new_tool.get("error") == "flawed_plan_requires_reorchestration":
                    self._trace("escape_hatch_failed", "flawed_plan_requires_reorchestration")
                    return {
                        "success": False,
                        "tool_name": None,
                        "observation": str(
                            new_tool.get("observation")
                            or (
                                "Observation: ToolGen aborted because the Orchestrator plan is flawed. "
                                "Re-orchestrate with a corrected prose-only plan before requesting a new tool."
                            )
                        ),
                    }
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
        self,
        *,
        query_text: Optional[str] = None,
        tool_plan: Optional[Mapping[str, Any]] = None,
        top_k: int = 5,
        query_entity_count: Optional[int] = None,
        include_control_entries: bool = True,
        force_include_tool_name: Optional[str] = None,
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

        def _extract_tool_archetype(tool) -> str:
            """Extract archetype label from a tool's input_schema or description/docstring."""
            # 1. Check explicit metadata first.
            # Phase 1: also check tool_type as a fallback persistence field — some older
            # tools may have had archetype stored there before input_schema injection.
            for attr_name in ("target_archetype", "archetype", "tool_type"):
                try:
                    attr_val = getattr(tool, attr_name, None)
                except Exception:
                    attr_val = None
                if attr_val:
                    candidate = str(attr_val).strip().upper()
                    if candidate in ARCHETYPE_REGISTRY:
                        return candidate

            # 2. Check input_schema.properties.target_archetype for const/default/enum
            schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
            props = schema.get("properties", {})
            arch_prop = props.get("target_archetype", {}) if isinstance(props, dict) else {}
            if isinstance(arch_prop, dict):
                const_val = arch_prop.get("const")
                if const_val and str(const_val).upper() in ARCHETYPE_REGISTRY:
                    return str(const_val).upper()
                enum_vals = arch_prop.get("enum")
                if isinstance(enum_vals, list) and len(enum_vals) == 1:
                    return str(enum_vals[0]).upper()
                default_val = arch_prop.get("default")
                if default_val:
                    return str(default_val).upper()
            # 3. Fall back to registry key scan in name/description/docstring
            scan_sources = [
                getattr(tool, "name", "") or "",
                tool.docstring or "",
                tool.description or "",
            ]
            for src in scan_sources:
                src_upper = src.upper()
                # Prefer exact token boundaries where possible.
                for arch_key in ARCHETYPE_REGISTRY.keys():
                    pattern = r"(?<![A-Z0-9_])" + re.escape(arch_key) + r"(?![A-Z0-9_])"
                    if re.search(pattern, src_upper):
                        return arch_key
            # Include name in second-pass: generated tools embed the archetype key
            # in their name (e.g. "attribute_intersector_macro_generated_tool") but
            # the boundary regex above rejects it when "_" follows the key.
            for src in (
                getattr(tool, "name", "") or "",
                tool.docstring or "",
                tool.description or "",
            ):
                src_upper = src.upper()
                for arch_key in ARCHETYPE_REGISTRY.keys():
                    if arch_key in src_upper:
                        return arch_key
            return "UNKNOWN"

        # Semantic count-mode gate: never show COUNTER/COUNTING_INTERSECTOR tools
        # to the LLM for entity-retrieval questions. Python-level filter prevents
        # catalog noise from causing wrong archetype selection.
        _count_kws = {"how many", "count", "number of", "total"}
        is_count_query = any(kw in (query_text or "").lower() for kw in _count_kws)
        request_archetype = (
            str(tool_plan.get("target_archetype") or "").strip().upper()
            if isinstance(tool_plan, Mapping)
            else ""
        )
        request_output_form = self._expected_request_output_form(
            query_text=query_text, tool_plan=tool_plan
        )

        def _log_reuse_gate(
            *,
            tool_name: str,
            tool_archetype: str,
            gate: str,
            compatible: bool,
            reason: Optional[str] = None,
            tool_output_form: str = "",
        ) -> None:
            try:
                self._append_tool_value_trace(
                    "tool_reuse_gate_decision",
                    tool_name=tool_name,
                    tool_archetype=tool_archetype,
                    request_archetype=request_archetype,
                    gate=gate,
                    compatible=compatible,
                    reason=reason,
                    query_entity_count=query_entity_count,
                    request_output_form=request_output_form or None,
                    tool_output_form=tool_output_form or None,
                )
            except Exception:
                pass
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_compatibility_decision",
                        "tool_name": tool_name,
                        "tool_archetype": tool_archetype,
                        "request_archetype": request_archetype or None,
                        "gate": gate,
                        "compatible": compatible,
                        "reason": reason,
                        "query_entity_count": query_entity_count,
                        "request_output_form": request_output_form or None,
                        "tool_output_form": tool_output_form or None,
                        "environment_label": self._resolved_environment_label(),
                    }
                )
            except Exception:
                pass

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
            archetype_label = _extract_tool_archetype(t)
            # A tool just generated for this exact task is always included — all
            # semantic/archetype/output-form gates are bypassed for it.  Hard
            # safety gates (validate_tool_code, spec alignment) already ran at
            # registration time; these gates only guard cross-task reuse.
            _is_forced = bool(force_include_tool_name and t.name == force_include_tool_name)
            if _is_forced:
                print(
                    f"[FORCE_INCLUDE] '{t.name}' is the just-generated tool — "
                    f"forced same-turn invocation path engaged, bypassing compatibility gates."
                )
                try:
                    self._append_generated_tools_log(
                        {
                            "event": "tool_invoker_compatibility_decision",
                            "tool_name": t.name,
                            "tool_archetype": archetype_label,
                            "gate": "force_bypass",
                            "compatible": True,
                            "reason": "forced_same_turn_invocation",
                            "environment_label": self._resolved_environment_label(),
                        }
                    )
                except Exception:
                    pass
            # Arity gate: skip tools whose archetype doesn't match the query's entity count.
            # ATTRIBUTE_INTERSECTOR requires exactly 3 entities (2 subjects + 1 attribute literal).
            # All other archetypes use a >= min threshold.
            if query_entity_count is not None:
                min_ents = self._get_archetype_min_entities(archetype_label)
                _exact_required = archetype_label == "ATTRIBUTE_INTERSECTOR"
                _arity_mismatch = (
                    (_exact_required and query_entity_count != 3)
                    or (not _exact_required and query_entity_count < min_ents)
                )
                if _arity_mismatch:
                    _needed_str = "==3" if _exact_required else f">={min_ents}"
                    print(
                        f"[ARITY_GATE] {'Bypassing (forced)' if _is_forced else 'Skipping'} '{t.name}' "
                        f"(archetype={archetype_label}, needs={_needed_str}, query has {query_entity_count})"
                    )
                    _log_reuse_gate(
                        tool_name=t.name,
                        tool_archetype=archetype_label,
                        gate="arity_gate",
                        compatible=_is_forced,
                        reason=f"entity_count_mismatch:{query_entity_count}:{_needed_str}",
                    )
                    if not _is_forced:
                        continue
            # Count-mode gate: COUNTER and COUNTING_INTERSECTOR are only relevant
            # when the query explicitly asks "how many / count / total".
            if archetype_label in {"COUNTER", "COUNTING_INTERSECTOR"} and not is_count_query:
                print(
                    f"[COUNT_GATE] {'Bypassing (forced)' if _is_forced else 'Skipping'} '{t.name}' "
                    f"(archetype={archetype_label}, not a count query)"
                )
                _log_reuse_gate(
                    tool_name=t.name,
                    tool_archetype=archetype_label,
                    gate="count_gate",
                    compatible=_is_forced,
                    reason="count_tool_for_non_count_query",
                    tool_output_form=self._tool_output_form(
                        t, archetype_label=archetype_label
                    ),
                )
                if not _is_forced:
                    continue
            output_compatible, output_reason, _, tool_output_form = (
                self._is_tool_output_form_compatible(
                    t,
                    archetype_label=archetype_label,
                    query_text=query_text,
                    tool_plan=tool_plan,
                )
            )
            if not output_compatible:
                print(
                    f"[OUTPUT_FORM_GATE] {'Bypassing (forced)' if _is_forced else 'Skipping'} '{t.name}' "
                    f"(archetype={archetype_label}, reason={output_reason})"
                )
                _log_reuse_gate(
                    tool_name=t.name,
                    tool_archetype=archetype_label,
                    gate="output_form_gate",
                    compatible=_is_forced,
                    reason=output_reason,
                    tool_output_form=tool_output_form,
                )
                if not _is_forced:
                    continue
            compatible, compatibility_reason = self._is_tool_semantically_compatible(
                t,
                archetype_label=archetype_label,
                query_text=query_text,
                tool_plan=tool_plan,
            )
            if not compatible:
                print(
                    f"[SEMANTIC_GATE] {'Bypassing (forced)' if _is_forced else 'Skipping'} '{t.name}' "
                    f"(archetype={archetype_label}, reason={compatibility_reason})"
                )
                _log_reuse_gate(
                    tool_name=t.name,
                    tool_archetype=archetype_label,
                    gate="semantic_gate",
                    compatible=_is_forced,
                    reason=compatibility_reason,
                    tool_output_form=tool_output_form,
                )
                if not _is_forced:
                    continue
            _log_reuse_gate(
                tool_name=t.name,
                tool_archetype=archetype_label,
                gate="semantic_gate",
                compatible=True,
                tool_output_form=tool_output_form,
            )
            base_docstring = self._format_orchestrator_docstring(t)
            docstring_clean = base_docstring.strip()
            if docstring_clean.startswith("[ARCHETYPE:"):
                docstring_with_archetype = docstring_clean
            else:
                docstring_with_archetype = f"[ARCHETYPE: {archetype_label}] {docstring_clean}"
            compact.append(
                {
                    "name": t.name,
                    "archetype": archetype_label,
                    "signature": t.signature,
                    "docstring": docstring_with_archetype,
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
        # Light reliability bias: within the candidate window, reorder so tools
        # with higher reliability_score and fewer negative_marks appear first.
        # The just-generated forced tool always sorts first so it is maximally
        # salient to the Invoker LLM regardless of its (zero) reliability score.
        compact.sort(
            key=lambda t: (
                0 if (force_include_tool_name and t.get("name") == force_include_tool_name) else 1,
                -(t.get("reliability_score") or 0.0),
                t.get("negative_marks") or 0,
            )
        )
        # Append the escape hatch baseline tool (Phase 4) only for
        # orchestrator/control reasoning. Invoker catalogs should contain
        # only real registry-backed tools.
        if include_control_entries:
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
            "action": "use_tool|request_new_tool|no_tool",
            "tool_name": "only if action=use_tool",
            "tool_type": "advisory|macro (only if action=request_new_tool)",
            "archetype_reasoning": "chain-of-thought: logical shape of the question (only if action=request_new_tool)",
            "target_archetype": "PURE queries only — e.g. COUNTER (only if action=request_new_tool AND single topological stage)",
            "composite_topology": "HYBRID queries only — ordered array of 2-3 archetype names, e.g. [\"INTERSECTOR\", \"SUPERLATIVE_FINDER\"] (only if action=request_new_tool AND multiple stages needed)",
            "recovery_policy": "OPTIONAL for request_new_tool — strict_plan|bounded_completion",
            "execution_style": "OPTIONAL for request_new_tool — walk_first|relation_first|probe_then_commit|partial_value_first|attribute_mapping_first|diagnostic_first",
            "preferred_tool_mode": "OPTIONAL for request_new_tool — full_solve|progress_tool|diagnostic_probe",
            "fallback_strategies": "OPTIONAL for request_new_tool — array of 1-3 alternate strategy families such as [\"relation_first\", \"probe_then_commit\"]",
            "target_concept": "for use_tool/request_new_tool — the canonical downstream category, role, or answer-type concept; prefer a singular ontology-friendly noun or type hint rather than raw plural surface text",
            "entity_target_concepts": "HARD REQUIRED for request_new_tool — array of per-entity semantic type hints parallel to entities; use credible entity-type hints, not blind copies of the raw entity strings",
            "domain_hints": "OPTIONAL broad domain/type hints kept separate from entity_target_concepts, e.g. [\"music\", \"people.profession\"]",
            "topological_execution_plan": "REQUIRED for request_new_tool — array of numbered prose-only steps naming exact helpers and $VAR aliases, with NO literal Python calls, kwargs, or dictionaries",
            "reason": "short reason (INPUT:...GOAL:... format if request_new_tool)",
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
        _sug = suggestion or {}
        tool_plan = _sug.get("tool_plan")
        if not isinstance(tool_plan, Mapping):
            tool_plan = self._build_tool_plan(_sug)
        payload["tool_plan"] = tool_plan
        payload["tool_plan_note"] = (
            "tool_plan is the authoritative semantic specification from the "
            "Orchestrator. Select a tool and map payload fields from this plan. "
            "Do not re-derive semantics from raw task text."
        )
        # Preserve legacy passthrough keys for compatibility, but source them
        # from the canonical tool_plan so the Invoker sees one authoritative plan.
        if tool_plan.get("target_concept"):
            payload["orchestrator_target_concept"] = tool_plan.get("target_concept")
        if tool_plan.get("execution_style"):
            payload["orchestrator_execution_style"] = tool_plan.get("execution_style")
        if tool_plan.get("preferred_tool_mode"):
            payload["orchestrator_preferred_tool_mode"] = tool_plan.get(
                "preferred_tool_mode"
            )
        if tool_plan.get("fallback_strategies"):
            payload["orchestrator_fallback_strategies"] = tool_plan.get(
                "fallback_strategies"
            )
        if tool_plan.get("entity_target_concepts"):
            payload["orchestrator_entity_target_concepts"] = tool_plan.get(
                "entity_target_concepts"
            )
        if tool_plan.get("recovery_policy"):
            payload["orchestrator_recovery_policy"] = tool_plan.get("recovery_policy")
        if tool_plan.get("topological_execution_plan"):
            payload["orchestrator_topological_execution_plan"] = tool_plan.get(
                "topological_execution_plan"
            )
        if tool_plan.get("composite_topology"):
            payload["orchestrator_composite_topology"] = tool_plan.get(
                "composite_topology"
            )
        if tool_plan.get("attribute_target_concept"):
            payload["orchestrator_attribute_target_concept"] = tool_plan.get(
                "attribute_target_concept"
            )
        if tool_plan.get("intermediate_target_concepts"):
            payload["orchestrator_intermediate_target_concepts"] = tool_plan.get(
                "intermediate_target_concepts"
            )
        if tool_plan.get("domain_hints"):
            payload["orchestrator_domain_hints"] = tool_plan.get("domain_hints")
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

        # FAST-PATH DISABLED: The blind trigger-based fast path (auto-selecting the
        # first _generated_tool on size_trigger/error_trigger/derailment_trigger) caused
        # Catalog Bias — the Orchestrator LLM was never asked to evaluate archetype fit.
        # The Orchestrator LLM must always run a full evaluation so that exact archetype
        # matching (COUNTER vs INTERSECTOR vs other specialized archetypes) drives tool selection.
        # Original fast-path block commented out below for reference:
        #
        # if observation_triggers:
        #     for trigger in observation_triggers:
        #         trigger_type = trigger.get("type")
        #         if trigger_type in {"size_trigger", "error_trigger"}:
        #             filter_tool = next(...)
        #             if filter_tool:
        #                 return {"action": "use_tool", "tool_name": filter_tool["name"], ...}
        #             else:
        #                 return {"action": "request_new_tool", ...}
        #         if trigger_type == "derailment_trigger":
        #             ...high/medium severity returns...

        forced_tool_name = getattr(self, "_just_generated_tool", None)
        # Mirror to a separate attribute so the Tool Invoker can read it even
        # after _just_generated_tool is cleared below (Orchestrator consumes
        # it first; Invoker downstream needs its own copy).
        if forced_tool_name:
            setattr(self, "_forced_invoker_tool", forced_tool_name)
        else:
            # Clear any stale forced invocation from a prior turn where
            # _tool_invoker_decision was never reached (e.g., post-escape
            # action resolved to no_tool).  Without this, the attribute leaks
            # into the next Orchestrator→Invoker cycle and incorrectly forces
            # an already-expired tool name.
            setattr(self, "_forced_invoker_tool", None)
        prompt = self._orchestrator_request_prompt(
            query,
            chat_history,
            solver_recommendation=solver_recommendation,
            stagnation_count=stagnation_count,
            forced_tool_name=forced_tool_name,
        )
        # Inject trigger context so the Orchestrator LLM is aware of why it was
        # called (e.g., size explosion, error, derailment) without the backend
        # making a blind archetype-unaware decision on its behalf.
        if observation_triggers:
            trigger_summary = "; ".join(
                f"{t.get('type', '?')}: {t.get('reason', '')}"
                for t in observation_triggers
            )
            prompt = f"[SYSTEM TRIGGERS: {trigger_summary}]\n\n" + prompt
        if forced_tool_name:
            setattr(self, "_just_generated_tool", None)
        _qec = self._extract_query_entity_count(query)
        try:
            tools = self._orchestrator_compact_existing_tools(
                query_text=retrieval_query,
                query_entity_count=_qec,
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
        upgrade_goal = payload.get("upgrade_goal")
        if action == "request_new_tool":
            if not isinstance(upgrade_goal, str) or not upgrade_goal.strip():
                # Backward-compat fallback: older orchestrator prompts encode
                # the upgrade instruction directly in reason.
                upgrade_goal = str(payload.get("reason") or "")
        else:
            upgrade_goal = ""

        if isinstance(payload, dict):
            entities_val = payload.get("entities")
            if not isinstance(entities_val, list) or not any(
                isinstance(item, str) and item.strip() for item in entities_val
            ):
                extracted_entities = self._extract_query_entities(query)
                if extracted_entities:
                    payload["entities"] = extracted_entities
        tool_plan = self._build_tool_plan(payload)
        try:
            self._log_flow_event(
                "orchestrator_tool_plan",
                chat_history=chat_history,
                tool_plan=tool_plan,
            )
        except Exception:
            pass
        try:
            self._trace(
                "orchestrator_tool_plan",
                json.dumps(tool_plan, ensure_ascii=True, default=str),
            )
        except Exception:
            pass
        # Emit orchestrator_plan_trace.jsonl entry for KG-type inspection
        if action in {"request_new_tool", "use_tool"}:
            try:
                _plan_steps = tool_plan.get("topological_execution_plan") or []
                _raw_result = None  # omit full payload; plan fields below are sufficient
                self._append_orchestrator_plan_trace(
                    action=action,
                    task_name=getattr(self, "_current_task_label", None),
                    environment=getattr(self, "_resolved_environment_label", lambda: None)(),
                    tool_name=str(payload.get("tool_name") or "").strip() or None,
                    tool_type=str(payload.get("tool_type") or "").strip() or None,
                    target_archetype=tool_plan.get("target_archetype") or None,
                    composite_topology=tool_plan.get("composite_topology") or None,
                    recovery_policy=tool_plan.get("recovery_policy") or None,
                    execution_style=tool_plan.get("execution_style") or None,
                    preferred_tool_mode=tool_plan.get("preferred_tool_mode") or None,
                    fallback_strategies=tool_plan.get("fallback_strategies") or None,
                    entities=tool_plan.get("entities") or [],
                    entity_target_concepts=tool_plan.get("entity_target_concepts") or [],
                    domain_hints=tool_plan.get("domain_hints") or [],
                    target_concept=tool_plan.get("target_concept") or None,
                    intermediate_target_concepts=tool_plan.get("intermediate_target_concepts") or [],
                    attribute_target_concept=tool_plan.get("attribute_target_concept") or None,
                    topological_execution_plan=_plan_steps,
                    reason=str(payload.get("reason") or "").strip() or None,
                    is_request_new_tool=(action == "request_new_tool"),
                    upgrade_goal=str(upgrade_goal or "").strip() or None,
                )
            except Exception:
                pass

        return {
            "action": action,
            "tool_name": str(tool_name).strip() if tool_name else None,
            "reason": payload.get("reason"),
            "tool_type": payload.get("tool_type"),
            "target_archetype": tool_plan.get("target_archetype", ""),
            "composite_topology": tool_plan.get("composite_topology", []),
            "recovery_policy": tool_plan.get("recovery_policy", ""),
            "execution_style": tool_plan.get("execution_style", ""),
            "preferred_tool_mode": tool_plan.get("preferred_tool_mode", ""),
            "fallback_strategies": tool_plan.get("fallback_strategies", []),
            "entities": tool_plan.get("entities", []),
            "target_concept": tool_plan.get("target_concept", ""),
            "entity_target_concepts": tool_plan.get("entity_target_concepts", []),
            "domain_hints": tool_plan.get("domain_hints") or [],  # Phase 2: top-level passthrough
            "attribute_target_concept": tool_plan.get("attribute_target_concept", ""),
            "intermediate_target_concepts": tool_plan.get(
                "intermediate_target_concepts", []
            ),
            "upgrade_goal": upgrade_goal,
            "insufficiency": payload.get("insufficiency"),
            "needed_capabilities": payload.get("needed_capabilities"),
            "evidence": payload.get("evidence"),
            "must_differ_from_existing": payload.get("must_differ_from_existing"),
            "self_test_cases": payload.get("self_test_cases"),
            "topological_execution_plan": tool_plan.get(
                "topological_execution_plan", []
            ),
            "tool_plan": tool_plan,
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
        _qec = self._extract_query_entity_count(query)
        try:
            tools = self._orchestrator_compact_existing_tools(query_entity_count=_qec)
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
        _qec = self._extract_query_entity_count(query)
        suggestion_tool_plan = (
            self._build_tool_plan(suggestion) if isinstance(suggestion, Mapping) else {}
        )
        # _forced_invoker_tool is set by the Orchestrator immediately after
        # successful tool generation; the Invoker must use it for this turn.
        _invoker_forced_tool = getattr(self, "_forced_invoker_tool", None)
        try:
            tools = self._orchestrator_compact_existing_tools(
                query_text=query,
                tool_plan=suggestion_tool_plan,
                query_entity_count=_qec,
                include_control_entries=False,
                force_include_tool_name=_invoker_forced_tool,
            )
        except Exception:
            tools = []
        control_entries_count = 1 if getattr(self, "_REQUEST_NEW_TOOL_ENTRY", None) else 0
        tool_list_text = json.dumps(tools, ensure_ascii=True, default=str)
        invocation_trigger = (
            str((suggestion or {}).get("invocation_trigger") or "").strip()
            if isinstance(suggestion, Mapping)
            else ""
        )
        created_tool_name = (
            str((suggestion or {}).get("created_tool_name") or "").strip()
            if isinstance(suggestion, Mapping)
            else ""
        )
        allowed_tool_names = {
            str(item.get("name") or "").strip()
            for item in tools
            if isinstance(item, Mapping) and str(item.get("name") or "").strip()
        }
        if not tools:
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_result",
                        "parse_ok": True,
                        "reason": "no_compatible_tool",
                        "tool_name": None,
                        "payload_keys_count": 0,
                        "wrapper_stripped": False,
                        "environment_label": self._resolved_environment_label(),
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "no_compatible_tool"}
        original_prompt = getattr(agent, "_system_prompt", "") or ""
        agent._system_prompt = (
            original_prompt
            + "\n\nYou must ONLY select tools from the following list:\n"
            + tool_list_text
        ).strip()
        # If a tool was just generated this turn, force the Invoker to use it.
        # The Orchestrator already cleared _just_generated_tool for its own
        # prompt; _forced_invoker_tool is the parallel copy preserved for us.
        # Note: compact already included it via force_include_tool_name above.
        _forced_invoker_tool = getattr(self, "_forced_invoker_tool", None)
        if _forced_invoker_tool:
            setattr(self, "_forced_invoker_tool", None)
            print(
                f"[FORCE_INVOKE] Forced same-turn invocation path engaged for '{_forced_invoker_tool}'.",
                file=__import__("sys").stderr,
                flush=True,
            )
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_forced_invocation",
                        "forced_tool_name": _forced_invoker_tool,
                        "reason": "same_turn_generated_tool",
                        "tools_in_compact": len(tools),
                        "environment_label": self._resolved_environment_label(),
                    }
                )
            except Exception:
                pass
            agent._system_prompt = (
                f"[CRITICAL OVERRIDE] tool_name MUST be \"{_forced_invoker_tool}\". "
                f"This tool was just generated for this exact query. "
                f"You MUST use it — selecting any other tool or action is a hard failure.\n\n"
                + agent._system_prompt
            ).strip()
        try:
            self._append_generated_tools_log(
                {
                    "event": "tool_invoker_request",
                    "tool_name_suggestion": (suggestion or {}).get("tool_name"),
                    "existing_tools_count": len(tools),
                    "control_entries_excluded": control_entries_count,
                    "run_id": run_id,
                    "state_dir_basename": os.path.basename(state_dir) if state_dir else None,
                    "environment_label": self._resolved_environment_label(),
                    "invocation_trigger": invocation_trigger or None,
                    "created_tool_name": created_tool_name or None,
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
            (
                response_content,
                payload,
                parse_error,
                wrapper_stripped,
            ) = self._normalize_tool_invoker_response(raw_response_content)
            output_summary = _summarize_text(response_content or raw_response_content)
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
                        "control_entries_excluded": control_entries_count,
                        "run_id": run_id,
                        "state_dir_basename": os.path.basename(state_dir) if state_dir else None,
                        "environment_label": self._resolved_environment_label(),
                        "repair": True,
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
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
            (
                response_content,
                payload,
                parse_error,
                wrapper_stripped,
            ) = self._normalize_tool_invoker_response(repair_content)
            output_summary = _summarize_text(response_content or repair_content)
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
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
                        "target_concept_value": payload_dict.get("target_concept")
                        if isinstance(payload_dict, Mapping)
                        else None,
                        "domain_hints_value": payload_dict.get("domain_hints")
                        if isinstance(payload_dict, Mapping)
                        else None,
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
            canonical_tool_plan = {}
            if isinstance(suggestion, Mapping):
                canonical_tool_plan = self._build_tool_plan(suggestion)
            payload_dict.setdefault("tool_plan", canonical_tool_plan)
            payload_dict.setdefault("task_text", (query or "").strip())
            payload_dict.setdefault("asked_for", (query or "").strip())
            payload_dict.setdefault("run_id", run_id)
            payload_dict.setdefault("state_dir", state_dir)
            payload_dict.setdefault("trace", [])
            payload_dict.setdefault("actions_spec", actions_spec or self._available_actions_spec())
            payload_dict.setdefault(
                "entities",
                list((canonical_tool_plan or {}).get("entities") or []),
            )
            payload_dict.setdefault("env_observation", "")
            if canonical_tool_plan:
                for key in (
                    "target_concept",
                    "execution_style",
                    "preferred_tool_mode",
                    "fallback_strategies",
                    "entity_target_concepts",
                    "attribute_target_concept",
                    "intermediate_target_concepts",
                    "topological_execution_plan",
                    "composite_topology",
                    "recovery_policy",
                    "target_archetype",
                    "domain_hints",
                ):
                    if canonical_tool_plan.get(key) not in (None, "", [], {}):
                        payload_dict.setdefault(key, canonical_tool_plan.get(key))
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
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
                        "target_concept_value": payload_dict.get("target_concept")
                        if isinstance(payload_dict, Mapping)
                        else None,
                        "domain_hints_value": payload_dict.get("domain_hints")
                        if isinstance(payload_dict, Mapping)
                        else None,
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "invalid_payload"}
        normalized_tool_name = str(tool_name).strip() if tool_name else ""
        if normalized_tool_name.lower() == "none":
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_result",
                        "parse_ok": True,
                        "reason": "no_compatible_tool",
                        "tool_name": None,
                        "payload_keys_count": 0,
                        "wrapper_stripped": wrapper_stripped,
                        "environment_label": self._resolved_environment_label(),
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "no_compatible_tool"}
        if normalized_tool_name and normalized_tool_name not in allowed_tool_names:
            try:
                self._append_generated_tools_log(
                    {
                        "event": "tool_invoker_result",
                        "parse_ok": True,
                        "reason": "invalid_tool_name",
                        "tool_name": normalized_tool_name,
                        "payload_keys_count": len(payload_dict or {}),
                        "output": output_summary,
                        "wrapper_stripped": wrapper_stripped,
                        "environment_label": self._resolved_environment_label(),
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
                    }
                )
            except Exception:
                pass
            return {"tool_name": None, "payload": None, "reason": "invalid_tool_name"}
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
                        "invocation_trigger": invocation_trigger or None,
                        "created_tool_name": created_tool_name or None,
                        "target_concept_value": payload_dict.get("target_concept")
                        if isinstance(payload_dict, Mapping)
                        else None,
                        "domain_hints_value": payload_dict.get("domain_hints")
                        if isinstance(payload_dict, Mapping)
                        else None,
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
                    "invocation_trigger": invocation_trigger or None,
                    "created_tool_name": created_tool_name or None,
                    "forced_invocation": bool(_invoker_forced_tool and tool_name == _invoker_forced_tool),
                    "target_concept_value": payload_dict.get("target_concept")
                    if isinstance(payload_dict, Mapping)
                    else None,
                    "domain_hints_value": payload_dict.get("domain_hints")
                    if isinstance(payload_dict, Mapping)
                    else None,
                }
            )
        except Exception:
            pass
        return {
            "tool_name": str(tool_name).strip() if tool_name else None,
            "payload": payload_dict,
            "reason": reason_text,
        }
