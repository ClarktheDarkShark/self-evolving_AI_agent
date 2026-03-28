#tool_validation.py

from __future__ import annotations

import ast
import inspect
import re
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from . import kg_utils as _kg_utils


@dataclass
class ToolValidationResult:
    success: bool
    error: Optional[str] = None
    smoke_output: Any = None
    self_test_passed: bool = False


def _dummy_value(param: inspect.Parameter) -> Any:
    annotation = param.annotation
    name = param.name.lower()
    if annotation in (str,):
        return "test"
    if annotation in (int,):
        return 0
    if annotation in (float,):
        return 0.0
    if annotation in (bool,):
        return False
    if annotation in (dict,):
        return {}
    if annotation in (list,):
        return []
    if "text" in name or "query" in name or "task" in name:
        return "test"
    return None


def _build_smoke_args(run_fn: Callable[..., Any]) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(run_fn)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not param.empty:
            continue
        value = _dummy_value(param)
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            args.append(value)
        else:
            kwargs[param.name] = value
    return args, kwargs


def _build_variation_args(
    run_fn: Callable[..., Any], base_args: list[Any], base_kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(run_fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    args = list(base_args)
    kwargs = dict(base_kwargs)
    if not params:
        return args, kwargs

    def _alt(val: Any) -> Any:
        if isinstance(val, str):
            return val + " alt"
        if isinstance(val, bool):
            return not val
        if isinstance(val, int):
            return val + 1
        if isinstance(val, float):
            return val + 1.0
        if isinstance(val, dict):
            return {**val, "alt": True}
        if isinstance(val, list):
            return val + ["alt"]
        if val is None:
            return "alt"
        return val

    # Prefer modifying the first required parameter.
    if args:
        args[0] = _alt(args[0])
        return args, kwargs

    # Otherwise modify the first kwarg.
    for key in list(kwargs.keys()):
        kwargs[key] = _alt(kwargs[key])
        return args, kwargs

    # If no args/kwargs were generated, fallback to a single string arg.
    return ["alt"], {}


def _looks_like_kg_macro(code: str) -> bool:
    markers = (
        "kg_utils",
        "target_concept",
        "entity_target_concepts",
        "intermediate_target_concepts",
        "actions_spec",
        "get_neighbors",
        "get_relations",
    )
    return sum(1 for marker in markers if marker in (code or "")) >= 2


class _KGSemanticRediscoveryVisitor(ast.NodeVisitor):
    _SEMANTIC_NAME_HINTS = (
        "target",
        "concept",
        "profession",
        "category",
        "relation",
        "entity",
        "entities",
        "label",
        "noun",
        "domain",
    )
    _TEXT_FN_ATTRS = {"search", "match", "findall", "finditer", "split", "partition", "rsplit"}

    def __init__(self) -> None:
        self.task_text_vars: set[str] = set()
        self.violations: list[str] = []

    @staticmethod
    def _payload_text_get(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "payload"
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value) in {"task_text", "asked_for"}
        )

    def _contains_text_source(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if self._payload_text_get(child):
                return True
            if isinstance(child, ast.Name) and child.id in self.task_text_vars:
                return True
        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._payload_text_get(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.task_text_vars.add(target.id)
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and any(hint in target.id.lower() for hint in self._SEMANTIC_NAME_HINTS)
                and self._contains_text_source(node.value)
            ):
                self.violations.append(
                    f"semantic_target_assignment_from_text:{target.id}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        text_parse = False
        if isinstance(func, ast.Attribute):
            if func.attr in self._TEXT_FN_ATTRS and self._contains_text_source(func.value):
                text_parse = True
            elif (
                isinstance(func.value, ast.Name)
                and func.value.id == "re"
                and func.attr in self._TEXT_FN_ATTRS
                and any(self._contains_text_source(arg) for arg in node.args)
            ):
                text_parse = True
        if text_parse:
            self.violations.append("semantic_parse_from_task_text")
        self.generic_visit(node)


class _KGHardcodedEntityInLabelVisitor(ast.NodeVisitor):
    """Detect hardcoded entity-specific string literals used as candidate_map / dict keys.

    A string literal like "resolved_Milk" or "walk_Goat_to_cheese" embedded directly in
    code (not derived via f-string from payload fields) is a task-imprinting generalization
    failure.  We flag string constants whose value matches the pattern
    ``<role_prefix>_<CapitalizedWord>`` where the capitalized word is not a known generic
    placeholder like Entity, Anchor, Item, etc.
    """

    _ROLE_PREFIXES = frozenset(
        {"resolved", "walk", "walked", "filter", "intersect", "intersected", "anchor"}
    )
    # Generic placeholder words that are acceptable as non-task-specific labels.
    _GENERIC_WORDS = frozenset(
        {
            "entity", "anchor", "item", "element", "node", "concept", "result",
            "set", "group", "target", "source", "base", "variable", "var",
        }
    )
    _PATTERN = re.compile(
        r"^(?P<prefix>[a-z]+)_(?P<word>[A-Z][a-zA-Z]{2,})(?:_.*)?$"
    )

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            self.generic_visit(node)
            return
        m = self._PATTERN.match(node.value)
        if m:
            prefix = m.group("prefix").lower()
            word = m.group("word").lower()
            if prefix in self._ROLE_PREFIXES and word not in self._GENERIC_WORDS:
                self.violations.append(
                    f"hardcoded_entity_in_label:{node.value}"
                )
        self.generic_visit(node)


def _kg_hardcoded_entity_in_label_issues(code: str) -> list[str]:
    """Return hardcoded-entity-in-label violations for KG macro tools."""
    if not _looks_like_kg_macro(code):
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    visitor = _KGHardcodedEntityInLabelVisitor()
    visitor.visit(tree)
    return visitor.violations


def _kg_attribute_target_as_anchor_issues(code: str) -> list[str]:
    """Return violations where attribute_target_concept is passed to resolve_entity_to_vars."""
    if not _looks_like_kg_macro(code):
        return []
    if "attribute_target_concept" not in code or "resolve_entity_to_vars" not in code:
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    # Collect variable names assigned from payload.get("attribute_target_concept")
    atc_var_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        rhs = node.value
        if (
            isinstance(rhs, ast.Call)
            and isinstance(rhs.func, ast.Attribute)
            and rhs.func.attr == "get"
            and rhs.args
            and isinstance(rhs.args[0], ast.Constant)
            and rhs.args[0].value == "attribute_target_concept"
        ):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    atc_var_names.add(tgt.id)
    if not atc_var_names:
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "kg_utils"
            and func.attr == "resolve_entity_to_vars"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            if node.args[0].id in atc_var_names:
                violations.append(
                    f"attribute_target_concept var '{node.args[0].id}' passed to resolve_entity_to_vars as entity anchor"
                )
    return violations


def _kg_semantic_rediscovery_issues(code: str) -> list[str]:
    if not _looks_like_kg_macro(code):
        return []
    plan_authority_present = any(
        token in code
        for token in (
            "tool_plan",
            "topological_execution_plan",
            "target_concept",
            "entity_target_concepts",
            "intermediate_target_concepts",
        )
    )
    if not plan_authority_present:
        return []
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    visitor = _KGSemanticRediscoveryVisitor()
    visitor.visit(tree)
    return visitor.violations


def validate_tool_code(
    code: str, *, timeout_s: float = 2.0
) -> ToolValidationResult:
    for heading in ("INVOKE_WITH:", "RUN_PAYLOAD_REQUIRED:", "RUN_PAYLOAD_OPTIONAL:"):
        if heading not in (code or ""):
            return ToolValidationResult(
                success=False,
                error=f"missing_invoke_contract:{heading}",
            )
    kg_semantic_issues = _kg_semantic_rediscovery_issues(code or "")
    if kg_semantic_issues:
        return ToolValidationResult(
            success=False,
            error="kg_semantic_rediscovery:" + ",".join(kg_semantic_issues),
        )
    kg_hardcode_issues = _kg_hardcoded_entity_in_label_issues(code or "")
    if kg_hardcode_issues:
        return ToolValidationResult(
            success=False,
            error="kg_hardcoded_entity_in_label:" + ",".join(kg_hardcode_issues),
        )
    kg_atc_anchor_issues = _kg_attribute_target_as_anchor_issues(code or "")
    if kg_atc_anchor_issues:
        return ToolValidationResult(
            success=False,
            error="kg_attribute_target_as_anchor:" + ",".join(kg_atc_anchor_issues),
        )
    # Strip any stray `import kg_utils` lines — kg_utils is a pre-injected global
    # and a bare import would raise ModuleNotFoundError at exec time.
    _clean_code = re.sub(r"^\s*import\s+kg_utils\b.*\n?", "", code, flags=re.MULTILINE)
    try:
        compiled = compile(_clean_code, "<generated_tool>", "exec")
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"compile failed: {exc}")

    module = types.ModuleType("generated_tool")
    # Pre-inject kg_utils so generated code can reference it as a global
    # without an import statement (mirrors the task.py safe_globals injection).
    helper_facade = _kg_utils.get_macro_helper_facade()
    module.__dict__["kg_utils"] = helper_facade
    try:
        exec(compiled, module.__dict__)
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"exec failed: {exc}")

    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        return ToolValidationResult(success=False, error="run() not found or not callable")

    self_test_fn = getattr(module, "self_test", None)
    if not callable(self_test_fn):
        return ToolValidationResult(
            success=False,
            error="self_test() not found or not callable",
        )

    sig = inspect.signature(run_fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    required = [
        p
        for p in params
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if len(required) != 1 or required[0].name != "payload":
        return ToolValidationResult(
            success=False,
            error="run() must accept exactly one required parameter named 'payload'",
        )

    # Build a smoke payload that includes mock callable actions_spec so
    # Macro tools pass callable() checks during validation.
    # NOTE: include 'entities' and other optional keys that generated tools
    # may declare as required, to avoid false-negative failures during smoke
    # testing.  The payload must be a superset of what real Orchestrator
    # payloads provide.
    def _mock_intersection(var1, var2):
        if not isinstance(var1, str) or not isinstance(var2, str):
            raise ValueError("intersection arguments must be strings")
        if var1 == var2:
            raise ValueError("intersection requires two distinct variables")
        return f"Variable #100 = intersection({var1}, {var2})"

    smoke_payload: dict = {
        "_smoke": True,
        "task_text": "smoke test",
        "asked_for": "smoke test",
        "trace": [],
        "run_id": "smoke",
        "state_dir": "/tmp",
        "entities": [],
        "env_observation": "",
        "constraints": {},
        "target_archetype": "UNKNOWN",
        "upgrade_goal": "",
        "kg_utils": helper_facade,
        "actions_spec": {
            "get_relations": lambda *_args: "Relations of mock: [mock.rel]",
            "get_neighbors": lambda *_args: "Variable #99 = get_neighbors(mock, mock.rel)",
            "intersection": _mock_intersection,
            "union": lambda *_args: "Variable #103 = union(#98, #99)",
            "difference": lambda *_args: "Variable #104 = difference(#98, #99)",
            "get_attributes": lambda *_args: "Attributes of #99: [mock.attr]",
            "argmax": lambda *_args: "Variable #101 = argmax(#99, mock.attr)",
            "argmin": lambda *_args: "Variable #102 = argmin(#99, mock.attr)",
            "count": lambda *_args: "Count of #99 is 42",
        },
    }
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_fn, smoke_payload)
            result = future.result(timeout=timeout_s)
    except TimeoutError:
        return ToolValidationResult(success=False, error="smoke test timed out")
    except Exception as exc:
        return ToolValidationResult(success=False, error=f"smoke test failed: {exc}")

    if not isinstance(result, dict):
        return ToolValidationResult(
            success=False,
            error="run() must return a dict",
        )
    # --- SSOT schema check (primary — new tools must use this) ---
    # Keys: status (str), final_variable (str|None), observation (str)
    ssot_keys = {"status", "final_variable", "observation"}
    has_ssot = all(k in result for k in ssot_keys)
    if has_ssot:
        valid_statuses = {"SUCCESS", "MACRO EXHAUSTED", "ERROR"}
        status_val = result.get("status")
        if status_val not in valid_statuses:
            return ToolValidationResult(
                success=False,
                error=f"ssot_status_invalid: '{status_val}' must be one of {sorted(valid_statuses)}",
            )
        if not isinstance(result.get("observation"), str):
            return ToolValidationResult(
                success=False,
                error="ssot_observation must be str",
            )
        fv = result.get("final_variable")
        if fv is not None and not isinstance(fv, (str, int)):
            return ToolValidationResult(
                success=False,
                error="ssot_final_variable must be str, int, or None",
            )
        # SSOT schema is fully valid — skip legacy/advisory checks.
    else:
        # --- Legacy advisory schema (backward compat for existing tools) ---
        advisory_keys = {"pruned_observation", "answer_recommendation", "confidence_score"}
        legacy_keys = {"next_action", "next_action_candidates", "why_stuck"}
        has_advisory = any(k in result for k in advisory_keys)
        has_legacy = any(k in result for k in legacy_keys)
        if has_advisory:
            missing = [k for k in advisory_keys if k not in result]
            if missing:
                return ToolValidationResult(
                    success=False,
                    error=f"missing_advisory_keys:{','.join(missing)}",
                )
            if not isinstance(result.get("answer_recommendation"), str):
                return ToolValidationResult(
                    success=False,
                    error="answer_recommendation must be str",
                )
            confidence = result.get("confidence_score")
            if not isinstance(confidence, (int, float)):
                return ToolValidationResult(
                    success=False,
                    error="confidence_score must be float",
                )
        elif not has_legacy:
            return ToolValidationResult(
                success=False,
                error="missing_output_schema_keys",
            )

    self_test_passed = False
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self_test_fn)
            self_test_passed = bool(future.result(timeout=timeout_s))
    except TimeoutError:
        self_test_passed = False
    except Exception:
        self_test_passed = False

    return ToolValidationResult(
        success=True, smoke_output=result, self_test_passed=self_test_passed
    )
