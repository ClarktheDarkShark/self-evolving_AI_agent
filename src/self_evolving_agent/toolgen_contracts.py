from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import sys
import sysconfig
from dataclasses import dataclass
from typing import Iterable, Optional

TOOL_START = "###TOOL_START"
TOOL_END = "###TOOL_END"

SCHEMA_CLAUSE = (
    "INPUT_SCHEMA: required=task_text,asked_for,trace,actions_spec,run_id,state_dir; "
    "optional=constraints,output_contract,draft_response,candidate_output,env_observation >"
)

SCHEMA_ECHO_REQUIRED = [
    "RUN_PAYLOAD_REQUIRED",
    "RUN_PAYLOAD_OPTIONAL",
    "Example",
]

SCHEMA_ECHO_OPTIONAL_GROUP = [
    "INVOKE_WITH",
    "INVOKE_EXAMPLE",
]

FORBIDDEN_SUBSTRINGS = [
    "sudo",
    "useradd",
    "usermod",
    "groupadd",
    "chmod",
    "chgrp",
    "eval(",
    "exec(",
    "subprocess",
    "urllib",
    "socket(",
    "http.client",
]


@dataclass
class ToolgenContractResult:
    ok: bool
    errors: list[str]
    code: str
    normalized_code: str
    code_sha256: str
    code_len: int
    docstring_preview: str


def extract_marked_code(raw_text: str) -> tuple[Optional[str], list[str]]:
    errors: list[str] = []
    if not raw_text:
        return None, ["A:empty_output"]
    start_count = raw_text.count(TOOL_START)
    end_count = raw_text.count(TOOL_END)
    if start_count != 1:
        errors.append("A:marker_start_count")
    if end_count != 1:
        errors.append("A:marker_end_count")
    if errors:
        return None, errors
    start = raw_text.find(TOOL_START)
    end = raw_text.find(TOOL_END, start + len(TOOL_START))
    if start < 0 or end < 0 or end <= start:
        return None, errors + ["A:marker_order"]
    if not raw_text.strip().startswith(TOOL_START) or not raw_text.strip().endswith(TOOL_END):
        errors.append("A:marker_extra_text")
    code = raw_text[start + len(TOOL_START):end]
    if code.startswith("\r\n"):
        code = code[2:]
    elif code.startswith("\n"):
        code = code[1:]
    return code, errors


def normalize_tool_code(code: str) -> str:
    if not code:
        return ""
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    text = text.rstrip()
    return text + "\n"


def _docstring_preview(code: str, max_len: int = 200) -> str:
    try:
        tree = ast.parse(code)
        doc = ast.get_docstring(tree) or ""
    except Exception:
        doc = ""
    if not doc:
        return ""
    return doc if len(doc) <= max_len else doc[: max_len - 3] + "..."


def _iter_imports(tree: ast.AST) -> Iterable[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def _is_stdlib_module(name: str) -> bool:
    if not name:
        return False
    root = name.split(".", 1)[0]
    if root in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names and root in stdlib_names:
        return True
    spec = importlib.util.find_spec(root)
    if not spec or not spec.origin:
        return False
    if spec.origin == "built-in":
        return True
    origin = os.path.abspath(spec.origin)
    if "site-packages" in origin or "dist-packages" in origin:
        return False
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if stdlib_path and origin.startswith(os.path.abspath(stdlib_path)):
        return True
    return False


def _check_run_docstring_invariant(_code: str, tree: ast.AST) -> list[str]:
    errors: list[str] = []
    run_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_fn = node
            break
    if run_fn is None or not run_fn.body:
        return ["B:docstring_missing"]
    first = run_fn.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)):
        errors.append("B:docstring_start_missing")
        return errors
    doc = ast.get_docstring(run_fn) or ""
    if not doc:
        errors.append("B:docstring_end_missing")
        return errors
    required_terms = ["contract guard", "prereqs", "limitations"]
    doc_lower = doc.lower()
    missing_terms = [term for term in required_terms if term not in doc_lower]
    if missing_terms:
        errors.append("B:docstring_clause_missing")
    return errors


def _check_schema_echo(code: str) -> list[str]:
    errors: list[str] = []
    lines = code.splitlines()
    head = "\n".join(lines[:80])
    def _has_heading(name: str) -> bool:
        pattern = r"^\s*#\s*%s\s*:" % re.escape(name)
        return re.search(pattern, head, re.MULTILINE) is not None
    for name in SCHEMA_ECHO_REQUIRED:
        if not _has_heading(name):
            errors.append(f"G:schema_echo_missing:# {name}:")
    optional_present = any(_has_heading(name) for name in SCHEMA_ECHO_OPTIONAL_GROUP)
    if optional_present:
        for name in SCHEMA_ECHO_OPTIONAL_GROUP:
            if not _has_heading(name):
                errors.append(f"G:schema_echo_missing:# {name}:")
    return errors


def _check_signatures(code: str) -> list[str]:
    errors: list[str] = []
    if not re.search(r"^def\s+run\(payload:\s*dict\)\s*->\s*dict\s*:", code, re.MULTILINE):
        errors.append("D:run_signature_missing")
    if not re.search(r"^def\s+self_test\(\)\s*->\s*bool\s*:", code, re.MULTILINE):
        errors.append("D:self_test_signature_missing")
    return errors


def _check_imports(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in _iter_imports(tree):
        parent = parents.get(node)
        if not isinstance(parent, ast.Module):
            errors.append("C:non_top_level_import")
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            names = [node.module] if node.module else []
        for name in names:
            if name and not _is_stdlib_module(name):
                errors.append(f"C:non_stdlib_import:{name}")
    return errors


def _check_run_try_except(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    run_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_fn = node
            break
    if run_fn is None:
        return ["F:run_missing"]
    if not run_fn.body:
        return ["F:run_empty_body"]
    body_idx = 0
    first = run_fn.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        body_idx = 1
    if body_idx >= len(run_fn.body) or not isinstance(run_fn.body[body_idx], ast.Try):
        errors.append("F:run_not_wrapped")
        return errors
    try_node = run_fn.body[body_idx]
    if not try_node.handlers:
        errors.append("F:run_missing_except")
        return errors

    def _return_has_error_dict(ret: ast.Return) -> bool:
        if not isinstance(ret.value, ast.Dict):
            return False
        keys = []
        for key in ret.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
            else:
                keys.append(None)
        # Accept any dict that contains an "error" key (advisory or legacy).
        if "error" in keys:
            return True
        # Also accept legacy dicts with {"status": "error", "errors", "warnings"}.
        legacy_needed = {"status", "errors", "warnings"}
        if legacy_needed.issubset(set(keys)):
            for k, v in zip(keys, ret.value.values):
                if k == "status" and isinstance(v, ast.Constant) and v.value == "error":
                    return True
        return False

    except_ok = False
    for handler in try_node.handlers:
        for node in ast.walk(handler):
            if isinstance(node, ast.Return) and _return_has_error_dict(node):
                except_ok = True
                break
        if except_ok:
            break
    if not except_ok:
        errors.append("F:run_except_missing_error_return")
    return errors


def _check_output_schema_presence(code: str) -> list[str]:
    errors: list[str] = []
    if not code:
        return errors
    advisory_keys = ["pruned_observation", "answer_recommendation", "confidence_score"]
    legacy_keys = ["next_action", "next_action_candidates", "why_stuck"]
    has_advisory = any(key in code for key in advisory_keys)
    has_legacy = any(key in code for key in legacy_keys)
    if has_advisory:
        missing = [key for key in advisory_keys if key not in code]
        if missing:
            errors.append(f"H:advisory_keys_missing:{','.join(missing)}")
    elif not has_legacy:
        errors.append("H:output_schema_missing")
    return errors


def validate_toolgen_output(raw_text: str) -> ToolgenContractResult:
    code, errors = extract_marked_code(raw_text or "")
    normalized = normalize_tool_code(code or "")
    if not normalized:
        errors.append("A:empty_extracted_code")
    if normalized:
        errors.extend(_check_schema_echo(normalized))
        errors.extend(_check_signatures(normalized))
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in normalized:
                errors.append(f"E:forbidden_substring:{forbidden}")
        try:
            tree = ast.parse(normalized)
        except Exception:
            errors.append("F:syntax_error")
        else:
            errors.extend(_check_imports(tree))
            errors.extend(_check_run_docstring_invariant(normalized, tree))
            errors.extend(_check_run_try_except(tree))
        errors.extend(_check_output_schema_presence(normalized))
    code_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""
    return ToolgenContractResult(
        ok=len(errors) == 0,
        errors=errors,
        code=code or "",
        normalized_code=normalized,
        code_sha256=code_sha256,
        code_len=len(normalized),
        docstring_preview=_docstring_preview(normalized),
    )
