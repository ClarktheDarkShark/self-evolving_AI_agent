from __future__ import annotations

import ast
import re
from typing import Any, Dict, List


FORBIDDEN_SUBSTRINGS = [
    "sudo",
    "useradd",
    "usermod",
    "groupadd",
    "chmod",
    "chgrp",
]

FORBIDDEN_IMPORTS = [
    "requests",
    "httpx",
    "aiohttp",
    "boto3",
    "openai",
    "numpy",
    "pandas",
    "torch",
    "tensorflow",
    "subprocess",
]

MAX_LINES = 1000
ANCHOR_MARKERS = [
    "# === PHASE2_INSERT_TRACE_NORMALIZATION ===",
    "# === PHASE3_INSERT_NEXT_ACTION ===",
    "# === PHASE4_INSERT_ANSWER_GATES ===",
    "# === PHASE5_INSERT_SELF_TESTS ===",
]


def validate_tool_source(code: str) -> Dict[str, Any]:
    errors: List[str] = []
    metrics: Dict[str, Any] = {
        "line_count": len(code.splitlines()) if code else 0,
        "char_len": len(code or ""),
    }
    syntax_details: Dict[str, Any] = {}
    try:
        ast.parse(code or "")
        metrics["ast_parse_ok"] = True
    except SyntaxError as exc:
        metrics["ast_parse_ok"] = False
        syntax_details = {
            "error_type": type(exc).__name__,
            "msg": str(exc),
            "lineno": exc.lineno,
            "col_offset": exc.offset,
        }
        errors.append("ast_parse_failed")
    except Exception as exc:
        metrics["ast_parse_ok"] = False
        syntax_details = {"error_type": type(exc).__name__, "msg": str(exc)}
        errors.append("ast_parse_failed")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "metrics": metrics,
        "syntax_error_details": syntax_details,
    }


def audit_tool_code(code: str) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not code:
        return {"ok": False, "errors": ["empty_code"], "warnings": []}

    lines = code.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return {"ok": False, "errors": ["empty_code"], "warnings": []}

    if non_empty[0].strip() != "###TOOL_START":
        errors.append("missing_tool_start")
    if non_empty[-1].strip() != "###TOOL_END":
        errors.append("missing_tool_end")

    if code.count('"""') != 2:
        errors.append("docstring_triple_quotes_invalid")
    if "'''" in code:
        errors.append("docstring_triple_quotes_invalid_single")
    else:
        try:
            start_idx = lines.index('"""')
            end_idx = lines.index('"""', start_idx + 1)
            if end_idx - start_idx != 2:
                errors.append("docstring_not_three_lines")
        except ValueError:
            errors.append("docstring_missing")

    if not re.search(r"^#\s*tool_name\s*:\s*[a-zA-Z0-9_]+_generated_tool\s*$", code, re.MULTILINE):
        errors.append("tool_name_missing")

    if not re.search(r"^def\s+run\(payload:\s*dict\)\s*->\s*dict\s*:", code, re.MULTILINE):
        errors.append("run_signature_missing")
    if not re.search(r"^def\s+self_test\(\)\s*->\s*bool\s*:", code, re.MULTILINE):
        errors.append("self_test_missing")

    for heading in ("INVOKE_WITH:", "RUN_PAYLOAD_REQUIRED:", "RUN_PAYLOAD_OPTIONAL:"):
        if heading not in code:
            errors.append(f"invoke_contract_missing:{heading}")

    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in code:
            errors.append(f"forbidden_substring:{forbidden}")

    for imp in FORBIDDEN_IMPORTS:
        if re.search(rf"^\s*(from|import)\s+{re.escape(imp)}\b", code, re.MULTILINE):
            errors.append(f"forbidden_import:{imp}")

    if len(lines) > MAX_LINES:
        errors.append(f"line_count_exceeded:{len(lines)}")

    anchors_remaining = [a for a in ANCHOR_MARKERS if a in code]
    if anchors_remaining:
        errors.append("anchors_remaining")

    ast_check = validate_tool_source(code)
    if not ast_check.get("metrics", {}).get("ast_parse_ok", False):
        errors.append("ast_parse_failed")

    audit_checks = {
        "anchors_removed": not anchors_remaining,
        "triple_quotes_ok": code.count('"""') == 2 and "'''" not in code,
        "ast_parse_ok": ast_check.get("metrics", {}).get("ast_parse_ok", False),
        "forbidden_ok": not any(f in code for f in FORBIDDEN_SUBSTRINGS),
        "required_helpers_ok": True,
    }

    ok = len(errors) == 0
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "audit_checks": audit_checks,
        "syntax_error_details": ast_check.get("syntax_error_details", {}),
    }
