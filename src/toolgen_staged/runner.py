from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Callable, Mapping, Optional

from .auditor import FORBIDDEN_SUBSTRINGS, audit_tool_code
from .integrator import integrate
from .prompts import load_prompt
from .utils import ensure_markers, extract_marked_block, strip_code_fences, summarize_text


LLMCall = Callable[[str, str], str]


class _StagedLogger:
    def __init__(self, log_path: Optional[str], tool_build_span_id: str) -> None:
        self._log_path = log_path
        self._seq = 0
        self._span_id = tool_build_span_id

    def log(self, event: str, **fields: Any) -> None:
        if not self._log_path:
            return
        self._seq += 1
        payload = {"t": self._seq, "event": event, "tool_build_span_id": self._span_id}
        payload.update(fields)
        try:
            dir_name = os.path.dirname(self._log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception:
            return


def _phase_output_dir(log_path: Optional[str], span_id: str) -> Optional[str]:
    if not log_path:
        return None
    parent = os.path.dirname(log_path)
    if not parent:
        return None
    return os.path.join(parent, "toolgen_staged_phases", span_id)


def _write_phase_output(output_dir: Optional[str], phase_name: str, content: str) -> Optional[str]:
    if not output_dir:
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{phase_name}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path
    except Exception:
        return None


def _phase_user_prompt(task_context: Mapping[str, Any]) -> str:
    stripped = dict(task_context)
    stripped.pop("system_prompt", None)
    return json.dumps(stripped, ensure_ascii=True, default=str)


def _repair_user_prompt(audit: Mapping[str, Any], code: str) -> str:
    payload = {
        "audit_errors": audit.get("errors", []),
        "audit_warnings": audit.get("warnings", []),
        "current_tool": code,
    }
    return json.dumps(payload, ensure_ascii=True, default=str)


def _phase_metrics(text: str, anchors: list[str]) -> dict[str, Any]:
    lines = text.splitlines()
    head = lines[:3]
    tail = lines[-3:] if len(lines) >= 3 else lines
    triple_double = text.count('"""')
    triple_single = text.count("'''")
    forbidden_hits = [s for s in FORBIDDEN_SUBSTRINGS if s in text]
    anchor_hits = [a for a in anchors if a in text]
    return {
        "char_len": len(text),
        "line_count": len(lines),
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        "head_3": head,
        "tail_3": tail,
        "forbidden_hits": forbidden_hits,
        "anchor_hits": anchor_hits,
        "triple_quote_counts": {"triple_double": triple_double, "triple_single": triple_single},
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_staged_toolgen(
    task_context: Mapping[str, Any],
    llm_call: LLMCall,
    *,
    log_path: Optional[str] = None,
    tool_build_span_id: str = "unknown",
    max_repairs: int = 3,
) -> str:
    logger = _StagedLogger(log_path, tool_build_span_id)
    logger.log("start", mode="staged")
    output_dir = _phase_output_dir(log_path, tool_build_span_id)

    phases = [
        ("phase1", "phase1_skeleton_contracts", ""),
        ("phase2", "phase2_trace_normalization", "# === PHASE2_INSERT_TRACE_NORMALIZATION ==="),
        ("phase3", "phase3_next_action", "# === PHASE3_INSERT_NEXT_ACTION ==="),
        ("phase4", "phase4_answer_gates", "# === PHASE4_INSERT_ANSWER_GATES ==="),
        ("phase5", "phase5_self_tests", "# === PHASE5_INSERT_SELF_TESTS ==="),
    ]

    parts: dict[str, str] = {}
    anchors = [p[2] for p in phases if p[2]]
    for phase_name, prompt_name, anchor in phases:
        system_prompt = load_prompt(prompt_name)
        user_prompt = _phase_user_prompt(task_context)
        logger.log("phase_start", phase=phase_name)
        start = time.monotonic()
        raw = llm_call(system_prompt, user_prompt)
        duration_ms = int((time.monotonic() - start) * 1000)
        cleaned = strip_code_fences(raw)
        parts[phase_name] = cleaned.strip()
        output_path = _write_phase_output(output_dir, phase_name, parts[phase_name])
        metrics = _phase_metrics(parts[phase_name], anchors)
        logger.log(
            "phase_done",
            phase=phase_name,
            duration_ms=duration_ms,
            output=summarize_text(parts[phase_name]),
            output_metrics=metrics,
            output_path=output_path,
        )

    integrated = integrate(parts)
    integrated = ensure_markers(integrated)
    replacement_map = {
        anchor: hashlib.sha1((parts.get(key) or "").encode("utf-8")).hexdigest()
        for key, anchor in [
            ("phase2", "# === PHASE2_INSERT_TRACE_NORMALIZATION ==="),
            ("phase3", "# === PHASE3_INSERT_NEXT_ACTION ==="),
            ("phase4", "# === PHASE4_INSERT_ANSWER_GATES ==="),
            ("phase5", "# === PHASE5_INSERT_SELF_TESTS ==="),
        ]
    }
    logger.log(
        "integrated",
        output=summarize_text(integrated),
        replacement_map=replacement_map,
        final_metrics={
            "line_count": len(integrated.splitlines()),
            "char_len": len(integrated),
            "sha256": _sha256(integrated),
            "triple_quote_count": integrated.count('"""') + integrated.count("'''"),
            "forbidden_hits": [s for s in FORBIDDEN_SUBSTRINGS if s in integrated],
        },
    )

    audit = audit_tool_code(integrated)
    logger.log(
        "audit",
        ok=audit.get("ok"),
        errors=audit.get("errors"),
        warnings=audit.get("warnings"),
        audit_checks=audit.get("audit_checks"),
        syntax_error_details=audit.get("syntax_error_details"),
    )

    if not audit.get("ok"):
        repair_prompt = load_prompt("phase6_auditor")
        for attempt in range(1, max_repairs + 1):
            user_prompt = _repair_user_prompt(audit, integrated)
            logger.log("repair_start", attempt=attempt)
            raw = llm_call(repair_prompt, user_prompt)
            fixed = strip_code_fences(raw)
            fixed_block = extract_marked_block(fixed) or fixed
            integrated = ensure_markers(fixed_block)
            audit = audit_tool_code(integrated)
            logger.log(
                "repair_done",
                attempt=attempt,
                ok=audit.get("ok"),
                errors=audit.get("errors"),
                warnings=audit.get("warnings"),
                audit_checks=audit.get("audit_checks"),
                syntax_error_details=audit.get("syntax_error_details"),
            )
            if audit.get("ok"):
                break

    if not audit.get("ok"):
        logger.log("audit_failed_final", errors=audit.get("errors"), warnings=audit.get("warnings"))

    return integrated
