from __future__ import annotations

from src.toolgen_staged.auditor import audit_tool_code
from src.toolgen_staged.runner import run_staged_toolgen


PHASE1_SKELETON = '''###TOOL_START
# tool_name: staged_smoke_generated_tool
"""
Tool helper with contract guard, prereqs, next-action suggestion, limitations. INPUT_SCHEMA: required=task_text,asked_for,trace,actions_spec,run_id,state_dir; optional=constraints,output_contract,draft_response,candidate_output,env_observation; limitations (no external calls; local JSON state only; analyzes payload; does not call tools).
"""
import json
import os
import re
import tempfile

# === PHASE2_INSERT_TRACE_NORMALIZATION ===
# === PHASE3_INSERT_NEXT_ACTION ===
# === PHASE4_INSERT_ANSWER_GATES ===

def _load_state(state_dir, run_id):
    path = os.path.join(state_dir, f"{run_id}.json")
    if not os.path.exists(path):
        return {"cursor": "", "history": [], "notes": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_state(state_dir, run_id, state):
    os.makedirs(state_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=state_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, os.path.join(state_dir, f"{run_id}.json"))

def run(payload: dict) -> dict:
    try:
        errors = []
        for k in ("task_text","asked_for","trace","actions_spec","run_id","state_dir"):
            if k not in payload:
                errors.append(f"missing_payload_key:{k}")
        if errors:
            return {"status":"blocked","next_action":None,"answer_recommendation":None,"plan":[],"validation":{},"rationale":["missing"],"errors":errors,"warnings":[]}
        trace = _normalize_trace(payload.get("trace"))
        answer = _answer_gate(payload.get("asked_for",""), trace, payload.get("constraints", []),
                              payload.get("output_contract", {}), payload.get("draft_response"),
                              payload.get("candidate_output"), payload.get("env_observation"))
        if answer["status"] == "can_answer":
            status = "can_answer"
            next_action = None
            answer_rec = answer["answer_recommendation"]
        else:
            status = "need_step"
            next_action = _choose_next_action(payload.get("actions_spec") or {}, trace, payload.get("asked_for",""))
            answer_rec = None
        state = _load_state(payload["state_dir"], payload["run_id"])
        state["cursor"] = trace[-1]["action"] if trace else ""
        state["history"].append(state["cursor"])
        _save_state(payload["state_dir"], payload["run_id"], state)
        return {"status":status,"next_action":next_action,"answer_recommendation":answer_rec,"plan":[],
                "validation":answer["validation"],"rationale":answer["rationale"],"errors":answer["errors"],"warnings":answer["warnings"]}
    except Exception as e:
        return {"status":"error","next_action":None,"answer_recommendation":None,"plan":[],"validation":{},"rationale":[str(e)],"errors":[str(e)],"warnings":[]}

def self_test() -> bool:
    try:
        # === PHASE5_INSERT_SELF_TESTS ===
        return True
    except Exception:
        return False
###TOOL_END
'''.strip()


PHASE2_SNIPPET = """
def _parse_raw_action(raw):
    if not isinstance(raw, str):
        return "", {}
    raw = raw.strip()
    if "(" not in raw or not raw.endswith(")"):
        return raw, {}
    name, rest = raw.split("(", 1)
    args_text = rest[:-1]
    parts = [p.strip() for p in args_text.split(",") if p.strip()]
    args = {"arg"+str(i): p for i, p in enumerate(parts)}
    return name.strip(), args

def _normalize_trace(trace):
    if isinstance(trace, list) and all(isinstance(t, str) for t in trace):
        return [{"action": t, "ok": None, "output": None, "args": {}, "error": None} for t in trace]
    normalized = []
    if not isinstance(trace, list):
        return normalized
    for step in trace:
        if not isinstance(step, dict):
            action, args = _parse_raw_action(str(step))
            normalized.append({"action": action, "ok": None, "output": None, "args": args, "error": None})
            continue
        action = step.get("action") or step.get("raw") or ""
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if action and action.startswith("get_") and "raw" in step:
            parsed_action, parsed_args = _parse_raw_action(step.get("raw", ""))
            if parsed_action:
                action = parsed_action
            if parsed_args:
                args = parsed_args
        normalized.append({"action": action, "ok": step.get("ok"), "output": step.get("output"), "args": args, "error": step.get("error")})
    return normalized
""".strip()


PHASE3_SNIPPET = """
def _choose_next_action(actions_spec, trace, asked_for):
    actions = sorted(actions_spec.keys()) if isinstance(actions_spec, dict) else []
    last_action = trace[-1]["action"] if trace else None
    for name in ["get_relations", "get_neighbors", "intersection", "count"]:
        if name in actions and name != last_action:
            if name == "get_neighbors":
                return {"action": name, "args": {"var": "#0", "relation": ""}}
            if name == "get_relations":
                return {"action": name, "args": {"var": "#0"}}
            return {"action": name, "args": {}}
    return {"action": actions[0], "args": {}} if actions else None
""".strip()


PHASE4_SNIPPET = """
def _desired_kind(asked_for):
    text = (asked_for or "").lower()
    if "how many" in text or "count" in text or "number" in text:
        return "numeric"
    return "string"

def _answer_gate(asked_for, trace, constraints, output_contract, draft_response, candidate_output, env_observation):
    last_ok = None
    for step in trace:
        if step.get("ok") is True:
            last_ok = step.get("output")
    candidate = candidate_output if candidate_output not in (None, "") else last_ok
    if isinstance(candidate, str) and "Variable #" in candidate:
        candidate = None
    status = "can_answer" if candidate is not None else "need_step"
    validation = {"contract_ok": True, "contract_violations": [], "solver_suggestion": None}
    return {"status": status, "answer_recommendation": candidate, "validation": validation, "rationale": ["ok"], "errors": [], "warnings": []}
""".strip()


PHASE5_SNIPPET = """
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            payload = {"task_text": "t", "asked_for": "count", "trace": [], "actions_spec": {"get_relations": {}}, "run_id": "r1", "state_dir": td}
            res = run(payload)
            if not isinstance(res, dict):
                return False
        return True
""".rstrip()


def _fake_llm(system_prompt: str, user_prompt: str) -> str:
    if "Phase 1" in system_prompt:
        return PHASE1_SKELETON
    if "Phase 2" in system_prompt:
        return PHASE2_SNIPPET
    if "Phase 3" in system_prompt:
        return PHASE3_SNIPPET
    if "Phase 4" in system_prompt:
        return PHASE4_SNIPPET
    if "Phase 5" in system_prompt:
        return PHASE5_SNIPPET
    return PHASE1_SKELETON


def test_staged_toolgen_smoke():
    task_context = {"user_prompt": "{}", "system_prompt": "system"}
    code = run_staged_toolgen(task_context, _fake_llm, max_repairs=0)
    audit = audit_tool_code(code)
    assert audit["ok"], audit
