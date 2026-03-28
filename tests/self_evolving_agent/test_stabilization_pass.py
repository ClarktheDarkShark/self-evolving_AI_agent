"""Targeted tests for the pre-Phase-2 stabilization pass.

Tests cover all four required changes:
  A. Role-aware anchor selection
  B. Canonical live-context helper discipline
  C. Grounded handoff requirement
  D. Anti-hardcoding (task imprinting elimination)
"""
import ast
import pathlib
import sys
import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_evolving_agent.controller_logging import ControllerLoggingMixin
from src.self_evolving_agent.controller_orchestrator import ControllerOrchestratorMixin
from src.self_evolving_agent.controller_toolgen import ControllerToolgenMixin
from src.self_evolving_agent.controller_prompts import (
    MACRO_TOOLGEN_USER_KG,
    TOOLGEN_VALIDATOR_SYSTEM_PROMPT,
)
from src.self_evolving_agent.tool_validation import (
    validate_tool_code,
    _kg_hardcoded_entity_in_label_issues,
    _kg_attribute_target_as_anchor_issues,
)


# ---------------------------------------------------------------------------
# Test controller stub
# ---------------------------------------------------------------------------

class _DummyController(
    ControllerLoggingMixin,
    ControllerToolgenMixin,
    ControllerOrchestratorMixin,
):
    def __init__(self, tmp_path: pathlib.Path) -> None:
        self._generated_tools_log_path = tmp_path / "generated_tools.jsonl"
        self._toolgen_debug_logger = None
        self._trace_hook = None
        self._current_task_label = "test_task"

    def _resolved_environment_label(self) -> str:
        return "knowledge_graph"

    def _get_run_task_metadata(self) -> dict:
        return {"task_name": "test_task", "sample_index": 0}


def _make_controller(tmp_path: pathlib.Path) -> _DummyController:
    return _DummyController(tmp_path)


def _tmp(name: str = "stabilization") -> pathlib.Path:
    p = pathlib.Path("/tmp") / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Minimal tool template for generating legal tool code in tests
# ---------------------------------------------------------------------------

_TOOL_HEADER = """\
# tool_name: test_macro_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: ["env_observation", "domain_hints", "target_concept", "attribute_target_concept", "entity_target_concepts", "intermediate_target_concepts", "topological_execution_plan", "composite_topology", "target_archetype", "upgrade_goal", "recovery_policy", "execution_style", "preferred_tool_mode", "minimum_acceptable_deliverable", "fallback_strategies", "tool_plan", "toolgen_retry_context", "variable_list"]
# INVOKE_EXAMPLE: {"args":[{"task_text":"...","asked_for":"...","trace":[],"actions_spec":{},"run_id":"r1","state_dir":"./state","entities":["A"]}],"kwargs":{}}
"""

_TOOL_BOILERPLATE_RUN_PREFIX = '''\
\"\"\"KG macro test.\"\"\"

import json

def run(payload: dict) -> dict:
    """
    contract guard: payload must contain the required run keys.
    prereqs: kg_utils facade and needed actions_spec primitives are available.
    limitations: test tool.
    """
    try:
'''

_TOOL_BOILERPLATE_SUFFIX = '''\
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": f"Tool error: {str(e)}"}

def self_test() -> bool:
    return True
'''

def _make_tool(body: str) -> str:
    """Wrap body lines (indented 8 spaces) with legal tool scaffolding."""
    return _TOOL_HEADER + _TOOL_BOILERPLATE_RUN_PREFIX + body + _TOOL_BOILERPLATE_SUFFIX


# ---------------------------------------------------------------------------
# A. ROLE-AWARE ANCHOR SELECTION TESTS
# ---------------------------------------------------------------------------

def test_A1_prompt_forbids_blanket_resolve_when_semantic_fields_present():
    """ToolGen prompt must prohibit blanket entity-resolve loops when richer semantic
    fields (entity_target_concepts, attribute_target_concept) are present."""
    assert "attribute_target_concept" in MACRO_TOOLGEN_USER_KG
    assert "entity_target_concepts" in MACRO_TOOLGEN_USER_KG
    # Key rule: blanket loops are forbidden unless plan explicitly requires it
    assert "blanket" in MACRO_TOOLGEN_USER_KG.lower() or \
           "resolve all entities" in MACRO_TOOLGEN_USER_KG.lower()
    # Anchor selection from plan is required
    assert "ANCHOR SELECTION FROM PLAN" in MACRO_TOOLGEN_USER_KG


def test_A2_prompt_requires_attribute_target_concept_as_filter_not_anchor():
    """ToolGen prompt must say attribute_target_concept is filter/modifier, not anchor."""
    text = MACRO_TOOLGEN_USER_KG
    assert "attribute_target_concept" in text
    assert "filter" in text.lower() or "modifier" in text.lower()
    # Rule must be present in the OPERAND ROLE PRESERVATION section or ANCHOR SELECTION section
    assert "ANCHOR SELECTION FROM PLAN" in text
    # And must NOT say to resolve it as an entity
    anchor_section_start = text.find("ANCHOR SELECTION FROM PLAN")
    anchor_section = text[anchor_section_start:anchor_section_start + 800]
    assert "attribute_target_concept" in anchor_section
    assert "resolve_entity_to_vars" in anchor_section  # must mention the rule


def test_A3_smell_detector_flags_blanket_entity_resolve_loop_when_attribute_present():
    """Smell detector must flag blanket entity-resolve loops when attribute_target_concept present."""
    ctrl = _make_controller(_tmp("A3"))
    # Code with attribute_target_concept in payload but blanket loop over all entities
    code_with_blanket_loop = """\
import json
attribute_target_concept = "texture"
entities = ["Goat", "Milk"]
for entity in entities:
    result = kg_utils.resolve_entity_to_vars(entity, None, None, None)
"""
    smells = ctrl._toolgen_semantic_code_smells(code_with_blanket_loop)
    assert "blanket_entity_resolve_loop" in smells, (
        f"Expected blanket_entity_resolve_loop smell when attribute_target_concept present, "
        f"got: {smells}"
    )


def test_A3b_smell_detector_does_not_flag_role_discriminated_resolve():
    """Smell detector must NOT flag resolve loops that use entity_target_concepts for role discrimination."""
    ctrl = _make_controller(_tmp("A3b"))
    # Code uses entity_target_concepts (no attribute_target_concept) → OK blanket loop
    code_with_role_loop = """\
import json
entities = ["CNES", "Astrium"]
entity_target_concepts = ["space_agency", "aerospace_company"]
for entity, hint in zip(entities, entity_target_concepts):
    result = kg_utils.resolve_entity_to_vars(entity, hint, None, None)
"""
    smells = ctrl._toolgen_semantic_code_smells(code_with_role_loop)
    assert "blanket_entity_resolve_loop" not in smells, (
        f"Should not flag role-discriminated resolve loop, got: {smells}"
    )


# ---------------------------------------------------------------------------
# B. CANONICAL LIVE-CONTEXT HELPER DISCIPLINE TESTS
# ---------------------------------------------------------------------------

def test_B1_prompt_requires_extract_var_ids_canonicalization_after_helper():
    """ToolGen prompt must require extract_var_ids + deduplication after set-producing helpers."""
    assert "extract_var_ids" in MACRO_TOOLGEN_USER_KG
    assert "LIVE-CONTEXT CANONICALIZATION" in MACRO_TOOLGEN_USER_KG
    canon_section_start = MACRO_TOOLGEN_USER_KG.find("LIVE-CONTEXT CANONICALIZATION")
    canon_section = MACRO_TOOLGEN_USER_KG[canon_section_start:canon_section_start + 600]
    assert "dict.fromkeys" in canon_section or "deduplicate" in canon_section.lower()


def test_B2_prompt_requires_live_context_for_resolve_semantic_filter():
    """ToolGen prompt must explicitly forbid using anchor vars as variable_list for resolve_semantic_filter."""
    text = MACRO_TOOLGEN_USER_KG
    assert "resolve_semantic_filter" in text
    # Must say variable_list must be live current candidate set, not anchor vars
    canon_start = text.find("LIVE-CONTEXT CANONICALIZATION")
    canon_section = text[canon_start:canon_start + 1200]
    assert "anchor" in canon_section.lower()
    assert "variable_list" in canon_section
    # Must have RIGHT/WRONG examples
    assert "WRONG" in canon_section
    assert "RIGHT" in canon_section


def test_B3_validator_prompt_defines_runtime_context_misuse():
    """Validator prompt must define runtime_context_misuse as a hard fail for wrong variable_list."""
    assert "LIVE-CONTEXT MISUSE" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT
    misuse_start = TOOLGEN_VALIDATOR_SYSTEM_PROMPT.find("LIVE-CONTEXT MISUSE")
    misuse_section = TOOLGEN_VALIDATOR_SYSTEM_PROMPT[misuse_start:misuse_start + 1200]
    assert "RUNTIME_CONTEXT_MISUSE" in misuse_section
    assert "variable_list" in misuse_section
    assert "anchor" in misuse_section.lower()
    # Must say this is NOT a minor smell
    assert "minor" in misuse_section.lower() or "not a minor" in misuse_section.lower()


def test_B4_helper_signature_smell_already_detected():
    """Existing helper_signature_mismatch smell must still be detected for wrong arg counts."""
    ctrl = _make_controller(_tmp("B4"))
    # resolve_semantic_filter requires 3 positional args — provide only 2
    code_wrong_sig = """\
import json
def run(payload: dict) -> dict:
    result = kg_utils.resolve_semantic_filter("base_var", "concept")
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    smells = ctrl._toolgen_semantic_code_smells(code_wrong_sig)
    assert "helper_signature_mismatch" in smells, (
        f"Expected helper_signature_mismatch for wrong arg count, got: {smells}"
    )


# ---------------------------------------------------------------------------
# C. GROUNDED HANDOFF REQUIREMENT TESTS
# ---------------------------------------------------------------------------

def test_C1_prompt_contains_built_set_handoff_rule():
    """ToolGen prompt must contain the built-set handoff rule with all three states."""
    assert "BUILT-SET HANDOFF RULE" in MACRO_TOOLGEN_USER_KG
    rule_start = MACRO_TOOLGEN_USER_KG.find("BUILT-SET HANDOFF RULE")
    rule_section = MACRO_TOOLGEN_USER_KG[rule_start:rule_start + 800]
    assert "built_target_set" in rule_section
    assert "built_both_sets" in rule_section
    assert "built_intersection_set" in rule_section
    # Must require final_variable=#N
    assert "final_variable" in rule_section
    assert "SUCCESS" in rule_section


def test_C2_validator_prompt_has_mandatory_grounded_handoff_section():
    """Validator prompt must have mandatory grounded handoff section as top-priority repair."""
    assert "MANDATORY GROUNDED HANDOFF" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT
    section_start = TOOLGEN_VALIDATOR_SYSTEM_PROMPT.find("MANDATORY GROUNDED HANDOFF")
    section = TOOLGEN_VALIDATOR_SYSTEM_PROMPT[section_start:section_start + 800]
    assert "built_target_set" in section
    assert "missing_grounded_handoff_after_real_progress" in section
    assert "TOP-PRIORITY" in section or "top-priority" in section.lower()
    # Must say grade must be 4 or lower
    assert "4" in section


def test_C3_validation_policy_caps_grade_for_missing_grounded_handoff():
    """Validation policy must cap grade <= 4 when built_target_set reached but final_variable=None."""
    ctrl = _make_controller(_tmp("C3"))
    base_validation = {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "summary": "built target set",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
        "usefulness_passed": True,
        "partial_value_usable": True,
    }
    # live_progress_summary: built a target set but exhausted with no final variable
    live_summary = {
        "execution_status": "MACRO EXHAUSTED",
        "has_final_variable": False,
        "has_context": True,
        "material_progress": True,
        "handoff_state": "exhausted",
        "value_delivered": "built_target_set",
    }
    execution_validation = {
        "status": "MACRO EXHAUSTED",
        "final_variable": None,
        "observation": (
            "MACRO EXHAUSTED: Resulting set is empty. "
            "walked_entity_to_target set built. "
            "minted_variables: {\"walked_set\": \"#3\"}"
        ),
    }
    result = ctrl._toolgen_apply_validation_policy(
        validation=base_validation,
        execution_validation=execution_validation,
        live_progress_summary=live_summary,
        round_context={"round": 1, "active_strategy_family": "walk_first"},
        tool_plan={"preferred_tool_mode": "progress_tool", "target_concept": "cheese"},
        tool_code="def run(p):\n    pass\ndef self_test():\n    return True\n",
    )
    assert result.get("grade", 10) <= 4, (
        f"Expected grade <= 4 for missing_grounded_handoff, got {result.get('grade')}"
    )
    assert result.get("missing_grounded_handoff_after_real_progress") is True
    assert result.get("repair_mode") == "rewrite_code"
    issues = result.get("issues", [])
    assert any("missing_grounded_handoff_after_real_progress" in str(i) for i in issues), (
        f"Expected missing_grounded_handoff issue as first item, got: {issues}"
    )
    # Must be first item (top-priority)
    assert "missing_grounded_handoff" in str(issues[0])


def test_C4_validation_policy_does_not_flag_if_final_variable_present():
    """Validation policy must NOT flag missing_grounded_handoff when final_variable is present."""
    ctrl = _make_controller(_tmp("C4"))
    base_validation = {
        "grade": 8,
        "issues": [],
        "fixes": [],
        "summary": "built target set and returned it",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
        "usefulness_passed": True,
        "partial_value_usable": True,
    }
    live_summary = {
        "execution_status": "SUCCESS",
        "has_final_variable": True,
        "has_context": True,
        "material_progress": True,
        "handoff_state": "complete",
        "value_delivered": "built_target_set",
    }
    execution_validation = {
        "status": "SUCCESS",
        "final_variable": "#3",
        "observation": "Built target set. minted_variables: {\"walked_set\": \"#3\"}",
    }
    result = ctrl._toolgen_apply_validation_policy(
        validation=base_validation,
        execution_validation=execution_validation,
        live_progress_summary=live_summary,
        round_context={"round": 1, "active_strategy_family": "walk_first"},
        tool_plan={"preferred_tool_mode": "progress_tool", "target_concept": "cheese"},
        tool_code="def run(p):\n    pass\ndef self_test():\n    return True\n",
    )
    assert not result.get("missing_grounded_handoff_after_real_progress", False), (
        "Should not flag missing_grounded_handoff when final_variable is present"
    )


def test_C5_runtime_repair_brief_identifies_built_set_no_handoff():
    """Runtime repair brief must identify built-set-no-handoff as distinct redesign direction."""
    ctrl = _make_controller(_tmp("C5"))
    live_summary = {
        "execution_status": "MACRO EXHAUSTED",
        "has_final_variable": False,
        "has_context": True,
        "material_progress": True,
        "handoff_state": "exhausted",
        "value_delivered": "built_target_set",
        "has_live_result": True,
    }
    brief = ctrl._toolgen_synthesize_runtime_repair_brief(
        live_progress_summary=live_summary,
        round_context={},
        validation={},
    )
    assert brief is not None, "Expected a repair brief for built-set-no-handoff pattern"
    assert brief.get("redesign_direction") == "emit_grounded_handoff_from_built_set"
    assert brief.get("missing_value_type") == "missing_grounded_handoff_after_real_progress"


# ---------------------------------------------------------------------------
# D. ANTI-HARDCODING TESTS
# ---------------------------------------------------------------------------

def test_D1_prompt_contains_label_derivation_mandate():
    """ToolGen prompt must contain LABEL DERIVATION MANDATE with anti-hardcoding rules."""
    assert "LABEL DERIVATION MANDATE" in MACRO_TOOLGEN_USER_KG
    section_start = MACRO_TOOLGEN_USER_KG.find("LABEL DERIVATION MANDATE")
    section = MACRO_TOOLGEN_USER_KG[section_start:section_start + 800]
    assert "f-string" in section.lower() or "f\"" in section
    assert "hardcod" in section.lower()
    # Must have WRONG/RIGHT examples
    assert "WRONG" in section
    assert "RIGHT" in section


def test_D2_validator_prompt_treats_hardcoded_entity_as_generalization_failure():
    """Validator prompt must classify hardcoded entity names as generalization_failure."""
    assert "TASK IMPRINTING" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT or \
           "hardcoded_entity_in_label" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT
    text = TOOLGEN_VALIDATOR_SYSTEM_PROMPT
    # Find the relevant section
    idx = text.find("TASK IMPRINTING")
    if idx < 0:
        idx = text.find("hardcoded_entity_in_label")
    section = text[idx:idx + 600]
    assert "generalization" in section.lower()
    # Must NOT say this is a style issue
    assert "not a style" in section.lower() or "not merely" in section.lower() or \
           "is not a style" in section.lower()


def test_D3_static_checker_flags_hardcoded_entity_in_label():
    """Static checker must flag string literals like 'resolved_Milk' as hardcoded_entity_in_label."""
    # Code must look like a KG macro for the checker to activate.
    # _looks_like_kg_macro requires >= 2 of the KG marker tokens.
    code_with_hardcoded = """\
import json
actions_spec = {}
target_concept = "cheese"
candidate_map = {}
candidate_map["resolved_Milk"] = "#1"
candidate_map["walk_Goat_to_cheese"] = "#2"
"""
    issues = _kg_hardcoded_entity_in_label_issues(code_with_hardcoded)
    assert len(issues) > 0, (
        f"Expected hardcoded_entity_in_label issues for 'resolved_Milk', got none"
    )
    # Verify at least the Milk label was flagged
    assert any("Milk" in issue or "resolved_Milk" in issue for issue in issues), (
        f"Expected 'resolved_Milk' to be flagged, got: {issues}"
    )


def test_D4_static_checker_does_not_flag_payload_derived_labels():
    """Static checker must NOT flag f-string derived labels from payload fields."""
    # Code with dynamically derived labels — these are GOOD
    code_with_dynamic = """\
import json
candidate_map = {}
entities = ["Milk", "Goat"]
target_concept = "cheese"
for entity in entities:
    key = f"resolved_{entity}"
    candidate_map[key] = "#1"
walk_key = f"walk_{entity}_to_{target_concept}"
candidate_map[walk_key] = "#2"
"""
    issues = _kg_hardcoded_entity_in_label_issues(code_with_dynamic)
    # f-string expressions generate JoinedStr nodes, not Constant nodes, so no violations
    assert len(issues) == 0, (
        f"Should not flag f-string derived labels, got: {issues}"
    )


def test_D5_static_checker_flags_hardcoded_in_observation_strings():
    """Static checker must flag hardcoded entity names in observation string literals."""
    code_with_hardcoded_obs = """\
import json
obs = "resolved_Milk anchor set."
candidate_map = {"resolved_Milk": "#1"}
"""
    # This code uses 'kg_utils' / 'actions_spec' patterns to trigger the KG macro check
    code_kg = "actions_spec = {}\ntarget_concept = 'cheese'\n" + code_with_hardcoded_obs
    issues = _kg_hardcoded_entity_in_label_issues(code_kg)
    assert len(issues) > 0, (
        f"Expected issues for hardcoded 'resolved_Milk' in observation string, got none. "
        f"Note: this requires the code to look like a KG macro."
    )


def test_D6_validate_tool_code_rejects_hardcoded_entity_labels():
    """validate_tool_code must reject tools with hardcoded entity names in labels."""
    tool_with_hardcoded = _make_tool("""\
        payload = payload or {}
        entities = payload.get("entities", [])
        actions_spec = payload.get("actions_spec", {})
        target_concept = payload.get("target_concept", "")
        candidate_map = {}
        candidate_map["resolved_Milk"] = "#1"
        candidate_map["walk_Goat_to_cheese"] = "#2"
        return {
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": "MACRO EXHAUSTED: Resulting set is empty. minted_variables: " + json.dumps(candidate_map),
        }
""")
    result = validate_tool_code(tool_with_hardcoded)
    assert not result.success, (
        "Expected validate_tool_code to reject tool with hardcoded entity labels"
    )
    assert "hardcoded_entity_in_label" in (result.error or ""), (
        f"Expected hardcoded_entity_in_label error, got: {result.error}"
    )


def test_D6b_validate_tool_code_accepts_dynamic_labels():
    """validate_tool_code must accept tools with dynamically derived labels."""
    tool_with_dynamic = _make_tool("""\
        payload = payload or {}
        entities = payload.get("entities", [])
        actions_spec = payload.get("actions_spec", {})
        target_concept = payload.get("target_concept", "entity")
        candidate_map = {}
        for entity in entities:
            key = f"resolved_{entity}"
            candidate_map[key] = "#99"
        return {
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": "MACRO EXHAUSTED: Resulting set is empty. minted_variables: " + json.dumps(candidate_map),
        }
""")
    result = validate_tool_code(tool_with_dynamic)
    assert "hardcoded_entity_in_label" not in (result.error or ""), (
        f"Should not reject tool with dynamic labels, got: {result.error}"
    )


# ---------------------------------------------------------------------------
# Prompt-level integrity checks (sanity)
# ---------------------------------------------------------------------------

def test_prompt_mandatory_grounded_handoff_in_self_check():
    """Self-check section of ToolGen prompt must include grounded handoff check."""
    assert "SELF-CHECK BEFORE EMITTING" in MACRO_TOOLGEN_USER_KG
    sc_start = MACRO_TOOLGEN_USER_KG.find("SELF-CHECK BEFORE EMITTING")
    sc_section = MACRO_TOOLGEN_USER_KG[sc_start:sc_start + 2000]
    assert "progress_tool" in sc_section
    assert "final_variable" in sc_section


def test_prompt_self_check_includes_anti_hardcoding():
    """Self-check section must include candidate_map key derivation requirement."""
    sc_start = MACRO_TOOLGEN_USER_KG.find("SELF-CHECK BEFORE EMITTING")
    sc_section = MACRO_TOOLGEN_USER_KG[sc_start:sc_start + 2000]
    assert "candidate_map" in sc_section
    assert "f-string" in sc_section.lower() or "payload" in sc_section


def test_prompt_self_check_includes_live_context_check():
    """Self-check section must include resolve_semantic_filter live-context check."""
    sc_start = MACRO_TOOLGEN_USER_KG.find("SELF-CHECK BEFORE EMITTING")
    sc_section = MACRO_TOOLGEN_USER_KG[sc_start:sc_start + 1500]
    assert "resolve_semantic_filter" in sc_section


def test_prompt_self_check_includes_anchor_selection_check():
    """Self-check section must include anchor-selection-from-plan check."""
    sc_start = MACRO_TOOLGEN_USER_KG.find("SELF-CHECK BEFORE EMITTING")
    sc_section = MACRO_TOOLGEN_USER_KG[sc_start:sc_start + 1500]
    assert "anchor" in sc_section.lower()
    assert "entity_target_concepts" in sc_section


# ---------------------------------------------------------------------------
# E. STEP 1 STABILIZATION: FINALIZATION, PRESERVED-ANCHOR, OPERAND ROLE
# ---------------------------------------------------------------------------

# E1 — built_filter_ready_set triggers missing handoff grade cap (PART A)

def test_E1_built_filter_ready_set_triggers_missing_handoff_cap():
    """Policy must cap grade <= 4 when built_filter_ready_set reached but final_variable=None."""
    ctrl = _make_controller(_tmp("E1"))
    base_validation = {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "summary": "filter-ready set built",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
        "usefulness_passed": True,
        "partial_value_usable": True,
    }
    live_summary = {
        "execution_status": "MACRO EXHAUSTED",
        "has_final_variable": False,
        "has_context": True,
        "material_progress": True,
        "handoff_state": "exhausted",
        "value_delivered": "built_filter_ready_set",
        "has_live_result": True,
    }
    execution_validation = {
        "status": "MACRO EXHAUSTED",
        "final_variable": None,
        "observation": (
            "MACRO EXHAUSTED: Resulting set is empty. "
            "minted_variables: {\"filter_result\": \"#5\"}"
        ),
    }
    result = ctrl._toolgen_apply_validation_policy(
        validation=base_validation,
        execution_validation=execution_validation,
        live_progress_summary=live_summary,
        round_context={"round": 2, "active_strategy_family": "probe_then_commit"},
        tool_plan={"preferred_tool_mode": "progress_tool", "target_concept": "drug"},
        tool_code="def run(p):\n    pass\ndef self_test():\n    return True\n",
    )
    assert result.get("grade", 10) <= 4, (
        f"Expected grade <= 4 for built_filter_ready_set missing handoff, got {result.get('grade')}"
    )
    assert result.get("missing_grounded_handoff_after_real_progress") is True
    assert result.get("repair_mode") == "rewrite_code"


def test_E1b_built_filter_ready_set_in_prompt_rule():
    """BUILT-SET HANDOFF RULE in prompt must include built_filter_ready_set."""
    rule_start = MACRO_TOOLGEN_USER_KG.find("BUILT-SET HANDOFF RULE")
    assert rule_start >= 0, "BUILT-SET HANDOFF RULE section must be present"
    rule_section = MACRO_TOOLGEN_USER_KG[rule_start:rule_start + 1200]
    assert "built_filter_ready_set" in rule_section, (
        "built_filter_ready_set must be listed in BUILT-SET HANDOFF RULE"
    )


def test_E1c_built_filter_ready_set_in_validator_prompt():
    """Validator prompt mandatory grounded handoff section must include built_filter_ready_set."""
    section_start = TOOLGEN_VALIDATOR_SYSTEM_PROMPT.find("MANDATORY GROUNDED HANDOFF")
    assert section_start >= 0
    section = TOOLGEN_VALIDATOR_SYSTEM_PROMPT[section_start:section_start + 1000]
    assert "built_filter_ready_set" in section, (
        "built_filter_ready_set must be listed in validator MANDATORY GROUNDED HANDOFF"
    )


def test_E1d_finalization_code_pattern_in_prompt():
    """FINALIZATION CODE PATTERN section must be present in ToolGen prompt."""
    assert "FINALIZATION CODE PATTERN" in MACRO_TOOLGEN_USER_KG
    fp_start = MACRO_TOOLGEN_USER_KG.find("FINALIZATION CODE PATTERN")
    fp_section = MACRO_TOOLGEN_USER_KG[fp_start:fp_start + 1200]
    # Must show both set-task and count-task patterns
    assert "set-tasks" in fp_section or "set_tasks" in fp_section.lower() or "INTERSECTOR" in fp_section
    assert "count" in fp_section.lower()
    assert "final_variable" in fp_section and "None" in fp_section
    assert "SUCCESS" in fp_section


# E2 — attribute_target_concept_resolved_as_anchor smell (PART C)

def test_E2_smell_detector_flags_attribute_target_as_anchor():
    """Smell detector must flag attribute_target_concept passed to resolve_entity_to_vars."""
    ctrl = _make_controller(_tmp("E2"))
    code = """\
import json
def run(payload):
    attribute_tc = payload.get("attribute_target_concept")
    entities = payload.get("entities") or []
    actions_spec = payload.get("actions_spec") or {}
    res = kg_utils.resolve_entity_to_vars(attribute_tc, None, actions_spec, None, max_k=1)
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    smells = ctrl._toolgen_semantic_code_smells(code)
    assert "attribute_target_concept_resolved_as_anchor" in smells, (
        f"Expected attribute_target_concept_resolved_as_anchor smell, got: {smells}"
    )


def test_E2b_smell_detector_does_not_flag_attribute_used_as_filter():
    """Smell detector must NOT flag attribute_target_concept used as target_concept for walk."""
    ctrl = _make_controller(_tmp("E2b"))
    code = """\
import json
def run(payload):
    attribute_tc = payload.get("attribute_target_concept")
    entities = payload.get("entities") or []
    entity = entities[0] if entities else ""
    actions_spec = payload.get("actions_spec") or {}
    # correct: use attribute_tc as target, not as entity anchor
    res = kg_utils.walk_to_target(actions_spec, ["#1"], attribute_tc, None, max_calls=6)
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    smells = ctrl._toolgen_semantic_code_smells(code)
    assert "attribute_target_concept_resolved_as_anchor" not in smells, (
        f"Should not flag attribute_target_concept used as filter/target, got: {smells}"
    )


def test_E2c_static_checker_rejects_attribute_target_as_anchor():
    """validate_tool_code must reject tools where attribute_target_concept is passed to resolve_entity_to_vars."""
    tool = _make_tool("""\
        payload = payload or {}
        attribute_tc = payload.get("attribute_target_concept")
        entities = payload.get("entities") or []
        actions_spec = payload.get("actions_spec") or {}
        domain_hints = payload.get("domain_hints")
        target_concept = payload.get("target_concept")
        res = kg_utils.resolve_entity_to_vars(attribute_tc, target_concept, actions_spec, domain_hints, max_k=1)
        raw_ids = kg_utils.extract_var_ids(res)
        ids = [v for v in raw_ids if isinstance(v, str) and v.startswith("#")]
        final_var = next((v for v in ids), None)
        if final_var:
            return {"status": "SUCCESS", "final_variable": final_var, "observation": "ok"}
        return {"status": "MACRO EXHAUSTED", "final_variable": None, "observation": "MACRO EXHAUSTED: Resulting set is empty. Suggested action: none; no actionable anchor available. minted_variables: {}"}
""")
    result = validate_tool_code(tool)
    assert not result.success, "Expected validate_tool_code to reject attribute_target as anchor"
    assert "kg_attribute_target_as_anchor" in (result.error or ""), (
        f"Expected kg_attribute_target_as_anchor error, got: {result.error}"
    )


def test_E2d_static_checker_function():
    """_kg_attribute_target_as_anchor_issues must return violations for the pattern."""
    code = """\
import json
actions_spec = {}
target_concept = "drug"
attribute_tc = payload.get("attribute_target_concept")
res = kg_utils.resolve_entity_to_vars(attribute_tc, target_concept, actions_spec, None, max_k=1)
"""
    issues = _kg_attribute_target_as_anchor_issues(code)
    assert len(issues) > 0, (
        f"Expected attribute_target_as_anchor violations, got none"
    )
    assert any("attribute_target_concept" in i or "attribute_tc" in i for i in issues), (
        f"Expected attribute_tc to be named in violation, got: {issues}"
    )


def test_E2e_policy_caps_grade_for_attribute_target_as_anchor():
    """Policy must cap grade <= 3 for attribute_target_concept_resolved_as_anchor smell."""
    ctrl = _make_controller(_tmp("E2e"))
    # Tool code that contains the ATC-as-anchor pattern in its text
    code_with_atc_anchor = """\
import json
def run(payload):
    attribute_tc = payload.get("attribute_target_concept")
    entities = payload.get("entities") or []
    actions_spec = payload.get("actions_spec") or {}
    domain_hints = payload.get("domain_hints")
    for ent in entities:
        kg_utils.resolve_entity_to_vars(attribute_tc, None, actions_spec, domain_hints, max_k=1)
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    base_validation = {
        "grade": 6,
        "issues": [],
        "fixes": [],
        "summary": "attribute tc used as anchor",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
        "usefulness_passed": False,
        "partial_value_usable": False,
    }
    live_summary = {
        "execution_status": "ERROR",
        "has_final_variable": False,
        "material_progress": False,
        "value_delivered": "none",
        "has_live_result": True,
    }
    result = ctrl._toolgen_apply_validation_policy(
        validation=base_validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": "test"},
        live_progress_summary=live_summary,
        round_context={"round": 1, "active_strategy_family": "probe_then_commit"},
        tool_plan={"preferred_tool_mode": "progress_tool", "attribute_target_concept": "formulation"},
        tool_code=code_with_atc_anchor,
    )
    assert result.get("grade", 10) <= 3, (
        f"Expected grade <= 3 for attribute_target_concept_resolved_as_anchor, got {result.get('grade')}"
    )
    issues = result.get("issues") or []
    assert any("attribute_target_concept_resolved_as_anchor" in str(i) for i in issues), (
        f"Expected attribute_target_concept_resolved_as_anchor in issues, got: {issues}"
    )


# E3 — preserved-anchor contract (PART B)

def test_E3_smell_detector_flags_preserved_ids_assigned_to_multiple_anchors():
    """Smell detector must flag when same preserved_ids is assigned to multiple anchor vars."""
    ctrl = _make_controller(_tmp("E3"))
    code = """\
import json
def run(payload):
    preserved_ids = payload.get("variable_list") or []
    entities = payload.get("entities") or []
    actions_spec = payload.get("actions_spec") or {}
    # BUG: same list assigned to both anchors
    anchor_a_ids = preserved_ids
    anchor_b_ids = preserved_ids
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    smells = ctrl._toolgen_semantic_code_smells(code)
    assert "preserved_ids_assigned_to_multiple_anchors" in smells, (
        f"Expected preserved_ids_assigned_to_multiple_anchors smell, got: {smells}"
    )


def test_E3b_smell_detector_does_not_flag_sliced_preserved_anchors():
    """Smell detector must NOT flag per-anchor slicing of preserved vars."""
    ctrl = _make_controller(_tmp("E3b"))
    code = """\
import json
def run(payload):
    variable_list = payload.get("variable_list") or []
    entities = payload.get("entities") or []
    actions_spec = payload.get("actions_spec") or {}
    # CORRECT: each anchor gets its own slice
    anchor_a_ids = [variable_list[0]] if len(variable_list) > 0 else []
    anchor_b_ids = [variable_list[1]] if len(variable_list) > 1 else []
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    smells = ctrl._toolgen_semantic_code_smells(code)
    assert "preserved_ids_assigned_to_multiple_anchors" not in smells, (
        f"Should not flag sliced preserved anchors, got: {smells}"
    )


def test_E3c_preserved_anchor_contract_in_prompt():
    """CANONICAL PRESERVED-ANCHOR CONTRACT section must be in ToolGen prompt."""
    assert "CANONICAL PRESERVED-ANCHOR CONTRACT" in MACRO_TOOLGEN_USER_KG, (
        "CANONICAL PRESERVED-ANCHOR CONTRACT section must be present in MACRO_TOOLGEN_USER_KG"
    )
    section_start = MACRO_TOOLGEN_USER_KG.find("CANONICAL PRESERVED-ANCHOR CONTRACT")
    section = MACRO_TOOLGEN_USER_KG[section_start:section_start + 3000]
    # Must show the correct per-anchor slicing pattern
    assert "variable_list" in section
    assert "resolved_anchors" in section
    # Must list forbidden patterns
    assert "FORBIDDEN" in section
    assert "multiple" in section.lower() or "both" in section.lower()


def test_E3d_preserved_anchor_self_check_in_prompt():
    """SELF-CHECK section must include preserved-anchor contract check."""
    sc_start = MACRO_TOOLGEN_USER_KG.find("SELF-CHECK BEFORE EMITTING")
    sc_section = MACRO_TOOLGEN_USER_KG[sc_start:sc_start + 2000]
    assert "PRESERVED-ANCHOR CHECK" in sc_section or "preserved" in sc_section.lower(), (
        "SELF-CHECK must include preserved-anchor check"
    )
    assert "variable_list" in sc_section


def test_E3e_policy_caps_grade_for_preserved_multianchor_collapse():
    """Policy must cap grade <= 4 for preserved_ids_assigned_to_multiple_anchors."""
    ctrl = _make_controller(_tmp("E3e"))
    code_with_multianchor = """\
import json
def run(payload):
    preserved_ids = payload.get("variable_list") or []
    entities = payload.get("entities") or []
    actions_spec = payload.get("actions_spec") or {}
    anchor_a_ids = preserved_ids
    anchor_b_ids = preserved_ids
    walk_a = kg_utils.walk_to_target(actions_spec, anchor_a_ids, "target", None, max_calls=6)
    walk_b = kg_utils.walk_to_target(actions_spec, anchor_b_ids, "target", None, max_calls=6)
    return {"status": "ERROR", "final_variable": None, "observation": "test"}
def self_test(): return True
"""
    base_validation = {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "summary": "multianchor collapse",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
        "usefulness_passed": False,
        "partial_value_usable": False,
    }
    live_summary = {
        "execution_status": "ERROR",
        "has_final_variable": False,
        "material_progress": False,
        "value_delivered": "none",
        "has_live_result": True,
    }
    result = ctrl._toolgen_apply_validation_policy(
        validation=base_validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": "test"},
        live_progress_summary=live_summary,
        round_context={
            "round": 2,
            "active_strategy_family": "probe_then_commit",
            "best_achieved_state": "resolved_both_anchors",
        },
        tool_plan={"preferred_tool_mode": "progress_tool", "target_concept": "drug"},
        tool_code=code_with_multianchor,
    )
    assert result.get("grade", 10) <= 4, (
        f"Expected grade <= 4 for preserved_ids_assigned_to_multiple_anchors, got {result.get('grade')}"
    )
    issues = result.get("issues") or []
    assert any("preserved_ids_assigned_to_multiple_anchors" in str(i) for i in issues), (
        f"Expected preserved_ids_assigned_to_multiple_anchors in issues, got: {issues}"
    )


# E4 — diagnostic logging fields present in policy output

def test_E4_diagnostic_fields_present_in_policy_output():
    """Policy result must include Step 1 diagnostic logging fields."""
    ctrl = _make_controller(_tmp("E4"))
    base_validation = {
        "grade": 5,
        "issues": [],
        "fixes": [],
        "summary": "test",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
        "usefulness_passed": True,
        "partial_value_usable": True,
    }
    live_summary = {
        "execution_status": "MACRO EXHAUSTED",
        "has_final_variable": False,
        "material_progress": True,
        "value_delivered": "built_target_set",
        "has_live_result": True,
    }
    result = ctrl._toolgen_apply_validation_policy(
        validation=base_validation,
        execution_validation={"status": "MACRO EXHAUSTED", "final_variable": None, "observation": "MACRO EXHAUSTED: Resulting set is empty. minted_variables: {}"},
        live_progress_summary=live_summary,
        round_context={"round": 1, "active_strategy_family": "probe_then_commit"},
        tool_plan={"preferred_tool_mode": "progress_tool"},
        tool_code="def run(p):\n    pass\ndef self_test():\n    return True\n",
    )
    required_fields = [
        "required_state_reached",
        "finalization_attempted",
        "finalization_block_reason",
        "preserved_vars_present_in_code",
        "used_preserved_vars",
        "fallback_reresolve_used",
        "preserved_state_missing_error",
        "reresolve_as_main_output",
        "preserved_ids_multianchor_collapse",
        "attribute_target_present_in_code",
        "attribute_used_as_anchor",
        "operand_role_collapse_detected",
        "count_called",
        "count_variable_extracted",
        "returned_count_variable",
    ]
    missing = [f for f in required_fields if f not in result]
    assert not missing, f"Missing diagnostic fields in policy result: {missing}"


# ---------------------------------------------------------------------------
# PART L — kg-relhint-fallback-pass: relation hint extraction
# ---------------------------------------------------------------------------

def test_L1_relation_hints_extracted_from_plan_steps():
    """Dotted KG schema paths with underscores are extracted from plan prose."""
    steps = [
        "Use food.cheese_milk_source.cheeses to walk from anchor to cheeses",
        "Then apply spaceflight.satellite_manufacturer.spacecraft_manufactured",
    ]
    hints = ControllerOrchestratorMixin._kg_extract_relation_hints(steps)
    assert "food.cheese_milk_source.cheeses" in hints
    assert "spaceflight.satellite_manufacturer.spacecraft_manufactured" in hints


def test_L2_relation_hints_excludes_no_underscore():
    """Dotted paths without underscores are NOT extracted (not KG schema paths)."""
    steps = ["Use common.topic.article and org.name.value to find things"]
    hints = ControllerOrchestratorMixin._kg_extract_relation_hints(steps)
    # Neither contains underscores
    assert "common.topic.article" not in hints
    assert "org.name.value" not in hints


def test_L3_relation_hints_capped_at_12():
    """Relation hint extraction is capped at 12 unique hints."""
    steps = [
        f"Use ns{i}.some_thing_{i}.target to walk entity" for i in range(20)
    ]
    hints = ControllerOrchestratorMixin._kg_extract_relation_hints(steps)
    assert len(hints) <= 12


def test_L4_relation_hints_empty_plan_returns_empty():
    hints = ControllerOrchestratorMixin._kg_extract_relation_hints([])
    assert hints == []


def test_L5_relation_hints_attached_to_build_tool_plan():
    """_build_tool_plan attaches relation_hints extracted from topological_execution_plan."""
    mixin = ControllerOrchestratorMixin()
    payload = {
        "task_type": "knowledge_graph",
        "target_archetype": "COUNTING_INTERSECTOR",
        "entities": ["CNES", "Astrium"],
        "attribute_target_concept": "",
        "topological_execution_plan": [
            "Walk CNES via spaceflight.satellite_manufacturer.spacecraft_manufactured",
            "Walk Astrium via spaceflight.satellite_manufacturer.spacecraft_manufactured",
            "Intersect and count",
        ],
    }
    plan = mixin._build_tool_plan(payload)
    assert "relation_hints" in plan
    assert "spaceflight.satellite_manufacturer.spacecraft_manufactured" in plan["relation_hints"]


# ---------------------------------------------------------------------------
# PART M — kg-relhint-fallback-pass: family fallback split
# ---------------------------------------------------------------------------

def test_M1_fallback_split_fires_3_entities_no_hints():
    """With 3 entities, no entity_target_concepts, attribute present: fallback split."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=["Gouda", "Edam", "semi-firm"],
        attribute_target_concept="food.cheese.texture",
    )
    assert result["anchor_operands"] == ["Gouda", "Edam"]
    assert result["filter_value_operand"] == "semi-firm"
    assert result["family_fallback_split_used"] is True
    assert result["family_normalization_fallback_used"] is True


def test_M2_fallback_split_does_not_fire_with_2_entities():
    """With exactly 2 entities (even with attribute), fallback does not fire."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=2,
        has_attribute_target=True,
        entities=["Gouda", "Edam"],
        attribute_target_concept="food.cheese.texture",
    )
    assert result.get("family_fallback_split_used") is False
    assert result["anchor_operands"] == ["Gouda", "Edam"]
    assert result.get("filter_value_operand") is None


def test_M3_fallback_split_does_not_fire_without_attribute():
    """Without has_attribute_target, fallback split stays dormant."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="COUNTING_INTERSECTOR",
        entity_count=3,
        has_attribute_target=False,
        entities=["A", "B", "C"],
        attribute_target_concept="",
    )
    assert result.get("family_fallback_split_used") is False
    assert "C" in result["anchor_operands"]


def test_M4_fallback_split_does_not_override_entity_target_concepts():
    """When entity_target_concepts provides a real hint, fallback is skipped."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=["Gouda", "Edam", "semi-firm"],
        attribute_target_concept="food.cheese.texture",
        entity_target_concepts=[
            "spaceflight.spacecraft",
            "spaceflight.spacecraft",
            "food.cheese.texture",  # matches ATC → goes to filter_value_operand directly
        ],
    )
    # Primary split handled it; fallback should not be needed
    assert result["anchor_operands"] == ["Gouda", "Edam"]
    assert result["filter_value_operand"] == "semi-firm"
    # Fallback was not the trigger here
    assert result.get("family_fallback_split_used") is False


def test_M5_fallback_split_reason_logged():
    """family_fallback_split_reason is set when fallback fires."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=["X", "Y", "Z"],
        attribute_target_concept="ns.concept.attr",
    )
    assert result["family_fallback_split_used"] is True
    assert result["family_fallback_split_reason"] is not None


# ---------------------------------------------------------------------------
# PART N — kg-relhint-fallback-pass: skeleton hardening (PART 3)
# ---------------------------------------------------------------------------

def test_N1_skeleton_includes_relation_hints_in_header():
    """When relation_hints present, skeleton header includes them."""
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block({
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["CNES", "Astrium"],
        "filter_operand": "",
        "filter_value_operand": "",
        "terminal_artifact_kind": "count_variable",
        "target_concept": "spaceflight.spacecraft",
        "attribute_target_concept": "",
        "required_next_stage": "built_target_set",
        "relation_hints": ["spaceflight.satellite_manufacturer.spacecraft_manufactured"],
    })
    assert "relation_hints" in block
    assert "spaceflight.satellite_manufacturer.spacecraft_manufactured" in block
    assert "domain_hints" in block


def test_N2_skeleton_includes_required_next_stage():
    """required_next_stage is shown in the skeleton header."""
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block({
        "template_family": "two_anchor_intersect_filter_set",
        "anchor_operands": ["A", "B"],
        "filter_operand": "food.cheese.texture",
        "filter_value_operand": "semi-firm",
        "terminal_artifact_kind": "filtered_set_variable",
        "target_concept": "food.cheese",
        "attribute_target_concept": "food.cheese.texture",
        "required_next_stage": "built_filter_ready_set",
        "relation_hints": [],
    })
    assert "built_filter_ready_set" in block
    assert "required_next_stage" in block


def test_N3_skeleton_walk_step_mentions_relation_hints_when_present():
    """Walk step in skeleton body references relation_hints when they exist."""
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block({
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["X", "Y"],
        "filter_operand": "",
        "filter_value_operand": "",
        "terminal_artifact_kind": "count_variable",
        "target_concept": "some.concept",
        "attribute_target_concept": "",
        "required_next_stage": "built_target_set",
        "relation_hints": ["foo.bar_baz.qux"],
    })
    # The walk step should mention the hints
    assert "foo.bar_baz.qux" in block
    assert "relation_hints" in block


def test_N4_skeleton_no_relation_hints_hint_line_absent():
    """When relation_hints is empty, no relation_hints annotation in header."""
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block({
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["X", "Y"],
        "filter_operand": "",
        "filter_value_operand": "",
        "terminal_artifact_kind": "count_variable",
        "target_concept": "some.concept",
        "attribute_target_concept": "",
        "required_next_stage": "built_target_set",
        "relation_hints": [],
    })
    # No relation_hints line in header when empty
    assert "relation_hints: []" not in block


# ---------------------------------------------------------------------------
# PART O — kg-relhint-fallback-pass: new validator smells (PART 5)
# ---------------------------------------------------------------------------

class _CombinedMixin(ControllerOrchestratorMixin, ControllerToolgenMixin):
    pass

_combined = _CombinedMixin()


def test_O1_wrong_anchor_count_for_family_fires_when_anchor_operands_missing():
    """Smell fires when two-anchor family tool code doesn't read anchor_operands."""
    tool_code = """
def run(payload):
    entities = payload.get('entities') or []
    anchor_a = entities[0]
    anchor_b = entities[1]
    return {'status': 'SUCCESS', 'final_variable': '#2'}
"""
    plan = {
        "task_type": "knowledge_graph",
        "entities": ["A", "B"],
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["A", "B"],
        "terminal_artifact_kind": "count_variable",
        "attribute_target_concept": "",
    }
    smells = _combined._toolgen_dynamic_plan_code_smells(tool_code, plan)
    assert "wrong_anchor_count_for_family" in smells


def test_O2_wrong_anchor_count_for_family_absent_when_anchor_operands_present():
    """Smell is absent when tool code reads anchor_operands from payload."""
    tool_code = """
def run(payload):
    anchor_ops = payload.get('anchor_operands') or []
    return {'status': 'SUCCESS', 'final_variable': '#2'}
"""
    plan = {
        "task_type": "knowledge_graph",
        "entities": ["A", "B"],
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["A", "B"],
        "terminal_artifact_kind": "count_variable",
        "attribute_target_concept": "",
    }
    smells = _combined._toolgen_dynamic_plan_code_smells(tool_code, plan)
    assert "wrong_anchor_count_for_family" not in smells


def test_O3_relation_hints_ignored_fires_when_hints_present_but_unused():
    """Smell fires when relation_hints are in the plan but not in the tool code."""
    tool_code = """
def run(payload):
    anchor_ops = payload.get('anchor_operands') or []
    domain_hints = payload.get('domain_hints') or []
    return {'status': 'SUCCESS', 'final_variable': '#2'}
"""
    plan = {
        "task_type": "knowledge_graph",
        "entities": ["A", "B"],
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["A", "B"],
        "terminal_artifact_kind": "count_variable",
        "attribute_target_concept": "",
        "topological_execution_plan": [
            "Walk using spaceflight.satellite_manufacturer.spacecraft_manufactured",
        ],
    }
    smells = _combined._toolgen_dynamic_plan_code_smells(tool_code, plan)
    assert "relation_hints_ignored" in smells


def test_O4_relation_hints_ignored_absent_when_no_hints():
    """Smell does not fire when there are no relation_hints to use."""
    tool_code = """
def run(payload):
    anchor_ops = payload.get('anchor_operands') or []
    return {'status': 'SUCCESS', 'final_variable': '#2'}
"""
    plan = {
        "task_type": "knowledge_graph",
        "entities": ["A", "B"],
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["A", "B"],
        "terminal_artifact_kind": "count_variable",
        "attribute_target_concept": "",
        "topological_execution_plan": [],
    }
    smells = _combined._toolgen_dynamic_plan_code_smells(tool_code, plan)
    assert "relation_hints_ignored" not in smells


def test_O5_relation_hints_ignored_absent_when_hints_referenced():
    """Smell does not fire when tool code references relation_hints."""
    tool_code = """
def run(payload):
    anchor_ops = payload.get('anchor_operands') or []
    hints = payload.get('relation_hints') or []
    return {'status': 'SUCCESS', 'final_variable': '#2'}
"""
    plan = {
        "task_type": "knowledge_graph",
        "entities": ["A", "B"],
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["A", "B"],
        "terminal_artifact_kind": "count_variable",
        "attribute_target_concept": "",
        "topological_execution_plan": [
            "Walk using spaceflight.satellite_manufacturer.spacecraft_manufactured",
        ],
    }
    smells = _combined._toolgen_dynamic_plan_code_smells(tool_code, plan)
    assert "relation_hints_ignored" not in smells


def test_O6_family_repair_instruction_includes_relation_hints():
    """_toolgen_family_repair_instruction embeds relation_hints in its text."""
    instr = ControllerToolgenMixin._toolgen_family_repair_instruction(
        template_family="two_anchor_intersect_count",
        anchor_operands=["CNES", "Astrium"],
        relation_hints=["spaceflight.satellite_manufacturer.spacecraft_manufactured"],
    )
    assert "relation_hints" in instr
    assert "spaceflight.satellite_manufacturer.spacecraft_manufactured" in instr
