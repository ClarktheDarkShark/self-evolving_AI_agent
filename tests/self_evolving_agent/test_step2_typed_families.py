"""Step 2 tests: typed family selection, operand roles, terminal artifact enforcement.

Tests cover all Step 2 parts:
  A. _kg_derive_template_family_and_roles derivation
  B. anchor_operands / filter_operand / terminal_artifact_kind propagation
  C. positional_entity_role_guessing smell detection and policy caps
  D. family-aware repair instructions
  E. prompt/validator sections
  F. diagnostic logging fields
"""
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.self_evolving_agent.controller_orchestrator import ControllerOrchestratorMixin
from src.self_evolving_agent.controller_toolgen import ControllerToolgenMixin
from src.self_evolving_agent.controller_prompts import (
    MACRO_TOOLGEN_USER_KG,
    TOOLGEN_VALIDATOR_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _Mixin(ControllerOrchestratorMixin, ControllerToolgenMixin):
    """Minimal concrete class for testing mixin methods."""
    def _resolved_environment_label(self) -> str:
        return "knowledge_graph"


_mixin = _Mixin()


def _smells(code: str) -> list[str]:
    return ControllerToolgenMixin._toolgen_semantic_code_smells(code)


def _make_payload(**kwargs) -> dict:
    base = {
        "entities": ["EntityA", "EntityB"],
        "target_concept": "food.cheese",
        "actions_spec": {},
        "task_text": "test",
        "asked_for": "answer",
        "trace": [],
        "run_id": "r1",
        "state_dir": "./state",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# PART A — _kg_derive_template_family_and_roles
# ---------------------------------------------------------------------------

def test_A1_counting_intersector_maps_to_two_anchor_count():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="COUNTING_INTERSECTOR",
        entity_count=2,
        has_attribute_target=False,
        entities=["A", "B"],
        attribute_target_concept="",
    )
    assert result["template_family"] == "two_anchor_intersect_count"
    assert result["terminal_artifact_kind"] == "count_variable"


def test_A2_counter_single_entity_maps_to_one_anchor():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="COUNTER",
        entity_count=1,
        has_attribute_target=False,
        entities=["A"],
        attribute_target_concept="",
    )
    assert result["template_family"] == "one_anchor_filter_then_count_or_progress"
    assert result["terminal_artifact_kind"] == "set_variable"


def test_A3_attribute_intersector_maps_to_extract_attribute():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="ATTRIBUTE_INTERSECTOR",
        entity_count=2,
        has_attribute_target=True,
        entities=["A", "B"],
        attribute_target_concept="food.cheese.milk_source",
    )
    assert result["template_family"] == "two_anchor_intersect_extract_attribute"
    assert result["terminal_artifact_kind"] == "attribute_values"


def test_A3b_attribute_intersector_with_ancestor_hints_keeps_both_anchors():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="ATTRIBUTE_INTERSECTOR",
        entity_count=2,
        has_attribute_target=True,
        entities=["Aspirin", "Ibuprofen"],
        attribute_target_concept="medicine.drug.dosage_form",
        entity_target_concepts=["medicine.drug", "medicine.drug"],
    )
    assert result["template_family"] == "two_anchor_intersect_extract_attribute"
    assert result["anchor_operands"] == ["Aspirin", "Ibuprofen"]
    assert result.get("filter_value_operand") is None
    assert result["terminal_artifact_kind"] == "attribute_values"


def test_A3c_extract_attribute_family_handoff_floor_is_target_set():
    plan = _mixin._build_tool_plan(
        {
            "target_archetype": "ATTRIBUTE_INTERSECTOR",
            "entities": ["Aspirin", "Ibuprofen"],
            "entity_target_concepts": ["medicine.drug", "medicine.drug"],
            "target_concept": "medicine.drug",
            "attribute_target_concept": "medicine.drug.dosage_form",
            "topological_execution_plan": [
                "1. Resolve both anchors.",
                "2. Walk both anchors to the target drug set.",
                "3. Intersect the target sets.",
                "4. Extract the dosage-form attribute from the intersection.",
            ],
        }
    )

    assert plan["template_family"] == "two_anchor_intersect_extract_attribute"
    assert _mixin._toolgen_progress_minimum_handoff_state_for_plan(plan) == "built_target_set"


def test_A3d_extract_attribute_handoff_floor_uses_nested_tool_plan():
    plan = _mixin._build_tool_plan(
        {
            "target_archetype": "ATTRIBUTE_INTERSECTOR",
            "entities": ["Aspirin", "Ibuprofen"],
            "entity_target_concepts": ["medicine.drug", "medicine.drug"],
            "target_concept": "medicine.drug",
            "attribute_target_concept": "medicine.drug.dosage_form",
            "topological_execution_plan": [
                "1. Resolve each drug anchor.",
                "2. Walk both anchors to the target drug set.",
                "3. Intersect the target sets.",
                "4. Extract the dosage-form attribute from the intersection.",
            ],
        }
    )

    assert (
        _mixin._toolgen_progress_minimum_handoff_state_for_plan({"tool_plan": plan})
        == "built_target_set"
    )


def test_A4_intersector_two_entities_maps_to_filter_set():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=2,
        has_attribute_target=True,
        entities=["A", "B"],
        attribute_target_concept="food.cheese.milk_source",
    )
    assert result["template_family"] == "two_anchor_intersect_filter_set"
    assert result["terminal_artifact_kind"] == "filtered_set_variable"


def test_A5_intersector_single_entity_maps_to_one_anchor():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=1,
        has_attribute_target=False,
        entities=["A"],
        attribute_target_concept="",
    )
    assert result["template_family"] == "one_anchor_filter_then_count_or_progress"


def test_A6_fallback_split_demotes_last_entity_when_3_entities_no_hints():
    # PART 2: With 3 entities, no entity_target_concepts, and has_attribute_target=True,
    # the family fallback split should demote the last entity to filter_value_operand.
    entities = ["Goat", "Milk", "France"]
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=entities,
        attribute_target_concept="food.cheese.milk_source",
    )
    # Fallback split fires: first 2 are anchors, last is filter_value_operand
    assert result["anchor_operands"] == ["Goat", "Milk"]
    assert result["filter_value_operand"] == "France"
    assert result["family_fallback_split_used"] is True


def test_A7_filter_operand_is_attribute_target_concept():
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=2,
        has_attribute_target=True,
        entities=["A", "B"],
        attribute_target_concept="food.cheese.milk_source",
    )
    assert result["filter_operand"] == "food.cheese.milk_source"


def test_A8_build_tool_plan_injects_family_fields():
    payload = _make_payload(
        target_archetype="COUNTING_INTERSECTOR",
        attribute_target_concept="",
    )
    plan = _mixin._build_tool_plan(payload)
    assert "template_family" in plan
    assert plan["template_family"] == "two_anchor_intersect_count"
    assert "anchor_operands" in plan
    assert "terminal_artifact_kind" in plan
    assert plan["terminal_artifact_kind"] == "count_variable"


def test_A9_build_tool_plan_attribute_intersector():
    payload = _make_payload(
        target_archetype="ATTRIBUTE_INTERSECTOR",
        attribute_target_concept="food.cheese.milk_source",
    )
    plan = _mixin._build_tool_plan(payload)
    assert plan["template_family"] == "two_anchor_intersect_extract_attribute"
    assert plan["filter_operand"] == "food.cheese.milk_source"


# ---------------------------------------------------------------------------
# PART B — anchor_operands / filter_operand propagation to payload
# ---------------------------------------------------------------------------

def test_B1_round_strategy_context_propagates_family_fields():
    payload = _make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.milk_source",
    )
    result = _mixin._toolgen_apply_round_strategy_context(payload, {})
    assert "template_family" in result
    assert result["template_family"] == "two_anchor_intersect_filter_set"
    assert "anchor_operands" in result
    assert "filter_operand" in result
    assert result["filter_operand"] == "food.cheese.milk_source"
    assert "terminal_artifact_kind" in result
    assert result["terminal_artifact_kind"] == "filtered_set_variable"


def test_B2_round_strategy_context_propagates_to_toolgen_retry_context():
    payload = _make_payload(target_archetype="COUNTING_INTERSECTOR")
    result = _mixin._toolgen_apply_round_strategy_context(payload, {})
    retry_ctx = result.get("toolgen_retry_context") or {}
    assert retry_ctx.get("template_family") == "two_anchor_intersect_count"
    assert retry_ctx.get("terminal_artifact_kind") == "count_variable"


def test_B3_anchor_operands_in_toolgen_retry_context_matches_entities():
    entities = ["Goat", "France"]
    payload = _make_payload(entities=entities, target_archetype="INTERSECTOR")
    result = _mixin._toolgen_apply_round_strategy_context(payload, {})
    retry_ctx = result.get("toolgen_retry_context") or {}
    assert retry_ctx.get("anchor_operands") == entities


# ---------------------------------------------------------------------------
# PART C — positional_entity_role_guessing smell
# ---------------------------------------------------------------------------

_CODE_POSITIONAL_INDEX = """
# tool_name: test_positional_index_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: []
# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}

\"\"\"KG macro.\"\"\"
import json

def run(payload: dict) -> dict:
    \"\"\"
    contract guard: test.
    prereqs: kg_utils available.
    limitations: test.
    \"\"\"
    try:
        entities = payload.get("entities") or []
        anchor_a = entities[0]
        anchor_b = entities[1]
        return {"status": "ERROR", "final_variable": None, "observation": "test"}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": str(e)}

def self_test() -> bool:
    return True
"""

_CODE_POSITIONAL_COUNTER = """
# tool_name: test_positional_counter_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: []
# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}

\"\"\"KG macro.\"\"\"
import json

def run(payload: dict) -> dict:
    \"\"\"
    contract guard: test.
    prereqs: kg_utils available.
    limitations: test.
    \"\"\"
    try:
        entities = payload.get("entities") or []
        anchors = []
        idx = 0
        for ent in entities:
            if idx < 2:
                anchors.append(ent)
                idx += 1
        return {"status": "ERROR", "final_variable": None, "observation": "test"}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": str(e)}

def self_test() -> bool:
    return True
"""

_CODE_CORRECT_ANCHOR_OPERANDS = """
# tool_name: test_correct_anchor_operands_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: []
# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}

\"\"\"KG macro.\"\"\"
import json

def run(payload: dict) -> dict:
    \"\"\"
    contract guard: test.
    prereqs: kg_utils available.
    limitations: test.
    \"\"\"
    try:
        tool_plan = payload.get("tool_plan") or {}
        anchor_operands = tool_plan.get("anchor_operands") or payload.get("entities") or []
        entity_target_concepts = payload.get("entity_target_concepts") or []
        for i, ent in enumerate(anchor_operands):
            hint = entity_target_concepts[i] if i < len(entity_target_concepts) else None
            pass
        return {"status": "ERROR", "final_variable": None, "observation": "test"}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": str(e)}

def self_test() -> bool:
    return True
"""


def test_C1_smell_detects_entities_direct_index():
    smells = _smells(_CODE_POSITIONAL_INDEX)
    assert "positional_entity_role_guessing" in smells


def test_C2_smell_detects_loop_counter_idx_lt_2():
    smells = _smells(_CODE_POSITIONAL_COUNTER)
    assert "positional_entity_role_guessing" in smells


def test_C3_no_false_positive_for_anchor_operands_iteration():
    smells = _smells(_CODE_CORRECT_ANCHOR_OPERANDS)
    assert "positional_entity_role_guessing" not in smells


def test_C4_policy_caps_grade_for_positional_guessing():
    """Policy must cap grade <= 3 when positional_entity_role_guessing is detected."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.milk_source",
    ))
    validation = {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "repair_mode": "none",
        "material_progress": False,
        "partial_value_usable": False,
        "bankable_partial_progress": False,
        "required_next_achieved_state": "built_filter_ready_set",
        "semantic_code_smells": ["positional_entity_role_guessing"],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code=_CODE_POSITIONAL_INDEX,
    )
    assert result["grade"] <= 3
    assert result["repair_mode"] == "rewrite_code"
    assert result["partial_value_usable"] is False


def test_C5_positional_smell_in_blocking_kg_smells():
    """positional_entity_role_guessing must be in blocking_kg_smells — checked via source."""
    import inspect
    src = inspect.getsource(ControllerToolgenMixin._toolgen_validate_candidate_tool)
    assert "positional_entity_role_guessing" in src


def test_C6_positional_repair_instruction_mentions_anchor_operands():
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.milk_source",
    ))
    validation = {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "repair_mode": "none",
        "material_progress": False,
        "partial_value_usable": False,
        "bankable_partial_progress": False,
        "required_next_achieved_state": "built_filter_ready_set",
        "semantic_code_smells": ["positional_entity_role_guessing"],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code=_CODE_POSITIONAL_INDEX,
    )
    pri = result.get("primary_repair_instruction") or ""
    assert "anchor_operands" in pri.lower() or "OPERAND ROLE REPAIR" in pri


# ---------------------------------------------------------------------------
# PART D — family-aware repair instructions
# ---------------------------------------------------------------------------

def test_D1_repair_instruction_filter_set_family():
    brief = {
        "missing_value_type": "no_classified_value",
        "redesign_direction": "rewrite_toward_first_valuable_state",
        "template_family": "two_anchor_intersect_filter_set",
        "terminal_artifact_kind": "filtered_set_variable",
        "anchor_operands": ["Goat", "France"],
        "filter_operand": "food.cheese.milk_source",
        "secondary_issues": [],
        "runtime_failure_summary": "no value",
    }
    instruction = ControllerToolgenMixin._toolgen_runtime_repair_instruction(brief)
    assert "two_anchor_intersect_filter_set" in instruction
    assert "filtered_set_variable" in instruction or "filter" in instruction.lower()
    assert "resolve_semantic_filter" in instruction


def test_D2_repair_instruction_count_family():
    brief = {
        "missing_value_type": "no_classified_value",
        "redesign_direction": "rewrite_toward_first_valuable_state",
        "template_family": "two_anchor_intersect_count",
        "terminal_artifact_kind": "count_variable",
        "anchor_operands": ["Goat", "France"],
        "filter_operand": "",
        "secondary_issues": [],
        "runtime_failure_summary": "no value",
    }
    instruction = ControllerToolgenMixin._toolgen_runtime_repair_instruction(brief)
    assert "two_anchor_intersect_count" in instruction
    assert "count_variable" in instruction


def test_D3_repair_instruction_extract_attribute_family():
    brief = {
        "missing_value_type": "no_classified_value",
        "redesign_direction": "rewrite_toward_first_valuable_state",
        "template_family": "two_anchor_intersect_extract_attribute",
        "terminal_artifact_kind": "attribute_values",
        "anchor_operands": ["Goat", "France"],
        "filter_operand": "food.cheese.milk_source",
        "secondary_issues": [],
        "runtime_failure_summary": "no value",
    }
    instruction = ControllerToolgenMixin._toolgen_runtime_repair_instruction(brief)
    assert "extract_attribute" in instruction.lower() or "attribute" in instruction.lower()


def test_D4_repair_instruction_one_anchor_family():
    brief = {
        "missing_value_type": "no_classified_value",
        "redesign_direction": "rewrite_toward_first_valuable_state",
        "template_family": "one_anchor_filter_then_count_or_progress",
        "terminal_artifact_kind": "set_variable",
        "anchor_operands": ["France"],
        "filter_operand": "",
        "secondary_issues": [],
        "runtime_failure_summary": "no value",
    }
    instruction = ControllerToolgenMixin._toolgen_runtime_repair_instruction(brief)
    assert "one_anchor_filter_then_count_or_progress" in instruction


def test_D5_repair_instruction_no_family_falls_back_to_generic():
    brief = {
        "missing_value_type": "no_classified_value",
        "redesign_direction": "rewrite_toward_first_valuable_state",
        "secondary_issues": [],
        "runtime_failure_summary": "no value",
    }
    instruction = ControllerToolgenMixin._toolgen_runtime_repair_instruction(brief)
    # Should still return something meaningful
    assert len(instruction) > 20


def test_D6_count_family_repair_does_not_mention_filter_ready_set():
    """Count family repair must NOT tell tool to build built_filter_ready_set."""
    brief = {
        "missing_value_type": "no_classified_value",
        "redesign_direction": "rewrite_toward_first_valuable_state",
        "template_family": "two_anchor_intersect_count",
        "terminal_artifact_kind": "count_variable",
        "anchor_operands": ["A", "B"],
        "filter_operand": "",
        "secondary_issues": [],
        "runtime_failure_summary": "no value",
    }
    instruction = ControllerToolgenMixin._toolgen_runtime_repair_instruction(brief)
    assert "built_filter_ready_set" not in instruction


# ---------------------------------------------------------------------------
# PART E — prompt sections
# ---------------------------------------------------------------------------

def test_E1_typed_operand_role_section_in_prompt():
    assert "TYPED OPERAND-ROLE CONTRACT" in MACRO_TOOLGEN_USER_KG


def test_E2_family_stage_orders_in_prompt():
    assert "two_anchor_intersect_filter_set" in MACRO_TOOLGEN_USER_KG
    assert "two_anchor_intersect_count" in MACRO_TOOLGEN_USER_KG
    assert "two_anchor_intersect_extract_attribute" in MACRO_TOOLGEN_USER_KG
    assert "one_anchor_filter_then_count_or_progress" in MACRO_TOOLGEN_USER_KG


def test_E3_terminal_artifact_kind_in_prompt():
    assert "terminal_artifact_kind" in MACRO_TOOLGEN_USER_KG
    assert "filtered_set_variable" in MACRO_TOOLGEN_USER_KG
    assert "count_variable" in MACRO_TOOLGEN_USER_KG


def test_E4_positional_ban_in_prompt():
    assert "entities[0]" in MACRO_TOOLGEN_USER_KG
    assert "idx < 2" in MACRO_TOOLGEN_USER_KG
    assert "counter >= 2" in MACRO_TOOLGEN_USER_KG


def test_E5_anchor_operands_in_prompt():
    assert "anchor_operands" in MACRO_TOOLGEN_USER_KG


def test_E6_filter_operand_in_prompt():
    assert "filter_operand" in MACRO_TOOLGEN_USER_KG


def test_E7_typed_role_check_in_self_check():
    sc_start = MACRO_TOOLGEN_USER_KG.index("### 13. SELF-CHECK")
    sc_section = MACRO_TOOLGEN_USER_KG[sc_start : sc_start + 3000]
    assert "TYPED ROLE CHECK" in sc_section
    assert "TERMINAL ARTIFACT CHECK" in sc_section
    assert "FILTER OPERAND CHECK" in sc_section


def test_E8_positional_role_guessing_in_validator():
    assert "POSITIONAL ENTITY ROLE GUESSING" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT


def test_E9_terminal_artifact_mismatch_in_validator():
    assert "TERMINAL ARTIFACT FAMILY MISMATCH" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT


def test_E10_payload_optional_includes_family_fields():
    assert "template_family" in MACRO_TOOLGEN_USER_KG
    assert "anchor_operands" in MACRO_TOOLGEN_USER_KG
    assert "filter_operand" in MACRO_TOOLGEN_USER_KG
    assert "terminal_artifact_kind" in MACRO_TOOLGEN_USER_KG


# ---------------------------------------------------------------------------
# PART F — diagnostic logging fields
# ---------------------------------------------------------------------------

def test_F1_diagnostic_fields_present_in_policy_output():
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.milk_source",
    ))
    validation = {
        "grade": 5,
        "issues": [],
        "fixes": [],
        "repair_mode": "none",
        "material_progress": False,
        "partial_value_usable": False,
        "bankable_partial_progress": False,
        "semantic_code_smells": [],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    required_fields = [
        "template_family_selected",
        "terminal_artifact_expected",
        "terminal_artifact_returned",
        "terminal_artifact_match",
        "positional_role_guess_detected",
        "operand_roles_source",
        "family_repair_instruction_emitted",
    ]
    for field in required_fields:
        assert field in result, f"Missing diagnostic field: {field}"


def test_F2_template_family_selected_matches_plan():
    tool_plan = _mixin._build_tool_plan(_make_payload(target_archetype="COUNTING_INTERSECTOR"))
    validation = {
        "grade": 5,
        "issues": [],
        "fixes": [],
        "repair_mode": "none",
        "material_progress": False,
        "partial_value_usable": False,
        "bankable_partial_progress": False,
        "semantic_code_smells": [],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    assert result["template_family_selected"] == "two_anchor_intersect_count"
    assert result["terminal_artifact_expected"] == "count_variable"


def test_F3_operand_roles_source_is_typed_slots_when_anchor_operands_present():
    tool_plan = _mixin._build_tool_plan(_make_payload(target_archetype="INTERSECTOR"))
    validation = {
        "grade": 5,
        "issues": [],
        "fixes": [],
        "repair_mode": "none",
        "material_progress": False,
        "partial_value_usable": False,
        "bankable_partial_progress": False,
        "semantic_code_smells": [],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    assert result["operand_roles_source"] == "typed_slots"


def test_F4_positional_role_guess_detected_true_when_smell_present():
    tool_plan = _mixin._build_tool_plan(_make_payload(target_archetype="INTERSECTOR"))
    validation = {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "repair_mode": "none",
        "material_progress": False,
        "partial_value_usable": False,
        "bankable_partial_progress": False,
        "semantic_code_smells": ["positional_entity_role_guessing"],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code=_CODE_POSITIONAL_INDEX,
    )
    assert result["positional_role_guess_detected"] is True


# ---------------------------------------------------------------------------
# PART G — anchor_operands / filter_value_operand split (PART 1)
# ---------------------------------------------------------------------------

def test_G1_split_function_separates_filter_value_from_anchors():
    """_kg_split_anchor_vs_filter_operands must put matching entity in filter_value."""
    anchors, filters = ControllerOrchestratorMixin._kg_split_anchor_vs_filter_operands(
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
        attribute_target_concept="food.cheese.texture",
    )
    assert "semi-firm" in filters
    assert "semi-firm" not in anchors
    assert "Goat" in anchors
    assert "cows" in anchors


def test_G2_split_function_no_match_all_go_to_anchors():
    """Without matching hints, all entities are anchors."""
    anchors, filters = ControllerOrchestratorMixin._kg_split_anchor_vs_filter_operands(
        entities=["Goat", "France"],
        entity_target_concepts=["animal.livestock", "place.country"],
        attribute_target_concept="food.cheese.texture",
    )
    assert anchors == ["Goat", "France"]
    assert filters == []


def test_G3_split_function_no_entity_target_concepts_all_anchors():
    """Empty entity_target_concepts → all entities are anchors."""
    anchors, filters = ControllerOrchestratorMixin._kg_split_anchor_vs_filter_operands(
        entities=["Goat", "semi-firm"],
        entity_target_concepts=[],
        attribute_target_concept="food.cheese.texture",
    )
    assert anchors == ["Goat", "semi-firm"]
    assert filters == []


def test_G4_derive_roles_cheese_task_splits_correctly():
    """Cheese texture task: semi-firm goes to filter_value_operand, not anchor_operands."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=["Goat", "cows", "semi-firm"],
        attribute_target_concept="food.cheese.texture",
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
    )
    assert "semi-firm" not in result["anchor_operands"]
    assert "Goat" in result["anchor_operands"]
    assert "cows" in result["anchor_operands"]
    assert result.get("filter_value_operand") == "semi-firm"


def test_G5_anchor_count_drives_family_after_split():
    """After splitting, 2 anchors → two_anchor family even with 3 raw entities."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=["Goat", "cows", "semi-firm"],
        attribute_target_concept="food.cheese.texture",
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
    )
    assert result["template_family"] == "two_anchor_intersect_filter_set"


def test_G6_filter_value_operand_propagated_to_round_context():
    """filter_value_operand must reach round_context via _toolgen_apply_round_strategy_context."""
    payload = _make_payload(
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
        attribute_target_concept="food.cheese.texture",
        target_archetype="INTERSECTOR",
    )
    result = _mixin._toolgen_apply_round_strategy_context(payload, {})
    retry_ctx = result.get("toolgen_retry_context") or {}
    assert retry_ctx.get("filter_value_operand") == "semi-firm"
    assert "semi-firm" not in (retry_ctx.get("anchor_operands") or [])


def test_G7_filter_value_operand_in_policy_diagnostic_fields():
    """filter_value_operand must appear in _toolgen_apply_validation_policy output."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
        attribute_target_concept="food.cheese.texture",
        target_archetype="INTERSECTOR",
    ))
    validation = {
        "grade": 5, "issues": [], "fixes": [], "repair_mode": "none",
        "material_progress": False, "partial_value_usable": False,
        "bankable_partial_progress": False, "semantic_code_smells": [],
    }
    summary = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=summary,
        round_context={},
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    assert result.get("filter_value_operand") == "semi-firm"


# ---------------------------------------------------------------------------
# PART H — family-specific primary_repair_instruction wins (PART 2)
# ---------------------------------------------------------------------------

_DUMMY_VALIDATION_BASE = {
    "grade": 4,
    "issues": [],
    "fixes": [],
    "repair_mode": "rewrite_code",
    "material_progress": False,
    "partial_value_usable": False,
    "bankable_partial_progress": False,
    "semantic_code_smells": [],
}

_DUMMY_SUMMARY = {"value_delivered": "none", "execution_status": "ERROR", "achieved_state": "none"}


def test_H1_family_repair_wins_over_generic_stage_binding():
    """Family-specific primary_repair_instruction must be the final value even when
    stage_binding_instruction is also set."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.texture",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
    ))
    validation = dict(_DUMMY_VALIDATION_BASE)
    validation["required_next_achieved_state"] = "built_filter_ready_set"
    validation["stage_binding_instruction"] = "PLAN-STAGE: do X then Y"
    round_context = {
        "template_family": "two_anchor_intersect_filter_set",
        "anchor_operands": ["Goat", "cows"],
        "filter_operand": "food.cheese.texture",
        "filter_value_operand": "semi-firm",
        "terminal_artifact_kind": "filtered_set_variable",
        "material_progress_last_round": False,
    }
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=_DUMMY_SUMMARY,
        round_context=round_context,
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    pri = result.get("primary_repair_instruction") or ""
    assert "two_anchor_intersect_filter_set" in pri


def test_H2_family_repair_wins_for_count_family():
    """Count family repair instruction must be the final primary_repair_instruction."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="COUNTING_INTERSECTOR",
        entities=["Goat", "cows"],
        entity_target_concepts=["animal.livestock", "animal.livestock"],
    ))
    round_context = {
        "template_family": "two_anchor_intersect_count",
        "anchor_operands": ["Goat", "cows"],
        "filter_operand": "",
        "terminal_artifact_kind": "count_variable",
        "material_progress_last_round": False,
    }
    result = _mixin._toolgen_apply_validation_policy(
        validation=dict(_DUMMY_VALIDATION_BASE),
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=_DUMMY_SUMMARY,
        round_context=round_context,
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    pri = result.get("primary_repair_instruction") or ""
    assert "two_anchor_intersect_count" in pri


def test_H3_family_repair_path_used_field_set():
    """family_repair_path_used must be set when family repair instruction wins."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.texture",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
    ))
    round_context = {
        "template_family": "two_anchor_intersect_filter_set",
        "anchor_operands": ["Goat", "cows"],
        "filter_operand": "food.cheese.texture",
        "terminal_artifact_kind": "filtered_set_variable",
        "material_progress_last_round": False,
    }
    result = _mixin._toolgen_apply_validation_policy(
        validation=dict(_DUMMY_VALIDATION_BASE),
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=_DUMMY_SUMMARY,
        round_context=round_context,
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    assert result.get("family_repair_path_used") == "two_anchor_intersect_filter_set"


# ---------------------------------------------------------------------------
# PART I — family skeleton block (PART 3)
# ---------------------------------------------------------------------------

def test_I1_skeleton_block_nonempty_for_filter_set_family():
    rc = {"template_family": "two_anchor_intersect_filter_set", "anchor_operands": ["A", "B"],
          "filter_operand": "food.cheese.texture", "filter_value_operand": "semi-firm",
          "terminal_artifact_kind": "filtered_set_variable"}
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block(rc)
    assert "two_anchor_intersect_filter_set" in block
    assert "anchor_operands" in block
    assert "resolve_semantic_filter" in block
    assert "filtered_set_variable" in block


def test_I2_skeleton_block_nonempty_for_count_family():
    rc = {"template_family": "two_anchor_intersect_count", "anchor_operands": ["A", "B"],
          "terminal_artifact_kind": "count_variable"}
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block(rc)
    assert "two_anchor_intersect_count" in block
    assert "count_variable" in block


def test_I3_skeleton_block_nonempty_for_extract_attribute_family():
    rc = {"template_family": "two_anchor_intersect_extract_attribute",
          "anchor_operands": ["A", "B"], "filter_operand": "food.cheese.milk_source",
          "terminal_artifact_kind": "attribute_values"}
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block(rc)
    assert "two_anchor_intersect_extract_attribute" in block
    assert "attribute_values" in block


def test_I4_skeleton_block_nonempty_for_one_anchor_family():
    rc = {"template_family": "one_anchor_filter_then_count_or_progress",
          "anchor_operands": ["France"], "terminal_artifact_kind": "set_variable"}
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block(rc)
    assert "one_anchor_filter_then_count_or_progress" in block


def test_I5_skeleton_block_empty_for_unknown_family():
    rc = {"template_family": "unknown_family"}
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block(rc)
    assert block == ""


def test_I6_skeleton_block_empty_when_no_family():
    rc = {}
    block = ControllerToolgenMixin._toolgen_kg_family_skeleton_block(rc)
    assert block == ""


def test_I7_skeleton_block_injected_in_prompt_builder():
    """_toolgen_build_full_rewrite_prompt must include the skeleton block."""
    round_context = {
        "template_family": "two_anchor_intersect_filter_set",
        "anchor_operands": ["A", "B"],
        "filter_operand": "food.cheese.texture",
        "terminal_artifact_kind": "filtered_set_variable",
    }
    prompt = _mixin._toolgen_build_full_rewrite_prompt(
        base_prompt="BASE_PROMPT_CONTENT",
        round_context=round_context,
        feedback_note="",
        round_history=[],
        last_tool_code=None,
    )
    assert "FAMILY_SKELETON" in prompt
    assert "two_anchor_intersect_filter_set" in prompt


def test_I8_family_binding_context_activates_for_supported_family():
    plan = _mixin._build_tool_plan(
        _make_payload(
            target_archetype="COUNTING_INTERSECTOR",
            entities=["CNES", "Astrium"],
            entity_target_concepts=["space_agency", "aerospace_company"],
            target_concept="spacecraft",
        )
    )
    binding = _mixin._toolgen_kg_family_binding_context(
        tool_plan=plan,
        round_context=plan,
    )
    assert binding["family_binding_active"] is True
    assert binding["generic_path_blocked_for_family_bound_attempt"] is True
    assert binding["family_generation_path_used"] is True
    assert binding["family_repair_template_name"] == "two_anchor_intersect_count"
    assert binding["family_validator_path_used"] is True


def test_I9_family_binding_context_allows_generic_fallback_for_unsupported_family():
    plan = _mixin._build_tool_plan(
        _make_payload(
            target_archetype="COUNTING_INTERSECTOR",
            entities=["CNES", "Astrium"],
            entity_target_concepts=["space_agency", "aerospace_company"],
            target_concept="spacecraft",
        )
    )
    binding = _mixin._toolgen_kg_family_binding_context(
        tool_plan=plan,
        round_context={"template_family": "unsupported_family"},
    )
    assert binding["family_binding_active"] is False
    assert binding["family_binding_reason"] == "unsupported_template_family:unsupported_family"
    assert binding["generic_path_blocked_for_family_bound_attempt"] is False
    assert binding["generic_generation_blocked"] is False
    assert binding["generic_repair_blocked"] is False


def test_I10_family_bound_prompt_omits_prior_code_when_requested():
    round_context = {
        "template_family": "two_anchor_intersect_filter_set",
        "anchor_operands": ["Goat", "cows"],
        "filter_operand": "food.cheese.texture",
        "filter_value_operand": "semi-firm",
        "attribute_target_concept": "food.cheese.texture",
        "target_concept": "food.cheese",
        "terminal_artifact_kind": "filtered_set_variable",
        "required_next_stage": "built_filter_ready_set",
        "family_binding_active": True,
        "family_binding_reason": "supported_template_family:two_anchor_intersect_filter_set",
        "family_skeleton_selected": "two_anchor_intersect_filter_set",
        "generic_path_blocked_for_family_bound_attempt": True,
    }
    prompt = _mixin._toolgen_build_full_rewrite_prompt(
        base_prompt="BASE_PROMPT_CONTENT",
        round_context=round_context,
        feedback_note="repair this",
        round_history=[],
        last_tool_code="def stale_macro():\n    return None\n",
        phase1_retry_state={"omit_prior_code": True},
    )
    assert "FAMILY_BINDING_ACTIVE:" in prompt
    assert "family_binding_active=true" in prompt
    assert "Generic KG generation, generic repair prose, and generic patch/apply surgery are blocked for this attempt." in prompt
    assert "LAST_TOOL_CODE:" not in prompt


# ---------------------------------------------------------------------------
# PART J — family-aware validator smells (PART 4)
# ---------------------------------------------------------------------------

_CODE_FILTER_VALUE_AS_ANCHOR = """
# tool_name: test_fva_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: []
# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}

\"\"\"KG macro.\"\"\"
import json

def run(payload: dict) -> dict:
    \"\"\"
    contract guard: test.
    prereqs: kg_utils available.
    limitations: test.
    \"\"\"
    try:
        env_out = kg_utils.resolve_entity_to_vars("semi-firm", None, None, None, max_k=1)
        return {"status": "ERROR", "final_variable": None, "observation": "test"}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": str(e)}

def self_test() -> bool:
    return True
"""

_CODE_WRONG_TERMINAL_COUNT = """
# tool_name: test_wrong_terminal_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: []
# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}

\"\"\"KG macro.\"\"\"
import json

def run(payload: dict) -> dict:
    \"\"\"
    contract guard: test.
    prereqs: kg_utils available.
    limitations: test.
    \"\"\"
    try:
        return {"status": "SUCCESS", "final_variable": 42, "observation": "count"}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": str(e)}

def self_test() -> bool:
    return True
"""

_CODE_STAGE_ORDER_VIOLATION = """
# tool_name: test_stage_order_generated_tool
# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}
# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]
# RUN_PAYLOAD_OPTIONAL: []
# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}

\"\"\"KG macro - applies filter without intersection.\"\"\"
import json

def run(payload: dict) -> dict:
    \"\"\"
    contract guard: test.
    prereqs: kg_utils available.
    limitations: test.
    \"\"\"
    try:
        base_ids = ["#1", "#2"]
        inter_ids = ["#1", "#2"]
        out = kg_utils.resolve_semantic_filter("#1", "food.cheese.texture", inter_ids, asked_for="semi-firm")
        return {"status": "SUCCESS", "final_variable": "#1", "observation": "ok"}
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "ERROR", "final_variable": None, "observation": str(e)}

def self_test() -> bool:
    return True
"""


def test_J1_filter_value_in_anchor_operands_smell_detected():
    """Dynamic smell must detect filter value entity passed to resolve_entity_to_vars."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
        attribute_target_concept="food.cheese.texture",
        target_archetype="INTERSECTOR",
    ))
    smells = _mixin._toolgen_dynamic_plan_code_smells(_CODE_FILTER_VALUE_AS_ANCHOR, tool_plan)
    assert "filter_value_in_anchor_operands" in smells


def test_J2_wrong_terminal_artifact_smell_count_integer():
    """wrong_terminal_artifact_for_family when count family returns literal integer."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="COUNTING_INTERSECTOR",
        entities=["Goat", "France"],
        entity_target_concepts=["animal.livestock", "place.country"],
    ))
    smells = _mixin._toolgen_dynamic_plan_code_smells(_CODE_WRONG_TERMINAL_COUNT, tool_plan)
    assert "wrong_terminal_artifact_for_family" in smells


def test_J3_family_stage_order_violation_filter_without_intersect():
    """family_stage_order_violation when semantic filter runs without cross_intersect."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
        attribute_target_concept="food.cheese.texture",
        target_archetype="INTERSECTOR",
    ))
    smells = _mixin._toolgen_dynamic_plan_code_smells(_CODE_STAGE_ORDER_VIOLATION, tool_plan)
    assert "family_stage_order_violation" in smells


def test_J4_new_family_smells_in_blocking_kg_smells_source():
    """New smells must be present in the blocking_kg_smells section."""
    import inspect
    src = inspect.getsource(ControllerToolgenMixin._toolgen_validate_candidate_tool)
    assert "filter_value_in_anchor_operands" in src
    assert "wrong_terminal_artifact_for_family" in src
    assert "family_stage_order_violation" in src


def test_J5_new_diagnostic_fields_in_policy_output():
    """filter_value_in_anchor_operands_detected and wrong_terminal_artifact_detected must be in output."""
    tool_plan = _mixin._build_tool_plan(_make_payload(
        target_archetype="INTERSECTOR",
        attribute_target_concept="food.cheese.texture",
    ))
    validation = dict(_DUMMY_VALIDATION_BASE)
    result = _mixin._toolgen_apply_validation_policy(
        validation=validation,
        execution_validation={"status": "ERROR", "final_variable": None, "observation": ""},
        live_progress_summary=_DUMMY_SUMMARY,
        round_context={},
        tool_plan=tool_plan,
        tool_code="def run(p): pass",
    )
    assert "filter_value_in_anchor_operands_detected" in result
    assert "wrong_terminal_artifact_detected" in result
    assert "family_stage_order_violation_detected" in result
    assert "family_skeleton_used" in result
    assert "family_validator_checks_used" in result


# ---------------------------------------------------------------------------
# PART K — A6 update: entities without matching hints → all go to anchors
# ---------------------------------------------------------------------------

def test_K1_no_entity_target_concepts_fallback_split_fires():
    """PART 2: Without entity_target_concepts, 3 entities + has_attribute_target
    triggers family fallback split: first 2 become anchors, last becomes filter_value_operand."""
    entities = ["Goat", "Milk", "France"]
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=entities,
        attribute_target_concept="food.cheese.milk_source",
        entity_target_concepts=[],
    )
    assert result["anchor_operands"] == ["Goat", "Milk"]
    assert result.get("filter_value_operand") == "France"
    assert result["family_fallback_split_used"] is True


def test_K2_partial_matching_hints_only_matching_entity_filtered():
    """Only the entity whose hint matches ATC becomes filter_value_operand."""
    result = ControllerOrchestratorMixin._kg_derive_template_family_and_roles(
        target_archetype="INTERSECTOR",
        entity_count=3,
        has_attribute_target=True,
        entities=["Goat", "cows", "semi-firm"],
        attribute_target_concept="food.cheese.texture",
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
    )
    assert result["anchor_operands"] == ["Goat", "cows"]
    assert result.get("filter_value_operand") == "semi-firm"
    assert "terminal_artifact_kind" in result


def test_K3_build_tool_plan_with_entity_target_concepts_splits_correctly():
    """_build_tool_plan must produce correct anchor_operands when entity_target_concepts match ATC."""
    payload = _make_payload(
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["animal.livestock", "animal.livestock", "food.cheese.texture"],
        attribute_target_concept="food.cheese.texture",
        target_archetype="INTERSECTOR",
    )
    plan = _mixin._build_tool_plan(payload)
    assert "semi-firm" not in plan["anchor_operands"]
    assert plan.get("filter_value_operand") == "semi-firm"
    assert len(plan["anchor_operands"]) == 2
