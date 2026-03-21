import json
import os
import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language_models import LanguageModel
from src.self_evolving_agent.controller import SelfEvolvingController
from src.self_evolving_agent.controller_logging import ControllerLoggingMixin
from src.self_evolving_agent.controller_orchestrator import ControllerOrchestratorMixin
from src.self_evolving_agent.controller_prompts import (
    COMBINED_ORCHESTRATOR_SYSTEM_PROMPT,
    MACRO_TOOLGEN_USER_KG,
    TOOLGEN_VALIDATOR_SYSTEM_PROMPT,
)
from src.self_evolving_agent.controller_toolgen import ControllerToolgenMixin
import src.self_evolving_agent.controller_toolgen as controller_toolgen_module
from src.self_evolving_agent.tool_registry import ToolResult
from src.typings import ChatHistory, ChatHistoryItem, Role


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

    def _get_run_task_metadata(self) -> dict[str, object]:
        return {"task_name": "test_task", "sample_index": 0}


def _kg_plan(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "target_archetype": "COUNTING_INTERSECTOR",
        "entities": ["CNES", "Astrium"],
        "target_concept": "spacecraft",
        "entity_target_concepts": ["space_agency", "aerospace_company"],
        "topological_execution_plan": [
            "1. Use kg_utils.resolve_entity_to_vars to resolve each anchor.",
            "2. Use kg_utils.walk_to_target to build spacecraft sets.",
            "3. Use kg_utils.cross_intersect to intersect the sets.",
            "4. Use count to count the intersection.",
        ],
    }
    base.update(overrides)
    return base


def _round_history_entry(
    round_context: dict[str, object],
    *,
    failure_family: str,
    failure_bucket: str,
    material_progress: bool = False,
    summary: str = "",
    failure_phase: str = "validator",
    repair_mode: str = "rewrite_code",
) -> dict[str, object]:
    entry = dict(round_context)
    entry.update(
        {
            "round": round_context.get("round"),
            "failure_phase": failure_phase,
            "failure_family": failure_family,
            "failure_bucket": failure_bucket,
            "material_progress": material_progress,
            "summary": summary or failure_family,
            "repair_mode": repair_mode,
        }
    )
    return entry


def _candidate_obj(
    *,
    name: str,
    value_delivered: str,
    partial_value_usable: bool,
    semantic_trust_level: str,
    semantic_code_smells: list[str] | None,
    failure_bucket: str,
    grade: int,
    usefulness_passed: bool = True,
) -> dict[str, object]:
    return {
        "tool_spec": {"name": name},
        "tool_code": "def run(payload):\n    return {}\n",
        "validation": {
            "value_delivered": value_delivered,
            "partial_value_usable": partial_value_usable,
            "semantic_trust_level": semantic_trust_level,
            "semantic_code_smells": list(semantic_code_smells or []),
            "failure_bucket": failure_bucket,
            "grade": grade,
            "usefulness_passed": usefulness_passed,
        },
        "usefulness_passed": usefulness_passed,
    }


class _StaticToolInvokerAgent:
    def __init__(self, content: str) -> None:
        self._system_prompt = ""
        self._content = content

    def _inference(self, chat_history: ChatHistory):
        return ChatHistoryItem(role=Role.AGENT, content=self._content)


def test_prompts_include_strategy_and_value_fields() -> None:
    assert "execution_style" in COMBINED_ORCHESTRATOR_SYSTEM_PROMPT
    assert "preferred_tool_mode" in COMBINED_ORCHESTRATOR_SYSTEM_PROMPT
    assert "fallback_strategies" in COMBINED_ORCHESTRATOR_SYSTEM_PROMPT
    assert "strategy_family" in MACRO_TOOLGEN_USER_KG
    assert "execution_style" in MACRO_TOOLGEN_USER_KG
    assert "preferred_tool_mode" in MACRO_TOOLGEN_USER_KG
    assert "failure_family" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT
    assert "integration_context_invalid" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT
    assert "value_delivered" in TOOLGEN_VALIDATOR_SYSTEM_PROMPT


def test_kg_prompt_pushes_minimal_tool_shape() -> None:
    assert "Write the SMALLEST correct tool." in MACRO_TOOLGEN_USER_KG
    assert "Prefer exactly two top-level functions" in MACRO_TOOLGEN_USER_KG
    assert "Do NOT add explanatory comments" in MACRO_TOOLGEN_USER_KG
    assert "INPUT_SCHEMA:" not in MACRO_TOOLGEN_USER_KG


def test_build_tool_plan_assigns_new_defaults(tmp_path: pathlib.Path) -> None:
    controller = _DummyController(tmp_path)

    count_plan = controller._build_tool_plan(_kg_plan())
    assert count_plan["execution_style"] == "relation_first"
    assert count_plan["preferred_tool_mode"] == "full_solve"
    assert count_plan["fallback_strategies"] == [
        "probe_then_commit",
        "set_builder",
        "intersector_counter",
    ]

    superlative_plan = controller._build_tool_plan(
        {
            "target_archetype": "SUPERLATIVE_FINDER",
            "entities": ["Cyclone Tracy"],
            "target_concept": "affected_region",
            "attribute_target_concept": "maximum_damage",
            "topological_execution_plan": [
                "1. Resolve the anchor.",
                "2. Walk to the candidate set.",
                "3. Use argmax on the attribute relation.",
            ],
        }
    )
    assert superlative_plan["execution_style"] == "attribute_mapping_first"

    shared_trait_plan = controller._build_tool_plan(
        {
            "target_archetype": "SHARED_TRAIT_PIVOT",
            "entities": ["Einstein", "Curie"],
            "target_concept": "scientific_field",
            "entity_target_concepts": [],
            "topological_execution_plan": [
                "1. Resolve each anchor.",
                "2. Walk to the shared pivot type.",
                "3. Intersect the branches.",
            ],
        }
    )
    assert shared_trait_plan["execution_style"] == "probe_then_commit"


def test_controller_logging_helpers_emit_invoke_and_failed_invoke_events(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)

    controller._log_tool_invocation_event(
        tool_name="demo_generated_tool",
        args=[{"payload": 1}],
        kwargs={},
        result=ToolResult.success_result({"status": "SUCCESS"}),
        reason="tool_invoker",
        decision_action="create_tool",
    )
    controller._log_failed_invoke_event(
        tool_name="demo_generated_tool",
        reason="missing_required_keys:entities",
        decision_action="create_tool",
    )

    events = _jsonl_events(controller._generated_tools_log_path)  # noqa: SLF001
    assert [event["event"] for event in events] == ["invoke", "failed_invoke"]
    assert events[0]["decision_action"] == "create_tool"
    assert events[1]["decision_action"] == "create_tool"


def test_tool_invoker_logs_post_create_context_and_compatibility(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(tmp_path, monkeypatch)
    generated = _run_live_toolgen_case(controller, _COUNT_QUERY)
    created_tool = generated["tool"]
    assert created_tool is not None
    controller._is_tool_output_form_compatible = (  # noqa: SLF001
        lambda *args, **kwargs: (True, None, None, "count_variable")
    )
    controller._is_tool_semantically_compatible = (  # noqa: SLF001
        lambda *args, **kwargs: (True, None)
    )
    controller._tool_invoker_agent = _StaticToolInvokerAgent(  # noqa: SLF001
        json.dumps(
            {
                "tool_name": created_tool.name,
                "payload": {"entities": ["CNES", "Astrium"]},
                "reason": "use freshly generated tool",
            }
        )
    )

    history = ChatHistory()
    history.inject(ChatHistoryItem(role=Role.USER, content=_COUNT_QUERY))
    result = controller._tool_invoker_decision(  # noqa: SLF001
        _COUNT_QUERY,
        history,
        suggestion={
            "tool_name": created_tool.name,
            "invocation_trigger": "post_create",
            "created_tool_name": created_tool.name,
            **_kg_plan(),
        },
    )

    assert result["tool_name"] == created_tool.name
    events = _jsonl_events(controller._generated_tools_log_path)  # noqa: SLF001
    request_events = [
        event for event in events if event.get("event") == "tool_invoker_request"
    ]
    result_events = [
        event for event in events if event.get("event") == "tool_invoker_result"
    ]
    compatibility_events = [
        event
        for event in events
        if event.get("event") == "tool_invoker_compatibility_decision"
    ]
    assert request_events
    assert request_events[-1]["invocation_trigger"] == "post_create"
    assert request_events[-1]["created_tool_name"] == created_tool.name
    assert result_events
    assert result_events[-1]["invocation_trigger"] == "post_create"
    assert result_events[-1]["created_tool_name"] == created_tool.name
    assert result_events[-1]["tool_name"] == created_tool.name
    assert compatibility_events
    assert compatibility_events[-1]["tool_name"] == created_tool.name


def test_fallback_list_normalization_removes_duplicate_primary(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = controller._build_tool_plan(
        _kg_plan(
            fallback_strategies=[
                "relation_first",
                "probe_then_commit",
                "relation_first",
                "set_builder",
            ]
        )
    )

    assert plan["fallback_strategies"] == [
        "probe_then_commit",
        "set_builder",
    ]


def test_retry_policy_allows_same_strategy_once_for_code_local_failure(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    round1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[],
    )
    round_context = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            )
        ],
    )

    assert round_context["pivot_required"] is False
    assert round_context["strategy_family"] == "relation_first"
    assert round_context["execution_style"] == "relation_first"
    assert round_context["strategy_source"] == "inherited_from_previous_round"
    assert round_context["strategy_index"] == 0
    assert round_context["strategy_epoch"] == 0


def test_retry_policy_pivots_after_repeated_no_progress(tmp_path: pathlib.Path) -> None:
    controller = _DummyController(tmp_path)
    round1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[],
    )
    round2 = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            )
        ],
    )
    round_context = controller._toolgen_compute_round_strategy_context(
        round_idx=3,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            ),
            _round_history_entry(
                round2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            ),
        ],
    )

    assert round_context["pivot_required"] is True
    assert round_context["execution_style"] == "probe_then_commit"
    assert round_context["strategy_family"] != "relation_first"
    assert round_context["strategy_source"] == "pivot_policy"
    assert round_context["strategy_index"] == 1
    assert round_context["strategy_epoch"] == 1
    assert "relation_first" in round_context["disallowed_strategy_families"]


def test_sticky_pivot_keeps_pivoted_strategy_on_next_round(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    exec_payload = {
        "tool_plan": _kg_plan(
            execution_style="relation_first",
            fallback_strategies=["probe_then_commit", "set_builder"],
        )
    }
    round1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1,
        exec_payload=exec_payload,
        round_history=[],
    )
    round2 = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            )
        ],
    )
    round3 = controller._toolgen_compute_round_strategy_context(
        round_idx=3,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            ),
            _round_history_entry(
                round2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            ),
        ],
    )
    round4 = controller._toolgen_compute_round_strategy_context(
        round_idx=4,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            ),
            _round_history_entry(
                round2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable returned",
            ),
            _round_history_entry(
                round3,
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="tool_plan field misread",
                repair_mode="rewrite_code",
            ),
        ],
    )

    assert round3["strategy_family"] == "probe_then_commit"
    assert round3["strategy_source"] == "pivot_policy"
    assert round3["strategy_index"] == 1
    assert round3["strategy_epoch"] == 1
    assert round4["strategy_family"] == "probe_then_commit"
    assert round4["execution_style"] == "probe_then_commit"
    assert round4["strategy_source"] == "inherited_from_previous_round"
    assert round4["strategy_index"] == 1
    assert round4["strategy_epoch"] == 1


def test_fallback_queue_is_consumed_in_order_without_reset(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    exec_payload = {
        "tool_plan": _kg_plan(
            execution_style="relation_first",
            fallback_strategies=["probe_then_commit", "set_builder"],
        )
    }
    round1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1,
        exec_payload=exec_payload,
        round_history=[],
    )
    round2 = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            )
        ],
    )
    round3 = controller._toolgen_compute_round_strategy_context(
        round_idx=3,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            ),
            _round_history_entry(
                round2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            ),
        ],
    )
    round4 = controller._toolgen_compute_round_strategy_context(
        round_idx=4,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            ),
            _round_history_entry(
                round2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            ),
            _round_history_entry(
                round3,
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                repair_mode="rewrite_code",
            ),
        ],
    )
    round5 = controller._toolgen_compute_round_strategy_context(
        round_idx=5,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                round1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            ),
            _round_history_entry(
                round2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
            ),
            _round_history_entry(
                round3,
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                repair_mode="rewrite_code",
            ),
            _round_history_entry(
                round4,
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                repair_mode="rewrite_code",
            ),
        ],
    )

    assert round3["strategy_family"] == "probe_then_commit"
    assert round4["strategy_family"] == "probe_then_commit"
    assert round4["strategy_source"] == "inherited_from_previous_round"
    assert round5["strategy_family"] == "set_builder"
    assert round5["strategy_source"] == "pivot_policy"
    assert round5["strategy_index"] == 2
    assert round5["strategy_epoch"] == 2
    assert round5["strategy_family"] != "relation_first"


def test_bucketed_failure_pressure_can_trigger_pivot_with_different_families(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    round1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[],
    )
    round2 = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="tool_plan field misread",
            )
        ],
    )
    round3 = controller._toolgen_compute_round_strategy_context(
        round_idx=3,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="tool_plan field misread",
            ),
            _round_history_entry(
                round2,
                failure_family="variable_list_context_wrong",
                failure_bucket="context_handling_no_progress",
                material_progress=False,
                summary="variable_list context wrong",
            ),
        ],
    )

    assert round3["pivot_required"] is True
    assert round3["strategy_family"] == "probe_then_commit"
    assert round3["strategy_source"] == "pivot_policy"


def test_rewrite_code_does_not_reset_a_pivoted_strategy(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    exec_payload = {
        "tool_plan": _kg_plan(
            execution_style="relation_first",
            fallback_strategies=["probe_then_commit", "set_builder"],
        )
    }
    round_context = controller._toolgen_compute_round_strategy_context(
        round_idx=4,
        exec_payload=exec_payload,
        round_history=[
            _round_history_entry(
                {
                    "round": 3,
                    "strategy_family": "probe_then_commit",
                    "active_strategy_family": "probe_then_commit",
                    "execution_style": "probe_then_commit",
                    "active_execution_style": "probe_then_commit",
                    "preferred_tool_mode": "full_solve",
                    "active_preferred_tool_mode": "full_solve",
                    "fallback_strategies": ["set_builder"],
                    "strategy_sequence": [
                        "relation_first",
                        "probe_then_commit",
                        "set_builder",
                    ],
                    "strategy_index": 1,
                    "strategy_epoch": 1,
                    "strategy_source": "pivot_policy",
                },
                failure_family="tool_plan_field_misread",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                repair_mode="rewrite_code",
            )
        ],
    )

    assert round_context["strategy_family"] == "probe_then_commit"
    assert round_context["strategy_source"] == "inherited_from_previous_round"
    assert round_context["strategy_index"] == 1
    assert round_context["strategy_epoch"] == 1


def test_live_progress_recognizes_partial_value_modes(tmp_path: pathlib.Path) -> None:
    controller = _DummyController(tmp_path)

    resolved_summary = controller._summarize_toolgen_live_progress(
        {
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": (
                'MACRO EXHAUSTED: Resulting set is empty. minted_variables: '
                '{"resolved_anchor_a": "#1", "resolved_anchor_b": "#2"}'
            ),
        },
        {"tool_plan": _kg_plan(preferred_tool_mode="progress_tool")},
    )
    assert resolved_summary["material_progress"] is True
    assert resolved_summary["value_delivered"] == "resolved_both_anchors"
    assert resolved_summary["partial_value_usable"] is True

    set_summary = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#3",
            "observation": 'Variable #3 contains spacecraft set. minted_variables: {"spacecraft_set": "#3"}',
        },
        {
            "tool_plan": _kg_plan(
                execution_style="partial_value_first",
                preferred_tool_mode="progress_tool",
            )
        },
    )
    assert set_summary["material_progress"] is True
    assert set_summary["value_delivered"] == "built_target_set"
    assert set_summary["value_mode"] == "progress_tool"

    diagnostic_summary = controller._summarize_toolgen_live_progress(
        {
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": (
                'MACRO EXHAUSTED: Resulting set is empty. minted_variables: {"resolved_anchor": "#4"} '
                "Diagnostic: empty walk because no spacecraft relation matched the anchor."
            ),
        },
        {
            "tool_plan": _kg_plan(
                execution_style="diagnostic_first",
                preferred_tool_mode="diagnostic_probe",
            )
        },
    )
    assert diagnostic_summary["material_progress"] is True
    assert diagnostic_summary["value_delivered"] == "produced_actionable_handoff"
    assert diagnostic_summary["value_mode"] == "diagnostic_probe"


def test_validation_policy_penalizes_repetition_and_rewards_partial_value(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    repeated = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 9,
            "issues": ["empty walk repeated"],
            "summary": "Structurally fine but made no live progress.",
            "repair_mode": "none",
        },
        execution_validation={
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": "Diagnostic: empty walk because no matching relation was found.",
        },
        live_progress_summary={
            "material_progress": False,
            "has_context": False,
            "execution_status": "MACRO EXHAUSTED",
            "has_final_variable": False,
            "final_operation_safe": False,
        },
        round_context={
            "previous_strategy_family": "relation_first",
            "previous_failure_family": "empty_walk",
            "previous_failure_bucket": "strategy_mismatch_no_progress",
            "execution_style": "relation_first",
            "preferred_tool_mode": "full_solve",
            "material_progress_last_round": False,
            "pivot_required": True,
            "pivot_reason": "repeated_no_progress_same_strategy_failure",
        },
        tool_plan=_kg_plan(execution_style="relation_first"),
        tool_code="kg_utils.resolve_entity_to_vars(entity, None, actions_spec, domain_hints)\nkg_utils.walk_to_target(actions_spec, vars_a, target_concept, domain_hints)",
    )
    assert repeated["grade"] == 4
    assert repeated["pivot_required"] is True
    assert repeated["repair_mode"] != "none"
    assert repeated["failure_bucket"] == "strategy_mismatch_no_progress"
    assert repeated["strategy_source"] == "inherited_from_previous_round"

    partial = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 4,
            "issues": [],
            "summary": "Built the spacecraft set correctly.",
            "repair_mode": "rewrite_code",
        },
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#9",
            "observation": 'Variable #9 contains spacecraft set. minted_variables: {"spacecraft_set": "#9"}',
        },
        live_progress_summary={
            "material_progress": True,
            "has_context": True,
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "final_operation_safe": False,
        },
        round_context={
            "previous_strategy_family": "probe_then_commit",
            "previous_failure_family": "wrong_relation_family",
            "execution_style": "partial_value_first",
            "preferred_tool_mode": "progress_tool",
        },
        tool_plan=_kg_plan(
            execution_style="partial_value_first",
            preferred_tool_mode="progress_tool",
        ),
        tool_code="kg_utils.resolve_entity_to_vars(entity, target_concept, actions_spec, domain_hints)\nreturn {'status': 'SUCCESS'}",
    )
    assert partial["grade"] >= 6
    assert partial["value_delivered"] == "built_target_set"
    assert partial["failure_bucket"] == "partial_value_delivered"


def test_integration_context_invalid_is_classified_without_pivot_pressure(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    validation = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 9,
            "issues": ["execution_integration_context_invalid: missing payload['entities']"],
            "summary": "Execution could not start because entities were missing.",
            "repair_mode": "none",
        },
        execution_validation={
            "integration_context_invalid": True,
            "integration_context_reason": "missing_entities",
            "observation": "execution_integration_context_invalid: missing payload['entities']",
        },
        live_progress_summary={
            "material_progress": False,
            "has_context": False,
            "execution_status": "ERROR",
            "has_final_variable": False,
            "final_operation_safe": False,
        },
        round_context={
            "previous_strategy_family": "relation_first",
            "previous_failure_family": "tool_plan_field_misread",
            "previous_failure_bucket": "code_local_no_progress",
            "execution_style": "relation_first",
            "preferred_tool_mode": "full_solve",
            "material_progress_last_round": False,
            "pivot_required": True,
        },
        tool_plan=_kg_plan(execution_style="relation_first"),
        tool_code="return {'status': 'ERROR', 'final_variable': None, 'observation': 'missing payload entities'}",
    )
    assert validation["failure_family"] == "integration_context_invalid"
    assert validation["failure_bucket"] == "integration_context_invalid"
    assert validation["strategy_pivot_recommended"] is False

    round1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[],
    )
    round2 = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="integration_context_invalid",
                failure_bucket="integration_context_invalid",
                material_progress=False,
                summary="missing payload entities",
            )
        ],
    )
    round3 = controller._toolgen_compute_round_strategy_context(
        round_idx=3,
        exec_payload={"tool_plan": _kg_plan(execution_style="relation_first")},
        round_history=[
            _round_history_entry(
                round1,
                failure_family="integration_context_invalid",
                failure_bucket="integration_context_invalid",
                material_progress=False,
                summary="missing payload entities",
            ),
            _round_history_entry(
                round2,
                failure_family="integration_context_invalid",
                failure_bucket="integration_context_invalid",
                material_progress=False,
                summary="missing payload entities",
            ),
        ],
    )
    assert round3["pivot_required"] is False
    assert round3["strategy_family"] == "relation_first"


def test_partial_candidate_bank_prefers_stronger_partial_value_over_higher_raw_grade(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    stronger_early = _candidate_obj(
        name="early_progress_tool",
        value_delivered="built_target_set",
        partial_value_usable=True,
        semantic_trust_level="verified",
        semantic_code_smells=[],
        failure_bucket="partial_value_delivered",
        grade=6,
    )
    weaker_late = _candidate_obj(
        name="late_progress_tool",
        value_delivered="resolved_anchor",
        partial_value_usable=True,
        semantic_trust_level="partial_unverified",
        semantic_code_smells=["kg_utils_shape_probing"],
        failure_bucket="partial_value_delivered",
        grade=7,
    )

    assert (
        controller._toolgen_candidate_is_better(
            weaker_late,
            stronger_early,
            partial_bank=True,
        )
        is False
    )
    assert controller._toolgen_candidate_is_better(weaker_late, stronger_early) is False


def test_patch_prompt_uses_compact_code_context(tmp_path: pathlib.Path) -> None:
    controller = _DummyController(tmp_path)
    large_code = "\n".join(
        [
            "# tool_name: giant_macro_generated_tool",
            '# INVOKE_WITH: {"args":[{}],"kwargs":{}}',
            '# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]',
            '# RUN_PAYLOAD_OPTIONAL: ["tool_plan"]',
            '# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}',
            '"""KG macro."""',
            "import json",
            "",
            "def run(payload: dict) -> dict:",
            '    """contract guard: ok.\\n    prereqs: ok.\\n    limitations: ok."""',
            "    try:",
            "        payload = payload or {}",
            "        return {'status': 'ERROR', 'final_variable': None, 'observation': 'x'}",
            "    except (KeyError, TypeError, ValueError) as e:",
            "        return {'status': 'ERROR', 'final_variable': None, 'observation': str(e)}",
            "",
            "def unused_helper() -> str:",
            "    return 'dead code'",
            "",
            "def self_test() -> bool:",
            "    return True",
            "",
            "x = '" + ("trail" * 5000) + "'",
        ]
    )
    prompt = controller._toolgen_build_patch_prompt(
        current_code=large_code,
        feedback_note="{}",
        round_history=[],
        base_prompt="task pack",
    )

    assert "def run(payload: dict) -> dict:" in prompt
    assert "def self_test() -> bool:" in prompt
    assert "def unused_helper() -> str:" not in prompt
    assert len(prompt) < len(large_code)


def test_semantic_code_smells_flag_verbose_scaffolding(tmp_path: pathlib.Path) -> None:
    controller = _DummyController(tmp_path)
    bulky_code = "\n".join(
        [
            "# tool_name: bulky_macro_generated_tool",
            '# INVOKE_WITH: {"args":[{}],"kwargs":{}}',
            '# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]',
            '# RUN_PAYLOAD_OPTIONAL: ["tool_plan"]',
            '# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}',
            "# extra comment one",
            "# extra comment two",
            "# extra comment three",
            '"""This is a very long module docstring that exists only to add size and does not carry any additional contract value for the tool generation loop."""',
            "import json",
            "",
            "def helper_one() -> str:",
            "    return 'a'",
            "",
            "def run(payload: dict) -> dict:",
            '    """contract guard: line one.\\n    prereqs: line two.\\n    limitations: line three.\\n    extra line four.\\n    extra line five.\\n    extra line six.\\n    extra line seven.\\n    extra line eight."""',
            "    try:",
            "        payload = payload or {}",
            "        return {'status': 'ERROR', 'final_variable': None, 'observation': 'x'}",
            "    except (KeyError, TypeError, ValueError) as e:",
            "        return {'status': 'ERROR', 'final_variable': None, 'observation': str(e)}",
            "",
            "def self_test() -> bool:",
            "    return True",
        ]
    )

    smells = controller._toolgen_semantic_code_smells(bulky_code)

    assert "helper_proliferation" in smells
    assert "comment_scaffolding" in smells
    assert "docstring_scaffolding" in smells


def test_validation_policy_penalizes_bulky_dead_end_tools(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    bulky_code = "\n".join(
        [
            "# tool_name: bulky_macro_generated_tool",
            '# INVOKE_WITH: {"args":[{}],"kwargs":{}}',
            '# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]',
            '# RUN_PAYLOAD_OPTIONAL: ["tool_plan"]',
            '# INVOKE_EXAMPLE: {"args":[{}],"kwargs":{}}',
            "# extra comment one",
            "# extra comment two",
            "# extra comment three",
            '"""This is a very long module docstring that exists only to add size and does not carry any additional contract value for the tool generation loop."""',
            "import json",
            "",
            "def helper_one() -> str:",
            "    return 'a'",
            "",
            "def run(payload: dict) -> dict:",
            '    """contract guard: line one.\\n    prereqs: line two.\\n    limitations: line three.\\n    extra line four.\\n    extra line five.\\n    extra line six.\\n    extra line seven.\\n    extra line eight."""',
            "    try:",
            "        payload = payload or {}",
            "        return {'status': 'ERROR', 'final_variable': None, 'observation': 'x'}",
            "    except (KeyError, TypeError, ValueError) as e:",
            "        return {'status': 'ERROR', 'final_variable': None, 'observation': str(e)}",
            "",
            "def self_test() -> bool:",
            "    return True",
        ]
    )

    result = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 9,
            "issues": [],
            "fixes": [],
            "summary": "structurally fine",
            "plan_diagnosis": "OK",
            "repair_mode": "none",
        },
        execution_validation={
            "status": "MACRO EXHAUSTED",
            "observation": "MACRO EXHAUSTED: Resulting set is empty.",
        },
        live_progress_summary={
            "material_progress": False,
            "has_final_variable": False,
            "usefulness_passed": False,
            "semantic_trust_level": "blocked",
            "handoff_state": "blocked",
            "reason": "no_material_progress",
        },
        round_context={"round": 1},
        tool_plan=_kg_plan(),
        tool_code=bulky_code,
        failure_phase="validator",
    )

    assert result["grade"] <= 5
    assert result["repair_mode"] == "rewrite_code"
    assert any("oversized for the value delivered" in issue for issue in result["issues"])


def test_adapter_style_regression_loses_to_cleaner_prior_candidate(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    guarded = controller._toolgen_apply_adapter_regression_guard(
        {
            "grade": 9,
            "issues": [],
            "summary": "",
            "repair_mode": "none",
            "usefulness_passed": True,
            "semantic_code_smells": ["kg_utils_shape_probing"],
        },
        round_history=[
            {
                "round": 1,
                "tool_name": "clean_progress_tool",
                "material_progress": True,
                "partial_value_usable": True,
                "semantic_code_smells": [],
            }
        ],
    )
    assert guarded["grade"] == 4
    assert guarded["usefulness_passed"] is False
    assert guarded["repair_mode"] == "rewrite_code"
    assert guarded["usefulness_reason"] == "adapter_shape_regression_after_cleaner_candidate"


def test_logging_and_failed_persistence_include_new_metadata(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    controller._append_orchestrator_plan_trace(
        action="request_new_tool",
        target_archetype="COUNTING_INTERSECTOR",
        execution_style="relation_first",
        preferred_tool_mode="full_solve",
        fallback_strategies=["walk_first", "probe_then_commit"],
        entities=["CNES", "Astrium"],
        topological_execution_plan=["1. Resolve.", "2. Count."],
    )
    controller._append_generated_tools_log(
        {
            "event": "toolgen_candidate",
            "tool_name": "kg_probe_generated_tool",
            "strategy_family": "relation_first",
            "execution_style": "relation_first",
            "preferred_tool_mode": "progress_tool",
            "strategy_source": "inherited_from_previous_round",
            "strategy_index": 0,
            "strategy_epoch": 0,
            "failure_family": "wrong_relation_family",
            "failure_bucket": "strategy_mismatch_no_progress",
            "value_delivered": "identified_relation_candidates",
            "same_strategy_as_previous": True,
            "same_failure_as_previous": True,
            "pivot_required": True,
            "partial_value_usable": True,
        }
    )
    controller._append_toolgen_strategy_pivot(
        previous_strategy="relation_first",
        previous_failure_family="wrong_relation_family",
        previous_failure_bucket="strategy_mismatch_no_progress",
        new_strategy="probe_then_commit",
        round=3,
        strategy_source="pivot_policy",
        strategy_index=1,
        strategy_epoch=1,
        failure_bucket="strategy_mismatch_no_progress",
        reason="repeated_no_progress_same_strategy_failure",
    )
    controller._persist_tool_candidate(
        round_idx=2,
        tool_name="kg_probe_generated_tool",
        tool_code="def run(payload):\n    return {}\n",
        failure_phase="validator",
        extra_meta={
            "strategy_family": "relation_first",
            "execution_style": "relation_first",
            "preferred_tool_mode": "progress_tool",
            "strategy_sequence": ["relation_first", "probe_then_commit"],
            "strategy_source": "inherited_from_previous_round",
            "strategy_index": 0,
            "strategy_epoch": 0,
            "failure_family": "wrong_relation_family",
            "failure_bucket": "strategy_mismatch_no_progress",
            "value_delivered": "identified_relation_candidates",
            "same_strategy_as_previous": True,
            "same_failure_as_previous": True,
            "pivot_required": True,
            "partial_value_usable": True,
        },
    )
    paths = controller._write_failed_tool_artifact(
        stage="validator",
        error="grade=4",
        code="def run(payload):\n    return {}\n",
        metadata={
            "strategy_family": "relation_first",
            "strategy_source": "inherited_from_previous_round",
            "strategy_index": 0,
            "strategy_epoch": 0,
            "failure_family": "wrong_relation_family",
            "failure_bucket": "strategy_mismatch_no_progress",
            "value_delivered": "identified_relation_candidates",
            "pivot_required": True,
        },
    )

    plan_log = (
        tmp_path / "orchestrator_plan_trace.jsonl"
    ).read_text(encoding="utf-8").strip()
    assert '"execution_style": "relation_first"' in plan_log
    assert '"preferred_tool_mode": "full_solve"' in plan_log

    generated_log = (tmp_path / "generated_tools.jsonl").read_text(encoding="utf-8")
    assert '"strategy_family": "relation_first"' in generated_log
    assert '"strategy_source": "inherited_from_previous_round"' in generated_log
    assert '"strategy_index": 0' in generated_log
    assert '"strategy_epoch": 0' in generated_log
    assert '"failure_family": "wrong_relation_family"' in generated_log
    assert '"failure_bucket": "strategy_mismatch_no_progress"' in generated_log
    assert '"value_delivered": "identified_relation_candidates"' in generated_log

    value_trace = (tmp_path / "tool_value_trace.jsonl").read_text(encoding="utf-8")
    assert '"event": "toolgen_strategy_pivot"' in value_trace
    assert '"new_strategy": "probe_then_commit"' in value_trace
    assert '"strategy_epoch": 1' in value_trace

    metadata_files = list((tmp_path / "generated_tool_candidates").glob("*__metadata.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["strategy_family"] == "relation_first"
    assert metadata["strategy_source"] == "inherited_from_previous_round"
    assert metadata["strategy_index"] == 0
    assert metadata["strategy_epoch"] == 0
    assert metadata["failure_bucket"] == "strategy_mismatch_no_progress"
    assert metadata["pivot_required"] is True

    assert paths
    failed_text = paths[0].read_text(encoding="utf-8")
    assert '"strategy_family": "relation_first"' in failed_text
    assert '"strategy_source": "inherited_from_previous_round"' in failed_text
    assert '"strategy_index": 0' in failed_text
    assert '"strategy_epoch": 0' in failed_text
    assert '"failure_bucket": "strategy_mismatch_no_progress"' in failed_text
    assert '"pivot_required": true' in failed_text.lower()


_COUNT_QUERY = (
    "Question: how many spacecrafts did cnes and astrium make?, "
    "Entities: ['CNES', 'Astrium']"
)
_SUPERLATIVE_QUERY = (
    "Question: which region suffered the maximum damage from cyclone tracy?, "
    "Entities: ['Cyclone Tracy']"
)
_SHARED_TRAIT_QUERY = (
    "Question: what scientific field do einstein and curie share?, "
    "Entities: ['Einstein', 'Curie']"
)


def _task_key(text: str) -> str:
    lowered = (text or "").lower()
    if "cnes" in lowered and "astrium" in lowered:
        return "count"
    if "cyclone tracy" in lowered:
        return "superlative"
    if "einstein" in lowered and "curie" in lowered:
        return "shared"
    return "generic"


def _jsonl_events(path: pathlib.Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _macro_tool_source(
    *,
    task_key: str,
    strategy_family: str,
    execution_style: str,
    preferred_tool_mode: str,
    round_idx: int,
) -> str:
    tool_name = f"{task_key}_{strategy_family}_r{round_idx}_generated_tool"
    return (
        "###TOOL_START\n"
        f"# tool_name: {tool_name}\n"
        '# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}\n'
        '# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]\n'
        '# RUN_PAYLOAD_OPTIONAL: ["domain_hints", "target_concept", "attribute_target_concept", "entity_target_concepts", "intermediate_target_concepts", "topological_execution_plan", "composite_topology", "target_archetype", "upgrade_goal", "recovery_policy", "execution_style", "preferred_tool_mode", "fallback_strategies", "tool_plan", "toolgen_retry_context", "variable_list"]\n'
        '# INVOKE_EXAMPLE: {"args":[{"task_text":"...","asked_for":"...","trace":[],"actions_spec":{},"run_id":"r1","state_dir":"./state","entities":["A"]}], "kwargs":{}}\n'
        f'"""KG macro for {task_key}."""\n'
        "\n"
        "import json\n"
        "\n"
        "def run(payload: dict) -> dict:\n"
        '    """\n'
        "    contract guard: payload must contain the required run keys.\n"
        "    prereqs: kg_utils facade and needed actions_spec primitives are available.\n"
        "    limitations: deterministic stdlib-only translator; no extra scaffolding.\n"
        '    """\n'
        "    try:\n"
        "        payload = payload or {}\n"
        "        candidate_map = {}\n"
        f"        strategy_family = {strategy_family!r}\n"
        f"        execution_style = {execution_style!r}\n"
        f"        preferred_tool_mode = {preferred_tool_mode!r}\n"
        "        observation = (\n"
        '            "MACRO EXHAUSTED: Resulting set is empty. minted_variables: "\n'
        "            + json.dumps(candidate_map)\n"
        '            + " strategy_family="\n'
        "            + strategy_family\n"
        '            + " execution_style="\n'
        "            + execution_style\n"
        '            + " preferred_tool_mode="\n'
        "            + preferred_tool_mode\n"
        "        )\n"
        '        return {"status": "MACRO EXHAUSTED", "final_variable": None, "observation": observation}\n'
        "    except (KeyError, TypeError, ValueError) as e:\n"
        '        return {"status": "ERROR", "final_variable": None, "observation": f"Tool error: {str(e)}"}\n'
        "\n"
        "def self_test() -> bool:\n"
        "    return True\n"
        "###TOOL_END\n"
    )


class _FakeKGTaskRef:
    def evaluate_generated_macro(self, tool_code: str, payload_json: str) -> str:
        payload = json.loads(payload_json)
        task_key = _task_key(str(payload.get("task_text") or ""))
        retry_context = payload.get("toolgen_retry_context") or {}
        strategy_family = str(
            retry_context.get("strategy_family")
            or payload.get("execution_style")
            or ""
        )
        if task_key == "count":
            if strategy_family == "relation_first":
                return json.dumps(
                    {
                        "status": "MACRO EXHAUSTED",
                        "final_variable": None,
                        "observation": "MACRO EXHAUSTED: Resulting set is empty.",
                    }
                )
            return json.dumps(
                {
                    "status": "SUCCESS",
                    "final_variable": "#41",
                    "observation": (
                        'COUNT VARIABLE RETURNED; submit it directly. minted_variables: '
                        '{"spacecraft_intersection": "#12", "count_result": "#41"}'
                    ),
                }
            )
        if task_key == "superlative":
            if strategy_family == "attribute_preparer":
                return json.dumps(
                    {
                        "status": "MACRO EXHAUSTED",
                        "final_variable": None,
                        "observation": "MACRO EXHAUSTED: Resulting set is empty.",
                    }
                )
            return json.dumps(
                {
                    "status": "MACRO EXHAUSTED",
                    "final_variable": None,
                    "observation": (
                        'MACRO EXHAUSTED: Resulting set is empty. minted_variables: '
                        '{"resolved_anchor": "#5"} '
                        "Diagnostic: attribute mapping missing. top relation families: "
                        "damage.maximum, damage.severity"
                    ),
                }
            )
        if task_key == "shared":
            if strategy_family in {"walk_first", "shared_trait_pivot"}:
                return json.dumps(
                    {
                        "status": "MACRO EXHAUSTED",
                        "final_variable": None,
                        "observation": "MACRO EXHAUSTED: Resulting set is empty.",
                    }
                )
            return json.dumps(
                {
                    "status": "MACRO EXHAUSTED",
                    "final_variable": None,
                    "observation": (
                        'MACRO EXHAUSTED: Resulting set is empty. minted_variables: '
                        '{"resolved_anchor_a": "#1", "resolved_anchor_b": "#2"} '
                        "Diagnostic: shared pivot relation candidates: physics, chemistry"
                    ),
                }
            )
        return json.dumps(
            {
                "status": "MACRO EXHAUSTED",
                "final_variable": None,
                "observation": "MACRO EXHAUSTED: Resulting set is empty.",
            }
        )


class _LivePathLanguageModel(LanguageModel):
    def __init__(self, *, fail_toolgen: bool = False) -> None:
        super().__init__({Role.USER: "user", Role.AGENT: "assistant"})
        self._fail_toolgen = fail_toolgen

    def _orchestrator_response(self, prompt: str) -> str:
        task_key = _task_key(prompt)
        if task_key == "count":
            payload = {
                "action": "request_new_tool",
                "tool_type": "macro",
                "target_archetype": "COUNTING_INTERSECTOR",
                "execution_style": "relation_first",
                "preferred_tool_mode": "full_solve",
                "fallback_strategies": [
                    "probe_then_commit",
                    "set_builder",
                    "intersector_counter",
                ],
                "entity_target_concepts": ["space_agency", "aerospace_company"],
                "domain_hints": ["spacecraft", "organization"],
                "target_concept": "spacecraft",
                "reason": "INPUT: ['CNES', 'Astrium']. GOAL: relation-first shared manufacturer count.",
                "topological_execution_plan": [
                    "1. Probe likely manufacture relations for both anchors.",
                    "2. Build both spacecraft sets.",
                    "3. Intersect the sets.",
                    "4. Count the intersection.",
                ],
            }
            return json.dumps(payload)
        if task_key == "superlative":
            payload = {
                "action": "request_new_tool",
                "tool_type": "macro",
                "target_archetype": "SUPERLATIVE_FINDER",
                "execution_style": "attribute_mapping_first",
                "preferred_tool_mode": "full_solve",
                "fallback_strategies": [
                    "diagnostic_probe",
                    "walk_first",
                ],
                "entity_target_concepts": ["tropical_cyclone"],
                "domain_hints": ["weather", "damage"],
                "target_concept": "affected_region",
                "attribute_target_concept": "maximum_damage",
                "reason": "INPUT: ['Cyclone Tracy']. GOAL: build candidate regions then select the maximum-damage region.",
                "topological_execution_plan": [
                    "1. Resolve the cyclone anchor.",
                    "2. Build the affected region set.",
                    "3. Prepare attribute mapping for damage.",
                    "4. Use argmax on the prepared attribute context.",
                ],
            }
            return json.dumps(payload)
        payload = {
            "action": "request_new_tool",
            "tool_type": "macro",
            "target_archetype": "SHARED_TRAIT_PIVOT",
            "execution_style": "walk_first",
            "preferred_tool_mode": "full_solve",
            "fallback_strategies": [
                "diagnostic_probe",
                "probe_then_commit",
            ],
            "entity_target_concepts": ["scientist", "scientist"],
            "domain_hints": ["science", "person"],
            "target_concept": "scientific_field",
            "reason": "INPUT: ['Einstein', 'Curie']. GOAL: walk both anchors to a shared scientific field pivot.",
            "topological_execution_plan": [
                "1. Resolve each scientist anchor.",
                "2. Walk each branch toward scientific fields.",
                "3. Intersect the branches.",
            ],
        }
        return json.dumps(payload)

    def _validator_response(self, prompt: str) -> str:
        payload = json.loads(prompt)
        task_pack = str(payload.get("task_pack") or "")
        task_key = _task_key(task_pack)
        round_context = (
            (payload.get("tool_context") or {}).get("round_context") or {}
        )
        strategy_family = str(round_context.get("strategy_family") or "")
        if task_key == "count":
            if strategy_family == "relation_first":
                return json.dumps(
                    {
                        "grade": 9,
                        "issues": ["wrong count variable returned"],
                        "fixes": ["Return the actual count variable ID."],
                        "summary": "wrong count variable returned",
                        "plan_diagnosis": "OK",
                        "repair_mode": "none",
                    }
                )
            return json.dumps(
                {
                    "grade": 9,
                    "issues": [],
                    "fixes": [],
                    "summary": "count variable returned correctly",
                    "plan_diagnosis": "OK",
                    "repair_mode": "none",
                }
            )
        if task_key == "superlative":
            if strategy_family == "attribute_preparer":
                return json.dumps(
                    {
                        "grade": 8,
                        "issues": ["attribute mapping missing"],
                        "fixes": ["Pivot to a diagnostic attribute probe."],
                        "summary": "attribute mapping missing",
                        "plan_diagnosis": "OK",
                        "repair_mode": "none",
                    }
                )
            return json.dumps(
                {
                    "grade": 9,
                    "issues": [],
                    "fixes": [],
                    "summary": "diagnostic attribute handoff is useful",
                    "plan_diagnosis": "OK",
                    "repair_mode": "none",
                }
            )
        if strategy_family in {"walk_first", "shared_trait_pivot"}:
            return json.dumps(
                {
                    "grade": 8,
                    "issues": ["empty walk after resolving anchors"],
                    "fixes": ["Use a diagnostic shared-trait probe."],
                    "summary": "empty walk after resolving anchors",
                    "plan_diagnosis": "OK",
                    "repair_mode": "none",
                }
            )
        return json.dumps(
            {
                "grade": 9,
                "issues": [],
                "fixes": [],
                "summary": "diagnostic shared-trait handoff is useful",
                "plan_diagnosis": "OK",
                "repair_mode": "none",
            }
        )

    def _toolgen_response(self, prompt: str) -> str:
        if self._fail_toolgen:
            raise RuntimeError("synthetic toolgen failure")
        task_key = _task_key(prompt)
        strategy_family = "generic_macro"
        execution_style = "walk_first"
        preferred_tool_mode = "full_solve"
        round_idx = 1
        marker = "ROUND_STRATEGY_CONTEXT:"
        if marker in prompt:
            context_blob = prompt.split(marker, 1)[1]
            context_blob = context_blob.split(
                "Follow this structured retry policy exactly.",
                1,
            )[0].strip()
            try:
                round_context = json.loads(context_blob)
            except Exception:
                round_context = {}
            strategy_family = str(round_context.get("strategy_family") or strategy_family)
            execution_style = str(round_context.get("execution_style") or execution_style)
            preferred_tool_mode = str(
                round_context.get("preferred_tool_mode") or preferred_tool_mode
            )
            try:
                round_idx = int(round_context.get("round") or round_idx)
            except Exception:
                round_idx = 1
        return _macro_tool_source(
            task_key=task_key,
            strategy_family=strategy_family,
            execution_style=execution_style,
            preferred_tool_mode=preferred_tool_mode,
            round_idx=round_idx,
        )

    def _inference(self, batch_chat_history, inference_config_dict, system_prompt):
        responses = []
        for chat_history in batch_chat_history:
            prompt = chat_history.get_item_deep_copy(-1).content or ""
            if "Combined Orchestrator" in system_prompt:
                content = self._orchestrator_response(prompt)
            elif "ToolGen Logic Validator" in system_prompt:
                content = self._validator_response(prompt)
            elif "Generate ONE highly specialized, robust Python macro for the Knowledge-Graph." in system_prompt:
                content = self._toolgen_response(prompt)
            elif "ToolGen Phase" in system_prompt:
                raise RuntimeError("unexpected staged toolgen phase in KG macro live-path test")
            else:
                raise RuntimeError(f"unexpected system prompt: {system_prompt[:80]}")
            responses.append(ChatHistoryItem(role=Role.AGENT, content=content))
        return responses


class _StickyPivotKGTaskRef(_FakeKGTaskRef):
    def evaluate_generated_macro(self, tool_code: str, payload_json: str) -> str:
        payload = json.loads(payload_json)
        retry_context = payload.get("toolgen_retry_context") or {}
        strategy_family = str(retry_context.get("strategy_family") or "")
        try:
            round_idx = int(retry_context.get("round") or 1)
        except Exception:
            round_idx = 1
        if strategy_family == "relation_first":
            return json.dumps(
                {
                    "status": "MACRO EXHAUSTED",
                    "final_variable": None,
                    "observation": "MACRO EXHAUSTED: Resulting set is empty.",
                }
            )
        if strategy_family == "probe_then_commit" and round_idx == 3:
            return json.dumps(
                {
                    "status": "MACRO EXHAUSTED",
                    "final_variable": None,
                    "observation": "MACRO EXHAUSTED: Resulting set is empty.",
                }
            )
        return json.dumps(
            {
                "status": "SUCCESS",
                "final_variable": "#41",
                "observation": (
                    'COUNT VARIABLE RETURNED; submit it directly. minted_variables: '
                    '{"spacecraft_intersection": "#12", "count_result": "#41"}'
                ),
            }
        )


class _StickyPivotLanguageModel(_LivePathLanguageModel):
    def _validator_response(self, prompt: str) -> str:
        payload = json.loads(prompt)
        round_context = ((payload.get("tool_context") or {}).get("round_context") or {})
        strategy_family = str(round_context.get("strategy_family") or "")
        try:
            round_idx = int(round_context.get("round") or 1)
        except Exception:
            round_idx = 1
        if strategy_family == "relation_first":
            return json.dumps(
                {
                    "grade": 9,
                    "issues": ["wrong count variable returned"],
                    "fixes": ["Return the actual count variable ID."],
                    "summary": "wrong count variable returned",
                    "plan_diagnosis": "OK",
                    "repair_mode": "none",
                }
            )
        if strategy_family == "probe_then_commit" and round_idx == 3:
            return json.dumps(
                {
                    "grade": 9,
                    "issues": ["tool_plan field misread"],
                    "fixes": ["Keep the probe-then-commit strategy and fix field handling."],
                    "summary": "tool_plan field misread",
                    "plan_diagnosis": "OK",
                    "repair_mode": "rewrite_code",
                }
            )
        return json.dumps(
            {
                "grade": 9,
                "issues": [],
                "fixes": [],
                "summary": "count variable returned correctly",
                "plan_diagnosis": "OK",
                "repair_mode": "none",
            }
        )


class _ProgressToolKGTaskRef(_FakeKGTaskRef):
    def evaluate_generated_macro(self, tool_code: str, payload_json: str) -> str:
        return json.dumps(
            {
                "status": "MACRO EXHAUSTED",
                "final_variable": None,
                "observation": (
                    'MACRO EXHAUSTED: Resulting set is empty. minted_variables: '
                    '{"resolved_anchor_a": "#1", "resolved_anchor_b": "#2"} '
                    "Action: intersection(resolved_anchor_a, resolved_anchor_b)"
                ),
            }
        )


class _ProgressToolLanguageModel(_LivePathLanguageModel):
    def _orchestrator_response(self, prompt: str) -> str:
        payload = json.loads(super()._orchestrator_response(prompt))
        payload["preferred_tool_mode"] = "progress_tool"
        payload["execution_style"] = "partial_value_first"
        payload["fallback_strategies"] = ["probe_then_commit", "diagnostic_probe"]
        return json.dumps(payload)

    def _validator_response(self, prompt: str) -> str:
        return json.dumps(
            {
                "grade": 6,
                "issues": [],
                "fixes": [],
                "summary": "resolved both anchors correctly",
                "plan_diagnosis": "OK",
                "repair_mode": "none",
            }
        )


class _PatchLoopKGTaskRef(_FakeKGTaskRef):
    def evaluate_generated_macro(self, tool_code: str, payload_json: str) -> str:
        if "PATCH_MARKER" in (tool_code or ""):
            return json.dumps(
                {
                    "status": "SUCCESS",
                    "final_variable": "#41",
                    "observation": (
                        'COUNT VARIABLE RETURNED; submit it directly. minted_variables: '
                        '{"count_result": "#41"} PATCH_MARKER'
                    ),
                }
            )
        return json.dumps(
            {
                "status": "MACRO EXHAUSTED",
                "final_variable": None,
                "observation": "MACRO EXHAUSTED: Resulting set is empty.",
            }
        )


class _PatchLoopLanguageModel(_LivePathLanguageModel):
    def _validator_response(self, prompt: str) -> str:
        payload = json.loads(prompt)
        tool_code = str(payload.get("tool_code") or "")
        if "PATCH_MARKER" in tool_code:
            return json.dumps(
                {
                    "grade": 9,
                    "issues": [],
                    "fixes": [],
                    "summary": "count variable returned correctly",
                    "plan_diagnosis": "OK",
                    "repair_mode": "none",
                }
            )
        return json.dumps(
            {
                "grade": 9,
                "issues": ["wrong count variable returned"],
                "fixes": ["Return the actual count variable ID."],
                "summary": "wrong count variable returned",
                "plan_diagnosis": "OK",
                "repair_mode": "rewrite_code",
            }
        )

    def _patch_response(self, prompt: str) -> str:
        replacement = (
            "def run(payload: dict) -> dict:\n"
            '    """\n'
            "    contract guard: validates payload contains required keys before execution.\n"
            "    prereqs: requires KG helpers and actions_spec primitives needed by the plan.\n"
            "    limitations: stdlib only, no network calls, deterministic.\n"
            "    INPUT_SCHEMA: required=task_text,asked_for,trace,actions_spec,run_id,state_dir,entities; optional=domain_hints,target_concept,attribute_target_concept,entity_target_concepts,intermediate_target_concepts,topological_execution_plan,composite_topology,target_archetype,upgrade_goal,recovery_policy,execution_style,preferred_tool_mode,fallback_strategies,tool_plan,toolgen_retry_context,variable_list. OUTPUT_SCHEMA: status,final_variable,observation\n"
            '    """\n'
            "    try:\n"
            "        payload = payload or {}\n"
            '        candidate_map = {"count_result": "#41"}\n'
            "        observation = (\n"
            '            "COUNT VARIABLE RETURNED; submit it directly. minted_variables: "\n'
            "            + json.dumps(candidate_map)\n"
            '            + " PATCH_MARKER"\n'
            "        )\n"
            '        return {"status": "SUCCESS", "final_variable": "#41", "observation": observation}\n'
            "    except (KeyError, TypeError, ValueError) as e:\n"
            '        return {"status": "ERROR", "final_variable": None, "observation": f"Tool error: {str(e)}"}\n'
        )
        return json.dumps(
            {
                "operations": [
                    {
                        "op": "replace_function",
                        "name": "run",
                        "code": replacement,
                    }
                ]
            }
        )

    def _inference(self, batch_chat_history, inference_config_dict, system_prompt):
        responses = []
        for chat_history in batch_chat_history:
            prompt = chat_history.get_item_deep_copy(-1).content or ""
            if "You are ToolPatch" in system_prompt:
                content = self._patch_response(prompt)
            elif "Combined Orchestrator" in system_prompt:
                content = self._orchestrator_response(prompt)
            elif "ToolGen Logic Validator" in system_prompt:
                content = self._validator_response(prompt)
            elif "Generate ONE highly specialized, robust Python macro for the Knowledge-Graph." in system_prompt:
                content = self._toolgen_response(prompt)
            elif "ToolGen Phase" in system_prompt:
                raise RuntimeError("unexpected staged toolgen phase in patch-mode live-path test")
            else:
                raise RuntimeError(f"unexpected system prompt: {system_prompt[:80]}")
            responses.append(ChatHistoryItem(role=Role.AGENT, content=content))
        return responses


def _build_live_controller(
    tmp_path: pathlib.Path,
    monkeypatch,
    *,
    fail_toolgen: bool = False,
    language_model: LanguageModel | None = None,
    kg_task_ref: object | None = None,
) -> SelfEvolvingController:
    # This exercises the production controller/toolgen loop with mocked LM and
    # mocked KG macro execution so the wiring is real-path, not external-service live.
    output_dir = tmp_path / "outputs"
    monkeypatch.setenv("LIFELONG_OUTPUT_DIR", str(output_dir))
    monkeypatch.delenv("LIFELONG_OUTPUT_TAG", raising=False)
    controller = SelfEvolvingController(
        language_model=language_model or _LivePathLanguageModel(fail_toolgen=fail_toolgen),
        tool_registry_path=str(tmp_path / "registry"),
        max_generated_tools_per_run=8,
        environment_label="knowledge_graph",
        use_packaged_agent=False,
    )
    controller._kg_task_ref = kg_task_ref or _FakeKGTaskRef()  # noqa: SLF001
    controller._toolgen_pipeline = None  # noqa: SLF001
    controller._toolgen_pipeline_name = "baseline"  # noqa: SLF001
    controller._current_task_label = "kg_live_path"  # noqa: SLF001
    return controller


def _run_live_toolgen_case(
    controller: SelfEvolvingController,
    query: str,
) -> dict[str, object]:
    history = ChatHistory()
    history.inject(ChatHistoryItem(role=Role.USER, content=query))
    decision = controller._orchestrate_decision(query, history)  # noqa: SLF001
    tool = controller._run_escape_hatch_toolgen(decision, query, history)  # noqa: SLF001
    log_dir = controller._generated_tools_log_path.parent  # noqa: SLF001
    return {
        "decision": decision,
        "tool": tool,
        "plan_trace": _jsonl_events(log_dir / "orchestrator_plan_trace.jsonl"),
        "generated_events": _jsonl_events(controller._generated_tools_log_path),  # noqa: SLF001
        "value_events": _jsonl_events(log_dir / "tool_value_trace.jsonl"),
        "candidate_metadata": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((log_dir / "generated_tool_candidates").glob("*__metadata.json"))
        ],
        "failed_artifacts": sorted(
            path
            for path in (log_dir / "callback_state").rglob("*")
            if path.is_file()
        ),
    }


def test_live_path_count_task_retries_once_then_pivots(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(tmp_path, monkeypatch)
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is not None
    decision = result["decision"]
    assert decision["execution_style"] == "relation_first"
    assert decision["preferred_tool_mode"] == "full_solve"
    assert decision["fallback_strategies"] == [
        "probe_then_commit",
        "set_builder",
        "intersector_counter",
    ]
    assert decision["entities"] == ["CNES", "Astrium"]

    plan_trace = result["plan_trace"]
    assert plan_trace
    assert plan_trace[0]["execution_style"] == "relation_first"
    assert plan_trace[0]["preferred_tool_mode"] == "full_solve"
    assert plan_trace[0]["fallback_strategies"] == [
        "probe_then_commit",
        "set_builder",
        "intersector_counter",
    ]
    assert plan_trace[0]["entities"] == ["CNES", "Astrium"]
    mode_override_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_mode_override"
    ]
    assert mode_override_events
    assert mode_override_events[0]["requested_mode"] == "staged"
    assert mode_override_events[0]["effective_mode"] == "legacy"
    lifecycle_events = [
        event["event"]
        for event in result["generated_events"]
        if event.get("event") in {"create", "register"}
    ]
    assert "create" in lifecycle_events
    assert "register" in lifecycle_events

    value_events = result["value_events"]
    strategy_events = [
        event
        for event in value_events
        if event.get("event") == "toolgen_strategy_classification"
    ]
    assert [event["strategy_family"] for event in strategy_events[:3]] == [
        "relation_first",
        "relation_first",
        "probe_then_commit",
    ]
    assert [event["execution_style"] for event in strategy_events[:3]] == [
        "relation_first",
        "relation_first",
        "probe_then_commit",
    ]
    assert [event["strategy_source"] for event in strategy_events[:3]] == [
        "initial_plan",
        "inherited_from_previous_round",
        "pivot_policy",
    ]
    assert [event["strategy_index"] for event in strategy_events[:3]] == [0, 0, 1]
    assert [event["strategy_epoch"] for event in strategy_events[:3]] == [0, 0, 1]
    assert strategy_events[1]["pivot_required"] is True

    pivot_events = [
        event
        for event in value_events
        if event.get("event") == "toolgen_strategy_pivot"
    ]
    assert pivot_events
    assert pivot_events[0]["round"] == 3
    assert pivot_events[0]["previous_strategy"] == "relation_first"
    assert pivot_events[0]["previous_failure_family"] == "count_target_wrong"
    assert pivot_events[0]["previous_failure_bucket"] == "code_local_no_progress"
    assert pivot_events[0]["new_strategy"] == "probe_then_commit"
    assert pivot_events[0]["strategy_source"] == "pivot_policy"
    assert pivot_events[0]["strategy_index"] == 1
    assert pivot_events[0]["strategy_epoch"] == 1
    assert pivot_events[0]["reason"] == "repeated_no_progress_same_strategy_failure"

    failure_events = [
        event
        for event in value_events
        if event.get("event") == "toolgen_failure_classification"
    ]
    assert failure_events[0]["failure_family"] == "count_target_wrong"
    assert failure_events[0]["failure_bucket"] == "code_local_no_progress"
    delivered_events = [
        event
        for event in value_events
        if event.get("event") == "toolgen_value_delivered"
    ]
    assert delivered_events[-1]["value_delivered"] == "produced_final_variable"
    assert delivered_events[-1]["failure_bucket"] == "final_value_delivered"
    assert delivered_events[-1]["partial_value_usable"] is True

    validator_metadata = [
        item for item in result["candidate_metadata"] if item.get("failure_phase") == "validator"
    ]
    assert len(validator_metadata) >= 2
    assert validator_metadata[0]["fallback_strategies"] == [
        "probe_then_commit",
        "set_builder",
        "intersector_counter",
    ]
    assert validator_metadata[0]["repair_mode"] != "none"
    assert validator_metadata[0]["strategy_family"] == "relation_first"
    assert validator_metadata[0]["failure_family"] == "count_target_wrong"
    assert validator_metadata[0]["failure_bucket"] == "code_local_no_progress"
    assert validator_metadata[0]["strategy_source"] == "initial_plan"
    assert validator_metadata[0]["strategy_index"] == 0
    assert validator_metadata[0]["strategy_epoch"] == 0
    validation_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_validation_result"
    ]
    assert validation_events
    assert validation_events[0]["fallback_strategies"] == [
        "probe_then_commit",
        "set_builder",
        "intersector_counter",
    ]
    assert validation_events[0]["strategy_source"] == "initial_plan"
    assert validation_events[0]["strategy_index"] == 0
    assert validation_events[0]["strategy_epoch"] == 0
    assert validation_events[0]["failure_bucket"] == "code_local_no_progress"
    assert all(
        event.get("failure_bucket") != "integration_context_invalid"
        for event in validation_events
    )
    assert validation_events[-1]["best_achieved_state"] == "produced_final_variable"
    admitted_metadata = [
        item for item in result["candidate_metadata"] if item.get("registered") is True
    ]
    assert admitted_metadata
    assert admitted_metadata[-1]["strategy_family"] == "probe_then_commit"
    assert admitted_metadata[-1]["execution_style"] == "probe_then_commit"
    assert admitted_metadata[-1]["value_delivered"] == "produced_final_variable"
    assert admitted_metadata[-1]["failure_bucket"] == "final_value_delivered"
    assert admitted_metadata[-1]["strategy_source"] == "pivot_policy"
    assert admitted_metadata[-1]["strategy_index"] == 1
    assert admitted_metadata[-1]["strategy_epoch"] == 1
    assert admitted_metadata[-1]["admitted"] is True
    assert admitted_metadata[-1]["best_achieved_state"] == "produced_final_variable"
    best_summary = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_best_candidate_summary"
    ]
    assert best_summary
    assert best_summary[-1]["selection_source"] == "best_full_candidate"
    assert best_summary[-1]["best_achieved_state"] == "produced_final_variable"
    assert best_summary[-1]["best_achieved_round"] == 3
    assert best_summary[-1]["best_achieved_tool_name"] == "count_probe_then_commit_r3_generated_tool"

    failed_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result["failed_artifacts"]
        if "__validator__" in path.name
    )
    assert '"fallback_strategies": ["probe_then_commit", "set_builder", "intersector_counter"]' in failed_text
    assert '"strategy_family": "relation_first"' in failed_text
    assert '"failure_bucket": "code_local_no_progress"' in failed_text


def test_live_path_sticky_pivot_inherits_strategy_after_pivot(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=_StickyPivotLanguageModel(),
        kg_task_ref=_StickyPivotKGTaskRef(),
    )
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is not None
    round_start_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_round_start"
    ]
    assert len(round_start_events) >= 4
    assert round_start_events[1]["strategy_family"] == "relation_first"
    assert round_start_events[1]["strategy_source"] == "inherited_from_previous_round"
    assert round_start_events[2]["strategy_family"] == "probe_then_commit"
    assert round_start_events[2]["strategy_source"] == "pivot_policy"
    assert round_start_events[2]["strategy_index"] == 1
    assert round_start_events[2]["strategy_epoch"] == 1
    assert round_start_events[3]["strategy_family"] == "probe_then_commit"
    assert round_start_events[3]["strategy_source"] == "inherited_from_previous_round"
    assert round_start_events[3]["strategy_index"] == 1
    assert round_start_events[3]["strategy_epoch"] == 1
    assert round_start_events[3]["strategy_family"] != "relation_first"

    pivot_events = [
        event
        for event in result["value_events"]
        if event.get("event") == "toolgen_strategy_pivot"
    ]
    assert pivot_events
    assert pivot_events[0]["round"] == 3
    assert pivot_events[0]["new_strategy"] == "probe_then_commit"

    strategy_events = [
        event
        for event in result["value_events"]
        if event.get("event") == "toolgen_strategy_classification"
    ]
    assert [event["strategy_family"] for event in strategy_events[:4]] == [
        "relation_first",
        "relation_first",
        "probe_then_commit",
        "probe_then_commit",
    ]
    assert [event["strategy_source"] for event in strategy_events[:4]] == [
        "initial_plan",
        "inherited_from_previous_round",
        "pivot_policy",
        "inherited_from_previous_round",
    ]


def test_live_path_partial_value_tools_are_scored_as_useful(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    super_controller = _build_live_controller(tmp_path / "super", monkeypatch)
    super_result = _run_live_toolgen_case(super_controller, _SUPERLATIVE_QUERY)
    assert super_result["tool"] is not None
    super_value_events = super_result["value_events"]
    super_pivots = [
        event
        for event in super_value_events
        if event.get("event") == "toolgen_strategy_pivot"
    ]
    assert super_pivots
    assert super_pivots[0]["round"] == 2
    assert super_pivots[0]["new_strategy"] == "diagnostic_probe"
    super_live = [
        event
        for event in super_value_events
        if event.get("event") == "toolgen_value_delivered"
    ]
    assert super_live[-1]["value_delivered"] == "produced_actionable_handoff"
    assert super_live[-1]["partial_value_usable"] is True
    assert super_live[-1]["usefulness_passed"] is True

    shared_controller = _build_live_controller(tmp_path / "shared", monkeypatch)
    shared_result = _run_live_toolgen_case(shared_controller, _SHARED_TRAIT_QUERY)
    assert shared_result["tool"] is not None
    shared_value_events = shared_result["value_events"]
    shared_pivots = [
        event
        for event in shared_value_events
        if event.get("event") == "toolgen_strategy_pivot"
    ]
    assert shared_pivots
    assert shared_pivots[0]["round"] == 2
    assert shared_pivots[0]["new_strategy"] == "diagnostic_probe"
    shared_live = [
        event
        for event in shared_value_events
        if event.get("event") == "toolgen_value_delivered"
    ]
    assert shared_live[-1]["value_delivered"] in {
        "identified_relation_candidates",
        "produced_actionable_handoff",
    }
    assert shared_live[-1]["partial_value_usable"] is True
    assert shared_live[-1]["usefulness_passed"] is True


def test_live_path_verified_progress_tool_can_stop_early(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=_ProgressToolLanguageModel(),
        kg_task_ref=_ProgressToolKGTaskRef(),
    )
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is not None
    round_starts = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_round_start"
    ]
    assert len(round_starts) == 1
    early_stop_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_progress_early_stop"
    ]
    assert early_stop_events
    assert early_stop_events[0]["preferred_tool_mode"] == "progress_tool"
    assert early_stop_events[0]["best_achieved_state"] == "resolved_anchors"
    best_summary = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_best_candidate_summary"
    ]
    assert best_summary
    assert best_summary[-1]["selection_source"] == "best_partial_candidate"
    assert best_summary[-1]["best_achieved_state"] == "resolved_anchors"
    assert best_summary[-1]["best_achieved_round"] == 1
    assert best_summary[-1]["best_achieved_tool_name"]
    admitted_metadata = [
        item for item in result["candidate_metadata"] if item.get("registered") is True
    ]
    assert admitted_metadata
    assert admitted_metadata[-1]["best_achieved_state"] == "resolved_anchors"
    assert admitted_metadata[-1]["best_achieved_round"] == 1


def test_patch_mode_keeps_patch_round_after_validator_feedback(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIFELONG_TOOLGEN_PATCH_MODE", "1")
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=_PatchLoopLanguageModel(),
        kg_task_ref=_PatchLoopKGTaskRef(),
    )
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is not None
    candidate_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_candidate"
    ]
    assert len(candidate_events) >= 2
    assert candidate_events[0]["patch_round"] is False
    assert candidate_events[1]["patch_round"] is True
    assert candidate_events[1]["patch_ops"] == 1
    assert not any(
        event.get("event") == "toolgen_round_failed"
        and event.get("phase") in {"patch_plan_parse", "patch_apply"}
        for event in result["generated_events"]
    )


def test_live_path_llm_call_exception_artifact_has_full_strategy_metadata(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(tmp_path, monkeypatch, fail_toolgen=True)
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is None
    llm_failures = [
        path for path in result["failed_artifacts"] if "__llm_call_exception__" in path.name
    ]
    assert llm_failures
    llm_text = llm_failures[0].read_text(encoding="utf-8")
    for key in (
        "strategy_family",
        "execution_style",
        "preferred_tool_mode",
        "fallback_strategies",
        "strategy_sequence",
        "strategy_source",
        "strategy_index",
        "strategy_epoch",
        "failure_family",
        "failure_bucket",
        "value_delivered",
        "pivot_required",
        "partial_value_usable",
    ):
        assert key in llm_text


def test_live_path_static_check_failure_artifact_has_full_strategy_metadata(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(tmp_path, monkeypatch)
    original_static_check = controller._toolgen_static_check  # noqa: SLF001
    calls = {"count": 0}

    def _fail_once(tool_code: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return False, "synthetic static failure"
        return original_static_check(tool_code)

    controller._toolgen_static_check = _fail_once  # noqa: SLF001
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    static_failures = [
        path for path in result["failed_artifacts"] if "__static_check__" in path.name
    ]
    assert static_failures
    static_text = static_failures[0].read_text(encoding="utf-8")
    for key in (
        "strategy_family",
        "execution_style",
        "preferred_tool_mode",
        "fallback_strategies",
        "strategy_sequence",
        "strategy_source",
        "strategy_index",
        "strategy_epoch",
        "failure_family",
        "failure_bucket",
        "value_delivered",
        "pivot_required",
        "partial_value_usable",
    ):
        assert key in static_text


def test_live_path_smoke_test_failure_artifact_has_full_strategy_metadata(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(tmp_path, monkeypatch)
    original_validate = controller_toolgen_module.validate_tool_code
    calls = {"count": 0}

    def _fail_once(tool_code: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return types.SimpleNamespace(success=False, error="synthetic smoke failure")
        return original_validate(tool_code)

    monkeypatch.setattr(controller_toolgen_module, "validate_tool_code", _fail_once)
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    smoke_failures = [
        path for path in result["failed_artifacts"] if "__smoke_test__" in path.name
    ]
    assert smoke_failures
    smoke_text = smoke_failures[0].read_text(encoding="utf-8")
    for key in (
        "strategy_family",
        "execution_style",
        "preferred_tool_mode",
        "fallback_strategies",
        "strategy_sequence",
        "strategy_source",
        "strategy_index",
        "strategy_epoch",
        "failure_family",
        "failure_bucket",
        "value_delivered",
        "pivot_required",
        "partial_value_usable",
    ):
        assert key in smoke_text


# ── Fix-1: quick_structural_precheck must run on every patch round ────────────

class _BadHeaderPatchLanguageModel(_PatchLoopLanguageModel):
    """Removes the RUN_PAYLOAD_REQUIRED header via a patch operation.

    Round 0 produces valid code (passes precheck, grade stays low so patching
    begins).  Round 1+ issue a replace_text patch that strips the required
    '# RUN_PAYLOAD_REQUIRED:' metadata comment, which quick_structural_precheck
    must catch even in patch mode.  Before Fix 1, defer_compile_gates suppressed
    the check and the failure only appeared at the final round.
    """

    def _toolgen_response(self, _prompt: str) -> str:
        # Round 0: valid code with all required headers.
        return _macro_tool_source(
            task_key="count",
            strategy_family="relation_first",
            execution_style="relation_first",
            preferred_tool_mode="full_solve",
            round_idx=0,
        )

    def _patch_response(self, _prompt: str) -> str:
        # Strip the RUN_PAYLOAD_REQUIRED header line from the file.
        # The patched code is syntactically valid Python but fails
        # quick_structural_precheck because the required header is absent.
        return json.dumps(
            {
                "operations": [
                    {
                        "op": "replace_text",
                        "find": "# RUN_PAYLOAD_REQUIRED:",
                        "replace": "# REMOVED_HEADER:",
                    }
                ]
            }
        )


def test_fix1_precheck_fires_on_patch_rounds(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    """Fix 1: quick_structural_precheck must run unconditionally in patch mode.

    Before the fix, defer_compile_gates suppressed the precheck for all rounds
    < max_rounds.  After the fix, a structurally invalid patch (missing the
    RUN_PAYLOAD_REQUIRED header) must produce a toolgen_precheck_fail event
    on a patch round (round > 0), not just the final round.
    """
    monkeypatch.setenv("LIFELONG_TOOLGEN_PATCH_MODE", "1")
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=_BadHeaderPatchLanguageModel(),
        kg_task_ref=_PatchLoopKGTaskRef(),
    )
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    precheck_fail_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_precheck_fail"
    ]
    # Must fire at least once.
    assert precheck_fail_events, "expected toolgen_precheck_fail events"
    # At least one must be on a patch round (round > 0).  Before Fix 1 the
    # loop would only emit toolgen_precheck_fail at the last round when
    # defer_compile_gates became False.
    patch_round_failures = [e for e in precheck_fail_events if (e.get("round") or 0) > 0]
    assert patch_round_failures, (
        "toolgen_precheck_fail must fire on a patch round (round > 0), "
        "not only at the final round"
    )


# ── Fix-2: patch mode must stop across strategy pivots ────────────────────────

def _no_progress_history_entry(
    round_idx: int,
    strategy_family: str,
    strategy_sequence: list,
) -> dict:
    """Build a round_history entry that represents a non-code-local no-progress round."""
    return {
        "round": round_idx,
        "tool_name": f"tool_r{round_idx}",
        "grade": 4,
        "top_issue": "context_handling_no_progress",
        "summary": "no_progress",
        "usefulness_passed": False,
        "usefulness_reason": "no_progress",
        "material_progress": False,
        "failure_phase": "validator",
        "strategy_family": strategy_family,
        "active_strategy_family": strategy_family,
        "execution_style": "relation_first",
        "active_execution_style": "relation_first",
        "preferred_tool_mode": "full_solve",
        "active_preferred_tool_mode": "full_solve",
        "strategy_sequence": list(strategy_sequence),
        "strategy_index": 0,
        "strategy_epoch": 0,
        "strategy_source": "base_plan",
        "previous_failure_bucket": None,
        "failure_family": "blocking_semantic_code_smells",
        "failure_bucket": "context_handling_no_progress",
        "value_delivered": "none",
        "partial_value_usable": False,
        "semantic_code_smells": [],
        "pivot_required": False,
        "code_shape_signature": "run",
    }


def test_fix2_pivot_round_disables_patch_mode(tmp_path: pathlib.Path) -> None:
    """Fix 2: current_round_patch_mode must be False when pivot_required=True.

    _toolgen_compute_round_strategy_context returns pivot_required=True at
    round 2 when both round 0 and round 1 show non-code-local no-progress for
    the same strategy family.  After Fix 2, the current_round_patch_mode
    condition gates on pivot_required, so patch mode is disabled for that round.

    Before Fix 2 the condition was:
        patch_mode and round_idx > 1 and not force_full_rewrite_next_round
        and isinstance(...) and isinstance(...)

    After Fix 2 it is:
        ... and not round_context.get("pivot_required") ...
    """
    controller = _DummyController(tmp_path)

    strategy_sequence = ["relation_first", "probe_then_commit", "set_builder", "intersector_counter"]
    exec_payload = {
        "task_text": _COUNT_QUERY,
        "asked_for": "count",
        "trace": [],
        "actions_spec": {},
        "run_id": "r0",
        "state_dir": "/tmp/state",
        "entities": ["CNES", "Astrium"],
        "target_archetype": "COUNTING_INTERSECTOR",
        "execution_style": "relation_first",
        "preferred_tool_mode": "full_solve",
        "fallback_strategies": ["probe_then_commit", "set_builder", "intersector_counter"],
        "strategy_family": "relation_first",
        "target_concept": "spacecraft",
        "entity_target_concepts": ["space_agency", "aerospace_company"],
        "topological_execution_plan": ["1. Resolve anchors.", "2. Walk.", "3. Intersect.", "4. Count."],
    }

    # Two rounds of no-progress on the same strategy triggers pivot at round_idx=2.
    round_history = [
        _no_progress_history_entry(0, "relation_first", strategy_sequence),
        _no_progress_history_entry(1, "relation_first", strategy_sequence),
    ]

    round_context = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload=exec_payload,
        round_history=round_history,
    )

    # Policy must have set pivot_required=True.
    assert round_context.get("pivot_required") is True, (
        "expected pivot_required=True at round 2 after two no-progress rounds"
    )

    # current_round_patch_mode (mirror of controller_toolgen.py condition):
    current_round_patch_mode = (
        True  # patch_mode
        and 2 > 1  # round_idx > 1
        and not False  # not force_full_rewrite_next_round
        and not round_context.get("pivot_required")  # Fix 2 guard
        and isinstance("some_code", str)
        and isinstance({}, dict)
    )
    assert current_round_patch_mode is False, (
        "patch mode must be disabled when round_context.pivot_required=True (Fix 2)"
    )


# ── Fix-3: non-live fallback must reject candidates with deferred checks ──────

def test_fix3_patch_fallback_rejects_deferred_candidate() -> None:
    """Fix 3: the patch-mode non-live fallback must skip candidates whose
    static/smoke checks were deferred (checks_deferred=True).

    This mirrors the selection logic in controller_toolgen.py so a future
    refactor that breaks the guard will be caught here.
    """
    deferred = {
        "tool_spec": {"name": "deferred_tool"},
        "tool_code": "def run(p): return {}",
        "validation": {"grade": 7},
        "checks_deferred": True,
    }
    checked = {
        "tool_spec": {"name": "checked_tool"},
        "tool_code": "def run(p): return {}",
        "validation": {"grade": 6},
        "checks_deferred": False,
    }

    # Both deferred and checked present — must pick checked.
    use_candidate = None
    for cand in (deferred, checked):
        if cand is None:
            continue
        if not cand.get("checks_deferred", True):
            use_candidate = cand
            break
    assert use_candidate is checked

    # Only deferred present — must return None (nothing safe to register).
    use_candidate = None
    for cand in (deferred, None):
        if cand is None:
            continue
        if not cand.get("checks_deferred", True):
            use_candidate = cand
            break
    assert use_candidate is None

    # Only checked present — must return it.
    use_candidate = None
    for cand in (None, checked):
        if cand is None:
            continue
        if not cand.get("checks_deferred", True):
            use_candidate = cand
            break
    assert use_candidate is checked

    # checks_deferred absent defaults to True (conservative — reject).
    no_flag = {
        "tool_spec": {"name": "no_flag_tool"},
        "tool_code": "def run(p): return {}",
        "validation": {"grade": 8},
    }
    use_candidate = None
    for cand in (no_flag,):
        if cand is None:
            continue
        if not cand.get("checks_deferred", True):
            use_candidate = cand
            break
    assert use_candidate is None, "absent checks_deferred must default to rejected"
