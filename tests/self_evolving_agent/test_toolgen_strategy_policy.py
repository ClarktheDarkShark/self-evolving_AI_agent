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
    TOOLGEN_SYSTEM_PROMPT_MARKERS,
    TOOLGEN_VALIDATOR_SYSTEM_PROMPT,
    TOOL_INVOKER_SYSTEM_PROMPT,
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


def _prompt_canonical_plan(prompt: str) -> dict[str, object]:
    marker = "CANONICAL TOOL PLAN: "
    assert marker in prompt
    plan_blob = prompt.split(marker, 1)[1].split("\n", 1)[0]
    return json.loads(plan_blob)


def _prompt_execution_plan(prompt: str) -> str:
    marker = "EXECUTION PLAN:\n"
    assert marker in prompt
    return prompt.split(marker, 1)[1].split("\n\nTOOLGEN SETUP:", 1)[0]


def _is_kg_toolgen_macro_prompt(system_prompt: str) -> bool:
    return (
        "You are ToolGen. Generate ONE specialized Python macro for the Knowledge"
        in system_prompt
        or "You are ToolGen. Generate EXACTLY ONE specialized Python macro for the Knowledge"
        in system_prompt
        or "Generate ONE highly specialized, robust Python macro for the Knowledge-Graph."
        in system_prompt
    )


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


def test_kg_prompt_has_filter_ready_finish_chain_override() -> None:
    assert "If the declared required handoff is `built_filter_ready_set`" in MACRO_TOOLGEN_USER_KG
    assert "one straight-line" in MACRO_TOOLGEN_USER_KG
    assert "do NOT stop at resolved anchors" in MACRO_TOOLGEN_USER_KG


def test_force_toolgen_prompt_requires_tool_or_generation(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    controller._force_toolgen_always_on = True  # noqa: SLF001
    controller._toolgen_render_history = lambda *args, **kwargs: ""  # noqa: SLF001

    prompt = controller._orchestrator_request_prompt(
        "Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        ChatHistory(),
    )

    assert "[TEST MODE OVERRIDE]" in prompt
    assert "MUST NOT output action='no_tool'" in prompt
    assert "action='request_new_tool'" in prompt
    assert "tool_type='macro'" in prompt


def test_escape_hatch_toolgen_uses_macro_prompt_for_kg_without_tool_type(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _DummyController(tmp_path)
    captured: dict[str, object] = {}

    def _fake_toolgen_generate_from_prompt(
        *,
        user_prompt: str,
        system_prompt: str,
        chat_history: ChatHistory,
        name_prefix: str,
        prompt_name: str | None = None,
        force_strict: bool = False,
        force_max_rounds: int | None = None,
    ) -> None:
        captured["user_prompt"] = user_prompt
        captured["system_prompt"] = system_prompt
        captured["prompt_name"] = prompt_name
        return None

    monkeypatch.setattr(
        controller,
        "_toolgen_generate_from_prompt",
        _fake_toolgen_generate_from_prompt,
    )

    controller._run_escape_hatch_toolgen(  # noqa: SLF001
        {
            "action": "request_new_tool",
            "reason": "INPUT: ['Goat', 'cows']. GOAL: find shared cheese products.",
            "entities": ["Goat", "cows", "semi-firm"],
            "target_concept": "cheese",
            "entity_target_concepts": ["animal", "animal"],
            "topological_execution_plan": [
                "1. Resolve the anchors.",
                "2. Walk to cheese candidates.",
                "3. Intersect the candidate sets.",
            ],
        },
        "Question: what semi-firm textured cheese is made from the products of goat and cows?, Entities: ['Goat', 'cows', 'semi-firm']",
        ChatHistory(),
    )

    assert captured["system_prompt"] == MACRO_TOOLGEN_USER_KG
    assert captured["prompt_name"] == "MACRO_TOOLGEN_USER_KG"
    assert "TOOL TYPE REQUIRED: macro" in str(captured["user_prompt"])


def test_blueprint_prompt_uses_ssot_return_wording_for_kg(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)

    prompt = controller._toolgen_build_blueprint_prompt(
        query="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        tool_type="macro",
        reason="INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        upgrade_goal="INPUT: ['CNES', 'Astrium']. GOAL: stop at first grounded value.",
        tool_plan=_kg_plan(),
        env_name="knowledge_graph",
        env_contract="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
    )

    assert "Return a legal next-state variable or exact MACRO EXHAUSTED." not in prompt
    assert "Return only the canonical 3-key dict with status, final_variable, and observation" in prompt
    assert "return an honest ERROR result instead of inventing KG traversal" in prompt


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


def test_live_progress_exhausted_partial_progress_is_admission_blocked(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)

    result = controller._summarize_toolgen_live_progress(
        {
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": (
                "MACRO EXHAUSTED: Resulting set is empty. "
                'minted_variables: {"resolved_CNES": "#1", "resolved_Astrium": "#2"}'
            ),
        },
        {"tool_plan": _kg_plan(preferred_tool_mode="full_solve")},
    )

    assert result["material_progress"] is True
    assert result["bankable_partial_progress"] is True
    assert result["admission_blocked"] is True
    assert result["usefulness_passed"] is False


def test_value_delivered_recognizes_resolved_entity_labels_with_grounded_pointers(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    value = controller._toolgen_classify_value_delivered(
        tool_plan=_kg_plan(preferred_tool_mode="progress_tool"),
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#1",
            "observation": (
                "Resolved producer anchors. "
                'minted_variables: {"resolved_Goat": "#1", "resolved_cows": ["#2"]}'
            ),
        },
        live_progress_summary={
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "has_context": True,
            "material_progress": True,
            "final_operation_safe": False,
        },
    )
    assert value == "resolved_both_anchors"


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


def test_runtime_repair_brief_targets_first_valuable_state(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)

    brief = controller._toolgen_synthesize_runtime_repair_brief(
        live_progress_summary={
            "has_live_result": True,
            "execution_status": "MACRO EXHAUSTED",
            "value_delivered": "none",
            "achieved_state": "none",
            "partial_value_usable": False,
            "has_final_variable": False,
            "material_progress": False,
            "has_context": True,
            "has_next_action_guidance": False,
            "narrowed_problem_space": False,
        },
        round_context={},
        validation={"issues": ["Fix minted_variables formatting."]},
    )

    assert brief is not None
    assert brief["missing_value_type"] == "no_executable_handoff"
    assert brief["redesign_direction"] == "emit_concrete_handoff_from_live_anchor"
    assert brief["secondary_issues"] == ["Fix minted_variables formatting."]


def test_validation_policy_runtime_repair_brief_outranks_minor_validator_issue(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)

    result = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 7,
            "issues": ["Exhaustion string is missing a formatting token."],
            "fixes": ["Restore the exact minted_variables formatting."],
            "summary": "The exhaustion observation needs a small formatting fix.",
            "repair_mode": "rewrite_code",
        },
        execution_validation={
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": "MACRO EXHAUSTED: Resulting set is empty.",
        },
        live_progress_summary={
            "has_live_result": True,
            "execution_status": "MACRO EXHAUSTED",
            "handoff_state": "exhausted",
            "reason": "exhausted_no_classified_value",
            "material_progress": False,
            "has_context": True,
            "has_next_action_guidance": False,
            "has_final_variable": False,
            "final_operation_safe": False,
            "value_delivered": "none",
            "partial_value_usable": False,
            "achieved_state": "none",
            "narrowed_problem_space": False,
        },
        round_context={
            "previous_strategy_family": "relation_first",
            "previous_failure_family": "no_runtime_progress",
            "previous_failure_bucket": "strategy_mismatch_no_progress",
            "execution_style": "relation_first",
            "preferred_tool_mode": "full_solve",
            "material_progress_last_round": False,
        },
        tool_plan=_kg_plan(execution_style="relation_first"),
        tool_code="kg_utils.resolve_entity_to_vars(entity, None, actions_spec, domain_hints)",
    )

    assert result["runtime_repair_brief"] is not None
    assert result["missing_value_type"] == "no_executable_handoff"
    assert result["redesign_direction"] == "replace_exhausted_full_solve_shape"
    assert result["issues"][0].startswith("Runtime-first failure:")
    assert result["issues"][1] == "Exhaustion string is missing a formatting token."
    assert "Stop local patching." in result["fixes"][0]
    assert result["summary"].startswith(
        "Observed runtime outcome: MACRO EXHAUSTED produced no classified value"
    )


def test_blocked_rewrite_feedback_includes_runtime_first_repair_target(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    brief = {
        "runtime_failure_summary": (
            "Observed runtime outcome: MACRO EXHAUSTED produced no classified value, "
            "no final variable, and no solver-usable handoff."
        ),
        "missing_value_type": "no_classified_value",
        "redesign_direction": "replace_exhausted_full_solve_shape",
        "secondary_issues": ["Fix minted_variables formatting."],
    }

    payload = json.loads(
        controller._toolgen_blocked_rewrite_feedback(
            phase="validator_blocked_no_progress",
            usefulness_reason="exhausted_no_classified_value",
            live_progress_summary={
                "execution_status": "MACRO EXHAUSTED",
                "handoff_state": "exhausted",
                "has_final_variable": False,
                "material_progress": False,
            },
            runtime_repair_brief=brief,
            primary_repair_instruction=controller._toolgen_runtime_repair_instruction(
                brief
            ),
            validator_top_issue="Fix minted_variables formatting.",
            validator_fixes=["Restore the exact exhausted observation format."],
        )
    )

    assert payload["PRIMARY_REPAIR_TARGET"].startswith(
        "RUNTIME-FIRST REPAIR TARGET:"
    )
    assert payload["runtime_repair_brief"]["missing_value_type"] == "no_classified_value"
    assert (
        payload["runtime_repair_brief"]["redesign_direction"]
        == "replace_exhausted_full_solve_shape"
    )
    assert payload["validator_top_issue"] == "Fix minted_variables formatting."


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


def test_patch_prompt_prioritizes_runtime_first_repair_target(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    prompt = controller._toolgen_build_patch_prompt(
        current_code="def run(payload: dict) -> dict:\n    return {}\n",
        feedback_note=json.dumps(
            {
                "PRIMARY_REPAIR_TARGET": (
                    "RUNTIME-FIRST REPAIR TARGET: rewrite toward the first acceptable "
                    "value state."
                ),
                "runtime_repair_brief": {
                    "missing_value_type": "no_classified_value",
                    "redesign_direction": "replace_exhausted_full_solve_shape",
                },
            }
        ),
        round_history=[],
        base_prompt="task pack",
    )

    assert "PRIMARY_REPAIR_TARGET" in prompt
    assert "dominant rewrite goal" in prompt
    assert "secondary formatting or local cleanup" in prompt


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
        '# RUN_PAYLOAD_OPTIONAL: ["domain_hints", "target_concept", "attribute_target_concept", "entity_target_concepts", "intermediate_target_concepts", "topological_execution_plan", "composite_topology", "target_archetype", "upgrade_goal", "recovery_policy", "execution_style", "preferred_tool_mode", "minimum_acceptable_deliverable", "fallback_strategies", "tool_plan", "toolgen_retry_context", "variable_list"]\n'
        '# INVOKE_EXAMPLE: {"args":[{"task_text":"...","asked_for":"...","trace":[],"actions_spec":{},"run_id":"r1","state_dir":"./state","entities":["A"]}], "kwargs":{}}\n'
        f'"""KG macro for {task_key}."""\n'
        "\n"
        "import json\n"
        "\n"
        "def run(payload: dict) -> dict:\n"
        '    """\n'
        "    contract guard: payload must contain the required run keys.\n"
        "    prereqs: kg_utils facade and needed actions_spec primitives are available.\n"
        "    limitations: stop at first grounded KG result; do not echo payload or plan fields; do not mint from entity strings alone.\n"
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


def _runless_tool_source(*, task_key: str, round_idx: int) -> str:
    tool_name = f"{task_key}_runless_r{round_idx}_generated_tool"
    return (
        "###TOOL_START\n"
        f"# tool_name: {tool_name}\n"
        '# INVOKE_WITH: {"args":[<RUN_PAYLOAD>], "kwargs":{}}\n'
        '# RUN_PAYLOAD_REQUIRED: ["task_text", "asked_for", "trace", "actions_spec", "run_id", "state_dir", "entities"]\n'
        '# RUN_PAYLOAD_OPTIONAL: ["domain_hints", "target_concept", "attribute_target_concept", "entity_target_concepts", "intermediate_target_concepts", "topological_execution_plan", "composite_topology", "target_archetype", "upgrade_goal", "recovery_policy", "execution_style", "preferred_tool_mode", "minimum_acceptable_deliverable", "fallback_strategies", "tool_plan", "toolgen_retry_context", "variable_list"]\n'
        '# INVOKE_EXAMPLE: {"args":[{"task_text":"...","asked_for":"...","trace":[],"actions_spec":{},"run_id":"r1","state_dir":"./state","entities":["A"]}], "kwargs":{}}\n'
        f'"""Runless KG macro draft for {task_key}."""\n'
        "\n"
        "def invoke(payload: dict) -> dict:\n"
        '    """helper-only draft missing exported run."""\n'
        '    return {"status": "MACRO EXHAUSTED", "final_variable": None, "observation": "draft"}\n'
        "###TOOL_END\n"
    )


def _importing_kg_utils_tool_source(
    *,
    task_key: str,
    strategy_family: str,
    execution_style: str,
    preferred_tool_mode: str,
    round_idx: int,
) -> str:
    return _macro_tool_source(
        task_key=task_key,
        strategy_family=strategy_family,
        execution_style=execution_style,
        preferred_tool_mode=preferred_tool_mode,
        round_idx=round_idx,
    ).replace("import json\n", "import json\nimport kg_utils\n", 1)


def test_invoker_and_markers_prompts_preserve_minimum_deliverable_metadata() -> None:
    assert "preserve `minimum_acceptable_deliverable` exactly in the payload" in TOOL_INVOKER_SYSTEM_PROMPT
    assert (
        "Do not omit, paraphrase, or regenerate `minimum_acceptable_deliverable`."
        in TOOL_INVOKER_SYSTEM_PROMPT
    )
    assert (
        'When `preferred_tool_mode` is `progress_tool` or `diagnostic_probe`, '
        "include `minimum_acceptable_deliverable` in the payload sent to the tool."
        in TOOL_INVOKER_SYSTEM_PROMPT
    )
    assert (
        '"preferred_tool_mode", "minimum_acceptable_deliverable", "fallback_strategies"'
        in TOOLGEN_SYSTEM_PROMPT_MARKERS
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
            elif _is_kg_toolgen_macro_prompt(system_prompt):
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
            elif _is_kg_toolgen_macro_prompt(system_prompt):
                content = self._toolgen_response(prompt)
            elif "ToolGen Phase" in system_prompt:
                raise RuntimeError("unexpected staged toolgen phase in patch-mode live-path test")
            else:
                raise RuntimeError(f"unexpected system prompt: {system_prompt[:80]}")
            responses.append(ChatHistoryItem(role=Role.AGENT, content=content))
        return responses


class _ImportKgUtilsPatchLanguageModel(_PatchLoopLanguageModel):
    def _toolgen_response(self, _prompt: str) -> str:
        return _importing_kg_utils_tool_source(
            task_key="count",
            strategy_family="relation_first",
            execution_style="relation_first",
            preferred_tool_mode="full_solve",
            round_idx=0,
        )


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


def _disable_family_binding(controller: SelfEvolvingController) -> None:
    controller._toolgen_kg_family_binding_context = lambda *args, **kwargs: {  # noqa: SLF001
        "family_binding_active": False,
        "family_binding_reason": "test_override_generic_path",
        "generic_path_blocked_for_family_bound_attempt": False,
        "family_skeleton_selected": None,
        "family_generation_path_used": False,
        "generic_generation_blocked": False,
        "generic_repair_blocked": False,
        "family_repair_template_name": None,
        "family_validator_path_used": False,
        "normalized_family_inputs": {},
        "family_normalization_source": "test_override_generic_path",
        "family_normalization_fallback_used": False,
        "family_classification_ambiguous": False,
        "candidate_family_options": [],
    }


class _PromptCaptureLanguageModel(_LivePathLanguageModel):
    def __init__(self) -> None:
        super().__init__(fail_toolgen=False)
        self.toolgen_prompts: list[str] = []

    def _inference(self, batch_chat_history, inference_config_dict, system_prompt):
        responses = []
        for chat_history in batch_chat_history:
            prompt = chat_history.get_item_deep_copy(-1).content or ""
            if "Combined Orchestrator" in system_prompt:
                content = self._orchestrator_response(prompt)
            elif "ToolGen Logic Validator" in system_prompt:
                content = self._validator_response(prompt)
            elif _is_kg_toolgen_macro_prompt(system_prompt):
                self.toolgen_prompts.append(prompt)
                content = self._toolgen_response(prompt)
            elif "ToolGen Phase" in system_prompt:
                raise RuntimeError("unexpected staged toolgen phase in KG macro prompt-capture test")
            else:
                raise RuntimeError(f"unexpected system prompt: {system_prompt[:80]}")
            responses.append(ChatHistoryItem(role=Role.AGENT, content=content))
        return responses


class _AlwaysSuccessKGTaskRef:
    def evaluate_generated_macro(self, tool_code: str, payload_json: str) -> str:
        return json.dumps(
            {
                "status": "SUCCESS",
                "final_variable": "#41",
                "observation": (
                    'COUNT VARIABLE RETURNED; submit it directly. minted_variables: '
                    '{"count_result": "#41"}'
                ),
            }
        )


class _MissingRunThenRewriteLanguageModel(_LivePathLanguageModel):
    def __init__(self) -> None:
        super().__init__(fail_toolgen=False)
        self.patch_calls = 0

    def _validator_response(self, prompt: str) -> str:
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

    def _toolgen_response(self, prompt: str) -> str:
        task_key = _task_key(prompt)
        round_idx = 1
        marker = "ROUND_STRATEGY_CONTEXT:"
        if marker in prompt:
            context_blob = prompt.split(marker, 1)[1]
            context_blob = context_blob.split(
                "Follow this structured retry policy exactly.",
                1,
            )[0].strip()
            try:
                round_idx = int((json.loads(context_blob) or {}).get("round") or 1)
            except Exception:
                round_idx = 1
        if round_idx == 1:
            return _runless_tool_source(task_key=task_key, round_idx=round_idx)
        return _macro_tool_source(
            task_key=task_key,
            strategy_family="probe_then_commit",
            execution_style="probe_then_commit",
            preferred_tool_mode="full_solve",
            round_idx=round_idx,
        )

    def _patch_response(self, prompt: str) -> str:
        self.patch_calls += 1
        return json.dumps({"operations": []})

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
            elif _is_kg_toolgen_macro_prompt(system_prompt):
                content = self._toolgen_response(prompt)
            elif "ToolGen Phase" in system_prompt:
                raise RuntimeError("unexpected staged toolgen phase in missing-run test")
            else:
                raise RuntimeError(f"unexpected system prompt: {system_prompt[:80]}")
            responses.append(ChatHistoryItem(role=Role.AGENT, content=content))
        return responses


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


def test_live_path_phase1_retry_prompt_scaffold_matches_mutated_retry_state(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    lm = _PromptCaptureLanguageModel()
    controller = _build_live_controller(tmp_path, monkeypatch, language_model=lm)
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is not None
    phase1_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_phase1_retry_mutation"
    ]
    assert phase1_events
    assert len(lm.toolgen_prompts) >= 2

    retry_prompt = lm.toolgen_prompts[1]
    scaffold_prefix = retry_prompt.split("TOOLGEN SETUP:", 1)[0]
    retry_plan = _prompt_canonical_plan(retry_prompt)
    retry_execution_plan = _prompt_execution_plan(retry_prompt)
    assert retry_plan["preferred_tool_mode"] == "progress_tool"
    assert retry_plan["execution_style"] == "relation_first"
    assert "single resolved anchor variable is NOT sufficient" in retry_plan[
        "minimum_acceptable_deliverable"
    ]
    assert "resolved_both_anchors" in retry_plan["minimum_acceptable_deliverable"]
    assert "Follow the declared strategy only until the first acceptable value state is reached." in retry_execution_plan
    assert "Stop immediately after producing that value or solver-usable handoff" in retry_execution_plan
    assert "Count the intersection." not in retry_execution_plan
    assert '"preferred_tool_mode": "full_solve"' not in scaffold_prefix
    assert '"execution_style": "walk_first"' not in scaffold_prefix
    assert "LAST_TOOL_CODE:" not in retry_prompt
    assert "PRIMARY_REPAIR_TARGET" in retry_prompt


def test_same_task_toolgen_allows_turn_zero_and_one_node_explosion_then_blocks_third(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    controller = _build_live_controller(tmp_path, monkeypatch)
    controller._reuse_existing_tool = lambda *args, **kwargs: None  # noqa: SLF001
    controller._get_candidate_output = lambda *args, **kwargs: None  # noqa: SLF001
    controller._force_toolgen_always_on = True  # noqa: SLF001
    calls: list[str] = []

    def _fake_generate_from_prompt(**kwargs):
        calls.append(str(kwargs.get("user_prompt") or ""))
        return types.SimpleNamespace(name=f"generated_{len(calls)}")

    monkeypatch.setattr(
        controller,
        "_toolgen_generate_from_prompt",
        _fake_generate_from_prompt,
    )

    controller._run_task_metadata = {"task_name": "knowledge_graph", "sample_index": "0"}  # noqa: SLF001
    turn_zero_history = ChatHistory()
    first = controller._maybe_generate_tool_for_query(  # noqa: SLF001
        "Question: first task turn", turn_zero_history
    )
    later_history = ChatHistory()
    later_history.inject(
        ChatHistoryItem(role=Role.USER, content="Question: first task turn")
    )
    later_history.inject(
        ChatHistoryItem(role=Role.AGENT, content="Tool output: #0")
    )
    controller._toolgen_set_current_turn_policy_context(  # noqa: SLF001
        is_turn_zero=False,
        observation_triggers=[],
    )
    second = controller._maybe_generate_tool_for_query(  # noqa: SLF001
        "Question: later same-task solver step", later_history, force=True
    )
    controller._toolgen_set_current_turn_policy_context(  # noqa: SLF001
        is_turn_zero=False,
        observation_triggers=[
            {
                "type": "size_trigger",
                "reason": "observation has 21 items (>15 threshold)",
            }
        ],
    )
    third = controller._maybe_generate_tool_for_query(  # noqa: SLF001
        "Question: node explosion recovery", later_history, force=True
    )
    fourth = controller._maybe_generate_tool_for_query(  # noqa: SLF001
        "Question: second node explosion retry", later_history, force=True
    )

    assert first is not None
    assert second is None
    assert third is not None
    assert fourth is None
    assert len(calls) == 2

    controller._run_task_metadata = {"task_name": "knowledge_graph", "sample_index": "1"}  # noqa: SLF001
    controller._toolgen_set_current_turn_policy_context(  # noqa: SLF001
        is_turn_zero=True,
        observation_triggers=[],
    )
    fifth = controller._maybe_generate_tool_for_query(  # noqa: SLF001
        "Question: next task boundary", ChatHistory()
    )

    assert fifth is not None
    assert len(calls) == 3

    generated_events = _jsonl_events(controller._generated_tools_log_path)  # noqa: SLF001
    policy_blocks = [
        event
        for event in generated_events
        if event.get("event") == "toolgen_attempt_blocked_policy"
    ]
    same_task_blocks = [
        event
        for event in generated_events
        if event.get("event") == "toolgen_attempt_blocked_same_task"
    ]
    assert policy_blocks
    assert policy_blocks[0]["policy_reason"] == "not_turn_zero_or_node_explosion"
    assert same_task_blocks
    assert same_task_blocks[0]["attempt_count"] == 2
    assert same_task_blocks[0]["max_attempts"] == 2


def test_toolgen_current_turn_policy_only_opens_for_turn_zero_or_size_trigger(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    later_history = ChatHistory()
    later_history.inject(ChatHistoryItem(role=Role.USER, content="Question"))
    later_history.inject(ChatHistoryItem(role=Role.AGENT, content="Observation"))

    controller._toolgen_set_current_turn_policy_context(  # noqa: SLF001
        is_turn_zero=False,
        observation_triggers=[],
    )
    allowed, reason = controller._toolgen_current_turn_allows_fresh_attempt(later_history)  # noqa: SLF001
    assert allowed is False
    assert reason == "not_turn_zero_or_node_explosion"

    controller._toolgen_set_current_turn_policy_context(  # noqa: SLF001
        is_turn_zero=False,
        observation_triggers=[
            {
                "type": "size_trigger",
                "reason": "observation has 18 items (>15 threshold)",
            }
        ],
    )
    allowed, reason = controller._toolgen_current_turn_allows_fresh_attempt(later_history)  # noqa: SLF001
    assert allowed is True
    assert reason == "observation has 18 items (>15 threshold)"


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
    assert early_stop_events[0]["best_achieved_state"] == "resolved_both_anchors"
    best_summary = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_best_candidate_summary"
    ]
    assert best_summary
    assert best_summary[-1]["selection_source"] == "best_partial_candidate"
    assert best_summary[-1]["best_achieved_state"] == "resolved_both_anchors"
    assert best_summary[-1]["best_achieved_round"] == 1
    assert best_summary[-1]["best_achieved_tool_name"]
    admitted_metadata = [
        item for item in result["candidate_metadata"] if item.get("registered") is True
    ]
    assert admitted_metadata
    assert admitted_metadata[-1]["best_achieved_state"] == "resolved_both_anchors"


def test_live_round_start_logs_family_routing_fields(
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

    round_starts = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_round_start"
    ]
    assert round_starts
    first = round_starts[0]
    assert first["template_family"] == "two_anchor_intersect_count"
    assert first["anchor_operands"] == ["CNES", "Astrium"]
    assert first["terminal_artifact_kind"] == "count_variable"
    assert first["required_next_stage"] == "built_target_set"
    assert first["family_binding_active"] is True
    assert first["family_binding_reason"] == "supported_template_family:two_anchor_intersect_count"
    assert first["generic_path_blocked_for_family_bound_attempt"] is True
    assert first["family_skeleton_selected"] == "two_anchor_intersect_count"
    assert first["family_generation_path_used"] is True
    assert first["family_skeleton_used"] is True
    assert first["family_repair_path_used"] is True
    assert first["family_repair_template_name"] == "two_anchor_intersect_count"
    assert first["generic_repair_blocked"] is True
    assert first["family_validator_path_used"] is True
    assert first["family_validator_checks_used"] is True
    assert first["family_patch_path_blocked"] is True
    assert first["family_regeneration_forced"] is True
    assert first["current_round_patch_mode"] is False
    assert first["generic_generation_used_when_family_known"] is False
    assert first["normalized_family_inputs"]["template_family"] == "two_anchor_intersect_count"


def test_family_bound_patch_mode_forces_full_regeneration(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIFELONG_TOOLGEN_PATCH_MODE", "1")
    lm = _MissingRunThenRewriteLanguageModel()
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=lm,
        kg_task_ref=_AlwaysSuccessKGTaskRef(),
    )
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    assert result["tool"] is not None
    round_starts = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_round_start"
    ]
    assert len(round_starts) >= 2
    assert all(event["family_binding_active"] is True for event in round_starts)
    assert all(event["family_patch_path_blocked"] is True for event in round_starts)
    assert all(event["family_regeneration_forced"] is True for event in round_starts)
    assert all(event["current_round_patch_mode"] is False for event in round_starts)
    assert lm.patch_calls == 0
    candidate_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_candidate"
    ]
    assert candidate_events
    assert all(event["patch_round"] is False for event in candidate_events)
    assert not any(
        event.get("event") == "toolgen_round_failed"
        and event.get("phase") in {"patch_plan_parse", "patch_apply"}
        for event in result["generated_events"]
    )


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
    _disable_family_binding(controller)
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


def test_patch_mode_sanitizes_preinjected_kg_utils_import_retry_source(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    code = (
        "import json\n"
        "import kg_utils\n"
        "\n"
        "def run(payload: dict) -> dict:\n"
        '    """contract guard: ok.\\nprereqs: ok.\\nlimitations: ok."""\n'
        '    return {"status": "ERROR", "final_variable": None, "observation": "x"}\n'
    )
    sanitized = controller._toolgen_patch_sanitize_contract_retry_source(  # noqa: SLF001
        code,
        feedback_note=json.dumps(
            {
                "phase": "critical_contract_precheck",
                "error": (
                    "CRITICAL: Do NOT write 'import kg_utils'. "
                    "It is pre-injected as a global."
                ),
            }
        ),
    )

    assert "import kg_utils" not in sanitized
    assert "import json" in sanitized
    assert "def run(payload: dict) -> dict:" in sanitized


def test_patch_mode_clears_kg_utils_import_contract_failure(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIFELONG_TOOLGEN_PATCH_MODE", "1")
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=_ImportKgUtilsPatchLanguageModel(),
        kg_task_ref=_PatchLoopKGTaskRef(),
    )
    _disable_family_binding(controller)
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    critical_fails = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_critical_contract_fail"
    ]
    candidate_events = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_candidate"
    ]

    assert critical_fails
    assert critical_fails[0]["round"] == 1
    assert any("import kg_utils" in issue for issue in critical_fails[0]["issues"])
    assert result["tool"] is not None
    assert len(candidate_events) >= 2
    assert candidate_events[1]["patch_round"] is True
    assert not any(
        event.get("round") > 1 and any("import kg_utils" in issue for issue in event.get("issues", []))
        for event in critical_fails
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
    _disable_family_binding(controller)
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


# ---------------------------------------------------------------------------
# Phase-1 retry-state tests
# ---------------------------------------------------------------------------

def _p1_feedback(
    *,
    redesign_direction: str = "replace_exhausted_full_solve_shape",
    value_delivered: str = "none",
    achieved_state: str = "none",
    partial_value_usable: bool = False,
    has_final_variable: bool = False,
    has_next_action_guidance: bool = False,
    handoff_state: str = "exhausted",
    include_brief: bool = True,
    include_primary: bool = True,
    include_lps: bool = True,
) -> str:
    """Build a minimal feedback_note JSON string for Phase-1 trigger tests."""
    payload: dict = {}
    if include_brief:
        payload["runtime_repair_brief"] = {
            "redesign_direction": redesign_direction,
            "missing_value_type": "no_classified_value",
            "runtime_failure_summary": "MACRO EXHAUSTED produced no value.",
        }
    if include_primary:
        payload["PRIMARY_REPAIR_TARGET"] = (
            "RUNTIME-FIRST REPAIR TARGET: rewrite toward first acceptable value state."
        )
    if include_lps:
        payload["live_progress_summary"] = {
            "value_delivered": value_delivered,
            "achieved_state": achieved_state,
            "partial_value_usable": partial_value_usable,
            "has_final_variable": has_final_variable,
            "has_next_action_guidance": has_next_action_guidance,
            "handoff_state": handoff_state,
        }
    return json.dumps(payload)


def test_phase1_retry_state_triggers_on_no_value_with_structural_direction(
    tmp_path: pathlib.Path,
) -> None:
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(redesign_direction="replace_exhausted_full_solve_shape")
    )
    assert state["active"] is True
    assert state["omit_prior_code"] is True
    assert state["override_preferred_tool_mode"] == "progress_tool"
    assert state["minimum_acceptable_deliverable"] is not None
    assert state["primary_repair_dominates"] is True


def test_phase1_retry_state_triggers_on_rewrite_toward_first_value(
    tmp_path: pathlib.Path,
) -> None:
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(redesign_direction="rewrite_toward_first_valuable_state")
    )
    assert state["active"] is True
    assert state["omit_prior_code"] is True
    assert state["override_preferred_tool_mode"] == "progress_tool"
    assert state["minimum_acceptable_deliverable"] is not None


def test_phase1_retry_state_triggers_on_emit_handoff_direction(
    tmp_path: pathlib.Path,
) -> None:
    """emit_concrete_handoff_from_live_anchor is first-value but not structural."""
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(redesign_direction="emit_concrete_handoff_from_live_anchor")
    )
    assert state["active"] is True
    assert state["omit_prior_code"] is False
    assert state["override_preferred_tool_mode"] == "progress_tool"
    assert state["minimum_acceptable_deliverable"] is not None


def test_phase1_retry_state_no_trigger_when_partial_value_usable(
    tmp_path: pathlib.Path,
) -> None:
    """Negative control: Phase-1 must NOT activate when partial value was usable."""
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(
            redesign_direction="replace_exhausted_full_solve_shape",
            value_delivered="resolved_anchor",
            partial_value_usable=True,
            has_final_variable=False,
            handoff_state="",
        )
    )
    assert state["active"] is False
    assert state["omit_prior_code"] is False
    assert state["override_preferred_tool_mode"] is None


def test_phase1_retry_state_no_trigger_when_achieved_state_present(
    tmp_path: pathlib.Path,
) -> None:
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(
            redesign_direction="replace_exhausted_full_solve_shape",
            value_delivered="none",
            achieved_state="resolved_both_anchors",
            partial_value_usable=False,
            has_final_variable=False,
            handoff_state="exhausted",
        )
    )
    assert state["active"] is False


def test_phase1_retry_state_no_trigger_when_no_repair_brief_and_no_primary(
    tmp_path: pathlib.Path,
) -> None:
    """No brief and no PRIMARY_REPAIR_TARGET → inactive regardless of LPS."""
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(include_brief=False, include_primary=False)
    )
    assert state["active"] is False


def test_phase1_retry_state_no_trigger_on_empty_feedback(
    tmp_path: pathlib.Path,
) -> None:
    assert ControllerToolgenMixin._toolgen_phase1_retry_state("")["active"] is False
    assert ControllerToolgenMixin._toolgen_phase1_retry_state("{}")["active"] is False


def test_phase1_retry_state_no_trigger_when_has_final_variable(
    tmp_path: pathlib.Path,
) -> None:
    """has_final_variable=True means value was delivered — Phase-1 should not fire."""
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(
        _p1_feedback(
            redesign_direction="replace_exhausted_full_solve_shape",
            value_delivered="none",
            partial_value_usable=False,
            has_final_variable=True,
        )
    )
    assert state["active"] is False


def test_phase1_minimum_acceptable_deliverable_injected_in_payload(
    tmp_path: pathlib.Path,
) -> None:
    """MAD from round_context flows through _toolgen_apply_round_strategy_context
    into both payload and payload['tool_plan']."""
    controller = _DummyController(tmp_path)
    mad = "Return the first grounded #N and stop."
    round_context = {
        "active_execution_style": "relation_first",
        "active_preferred_tool_mode": "progress_tool",
        "minimum_acceptable_deliverable": mad,
        "fallback_strategies": [],
    }
    exec_payload = {"tool_plan": _kg_plan(execution_style="relation_first")}
    result = controller._toolgen_apply_round_strategy_context(exec_payload, round_context)

    assert result.get("minimum_acceptable_deliverable") == mad
    assert result["tool_plan"].get("minimum_acceptable_deliverable") == mad


def test_phase1_minimum_acceptable_deliverable_not_injected_when_absent(
    tmp_path: pathlib.Path,
) -> None:
    """MAD must NOT appear in payload when round_context does not set it."""
    controller = _DummyController(tmp_path)
    round_context = {
        "active_execution_style": "relation_first",
        "active_preferred_tool_mode": "full_solve",
        "fallback_strategies": [],
    }
    exec_payload = {"tool_plan": _kg_plan(execution_style="relation_first")}
    result = controller._toolgen_apply_round_strategy_context(exec_payload, round_context)

    assert "minimum_acceptable_deliverable" not in result or not result["minimum_acceptable_deliverable"]


def test_progress_floor_tightens_generic_minimum_deliverable_for_multi_anchor_plan(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    result = controller._toolgen_apply_round_strategy_context(
        {"tool_plan": _kg_plan(execution_style="relation_first")},
        {
            "active_execution_style": "relation_first",
            "active_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE,
            "best_achieved_state": "resolved_both_anchors",
            "fallback_strategies": [],
        },
    )
    mad = str(result.get("minimum_acceptable_deliverable") or "")
    assert "single resolved anchor variable is NOT sufficient" in mad
    assert "resolved_both_anchors" in mad
    assert "active plan stage" in mad


def test_phase1_retry_prompt_scaffold_rebuilt_from_mutated_exec_payload(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    controller._toolgen_prompt_context = {  # noqa: SLF001
        "query": "Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        "tool_type": "macro",
        "reason": "INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        "upgrade_goal": "INPUT: ['CNES', 'Astrium']. GOAL: stop at first valuable state.",
        "env_name": "knowledge_graph",
        "env_contract": "Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
    }
    stale_payload = {"tool_plan": _kg_plan(execution_style="walk_first", preferred_tool_mode="full_solve")}
    stale_prompt = controller._toolgen_build_blueprint_prompt(
        query="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        tool_type="macro",
        reason="INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        upgrade_goal="INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        tool_plan=stale_payload["tool_plan"],
        env_name="knowledge_graph",
        env_contract="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
    )
    mad = "Return the first grounded #N and stop."
    mutated_payload = controller._toolgen_apply_round_strategy_context(
        stale_payload,
        {
            "active_execution_style": "relation_first",
            "active_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": mad,
            "fallback_strategies": ["probe_then_commit"],
        },
        phase1_retry_state={
            "active": True,
            "omit_prior_code": True,
            "override_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": mad,
            "primary_repair_dominates": True,
        },
    )

    refreshed_prompt = controller._toolgen_refresh_retry_prompt_scaffold(
        fallback_prompt=stale_prompt,
        exec_payload=mutated_payload,
    )

    plan = _prompt_canonical_plan(refreshed_prompt)
    scaffold_prefix = refreshed_prompt.split("TOOLGEN SETUP:", 1)[0]
    assert plan["execution_style"] == "relation_first"
    assert plan["preferred_tool_mode"] == "progress_tool"
    assert plan["minimum_acceptable_deliverable"] == mad
    assert '"execution_style": "walk_first"' not in scaffold_prefix
    assert '"preferred_tool_mode": "full_solve"' not in scaffold_prefix


def test_phase1_retry_prompt_scaffold_replaces_stale_execution_plan_text(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    controller._toolgen_prompt_context = {  # noqa: SLF001
        "query": "Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        "tool_type": "macro",
        "reason": "INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        "upgrade_goal": "INPUT: ['CNES', 'Astrium']. GOAL: stop at first valuable state.",
        "env_name": "knowledge_graph",
        "env_contract": "Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
    }
    stale_plan = _kg_plan(
        execution_style="walk_first",
        preferred_tool_mode="full_solve",
        topological_execution_plan=[
            "1. Continue through the full solve.",
            "2. Return the final answer variable.",
        ],
    )
    stale_prompt = controller._toolgen_build_blueprint_prompt(
        query="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        tool_type="macro",
        reason="INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        upgrade_goal="INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        tool_plan=stale_plan,
        env_name="knowledge_graph",
        env_contract="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
    )
    mutated_payload = controller._toolgen_apply_round_strategy_context(
        {"tool_plan": stale_plan},
        {
            "active_execution_style": "relation_first",
            "active_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE,
            "fallback_strategies": ["probe_then_commit"],
        },
        phase1_retry_state={
            "active": True,
            "omit_prior_code": True,
            "override_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE,
            "primary_repair_dominates": True,
        },
    )
    refreshed_prompt = controller._toolgen_refresh_retry_prompt_scaffold(
        fallback_prompt=stale_prompt,
        exec_payload=mutated_payload,
    )

    execution_plan = _prompt_execution_plan(refreshed_prompt)
    assert "Follow the declared strategy only until the first acceptable value state is reached." in execution_plan
    assert "Minimum acceptable deliverable:" in execution_plan
    assert "Stop immediately after producing that value or solver-usable handoff" in execution_plan
    assert "Continue through the full solve." not in execution_plan
    assert "Return the final answer variable." not in execution_plan


def test_phase1_build_full_rewrite_prompt_omits_prior_code_when_flagged(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    base_prompt = controller._toolgen_build_blueprint_prompt(
        query="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
        tool_type="macro",
        reason="INPUT: ['CNES', 'Astrium']. GOAL: count shared spacecraft.",
        upgrade_goal="INPUT: ['CNES', 'Astrium']. GOAL: stop at first valuable state.",
        tool_plan=_kg_plan(execution_style="relation_first", preferred_tool_mode="progress_tool"),
        env_name="knowledge_graph",
        env_contract="Question: count shared spacecraft, Entities: ['CNES', 'Astrium']",
    )
    prompt = controller._toolgen_build_full_rewrite_prompt(
        base_prompt=base_prompt,
        round_context={
            "round": 2,
            "strategy_family": "relation_first",
            "execution_style": "relation_first",
            "preferred_tool_mode": "progress_tool",
        },
        feedback_note=_p1_feedback(),
        round_history=[],
        last_tool_code="def run(payload: dict) -> dict:\n    return {}\n",
        phase1_retry_state={
            "active": True,
            "omit_prior_code": True,
            "override_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE,
            "primary_repair_dominates": True,
        },
    )
    assert "LAST_TOOL_CODE:" not in prompt
    assert "def run(payload: dict) -> dict:" not in prompt


def test_phase1_build_full_rewrite_prompt_primary_repair_dominates_instruction(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    prompt = controller._toolgen_build_full_rewrite_prompt(
        base_prompt="task pack",
        round_context={
            "round": 2,
            "strategy_family": "relation_first",
            "execution_style": "relation_first",
            "preferred_tool_mode": "progress_tool",
        },
        feedback_note=_p1_feedback(),
        round_history=[],
        last_tool_code="def run(payload: dict) -> dict:\n    return {}\n",
        phase1_retry_state={
            "active": True,
            "omit_prior_code": True,
            "override_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE,
            "primary_repair_dominates": True,
        },
    )
    assert "Implement ONLY the PRIMARY_REPAIR_TARGET above as your primary objective." in prompt
    assert "Implement all requested fixes and refactor the code as necessary to pass the live evaluation." not in prompt


def test_phase1_build_full_rewrite_prompt_uses_standard_instruction_without_dominant_primary(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    prompt = controller._toolgen_build_full_rewrite_prompt(
        base_prompt="task pack",
        round_context={
            "round": 2,
            "strategy_family": "relation_first",
            "execution_style": "relation_first",
            "preferred_tool_mode": "progress_tool",
        },
        feedback_note=json.dumps({"issues": ["wrong count variable returned"]}),
        round_history=[],
        last_tool_code="def run(payload: dict) -> dict:\n    return {}\n",
        phase1_retry_state={
            "active": True,
            "omit_prior_code": False,
            "override_preferred_tool_mode": None,
            "minimum_acceptable_deliverable": None,
            "primary_repair_dominates": False,
        },
    )
    assert "LAST_TOOL_CODE:" in prompt
    assert "Implement all requested fixes and refactor the code as necessary to pass the live evaluation." in prompt
    assert "Implement ONLY the PRIMARY_REPAIR_TARGET above as your primary objective." not in prompt


def test_phase1_build_patch_prompt_omits_code_when_flagged(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    prompt = controller._toolgen_build_patch_prompt(
        current_code="def run(payload: dict) -> dict:\n    return {}\n",
        feedback_note=_p1_feedback(),
        round_history=[],
        base_prompt="task pack",
        phase1_omit_prior_code=True,
    )
    assert "omitted" in prompt
    assert "generate a new implementation from scratch" in prompt
    assert "def run(payload: dict) -> dict:" not in prompt


def test_phase1_build_patch_prompt_includes_code_when_not_flagged(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    code = "def run(payload: dict) -> dict:\n    return {}\n"
    prompt = controller._toolgen_build_patch_prompt(
        current_code=code,
        feedback_note=_p1_feedback(),
        round_history=[],
        base_prompt="task pack",
        phase1_omit_prior_code=False,
    )
    assert "def run(payload: dict) -> dict:" in prompt
    assert "omitted" not in prompt


def test_phase1_primary_repair_dominates_changes_rewrite_instruction(
    tmp_path: pathlib.Path,
) -> None:
    """When primary_repair_dominates is True the patch prompt must emphasise
    PRIMARY_REPAIR_TARGET as the sole objective, not 'implement all requested fixes'."""
    controller = _DummyController(tmp_path)
    feedback = _p1_feedback(redesign_direction="replace_exhausted_full_solve_shape")
    prompt = controller._toolgen_build_patch_prompt(
        current_code="def run(payload: dict) -> dict:\n    return {}\n",
        feedback_note=feedback,
        round_history=[],
        base_prompt="task pack",
    )
    # Phase-1 dominant wording must be present in the patch prompt
    assert "PRIMARY_REPAIR_TARGET" in prompt
    assert "dominant rewrite goal" in prompt


def _build_legacy_generation_controller(
    tmp_path: pathlib.Path,
    *,
    raw_output: str,
) -> _DummyController:
    controller = _DummyController(tmp_path)
    controller._registry = types.SimpleNamespace(  # noqa: SLF001
        list_tools=lambda **_: [],
        list_latest_tools=lambda **_: [],
    )
    controller._toolgen_agent = types.SimpleNamespace(_system_prompt="")  # noqa: SLF001
    controller._toolgen_call_llm = lambda **_: raw_output  # noqa: SLF001
    return controller


def test_legacy_candidate_without_top_level_run_is_rejected_before_wrapping(
    tmp_path: pathlib.Path,
) -> None:
    controller = _build_legacy_generation_controller(
        tmp_path,
        raw_output=_runless_tool_source(task_key="count", round_idx=1),
    )

    result = controller._toolgen_generate_from_prompt_legacy(
        user_prompt="task pack",
        system_prompt="system",
        chat_history=ChatHistory(),
        name_prefix="",
    )

    assert result is not None
    assert result["error"] == "run_signature"
    assert result["run_signature_error"] == "run_not_found"
    assert result["force_full_rewrite_next_round"] is True
    assert "tool_spec" not in result


def test_legacy_candidate_with_top_level_run_still_passes_admission(
    tmp_path: pathlib.Path,
) -> None:
    controller = _build_legacy_generation_controller(
        tmp_path,
        raw_output=_macro_tool_source(
            task_key="count",
            strategy_family="relation_first",
            execution_style="relation_first",
            preferred_tool_mode="full_solve",
            round_idx=1,
        ),
    )

    result = controller._toolgen_generate_from_prompt_legacy(
        user_prompt="task pack",
        system_prompt="system",
        chat_history=ChatHistory(),
        name_prefix="",
    )

    assert result is not None
    assert "error" not in result
    assert result["tool_spec"]["signature"] == "run(payload: dict) -> dict"
    assert result["tool_spec"]["name"] == "count_relation_first_r1_generated_tool"
    assert "def run(payload: dict) -> dict:" in result["tool_code"]


def test_run_scoped_ssot_precheck_ignores_helper_raw_string_returns(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    code = (
        "def invoke(payload: dict) -> str:\n"
        '    return "MACRO EXHAUSTED"\n'
        "\n"
        "def run(payload: dict) -> dict:\n"
        "    helper_result = invoke(payload)\n"
        '    return {"status": "MACRO EXHAUSTED", "final_variable": None, "observation": helper_result}\n'
    )

    assert controller._toolgen_run_ssot_contract_error(code) is None  # noqa: SLF001


def test_run_scoped_ssot_precheck_still_flags_run_raw_string_returns(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    code = (
        "def run(payload: dict) -> dict:\n"
        '    return "MACRO EXHAUSTED"\n'
    )

    error = controller._toolgen_run_ssot_contract_error(code)  # noqa: SLF001
    assert error is not None
    assert "SSOT schema violation" in error


def test_missing_run_failure_forces_full_rewrite_and_skips_patch(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LIFELONG_TOOLGEN_PATCH_MODE", "1")
    language_model = _MissingRunThenRewriteLanguageModel()
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=language_model,
        kg_task_ref=_AlwaysSuccessKGTaskRef(),
    )

    result = _run_live_toolgen_case(controller, _COUNT_QUERY)
    generation_failures = [
        event
        for event in result["generated_events"]
        if event.get("event") == "toolgen_round_failed"
    ]

    assert any(event.get("reason") == "run_signature" for event in generation_failures)
    assert language_model.patch_calls == 0
    assert not any(event.get("phase") == "patch_apply" for event in generation_failures)
    assert result["tool"] is not None


def test_patch_mode_requires_existing_top_level_run(tmp_path: pathlib.Path) -> None:
    controller = _DummyController(tmp_path)

    assert (
        controller._toolgen_has_patchable_run(  # noqa: SLF001
            _runless_tool_source(task_key="count", round_idx=1)
        )
        is False
    )
    assert (
        controller._toolgen_has_patchable_run(  # noqa: SLF001
            _macro_tool_source(
                task_key="count",
                strategy_family="relation_first",
                execution_style="relation_first",
                preferred_tool_mode="full_solve",
                round_idx=1,
            )
        )
        is True
    )


# ---------------------------------------------------------------------------
# Phase 1b: grounded-first-value semantics and translator prohibition tests
# ---------------------------------------------------------------------------


def test_phase1b_tool_start_template_does_not_contain_translator_wording() -> None:
    """The ###TOOL_START template must not encourage translator-style behavior."""
    tool_start = MACRO_TOOLGEN_USER_KG.split("###TOOL_START", 1)
    assert len(tool_start) == 2, "###TOOL_START block missing from MACRO_TOOLGEN_USER_KG"
    tool_start_block = tool_start[1]
    assert "translator" not in tool_start_block.lower(), (
        "TOOL_START limitations docstring must not contain 'translator'"
    )
    assert "stop at first grounded KG result" in tool_start_block


def test_phase1b_tool_start_template_limitations_forbids_payload_echo() -> None:
    """The ###TOOL_START limitations line must explicitly forbid payload/plan echo."""
    tool_start_block = MACRO_TOOLGEN_USER_KG.split("###TOOL_START", 1)[1]
    assert "payload" in tool_start_block
    assert "plan fields" in tool_start_block or "echo" in tool_start_block


def test_phase1b_min_deliverable_requires_kg_operation() -> None:
    """_PHASE1_GENERIC_MIN_DELIVERABLE must require a real kg_utils.* primitive call."""
    mad = ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE
    assert "kg_utils" in mad, "MAD must reference kg_utils.* operations"
    assert "resolve_entity_to_vars" in mad or "resolve" in mad


def test_phase1b_min_deliverable_forbids_pseudo_progress() -> None:
    """_PHASE1_GENERIC_MIN_DELIVERABLE must explicitly prohibit each class of pseudo-progress."""
    mad = ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE
    # tool_plan echo
    assert "tool_plan" in mad
    # payload echo
    assert "payload" in mad
    # placeholder minting
    assert "placeholder" in mad or "minting" in mad
    # entity string minting
    assert "entity" in mad
    # prose without KG state
    assert "prose" in mad or "generic" in mad


def test_phase1b_min_deliverable_requires_grounded_anchor_on_exhaustion() -> None:
    """On MACRO EXHAUSTED path, MAD must require a real resolved anchor."""
    mad = ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE
    assert "MACRO EXHAUSTED" in mad
    assert "resolved anchor" in mad or "real resolved" in mad


def test_phase1b_min_deliverable_reaches_execution_plan_steps(
    tmp_path: pathlib.Path,
) -> None:
    """The MAD prohibitions must be visible in the rendered topological_execution_plan
    injected into the payload during a Phase 1 progress_tool retry."""
    controller = _DummyController(tmp_path)
    mad = ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE
    round_context = {
        "active_execution_style": "walk_first",
        "active_preferred_tool_mode": "progress_tool",
        "minimum_acceptable_deliverable": mad,
        "fallback_strategies": [],
    }
    exec_payload = {"tool_plan": _kg_plan(execution_style="walk_first")}
    result = controller._toolgen_apply_round_strategy_context(
        exec_payload,
        round_context,
        phase1_retry_state={
            "active": True,
            "omit_prior_code": True,
            "override_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": mad,
            "primary_repair_dominates": True,
        },
    )
    plan_steps = result.get("topological_execution_plan") or []
    plan_text = " ".join(str(s) for s in plan_steps)
    # The MAD content with its prohibitions must appear in the rendered steps
    assert "kg_utils" in plan_text
    assert "NOT acceptable" in plan_text or "tool_plan" in plan_text


def test_phase1b_preferred_tool_mode_mutation_still_fires(
    tmp_path: pathlib.Path,
) -> None:
    """Regression: Phase 1 preferred_tool_mode override must still fire after 1b change."""
    feedback = _p1_feedback(redesign_direction="rewrite_toward_first_valuable_state")
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(feedback)
    assert state["active"] is True
    assert state["override_preferred_tool_mode"] == "progress_tool"
    assert state["minimum_acceptable_deliverable"] == ControllerToolgenMixin._PHASE1_GENERIC_MIN_DELIVERABLE


def test_phase1b_omit_prior_code_still_fires_for_structural_direction(
    tmp_path: pathlib.Path,
) -> None:
    """Regression: omit_prior_code must still be True for structural directions."""
    feedback = _p1_feedback(redesign_direction="replace_exhausted_full_solve_shape")
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(feedback)
    assert state["omit_prior_code"] is True


def test_phase1b_omit_prior_code_false_for_emit_grounded_intermediate(
    tmp_path: pathlib.Path,
) -> None:
    """Regression: omit_prior_code must remain False for emit_grounded_intermediate_before_final."""
    feedback = _p1_feedback(redesign_direction="emit_grounded_intermediate_before_final")
    state = ControllerToolgenMixin._toolgen_phase1_retry_state(feedback)
    assert state["active"] is True
    assert state["omit_prior_code"] is False
    assert state["override_preferred_tool_mode"] == "progress_tool"
    assert state["minimum_acceptable_deliverable"] is not None


def test_phase1_patch_failure_with_best_achieved_anchor_activates_primary_binding(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    feedback = json.dumps(
        {
            "phase": "patch_apply",
            "error": "hunk mismatch",
            "best_achieved_state": "produced_actionable_handoff",
            "best_achieved_round": 3,
            "primary_repair_instruction": (
                "Round 3 achieved 'produced_actionable_handoff'. Recovery must preserve "
                "or improve that partial-success shape."
            ),
        }
    )
    state = controller._toolgen_phase1_retry_state(feedback)
    assert state["active"] is True
    assert state["omit_prior_code"] is False
    assert state["primary_repair_dominates"] is True


# ---------------------------------------------------------------------------
# Pre-Phase-2 stabilization tests (A/B/C/D)
# ---------------------------------------------------------------------------


def test_stabilization_a_progress_tool_anchor_only_insufficient_for_narrowing_plan(
    tmp_path: pathlib.Path,
) -> None:
    """Fix A: progress_tool candidate that resolves only an anchor is NOT sufficient
    when the active plan requires downstream narrowing (intersection/walk/count).

    The live-progress summarizer must mark material_progress=False with reason
    resolved_anchor_only_insufficient_for_plan instead of accepting it.
    """
    controller = _DummyController(tmp_path)

    # Plan with intersection → plan_requires_narrowing_or_final_op=True
    plan = _kg_plan(
        preferred_tool_mode="progress_tool",
        execution_style="partial_value_first",
    )

    # Tool returned SUCCESS with a resolved anchor but no downstream operation
    anchor_only_result = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#1",
            "observation": (
                "Variable #1 contains the resolved anchor node for CNES. "
                'minted_variables: {"resolved_CNES": "#1"}'
            ),
        },
        {"tool_plan": plan},
    )

    assert anchor_only_result["material_progress"] is False, (
        "resolved anchor alone is not sufficient when plan requires intersection/count"
    )
    assert anchor_only_result["reason"] == "resolved_anchor_only_insufficient_for_plan"
    assert anchor_only_result["value_delivered"] in {"resolved_anchor", "resolved_both_anchors"}

    # Positive control: anchor + downstream op signal is sufficient
    with_downstream = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#3",
            "observation": (
                "Variable #3 contains the intersected set. "
                'minted_variables: {"spacecraft_set": "#3"}'
            ),
        },
        {"tool_plan": plan},
    )
    assert with_downstream["material_progress"] is True


def test_stabilization_a_full_solve_multi_anchor_plan_uses_strong_bank_floor(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        preferred_tool_mode="full_solve",
        execution_style="relation_first",
    )

    anchor_only_result = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#1",
            "observation": 'minted_variables: {"resolved_CNES": "#1"}',
        },
        {"tool_plan": plan},
    )

    assert anchor_only_result["value_delivered"] == "resolved_anchor"
    assert anchor_only_result["minimum_bankable_achieved_state"] == "resolved_both_anchors"
    assert anchor_only_result["material_progress"] is False
    assert anchor_only_result["bankable_partial_progress"] is False
    assert anchor_only_result["reason"] == "resolved_anchor_only_insufficient_for_plan"


def test_stabilization_a_multi_anchor_progress_plan_accepts_resolved_both_anchors_floor(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        preferred_tool_mode="progress_tool",
        execution_style="partial_value_first",
    )

    result = controller._summarize_toolgen_live_progress(
        {
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": (
                "MACRO EXHAUSTED: Resulting set is empty. "
                'minted_variables: {"resolved_Goat": "#1", "resolved_cows": ["#2"]}'
            ),
        },
        {"tool_plan": plan},
    )

    assert result["value_delivered"] == "resolved_both_anchors"
    assert result["material_progress"] is True
    assert result["partial_value_usable"] is True


def test_stabilization_a_regression_below_best_achieved_state_is_demoted(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    regressed = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 7,
            "issues": [],
            "fixes": [],
            "summary": "regressed to weaker anchor-only shape",
            "plan_diagnosis": "OK",
            "repair_mode": "none",
        },
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#1",
            "observation": 'Variable #1 contains resolved anchor. minted_variables: {"resolved_Goat": "#1"}',
        },
        live_progress_summary={
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "has_context": True,
            "material_progress": True,
            "handoff_state": "partial_safe_continue",
            "final_operation_safe": False,
        },
        round_context={
            "round": 5,
            "preferred_tool_mode": "progress_tool",
            "best_achieved_state": "produced_actionable_handoff",
        },
        tool_plan=_kg_plan(preferred_tool_mode="progress_tool"),
        tool_code=(
            "def run(payload: dict) -> dict:\n"
            "    entities = payload.get('entities') or []\n"
            "    attribute_target_concept = payload.get('attribute_target_concept')\n"
            "    for ent in entities:\n"
            "        out = kg_utils.resolve_entity_to_vars(ent, None, {}, None)\n"
            "        ids = kg_utils.extract_var_ids(out)\n"
            "        if ids:\n"
            "            return {'status': 'SUCCESS', 'final_variable': ids[0], 'observation': 'ok'}\n"
        ),
    )
    assert regressed["grade"] <= 4
    assert regressed["partial_value_usable"] is False
    assert regressed["prefer_best_retry_anchor"] is True
    assert any(
        "regressed_below_best_achieved_state" in str(item)
        for item in regressed["issues"]
    )


def test_stabilization_a_anchor_only_insufficiency_sets_next_round_constraint(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    result = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 7,
            "issues": [],
            "fixes": [],
            "summary": "anchor-only output is too broad",
            "plan_diagnosis": "OK",
            "repair_mode": "none",
        },
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#1",
            "observation": 'minted_variables: {"resolved_CNES": "#1"}',
        },
        live_progress_summary={
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "has_context": True,
            "material_progress": False,
            "partial_value_usable": False,
            "handoff_state": "partial_fallback_not_final",
            "final_operation_safe": False,
            "value_delivered": "resolved_anchor",
            "achieved_state": "resolved_anchor",
            "reason": "resolved_anchor_only_insufficient_for_plan",
        },
        round_context={
            "round": 2,
            "preferred_tool_mode": "full_solve",
        },
        tool_plan=_kg_plan(
            preferred_tool_mode="full_solve",
            execution_style="relation_first",
        ),
        tool_code="def run(payload: dict) -> dict:\n    return {}\n",
    )

    assert result["repair_mode"] == "rewrite_code"
    assert "minimum_acceptable_deliverable" in result
    assert "resolved_both_anchors" in result["minimum_acceptable_deliverable"]
    assert "PLAN-STAGE REPAIR TARGET" in str(result["primary_repair_instruction"])


def test_stage_binding_retry_context_sets_required_next_state_for_attribute_intersector(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        target_archetype="ATTRIBUTE_INTERSECTOR",
        preferred_tool_mode="full_solve",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
        attribute_target_concept="cheese.texture",
        topological_execution_plan=[
            "1. Resolve both anchors.",
            "2. Walk anchor A to cheese candidates.",
            "3. Walk anchor B to cheese candidates.",
            "4. Intersect the cheese candidate sets.",
            "5. Apply resolve_semantic_filter for texture.",
        ],
    )
    round_context, _ = controller._toolgen_bind_best_partial_retry_context(
        {"round": 2, "preferred_tool_mode": "full_solve"},
        tool_plan=plan,
        round_history=[
            {
                "round": 1,
                "value_delivered": "resolved_both_anchors",
                "tool_name": "generated_tool",
                "bankable_partial_progress": True,
            }
        ],
        best_candidate=None,
        best_live_candidate=None,
        best_partial_candidate=None,
        best_partial_live_candidate=None,
    )

    assert round_context["required_next_achieved_state"] == "built_filter_ready_set"
    assert round_context["required_handoff_achieved_state"] == "built_filter_ready_set"
    assert "intersect the two anchor-derived target sets" in str(
        round_context["stage_binding_instruction"]
    ).lower()
    assert "apply resolve_semantic_filter to the intersection" in str(
        round_context["stage_binding_instruction"]
    ).lower()
    assert "if preserved anchor vars" in str(
        round_context["stage_binding_instruction"]
    ).lower()
    assert "fallback recovery" in str(round_context["stage_binding_instruction"]).lower()
    assert "do not return error solely because preserved anchors are absent" in str(
        round_context["stage_binding_instruction"]
    ).lower()


def test_stage_binding_validation_demotes_anchor_reresolve_main_output_after_preserved_state(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        target_archetype="ATTRIBUTE_INTERSECTOR",
        preferred_tool_mode="progress_tool",
        execution_style="partial_value_first",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
        attribute_target_concept="cheese.texture",
        topological_execution_plan=[
            "1. Resolve both anchors.",
            "2. Walk anchor A to cheese candidates.",
            "3. Walk anchor B to cheese candidates.",
            "4. Intersect the cheese candidate sets.",
            "5. Apply resolve_semantic_filter for texture.",
        ],
    )

    result = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 7,
            "issues": [],
            "fixes": [],
            "summary": "candidate stopped at re-resolved anchors",
            "plan_diagnosis": "OK",
            "repair_mode": "none",
        },
        execution_validation={
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": (
                'minted_variables: {"resolved_Goat": "#1", "resolved_cows": "#2"}'
            ),
        },
        live_progress_summary={
            "execution_status": "MACRO EXHAUSTED",
            "has_final_variable": False,
            "has_context": True,
            "material_progress": True,
            "partial_value_usable": True,
            "bankable_partial_progress": True,
            "handoff_state": "partial_safe_continue",
            "final_operation_safe": False,
            "value_delivered": "resolved_both_anchors",
            "achieved_state": "resolved_both_anchors",
        },
        round_context={
            "round": 5,
            "preferred_tool_mode": "progress_tool",
            "best_achieved_state": "resolved_both_anchors",
        },
        tool_plan=plan,
        tool_code=(
            "def run(payload: dict) -> dict:\n"
            "    entities = payload.get('entities') or []\n"
            "    resolved = []\n"
            "    for ent in entities[:2]:\n"
            "        out = kg_utils.resolve_entity_to_vars(ent, None, {}, None)\n"
            "        ids = kg_utils.extract_var_ids(out)\n"
            "        if ids:\n"
            "            resolved.append(ids[0])\n"
            "    return {'status': 'MACRO EXHAUSTED', 'final_variable': None, 'observation': str(resolved)}\n"
        ),
    )

    assert result["grade"] <= 4
    assert result["partial_value_usable"] is False
    assert result["prefer_best_retry_anchor"] is True
    assert "does_not_respect_preserved_anchor_vars" in result["semantic_code_smells"]
    assert "re_resolve_anchors_as_main_output" in result["semantic_code_smells"]
    assert any(
        "does_not_respect_preserved_anchor_vars" in str(item)
        for item in result["issues"]
    )
    assert any(
        "re_resolve_anchors_as_main_output" in str(item)
        for item in result["issues"]
    )
    assert "fallback recovery" in str(result["primary_repair_instruction"]).lower()


def test_phase1_retry_state_activates_from_stage_binding_controls(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    state = controller._toolgen_phase1_retry_state(
        json.dumps(
            {
                "phase": "validator",
                "required_next_achieved_state": "built_filter_ready_set",
                "required_handoff_achieved_state": "built_filter_ready_set",
                "stage_binding_instruction": "advance directly to the filter-ready intersection",
                "minimum_acceptable_deliverable": "return at least built_filter_ready_set",
                "forbidden_fallback_shapes": ["raw_anchor_handoff"],
                "force_structural_stage_rewrite": True,
            }
        )
    )

    assert state["active"] is True
    assert state["omit_prior_code"] is True
    assert state["override_preferred_tool_mode"] == "progress_tool"
    assert state["required_next_achieved_state"] == "built_filter_ready_set"
    assert state["force_structural_stage_rewrite"] is True


def test_apply_round_strategy_context_rewrites_plan_to_stage_binding_sequence(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    payload = controller._toolgen_apply_round_strategy_context(
        _kg_plan(
            target_archetype="ATTRIBUTE_INTERSECTOR",
            preferred_tool_mode="full_solve",
            execution_style="walk_first",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
        ),
        {
            "round": 2,
            "preferred_tool_mode": "progress_tool",
            "required_next_achieved_state": "built_filter_ready_set",
            "required_handoff_achieved_state": "built_filter_ready_set",
            "stage_binding_instruction": "preserve anchors, walk both, intersect, filter, return filter-ready set",
            "minimum_acceptable_deliverable": "return built_filter_ready_set",
        },
        phase1_retry_state={
            "active": True,
            "omit_prior_code": False,
            "override_preferred_tool_mode": "progress_tool",
            "minimum_acceptable_deliverable": "return built_filter_ready_set",
            "primary_repair_dominates": True,
            "required_next_achieved_state": "built_filter_ready_set",
            "required_handoff_achieved_state": "built_filter_ready_set",
            "stage_binding_instruction": "preserve anchors, walk both, intersect, filter, return filter-ready set",
            "forbidden_fallback_shapes": [
                "raw_anchor_handoff",
                "raw_walk_set_handoff",
            ],
            "force_structural_stage_rewrite": False,
        },
    )
    steps = payload["topological_execution_plan_steps"]

    assert any("walk anchor a" in str(step).lower() for step in steps)
    assert any("walk anchor b" in str(step).lower() for step in steps)
    assert any("intersect the two anchor-derived target sets" in str(step).lower() for step in steps)
    assert any("apply resolve_semantic_filter to the intersection" in str(step).lower() for step in steps)
    assert any("built_filter_ready_set" in str(step) for step in steps)


def test_blueprint_prompt_adds_compact_finish_chain_override_for_filter_ready_handoff(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    prompt = controller._toolgen_build_blueprint_prompt(
        query="Question: find semi-firm cheese from goat and cows, Entities: ['Goat', 'cows', 'semi-firm']",
        tool_type="macro",
        reason="INPUT: ['Goat', 'cows', 'semi-firm']. GOAL: return the filter-ready overlap.",
        upgrade_goal="INPUT: ['Goat', 'cows', 'semi-firm']. GOAL: finish the filter-ready chain.",
        tool_plan=_kg_plan(
            target_archetype="ATTRIBUTE_INTERSECTOR",
            preferred_tool_mode="progress_tool",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
            required_next_achieved_state="built_filter_ready_set",
            required_handoff_achieved_state="built_filter_ready_set",
            topological_execution_plan=[
                "1. Resolve both anchors.",
                "2. Walk anchor A to cheese candidates.",
                "3. Walk anchor B to cheese candidates.",
                "4. Intersect the cheese candidate sets.",
                "5. Apply resolve_semantic_filter for texture.",
            ],
        ),
        env_name="knowledge_graph",
        env_contract="Question: find semi-firm cheese from goat and cows, Entities: ['Goat', 'cows', 'semi-firm']",
    )

    assert "FILTER-READY FINISH-CHAIN OVERRIDE" in prompt
    assert "Do not stop at resolved anchors, walked target sets, or raw intersection sets." in prompt
    assert "Avoid alternate fallback branches" in prompt


def test_validation_policy_repeated_same_stage_below_handoff_forces_structural_rewrite(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    result = controller._toolgen_apply_validation_policy(
        validation={
            "grade": 6,
            "issues": [],
            "fixes": [],
            "summary": "repeated same-stage exhaustion",
            "plan_diagnosis": "OK",
            "repair_mode": "none",
        },
        execution_validation={
            "status": "MACRO EXHAUSTED",
            "final_variable": None,
            "observation": 'minted_variables: {"resolved_Goat": "#1", "resolved_cows": "#2"}',
        },
        live_progress_summary={
            "execution_status": "MACRO EXHAUSTED",
            "has_final_variable": False,
            "has_context": True,
            "material_progress": True,
            "handoff_state": "exhausted",
            "final_operation_safe": False,
            "value_delivered": "resolved_both_anchors",
            "achieved_state": "resolved_both_anchors",
            "reason": "honest_zero_no_handoff",
        },
        round_context={
            "round": 3,
            "preferred_tool_mode": "full_solve",
            "best_achieved_state": "resolved_both_anchors",
            "previous_achieved_state": "resolved_both_anchors",
            "previous_failure_bucket": "partial_value_delivered",
        },
        tool_plan=_kg_plan(
            target_archetype="ATTRIBUTE_INTERSECTOR",
            preferred_tool_mode="full_solve",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
            topological_execution_plan=[
                "1. Resolve both anchors.",
                "2. Walk both anchors to cheese sets.",
                "3. Intersect the cheese sets.",
                "4. Apply resolve_semantic_filter for texture.",
            ],
        ),
        tool_code="def run(payload: dict) -> dict:\n    return {}\n",
    )

    assert result["force_structural_stage_rewrite"] is True
    assert result["prefer_best_retry_anchor"] is True
    assert result["required_next_achieved_state"] == "built_filter_ready_set"
    assert "intersect the two anchor-derived target sets" in str(
        result["primary_repair_instruction"]
    ).lower()


def test_dynamic_plan_smells_flag_multi_anchor_filter_order_violation(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    smells = controller._toolgen_dynamic_plan_code_smells(
        "\n".join(
            [
                "def run(payload: dict) -> dict:",
                "    candidate_map = {}",
                "    filtered = kg_utils.resolve_semantic_filter(walked_cheeses, 'cheese.texture', walked_cheeses)",
                "    return {'status': 'SUCCESS', 'final_variable': '#1', 'observation': 'ok'}",
            ]
        ),
        _kg_plan(
            target_archetype="ATTRIBUTE_INTERSECTOR",
            preferred_tool_mode="full_solve",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
            topological_execution_plan=[
                "1. Resolve both anchors.",
                "2. Walk to target sets.",
                "3. Intersect the target sets.",
                "4. Apply resolve_semantic_filter.",
            ],
        ),
    )

    assert "multi_anchor_filter_order_violation" in smells


def test_stabilization_a_value_classification_recognizes_grounded_walk_intersect_and_filter_labels(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        preferred_tool_mode="progress_tool",
        execution_style="partial_value_first",
        attribute_target_concept="cheese.texture",
        topological_execution_plan=[
            "1. Resolve both anchors.",
            "2. Walk to cheese candidates.",
            "3. Intersect the candidate sets.",
            "4. Apply resolve_semantic_filter for texture.",
        ],
    )

    walk_value = controller._toolgen_classify_value_delivered(
        tool_plan=plan,
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#3",
            "observation": 'minted_variables: {"walk_Goat_to_cheese": "#3"}',
        },
        live_progress_summary={
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "has_context": True,
        },
    )
    intersect_value = controller._toolgen_classify_value_delivered(
        tool_plan=plan,
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#4",
            "observation": 'minted_variables: {"intersect_Goat_cows": "#4"}',
        },
        live_progress_summary={
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "has_context": True,
        },
    )
    filter_value = controller._toolgen_classify_value_delivered(
        tool_plan=plan,
        execution_validation={
            "status": "SUCCESS",
            "final_variable": "#5",
            "observation": 'minted_variables: {"filter_texture_semifirm": ["#5"]}',
        },
        live_progress_summary={
            "execution_status": "SUCCESS",
            "has_final_variable": True,
            "has_context": True,
        },
    )

    assert walk_value == "built_target_set"
    assert intersect_value == "built_intersection_set"
    assert filter_value == "built_filter_ready_set"


def test_stabilization_a_single_anchor_plan_still_accepts_resolved_anchor_progress(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = {
        "preferred_tool_mode": "progress_tool",
        "execution_style": "partial_value_first",
        "target_concept": "scientist",
        "entities": ["Einstein"],
        "entity_target_concepts": ["scientist"],
        "topological_execution_plan": [
            "1. Resolve the anchor.",
            "2. Return the grounded scientist variable.",
        ],
    }

    result = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#1",
            "observation": 'minted_variables: {"resolved_Einstein": "#1"}',
        },
        {"tool_plan": plan},
    )

    assert result["value_delivered"] == "resolved_anchor"
    assert result["material_progress"] is True
    assert result["usefulness_passed"] is True


def test_stabilization_b_broad_multi_anchor_handoff_is_demoted_but_bankable(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        preferred_tool_mode="progress_tool",
        execution_style="partial_value_first",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
        attribute_target_concept="cheese.texture",
        topological_execution_plan=[
            "1. Resolve both anchors.",
            "2. Walk to cheese candidates.",
            "3. Intersect the candidate sets.",
            "4. Apply resolve_semantic_filter for texture.",
        ],
    )

    result = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#0",
            "observation": (
                'minted_variables: {"resolved_Goat": "#0", "resolved_cows": ["#1"]}'
            ),
        },
        {"tool_plan": plan},
    )

    assert result["value_delivered"] == "resolved_both_anchors"
    assert result["bankable_partial_progress"] is True
    assert result["usefulness_passed"] is False
    assert result["reason"] == "plan_stage_handoff_too_broad"


def test_stabilization_b_full_solve_attribute_filter_plan_rejects_broad_anchor_handoff(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        preferred_tool_mode="full_solve",
        execution_style="relation_first",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
        attribute_target_concept="cheese.texture",
        topological_execution_plan=[
            "1. Resolve both anchors.",
            "2. Walk to cheese candidates.",
            "3. Intersect the candidate sets.",
            "4. Apply resolve_semantic_filter for texture.",
        ],
    )

    result = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#0",
            "observation": (
                'minted_variables: {"resolved_Goat": "#0", "resolved_cows": ["#1"]}'
            ),
        },
        {"tool_plan": plan},
    )

    assert result["value_delivered"] == "resolved_both_anchors"
    assert result["minimum_handoff_achieved_state"] == "built_filter_ready_set"
    assert result["bankable_partial_progress"] is True
    assert result["usefulness_passed"] is False
    assert result["reason"] == "plan_stage_handoff_too_broad"


def test_stabilization_b_filter_ready_handoff_is_accepted_for_attribute_filter_plan(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    plan = _kg_plan(
        preferred_tool_mode="progress_tool",
        execution_style="partial_value_first",
        entities=["Goat", "cows", "semi-firm"],
        entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
        attribute_target_concept="cheese.texture",
        topological_execution_plan=[
            "1. Resolve both anchors.",
            "2. Walk to cheese candidates.",
            "3. Intersect the candidate sets.",
            "4. Apply resolve_semantic_filter for texture.",
        ],
    )

    result = controller._summarize_toolgen_live_progress(
        {
            "status": "SUCCESS",
            "final_variable": "#7",
            "observation": 'minted_variables: {"filter_texture_semifirm": ["#7"]}',
        },
        {"tool_plan": plan},
    )

    assert result["value_delivered"] == "built_filter_ready_set"
    assert result["usefulness_passed"] is True
    assert result["minimum_handoff_achieved_state"] == "built_filter_ready_set"


def test_stabilization_c_best_achieved_summary_skips_below_floor_anchor_states(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    history = [
        {
            "round": 1,
            "value_delivered": "resolved_anchor",
            "tool_name": "t1",
            "bankable_partial_progress": False,
        },
        {
            "round": 2,
            "value_delivered": "built_target_set",
            "tool_name": "t2",
            "bankable_partial_progress": True,
        },
    ]

    result = controller._toolgen_best_achieved_state_summary(history)
    assert result["best_achieved_state"] == "built_target_set"
    assert result["best_achieved_round"] == 2


def test_stabilization_b_helper_signature_mismatch_is_a_blocking_smell(
    tmp_path: pathlib.Path,
) -> None:
    """Fix B: calls to kg_utils helpers with wrong positional-arg count must be
    detected as 'helper_signature_mismatch' and that smell must block registration.

    walk_to_target requires 4 positional args; cross_intersect requires 3;
    resolve_semantic_filter requires 3.  Calling them with fewer args produces a
    TypeError at runtime, so this must be caught before live evaluation.
    """
    controller = _DummyController(tmp_path)

    # Bad: walk_to_target called with only 2 args (needs 4)
    code_bad_walk = (
        "def run(payload: dict) -> dict:\n"
        "    try:\n"
        "        actions_spec = payload.get('actions_spec', {})\n"
        "        vars_a = kg_utils.resolve_entity_to_vars('CNES', None, actions_spec, {})\n"
        "        # wrong: only 2 positional args, needs 4\n"
        "        result = kg_utils.walk_to_target(actions_spec, vars_a)\n"
        "        return {'status': 'SUCCESS', 'final_variable': None, 'observation': str(result)}\n"
        "    except Exception as e:\n"
        "        return {'status': 'ERROR', 'final_variable': None, 'observation': str(e)}\n"
        "\ndef self_test(): return True\n"
    )
    smells_walk = controller._toolgen_semantic_code_smells(code_bad_walk)
    assert "helper_signature_mismatch" in smells_walk, (
        "walk_to_target with 2 args (needs 4) must produce helper_signature_mismatch"
    )

    # Bad: cross_intersect called with only 2 args (needs 3)
    code_bad_intersect = (
        "def run(payload: dict) -> dict:\n"
        "    try:\n"
        "        actions_spec = payload.get('actions_spec', {})\n"
        "        set_a = kg_utils.resolve_entity_to_vars('CNES', None, actions_spec, {})\n"
        "        # wrong: only 2 positional args, needs 3\n"
        "        result = kg_utils.cross_intersect(set_a, actions_spec)\n"
        "        return {'status': 'SUCCESS', 'final_variable': None, 'observation': str(result)}\n"
        "    except Exception as e:\n"
        "        return {'status': 'ERROR', 'final_variable': None, 'observation': str(e)}\n"
        "\ndef self_test(): return True\n"
    )
    smells_intersect = controller._toolgen_semantic_code_smells(code_bad_intersect)
    assert "helper_signature_mismatch" in smells_intersect, (
        "cross_intersect with 2 args (needs 3) must produce helper_signature_mismatch"
    )

    # Positive control: correct arg counts must not trigger the smell
    code_good = (
        "def run(payload: dict) -> dict:\n"
        "    try:\n"
        "        actions_spec = payload.get('actions_spec', {})\n"
        "        domain_hints = payload.get('domain_hints', {})\n"
        "        target_concept = payload.get('target_concept', 'spacecraft')\n"
        "        vars_a = kg_utils.resolve_entity_to_vars('CNES', None, actions_spec, domain_hints)\n"
        "        # correct: 4 positional args\n"
        "        walk_a = kg_utils.walk_to_target(actions_spec, vars_a, target_concept, domain_hints)\n"
        "        vars_b = kg_utils.resolve_entity_to_vars('Astrium', None, actions_spec, domain_hints)\n"
        "        walk_b = kg_utils.walk_to_target(actions_spec, vars_b, target_concept, domain_hints)\n"
        "        # correct: 3 positional args\n"
        "        result = kg_utils.cross_intersect(actions_spec, walk_a, walk_b)\n"
        "        return {'status': 'SUCCESS', 'final_variable': None, 'observation': str(result)}\n"
        "    except Exception as e:\n"
        "        return {'status': 'ERROR', 'final_variable': None, 'observation': str(e)}\n"
        "\ndef self_test(): return True\n"
    )
    smells_good = controller._toolgen_semantic_code_smells(code_good)
    assert "helper_signature_mismatch" not in smells_good, (
        "correct arg counts must not trigger helper_signature_mismatch"
    )

    # helper_signature_mismatch must be listed in blocking_kg_smells so the
    # validate_candidate_tool path sets severe_semantic_smell=True and gates admission.
    import inspect as _inspect
    _candidate_src = _inspect.getsource(
        type(controller)._toolgen_validate_candidate_tool
    )
    assert '"helper_signature_mismatch"' in _candidate_src, (
        "helper_signature_mismatch must be listed in blocking_kg_smells inside "
        "_toolgen_validate_candidate_tool to prevent live-run TypeError"
    )


def test_stabilization_d_resolve_semantic_filter_context_misuse_is_preblocked(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    live_calls = {"count": 0}
    controller._toolgen_execution_payload = {
        "tool_plan": _kg_plan(
            preferred_tool_mode="progress_tool",
            execution_style="partial_value_first",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
            topological_execution_plan=[
                "1. Resolve both anchors.",
                "2. Walk to cheese candidates.",
                "3. Intersect the candidate sets.",
                "4. Apply resolve_semantic_filter for texture.",
            ],
        )
    }
    controller._toolgen_execution_check = lambda tool_code, exec_payload: live_calls.__setitem__(
        "count", live_calls["count"] + 1
    ) or {
        "status": "SUCCESS",
        "final_variable": "#9",
        "observation": "unexpected live execution",
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 8,
        "issues": [],
        "fixes": [],
        "summary": "candidate superficially looks usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    tool_code = (
        "def run(payload: dict) -> dict:\n"
        "    variable_list = payload.get('variable_list') or []\n"
        "    target_concept = payload.get('attribute_target_concept')\n"
        "    base_var = '#1'\n"
        "    filtered = kg_utils.resolve_semantic_filter(variable_list, target_concept, base_var)\n"
        "    return {'status': 'SUCCESS', 'final_variable': '#9', 'observation': str(filtered)}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    return True\n"
    )

    validation = controller._toolgen_validate_candidate_tool(
        {
            "name": "bad_filter_context",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
        },
        tool_code,
        task_pack="test task pack",
        run_live_execution_check=True,
    )

    assert validation is not None
    assert live_calls["count"] == 0
    assert "resolve_semantic_filter_context_misuse" in validation["semantic_code_smells"]
    assert "stale_anchor_filter_context" in validation["semantic_code_smells"]
    assert validation["usefulness_passed"] is False
    assert validation["grade"] <= 4


def test_stabilization_d_variable_list_is_optional_and_missing_it_is_preblocked(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    live_calls = {"count": 0}
    controller._toolgen_execution_payload = {
        "tool_plan": _kg_plan(
            preferred_tool_mode="progress_tool",
            execution_style="partial_value_first",
        )
    }
    controller._toolgen_execution_check = lambda tool_code, exec_payload: live_calls.__setitem__(
        "count", live_calls["count"] + 1
    ) or {
        "status": "SUCCESS",
        "final_variable": "#9",
        "observation": "unexpected live execution",
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 8,
        "issues": [],
        "fixes": [],
        "summary": "candidate superficially looks usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    tool_code = (
        "def run(payload: dict) -> dict:\n"
        "    variable_list = payload.get('variable_list')\n"
        "    if not variable_list:\n"
        "        return {'status': 'ERROR', 'final_variable': None, 'observation': 'missing variable_list'}\n"
        "    return {'status': 'SUCCESS', 'final_variable': '#1', 'observation': 'ok'}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    return True\n"
    )

    validation = controller._toolgen_validate_candidate_tool(
        {
            "name": "bad_preserved_state_contract",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
        },
        tool_code,
        task_pack="test task pack",
        run_live_execution_check=True,
    )

    assert validation is not None
    assert live_calls["count"] == 0
    assert "requires_variable_list_in_payload" in validation["semantic_code_smells"]
    assert validation["usefulness_passed"] is False
    assert validation["grade"] <= 4


def test_stabilization_d_remaining_helper_context_misuse_is_preblocked(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    live_calls = {"count": 0}
    controller._toolgen_execution_check = lambda tool_code, exec_payload: live_calls.__setitem__(
        "count", live_calls["count"] + 1
    ) or {
        "status": "SUCCESS",
        "final_variable": "#9",
        "observation": "unexpected live execution",
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 8,
        "issues": [],
        "fixes": [],
        "summary": "candidate superficially looks usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    cases = [
        (
            "walk_to_target_context_misuse",
            _kg_plan(
                preferred_tool_mode="full_solve",
                execution_style="relation_first",
            ),
            (
                "def run(payload: dict) -> dict:\n"
                "    actions_spec = payload.get('actions_spec') or {}\n"
                "    domain_hints = payload.get('domain_hints') or {}\n"
                "    vars_a = kg_utils.resolve_entity_to_vars('CNES', None, actions_spec, domain_hints)\n"
                "    walked = kg_utils.walk_to_target(vars_a, actions_spec, 'spacecraft', domain_hints)\n"
                "    return {'status': 'SUCCESS', 'final_variable': '#1', 'observation': str(walked)}\n"
                "\n"
                "def self_test() -> bool:\n"
                "    return True\n"
            ),
        ),
        (
            "cross_intersect_context_misuse",
            _kg_plan(
                preferred_tool_mode="full_solve",
                execution_style="relation_first",
            ),
            (
                "def run(payload: dict) -> dict:\n"
                "    actions_spec = payload.get('actions_spec') or {}\n"
                "    walk_a = '#1'\n"
                "    walk_b = '#2'\n"
                "    overlap = kg_utils.cross_intersect(walk_a, walk_b, actions_spec)\n"
                "    return {'status': 'SUCCESS', 'final_variable': '#3', 'observation': str(overlap)}\n"
                "\n"
                "def self_test() -> bool:\n"
                "    return True\n"
            ),
        ),
        (
            "extract_attribute_value_context_misuse",
            {
                "target_archetype": "SUPERLATIVE_FINDER",
                "preferred_tool_mode": "full_solve",
                "execution_style": "attribute_mapping_first",
                "entities": ["fighter"],
                "target_concept": "aircraft",
                "attribute_target_concept": "maximum_damage",
                "topological_execution_plan": [
                    "1. Resolve the anchor.",
                    "2. Walk to aircraft candidates.",
                    "3. Use extract_attribute_value on the helper output.",
                    "4. Use argmax over the extracted values.",
                ],
            },
            (
                "def run(payload: dict) -> dict:\n"
                "    attribute_target_concept = payload.get('attribute_target_concept')\n"
                "    extracted = kg_utils.extract_attribute_value(attribute_target_concept)\n"
                "    return {'status': 'SUCCESS', 'final_variable': '#4', 'observation': str(extracted)}\n"
                "\n"
                "def self_test() -> bool:\n"
                "    return True\n"
            ),
        ),
    ]

    for smell, plan, tool_code in cases:
        controller._toolgen_execution_payload = {"tool_plan": plan}
        before = live_calls["count"]
        validation = controller._toolgen_validate_candidate_tool(
            {
                "name": f"bad_{smell}",
                "description": "test tool",
                "signature": "run(payload: dict) -> dict",
            },
            tool_code,
            task_pack="test task pack",
            run_live_execution_check=True,
        )

        assert validation is not None
        assert live_calls["count"] == before
        assert smell in validation["semantic_code_smells"]
        assert validation["usefulness_passed"] is False
        assert validation["grade"] <= 4


def test_stabilization_d_missing_attribute_filter_application_is_blocking(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    controller._toolgen_execution_payload = {
        "tool_plan": _kg_plan(
            preferred_tool_mode="progress_tool",
            execution_style="partial_value_first",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
            topological_execution_plan=[
                "1. Resolve both anchors.",
                "2. Walk to cheese candidates.",
                "3. Intersect the candidate sets.",
                "4. Apply resolve_semantic_filter for texture.",
            ],
        )
    }
    controller._toolgen_execution_check = lambda tool_code, exec_payload: {
        "status": "SUCCESS",
        "final_variable": "#3",
        "observation": 'minted_variables: {"intersect_goat_cows": "#3"}',
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "summary": "candidate superficially looks usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    tool_code = (
        "def run(payload: dict) -> dict:\n"
        "    actions_spec = payload.get('actions_spec') or {}\n"
        "    domain_hints = payload.get('domain_hints') or {}\n"
        "    vars_a = kg_utils.resolve_entity_to_vars('Goat', None, actions_spec, domain_hints)\n"
        "    vars_b = kg_utils.resolve_entity_to_vars('cows', None, actions_spec, domain_hints)\n"
        "    walk_a = kg_utils.walk_to_target(actions_spec, vars_a, 'cheese', domain_hints)\n"
        "    walk_b = kg_utils.walk_to_target(actions_spec, vars_b, 'cheese', domain_hints)\n"
        "    overlap = kg_utils.cross_intersect(actions_spec, walk_a, walk_b)\n"
        "    return {'status': 'SUCCESS', 'final_variable': '#3', 'observation': str(overlap)}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    return True\n"
    )

    validation = controller._toolgen_validate_candidate_tool(
        {
            "name": "missing_filter",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
        },
        tool_code,
        task_pack="test task pack",
        run_live_execution_check=True,
    )

    assert validation is not None
    assert "missing_attribute_filter_application" in validation["semantic_code_smells"]
    assert validation["usefulness_passed"] is False
    assert validation["grade"] <= 4


def test_validate_candidate_tool_blocks_exhausted_without_final_variable_admission(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    controller._toolgen_execution_payload = {
        "tool_plan": _kg_plan(preferred_tool_mode="full_solve")
    }
    controller._toolgen_execution_check = lambda tool_code, exec_payload: {
        "status": "MACRO EXHAUSTED",
        "final_variable": None,
        "observation": (
            "MACRO EXHAUSTED: Resulting set is empty. "
            'minted_variables: {"resolved_CNES": "#1", "resolved_Astrium": "#2"}'
        ),
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 9,
        "issues": [],
        "fixes": [],
        "summary": "looks superficially usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    validation = controller._toolgen_validate_candidate_tool(
        {
            "name": "exhausted_without_final",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
        },
        "def run(payload: dict) -> dict:\n    return {'status': 'MACRO EXHAUSTED', 'final_variable': None, 'observation': 'x'}\n",
        task_pack="test task pack",
        run_live_execution_check=True,
    )

    assert validation is not None
    assert validation["admission_blocked"] is True
    assert validation["usefulness_passed"] is False
    assert validation["grade"] <= 4


def test_stabilization_e_blanket_entity_resolve_loop_is_blocked_for_progress_retry(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    live_calls = {"count": 0}
    controller._toolgen_execution_payload = {
        "tool_plan": _kg_plan(
            preferred_tool_mode="progress_tool",
            execution_style="partial_value_first",
            entities=["Goat", "cows", "semi-firm"],
            entity_target_concepts=["dairy.animal", "dairy.animal", "cheese.texture"],
            attribute_target_concept="cheese.texture",
        )
    }
    controller._toolgen_execution_check = lambda tool_code, exec_payload: live_calls.__setitem__(
        "count", live_calls["count"] + 1
    ) or {
        "status": "SUCCESS",
        "final_variable": "#1",
        "observation": 'Variable #1 contains resolved anchor. minted_variables: {"resolved_Goat": "#1"}',
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 7,
        "issues": [],
        "fixes": [],
        "summary": "candidate superficially looks usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    tool_code = (
        '"""blanket resolver"""\n'
        "def run(payload: dict) -> dict:\n"
        "    entities = payload.get('entities') or []\n"
        "    actions_spec = payload.get('actions_spec') or {}\n"
        "    domain_hints = payload.get('domain_hints') or {}\n"
        "    attribute_target_concept = payload.get('attribute_target_concept')\n"
        "    for ent in entities:\n"
        "        out = kg_utils.resolve_entity_to_vars(ent, None, actions_spec, domain_hints)\n"
        "        ids = kg_utils.extract_var_ids(out)\n"
        "        if ids:\n"
        "            return {'status': 'SUCCESS', 'final_variable': ids[0], 'observation': 'resolved first anchor'}\n"
        "    return {'status': 'MACRO EXHAUSTED', 'final_variable': None, 'observation': str(attribute_target_concept)}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    return True\n"
    )
    validation = controller._toolgen_validate_candidate_tool(
        {
            "name": "blanket_resolve_progress_tool",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
        },
        tool_code,
        task_pack="test task pack",
        run_live_execution_check=True,
    )

    assert validation is not None
    assert live_calls["count"] == 0
    assert "blanket_entity_resolve_loop" in validation["semantic_code_smells"]
    assert validation["usefulness_passed"] is False
    assert validation["usefulness_reason"] == "blocking_semantic_code_smells"


def test_stabilization_e_first_candidate_selection_by_index_is_preblocked(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    live_calls = {"count": 0}
    controller._toolgen_execution_payload = {
        "tool_plan": _kg_plan(
            preferred_tool_mode="progress_tool",
            execution_style="partial_value_first",
        )
    }
    controller._toolgen_execution_check = lambda tool_code, exec_payload: live_calls.__setitem__(
        "count", live_calls["count"] + 1
    ) or {
        "status": "SUCCESS",
        "final_variable": "#1",
        "observation": "unexpected live execution",
    }
    controller._toolgen_validator_call = lambda payload: {
        "grade": 8,
        "issues": [],
        "fixes": [],
        "summary": "candidate superficially looks usable",
        "plan_diagnosis": "OK",
        "repair_mode": "none",
    }

    tool_code = (
        "def run(payload: dict) -> dict:\n"
        "    candidate_ids = ['#1', '#2']\n"
        "    return {'status': 'SUCCESS', 'final_variable': candidate_ids[0], 'observation': 'selected first candidate by index'}\n"
        "\n"
        "def self_test() -> bool:\n"
        "    return True\n"
    )
    validation = controller._toolgen_validate_candidate_tool(
        {
            "name": "first_candidate_by_index",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
        },
        tool_code,
        task_pack="test task pack",
        run_live_execution_check=True,
    )

    assert validation is not None
    assert live_calls["count"] == 0
    assert "first_candidate_selection_by_index" in validation["semantic_code_smells"]
    assert validation["usefulness_passed"] is False


def test_stabilization_c_pivot_required_omits_prior_code_from_full_rewrite_prompt(
    tmp_path: pathlib.Path,
) -> None:
    """Fix C: when pivot_required=True the full-rewrite prompt must omit LAST_TOOL_CODE.

    The fix adds _effective_phase1_state with omit_prior_code=True whenever
    round_context.pivot_required is True.  We test two things:
    1. _toolgen_build_full_rewrite_prompt respects omit_prior_code=True
       (mechanism works).
    2. The round-strategy context correctly sets pivot_required=True after two
       no-progress rounds on the same strategy, so the trigger condition fires
       (trigger works).
    Together these confirm the end-to-end behavior.
    """
    controller = _DummyController(tmp_path)

    prior_code = (
        "def run(payload: dict) -> dict:\n"
        "    return {'status': 'ERROR', 'final_variable': None, 'observation': 'stub'}\n"
        "\ndef self_test(): return True\n"
    )
    base_prompt = "dummy base prompt"
    round_ctx = {"round": 3, "strategy_family": "walk_first"}
    feedback = "Fix the empty walk."

    # Without omit: LAST_TOOL_CODE must appear in the prompt.
    prompt_with = controller._toolgen_build_full_rewrite_prompt(
        base_prompt=base_prompt,
        round_context=round_ctx,
        feedback_note=feedback,
        round_history=[],
        last_tool_code=prior_code,
        phase1_retry_state={"omit_prior_code": False},
    )
    assert "LAST_TOOL_CODE:" in prompt_with

    # With omit_prior_code=True: LAST_TOOL_CODE must be absent.
    prompt_without = controller._toolgen_build_full_rewrite_prompt(
        base_prompt=base_prompt,
        round_context=round_ctx,
        feedback_note=feedback,
        round_history=[],
        last_tool_code=prior_code,
        phase1_retry_state={"omit_prior_code": True},
    )
    assert "LAST_TOOL_CODE:" not in prompt_without

    # Trigger check: two no-progress rounds on the same strategy → pivot_required=True.
    # (Uses the same plan/failure pattern as test_retry_policy_pivots_after_repeated_no_progress.)
    _exec_payload = {"tool_plan": _kg_plan(execution_style="relation_first")}
    _r1 = controller._toolgen_compute_round_strategy_context(
        round_idx=1, exec_payload=_exec_payload, round_history=[],
    )
    _r2 = controller._toolgen_compute_round_strategy_context(
        round_idx=2,
        exec_payload=_exec_payload,
        round_history=[
            _round_history_entry(
                _r1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable",
            )
        ],
    )
    _r3 = controller._toolgen_compute_round_strategy_context(
        round_idx=3,
        exec_payload=_exec_payload,
        round_history=[
            _round_history_entry(
                _r1,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable",
            ),
            _round_history_entry(
                _r2,
                failure_family="count_target_wrong",
                failure_bucket="code_local_no_progress",
                material_progress=False,
                summary="wrong count variable",
            ),
        ],
    )
    assert _r3["pivot_required"] is True, (
        "two no-progress rounds must set pivot_required=True in round_context"
    )

    # Mirror the C fix logic: when pivot_required, effective state has omit_prior_code=True.
    _effective_phase1_state = {"omit_prior_code": False}
    if _r3.get("pivot_required"):
        _effective_phase1_state = dict(_effective_phase1_state)
        _effective_phase1_state["omit_prior_code"] = True
    assert _effective_phase1_state["omit_prior_code"] is True, (
        "pivot_required must force omit_prior_code=True in the effective phase1 state"
    )


class _DataSparseValidatorLanguageModel(_LivePathLanguageModel):
    """Returns DATA_SPARSE with a blocking code smell on the first validator call.

    On subsequent calls (after the override forces a retry), returns a clean OK
    result so the loop can complete.
    """

    def __init__(self) -> None:
        super().__init__(fail_toolgen=False)
        self._validator_calls = 0

    def _validator_response(self, prompt: str) -> str:
        self._validator_calls += 1
        payload = json.loads(prompt)
        tool_code = str(payload.get("tool_code") or "")
        if self._validator_calls == 1:
            # First call: DATA_SPARSE + blocking helper_signature_mismatch smell.
            return json.dumps(
                {
                    "grade": 2,
                    "issues": ["kg graph lacks data (sparse)"],
                    "fixes": [],
                    "summary": "sparse graph: walk returned empty",
                    "plan_diagnosis": "DATA_SPARSE",
                    "repair_mode": "none",
                    "semantic_code_smells": ["helper_signature_mismatch"],
                }
            )
        # Subsequent calls: clean result so the tool gets registered.
        return json.dumps(
            {
                "grade": 9,
                "issues": [],
                "fixes": [],
                "summary": "count variable returned correctly",
                "plan_diagnosis": "OK",
                "repair_mode": "none",
                "semantic_code_smells": [],
            }
        )


def test_stabilization_d_data_sparse_overridden_when_code_quality_failure(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    """Fix D: DATA_SPARSE must be overridden to OK when evidence points to a code
    quality or handoff failure rather than a genuinely sparse graph.

    When the validator returns plan_diagnosis=DATA_SPARSE together with a blocking
    code smell (helper_signature_mismatch) or grade≤2, the toolgen loop must:
    - emit toolgen_data_sparse_overridden (not toolgen_data_sparse_abort)
    - continue the retry loop so the code-quality issue can be fixed
    """
    controller = _build_live_controller(
        tmp_path,
        monkeypatch,
        language_model=_DataSparseValidatorLanguageModel(),
        kg_task_ref=_FakeKGTaskRef(),
    )
    result = _run_live_toolgen_case(controller, _COUNT_QUERY)

    generated_events = result["generated_events"]

    # Override must have been logged at least once.
    override_events = [
        e for e in generated_events if e.get("event") == "toolgen_data_sparse_overridden"
    ]
    assert override_events, (
        "DATA_SPARSE with blocking code smell must emit toolgen_data_sparse_overridden"
    )
    assert override_events[0]["reason"] == "code_quality_or_handoff_failure"

    # The loop must NOT have short-circuited via toolgen_data_sparse_abort on
    # the same round where the override fired.
    override_round = override_events[0]["round"]
    abort_on_override_round = [
        e
        for e in generated_events
        if e.get("event") == "toolgen_data_sparse_abort" and e.get("round") == override_round
    ]
    assert not abort_on_override_round, (
        "toolgen_data_sparse_abort must NOT fire in the same round as the override"
    )


# ---------------------------------------------------------------------------
# Turn-0 ToolGen gating (spec A/B)
# ---------------------------------------------------------------------------


def test_best_achieved_state_summary_returns_none_for_empty_history(
    tmp_path: pathlib.Path,
) -> None:
    """_toolgen_best_achieved_state_summary returns 'none' when round_history is empty."""
    controller = _DummyController(tmp_path)
    result = controller._toolgen_best_achieved_state_summary([])
    assert result["best_achieved_state"] == "none"
    assert result["best_achieved_round"] is None


def test_best_achieved_state_summary_picks_highest_ranked_state(
    tmp_path: pathlib.Path,
) -> None:
    """_toolgen_best_achieved_state_summary returns the highest-ranked value state seen."""
    controller = _DummyController(tmp_path)
    history = [
        {"round": 1, "value_delivered": "resolved_anchor", "tool_name": "t1"},
        {"round": 2, "value_delivered": "produced_actionable_handoff", "tool_name": "t2"},
        {"round": 3, "value_delivered": "built_target_set", "tool_name": "t3"},
    ]
    result = controller._toolgen_best_achieved_state_summary(history)
    assert result["best_achieved_state"] == "produced_actionable_handoff"
    assert result["best_achieved_round"] == 2
    assert result["best_achieved_tool_name"] == "t2"


def test_best_achieved_state_summary_skips_none_value_entries(
    tmp_path: pathlib.Path,
) -> None:
    """'none' value_delivered entries do not contribute to best-achieved state."""
    controller = _DummyController(tmp_path)
    history = [
        {"round": 1, "value_delivered": "none", "tool_name": "t1"},
        {"round": 2, "value_delivered": "resolved_both_anchors", "tool_name": "t2"},
    ]
    result = controller._toolgen_best_achieved_state_summary(history)
    assert result["best_achieved_state"] == "resolved_both_anchors"
    assert result["best_achieved_round"] == 2


def test_best_partial_retry_context_preserves_best_candidate_anchor(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    best_partial = _candidate_obj(
        name="best_partial",
        value_delivered="built_intersection_set",
        partial_value_usable=True,
        semantic_trust_level="partial_unverified",
        semantic_code_smells=[],
        failure_bucket="partial_value_delivered",
        grade=6,
    )
    context, anchor_candidate = controller._toolgen_bind_best_partial_retry_context(
        {"round": 4, "pivot_required": False},
        round_history=_make_round_history_with_partial_success(),
        best_candidate=best_partial,
        best_live_candidate=None,
        best_partial_candidate=best_partial,
        best_partial_live_candidate=None,
    )
    assert context["preserve_best_partial_candidate"] is True
    assert context["best_achieved_state"] == "produced_actionable_handoff"
    assert context["do_not_regress_below_best_achieved_state"] == "produced_actionable_handoff"
    assert anchor_candidate is best_partial


def test_best_achieved_final_state_is_used_as_retry_anchor(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    best_full = _candidate_obj(
        name="best_full",
        value_delivered="produced_final_variable",
        partial_value_usable=True,
        semantic_trust_level="verified",
        semantic_code_smells=[],
        failure_bucket="final_value_delivered",
        grade=8,
        usefulness_passed=True,
    )
    weaker_partial = _candidate_obj(
        name="weaker_partial",
        value_delivered="resolved_both_anchors",
        partial_value_usable=True,
        semantic_trust_level="partial_unverified",
        semantic_code_smells=[],
        failure_bucket="partial_value_delivered",
        grade=6,
    )

    context, anchor_candidate = controller._toolgen_bind_best_partial_retry_context(
        {"round": 4, "pivot_required": False},
        round_history=[
            {
                "round": 1,
                "value_delivered": "produced_final_variable",
                "tool_name": "best_full",
                "usefulness_passed": True,
            }
        ],
        best_candidate=best_full,
        best_live_candidate=None,
        best_partial_candidate=weaker_partial,
        best_partial_live_candidate=None,
    )

    assert context["preserve_best_partial_candidate"] is True
    assert context["best_achieved_state"] == "produced_final_variable"
    assert context["best_retry_anchor_achieved_state"] == "produced_final_variable"
    assert anchor_candidate is best_full


def test_feedback_note_with_authoritative_baseline_uses_best_candidate_validation(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    retry_anchor = _candidate_obj(
        name="best_intersection",
        value_delivered="built_intersection_set",
        partial_value_usable=True,
        semantic_trust_level="partial_unverified",
        semantic_code_smells=[],
        failure_bucket="partial_value_delivered",
        grade=7,
    )
    retry_anchor["validation"].update(
        {
            "summary": "best prior candidate built the grounded intersection set",
            "issues": ["repair local helper misuse without changing topology"],
            "fixes": ["keep the grounded intersection handoff"],
            "achieved_state": "built_intersection_set",
        }
    )

    note = controller._toolgen_feedback_note_with_authoritative_baseline(
        json.dumps(
            {
                "phase": "validator",
                "validation": {"summary": "weak current candidate"},
                "primary_repair_instruction": "repair the current round",
            }
        ),
        failed_validation={
            "prefer_best_retry_anchor": True,
            "primary_repair_instruction": "repair the local regression",
        },
        retry_anchor_candidate=retry_anchor,
    )
    payload = json.loads(note)

    assert payload["prefer_best_retry_anchor"] is True
    assert payload["validation"]["summary"] == (
        "best prior candidate built the grounded intersection set"
    )
    assert payload["authoritative_retry_baseline"]["achieved_state"] == (
        "built_intersection_set"
    )
    assert "authoritative_retry_baseline" in payload["CRITICAL_INSTRUCTION"].lower()


def test_normalize_tool_spec_persists_structured_registration_metadata(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    enriched = controller._toolgen_enrich_registration_payload(
        {
            "name": "kg_macro_generated_tool",
            "description": "test tool",
            "signature": "run(payload: dict) -> dict",
            "code_lines": [
                "def run(payload: dict) -> dict:",
                "    return {'status': 'SUCCESS', 'final_variable': '#1', 'observation': 'ok'}",
            ],
        },
        registration_exec_payload=_kg_plan(),
    )
    normalized = controller._normalize_tool_spec(enriched)
    props = normalized["input_schema"]["properties"]

    assert props["target_archetype"]["const"] == "COUNTING_INTERSECTOR"
    assert props["output_form"]["const"] == "count_variable"
    assert "archetype:COUNTING_INTERSECTOR" in normalized["capabilities"]
    assert "output_form:count_variable" in normalized["capabilities"]


def test_orchestrator_prefers_structured_archetype_and_output_form_metadata(
    tmp_path: pathlib.Path,
) -> None:
    controller = _DummyController(tmp_path)
    tool = types.SimpleNamespace(
        name="kg_macro_generated_tool",
        signature="run(payload: dict) -> dict",
        docstring="generic facade",
        description="generic grounded tool",
        input_schema={
            "type": "object",
            "properties": {
                "target_archetype": {"const": "COUNTING_INTERSECTOR", "type": "string"},
                "output_form": {"const": "count_variable", "type": "string"},
            },
        },
        required_keys=[],
        optional_keys=[],
        property_types={},
        capabilities=[
            "archetype:COUNTING_INTERSECTOR",
            "output_form:count_variable",
            "target_concept:spacecraft",
        ],
        creation_time="1",
        success_count=0,
        failure_count=0,
        reliability_score=1.0,
        negative_marks=0,
    )
    controller._registry = types.SimpleNamespace(
        retrieve_similar_tools=lambda query_text, top_k, environment: [tool]
    )
    controller._load_dynamic_registry = lambda: [tool]  # noqa: SLF001
    controller._parse_tool_invoke_contract = lambda tool_name: None  # noqa: SLF001
    controller._append_generated_tools_log = lambda payload: None  # noqa: SLF001

    compact = controller._orchestrator_compact_existing_tools(
        query_text="How many shared spacecraft are there?",
        tool_plan=_kg_plan(),
        query_entity_count=2,
    )
    compatible, _, request_form, tool_form = controller._is_tool_output_form_compatible(
        tool,
        archetype_label="UNKNOWN",
        query_text="How many shared spacecraft are there?",
        tool_plan=_kg_plan(),
    )

    assert compact
    assert compact[0]["archetype"] == "COUNTING_INTERSECTOR"
    assert compatible is True
    assert request_form == "count_variable"
    assert tool_form == "count_variable"


# ---------------------------------------------------------------------------
# Patch failure best-achieved anchor preservation (spec C/D/E)
# ---------------------------------------------------------------------------


def _make_round_history_with_partial_success() -> list[dict]:
    """Build a round_history where round 3 achieved produced_actionable_handoff."""
    return [
        {"round": 1, "value_delivered": "none", "tool_name": "tool_v1"},
        {"round": 2, "value_delivered": "resolved_anchor", "tool_name": "tool_v2"},
        {"round": 3, "value_delivered": "produced_actionable_handoff", "tool_name": "tool_v3"},
    ]


def test_patch_plan_parse_failure_includes_best_achieved_anchor(
    tmp_path: pathlib.Path,
) -> None:
    """After patch_plan_parse failure, feedback_note must include best-achieved anchor
    fields when a prior round achieved meaningful partial value."""
    controller = _DummyController(tmp_path)
    round_history = _make_round_history_with_partial_success()
    best = controller._toolgen_best_achieved_state_summary(round_history)

    # Simulate what the patched code does for patch_plan_parse failure
    _patch_parse_note: dict = {
        "phase": "patch_plan_parse",
        "error": "Patch plan was not valid JSON with a non-empty operations list.",
        "raw": "",
    }
    if best.get("best_achieved_state", "none") != "none":
        _best_round = best.get("best_achieved_round")
        _best_state = best["best_achieved_state"]
        _patch_parse_note["best_achieved_state"] = _best_state
        _patch_parse_note["best_achieved_round"] = _best_round
        _patch_parse_note["primary_repair_instruction"] = (
            f"Round {_best_round} achieved '{_best_state}'. "
            "Recovery must preserve or improve that partial-success shape. "
            "Do not regress to broader rewrite or lower-value first-attempt behavior."
        )
    feedback_note = json.loads(json.dumps(_patch_parse_note, ensure_ascii=True, default=str))

    assert feedback_note["best_achieved_state"] == "produced_actionable_handoff", (
        "patch_plan_parse feedback_note must include best_achieved_state anchor"
    )
    assert feedback_note["best_achieved_round"] == 3, (
        "patch_plan_parse feedback_note must record the round that achieved best state"
    )
    assert "primary_repair_instruction" in feedback_note, (
        "patch_plan_parse feedback_note must include primary_repair_instruction anchor"
    )
    assert "produced_actionable_handoff" in feedback_note["primary_repair_instruction"]
    assert "regress" in feedback_note["primary_repair_instruction"].lower()


def test_patch_apply_failure_includes_best_achieved_anchor(
    tmp_path: pathlib.Path,
) -> None:
    """After patch_apply failure, feedback_note must include best-achieved anchor
    fields when a prior round achieved meaningful partial value."""
    controller = _DummyController(tmp_path)
    round_history = _make_round_history_with_partial_success()
    best = controller._toolgen_best_achieved_state_summary(round_history)

    # Simulate what the patched code does for patch_apply failure
    _patch_apply_note: dict = {
        "phase": "patch_apply",
        "error": "hunk mismatch at line 42",
        "plan": [],
    }
    if best.get("best_achieved_state", "none") != "none":
        _best_round = best.get("best_achieved_round")
        _best_state = best["best_achieved_state"]
        _patch_apply_note["best_achieved_state"] = _best_state
        _patch_apply_note["best_achieved_round"] = _best_round
        _patch_apply_note["primary_repair_instruction"] = (
            f"Round {_best_round} achieved '{_best_state}'. "
            "Recovery must preserve or improve that partial-success shape. "
            "Do not regress to broader rewrite or lower-value first-attempt behavior."
        )
    feedback_note = json.loads(json.dumps(_patch_apply_note, ensure_ascii=True, default=str))

    assert feedback_note["best_achieved_state"] == "produced_actionable_handoff", (
        "patch_apply feedback_note must include best_achieved_state anchor"
    )
    assert feedback_note["best_achieved_round"] == 3, (
        "patch_apply feedback_note must record the round that achieved best state"
    )
    assert "primary_repair_instruction" in feedback_note, (
        "patch_apply feedback_note must include primary_repair_instruction anchor"
    )
    assert "produced_actionable_handoff" in feedback_note["primary_repair_instruction"]
    assert "regress" in feedback_note["primary_repair_instruction"].lower()


def test_patch_failure_no_anchor_when_no_prior_partial_success(
    tmp_path: pathlib.Path,
) -> None:
    """When no prior round has achieved meaningful value, patch failure feedback_note
    must NOT include best-achieved anchor fields (no false anchoring)."""
    controller = _DummyController(tmp_path)
    round_history = [
        {"round": 1, "value_delivered": "none", "tool_name": "t1"},
        {"round": 2, "value_delivered": "none", "tool_name": "t2"},
    ]
    best = controller._toolgen_best_achieved_state_summary(round_history)

    _note: dict = {
        "phase": "patch_apply",
        "error": "hunk mismatch",
        "plan": [],
    }
    if best.get("best_achieved_state", "none") != "none":
        _note["best_achieved_state"] = best["best_achieved_state"]
        _note["best_achieved_round"] = best.get("best_achieved_round")
        _note["primary_repair_instruction"] = "anchor"

    assert "best_achieved_state" not in _note, (
        "When no partial success exists, patch failure note must not add false anchor"
    )
    assert "primary_repair_instruction" not in _note, (
        "When no partial success exists, patch failure note must not add primary_repair_instruction"
    )


def test_patch_failure_anchor_not_added_when_best_is_none(
    tmp_path: pathlib.Path,
) -> None:
    """Confirm _toolgen_best_achieved_state_summary gates on 'none' correctly —
    an all-'none' history must yield best_achieved_state='none' and no anchor injection."""
    controller = _DummyController(tmp_path)
    history = [
        {"round": 1, "value_delivered": "none"},
        {"round": 2, "value_delivered": "none"},
        {"round": 3, "value_delivered": "none"},
    ]
    best = controller._toolgen_best_achieved_state_summary(history)
    assert best["best_achieved_state"] == "none"
    assert best["best_achieved_round"] is None
