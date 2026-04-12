from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.offline_compare_family_candidate as offline_compare
import src.agents.instance.sage_agent_controller as sage_agent_controller_module
from src.agents.exceptions import AgentUnknownException
from src.agents.instance.sage_agent_controller import SAGEAgentController
from src.sage.candidate_compare import (
    baseline_satisfies_locked_family,
    load_archived_baseline_record,
    summarize_sample_semantic_metrics,
    write_archived_baseline_record,
)
from src.sage.family_policy_evolution import ENV_COMPARE_LOCK_FAMILY
from src.sage.policy_contracts import FamilyPolicyBundle
from src.typings import ChatHistory, ChatHistoryItem, Role


RUNTIME_SINGLE_ANCHOR_BINDING_ENV = "SAGE_RUNTIME_SINGLE_ANCHOR_ANSWER_BINDING"
RUNTIME_SINGLE_ANCHOR_TARGET_ENV = "SAGE_RUNTIME_SINGLE_ANCHOR_TARGET_SEMANTICS"
RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC_ENV = "SAGE_RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC"
RUNTIME_SINGLE_ANCHOR_CHAIN_LOW_TRUST_DYNAMIC_ENV = (
    "SAGE_RUNTIME_SINGLE_ANCHOR_CHAIN_LOW_TRUST_DYNAMIC"
)
RUNTIME_PRESERVE_CHAIN_QUERY_SHAPE_ON_REFRESH_ENV = (
    "SAGE_RUNTIME_PRESERVE_CHAIN_QUERY_SHAPE_ON_REFRESH"
)
RUNTIME_JOINED_COUNT_TARGET_ENV = "SAGE_RUNTIME_JOINED_COUNT_TARGET_BOUNDARY"
from src.sage.reusable_tool_families import select_reusable_tool


def _make_controller() -> SAGEAgentController:
    controller = object.__new__(SAGEAgentController)
    controller._emit_generated_tools_event = lambda payload: None
    return controller


def _make_inference_ready_controller() -> SAGEAgentController:
    controller = _make_controller()
    controller._manual_fallback_active_runs = set()
    controller._manual_fallback_agent = None
    controller._tool_invoked_in_last_inference = None
    controller._language_model = type("DummyLM", (), {"role_dict": {}})()
    controller._inference_config_dict = {}
    controller._kwargs = {}
    controller._build_query_tool_name = lambda task_question: "sage_test_tool"
    controller._split_task_question = lambda task_question: (task_question, [])
    controller._extract_answer_target_phrase = lambda question_text: ""
    controller._build_question_interpretation = lambda **kwargs: {
        "question_inputs": [],
        "preferred_scaffolds": [],
    }
    controller._extract_grounding_entities_from_question_interpretation = (
        lambda question_interpretation: []
    )
    controller._infer_domain_hints = lambda question_text: []
    controller._build_grounded_relation_candidates_with_dynamic_fallback = (
        lambda **kwargs: []
    )
    controller._refine_question_interpretation_with_grounding = (
        lambda **kwargs: kwargs["question_interpretation"]
    )
    controller._build_sage_grounding_card = lambda *args, **kwargs: ""
    controller._run_text_prompt = lambda **kwargs: "generate_tool"
    controller._parse_orchestrator_action = lambda raw_output: "generate_tool"
    return controller


def test_select_reusable_tool_respects_compare_locked_family(monkeypatch) -> None:
    monkeypatch.setenv(ENV_COMPARE_LOCK_FAMILY, "single_anchor_lookup")

    selection = select_reusable_tool(
        {
            "answer_mode": "entity",
            "query_shape": "single_anchor_chain_lookup",
            "shared_answer_variable": "answer",
            "projection": ["answer", "answer_name"],
            "relation_paths": [
                {
                    "relation": "people.person.parents",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "curated",
                }
            ],
        }
    )

    assert selection is None


def test_validate_query_plan_grounding_rejects_family_lock_escape(monkeypatch) -> None:
    controller = _make_controller()
    monkeypatch.setenv(ENV_COMPARE_LOCK_FAMILY, "single_anchor_lookup")
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_chain_lookup",
            "anchored_entities": [
                {
                    "surface": "Saxe-Coburg-Gotha",
                    "chosen_alias": "Saxe-Coburg-Gotha",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "royalty.kingdom.rulers",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                }
            ],
        },
        relation_grounding=[
            {
                "relation": "royalty.kingdom.rulers",
                "grounding_source": "curated",
            }
        ],
    )

    assert "family_compare_locked_query_shape:single_anchor_lookup:single_anchor_chain_lookup" in errors


def test_compare_family_candidate_reuses_archived_baseline(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "archive" / "single_anchor_lookup" / "family_locked_candidate_compare_full" / "2026-03-31" / "3.json"
    context = offline_compare.family_policy_harness._build_evaluation_cache_context(
        family_name="single_anchor_lookup",
        bundle_version="2026-03-31",
        sample_index="3",
        evaluation_mode="family_locked_candidate_compare_full",
        extra_env={ENV_COMPARE_LOCK_FAMILY: "single_anchor_lookup"},
    )
    run_summary = {
        "sample_index": "3",
        "sample_status": "completed",
        "evaluation_outcome": "correct",
        "finish_reason": "done",
        "run_dir": str(tmp_path / "baseline_run"),
        "dangerous_overreach_count": 0,
    }
    semantic_metrics = {
        "sample_index": "3",
        "sample_status": "completed",
        "evaluation_outcome": "correct",
        "correct_completed": 1,
        "wrong_completed": 0,
        "fail_closed": 0,
        "dangerous_overreach_count": 0,
        "answer_target_failure_count": 0,
        "anchor_failure_count": 0,
        "family_lock_violation_count": 0,
        "family_lock_violation_details": [],
        "semantic_failure_labels": [],
    }
    write_archived_baseline_record(
        archive_path,
        context=context,
        run_summary=run_summary,
        semantic_metrics=semantic_metrics,
    )
    reused = load_archived_baseline_record(archive_path, expected_context=context)
    assert reused is not None

    call_log: list[tuple[str, str]] = []

    def _fake_run_sample_with_policy(**kwargs):
        call_log.append(
            (
                str(kwargs["sample_index"]),
                str(kwargs.get("override_version") or ""),
            )
        )
        return {
            "sample_index": str(kwargs["sample_index"]),
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "finish_reason": "done",
            "run_dir": str(tmp_path / f"run_{kwargs['override_version']}"),
            "dangerous_overreach_count": 0,
        }

    monkeypatch.setattr(
        offline_compare.family_policy_harness,
        "_run_sample_with_policy",
        _fake_run_sample_with_policy,
    )

    summary = offline_compare.compare_family_candidate(
        family_name="single_anchor_lookup",
        sample_indices=["3"],
        store_path=tmp_path / "store",
        candidate_version="2026-03-31__cand0001",
        baseline_version="2026-03-31",
        baseline_archive_root=tmp_path / "archive",
    )

    assert summary["baseline_reuse_count"] == 1
    assert call_log == [("3", "2026-03-31__cand0001")]


def test_compare_family_candidate_passes_extra_env(tmp_path, monkeypatch) -> None:
    observed_contexts: list[dict[str, str]] = []

    def _fake_run_sample_with_policy(**kwargs):
        observed_contexts.append(dict(kwargs.get("extra_env") or {}))
        return {
            "sample_index": str(kwargs["sample_index"]),
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "finish_reason": "done",
            "run_dir": str(tmp_path / f"run_{kwargs['override_version']}"),
            "dangerous_overreach_count": 0,
        }

    monkeypatch.setattr(
        offline_compare.family_policy_harness,
        "_run_sample_with_policy",
        _fake_run_sample_with_policy,
    )

    summary = offline_compare.compare_family_candidate(
        family_name="single_anchor_lookup",
        sample_indices=["3"],
        store_path=tmp_path / "store",
        candidate_version="2026-03-31__cand0001",
        baseline_version="2026-03-31",
        extra_env={"SAGE_RUNTIME_SINGLE_ANCHOR_ANSWER_BINDING": "1"},
    )

    assert summary["extra_env"] == {
        ENV_COMPARE_LOCK_FAMILY: "single_anchor_lookup",
        "SAGE_RUNTIME_SINGLE_ANCHOR_ANSWER_BINDING": "1",
    }
    assert observed_contexts == [
        {
            ENV_COMPARE_LOCK_FAMILY: "single_anchor_lookup",
            "SAGE_RUNTIME_SINGLE_ANCHOR_ANSWER_BINDING": "1",
        },
        {
            ENV_COMPARE_LOCK_FAMILY: "single_anchor_lookup",
            "SAGE_RUNTIME_SINGLE_ANCHOR_ANSWER_BINDING": "1",
        },
    ]


def test_summarize_sample_semantic_metrics_falls_back_to_summary_typed_fields() -> None:
    metrics = summarize_sample_semantic_metrics(
        {
            "sample_index": "15",
            "sample_status": "timeout",
            "evaluation_outcome": "",
            "finish_reason": "timed_out",
            "sage_primary_failure_kind": "timeout",
            "sage_stop_reason": "runner_timeout",
            "sage_completion_state": "fail_closed",
            "sage_repair_attempt_count": 0,
        }
    )

    assert metrics["timeout_count"] == 1
    assert metrics["primary_failure_kind"] == "timeout"
    assert metrics["stop_reason"] == "runner_timeout"
    assert metrics["repair_attempt_count"] == 0


def test_summarize_sample_semantic_metrics_infers_timeout_without_typed_failure() -> None:
    metrics = summarize_sample_semantic_metrics(
        {
            "sample_index": "21",
            "sample_status": "timeout",
            "evaluation_outcome": "",
            "finish_reason": "timed_out",
            "timed_out": True,
        }
    )

    assert metrics["timeout_count"] == 1
    assert metrics["primary_failure_kind"] == "timeout"
    assert metrics["stop_reason"] == "runner_timeout"


def test_baseline_satisfies_locked_family_rejects_shape_mismatch() -> None:
    assert not baseline_satisfies_locked_family(
        {
            "family_lock_violation_count": 0,
            "semantic_failure_labels": [
                "sage_query_plan_invalid:family_compare_locked_query_shape:count_over_joined_set:count_over_direct_relation"
            ],
        }
    )
    assert baseline_satisfies_locked_family(
        {
            "family_lock_violation_count": 0,
            "semantic_failure_labels": ["accepted_completion"],
        }
    )


def test_compare_family_candidate_screens_unsatisfied_baseline(tmp_path, monkeypatch) -> None:
    call_log: list[tuple[str, str]] = []

    def _fake_run_sample_with_policy(**kwargs):
        sample_index = str(kwargs["sample_index"])
        version = str(kwargs.get("override_version") or "")
        call_log.append((sample_index, version))
        if version == "2026-03-31" and sample_index == "20":
            return {
                "sample_index": sample_index,
                "sample_status": "not_completed",
                "evaluation_outcome": "incorrect",
                "finish_reason": "sage_query_plan_invalid:family_compare_locked_query_shape:count_over_joined_set:count_over_direct_relation",
                "run_dir": str(tmp_path / f"run_{version}_{sample_index}"),
                "dangerous_overreach_count": 0,
            }
        return {
            "sample_index": sample_index,
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "finish_reason": "done",
            "run_dir": str(tmp_path / f"run_{version}_{sample_index}"),
            "dangerous_overreach_count": 0,
        }

    monkeypatch.setattr(
        offline_compare.family_policy_harness,
        "_run_sample_with_policy",
        _fake_run_sample_with_policy,
    )

    summary = offline_compare.compare_family_candidate(
        family_name="count_over_joined_set",
        sample_indices=["20", "21"],
        store_path=tmp_path / "store",
        candidate_version="2026-03-31__cand0001",
        baseline_version="2026-03-31",
        screen_unsatisfied_baseline=True,
    )

    assert summary["screened_out_sample_count"] == 1
    assert summary["screened_out_samples"][0]["sample_index"] == "20"
    assert summary["baseline"]["sample_count"] == 1
    assert summary["candidate"]["sample_count"] == 1
    assert ("20", "2026-03-31__cand0001") not in call_log


def test_inference_routes_sage_query_plan_invalid_to_manual_fallback() -> None:
    controller = _make_inference_ready_controller()
    controller._generate_sage_query_plan = lambda **kwargs: (_ for _ in ()).throw(
        ValueError("sage_query_plan_invalid:family_policy_single_anchor_requires_direct_path")
    )
    controller._tool_evolution_skip_manual_fallback_enabled = lambda: False
    controller._delegate_to_manual_solver = lambda **kwargs: ChatHistoryItem(
        role=Role.AGENT,
        content="manual fallback answer",
    )

    chat_history = ChatHistory(
        value=[ChatHistoryItem(role=Role.USER, content="Who influenced Quake 3 engine?")]
    )

    response = controller._inference(chat_history)

    assert response.content == "manual fallback answer"
    assert controller._manual_fallback_active_for_current_run()
    updated_user_turn = chat_history.get_item_deep_copy(-1).content or ""
    assert "SAGE tool advisory" in updated_user_turn
    assert "sage_query_plan_invalid:family_policy_single_anchor_requires_direct_path" in updated_user_turn


def test_inference_bypasses_manual_fallback_for_plan_invalid_when_skip_enabled() -> None:
    controller = _make_inference_ready_controller()
    controller._generate_sage_query_plan = lambda **kwargs: (_ for _ in ()).throw(
        ValueError("sage_query_plan_invalid:family_policy_single_anchor_requires_direct_path")
    )
    controller._tool_evolution_skip_manual_fallback_enabled = lambda: True

    chat_history = ChatHistory(
        value=[ChatHistoryItem(role=Role.USER, content="Who influenced Quake 3 engine?")]
    )

    try:
        controller._inference(chat_history)
        raise AssertionError("Expected AgentUnknownException")
    except AgentUnknownException as exc:
        assert (
            "sage_tool_failure_bypassed:sage_query_plan_invalid:family_policy_single_anchor_requires_direct_path"
            in str(exc)
        )
    assert not controller._manual_fallback_active_for_current_run()


def test_validate_query_plan_grounding_rejects_single_anchor_without_answer_binding(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_BINDING_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=("verify_projected_entity_matches_question_target",),
        )
        if family_name == "single_anchor_lookup"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "architecture.architect.architectural_style",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "architecture.architectural_style.architects",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "candidate_set",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                },
            ],
        },
        relation_grounding=[
            {
                "relation": "architecture.architect.architectural_style",
                "grounding_source": "curated",
            },
            {
                "relation": "architecture.architectural_style.architects",
                "grounding_source": "curated",
            },
        ],
    )

    assert "family_policy_single_anchor_requires_direct_path" in errors


def test_validate_query_plan_grounding_rejects_single_anchor_exploratory_target_drift(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_TARGET_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=(
                "verify_answer_target_semantics_not_just_executability",
            ),
        )
        if family_name == "single_anchor_lookup"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_lookup",
            "answer_target_phrase": "animals",
            "anchored_entities": [
                {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "people.person.pets",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "exploratory",
                }
            ],
        },
        relation_grounding=[
            {
                "relation": "biology.animal_owner.animals_owned",
                "direction": "forward",
                "from": "anchor",
                "to": "answer",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            }
        ],
    )

    assert (
        "family_policy_single_anchor_exploratory_answer_target_drift:animals"
        in errors
    )


def test_validate_query_plan_grounding_rejects_low_trust_dynamic_single_anchor_alias_repair(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=("reject_known_dangerous_overreach_patterns",),
        )
        if family_name == "single_anchor_lookup"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_lookup",
            "allow_exploratory_predicates": True,
            "anchored_entities": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "resolved_entity_id": "m.02qndhf",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "reason": "live probe repair selected alias 'J. Paul Reddam'",
                }
            ],
            "relation_paths": [
                {
                    "relation": "biology.animal_owner.animals_owned",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        relation_grounding=[
            {
                "relation": "biology.animal_owner.animals_owned",
                "direction": "forward",
                "from": "anchor",
                "to": "answer",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert "family_policy_single_anchor_low_trust_dynamic_alias_repair" in errors


def test_validate_query_plan_grounding_rejects_low_trust_dynamic_without_bundle_token(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=(),
        )
        if family_name == "single_anchor_lookup"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_lookup",
            "allow_exploratory_predicates": True,
            "anchored_entities": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "resolved_entity_id": "m.02qndhf",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "reason": "live probe repair selected alias 'J. Paul Reddam'",
                }
            ],
            "relation_paths": [
                {
                    "relation": "biology.animal_owner.animals_owned",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        relation_grounding=[
            {
                "relation": "biology.animal_owner.animals_owned",
                "direction": "forward",
                "from": "anchor",
                "to": "answer",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert "family_policy_single_anchor_low_trust_dynamic_alias_repair" in errors


def test_validate_query_plan_grounding_rejects_low_trust_dynamic_single_anchor_chain_alias_repair(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC_ENV, "1")
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_CHAIN_LOW_TRUST_DYNAMIC_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_chain_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=("reject_known_dangerous_overreach_patterns",),
        )
        if family_name == "single_anchor_chain_lookup"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_chain_lookup",
            "allow_exploratory_predicates": False,
            "shared_answer_variable": "answer",
            "candidate_set_variable": "owned",
            "anchored_entities": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "resolved_entity_id": "m.02qndhf",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "reason": "live probe repair selected alias 'J. Paul Reddam'",
                }
            ],
            "relation_paths": [
                {
                    "relation": "biology.animal_owner.animals_owned",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "owned",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        relation_grounding=[
            {
                "relation": "biology.animal_owner.animals_owned",
                "direction": "forward",
                "from": "anchor",
                "to": "owned",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert "family_policy_single_anchor_low_trust_dynamic_alias_repair" in errors


def test_validate_query_plan_grounding_allows_dynamic_single_anchor_without_alias_repair(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=("reject_known_dangerous_overreach_patterns",),
        )
        if family_name == "single_anchor_lookup"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "single_anchor_lookup",
            "allow_exploratory_predicates": False,
            "anchored_entities": [
                {
                    "surface": "National Wine Centre of Australia",
                    "chosen_alias": "National Wine Centre of Australia",
                    "resolved_entity_id": "m.03hd1z",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "National Wine Centre of Australia",
                    "chosen_alias": "National Wine Centre of Australia",
                    "reason": "explicit_entity:surface_constraint from question_inputs",
                }
            ],
            "relation_paths": [
                {
                    "relation": "education.educational_institution.parent_institution",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        relation_grounding=[
            {
                "relation": "education.educational_institution.parent_institution",
                "direction": "forward",
                "from": "anchor",
                "to": "answer",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert "family_policy_single_anchor_low_trust_dynamic_alias_repair" not in errors


def test_validate_sage_query_candidate_rejects_repaired_low_trust_dynamic_single_anchor(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_SINGLE_ANCHOR_LOW_TRUST_DYNAMIC_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="single_anchor_lookup",
            version="2026-03-31__cand_runtime",
            renderer_name="entity",
            validator_expectations=("reject_known_dangerous_overreach_patterns",),
        )
        if family_name == "single_anchor_lookup"
        else None,
    )
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    query_text = """
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?answer ?answer_name WHERE {
      fb:m.02qndhf fb:biology.animal_owner.animals_owned ?answer .
      OPTIONAL { ?answer fb:type.object.name ?answer_name . }
    }
    """

    errors = controller._validate_sage_query_candidate(
        raw_output=query_text,
        generated_code=f"""
query = '''{query_text}'''
sparql.setQuery(query)
""",
        query_text=query_text,
        query_texts=[query_text],
        query_plan={
            "query_shape": "single_anchor_lookup",
            "answer_mode": "entity",
            "allow_exploratory_predicates": True,
            "projection": ["answer", "answer_name"],
            "anchored_entities": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "resolved_entity_id": "m.02qndhf",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "paul reddam",
                    "chosen_alias": "J. Paul Reddam",
                    "reason": "live probe repair selected alias 'J. Paul Reddam'",
                }
            ],
            "relation_paths": [
                {
                    "relation": "biology.animal_owner.animals_owned",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
    )

    assert "family_policy_single_anchor_low_trust_dynamic_alias_repair" in errors


def test_preserve_chain_query_shape_on_refresh_for_chain_structured_plan(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_PRESERVE_CHAIN_QUERY_SHAPE_ON_REFRESH_ENV, "1")

    refreshed = controller._preserve_chain_query_shape_on_refresh(
        previous_query_plan={
            "query_shape": "single_anchor_chain_lookup",
            "relation_paths": [
                {
                    "relation": "architecture.architect.architectural_style",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "architecture.architectural_style.architects",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                },
            ],
        },
        refreshed_query_plan={
            "query_shape": "single_anchor_lookup",
            "shared_answer_variable": "answer",
            "candidate_set_variable": "style",
            "relation_paths": [
                {
                    "relation": "architecture.architect.architectural_style",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "architecture.architectural_style.architects",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                },
            ],
        },
    )

    assert refreshed["query_shape"] == "single_anchor_chain_lookup"


def test_validate_query_plan_grounding_rejects_joined_count_missing_target_boundary(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv(RUNTIME_JOINED_COUNT_TARGET_ENV, "1")
    monkeypatch.setattr(
        sage_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name="count_over_joined_set",
            version="2026-03-31__cand_runtime",
            renderer_name="count",
            validator_expectations=("verify_count_targets_requested_entity_set",),
        )
        if family_name == "count_over_joined_set"
        else None,
    )
    controller._relation_contract_match_details = lambda **kwargs: (True, "matched")
    controller._build_scaffold_signature = lambda query_plan: ""
    controller._relation_names_from_plan = lambda query_plan: []

    errors = controller._validate_query_plan_grounding(
        query_plan={
            "query_shape": "count_over_joined_set",
            "answer_mode": "count",
            "answer_target_phrase": "radio programs",
            "candidate_set_variable": "candidate_content",
            "count_set_variable": "count_candidate_content",
            "shared_answer_variable": "candidate_content",
            "anchored_entities": [
                {"surface": "Talk radio", "chosen_alias": "Talk radio", "role": "anchor_a"},
                {"surface": "Weekend Edition Sunday", "chosen_alias": "Weekend Edition Sunday", "role": "anchor_b"},
            ],
            "relation_paths": [
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "candidate_set",
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "broadcast.producer.produces",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "candidate_set",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                },
            ],
        },
        relation_grounding=[
            {
                "relation": "broadcast.content.producer",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.producer.produces",
                "grounding_source": "curated",
            },
        ],
    )

    assert "count_query_counts_wrong_variable:count_set_not_structurally_bound" in errors
    assert "family_policy_count_answer_target_not_preserved:radio programs" in errors
