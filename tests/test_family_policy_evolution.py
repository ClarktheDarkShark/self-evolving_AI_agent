from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pal.family_policy_evolution import (
    ENV_ENABLE_EVOLUTION,
    ENV_ENABLED_FAMILIES,
    ENV_STORE_PATH,
    build_family_policy_store,
    build_success_plan_archetype,
    classify_family_failure,
)
from src.pal.reusable_tool_families import (
    get_baseline_reusable_family_policy_bundles,
    get_reusable_family_policy_bundle,
)
import scripts.run_kg_family_policy_evolution as family_policy_harness
import scripts.pal_kg_batch_runner as kg_batch_runner
from scripts.run_kg_family_policy_evolution import _maybe_create_candidate_from_run_summary


def test_reusable_family_bundle_defaults_to_baseline_when_evolution_disabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv(ENV_ENABLE_EVOLUTION, raising=False)
    monkeypatch.delenv(ENV_ENABLED_FAMILIES, raising=False)
    monkeypatch.delenv(ENV_STORE_PATH, raising=False)

    bundle = get_reusable_family_policy_bundle("count_over_direct_relation")

    assert bundle is not None
    assert bundle.version == "2026-03-31"
    assert bundle.blocked_scaffold_signatures == ()


def test_family_policy_candidate_creation_preserves_active_version(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="count_over_direct_relation",
        scaffold_signature=(
            "count_over_direct_relation|candidate_set|biology.organism.diseases_transmitted"
        ),
        relation_names=["biology.organism.diseases_transmitted"],
        failure_reasons=["repairable_bad_count_set"],
        trigger_context={"sample_index": "9"},
    )

    assert candidate is not None
    assert store.get_active_version("count_over_direct_relation") == "2026-03-31"
    pending = store.get_pending_candidates("count_over_direct_relation")
    assert len(pending) == 1
    assert pending[0]["candidate_version"] == candidate.candidate_version
    assert "repair_policy" in pending[0]["fields_changed"]
    assert (
        "switch_to_joined_count_when_downstream_filter_or_projection_exists"
        in pending[0]["bundle"]["repair_policy"]
    )
    assert not pending[0]["bundle"]["blocked_scaffold_signatures"]


def test_wrong_trusted_completion_prefers_constructive_count_guidance(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="count_over_direct_relation",
        scaffold_signature="count_over_direct_relation|candidate_set|medicine.vector_of_disease.disease",
        relation_names=["medicine.vector_of_disease.disease"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "9"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("count_over_direct_relation")
    assert len(pending) == 1
    assert "validator_expectations" in pending[0]["fields_changed"]
    assert "repair_policy" in pending[0]["fields_changed"]
    assert (
        "verify_count_targets_requested_entity_set"
        in pending[0]["bundle"]["validator_expectations"]
    )
    assert (
        "switch_to_joined_count_when_downstream_filter_or_projection_exists"
        in pending[0]["bundle"]["repair_policy"]
    )


def test_family_policy_rejection_keeps_baseline_active(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )
    candidate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|type.object.type",
        relation_names=["type.object.type"],
        failure_reasons=["dangerous_overreach:ontology_dump"],
        trigger_context={"sample_index": "22"},
    )
    assert candidate is not None

    store.reject_candidate(
        "single_anchor_lookup",
        candidate_version=candidate.candidate_version,
        evaluation_results={"gate": "failed"},
        rejection_reason="held_out_regressed",
    )

    assert store.get_active_version("single_anchor_lookup") == "2026-03-31"
    family_payload = json.loads((tmp_path / "single_anchor_lookup.json").read_text())
    assert (
        family_payload["versions"][candidate.candidate_version]["promotion_decision"][
            "decision"
        ]
        == "rejected"
    )


def test_family_policy_promotion_switches_active_version(tmp_path, monkeypatch) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )
    candidate = store.create_candidate_update(
        family_name="multi_anchor_intersection",
        scaffold_signature="multi_anchor_intersection|answer|type.object.type",
        relation_names=["type.object.type"],
        failure_reasons=["dangerous_overreach:broad_type_expansion"],
        trigger_context={"sample_index": "28"},
    )
    assert candidate is not None

    store.promote_candidate(
        "multi_anchor_intersection",
        candidate_version=candidate.candidate_version,
        evaluation_results={"gate": "passed"},
        promotion_reason="all_regression_gates_passed",
    )

    monkeypatch.setenv(ENV_ENABLE_EVOLUTION, "1")
    monkeypatch.setenv(ENV_ENABLED_FAMILIES, "multi_anchor_intersection")
    monkeypatch.setenv(ENV_STORE_PATH, str(tmp_path))

    active_bundle = get_reusable_family_policy_bundle("multi_anchor_intersection")

    assert active_bundle is not None
    assert active_bundle.version == candidate.candidate_version
    assert (
        "dangerous_overreach:broad_type_expansion"
        in active_bundle.forbidden_overreach_patterns
    )


def test_promotion_preserves_trusted_success_bank_and_tool_evolution_context(tmp_path) -> None:
    family_name = "count_over_direct_relation"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["11"],
        source_version=active_version,
        evaluation_results={
            "run_summary": {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        },
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )
    store.set_tool_evolution_context(
        family_name,
        source_version=active_version,
        preferred_patterns=[{"pattern_signature": "success-1"}],
        avoid_patterns=[{"pattern_signature": "failure-1"}],
        last_signal={"signal_type": "trusted_success"},
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|candidate_set|cvg.game_version.publisher",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "20"},
    )
    assert candidate is not None

    store.promote_candidate(
        family_name,
        candidate_version=candidate.candidate_version,
        evaluation_results={"gate": "passed"},
        promotion_reason="all_regression_gates_passed",
    )

    payload = json.loads((tmp_path / f"{family_name}.json").read_text(encoding="utf-8"))
    metadata = payload["trusted_success_bank_metadata"]
    assert metadata["source_version"] == candidate.candidate_version
    assert metadata["evaluation_context"]["source_version"] == candidate.candidate_version
    context = payload["tool_evolution_context"]
    assert context["source_version"] == candidate.candidate_version
    assert payload["trusted_success_bank"] == ["11"]


def test_inline_promotion_gate_reuses_success_bank_with_extra_context_fields(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_direct_relation"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )
    active_version = store.get_active_version(family_name)
    bank_context = family_policy_harness._build_success_bank_context(
        family_name=family_name,
        active_version=active_version,
    )
    bank_context["success_plan_archetypes"] = [{"pattern_signature": "success-241"}]
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["241"],
        source_version=active_version,
        evaluation_context=bank_context,
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|disease|medicine.infectious_disease.vector",
        relation_names=["medicine.infectious_disease.vector"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "15"},
    )
    assert candidate is not None

    def _fake_run_sample_with_policy(**kwargs):
        label = str(kwargs.get("label") or "")
        sample_index = str(kwargs.get("sample_index") or "")
        if label.endswith("_candidate_trigger"):
            return {
                "sample_index": sample_index,
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
                "evaluation_cache_hit": False,
            }
        return {
            "sample_index": sample_index,
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "dangerous_overreach_count": 0,
            "evaluation_cache_hit": False,
        }

    monkeypatch.setattr(
        family_policy_harness,
        "_run_sample_with_policy",
        _fake_run_sample_with_policy,
    )

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=True,
        store_path=tmp_path,
        label_prefix="inline_test",
        trigger_baseline_summary={
            "sample_index": "15",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
        },
        parent_output_dir=None,
    )

    assert evaluation["prior_success"]["trusted_success_bank_reused"] is True
    assert evaluation["evaluation_stats"]["prior_success_evaluated"] == 1
    assert evaluation["promotion_gate"]["gate_checks"]["regression_guard_available"] is True


def test_harness_synthesizes_candidate_from_wrong_completed_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "pal_query_artifacts").mkdir(parents=True)
    (run_dir / "generated_tools.log").write_text(
        json.dumps(
            {
                "event": "pal_attempt_decision_finalized",
                "sample_index": "2",
                "selected_family": "count_over_direct_relation",
                "family_bundle_version": "2026-03-31",
                "tool_name": "pal_sparql_query_tool_demo",
                "dangerous_overreach": False,
                "dangerous_overreach_reasons": [],
                "materialization_denial_reasons": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "pal_query_artifacts" / "pal_sparql_query_tool_demo.plan.json").write_text(
        json.dumps(
            {
                "query_shape": "count_over_direct_relation",
                "projection_role": "species_set",
                "relation_paths": [
                    {
                        "relation": "fictional_universe.fictional_universe.species",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    event = _maybe_create_candidate_from_run_summary(
        family_name="count_over_direct_relation",
        run_summary={
            "sample_index": "2",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(run_dir),
        },
        store_path=tmp_path / "store",
    )

    assert event is not None
    assert event["event"] == "family_policy_candidate_synthesized_from_run"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    assert store.get_active_version("count_over_direct_relation") == "2026-03-31"
    pending = store.get_pending_candidates("count_over_direct_relation")
    assert len(pending) == 1
    assert pending[0]["trigger_context"]["sample_index"] == "2"


def test_harness_synthesizes_candidate_from_matching_sample_decision_only(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "pal_query_artifacts").mkdir(parents=True)
    (run_dir / "generated_tools.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "pal_attempt_decision_finalized",
                        "sample_index": "2",
                        "selected_family": "count_over_direct_relation",
                        "family_bundle_version": "2026-03-31",
                        "tool_name": "pal_sparql_query_tool_sample_2",
                        "dangerous_overreach": False,
                        "dangerous_overreach_reasons": [],
                        "materialization_denial_reasons": [],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_attempt_decision_finalized",
                        "sample_index": "11",
                        "selected_family": "count_over_direct_relation",
                        "family_bundle_version": "2026-03-31",
                        "tool_name": "pal_sparql_query_tool_sample_11",
                        "dangerous_overreach": False,
                        "dangerous_overreach_reasons": [],
                        "materialization_denial_reasons": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "pal_query_artifacts" / "pal_sparql_query_tool_sample_2.plan.json").write_text(
        json.dumps(
            {
                "query_shape": "count_over_direct_relation",
                "projection_role": "species_set",
                "relation_paths": [
                    {
                        "relation": "fictional_universe.fictional_universe.species",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "pal_query_artifacts" / "pal_sparql_query_tool_sample_11.plan.json").write_text(
        json.dumps(
            {
                "query_shape": "count_over_direct_relation",
                "projection_role": "answer",
                "relation_paths": [
                    {
                        "relation": "people.person.profession",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _maybe_create_candidate_from_run_summary(
        family_name="count_over_direct_relation",
        run_summary={
            "sample_index": "2",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(run_dir),
        },
        store_path=tmp_path / "store",
    )

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    pending = store.get_pending_candidates("count_over_direct_relation")
    assert len(pending) == 1
    repair_policy = tuple(pending[0]["bundle"]["repair_policy"])
    assert (
        "switch_to_joined_count_when_downstream_filter_or_projection_exists"
        in repair_policy
    )
    blocked = tuple(pending[0]["bundle"]["blocked_scaffold_signatures"])
    assert not blocked


def test_tool_evolution_context_round_trips_patterns(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )
    success_pattern = build_success_plan_archetype(
        {
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "resolved_entity_id": "m.02h66l4",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "reason": "explicit_entity:attribute_value.profession",
                }
            ],
            "join_structure": {"type": "count", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
        }
    )
    store.set_tool_evolution_context(
        "count_over_direct_relation",
        source_version="2026-03-31",
        preferred_patterns=[success_pattern],
        avoid_patterns=[{"pattern_signature": "count-over-vector"}],
        last_signal={"signal_type": "clean_failure"},
    )

    context = store.get_tool_evolution_context("count_over_direct_relation")
    assert context["source_version"] == "2026-03-31"
    assert context["preferred_patterns"][0]["relation_role_skeleton"] == [
        "anchor->count_set:dynamic_probe"
    ]
    assert context["avoid_patterns"][0]["pattern_signature"] == "count-over-vector"


def test_classify_wrong_trusted_count_failure_as_weak_applicability_boundary() -> None:
    failure_class = classify_family_failure(
        family_name="count_over_direct_relation",
        sample_status="completed",
        evaluation_outcome="incorrect",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=("trusted_incorrect_completion",),
        dangerous_overreach=False,
    )

    assert failure_class == "weak_applicability_boundary"


def test_classify_repairable_count_failure_without_hardcoded_unknown_error_path() -> None:
    failure_class = classify_family_failure(
        family_name="count_over_joined_set",
        sample_status="",
        evaluation_outcome="",
        relation_names=["music.recording.artist"],
        failure_reasons=(
            "pal_query_not_accepted:repairable_bad_count_set",
            "count_answer_target_unenforced:artist",
        ),
        dangerous_overreach=False,
    )

    assert failure_class == "weak_applicability_boundary"


def test_classify_repairable_anchor_grounding_failure_as_bad_routing() -> None:
    failure_class = classify_family_failure(
        family_name="single_anchor_lookup",
        sample_status="",
        evaluation_outcome="",
        relation_names=["location.country.languages_spoken"],
        failure_reasons=("pal_query_not_accepted:repairable_weak_grounding",),
        dangerous_overreach=False,
    )

    assert failure_class == "bad_routing"


def test_run_sample_with_policy_reuses_cache_only_for_identical_context(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(family_policy_harness, "_policy_fingerprint", lambda: "policy-a")
    monkeypatch.setattr(
        family_policy_harness,
        "_execution_environment_fingerprint",
        lambda: "exec-a",
    )

    def fake_run_sample(
        *,
        sample_index: str,
        label: str,
        parent_output_dir=None,
    ) -> dict[str, str]:
        calls.append((sample_index, label))
        run_dir = tmp_path / f"run_{len(calls)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "sample_index": sample_index,
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "run_dir": str(run_dir),
        }

    monkeypatch.setattr(family_policy_harness, "run_sample", fake_run_sample)

    first = family_policy_harness._run_sample_with_policy(
        sample_index="11",
        label="cache_probe",
        family_name="count_over_direct_relation",
        store_path=tmp_path / "store",
        promotion_enabled=False,
        override_version="2026-03-31",
    )
    second = family_policy_harness._run_sample_with_policy(
        sample_index="11",
        label="cache_probe",
        family_name="count_over_direct_relation",
        store_path=tmp_path / "store",
        promotion_enabled=False,
        override_version="2026-03-31",
    )

    assert first["evaluation_cache_hit"] is False
    assert second["evaluation_cache_hit"] is True
    assert len(calls) == 1

    monkeypatch.setattr(family_policy_harness, "_policy_fingerprint", lambda: "policy-b")

    third = family_policy_harness._run_sample_with_policy(
        sample_index="11",
        label="cache_probe",
        family_name="count_over_direct_relation",
        store_path=tmp_path / "store",
        promotion_enabled=False,
        override_version="2026-03-31",
    )

    assert third["evaluation_cache_hit"] is False
    assert len(calls) == 2


def test_batch_runner_places_child_run_under_parent_output_dir(
    tmp_path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    class DummyCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_subprocess_run(command, **kwargs):
        commands.append(list(command))
        run_dir = tmp_path / "parent" / "pal_batch_demo_2"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "task_outcomes.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "sample_index": "2",
                            "completed": False,
                            "correct": False,
                            "outcome": "incorrect",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return DummyCompleted()

    monkeypatch.setattr(kg_batch_runner.subprocess, "run", fake_subprocess_run)

    summary = kg_batch_runner.run_sample(
        sample_index="2",
        label="demo",
        parent_output_dir=tmp_path / "parent",
    )

    assert commands
    assert "m._run_one(" in commands[0][2]
    assert str((tmp_path / "parent").resolve()) in commands[0][2]
    assert summary["run_dir"] == str(tmp_path / "parent" / "pal_batch_demo_2")


def test_evaluate_candidate_fails_fast_when_trigger_not_improved(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_direct_relation"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|candidate_set|cvg.game_version.publisher",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "20"},
    )
    assert candidate is not None
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["11"],
        source_version=active_version,
        evaluation_results={
            "run_summary": {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        },
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )

    calls: list[tuple[str, str]] = []

    def fake_run_sample_with_policy(**kwargs):
        calls.append(
            (
                str(kwargs["sample_index"]),
                str(kwargs.get("override_version") or ""),
            )
        )
        return {
            "sample_index": str(kwargs["sample_index"]),
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
            "evaluation_cache_hit": False,
        }

    monkeypatch.setattr(family_policy_harness, "_run_sample_with_policy", fake_run_sample_with_policy)

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=False,
        store_path=tmp_path / "store",
        label_prefix="trigger_fail_fast",
        trigger_baseline_summary={
            "sample_index": "20",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(tmp_path / "baseline_run"),
        },
    )

    assert evaluation["gate_stage"] == "inline"
    assert evaluation["evaluation_stats"]["prior_success_evaluated"] == 0
    assert "trigger_failure_not_improved" in evaluation["promotion_gate"]["reasons"]
    assert calls == [("20", candidate.candidate_version)]


def test_evaluate_candidate_rejects_when_trigger_not_improved_with_regression_guard(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_direct_relation"
    store_path = tmp_path / "store"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|candidate_set|cvg.game_version.publisher",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "20"},
    )
    assert candidate is not None
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["11"],
        source_version=active_version,
        evaluation_results={
            "run_summary": {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        },
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )

    calls: list[tuple[str, str]] = []

    def fake_run_sample_with_policy(**kwargs):
        calls.append(
            (
                str(kwargs["sample_index"]),
                str(kwargs.get("override_version") or ""),
            )
        )
        return {
            "sample_index": str(kwargs["sample_index"]),
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
            "evaluation_cache_hit": False,
        }

    monkeypatch.setattr(family_policy_harness, "_run_sample_with_policy", fake_run_sample_with_policy)

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=True,
        store_path=store_path,
        label_prefix="trigger_rejects_with_guard",
        trigger_baseline_summary={
            "sample_index": "20",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(tmp_path / "baseline_run"),
        },
    )

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    versions = store._load_family_payload(family_name)["versions"]
    assert evaluation["promotion_gate"]["promote"] is False
    assert "trigger_failure_not_improved" in evaluation["promotion_gate"]["reasons"]
    assert "awaiting_prior_trusted_success" not in evaluation["promotion_gate"]["reasons"]
    assert versions[candidate.candidate_version]["status"] == "rejected"
    assert calls == [("20", candidate.candidate_version)]


def test_evaluate_candidate_fails_fast_on_prior_success_regression(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_direct_relation"
    store_path = tmp_path / "store"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|candidate_set|cvg.game_version.publisher",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "20"},
    )
    assert candidate is not None
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["11", "241"],
        source_version=active_version,
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )

    calls: list[tuple[str, str]] = []

    def fake_run_sample_with_policy(**kwargs):
        sample_index = str(kwargs["sample_index"])
        version = str(kwargs.get("override_version") or "")
        calls.append((sample_index, version))
        if sample_index == "20" and version == candidate.candidate_version:
            return {
                "sample_index": sample_index,
                "sample_status": "agent_unknown_error",
                "evaluation_outcome": "incorrect",
                "dangerous_overreach_count": 0,
                "evaluation_cache_hit": False,
            }
        if sample_index == "11" and version == active_version:
            return {
                "sample_index": sample_index,
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
                "evaluation_cache_hit": False,
            }
        if sample_index == "11" and version == candidate.candidate_version:
            return {
                "sample_index": sample_index,
                "sample_status": "completed",
                "evaluation_outcome": "incorrect",
                "dangerous_overreach_count": 0,
                "evaluation_cache_hit": False,
            }
        raise AssertionError(f"unexpected evaluation request:{sample_index}:{version}")

    monkeypatch.setattr(family_policy_harness, "_run_sample_with_policy", fake_run_sample_with_policy)

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=False,
        store_path=store_path,
        label_prefix="prior_regression",
        trigger_baseline_summary={
            "sample_index": "20",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(tmp_path / "baseline_run"),
        },
    )

    assert evaluation["gate_stage"] == "inline"
    assert evaluation["promotion_gate"]["promote"] is False
    assert "prior_success_regressed" in evaluation["promotion_gate"]["reasons"]
    assert evaluation["evaluation_stats"]["prior_success_evaluated"] == 1
    assert calls == [
        ("20", candidate.candidate_version),
        ("11", active_version),
        ("11", candidate.candidate_version),
    ]


def test_inline_promotion_gate_rejects_invalid_evaluations() -> None:
    gate = family_policy_harness._evaluate_inline_promotion_gate(
        trigger_baseline={
            "sample_index": "9",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
        },
        trigger_candidate={
            "sample_index": "9",
            "sample_status": "",
            "evaluation_outcome": "",
            "returncode": 1,
        },
        prior_success_baseline=[
            {
                "sample_index": "11",
                "sample_status": "",
                "evaluation_outcome": "",
                "returncode": 1,
            }
        ],
        prior_success_candidate=[
            {
                "sample_index": "11",
                "sample_status": "",
                "evaluation_outcome": "",
                "returncode": 1,
            }
        ],
    )

    assert gate["promote"] is False
    assert gate["gate_checks"]["trigger_candidate_valid"] is False
    assert gate["gate_checks"]["prior_success_evaluation_valid"] is False
    assert "trigger_evaluation_invalid" in gate["reasons"]
    assert "prior_success_evaluation_invalid" in gate["reasons"]


def test_evaluate_candidate_promotes_when_trigger_and_prior_success_pass(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_direct_relation"
    store_path = tmp_path / "store"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|candidate_set|cvg.game_version.publisher",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "20"},
    )
    assert candidate is not None
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["11"],
        source_version=active_version,
        evaluation_results={
            "run_summary": {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        },
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )

    calls: list[tuple[str, str]] = []

    def fake_run_sample_with_policy(**kwargs):
        sample_index = str(kwargs["sample_index"])
        version = str(kwargs.get("override_version") or "")
        calls.append((sample_index, version))
        table = {
            ("20", candidate.candidate_version): ("agent_unknown_error", "incorrect"),
            ("11", active_version): ("completed", "correct"),
            ("11", candidate.candidate_version): ("completed", "correct"),
        }
        status, outcome = table[(sample_index, version)]
        return {
            "sample_index": sample_index,
            "sample_status": status,
            "evaluation_outcome": outcome,
            "dangerous_overreach_count": 0,
            "answer_target_failure_count": 0,
            "anchor_failure_count": 0,
            "wrong_executable_count": int(status == "completed" and outcome != "correct"),
            "semantic_quality_score": 3 if outcome == "correct" else 1,
            "evaluation_cache_hit": False,
        }

    monkeypatch.setattr(family_policy_harness, "_run_sample_with_policy", fake_run_sample_with_policy)

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=True,
        store_path=store_path,
        label_prefix="inline_pass",
        trigger_baseline_summary={
            "sample_index": "20",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(tmp_path / "baseline_run"),
        },
    )

    assert evaluation["gate_stage"] == "inline"
    assert evaluation["evaluation_stats"]["prior_success_evaluated"] == 1
    assert evaluation["evaluation_stats"]["prior_success_baseline_reused"] is True
    assert evaluation["promotion_gate"]["promote"] is True
    assert store.get_active_version(family_name) == candidate.candidate_version
    assert calls == [
        ("20", candidate.candidate_version),
        ("11", candidate.candidate_version),
    ]


def test_evaluate_candidate_non_regression_mode_replays_prior_success_without_trigger_win(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_joined_set"
    store_path = tmp_path / "store"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=store_path,
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_joined_set|disease|medicine.infectious_disease.vector",
        relation_names=["medicine.infectious_disease.vector"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "15"},
    )
    assert candidate is not None
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["11"],
        source_version=active_version,
        evaluation_results={
            "run_summary": {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        },
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )
    monkeypatch.setenv("PAL_FAMILY_POLICY_PROMOTION_GATE_MODE", "non_regression")

    calls: list[tuple[str, str]] = []

    def fake_run_sample_with_policy(**kwargs):
        sample_index = str(kwargs["sample_index"])
        version = str(kwargs.get("override_version") or "")
        calls.append((sample_index, version))
        table = {
            ("15", candidate.candidate_version): ("agent_unknown_error", "incorrect"),
            ("11", active_version): ("completed", "correct"),
            ("11", candidate.candidate_version): ("completed", "correct"),
        }
        status, outcome = table[(sample_index, version)]
        return {
            "sample_index": sample_index,
            "sample_status": status,
            "evaluation_outcome": outcome,
            "dangerous_overreach_count": 0,
            "answer_target_failure_count": 0,
            "anchor_failure_count": 0,
            "wrong_executable_count": int(status == "completed" and outcome != "correct"),
            "semantic_quality_score": 3 if outcome == "correct" else 1,
            "evaluation_cache_hit": False,
        }

    monkeypatch.setattr(family_policy_harness, "_run_sample_with_policy", fake_run_sample_with_policy)

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=True,
        store_path=store_path,
        label_prefix="non_regression_replay",
        trigger_baseline_summary={
            "sample_index": "15",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "run_dir": str(tmp_path / "baseline_run"),
        },
    )

    assert evaluation["promotion_gate"]["gate_mode"] == "non_regression"
    assert evaluation["promotion_gate"]["gate_checks"]["trigger_improved"] is False
    assert evaluation["promotion_gate"]["gate_checks"]["trigger_not_regressed"] is True
    assert evaluation["evaluation_stats"]["prior_success_evaluated"] == 1
    assert evaluation["evaluation_stats"]["prior_success_baseline_reused"] is True
    assert evaluation["prior_success"]["trusted_success_bank_reused"] is True
    assert evaluation["prior_success"]["baseline_reused_from_bank_metadata"] is True
    assert evaluation["promotion_gate"]["promote"] is True
    assert calls == [
        ("15", candidate.candidate_version),
        ("11", candidate.candidate_version),
    ]


def test_inline_promotion_gate_soft_improvement_mode_rewards_safer_failure() -> None:
    gate = family_policy_harness._evaluate_inline_promotion_gate(
        trigger_baseline={
            "sample_index": "20",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 1,
        },
        trigger_candidate={
            "sample_index": "20",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
        },
        prior_success_baseline=[
            {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        ],
        prior_success_candidate=[
            {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        ],
    )

    assert gate["gate_checks"]["trigger_improved"] is False
    assert gate["gate_checks"]["trigger_soft_improved"] is True
    assert gate["gate_checks"]["trigger_gate_passed"] is False
    assert gate["promote"] is False


def test_inline_promotion_gate_soft_improvement_mode_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PAL_FAMILY_POLICY_PROMOTION_GATE_MODE", "soft_improvement")
    gate = family_policy_harness._evaluate_inline_promotion_gate(
        trigger_baseline={
            "sample_index": "20",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 1,
        },
        trigger_candidate={
            "sample_index": "20",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
        },
        prior_success_baseline=[
            {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        ],
        prior_success_candidate=[
            {
                "sample_index": "11",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "dangerous_overreach_count": 0,
            }
        ],
    )

    assert gate["gate_mode"] == "soft_improvement"
    assert gate["gate_checks"]["trigger_soft_improved"] is True
    assert gate["gate_checks"]["trigger_gate_passed"] is True
    assert gate["promote"] is True


def test_evaluate_candidate_stays_pending_until_prior_success_exists(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_direct_relation"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_direct_relation|candidate_set|cvg.game_version.publisher",
        relation_names=["cvg.game_version.publisher"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "20"},
    )
    assert candidate is not None
    calls: list[tuple[str, str]] = []

    def fake_run_sample_with_policy(**kwargs):
        calls.append(
            (
                str(kwargs["sample_index"]),
                str(kwargs.get("override_version") or ""),
            )
        )
        return {
            "sample_index": str(kwargs["sample_index"]),
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
            "evaluation_cache_hit": False,
        }

    monkeypatch.setattr(family_policy_harness, "_run_sample_with_policy", fake_run_sample_with_policy)

    evaluation = family_policy_harness._evaluate_candidate(
        family_name=family_name,
        candidate_version=candidate.candidate_version,
        promote=True,
        store_path=tmp_path / "store",
        label_prefix="awaiting_prior_success",
        trigger_baseline_summary={
            "sample_index": "20",
            "sample_status": "completed",
            "evaluation_outcome": "incorrect",
            "run_dir": str(tmp_path / "baseline_run"),
        },
    )

    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    assert evaluation["promotion_gate"]["promote"] is False
    assert "awaiting_prior_trusted_success" in evaluation["promotion_gate"]["reasons"]
    assert evaluation["evaluation_stats"]["executed_runs"] == 0
    assert calls == []
    assert store.get_active_version(family_name) == "2026-03-31"
    pending = store.get_pending_candidates(family_name)
    assert pending and pending[0]["candidate_version"] == candidate.candidate_version


def test_candidate_update_can_forbid_relation_family_for_validator_miss(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|type.object.type",
        relation_names=["type.object.type"],
        failure_reasons=["dangerous_overreach:ontology_dump"],
        failure_class="validator_miss",
        trigger_context={"sample_index": "22"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("single_anchor_lookup")
    assert "forbidden_relation_families" in pending[0]["fields_changed"]


def test_candidate_update_does_not_ban_non_type_relation_for_semantic_validator_miss(
    tmp_path,
) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="count_over_joined_set",
        scaffold_signature=(
            "count_over_joined_set|candidate|people.person.profession|"
            "people.profession.people_with_this_profession"
        ),
        relation_names=[
            "people.profession.people_with_this_profession",
            "people.person.profession",
        ],
        failure_reasons=["dangerous_overreach:weak_count_semantics"],
        failure_class="validator_miss",
        trigger_context={"sample_index": "11"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("count_over_joined_set")
    assert pending
    bundle = pending[0]["bundle"]
    assert (
        "dangerous_overreach:weak_count_semantics"
        in bundle["forbidden_overreach_patterns"]
    )
    assert bundle["forbidden_relation_families"] == []


def test_candidate_update_can_tighten_validator_and_repair_policy(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="multi_anchor_intersection",
        scaffold_signature="multi_anchor_intersection|answer|type.object.type",
        relation_names=["type.object.type"],
        failure_reasons=["trusted_incorrect_completion"],
        failure_class="bad_fallback_ordering",
        trigger_context={"sample_index": "6"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("multi_anchor_intersection")
    assert "repair_policy" in pending[0]["fields_changed"]

