from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pal.family_policy_evolution import (
    ENV_DEDUP_SIGNATURE,
    ENV_ENABLE_EVOLUTION,
    ENV_ENABLED_FAMILIES,
    ENV_STORE_PATH,
    ENV_STRICT_UPDATE_MAPPING,
    ENV_STRUCTURED_SUCCESS_FEATURES,
    build_candidate_signature,
    build_family_policy_store,
    build_success_plan_archetype,
    bundle_from_dict,
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
    assert pending[0]["fields_changed"] == ["validator_expectations"]
    assert pending[0]["bundle"]["repair_policy"] == [
        "repair direct count relation family before escalating to joined-count family"
    ]
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
    assert pending[0]["fields_changed"] == ["validator_expectations"]
    assert (
        "verify_count_targets_requested_entity_set"
        in pending[0]["bundle"]["validator_expectations"]
    )
    assert pending[0]["bundle"]["repair_policy"] == [
        "repair direct count relation family before escalating to joined-count family"
    ]


def test_strict_update_mapping_limits_joined_count_to_validator_constraints(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="count_over_joined_set",
        scaffold_signature="count_over_joined_set|candidate_set|biology.animal_breed.temperament",
        relation_names=["biology.animal_breed.temperament"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "8"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("count_over_joined_set")
    assert len(pending) == 1
    assert pending[0]["fields_changed"] == ["validator_expectations"]
    assert pending[0]["bundle"]["repair_policy"] == list(
        get_baseline_reusable_family_policy_bundles()["count_over_joined_set"].repair_policy
    )


def test_candidate_signature_dedup_skips_rejected_duplicate_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_DEDUP_SIGNATURE, "1")
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    first = store.create_candidate_update(
        family_name="count_over_joined_set",
        scaffold_signature="count_over_joined_set|candidate_set|biology.animal_breed.temperament",
        relation_names=["biology.animal_breed.temperament"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "8"},
    )
    assert first is not None
    store.reject_candidate(
        "count_over_joined_set",
        candidate_version=first.candidate_version,
        evaluation_results={"gate": "failed"},
        rejection_reason="smoke_failed",
    )

    duplicate = store.create_candidate_update(
        family_name="count_over_joined_set",
        scaffold_signature="count_over_joined_set|candidate_set|biology.animal_breed.country_of_origin",
        relation_names=["biology.animal_breed.country_of_origin"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "14"},
    )

    assert duplicate is None
    payload = json.loads((tmp_path / "count_over_joined_set.json").read_text())
    rejected_versions = [
        version_name
        for version_name, version_payload in payload["versions"].items()
        if version_payload["status"] == "rejected"
    ]
    assert rejected_versions == [first.candidate_version]
    signature = payload["versions"][first.candidate_version]["candidate_signature"]
    assert signature == build_candidate_signature(
        bundle=store.get_bundle("count_over_joined_set", version=first.candidate_version),
        fields_changed=["validator_expectations"],
    )


def test_strict_update_mapping_keeps_single_anchor_validator_constraints(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|royalty.kingdom.monarchs",
        relation_names=["royalty.kingdom.monarchs"],
        failure_reasons=[
            "pal_query_not_accepted:repairable_anchor_not_found",
            "anchor_not_found:'Saxe-Coburg-Gotha'",
        ],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "3"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("single_anchor_lookup")
    assert len(pending) == 1
    assert pending[0]["fields_changed"] == ["validator_expectations"]
    assert (
        "verify_projected_entity_matches_question_target"
        in pending[0]["bundle"]["validator_expectations"]
    )
    assert pending[0]["bundle"]["repair_policy"] == list(
        get_baseline_reusable_family_policy_bundles()["single_anchor_lookup"].repair_policy
    )


def test_strict_update_mapping_rejects_nonreusable_single_anchor_boundary_failure(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|people.person.pets",
        relation_names=["people.person.pets"],
        failure_reasons=["repairable_grounded_empty_result"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "18"},
    )

    assert candidate is None
    assert store.get_pending_candidates("single_anchor_lookup") == []


def test_strict_update_mapping_rejects_nonsemantic_single_anchor_validator_miss(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|people.person.parents",
        relation_names=["people.person.parents"],
        failure_reasons=["validator_missing:unrelated_contract_gap"],
        failure_class="validator_miss",
        trigger_context={"sample_index": "x1"},
    )

    assert candidate is None
    assert store.get_pending_candidates("single_anchor_lookup") == []


def test_strict_update_mapping_keeps_single_anchor_validator_miss_for_weak_entity_semantics(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    candidate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|digicams.digital_camera.format",
        relation_names=["digicams.digital_camera.format"],
        failure_reasons=[
            "entity_answer_target_unenforced:format?",
            "dangerous_overreach:weak_entity_semantics",
        ],
        failure_class="validator_miss",
        trigger_context={"sample_index": "19"},
    )

    assert candidate is not None
    pending = store.get_pending_candidates("single_anchor_lookup")
    assert len(pending) == 1
    assert pending[0]["fields_changed"] == ["blocked_scaffold_signatures"]
    assert (
        "reject_known_dangerous_overreach_patterns"
        not in pending[0]["bundle"]["validator_expectations"]
    )
    assert pending[0]["bundle"]["forbidden_overreach_patterns"] == []


def test_semantic_signature_dedup_skips_wording_only_single_anchor_variant(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_DEDUP_SIGNATURE, "1")
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    first = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|royalty.kingdom.monarchs",
        relation_names=["royalty.kingdom.monarchs"],
        failure_reasons=[
            "pal_query_not_accepted:repairable_anchor_not_found",
            "anchor_not_found:'Saxe-Coburg-Gotha'",
        ],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "3"},
    )

    assert first is not None
    store.reject_candidate(
        "single_anchor_lookup",
        candidate_version=first.candidate_version,
        evaluation_results={"gate": "failed"},
        rejection_reason="smoke_failed",
    )

    payload_path = tmp_path / "single_anchor_lookup.json"
    payload = json.loads(payload_path.read_text())
    version_payload = payload["versions"][first.candidate_version]
    mutated_bundle = dict(version_payload["bundle"])
    mutated_bundle["repair_policy"] = list(mutated_bundle.get("repair_policy") or []) + [
        "wording_only_variant",
    ]
    version_payload["bundle"] = mutated_bundle
    version_payload["candidate_signature"] = build_candidate_signature(
        bundle=bundle_from_dict(mutated_bundle),
        fields_changed=version_payload.get("fields_changed") or [],
    )
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    duplicate = store.create_candidate_update(
        family_name="single_anchor_lookup",
        scaffold_signature="single_anchor_lookup|answer|royalty.kingdom.rulers",
        relation_names=["royalty.kingdom.rulers"],
        failure_reasons=[
            "pal_query_not_accepted:repairable_anchor_not_found",
            "anchor_not_found:'Saxe-Coburg-Gotha'",
        ],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "3b"},
    )

    assert duplicate is None


def test_exact_duplicate_bundle_is_skipped_even_without_signature_flag(
    tmp_path,
) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )

    first = store.create_candidate_update(
        family_name="count_over_joined_set",
        scaffold_signature="count_over_joined_set|candidate_set|biology.animal_breed.temperament",
        relation_names=["biology.animal_breed.temperament"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "8"},
    )
    assert first is not None
    store.reject_candidate(
        "count_over_joined_set",
        candidate_version=first.candidate_version,
        evaluation_results={"gate": "failed"},
        rejection_reason="smoke_failed",
    )

    duplicate = store.create_candidate_update(
        family_name="count_over_joined_set",
        scaffold_signature="count_over_joined_set|candidate_set|biology.animal_breed.country_of_origin",
        relation_names=["biology.animal_breed.country_of_origin"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "14"},
    )

    assert duplicate is None


def test_build_success_plan_archetype_can_emit_structured_success_features(
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRUCTURED_SUCCESS_FEATURES, "1")

    archetype = build_success_plan_archetype(
        {
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "answer_target_phrase": "parent institution",
            "anchored_entities": [
                {
                    "surface": "National Wine Centre of Australia",
                    "chosen_alias": "National Wine Centre of Australia",
                    "resolved_entity_id": "m.03hd1z",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "answer",
            "relation_paths": [
                {
                    "from_role": "anchor",
                    "to_role": "answer",
                    "relation": "education.educational_institution.parent_institution",
                    "grounding_source": "dynamic_probe",
                }
            ],
        }
    )

    assert archetype["grounding_sources"] == ["dynamic_probe"]
    assert archetype["anchor_binding_modes"] == ["resolved_entity_id"]
    assert archetype["answer_target_present"] is True
    assert archetype["relation_signatures"] == [
        "anchor->education.educational_institution.parent_institution->answer->dynamic_probe"
    ]


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
        "preserve_all_anchor_constraints_on_same_answer_variable"
        in active_bundle.validator_expectations
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


def test_stage_a_screen_rejects_candidate_before_full_inline_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    family_name = "count_over_joined_set"
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path,
    )
    candidate = store.create_candidate_update(
        family_name=family_name,
        scaffold_signature="count_over_joined_set|candidate_set|biology.animal_breed.temperament",
        relation_names=["biology.animal_breed.temperament"],
        failure_reasons=["pal_query_not_accepted:repairable_bad_count_set"],
        failure_class="weak_applicability_boundary",
        trigger_context={"sample_index": "8"},
    )
    assert candidate is not None
    active_version = store.get_active_version(family_name)
    store.set_trusted_success_bank(
        family_name,
        sample_ids=["21"],
        source_version=active_version,
        evaluation_context=family_policy_harness._build_success_bank_context(
            family_name=family_name,
            active_version=active_version,
        ),
    )
    monkeypatch.setenv(family_policy_harness.ENV_STAGE_A_SCREEN, "1")

    def _fake_run_sample_with_policy(**kwargs):
        mode = str(kwargs.get("evaluation_mode") or "")
        version = str(kwargs.get("override_version") or "")
        if mode == family_policy_harness.STAGE_A_EVALUATION_MODE:
            if version == active_version:
                return {
                    "sample_index": str(kwargs["sample_index"]),
                    "sample_status": "completed",
                    "evaluation_outcome": "correct",
                    "dangerous_overreach_count": 0,
                    "evaluation_cache_hit": False,
                }
            return {
                "sample_index": str(kwargs["sample_index"]),
                "sample_status": "not_completed",
                "evaluation_outcome": "incorrect",
                "dangerous_overreach_count": 0,
                "evaluation_cache_hit": False,
            }
        raise AssertionError("full inline evaluation should be skipped after stage_a failure")

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
        label_prefix="stage_a_test",
        trigger_baseline_summary={
            "sample_index": "8",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "dangerous_overreach_count": 0,
        },
        parent_output_dir=None,
    )

    assert evaluation["gate_stage"] == "stage_a_screen"
    assert "failed_stage_a_pal_only_screen" in evaluation["promotion_gate"]["reasons"]
    assert evaluation["evaluation_stats"]["total_evaluation_requests"] == 2


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
    assert pending[0]["fields_changed"] == ["validator_expectations"]
    repair_policy = tuple(pending[0]["bundle"]["repair_policy"])
    assert repair_policy == (
        "repair direct count relation family before escalating to joined-count family",
    )
    blocked = tuple(pending[0]["bundle"]["blocked_scaffold_signatures"])


def test_harness_synthesizes_candidate_from_nonfinal_decision_with_repair_rejection(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    run_dir = tmp_path / "run"
    (run_dir / "pal_query_artifacts").mkdir(parents=True)
    (run_dir / "generated_tools.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "pal_attempt_decision",
                        "sample_index": "8",
                        "selected_family": "count_over_joined_set",
                        "family_bundle_version": "2026-03-31",
                        "tool_name": "pal_sparql_query_tool_sample_8",
                        "dangerous_overreach": False,
                        "dangerous_overreach_reasons": [],
                        "materialization_denial_reasons": [],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_repair_loop_rejected",
                        "sample_index": "8",
                        "final_verdict": "no_accepted_candidate",
                        "last_verdict": "repairable_bad_count_set",
                        "last_reasons": [
                            "count_query_zero_with_live_anchor_paths",
                            "count_scalar_returned:0",
                        ],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_attempt_decision",
                        "sample_index": "8",
                        "selected_family": None,
                        "family_bundle_version": None,
                        "tool_name": "pal_sparql_query_tool_sample_8",
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
    (run_dir / "pal_query_artifacts" / "pal_sparql_query_tool_sample_8.plan.json").write_text(
        json.dumps(
            {
                "query_shape": "count_over_joined_set",
                "projection_role": "breed_set",
                "relation_paths": [
                    {
                        "relation": "biology.breed_origin.breeds_originating_here",
                    },
                    {
                        "relation": "biology.animal_breed.temperament",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    event = _maybe_create_candidate_from_run_summary(
        family_name="count_over_joined_set",
        run_summary={
            "sample_index": "8",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "run_dir": str(run_dir),
        },
        store_path=tmp_path / "store",
    )

    assert event is not None
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    pending = store.get_pending_candidates("count_over_joined_set")
    assert len(pending) == 1
    assert pending[0]["fields_changed"] == ["validator_expectations"]
    assert (
        "verify_count_targets_requested_entity_set"
        in pending[0]["bundle"]["validator_expectations"]
    )


def test_harness_skips_single_anchor_candidate_for_low_trust_dynamic_fail_close(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    run_dir = tmp_path / "run"
    (run_dir / "pal_query_artifacts").mkdir(parents=True)
    (run_dir / "generated_tools.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "pal_attempt_decision",
                        "sample_index": "18",
                        "selected_family": "single_anchor_lookup",
                        "family_bundle_version": "2026-03-31__cand0001__cand0001",
                        "tool_name": "pal_sparql_query_tool_sample_18",
                        "dangerous_overreach": False,
                        "dangerous_overreach_reasons": [],
                        "materialization_denial_reasons": [],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_query_candidate_rejected",
                        "sample_index": "18",
                        "rejection_reasons": [
                            "family_policy_single_anchor_low_trust_dynamic_alias_repair"
                        ],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_repair_loop_rejected",
                        "sample_index": "18",
                        "final_verdict": "no_accepted_candidate",
                        "last_verdict": "repairable_anchor_path_empty",
                        "last_reasons": [
                            "anchor_path_empty:'J. Paul Reddam':people.person.pets",
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "pal_query_artifacts" / "pal_sparql_query_tool_sample_18.plan.json").write_text(
        json.dumps(
            {
                "query_shape": "single_anchor_lookup",
                "shared_answer_variable": "answer",
                "relation_paths": [
                    {
                        "relation": "people.person.pets",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    event = _maybe_create_candidate_from_run_summary(
        family_name="single_anchor_lookup",
        run_summary={
            "sample_index": "18",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "run_dir": str(run_dir),
        },
        store_path=tmp_path / "store",
    )

    assert event is None
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    assert not store.get_pending_candidates("single_anchor_lookup")


def test_harness_skips_single_anchor_chain_candidate_for_low_trust_dynamic_fail_close(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT_UPDATE_MAPPING, "1")
    run_dir = tmp_path / "run_chain"
    (run_dir / "pal_query_artifacts").mkdir(parents=True)
    (run_dir / "generated_tools.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "pal_attempt_decision",
                        "sample_index": "18",
                        "selected_family": "single_anchor_chain_lookup",
                        "family_bundle_version": "2026-03-31__cand0001__cand0001",
                        "tool_name": "pal_sparql_query_tool_sample_18",
                        "dangerous_overreach": False,
                        "dangerous_overreach_reasons": [],
                        "materialization_denial_reasons": [],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_query_candidate_rejected",
                        "sample_index": "18",
                        "rejection_reasons": [
                            "family_policy_single_anchor_low_trust_dynamic_alias_repair"
                        ],
                    }
                ),
                json.dumps(
                    {
                        "event": "pal_repair_loop_rejected",
                        "sample_index": "18",
                        "final_verdict": "no_accepted_candidate",
                        "last_verdict": "repairable_anchor_path_empty",
                        "last_reasons": [
                            "anchor_path_empty:'J. Paul Reddam':people.person.pets_or_owned_animals",
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (
        run_dir / "pal_query_artifacts" / "pal_sparql_query_tool_sample_18.plan.json"
    ).write_text(
        json.dumps(
            {
                "query_shape": "single_anchor_chain_lookup",
                "shared_answer_variable": "answer",
                "candidate_set_variable": "owned",
                "relation_paths": [
                    {
                        "relation": "biology.animal_owner.animals_owned",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    event = _maybe_create_candidate_from_run_summary(
        family_name="single_anchor_chain_lookup",
        run_summary={
            "sample_index": "18",
            "sample_status": "agent_unknown_error",
            "evaluation_outcome": "incorrect",
            "run_dir": str(run_dir),
        },
        store_path=tmp_path / "store",
    )

    assert event is None
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )
    assert not store.get_pending_candidates("single_anchor_chain_lookup")


def test_harness_scaffold_signature_matches_runtime_single_anchor_signature() -> None:
    signature = family_policy_harness._build_scaffold_signature(
        {
            "query_shape": "single_anchor_lookup",
            "shared_answer_variable": "format",
            "candidate_set_variable": "candidate_set",
            "relation_paths": [
                {"relation": "digicams.digital_camera.format"},
            ],
        }
    )

    assert signature == "single_anchor_lookup|format|digicams.digital_camera.format"


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


def test_run_sample_with_policy_bypasses_nonterminal_cached_result(
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
        if len(calls) == 1:
            return {
                "sample_index": sample_index,
                "sample_status": "running",
                "evaluation_outcome": "",
                "run_dir": str(run_dir),
                "returncode": -15,
            }
        return {
            "sample_index": sample_index,
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "run_dir": str(run_dir),
            "returncode": 0,
        }

    monkeypatch.setattr(family_policy_harness, "run_sample", fake_run_sample)

    first = family_policy_harness._run_sample_with_policy(
        sample_index="18",
        label="cache_probe_running",
        family_name="single_anchor_lookup",
        store_path=tmp_path / "store",
        promotion_enabled=False,
        override_version="2026-03-31",
    )
    second = family_policy_harness._run_sample_with_policy(
        sample_index="18",
        label="cache_probe_running",
        family_name="single_anchor_lookup",
        store_path=tmp_path / "store",
        promotion_enabled=False,
        override_version="2026-03-31",
    )
    third = family_policy_harness._run_sample_with_policy(
        sample_index="18",
        label="cache_probe_running",
        family_name="single_anchor_lookup",
        store_path=tmp_path / "store",
        promotion_enabled=False,
        override_version="2026-03-31",
    )

    assert first["evaluation_cache_hit"] is False
    assert first["sample_status"] == "running"
    assert second["evaluation_cache_hit"] is False
    assert second["sample_status"] == "completed"
    assert third["evaluation_cache_hit"] is True
    assert len(calls) == 2


def test_batch_runner_places_child_run_under_parent_output_dir(
    tmp_path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    class DummyPopen:
        def __init__(self, command, **kwargs):
            del kwargs
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
            self.returncode = 0
            self.pid = 12345

        def communicate(self, timeout=None):
            del timeout
            return "", ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr(kg_batch_runner.subprocess, "Popen", DummyPopen)

    summary = kg_batch_runner.run_sample(
        sample_index="2",
        label="demo",
        parent_output_dir=tmp_path / "parent",
    )

    assert commands
    assert "m._run_one(" in commands[0][2]
    assert str((tmp_path / "parent").resolve()) in commands[0][2]
    assert summary["run_dir"] == str(tmp_path / "parent" / "pal_batch_demo_2")


def test_batch_runner_timeout_cleans_up_orphans(
    tmp_path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []
    killed_groups: list[tuple[int, int]] = []
    killed_pids: list[int] = []

    def fake_subprocess_run(command, **kwargs):
        command_list = list(command)
        commands.append(command_list)
        if command_list[:2] == ["pkill", "-f"]:
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "", "stderr": ""},
            )()
        if command_list and command_list[0] == "lsof":
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "8000\n8001\n", "stderr": ""},
            )()
        raise AssertionError(f"unexpected subprocess.run call: {command_list}")

    class TimeoutPopen:
        def __init__(self, command, **kwargs):
            del command, kwargs
            self.pid = 4242
            self.returncode = None

        def communicate(self, timeout=None):
            raise kg_batch_runner.subprocess.TimeoutExpired(
                cmd=["python"],
                timeout=timeout or 1,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

        def poll(self):
            return None

    def fake_kill(pid: int, sig: int) -> None:
        del sig
        killed_pids.append(pid)

    def fake_killpg(pid: int, sig: int) -> None:
        killed_groups.append((pid, sig))

    monkeypatch.setattr(kg_batch_runner.subprocess, "Popen", TimeoutPopen)
    monkeypatch.setattr(kg_batch_runner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(kg_batch_runner.os, "kill", fake_kill)
    monkeypatch.setattr(kg_batch_runner.os, "killpg", fake_killpg)
    monkeypatch.setattr(kg_batch_runner.time, "sleep", lambda _: None)

    summary = kg_batch_runner.run_sample(
        sample_index="9",
        label="timeout_probe",
        parent_output_dir=tmp_path / "parent",
    )

    assert summary["returncode"] == 124
    assert summary["sample_status"] == "timeout"
    assert summary["finish_reason"] == "timed_out"
    assert summary["timed_out"] is True
    assert "partial stdout" in summary["stdout_tail"]
    assert "partial stderr" in summary["stderr_tail"]
    assert any(command[:2] == ["pkill", "-f"] for command in commands)
    assert killed_groups == [
        (4242, kg_batch_runner.signal.SIGTERM),
        (4242, kg_batch_runner.signal.SIGKILL),
    ]
    assert killed_pids == [8000, 8001]


def test_family_policy_store_serializes_cached_evaluation_bytes(tmp_path) -> None:
    store = build_family_policy_store(
        baseline_bundles=get_baseline_reusable_family_policy_bundles(),
        store_path=tmp_path / "store",
    )

    store.set_cached_evaluation(
        "count_over_direct_relation",
        cache_key="bytes-cache",
        context={"sample_index": "240"},
        result={
            "sample_status": "timeout",
            "stdout_tail": b"partial stdout",
            "stderr_tail": b"partial stderr",
        },
    )

    cached = store.get_cached_evaluation(
        "count_over_direct_relation",
        cache_key="bytes-cache",
    )

    assert cached is not None
    assert cached["result"]["stdout_tail"] == "partial stdout"
    assert cached["result"]["stderr_tail"] == "partial stderr"


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
    assert bundle["forbidden_overreach_patterns"] == []
    assert "validator_expectations" in pending[0]["fields_changed"]
    assert "require_count_answer_target_preservation" in bundle["validator_expectations"]
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
    assert pending[0]["fields_changed"] == ["blocked_scaffold_signatures"]
