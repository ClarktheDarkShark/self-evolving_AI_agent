import json
import pathlib
import sys
from unittest.mock import patch

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.agents.instance.pal_agent_controller as pal_agent_controller_module
from src.agents.exceptions import AgentUnknownException
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.agents.instance.pal_agent_controller import PALAgentController
from src.pal.invoker import PALInvocationResult
from src.pal.family_policy_evolution import (
    build_success_plan_archetype,
    merge_success_plan_archetypes,
)
from src.pal.plausibility_validator import (
    AnchorProbeResult,
    PlausibilityVerdict,
    VERDICT_ACCEPTED,
    VERDICT_REJECTED_DANGEROUS_OVERREACH,
    VERDICT_REPAIRABLE_BAD_COUNT_SET,
    VERDICT_REPAIRABLE_BAD_JOIN,
    VERDICT_REPAIRABLE_BAD_SUPERLATIVE,
    VERDICT_REPAIRABLE_GROUNDED_EMPTY,
    build_repair_feedback,
    validate_pal_execution,
)
from src.pal.kg_benchmark_adapter import (
    BenchmarkAdapterContext,
    BenchmarkMaterialization,
    adapt_pal_result_to_benchmark,
    classify_execution_artifact,
)
from src.pal.policy_contracts import FamilyPolicyBundle, build_trust_contract_evaluation
from src.pal.reusable_tool_families import select_reusable_tool
from src.typings import ChatHistory, ChatHistoryItem, Role


def _make_controller() -> PALAgentController:
    controller = object.__new__(PALAgentController)
    controller._emit_generated_tools_event = lambda payload: None
    controller._pending_macro_runs = {}
    controller._registered_bridge_tools = set()
    controller._tool_invoked_in_last_inference = None
    controller._manual_fallback_agent = None
    controller._manual_fallback_active_runs = set()
    return controller


def test_question_interpretation_extracts_dynamic_anchor_inputs() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text="the creators of wma's most recently released browser was called what?",
        explicit_entities=[],
        answer_target_phrase="creators",
    )

    surfaces = {
        str(item.get("surface") or "")
        for item in interpretation["question_inputs"]
    }
    scaffold_names = {
        str(item.get("name") or "")
        for item in interpretation["preferred_scaffolds"]
    }

    assert "wma" in {surface.lower() for surface in surfaces}
    assert "most recently" in {surface.lower() for surface in surfaces}


def test_question_interpretation_treats_occupation_surface_as_constraint_value() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text=(
            "what is the number of film characters that are with educators occupation "
            "and neyaphem species?"
        ),
        explicit_entities=["educators", "Neyaphem"],
        answer_target_phrase="film characters",
    )

    inputs_by_surface = {
        str(item.get("surface") or "").lower(): item
        for item in interpretation["question_inputs"]
    }

    assert inputs_by_surface["educators"]["kind"] == "attribute_value"
    assert inputs_by_surface["educators"]["role_hint"] == "constraint_value"


def test_build_success_plan_archetype_keeps_only_structural_pattern() -> None:
    archetype = build_success_plan_archetype(
        {
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Serbia",
                    "chosen_alias": "Serbia",
                    "role": "anchor_a",
                },
                {
                    "surface": "Smooth Fox Terrier",
                    "chosen_alias": "Smooth Fox Terrier",
                    "role": "anchor_b",
                },
            ],
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "candidate_set",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "candidate_set",
                    },
                ],
            },
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "relation_paths": [
                {
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
            ],
        }
    )

    assert archetype["query_shape"] == "count_over_joined_set"
    assert archetype["anchor_roles"] == ["anchor_a", "anchor_b"]
    assert archetype["anchor_constraints"] == [
        "anchor_a->candidate_set",
        "anchor_b->candidate_set",
    ]
    assert archetype["relation_role_skeleton"] == [
        "anchor_a->candidate_set:curated",
        "anchor_b->candidate_set:curated",
        "candidate_set->count_set:curated",
    ]
    assert "preserve_multiple_anchor_constraints" in archetype["structural_notes"]
    assert "Serbia" not in json.dumps(archetype)


def test_merge_success_plan_archetypes_dedupes_by_signature() -> None:
    base = build_success_plan_archetype(
        {
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [{"role": "anchor"}],
            "join_structure": {"type": "count", "anchor_constraints": []},
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "relation_paths": [
                {
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
        }
    )

    merged = merge_success_plan_archetypes([base], dict(base))

    assert len(merged) == 1
    assert merged[0]["pattern_signature"] == base["pattern_signature"]


def test_grounding_card_omits_family_success_patterns_by_default(monkeypatch) -> None:
    controller = _make_controller()

    class _FakeStore:
        def get_trusted_success_bank_metadata(self, family_name: str):
            assert family_name == "count_over_joined_set"
            return {
                "source_version": "2026-03-31",
                "evaluation_context": {
                    "success_plan_archetypes": [
                        {
                            "anchor_roles": ["anchor_a", "anchor_b"],
                            "anchor_constraints": [
                                "anchor_a->candidate_set",
                                "anchor_b->candidate_set",
                            ],
                            "relation_role_skeleton": [
                                "anchor_a->candidate_set:curated",
                                "anchor_b->candidate_set:curated",
                                "candidate_set->count_set:curated",
                            ],
                            "structural_notes": [
                                "preserve_multiple_anchor_constraints",
                                "count_target_distinct_from_candidate_set",
                            ],
                        }
                    ]
                },
            }

    controller._get_family_policy_store = lambda: _FakeStore()
    monkeypatch.setattr(
        controller,
        "_infer_query_shape",
        lambda **kwargs: "count_over_joined_set",
    )
    monkeypatch.setattr(
        controller,
        "_build_question_interpretation",
        lambda **kwargs: {"question_inputs": [], "preferred_scaffolds": []},
    )
    monkeypatch.setattr(
        controller,
        "_refine_question_interpretation_with_grounding",
        lambda **kwargs: kwargs["question_interpretation"],
    )

    grounding_card = controller._build_pal_grounding_card(
        "Question: how many dialects share two constraints?, Entities: ['A', 'B']",
        relation_grounding=[],
    )

    assert "- family_success_patterns:" not in grounding_card


def test_family_policy_candidate_uses_current_session_status_when_available(
    monkeypatch,
) -> None:
    controller = _make_controller()
    captured: dict[str, object] = {}
    monkeypatch.setenv("PAL_ENABLE_FAMILY_POLICY_EVOLUTION", "1")

    class _FakeStore:
        def create_candidate_update(self, **kwargs):
            captured.update(kwargs)
            return type(
                "_Candidate",
                (),
                {
                    "base_version": "2026-03-31",
                    "candidate_version": "2026-03-31__cand0001",
                    "fields_changed": ("validator_expectations",),
                    "reason_for_change": "family_failure:pal_query_not_accepted:repairable_bad_count_set",
                    "trigger_context": dict(kwargs.get("trigger_context") or {}),
                },
            )()

    controller._get_family_policy_store = lambda: _FakeStore()
    controller._current_session = type(
        "_Session",
        (),
        {
            "task_name": "knowledge_graph",
            "sample_index": "15",
            "sample_status": "completed",
            "evaluation_record": {"outcome": "incorrect"},
        },
    )()
    controller._resolve_family_policy_trigger_query_plan = lambda **kwargs: {
        "query_shape": "count_over_joined_set"
    }
    controller._relation_names_from_plan = lambda plan: ["music.recording.artist"]
    controller._build_scaffold_signature = lambda plan: "count_over_joined_set|anchor|music.recording.artist"
    controller._collect_family_policy_failure_reasons = lambda **kwargs: (
        "pal_query_not_accepted:repairable_bad_count_set",
        "count_answer_target_unenforced:artist",
    )

    controller._maybe_record_family_policy_candidate(
        generated_tool_name="pal_sparql_query_tool_demo",
        query_plan={"query_shape": "count_over_joined_set"},
        failure_reason="pal_query_not_accepted:repairable_bad_count_set",
        repair_loop_log=None,
        trust_contract=None,
    )

    assert captured["failure_class"] == "weak_applicability_boundary"
    assert captured["trigger_context"]["sample_index"] == "15"


def test_grounding_card_surfaces_active_family_policy(monkeypatch) -> None:
    controller = _make_controller()

    monkeypatch.setattr(
        pal_agent_controller_module,
        "get_reusable_family_policy_bundle",
        lambda family_name: FamilyPolicyBundle(
            family_name=family_name,
            version="2026-03-31__cand1234",
            renderer_name="count",
            applicability_conditions=(
                "query_shape=count_over_direct_relation",
                "counted relation grounded",
                "require_explicit_answer_role_alignment",
            ),
            validator_expectations=(
                "counted variable must be structurally bound",
                "verify_answer_target_semantics_not_just_executability",
                "verify_count_targets_requested_entity_set",
            ),
            repair_policy=(
                "repair direct count relation family before escalating to joined-count family",
                "switch_to_joined_count_when_downstream_filter_or_projection_exists",
            ),
        ),
    )
    monkeypatch.setattr(
        controller,
        "_infer_query_shape",
        lambda **kwargs: "count_over_direct_relation",
    )
    monkeypatch.setattr(
        controller,
        "_build_question_interpretation",
        lambda **kwargs: {"question_inputs": [], "preferred_scaffolds": []},
    )
    monkeypatch.setattr(
        controller,
        "_refine_question_interpretation_with_grounding",
        lambda **kwargs: kwargs["question_interpretation"],
    )

    grounding_card = controller._build_pal_grounding_card(
        "Question: how many breeds share the same temperament as bull terrier?, Entities: ['Bull Terrier']",
        relation_grounding=[],
    )

    assert "- active_family_policy:" in grounding_card
    assert "bundle_version=2026-03-31__cand1234" in grounding_card
    assert "Keep the answer/count target aligned" in grounding_card
    assert "use the joined-count family" in grounding_card
    assert "counted relation grounded" not in grounding_card
    assert (
        "verify_answer_target_semantics_not_just_executability" not in grounding_card
    )
    assert (
        "repair direct count relation family before escalating to joined-count family"
        in grounding_card
    )


def test_attribute_value_alias_candidates_include_singular_profession_forms() -> None:
    controller = _make_controller()

    aliases = controller._build_entity_alias_candidates(
        "educators",
        entity_clue="attribute_value.occupation",
    )

    assert "educators" in aliases
    assert "educator" in aliases
    assert "Educator" in aliases


def test_attribute_value_alias_candidates_include_bayer_filter_alias() -> None:
    controller = _make_controller()

    aliases = controller._build_entity_alias_candidates(
        "bayer",
        entity_clue="attribute_value.feature",
    )

    assert "Bayer filter" in aliases


def test_grounding_candidates_use_fictional_character_occupation_for_character_questions() -> None:
    controller = _make_controller()

    candidates = controller._build_grounded_relation_candidates(
        "Question: what is the number of film characters that are with educators occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
    )

    assert any(
        candidate.get("relation") == "fictional_universe.fictional_character.occupation"
        for candidate in candidates
    )
    assert not any(
        candidate.get("relation") == "people.person.profession"
        for candidate in candidates
    )


def test_grounding_candidates_include_camera_attribute_filters() -> None:
    controller = _make_controller()

    candidates = controller._build_grounded_relation_candidates(
        "Question: what is the sensor type of a digital camera that has the color filter array type of bayer and iso settings of 120?, Entities: ['bayer', '120']"
    )

    assert any(
        candidate.get("relation") == "digicams.digital_camera.sensor_type"
        for candidate in candidates
    )
    assert any(
        candidate.get("relation") == "digicams.digital_camera.color_filter_array_type"
        and candidate.get("to_role") == "constraint_value"
        for candidate in candidates
    )
    assert any(
        candidate.get("relation") == "digicams.digital_camera.iso_setting"
        and candidate.get("to_role") == "constraint_value"
        for candidate in candidates
    )


def test_probe_relation_endpoint_labels_keep_with_this_relation_target_type() -> None:
    controller = _make_controller()

    source_label, target_label = controller._infer_probe_relation_endpoint_labels(
        "people.profession.people_with_this_profession"
    )

    assert source_label == "profession"
    assert target_label == "person"


def test_grounded_relation_sort_uses_explicit_anchor_clue_to_rank_curated_candidates() -> None:
    controller = _make_controller()

    ranked = controller._sort_grounded_relation_candidates_by_semantic_fit(
        relation_candidates=[
            {
                "relation": "music.artist.track",
                "direction": "forward",
                "from": "artist",
                "to": "track",
                "grounding_source": "curated",
                "from_role": "anchor",
                "to_role": "count_set",
                "support": "curated_music_predicate",
                "use_when": "find tracks performed/recorded by a music artist",
            },
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "profession",
                "to": "person",
                "grounding_source": "curated",
                "from_role": "constraint_value",
                "to_role": "count_set",
                "support": "curated_people_predicate",
                "use_when": "find people who have the given profession",
            },
        ],
        question_text="Question: how many songwriters work in the percussionist profession?",
        answer_target_phrase="songwriters",
        question_inputs=[
            {
                "surface": "Percussionist",
                "kind": "named_entity",
                "role_hint": "anchor",
                "reason": "explicit_entity:attribute_value.profession",
            }
        ],
    )

    assert ranked[0]["relation"] == "people.profession.people_with_this_profession"


def test_class_filtered_count_repair_plan_supports_constraint_value_filters() -> None:
    controller = _make_controller()

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "Percussionist",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "strategy": "Count the distinct candidate_set reached directly from the anchor.",
            "plan_rationale": [],
        },
        relation_grounding=[
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "candidate_set",
                "to": "profession",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            }
        ],
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_joined_set"
    assert any(
        str(item.get("role") or "") == "constraint_value"
        for item in rewritten["anchored_entities"]
    )


def test_class_filtered_count_repair_plan_accepts_relation_hinted_constraint_anchor_path() -> None:
    controller = _make_controller()

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "answer_target_phrase": "songwriters",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "person",
                        "notes": (
                            "The anchor profession (Percussionist) is used via "
                            "people.profession.people_with_this_profession to produce "
                            "the candidate person set."
                        ),
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "profession",
                    "to": "person",
                    "from_role": "constraint_value",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "shared_answer_variable": "answer_person",
            "candidate_set_variable": "candidate_person",
            "count_set_variable": "person",
            "strategy": "Count people in the anchor profession, then enforce songwriter.",
            "plan_rationale": [],
        },
        relation_grounding=[
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "profession",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            }
        ],
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_joined_set"
    assert any(
        str(item.get("role") or "") == "constraint_value"
        and str(item.get("chosen_alias") or "") == "songwriter"
        for item in rewritten["anchored_entities"]
    )
    anchor_notes = rewritten["join_structure"]["anchor_constraints"][0]["notes"]
    filter_notes = rewritten["join_structure"]["anchor_constraints"][1]["notes"]
    assert "people.profession.people_with_this_profession" in anchor_notes
    assert "people.person.profession" in filter_notes
    assert "'songwriter'" in filter_notes


def test_class_filtered_count_repair_plan_reuses_anchor_to_constraint_relation_for_candidate_filter() -> None:
    controller = _make_controller()

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "Percussionist",
                    "to": "person",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "person",
            "count_set_variable": "person",
            "strategy": "Count the distinct person reached directly from the anchor.",
            "plan_rationale": [],
        },
        relation_grounding=[
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "profession",
                "from_role": "anchor",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            }
        ],
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_joined_set"
    class_path = rewritten["relation_paths"][1]
    assert class_path["from_role"] == "candidate_set"
    assert class_path["to_role"] == "constraint_value"


def test_joined_count_boundary_repair_plan_reclassifies_downstream_projection() -> None:
    controller = _make_controller()

    rewritten = controller._build_joined_count_boundary_repair_plan(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "artists",
            "anchored_entities": [
                {
                    "surface": "LSO",
                    "chosen_alias": "LSO",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "artist",
            "candidate_set_variable": "recording",
            "count_set_variable": "artist",
            "relation_paths": [
                {
                    "relation": "music.recording.featured_artists",
                    "direction": "reverse",
                    "from": "recording",
                    "to": "LSO",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "music.recording.artist",
                    "direction": "forward",
                    "from": "recording",
                    "to": "artist",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "strategy": "Count artists reachable from recordings tied to the anchor.",
            "plan_rationale": [],
        }
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_joined_set"


def test_joined_count_boundary_repair_plan_reclassifies_downstream_constraint_filter() -> None:
    controller = _make_controller()

    rewritten = controller._build_joined_count_boundary_repair_plan(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "songwriters",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "person",
            "candidate_set_variable": "person",
            "count_set_variable": "person",
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "profession",
                    "to": "person",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "people.person.profession",
                    "direction": "forward",
                    "from": "person",
                    "to": "profession",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "strategy": "Count persons with the anchor profession, then filter their profession.",
            "plan_rationale": [],
        }
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_joined_set"


def test_normalize_pal_query_plan_reclassifies_direct_count_with_downstream_filter() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "songwriters",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "person",
            "candidate_set_variable": "person",
            "count_set_variable": "person",
            "join_structure": {"type": "count", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "profession",
                    "to": "person",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "people.person.profession",
                    "direction": "forward",
                    "from": "person",
                    "to": "profession",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "Count persons with the anchor profession, then filter their profession.",
            "plan_rationale": [],
        }
    )

    assert normalized["query_shape"] == "count_over_joined_set"
    assert any(
        "joined-count family" in str(item)
        for item in normalized["plan_rationale"]
    )


def test_normalize_pal_query_plan_reclassifies_two_hop_direct_count() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "artists",
            "anchored_entities": [
                {
                    "surface": "LSO",
                    "chosen_alias": "LSO",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "artist",
            "candidate_set_variable": "recording",
            "count_set_variable": "artist",
            "join_structure": {"type": "count", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "music.recording.featured_artists",
                    "direction": "reverse",
                    "from": "recording",
                    "to": "LSO",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "music.recording.artist",
                    "direction": "forward",
                    "from": "recording",
                    "to": "artist",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "Count artists by first binding recordings tied to the anchor.",
            "plan_rationale": [],
        }
    )

    assert normalized["query_shape"] == "count_over_joined_set"
    assert any(
        "joined-count family" in str(item)
        for item in normalized["plan_rationale"]
    )


def test_normalize_pal_query_plan_repairs_nonstructural_joined_count_variables() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_joined_set",
            "answer_target_phrase": "artists recorded the contribution by lso",
            "anchored_entities": [
                {
                    "surface": "lso",
                    "chosen_alias": "lso",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "artist",
            "candidate_set_variable": "artist_set",
            "count_set_variable": "artist_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "recording",
                        "notes": "recording pivot",
                    },
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "artist_set",
                        "notes": "counted answer set",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "music.recording_contribution.contributor",
                    "direction": "reverse",
                    "from": "recording",
                    "to": "lso",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.recording.artist",
                    "direction": "forward",
                    "from": "recording",
                    "to": "artist",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "Count artists who recorded the contribution by lso.",
            "plan_rationale": [],
        }
    )

    assert normalized["candidate_set_variable"] == "recording"
    assert normalized["count_set_variable"] == "artist"
    assert normalized["shared_answer_variable"] == "artist"
    assert normalized["join_structure"]["anchor_constraints"][1]["constrains_variable"] == "artist"


def test_grounded_relation_candidates_prioritize_discriminative_answer_target_tokens() -> None:
    controller = _make_controller()

    ranked = controller._prune_redundant_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "music.recording.featured_artists",
                "direction": "reverse",
                "from": "recording",
                "to": "lso",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
                "use_when": "find recordings that link to lso via featured artists",
            },
            {
                "relation": "music.recording_contribution.contributor",
                "direction": "reverse",
                "from": "contribution",
                "to": "lso",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
                "use_when": "find contributions that link to lso via recording contribution contributor",
            },
            {
                "relation": "music.album.artist",
                "direction": "reverse",
                "from": "album",
                "to": "lso",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
                "use_when": "find albums linked to lso by album artist",
            },
        ],
        question_text="Question: how many artists recorded the contribution by lso?",
        answer_target_phrase="artists recorded the contribution by lso",
    )

    assert ranked[0]["relation"] == "music.recording_contribution.contributor"


def test_class_filtered_count_repair_plan_skips_relation_encoded_answer_class() -> None:
    controller = _make_controller()

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: what is the number of infectious diseases that are transmitted "
            "by the aedes aegypti?, Entities: ['Aedes aegypti']"
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Aedes aegypti",
                    "chosen_alias": "Aedes aegypti",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "reverse",
                    "from": "disease",
                    "to": "Aedes aegypti",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "disease",
            "count_set_variable": "disease",
            "strategy": "Count diseases that list the anchor as their vector.",
            "plan_rationale": [],
        },
        relation_grounding=[
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "infectious disease",
                "to": "disease",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert rewritten is None


def test_class_filtered_count_candidate_prefers_generic_root_aligned_relation() -> None:
    controller = _make_controller()

    selected = controller._select_class_filtered_count_candidate(
        relation_grounding=[
            {
                "relation": "music.artist.profession",
                "direction": "forward",
                "from": "artist",
                "to": "profession",
                "from_role": "anchor",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "profession",
                "from_role": "anchor",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        preferred_relation_root="people",
    )

    assert selected is not None
    assert selected["relation"] == "people.person.profession"


def test_direct_count_grounding_validation_accepts_answer_candidate_role_equivalence() -> None:
    controller = _make_controller()

    matched, reason = controller._relation_contract_match_details(
        planned_path={
            "relation": "fictional_universe.fictional_universe.species",
            "direction": "forward",
            "from": "Seventh sphere",
            "to": "species",
            "from_role": "anchor",
            "to_role": "answer",
        },
        grounded_candidate={
            "relation": "fictional_universe.fictional_universe.species",
            "direction": "forward",
            "from": "anchor",
            "to": "candidate_set",
            "from_role": "anchor",
            "to_role": "candidate_set",
        },
        query_shape="count_over_direct_relation",
        anchor_count=1,
        anchored_entities=[
            {
                "surface": "Seventh sphere",
                "chosen_alias": "Seventh sphere",
                "role": "anchor",
            }
        ],
    )

    assert matched is True
    assert reason == "accepted_role_match"


def test_direct_count_grounding_validation_accepts_pivot_preserving_count_family() -> None:
    controller = _make_controller()
    query_plan = {
        "query_shape": "count_over_direct_relation",
        "relation_paths": [
            {
                "relation": "fictional_universe.fictional_setting.universe",
                "direction": "forward",
                "from": "Seventh sphere",
                "to": "universe",
                "from_role": "anchor",
                "to_role": "candidate_set",
            },
            {
                "relation": "fictional_universe.fictional_universe.species",
                "direction": "forward",
                "from": "universe",
                "to": "species",
                "from_role": "candidate_set",
                "to_role": "count_set",
            },
        ],
    }

    matched, reason = controller._relation_contract_match_details(
        planned_path=query_plan["relation_paths"][1],
        grounded_candidate={
            "relation": "fictional_universe.fictional_universe.species",
            "direction": "forward",
            "from": "fictional_world",
            "to": "species",
            "from_role": "anchor",
            "to_role": "count_set",
        },
        query_shape="count_over_direct_relation",
        anchor_count=1,
        anchored_entities=[
            {
                "surface": "Seventh sphere",
                "chosen_alias": "Seventh sphere",
                "role": "anchor",
            }
        ],
        query_plan=query_plan,
    )

    assert matched is True
    assert reason == "accepted_role_match"



def test_superlative_grounding_validation_accepts_candidate_anchor_and_ordering_equivalence() -> None:
    controller = _make_controller()

    matched, reason = controller._relation_contract_match_details(
        planned_path={
            "relation": "music.recording.length",
            "direction": "forward",
            "from": "candidate_set",
            "to": "length_attr",
            "from_role": "candidate_set",
            "to_role": "ordering_attribute",
        },
        grounded_candidate={
            "relation": "music.recording.length",
            "direction": "forward",
            "from": "recording",
            "to": "length",
            "from_role": "anchor",
            "to_role": "answer",
        },
        query_shape="superlative_chain",
        anchor_count=1,
        anchored_entities=[
            {
                "surface": "David Han",
                "chosen_alias": "David Han",
                "role": "anchor",
            }
        ],
    )

    assert matched is True
    assert reason == "accepted_role_match"


def test_joined_count_grounding_validation_accepts_generic_anchor_candidate_for_specific_anchor() -> None:
    controller = _make_controller()

    matched, reason = controller._relation_contract_match_details(
        planned_path={
            "relation": "broadcast.genre.content",
            "direction": "reverse",
            "from": "higher_education",
            "to": "genre_contents",
            "from_role": "anchor_a",
            "to_role": "count_set",
        },
        grounded_candidate={
            "relation": "broadcast.genre.content",
            "direction": "reverse",
            "from": "genre",
            "to": "content",
            "from_role": "anchor",
            "to_role": "count_set",
        },
        query_shape="count_over_joined_set",
        anchor_count=2,
        anchored_entities=[
            {
                "surface": "Higher Education",
                "chosen_alias": "m.03bv2kt",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "m.03fx9_c",
                "role": "anchor_b",
            },
        ],
    )

    assert matched is True
    assert reason == "accepted_role_match"


def test_joined_count_grounding_validation_keeps_specific_anchor_when_candidate_resolves_other_anchor() -> None:
    controller = _make_controller()

    matched, reason = controller._relation_contract_match_details(
        planned_path={
            "relation": "broadcast.genre.content",
            "direction": "reverse",
            "from": "higher_education",
            "to": "genre_contents",
            "from_role": "anchor_a",
            "to_role": "count_set",
        },
        grounded_candidate={
            "relation": "broadcast.genre.content",
            "direction": "reverse",
            "from": "To the Best of Our Knowledge",
            "to": "content",
            "from_role": "anchor",
            "to_role": "count_set",
        },
        query_shape="count_over_joined_set",
        anchor_count=2,
        anchored_entities=[
            {
                "surface": "Higher Education",
                "chosen_alias": "m.03bv2kt",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "m.03fx9_c",
                "role": "anchor_b",
            },
        ],
    )

    assert matched is False
    assert reason == "mismatch:from_role"


def test_superlative_role_coercion_treats_anchor_category_as_type_family() -> None:
    controller = _make_controller()

    coerced = controller._coerce_relation_roles_for_query_shape(
        normalized_item={
            "relation": "meteorology.tropical_cyclone.category",
            "direction": "forward",
            "from": "cyclone",
            "to": "category",
            "from_role": "anchor",
            "to_role": "answer",
        },
        query_shape="superlative_chain",
        ordering_attribute={},
        anchored_entities=[
            {
                "surface": "Hurricane Dolly",
                "chosen_alias": "Hurricane Dolly",
                "role": "anchor",
            }
        ],
        candidate_set_variable="candidate_set",
    )

    assert coerced["from_role"] == "anchor"
    assert coerced["to_role"] == "type_set"


def test_infer_relation_endpoint_role_prefers_anchored_entity_identity_over_explicit_constraint_role() -> None:
    controller = _make_controller()

    inferred = controller._infer_relation_endpoint_role(
        explicit_role="constraint_value",
        raw_endpoint="Percussionist",
        endpoint_side="to",
        direction="forward",
        query_shape="count_over_direct_relation",
        answer_mode="count",
        answer_target_phrase="songwriters",
        anchored_entities=[
            {
                "surface": "Percussionist",
                "chosen_alias": "Percussionist",
                "role": "anchor",
            }
        ],
        shared_answer_variable="",
        candidate_set_variable="candidate_person",
        count_set_variable="candidate_person",
        ordering_attribute={},
    )

    assert inferred == "anchor"


def test_infer_relation_endpoint_role_matches_shared_bridge_variable_by_token_family() -> None:
    controller = _make_controller()

    inferred = controller._infer_relation_endpoint_role(
        explicit_role="",
        raw_endpoint="producer",
        endpoint_side="from",
        direction="reverse",
        query_shape="count_over_joined_set",
        answer_mode="count",
        answer_target_phrase="contents",
        anchored_entities=[
            {
                "surface": "Higher Education",
                "chosen_alias": "Higher Education",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "To the Best of Our Knowledge",
                "role": "anchor_b",
            },
        ],
        shared_answer_variable="shared_producer",
        candidate_set_variable="candidate_content",
        count_set_variable="candidate_content",
        ordering_attribute={},
    )

    assert inferred == "shared_answer"


def test_infer_relation_endpoint_role_matches_candidate_variable_by_token_family() -> None:
    controller = _make_controller()

    inferred = controller._infer_relation_endpoint_role(
        explicit_role="",
        raw_endpoint="content",
        endpoint_side="to",
        direction="forward",
        query_shape="count_over_joined_set",
        answer_mode="count",
        answer_target_phrase="contents",
        anchored_entities=[
            {
                "surface": "Higher Education",
                "chosen_alias": "Higher Education",
                "role": "anchor_a",
            }
        ],
        shared_answer_variable="shared_producer",
        candidate_set_variable="candidate_content",
        count_set_variable="candidate_content",
        ordering_attribute={},
    )

    assert inferred == "candidate_set"


def test_normalize_joined_count_plan_keeps_bridge_candidate_and_count_target_distinct() -> None:
    controller = _make_controller()

    plan = controller._normalize_pal_query_plan(
        {
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "different species",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Seventh sphere",
                    "chosen_alias": "Seventh sphere",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "universe_var",
            "count_set_variable": "species",
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.universe",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "universe_var",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "universe_var",
                    "to": "species",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "count species by traversing from setting to universe to species",
        }
    )

    assert plan["relation_paths"][0]["from_role"] == "anchor"
    assert plan["relation_paths"][0]["to_role"] == "candidate_set"
    assert plan["relation_paths"][1]["from_role"] == "candidate_set"
    assert plan["relation_paths"][1]["to_role"] == "count_set"


def test_normalize_multi_anchor_plan_uniquifies_constraint_value_variables() -> None:
    controller = _make_controller()

    plan = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "answer_target_phrase": "sensor type",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {
                    "surface": "bayer",
                    "chosen_alias": "Bayer filter",
                    "role": "anchor_a",
                },
                {
                    "surface": "120",
                    "chosen_alias": "120",
                    "role": "anchor_b",
                },
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "relation_paths": [
                {
                    "relation": "digicams.digital_camera.color_filter_array_type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "constraint_value",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "digicams.digital_camera.iso_setting",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "constraint_value",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "digicams.digital_camera.sensor_type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "answer",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["answer", "answer_name"],
            "allow_exploratory_predicates": False,
            "strategy": "camera filter and iso intersection",
        }
    )

    constraint_targets = [
        path["to"]
        for path in plan["relation_paths"]
        if path["to_role"] == "constraint_value"
    ]
    assert constraint_targets == ["constraint_value", "constraint_value_2"]


def test_build_type_constraint_alias_candidates_includes_title_cased_singular_form() -> None:
    controller = _make_controller()

    candidates = controller._build_type_constraint_alias_candidates("songwriters")

    assert "songwriter" in candidates
    assert "Songwriter" in candidates


def test_build_grounded_type_constraint_alias_candidates_prefers_resolved_mid(
    monkeypatch,
) -> None:
    controller = _make_controller()

    monkeypatch.setattr(
        controller,
        "_probe_entity_name_count",
        lambda anchor_name, timeout_s=2.5: 1 if anchor_name == "songwriter" else 0,
    )
    monkeypatch.setattr(
        controller,
        "_probe_anchor_entity_ids",
        lambda anchor_name, relation=None, anchor_position="subject", timeout_s=2.5: (
            ["m.0fj9f"] if anchor_name == "songwriter" else []
        ),
    )

    candidates = controller._build_grounded_type_constraint_alias_candidates(
        "songwriters"
    )

    assert candidates[0] == "m.0fj9f"
    assert "songwriter" in candidates


def test_build_entity_alias_candidates_strips_trailing_generic_animal_class_word() -> None:
    controller = _make_controller()

    candidates = controller._build_entity_alias_candidates("maltese dog")

    assert "maltese dog" in candidates
    assert "Maltese Dog" in candidates
    assert "maltese" in candidates
    assert "Maltese" in candidates


def test_validate_pal_execution_accepts_multi_anchor_mid_pins_without_probe_results() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {
                    "surface": "Goat",
                    "chosen_alias": "Goat",
                    "resolved_entity_id": "m.03fwl",
                    "role": "anchor_a",
                },
                {
                    "surface": "Cattle",
                    "chosen_alias": "Cattle",
                    "resolved_entity_id": "m.01xq0k1",
                    "role": "anchor_b",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "shared_answer", "notes": "a"},
                    {"anchor_role": "anchor_b", "constrains_variable": "shared_answer", "notes": "b"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "food.cheese_milk_source.cheeses",
                    "direction": "reverse",
                    "from": "anchor_a",
                    "to": "shared_answer",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
                {
                    "relation": "food.cheese_milk_source.cheeses",
                    "direction": "reverse",
                    "from": "anchor_b",
                    "to": "shared_answer",
                    "from_role": "anchor_b",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "intersect two anchored milk-source constraints",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT ?shared_answer WHERE {\n"
            "  VALUES ?anchor_a { fb:m.03fwl }\n"
            "  VALUES ?anchor_b { fb:m.01xq0k1 }\n"
            "  ?anchor_a fb:food.cheese_milk_source.cheeses ?shared_answer .\n"
            "  ?anchor_b fb:food.cheese_milk_source.cheeses ?shared_answer .\n"
            "}"
        ),
        result_dict={
            "head": {"vars": ["shared_answer"]},
            "results": {"bindings": [{"shared_answer": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.cheese"}}]},
        },
        entities=["Goat", "Cattle"],
        anchor_probe_results=None,
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_validate_pal_execution_treats_constraint_value_probe_miss_as_optional() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                },
                {
                    "surface": "songwriters",
                    "chosen_alias": "songwriter",
                    "role": "constraint_value",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "shared_answer",
            "count_set_variable": "shared_answer",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "shared_answer", "notes": "anchor"},
                    {"anchor_role": "constraint_value", "constrains_variable": "shared_answer", "notes": "constraint"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "people.person.profession",
                    "direction": "reverse",
                    "from": "shared_answer",
                    "to": "Percussionist",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "people.person.profession",
                    "direction": "forward",
                    "from": "shared_answer",
                    "to": "constraint_value",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count intersection of primary anchor and class filter",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?shared_answer) AS ?count) WHERE {\n"
            "  VALUES ?anchor { fb:m.02h66l4 }\n"
            "  ?shared_answer fb:people.person.profession ?anchor .\n"
            "  ?shared_answer fb:people.person.profession ?constraint_value .\n"
            '  ?constraint_value fb:type.object.name ?constraint_value_label .\n'
            '  FILTER(LCASE(STR(?constraint_value_label)) = "songwriter")\n'
            "}"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "21"}}]},
        },
        entities=["Percussionist"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Percussionist",
                entity_count=3,
                path_count=921,
                relation_probed="people.person.profession",
                anchor_position="object",
                resolved_entity_id="m.02h66l4",
            ),
            AnchorProbeResult(
                anchor_name="songwriter",
                entity_count=0,
                path_count=None,
                relation_probed=None,
                anchor_position=None,
                resolved_entity_id=None,
            ),
        ],
    )

    assert verdict.verdict != VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_set_anchor_not_found" not in verdict.reasons


def test_validate_pal_execution_requires_explicit_constraint_value_anchor() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "film characters",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "Neyaphem",
                    "role": "anchor_b",
                },
                {
                    "surface": "educators",
                    "chosen_alias": "educators",
                    "role": "constraint_value",
                },
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "answer",
            "count_set_variable": "answer",
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "type_set",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "fictional_universe.fictional_character.species",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                },
            ],
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?answer) AS ?count) WHERE {\n"
            '  ?film_character_type_set fb:type.object.name "film character" .\n'
            '  ?constraint_value fb:type.object.name "educators" .\n'
            '  ?anchor_b fb:type.object.name "neyaphem" .\n'
            "  ?film_character_type_set fb:type.type.instance ?answer .\n"
            "  ?constraint_value fb:people.profession.people_with_this_profession ?answer .\n"
            "  ?answer fb:fictional_universe.fictional_character.species ?anchor_b .\n"
            "}"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["educators", "Neyaphem"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Neyaphem",
                entity_count=1,
                path_count=8,
                relation_probed="fictional_universe.fictional_character.species",
                anchor_position="object",
                resolved_entity_id="m.09tc50",
            ),
            AnchorProbeResult(
                anchor_name="educators",
                entity_count=0,
                path_count=None,
                relation_probed=None,
                anchor_position=None,
                resolved_entity_id=None,
            ),
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_set_anchor_not_found" in verdict.reasons
    assert "anchor_not_found:'educators'" in verdict.reasons


def test_validate_pal_execution_respects_relation_selection_hint_for_direct_count() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "songwriters",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "candidate",
            "count_set_variable": "candidate",
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "Percussionist",
                    "to": "candidate",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "strategy": "Count the distinct candidate reached directly from the anchor via people.profession.people_with_this_profession. Treat the answer target only as a relation-selection hint.",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?candidate) AS ?count) WHERE {\n"
            "  VALUES ?anchor { fb:m.02h66l4 }\n"
            "  ?anchor fb:people.profession.people_with_this_profession ?candidate .\n"
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "921"}}]},
        },
        entities=["Percussionist"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Percussionist",
                entity_count=3,
                path_count=921,
                relation_probed="people.profession.people_with_this_profession",
                anchor_position="subject",
                resolved_entity_id="m.02h66l4",
            )
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_joined_count_normalization_promotes_answer_target_endpoint_to_count_set() -> None:
    controller = _make_controller()

    normalized = controller._normalize_relation_contract_item(
        {
            "relation": "computer.computer.key_designers",
            "direction": "forward",
            "from": "computer",
            "to": "key designer",
            "from_role": "candidate_set",
            "to_role": "candidate_set",
            "grounding_source": "dynamic_probe",
        },
        query_shape="count_over_joined_set",
        answer_mode="count",
        answer_target_phrase="key designers",
        anchored_entities=[
            {
                "surface": "Richard Altwasser",
                "chosen_alias": "Richard Altwasser",
                "role": "anchor",
            }
        ],
        shared_answer_variable="answer",
        candidate_set_variable="candidate_set",
        count_set_variable="count",
        ordering_attribute={},
        allow_exploratory=False,
    )

    assert normalized is not None
    assert normalized["from_role"] == "candidate_set"
    assert normalized["to_role"] == "count_set"


def test_question_interpretation_extracts_shared_attribute_and_constraints() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text="how many different breeds from republic of brazil have the same temperament as the manchester terrier?",
        explicit_entities=[],
        answer_target_phrase="different breeds",
    )

    role_by_surface = {
        str(item.get("surface") or "").lower(): str(item.get("role_hint") or "")
        for item in interpretation["question_inputs"]
    }
    kind_by_surface = {
        str(item.get("surface") or "").lower(): str(item.get("kind") or "")
        for item in interpretation["question_inputs"]
    }
    scaffold_names = {
        str(item.get("name") or "")
        for item in interpretation["preferred_scaffolds"]
    }

    assert kind_by_surface["temperament"] == "shared_attribute"
    assert role_by_surface["republic of brazil"] == "constraint_value"
    assert role_by_surface["manchester terrier"] == "anchor_b"
    assert "count_shared_attribute" in scaffold_names


def test_question_interpretation_preserves_count_shared_attribute_with_question_prefix() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text="Question: how many different breeds from republic of brazil have the same temperament as the manchester terrier?",
        explicit_entities=["republic of brazil", "Manchester Terrier"],
        answer_target_phrase="different breeds",
    )

    scaffold_names = {
        str(item.get("name") or "")
        for item in interpretation["preferred_scaffolds"]
    }

    assert "count_shared_attribute" in scaffold_names


def test_question_interpretation_extracts_release_region_anchors_for_cvg_question() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text="Question: virtual console, which is developed by sega of japan, was released where?",
        explicit_entities=[],
        answer_target_phrase=controller._extract_answer_target_phrase(
            "Question: virtual console, which is developed by sega of japan, was released where?"
        ),
    )

    surfaces = {
        str(item.get("surface") or "").strip().lower()
        for item in interpretation["question_inputs"]
    }
    scaffold_names = {
        str(item.get("name") or "")
        for item in interpretation["preferred_scaffolds"]
    }

    assert controller._extract_answer_target_phrase(
        "Question: virtual console, which is developed by sega of japan, was released where?"
    ) == "region"
    assert "virtual console" in surfaces
    assert "sega of japan" in surfaces
    assert "projected_answer_intersection" in scaffold_names


def test_question_interpretation_extracts_leading_count_constraint_value() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text=(
            "Question: the serbia has how many breeds of dogs that has the same temperament as the smooth fox terrier?"
        ),
        explicit_entities=[],
        answer_target_phrase=controller._extract_answer_target_phrase(
            "Question: the serbia has how many breeds of dogs that has the same temperament as the smooth fox terrier?"
        ),
    )

    inputs = interpretation["question_inputs"]
    assert any(
        str(item.get("surface") or "").strip().lower() == "serbia"
        and str(item.get("role_hint") or "").strip() == "constraint_value"
        for item in inputs
    )
    assert any(
        str(item.get("surface") or "").strip().lower() == "smooth fox terrier"
        and str(item.get("role_hint") or "").strip() == "anchor_b"
        for item in inputs
    )


def test_question_interpretation_extracts_shared_attribute_from_attribute_of_anchor_phrase() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text=(
            "what number of contents about higher education are produced by the producer "
            "of to the best of our knowledge?"
        ),
        explicit_entities=["Higher Education", "To the Best of Our Knowledge"],
        answer_target_phrase="contents about higher education",
    )

    kind_by_surface = {
        str(item.get("surface") or "").lower(): str(item.get("kind") or "")
        for item in interpretation["question_inputs"]
    }
    scaffold_names = {
        str(item.get("name") or "")
        for item in interpretation["preferred_scaffolds"]
    }

    assert kind_by_surface["producer"] == "shared_attribute"
    assert "count_shared_attribute" in scaffold_names


def test_question_interpretation_retypes_explicit_class_phrase_inside_answer_target() -> None:
    controller = _make_controller()

    interpretation = controller._build_question_interpretation(
        question_text="Question: what is the number of research project cancer centers?",
        explicit_entities=["research project"],
        answer_target_phrase="research project cancer centers",
    )

    role_by_surface = {
        str(item.get("surface") or "").lower(): str(item.get("role_hint") or "")
        for item in interpretation["question_inputs"]
    }
    kind_by_surface = {
        str(item.get("surface") or "").lower(): str(item.get("kind") or "")
        for item in interpretation["question_inputs"]
    }

    assert kind_by_surface["research project"] == "class_phrase"
    assert role_by_surface["research project"] == "type_set"


def test_answer_target_relative_clause_is_not_split_as_class_phrase() -> None:
    controller = _make_controller()

    class_phrase, target_head = controller._split_answer_target_compound_phrase(
        "infectious diseases that can be transmitted by aedes aegypti"
    )

    assert class_phrase == ""
    assert target_head == "infectious diseases that can be transmitted by aedes aegypti"


def test_answer_target_compound_phrase_prefers_leading_head_before_relation_marker() -> None:
    controller = _make_controller()

    qualifier, head = controller._split_answer_target_compound_phrase(
        "artists recorded the contribution by lso"
    )
    assert qualifier == "recorded the contribution by lso"
    assert head == "artists"

    qualifier, head = controller._split_answer_target_compound_phrase(
        "infectious diseases can a flea transmit"
    )
    assert qualifier == "can a flea transmit"
    assert head == "infectious diseases"

    qualifier, head = controller._split_answer_target_compound_phrase(
        "contents about higher education"
    )
    assert qualifier == "about higher education"
    assert head == "contents"

    qualifier, head = controller._split_answer_target_compound_phrase(
        "game expansions"
    )
    assert qualifier == ""
    assert head == "game expansions"


def test_extract_answer_target_phrase_handles_name_of_relative_clause() -> None:
    controller = _make_controller()

    target = controller._extract_answer_target_phrase(
        "Question: what is the name of the red wine in which dutcher crossing winery winery produces?"
    )

    assert target == "red wine"


def test_infer_entity_clue_marks_from_location_as_origin_constraint() -> None:
    controller = _make_controller()

    clue = controller._infer_entity_clue(
        "Question: how many different breeds from republic of brazil have the same temperament as the manchester terrier?",
        "republic of brazil",
    )

    assert clue == "origin_constraint"


def test_infer_entity_clue_marks_has_phrase_inside_shared_category_question_as_feature() -> None:
    controller = _make_controller()

    clue = controller._infer_entity_clue(
        "Question: what martial art has the same category as mongolian wrestling and has strike?",
        "Strike",
    )

    assert clue == "attribute_value.feature"


def test_infer_entity_clue_marks_camera_filter_and_iso_values_as_feature_constraints() -> None:
    controller = _make_controller()

    question = (
        "Question: what is the sensor type of a digital camera that has the color filter array type "
        "of bayer and iso settings of 120?"
    )

    assert controller._infer_entity_clue(question, "bayer") == "attribute_value.feature"
    assert controller._infer_entity_clue(question, "120") == "attribute_value.feature"


def test_count_shared_attribute_plan_rewrite_counts_candidate_set_when_answer_target_is_entity() -> None:
    controller = _make_controller()

    plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {"surface": "republic of brazil", "chosen_alias": "Brazil", "role": "anchor_a"},
            {"surface": "Manchester Terrier", "chosen_alias": "Manchester Terrier", "role": "anchor_b"},
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {"anchor_role": "anchor_a", "constrains_variable": "shared_answer", "notes": "country constrains breeds"},
                {"anchor_role": "anchor_b", "constrains_variable": "shared_answer", "notes": "breed temperament equality"},
            ],
        },
        "relation_paths": [
            {
                "relation": "biology.breed_origin.breeds_originating_here",
                "direction": "reverse",
                "from": "country",
                "to": "breed",
                "from_role": "constraint_value",
                "to_role": "shared_answer",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "breed",
                "to": "temperament",
                "from_role": "shared_answer",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "Manchester Terrier",
                "to": "temperament",
                "from_role": "anchor_b",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count breeds with same temperament",
        "plan_rationale": ["initial"],
    }

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question="Question: how many different breeds from republic of brazil have the same temperament as the manchester terrier?, Entities: ['republic of brazil', 'Manchester Terrier']",
        query_plan=plan,
    )

    assert rewritten["shared_answer_variable"] == "breed"
    assert rewritten["count_set_variable"] == "breed"
    assert rewritten["candidate_set_variable"] == "breed"
    assert rewritten["join_structure"]["type"] == "count"
    assert rewritten["join_structure"]["anchor_constraints"][0]["constrains_variable"] == "breed"
    assert rewritten["join_structure"]["anchor_constraints"][1]["constrains_variable"] == "temperament"
    assert rewritten["relation_paths"][0]["to_role"] == "candidate_set"
    assert rewritten["relation_paths"][1]["from_role"] == "candidate_set"
    assert rewritten["relation_paths"][1]["to_role"] == "constraint_value"
    assert rewritten["relation_paths"][2]["to_role"] == "constraint_value"


def test_count_shared_attribute_plan_rewrite_counts_shared_values_when_answer_target_matches_attribute() -> None:
    controller = _make_controller()

    plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {"surface": "Aidi", "chosen_alias": "Aidi", "role": "anchor_a"},
            {
                "surface": "Australian Sheep Dog",
                "chosen_alias": "Australian Sheep Dog",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "",
        "count_set_variable": "count",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {"anchor_role": "anchor_a", "constrains_variable": "shared_answer", "notes": "aidi temperament"},
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "shared_answer",
                    "notes": "australian sheep dog temperament",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "Aidi",
                "to": "temperament",
                "from_role": "anchor_a",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "Australian Sheep Dog",
                "to": "temperament",
                "from_role": "anchor_b",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count common temperaments",
        "plan_rationale": ["initial"],
    }

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question="Question: for the aidi and australian sheep dog breeds, how many temperaments do they have in common?, Entities: ['Aidi', 'Australian Sheep Dog']",
        query_plan=plan,
    )

    assert rewritten["shared_answer_variable"] == "temperament"
    assert rewritten["count_set_variable"] == "temperament"
    assert rewritten["join_structure"]["anchor_constraints"][0]["constrains_variable"] == "temperament"
    assert rewritten["join_structure"]["anchor_constraints"][1]["constrains_variable"] == "temperament"
    assert rewritten["relation_paths"][0]["to_role"] == "count_set"
    assert rewritten["relation_paths"][1]["to_role"] == "count_set"


def test_count_shared_attribute_rewrite_keeps_explicit_shared_bridge_and_counts_entities() -> None:
    controller = _make_controller()

    plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "Higher Education",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "To the Best of Our Knowledge",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "producer",
        "candidate_set_variable": "producer",
        "count_set_variable": "producer",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "content",
                    "notes": "genre constraint",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "content",
                    "notes": "producer anchor",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "broadcast.content.producer",
                "direction": "forward",
                "from": "content",
                "to": "producer",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.producer.produces",
                "direction": "forward",
                "from": "producer",
                "to": "content",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.genre.content",
                "direction": "reverse",
                "from": "genre",
                "to": "content",
                "from_role": "anchor_a",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count content by shared producer",
        "plan_rationale": ["initial"],
    }

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what number of contents about higher education are produced by "
            "the producer of to the best of our knowledge?, "
            "Entities: ['Higher Education', 'To the Best of Our Knowledge']"
        ),
        query_plan=plan,
    )

    assert rewritten["shared_answer_variable"] == "content"
    assert rewritten["candidate_set_variable"] == "content"
    assert rewritten["count_set_variable"] == "content"
    assert rewritten["join_structure"]["anchor_constraints"][0]["constrains_variable"] == "content"
    assert rewritten["join_structure"]["anchor_constraints"][1]["constrains_variable"] == "producer"
    assert rewritten["relation_paths"][0]["to_role"] == "constraint_value"
    assert rewritten["relation_paths"][1]["from_role"] == "constraint_value"
    assert rewritten["relation_paths"][1]["to_role"] == "candidate_set"
    assert rewritten["relation_paths"][2]["to_role"] == "candidate_set"


def test_count_shared_attribute_plan_rewrite_materializes_generic_shared_answer_endpoint() -> None:
    controller = _make_controller()

    plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "Higher Education",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "To the Best of Our Knowledge",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "content",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count_set",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "count_set",
                    "notes": "Genre/topic anchor (Higher Education) constrains the content set via broadcast.genre.content (genre -> content).",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "candidate_set",
                    "notes": "Program anchor (To the Best of Our Knowledge) yields its producer (via broadcast.content.producer); that producer defines candidate_set of contents it produces (via broadcast.producer.produces).",
                },
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": "content",
                    "notes": "Final answer set is the intersection of candidate_set (contents produced by the producer) and count_set (contents in the Higher Education genre). Count is over that intersection.",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "broadcast.content.producer",
                "direction": "forward",
                "from": "anchor_b",
                "to": "producer",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.producer.produces",
                "direction": "forward",
                "from": "producer",
                "to": "candidate_set",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.genre.content",
                "direction": "reverse",
                "from": "anchor_a",
                "to": "count_set",
                "from_role": "anchor_a",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count content by shared producer",
        "plan_rationale": ["initial"],
    }

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what number of contents about higher education are produced by "
            "the producer of to the best of our knowledge?, "
            "Entities: ['Higher Education', 'To the Best of Our Knowledge']"
        ),
        query_plan=plan,
    )

    assert rewritten["shared_answer_variable"] == "content"
    assert rewritten["candidate_set_variable"] == "content"
    assert rewritten["count_set_variable"] == "content"
    assert rewritten["join_structure"]["anchor_constraints"][0]["constrains_variable"] == "content"
    assert rewritten["join_structure"]["anchor_constraints"][1]["constrains_variable"] == "producer"
    assert rewritten["relation_paths"][0]["to"] == "producer"
    assert rewritten["relation_paths"][0]["to_role"] == "constraint_value"
    assert rewritten["relation_paths"][1]["from"] == "producer"
    assert rewritten["relation_paths"][1]["from_role"] == "constraint_value"
    assert rewritten["relation_paths"][1]["to"] == "content"
    assert rewritten["relation_paths"][1]["to_role"] == "candidate_set"
    assert rewritten["relation_paths"][2]["to"] == "content"
    assert rewritten["relation_paths"][2]["to_role"] == "candidate_set"


def test_joined_count_rewrite_materializes_missing_anchor_constraint_path() -> None:
    controller = _make_controller()

    plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "Higher Education",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "To the Best of Our Knowledge",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "content",
        "candidate_set_variable": "content",
        "count_set_variable": "content",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "content",
                    "notes": "Anchor_a filters content via broadcast.genre.content",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "content",
                    "notes": "Anchor_b yields its producer via broadcast.content.producer and that producer filters candidate content",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "broadcast.content.producer",
                "direction": "forward",
                "from": "content",
                "to": "producer",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.producer.produces",
                "direction": "forward",
                "from": "producer",
                "to": "content",
                "from_role": "constraint_value",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "broadcast.genre.content",
                "direction": "reverse",
                "from": "genre",
                "to": "content",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count higher-education content produced by the producer of anchor_b",
        "plan_rationale": ["initial"],
    }

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what number of contents about higher education are produced by "
            "the producer of to the best of our knowledge?, "
            "Entities: ['Higher Education', 'To the Best of Our Knowledge']"
        ),
        query_plan=plan,
    )

    matching_paths = [
        relation_path
        for relation_path in rewritten["relation_paths"]
        if relation_path["relation"] == "broadcast.content.producer"
        and relation_path["from_role"] == "anchor_b"
        and relation_path["to_role"] == "constraint_value"
    ]
    assert matching_paths


def test_broadcast_producer_of_rewrite_prefers_content_producer_bridge() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what number of contents about higher education are produced by "
            "the producer of to the best of our knowledge?, "
            "Entities: ['Higher Education', 'To the Best of Our Knowledge']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "contents about higher education",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Higher Education",
                    "chosen_alias": "m.03bv2kt",
                    "role": "anchor_a",
                },
                {
                    "surface": "To the Best of Our Knowledge",
                    "chosen_alias": "m.03fx9_c",
                    "role": "anchor_b",
                },
            ],
            "shared_answer_variable": "content",
            "candidate_set_variable": "content",
            "count_set_variable": "content",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "content", "notes": "genre filter"},
                    {"anchor_role": "anchor_b", "constrains_variable": "producer", "notes": "producer filter"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "broadcast.genre.content",
                    "direction": "reverse",
                    "from": "Higher Education",
                    "to": "content",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "broadcast.producer.produces",
                    "direction": "reverse",
                    "from": "producer",
                    "to": "To the Best of Our Knowledge",
                    "from_role": "constraint_value",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "broadcast.producer.produces",
                    "direction": "forward",
                    "from": "producer",
                    "to": "content",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert any(
        str(path.get("relation") or "") == "broadcast.content.producer"
        and str(path.get("from_role") or "") == "anchor_b"
        and str(path.get("to_role") or "") == "constraint_value"
        for path in rewritten["relation_paths"]
    )
    assert not any(
        str(path.get("relation") or "") == "broadcast.producer.produces"
        and str(path.get("to_role") or "") == "anchor_b"
        for path in rewritten["relation_paths"]
    )


def test_question_scaffold_rewrite_is_noop_for_non_shared_attribute_count() -> None:
    controller = _make_controller()

    plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {"surface": "Unsteadiness", "chosen_alias": "Unsteadiness", "role": "anchor"}
        ],
        "shared_answer_variable": "treatment",
        "candidate_set_variable": "",
        "count_set_variable": "treatment",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "treatment", "notes": "count treatments"}
            ],
        },
        "relation_paths": [
            {
                "relation": "medicine.symptom.side_effect_of",
                "direction": "reverse",
                "from": "symptom",
                "to": "treatment",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "curated",
            }
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count direct relation",
        "plan_rationale": ["initial"],
    }

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question="Question: unsteadiness can be a side effect in how many medical treatments?, Entities: ['Unsteadiness']",
        query_plan=plan,
    )

    assert rewritten == plan


def test_generated_query_anchor_binding_rewrite_adds_name_or_alias_union() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT * WHERE {
  ?breedB fb:type.object.name "Australian Sheep Dog"@en .
  ?breedB fb:biology.animal_breed.temperament ?t .
}"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'

    rewritten_code, rewritten_queries = controller._rewrite_generated_code_anchor_bindings(
        generated_code=generated_code,
        query_texts=[query_text],
        query_plan={
            "anchored_entities": [
                {
                    "surface": "australian sheep dog",
                    "chosen_alias": "Australian Sheep Dog",
                    "role": "anchor_b",
                }
            ]
        },
    )

    assert rewritten_queries
    assert "fb:common.topic.alias" in rewritten_queries[0]
    assert 'FILTER(LCASE(STR(?breedB_label_1)) = "australian sheep dog")' in rewritten_queries[0]
    assert rewritten_code != generated_code


def test_generated_query_anchor_binding_rewrite_prefers_resolved_entity_id() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT * WHERE {
  ?language fb:type.object.name "Southern Min"@en .
  ?dialect fb:language.language_dialect.language ?language .
}"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'

    rewritten_code, rewritten_queries = controller._rewrite_generated_code_anchor_bindings(
        generated_code=generated_code,
        query_texts=[query_text],
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Southern Min",
                    "chosen_alias": "Southern Min",
                    "role": "anchor",
                    "resolved_entity_id": "m.01c44b",
                }
            ]
        },
    )

    assert rewritten_queries
    assert "VALUES ?language { fb:m.01c44b }" in rewritten_queries[0]
    assert '"Southern Min"@en' not in rewritten_queries[0]
    assert rewritten_code != generated_code


def test_generated_query_anchor_binding_rewrite_handles_filter_without_str() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT * WHERE {
  ?anchor fb:type.object.name ?anchor_name .
  FILTER(LCASE(?anchor_name) = "southern min")
  ?dialect fb:language.language_dialect.language ?anchor .
}"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'

    rewritten_code, rewritten_queries = controller._rewrite_generated_code_anchor_bindings(
        generated_code=generated_code,
        query_texts=[query_text],
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Southern Min",
                    "chosen_alias": "Southern Min",
                    "role": "anchor",
                }
            ]
        },
    )

    assert rewritten_queries
    assert "fb:common.topic.alias" in rewritten_queries[0]
    assert 'FILTER(LCASE(STR(?anchor_name)) = "southern min")' in rewritten_queries[0]
    assert rewritten_code != generated_code


def test_generated_query_anchor_binding_rewrite_collapses_alias_union_after_resolved_id() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT * WHERE {
  {
    ?anchor fb:type.object.name ?anchor_name .
    FILTER(LCASE(STR(?anchor_name)) = "southern min")
  }
  UNION
  {
    ?anchor fb:common.topic.alias ?anchor_alias .
    FILTER(LCASE(STR(?anchor_alias)) = "southern min")
  }
  ?dialect fb:language.language_dialect.language ?anchor .
}"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'

    rewritten_code, rewritten_queries = controller._rewrite_generated_code_anchor_bindings(
        generated_code=generated_code,
        query_texts=[query_text],
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Southern Min",
                    "chosen_alias": "Southern Min",
                    "role": "anchor",
                    "resolved_entity_id": "m.01c44b",
                }
            ]
        },
    )

    assert rewritten_queries
    assert "VALUES ?anchor { fb:m.01c44b }" in rewritten_queries[0]
    assert "fb:common.topic.alias" not in rewritten_queries[0]
    assert "fb:type.object.name ?anchor_name" not in rewritten_queries[0]
    assert rewritten_code != generated_code


def test_generated_query_anchor_binding_rewrite_collapses_same_line_union_after_resolved_id() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT * WHERE {
  { VALUES ?anchor { fb:m.01c44b } } UNION { ?anchor fb:common.topic.alias ?anchor_alias . FILTER(LCASE(STR(?anchor_alias)) = "southern min") }
  ?dialect fb:language.language_dialect.language ?anchor .
}"""

    rewritten = controller._collapse_resolved_anchor_union_binding_blocks(
        query_text=query_text,
        resolved_entity_id="m.01c44b",
        anchor_literal="Southern Min",
    )

    assert "UNION" not in rewritten
    assert "fb:common.topic.alias" not in rewritten
    assert "VALUES ?anchor { fb:m.01c44b }" in rewritten


def test_generated_query_anchor_binding_rewrite_updates_stale_direct_anchor_entity_id() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?candidate_set WHERE {
  fb:m.0l6tq fb:astronomy.celestial_object_category.objects ?candidate_set .
  ?candidate_set fb:astronomy.celestial_object.cosmological_distance ?distance .
} ORDER BY DESC(?distance) LIMIT 1"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'

    rewritten_code, rewritten_queries = controller._rewrite_generated_code_anchor_bindings(
        generated_code=generated_code,
        query_texts=[query_text],
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Nebula",
                    "chosen_alias": "Nebula",
                    "role": "anchor",
                    "resolved_entity_id": "m.05fny",
                }
            ],
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object_category.objects",
                    "direction": "reverse",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                }
            ],
        },
    )

    assert rewritten_queries
    assert "fb:m.05fny fb:astronomy.celestial_object_category.objects ?candidate_set" in rewritten_queries[0]
    assert "fb:m.0l6tq fb:astronomy.celestial_object_category.objects ?candidate_set" not in rewritten_queries[0]
    assert rewritten_code != generated_code


def test_same_attempt_anchor_entity_retry_reexecutes_with_resolved_id() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE {
  ?anchor fb:type.object.name ?anchor_name .
  FILTER(LCASE(?anchor_name) = "southern min")
  ?candidate_set fb:language.language_dialect.language ?anchor .
}"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'
    original_result = PALInvocationResult(
        success=True,
        payload={
            "results": {
                "bindings": [
                    {
                        "count": {
                            "type": "literal",
                            "value": "0",
                        }
                    }
                ]
            }
        },
    )
    retried_result = PALInvocationResult(
        success=True,
        payload={
            "results": {
                "bindings": [
                    {
                        "count": {
                            "type": "literal",
                            "value": "4",
                        }
                    }
                ]
            }
        },
    )

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=retried_result,
    ) as retry_exec:
        rewritten_code, new_result, repaired_plan, used_retry = (
            controller._retry_execution_with_resolved_anchor_ids(
                generated_code=generated_code,
                invocation_result=original_result,
                query_plan={
                    "answer_mode": "count",
                    "anchored_entities": [
                        {
                            "surface": "Southern Min",
                            "chosen_alias": "Southern Min",
                            "role": "anchor",
                        }
                    ],
                },
                anchor_probe_results=[
                    AnchorProbeResult(
                        anchor_name="Southern Min",
                        entity_count=1,
                        path_count=8,
                        relation_probed="language.language_dialect.language",
                        anchor_position="object",
                        resolved_entity_id="m.01c44b",
                    )
                ],
            )
        )

    assert used_retry is True
    assert retry_exec.called
    assert "VALUES ?anchor { fb:m.01c44b }" in rewritten_code
    assert new_result is retried_result
    assert repaired_plan["anchored_entities"][0]["resolved_entity_id"] == "m.01c44b"


def test_same_attempt_anchor_entity_retry_adopts_ambiguous_anchor_pin_even_without_count_gain() -> None:
    controller = _make_controller()

    query_text = """PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE {
  ?anchor fb:type.object.name ?anchor_name .
  FILTER(LCASE(STR(?anchor_name)) = "southern min")
  ?candidate_set fb:language.language_dialect.language ?anchor .
}"""
    generated_code = f'query = """{query_text}"""\nwrapper.setQuery(query)\n'
    original_result = PALInvocationResult(
        success=True,
        payload={
            "results": {
                "bindings": [
                    {
                        "count": {
                            "type": "literal",
                            "value": "4",
                        }
                    }
                ]
            }
        },
    )
    retried_result = PALInvocationResult(
        success=True,
        payload={
            "results": {
                "bindings": [
                    {
                        "count": {
                            "type": "literal",
                            "value": "4",
                        }
                    }
                ]
            }
        },
    )

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=retried_result,
    ) as retry_exec:
        rewritten_code, new_result, repaired_plan, used_retry = (
            controller._retry_execution_with_resolved_anchor_ids(
                generated_code=generated_code,
                invocation_result=original_result,
                query_plan={
                    "answer_mode": "count",
                    "anchored_entities": [
                        {
                            "surface": "Southern Min",
                            "chosen_alias": "Southern Min",
                            "role": "anchor",
                        }
                    ],
                },
                anchor_probe_results=[
                    AnchorProbeResult(
                        anchor_name="Southern Min",
                        entity_count=2,
                        path_count=4,
                        relation_probed="language.language_dialect.language",
                        anchor_position="object",
                        resolved_entity_id="m.01c44b",
                    )
                ],
            )
        )

    assert used_retry is True
    assert retry_exec.called
    assert "VALUES ?anchor { fb:m.01c44b }" in rewritten_code
    assert new_result is retried_result
    assert repaired_plan["anchored_entities"][0]["resolved_entity_id"] == "m.01c44b"


def test_retry_same_plan_after_alias_repair_for_pure_anchor_not_found_count() -> None:
    controller = _make_controller()

    should_retry = controller._should_retry_same_plan_after_alias_repair(
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "anchor_not_found:'republic of brazil'",
                "probe_count:'republic of brazil'=0",
                "count_set_anchor_not_found",
            ],
        ),
        alias_repair_feedback=[
            "anchor_alias_override:republic of brazil=>Brazil",
        ],
    )

    assert should_retry is True


def test_merge_feedback_items_prefers_latest_anchor_override() -> None:
    controller = _make_controller()

    merged = controller._merge_feedback_items(
        [
            "anchor_alias_override:Nebula=>Nebula Kepiting",
            "anchor_entity_override:Nebula=>m.0l6tq",
            "repair_hint:use_selected_aliases",
        ],
        [
            "anchor_alias_override:Nebula=>Nebula",
            "anchor_entity_override:Nebula=>m.05fny",
            "repair_hint:bind_anchor_to_resolved_entity_id",
        ],
    )

    assert "anchor_alias_override:Nebula=>Nebula" in merged
    assert "anchor_entity_override:Nebula=>m.05fny" in merged
    assert "anchor_alias_override:Nebula=>Nebula Kepiting" not in merged
    assert "anchor_entity_override:Nebula=>m.0l6tq" not in merged


def test_retry_same_plan_after_alias_repair_retries_path_empty_counts_when_anchor_changes() -> None:
    controller = _make_controller()

    should_retry = controller._should_retry_same_plan_after_alias_repair(
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "anchor_path_empty:'Brazil':biology.animal_breed.country_of_origin",
                "count_set_path_empty",
            ],
        ),
        alias_repair_feedback=[
            "anchor_alias_override:republic of brazil=>Brazil",
        ],
    )

    assert should_retry is True


def test_retry_same_plan_after_alias_repair_yields_to_structural_repair_feedback() -> None:
    controller = _make_controller()

    should_retry = controller._should_retry_same_plan_after_alias_repair(
        verdict=PlausibilityVerdict(
            verdict="repairable_anchor_path_empty",
            reasons=[
                "anchor_path_empty:'Saxe-Coburg and Gotha':royalty.kingdom.monarchs",
            ],
        ),
        alias_repair_feedback=[
            "anchor_entity_override:saxe-coburg-gotha=>m.01f0_j",
        ],
        repair_feedback=[
            "plausibility_feedback:dynamic_grounding_augmented — fresh dynamic candidates were added",
            "dead_relation_suppressed:royalty.kingdom.monarchs[anchor=Saxe-Coburg and Gotha, direction=forward, from_role=anchor, to_role=answer]",
        ],
    )

    assert should_retry is False


def test_repair_feedback_constraints_reject_count_that_drops_preserved_relation_family() -> None:
    controller = _make_controller()

    constrained = controller._apply_repair_feedback_constraints(
        query_plan={
            "answer_mode": "count",
            "relation_paths": [
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "forward",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        verdict=PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=[]),
        repair_feedback=[
            "repair_hint:preserve_count_target_family_after_pivot",
            "relation_still_available_in_other_roles:biology.organism.diseases_transmitted",
        ],
    )

    assert constrained.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert (
        "count_repair_dropped_preserved_relation_family:"
        "biology.organism.diseases_transmitted"
    ) in constrained.reasons


def test_repair_feedback_constraints_allow_count_that_preserves_relation_family() -> None:
    controller = _make_controller()

    constrained = controller._apply_repair_feedback_constraints(
        query_plan={
            "answer_mode": "count",
            "relation_paths": [
                {
                    "relation": "location.country.form_of_government",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "constraint_value",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "government.government_position_held.office_holder",
                    "direction": "reverse",
                    "from_role": "constraint_value",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
            ],
        },
        verdict=PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=[]),
        repair_feedback=[
            "repair_hint:preserve_count_target_family_after_pivot",
            "relation_still_available_in_other_roles:government.government_position_held.office_holder",
        ],
    )

    assert constrained.verdict == VERDICT_ACCEPTED
    assert constrained.reasons == []


def test_repair_feedback_constraints_reject_dynamic_only_direct_count_after_grounded_failure() -> None:
    controller = _make_controller()

    constrained = controller._apply_repair_feedback_constraints(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "infectious diseases",
            "relation_paths": [
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "reverse",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        verdict=PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=[]),
        repair_feedback=[
            "dead_relation_suppressed:biology.organism.diseases_transmitted[anchor=Aedes aegypti, direction=forward, from_role=anchor, to_role=count_set]",
            "anchor_path_empty:'Aedes aegypti':biology.organism.diseases_transmitted",
            "count_set_path_empty",
        ],
    )

    assert constrained.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert (
        "count_repair_collapsed_to_dynamic_only_after_grounded_count_failed"
        in constrained.reasons
    )


def test_repair_feedback_constraints_allow_dynamic_only_direct_count_for_simple_target() -> None:
    controller = _make_controller()

    constrained = controller._apply_repair_feedback_constraints(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "albums",
            "relation_paths": [
                {
                    "relation": "music.artist.album",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
        },
        verdict=PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=[]),
        repair_feedback=[
            "dead_relation_suppressed:music.artist.track[anchor=Prince, direction=forward, from_role=anchor, to_role=count_set]",
            "anchor_path_empty:'Prince':music.artist.track",
        ],
    )

    assert constrained.verdict == VERDICT_ACCEPTED
    assert constrained.reasons == []


def test_apply_probe_guided_anchor_entity_repairs_adds_resolved_entity_override() -> None:
    controller = _make_controller()

    repaired_query_plan, feedback = controller._apply_probe_guided_anchor_entity_repairs(
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Southern Min",
                    "chosen_alias": "Southern Min",
                    "role": "anchor",
                }
            ]
        },
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Southern Min",
                entity_count=3,
                path_count=8,
                relation_probed="language.language_dialect.language",
                anchor_position="object",
                resolved_entity_id="m.01c44b",
            )
        ],
    )

    assert (
        repaired_query_plan["anchored_entities"][0]["resolved_entity_id"] == "m.01c44b"
    )
    assert "anchor_entity_override:Southern Min=>m.01c44b" in feedback


def test_question_interpretation_grounding_entities_use_named_entities_only() -> None:
    controller = _make_controller()
    interpretation = {
        "question_inputs": [
            {
                "surface": "research project",
                "kind": "class_phrase",
                "role_hint": "type_set",
                "reason": "class/category qualifier",
            },
            {
                "surface": "Cancer Centers",
                "kind": "answer_target",
                "role_hint": "answer_target",
                "reason": "answer head",
            },
            {
                "surface": "RS-27A",
                "kind": "named_entity",
                "role_hint": "anchor",
                "reason": "engine anchor",
            },
        ]
    }

    assert controller._extract_grounding_entities_from_question_interpretation(
        interpretation
    ) == ["RS-27A"]


def test_question_interpretation_grounding_entities_skip_class_typed_duplicates() -> None:
    controller = _make_controller()
    interpretation = {
        "question_inputs": [
            {
                "surface": "research project",
                "kind": "named_entity",
                "role_hint": "anchor",
                "reason": "explicit entity payload",
            },
            {
                "surface": "research project",
                "kind": "class_phrase",
                "role_hint": "type_set",
                "reason": "class/category qualifier",
            },
            {
                "surface": "cancer centers",
                "kind": "answer_target",
                "role_hint": "answer_target",
                "reason": "answer head",
            },
        ]
    }

    assert controller._extract_grounding_entities_from_question_interpretation(
        interpretation
    ) == []


def test_question_interpretation_grounding_entities_keep_multi_anchor_answer_target_overlap() -> None:
    controller = _make_controller()
    interpretation = {
        "question_inputs": [
            {
                "surface": "Dutcher Crossing Winery",
                "kind": "named_entity",
                "role_hint": "anchor_a",
                "reason": "explicit entity payload",
            },
            {
                "surface": "Red Wine",
                "kind": "named_entity",
                "role_hint": "anchor_b",
                "reason": "explicit entity payload",
            },
            {
                "surface": "red wine",
                "kind": "answer_target",
                "role_hint": "answer_target",
                "reason": "derived answer target phrase",
            },
        ]
    }

    assert controller._extract_grounding_entities_from_question_interpretation(
        interpretation
    ) == ["Dutcher Crossing Winery", "Red Wine"]


def test_structured_question_inputs_suppress_raw_entity_fallback_for_grounding() -> None:
    controller = _make_controller()
    interpretation = controller._build_question_interpretation(
        question_text="what is the number of research project cancer centers?",
        explicit_entities=["research project"],
        answer_target_phrase=controller._extract_answer_target_phrase(
            "what is the number of research project cancer centers?"
        ),
    )

    interpreted_grounding_entities = (
        controller._extract_grounding_entities_from_question_interpretation(
            interpretation
        )
    )
    has_structured_non_entity_inputs = any(
        str(item.get("kind") or "").strip()
        in {"class_phrase", "type_constraint", "shared_attribute"}
        for item in interpretation["question_inputs"]
    )
    grounding_entities = interpreted_grounding_entities or (
        [] if has_structured_non_entity_inputs else ["research project"]
    )

    assert grounding_entities == []


def test_extract_answer_target_phrase_handles_number_of_questions() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "what is the number of research project cancer centers?"
        )
        == "research project cancer centers"
    )


def test_extract_answer_target_phrase_handles_type_of_questions() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "what type of fuel ran the engine on rs-27a?"
        )
        == "fuel"
    )


def test_extract_answer_target_phrase_handles_embedded_kind_of_attribute_clause() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "maltese dog and papillon have what kind of temperament?"
        )
        == "temperament"
    )


def test_extract_answer_target_phrase_handles_preceded_by_clause() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "what is the royal line preceded by the house of lancaster and succeeded by tudor dynasty?"
        )
        == "royal line"
    )


def test_extract_answer_target_phrase_handles_superlative_track_questions() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "david han's longest release track of recordings is what?"
        )
        == "release track"
    )
    assert (
        controller._extract_answer_target_phrase(
            "what release track is the longest among recordings by david han?"
        )
        == "release track"
    )


def test_extract_answer_target_phrase_prefers_count_target_over_keyword_verb_match() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "how many game expansions has valve corp released?"
        )
        == "game expansions"
    )


def test_extract_answer_target_phrase_handles_was_the_amount_of_questions() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "what was the amount of key designers that the computer designed by richard altwasser have?"
        )
        == "key designers"
    )


def test_extract_answer_target_phrase_handles_trailing_preposition_which_questions() -> None:
    controller = _make_controller()

    assert (
        controller._extract_answer_target_phrase(
            "john jordan is one of the leaders of which wine producer?"
        )
        == "wine producer"
    )


def test_build_entity_alias_candidates_strips_common_geopolitical_prefixes() -> None:
    controller = _make_controller()

    aliases = controller._build_entity_alias_candidates("republic of brazil")

    assert "Brazil" in aliases
    assert aliases.index("Brazil") < aliases.index("brazil")


def test_build_entity_alias_candidates_strips_regional_org_suffix_for_single_brand_name() -> None:
    controller = _make_controller()

    aliases = controller._build_entity_alias_candidates("sega of japan")

    assert "Sega" in aliases
    assert "sega" in aliases


def test_infer_query_shape_prefers_shared_type_intersection_with_shared_type_cues() -> None:
    controller = _make_controller()

    query_shape = controller._infer_query_shape(
        question_text="what other types of collections are in the same category as patch collecting collection?",
        entities=["Patch collecting collection"],
        answer_target_phrase="other types of collections",
        question_inputs=[
            {
                "surface": "category",
                "kind": "shared_attribute",
                "role_hint": "shared_attribute",
            },
            {
                "surface": "Patch collecting collection",
                "kind": "named_entity",
                "role_hint": "anchor",
            },
        ],
    )

    assert query_shape == "shared_type_intersection"


def test_infer_query_shape_prefers_shared_type_intersection_for_multi_anchor_type_questions() -> None:
    controller = _make_controller()

    query_shape = controller._infer_query_shape(
        question_text="list all types of museums that are of the same type as the museum of modern art and smithsonian institution",
        entities=["the museum of modern art", "Smithsonian Institution"],
        answer_target_phrase="types of museums",
        question_inputs=[
            {
                "surface": "the museum of modern art",
                "kind": "named_entity",
                "role_hint": "anchor_a",
            },
            {
                "surface": "Smithsonian Institution",
                "kind": "named_entity",
                "role_hint": "anchor_b",
            },
        ],
    )

    assert query_shape == "shared_type_intersection"


def test_infer_query_shape_uses_multi_anchor_intersection_for_relation_encoded_type_projection() -> None:
    controller = _make_controller()

    query_shape = controller._infer_query_shape(
        question_text=(
            "what is the sensor type of a digital camera that has the color filter array type "
            "of bayer and iso settings of 120?"
        ),
        entities=["bayer", "120"],
        answer_target_phrase="sensor type",
        question_inputs=[
            {
                "surface": "bayer",
                "kind": "attribute_value",
                "role_hint": "constraint_value",
            },
            {
                "surface": "120",
                "kind": "attribute_value",
                "role_hint": "constraint_value",
            },
            {
                "surface": "sensor type",
                "kind": "answer_target",
                "role_hint": "answer_target",
            },
        ],
        grounded_relation_candidates=[
            {
                "relation": "digicams.digital_camera.sensor_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "answer",
                "from_role": "candidate_set",
                "to_role": "answer",
                "use_when": "project a digital camera product to its sensor type",
            },
            {
                "relation": "digicams.digital_camera.color_filter_array_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "use_when": "constrain a digital camera product by its color filter array type such as Bayer filter",
            },
        ],
    )

    assert query_shape == "multi_anchor_intersection"


def test_infer_query_shape_prefers_joined_count_when_count_question_has_constraint_value() -> None:
    controller = _make_controller()

    query_shape = controller._infer_query_shape(
        question_text=(
            "what is the number of film characters that are with educators occupation "
            "and neyaphem species?"
        ),
        entities=["Neyaphem"],
        answer_target_phrase="film characters",
        question_inputs=[
            {
                "surface": "educators",
                "kind": "attribute_value",
                "role_hint": "constraint_value",
            },
            {
                "surface": "Neyaphem",
                "kind": "named_entity",
                "role_hint": "anchor_b",
            },
            {
                "surface": "film characters",
                "kind": "answer_target",
                "role_hint": "answer_target",
            },
        ],
    )

    assert query_shape == "count_over_joined_set"


def test_infer_query_shape_detects_internal_how_many_count_cues() -> None:
    controller = _make_controller()

    direct_shape = controller._infer_query_shape(
        question_text="the rank of goro has been given to how many book characters?",
        entities=["Goro"],
        answer_target_phrase="book characters",
        question_inputs=[
            {
                "surface": "Goro",
                "kind": "named_entity",
                "role_hint": "anchor",
            },
            {
                "surface": "book characters",
                "kind": "answer_target",
                "role_hint": "answer_target",
            },
        ],
    )
    joined_shape = controller._infer_query_shape(
        question_text=(
            "the czecho-slovakia has how many breeds of dogs that has the same "
            "temperament as the cairn terriers?"
        ),
        entities=["czecho-slovakia", "cairn terriers"],
        answer_target_phrase="breeds",
        question_inputs=[
            {
                "surface": "czecho-slovakia",
                "kind": "named_entity",
                "role_hint": "anchor_a",
            },
            {
                "surface": "cairn terriers",
                "kind": "named_entity",
                "role_hint": "anchor_b",
            },
            {
                "surface": "temperament",
                "kind": "shared_attribute",
                "role_hint": "shared_attribute",
            },
            {
                "surface": "breeds",
                "kind": "answer_target",
                "role_hint": "answer_target",
            },
        ],
    )

    assert direct_shape == "count_over_direct_relation"
    assert joined_shape == "count_over_joined_set"


def test_superlative_anchor_alternative_repair_prefers_grounded_objects_relation() -> None:
    controller = _make_controller()

    rewritten = controller._build_superlative_anchor_alternative_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Nebula", "chosen_alias": "m.05fny", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object.category",
                    "direction": "reverse",
                    "from": "category",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "astronomy.celestial_object.cosmological_distance",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "ordering_attribute",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "ordering_attribute": {
                "relation": "astronomy.celestial_object.cosmological_distance",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "ordering_attribute",
            },
            "ordering_direction": "max",
            "projection": ["candidate_set"],
            "strategy": "Use the category relation then order by distance.",
        },
        relation_grounding=[
            {
                "relation": "astronomy.celestial_object.category",
                "direction": "reverse",
                "from": "category",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "astronomy.celestial_object_category.objects",
                "direction": "reverse",
                "from": "category",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "astronomy.celestial_object.cosmological_distance",
                "direction": "forward",
                "from": "candidate_set",
                "to": "ordering_attribute",
                "from_role": "candidate_set",
                "to_role": "ordering_attribute",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Nebula",
                entity_count=1,
                path_count=0,
                relation_probed="astronomy.celestial_object.category",
                anchor_position="subject",
                resolved_entity_id="m.05fny",
            )
        ],
    )

    assert rewritten is not None
    assert rewritten["relation_paths"][0]["relation"] == "astronomy.celestial_object_category.objects"


def test_superlative_anchor_alternative_repair_prefers_grounded_objects_relation_when_current_path_is_live() -> None:
    controller = _make_controller()

    rewritten = controller._build_superlative_anchor_alternative_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Nebula", "chosen_alias": "m.05fny", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object.category",
                    "direction": "reverse",
                    "from": "category",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "astronomy.celestial_object.cosmological_distance",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "ordering_attribute",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "ordering_attribute": {
                "relation": "astronomy.celestial_object.cosmological_distance",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "ordering_attribute",
            },
            "ordering_direction": "max",
            "projection": ["candidate_set"],
            "strategy": "Use the category relation then order by distance.",
        },
        relation_grounding=[
            {
                "relation": "astronomy.celestial_object.category",
                "direction": "reverse",
                "from": "category",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "astronomy.celestial_object_category.objects",
                "direction": "reverse",
                "from": "category",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "astronomy.celestial_object.cosmological_distance",
                "direction": "forward",
                "from": "candidate_set",
                "to": "ordering_attribute",
                "from_role": "candidate_set",
                "to_role": "ordering_attribute",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Nebula",
                entity_count=1,
                path_count=111,
                relation_probed="astronomy.celestial_object.category",
                anchor_position="object",
                resolved_entity_id="m.05fny",
            )
        ],
    )

    assert rewritten is not None
    assert rewritten["relation_paths"][0]["relation"] == "astronomy.celestial_object_category.objects"


def test_infer_query_shape_treats_calendar_and_nebula_questions_as_superlatives() -> None:
    controller = _make_controller()

    calendar_shape = controller._infer_query_shape(
        question_text="based on the information within the gregorian calendar what is the last day of the week?",
        entities=["Gregorian calendar"],
        answer_target_phrase="last day",
        question_inputs=[
            {"surface": "Gregorian calendar", "kind": "named_entity", "role_hint": "anchor"},
            {"surface": "last", "kind": "ordering_cue", "role_hint": "ordering_attribute"},
        ],
    )
    nebula_shape = controller._infer_query_shape(
        question_text="what's the name of the nebula that's farthest away from us?",
        entities=["Nebula"],
        answer_target_phrase="nebula",
        question_inputs=[
            {"surface": "Nebula", "kind": "named_entity", "role_hint": "anchor"},
            {"surface": "farthest", "kind": "ordering_cue", "role_hint": "ordering_attribute"},
        ],
    )

    assert calendar_shape == "superlative_chain"
    assert nebula_shape == "superlative_chain"


def test_validate_pal_execution_treats_anchor_probe_as_optional_for_type_set_only_count_plan() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "research project",
                    "chosen_alias": "research project",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "type_set",
                    "to_role": "count_set",
                }
            ],
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "type_node",
                    }
                ],
            },
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?type_node fb:type.object.name \"research project cancer center\" . "
            "?type_node fb:type.type.instance ?candidate_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["research project"],
        anchor_probe_results=[AnchorProbeResult(anchor_name="research project", entity_count=0)],
    )

    assert "anchor_not_found:'research project'" not in verdict.reasons


def test_generic_type_relation_candidates_added_for_class_or_type_questions() -> None:
    controller = _make_controller()

    augmented = controller._augment_generic_type_relation_candidates(
        relation_candidates=[],
        answer_target_phrase="types of collections",
        question_interpretation={
            "question_inputs": [
                {
                    "surface": "types of collections",
                    "kind": "answer_target",
                    "role_hint": "answer_target",
                    "reason": "answer target",
                },
                {
                    "surface": "category",
                    "kind": "shared_attribute",
                    "role_hint": "shared_attribute",
                    "reason": "shared category cue",
                },
            ]
        },
    )

    relations = {str(item.get("relation") or "") for item in augmented}
    relation_roles = {
        (
            str(item.get("relation") or ""),
            str(item.get("from_role") or ""),
            str(item.get("to_role") or ""),
        )
        for item in augmented
    }
    assert "type.object.type" in relations
    assert "type.type.instance" in relations
    assert ("type.object.type", "candidate_set", "type_set") in relation_roles
    assert ("type.type.instance", "type_set", "candidate_set") in relation_roles


def test_grounding_card_surfaces_question_inputs_and_scaffolds() -> None:
    controller = _make_controller()

    grounding_card = controller._build_pal_grounding_card(
        "Question: what is the number of research project cancer centers?",
        relation_grounding=[],
        question_interpretation={
            "question_inputs": [
                {
                    "surface": "research project",
                    "kind": "class_phrase",
                    "role_hint": "type_set",
                    "reason": "class/category qualifier extracted from answer target",
                },
                {
                    "surface": "cancer centers",
                    "kind": "answer_target",
                    "role_hint": "answer_target",
                    "reason": "answer head noun phrase",
                },
            ],
            "preferred_scaffolds": [
                {
                    "name": "class_filtered_count",
                    "priority": 1,
                    "reason": "count question with a derived class/category phrase",
                }
            ],
        },
    )

    assert "- question_inputs:" in grounding_card
    assert "class_filtered_count" in grounding_card
    assert "research project" in grounding_card


def test_count_question_interpretation_keeps_answer_target_soft_for_single_anchor_counts() -> None:
    controller = _make_controller()

    question_text = "Question: how many songwriters work in the percussionist profession?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["Percussionist"],
        answer_target_phrase=answer_target,
    )

    assert answer_target == "songwriters"
    assert ("songwriters", "answer_target", "answer_target") in {
        (
            str(item.get("surface") or ""),
            str(item.get("kind") or ""),
            str(item.get("role_hint") or ""),
        )
        for item in interpretation["question_inputs"]
    }
    assert not any(
        str(item.get("kind") or "") == "class_phrase"
        for item in interpretation["question_inputs"]
    )
    assert interpretation["preferred_scaffolds"][0]["name"] == "direct_count"


def test_count_question_interpretation_does_not_split_compound_answer_target() -> None:
    controller = _make_controller()

    question_text = (
        "Question: what amount of comic book writers have a profession of "
        "documentary filmmaker?"
    )
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["Documentary Filmmaker"],
        answer_target_phrase=answer_target,
    )

    surfaces = {
        (
            str(item.get("surface") or ""),
            str(item.get("kind") or ""),
            str(item.get("role_hint") or ""),
        )
        for item in interpretation["question_inputs"]
    }

    assert answer_target == "comic book writers"
    assert ("comic book writers", "answer_target", "answer_target") in surfaces
    assert ("book writers", "answer_target", "answer_target") not in surfaces
    assert ("comic", "class_phrase", "type_set") not in surfaces


def test_count_question_interpretation_keeps_simple_direct_count_without_class_promotion() -> None:
    controller = _make_controller()

    question_text = "Question: how many language dialects does southern min have?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["Southern Min"],
        answer_target_phrase=answer_target,
    )

    assert answer_target == "language dialects"
    assert not any(
        str(item.get("kind") or "") == "class_phrase"
        for item in interpretation["question_inputs"]
    )
    assert interpretation["preferred_scaffolds"][0]["name"] == "direct_count"


def test_grounding_refinement_drops_attribute_qualifier_class_phrase() -> None:
    controller = _make_controller()

    question_text = "Question: what is the camera sensor type of panasonic lumix brand digital camera?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["panasonic lumix"],
        answer_target_phrase=answer_target,
    )

    assert ("camera", "class_phrase", "type_set") in {
        (
            str(item.get("surface") or ""),
            str(item.get("kind") or ""),
            str(item.get("role_hint") or ""),
        )
        for item in interpretation["question_inputs"]
    }

    refined = controller._refine_question_interpretation_with_grounding(
        question_text=question_text,
        answer_target_phrase=answer_target,
        question_interpretation=interpretation,
        grounded_relation_candidates=[
            {
                "relation": "digicams.digital_camera.sensor_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "answer",
                "from_role": "candidate_set",
                "to_role": "answer",
                "use_when": "project a digital camera product to its sensor type",
            }
        ],
    )

    surfaces = {
        (
            str(item.get("surface") or ""),
            str(item.get("kind") or ""),
            str(item.get("role_hint") or ""),
        )
        for item in refined["question_inputs"]
    }
    scaffold_names = {
        str(item.get("name") or "")
        for item in refined["preferred_scaffolds"]
    }

    assert ("camera", "class_phrase", "type_set") not in surfaces
    assert "type_instance_lookup" not in scaffold_names
    assert "shared_type_lookup" not in scaffold_names


def test_grounding_refinement_drops_shared_type_lookup_for_relation_encoded_sensor_type_projection() -> None:
    controller = _make_controller()

    question_text = (
        "Question: what is the sensor type of a digital camera that has the color filter array type "
        "of bayer and iso settings of 120?"
    )
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["bayer", "120"],
        answer_target_phrase=answer_target,
    )

    refined = controller._refine_question_interpretation_with_grounding(
        question_text=question_text,
        answer_target_phrase=answer_target,
        question_interpretation=interpretation,
        grounded_relation_candidates=[
            {
                "relation": "digicams.digital_camera.sensor_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "answer",
                "from_role": "candidate_set",
                "to_role": "answer",
                "use_when": "project a digital camera product to its sensor type",
            },
            {
                "relation": "digicams.digital_camera.color_filter_array_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "use_when": "constrain a digital camera product by its color filter array type such as Bayer filter",
            },
            {
                "relation": "digicams.digital_camera.iso_setting",
                "direction": "forward",
                "from": "candidate_set",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "use_when": "constrain a digital camera product by its ISO setting",
            },
        ],
    )

    scaffold_names = {
        str(item.get("name") or "")
        for item in refined["preferred_scaffolds"]
    }

    assert "projected_answer_intersection" in scaffold_names
    assert "shared_type_lookup" not in scaffold_names


def test_answer_target_extraction_handles_recent_superlative_phrase() -> None:
    controller = _make_controller()

    target = controller._extract_answer_target_phrase(
        "Question: what was the most recently formed cyclone in the same category as hurricane dolly?"
    )

    assert target == "formed cyclone"


def test_domain_hints_include_meteorology_and_fictional_book_superlatives() -> None:
    controller = _make_controller()

    assert controller._infer_domain_hints(
        "Question: what was the most recently formed cyclone in the same category as hurricane dolly?"
    ) == ["meteorology", "cyclone"]
    assert controller._infer_domain_hints(
        "Question: which short story of the sacred band of stepsons universe universe is know to have the earliest copyright date?"
    ) == ["fictional_universe", "book"]


def test_domain_hints_include_cvg_release_region_questions() -> None:
    controller = _make_controller()

    assert controller._infer_domain_hints(
        "Question: virtual console, which is developed by sega of japan, was released where?"
    ) == ["software", "video game", "cvg"]


def test_domain_hints_include_calendar_and_nebula_superlatives() -> None:
    controller = _make_controller()

    assert controller._infer_domain_hints(
        "Question: based on the information within the gregorian calendar what is the last day of the week?"
    ) == ["time", "calendar"]
    assert controller._infer_domain_hints(
        "Question: what's the name of the nebula that's farthest away from us?"
    ) == ["astronomy", "nebula"]


def test_build_class_filtered_count_repair_plan_uses_answer_class_constraint() -> None:
    controller = _make_controller()

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "strategy": "Count the people reached from Percussionist.",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "Percussionist",
                    "to": "person",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
        },
        relation_grounding=[
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "songwriter",
                "to": "person",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_joined_set"
    assert rewritten["count_set_variable"] == "shared_answer"
    assert any(
        str(item.get("role") or "") == "type_set"
        and str(item.get("chosen_alias") or "") == "songwriter"
        for item in rewritten["anchored_entities"]
    )
    assert any(
        str(path.get("relation") or "") == "people.profession.people_with_this_profession"
        and str(path.get("from_role") or "") == "type_set"
        for path in rewritten["relation_paths"]
    )


def test_build_class_filtered_count_repair_plan_skips_explicit_constraint_queries() -> None:
    controller = _make_controller()

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: what is the number of film characters that are with educators "
            "occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "film characters",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "Neyaphem",
                    "role": "anchor_b",
                },
                {
                    "surface": "educators",
                    "chosen_alias": "educators",
                    "role": "constraint_value",
                },
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.species",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "Neyaphem",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                },
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "educators",
                    "to": "candidate_set",
                    "from_role": "constraint_value",
                    "to_role": "count_set",
                },
            ],
        },
        relation_grounding=[],
    )

    assert rewritten is None


def test_grounding_keeps_curated_count_candidates_ahead_of_dynamic_fallbacks() -> None:
    controller = _make_controller()
    interpretation = controller._build_question_interpretation(
        question_text="Question: how many songwriters work in the percussionist profession?",
        explicit_entities=["Percussionist"],
        answer_target_phrase="songwriters",
    )

    with patch.object(
        controller,
        "_build_grounded_relation_candidates",
        return_value=[
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "Percussionist",
                "to": "person",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "curated",
                "support": "curated_people_predicate",
            }
        ],
    ), patch.object(
        controller,
        "_probe_dynamic_relation_candidates",
        return_value=[
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "Percussionist",
                "to": "person",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            }
        ],
    ), patch.object(
        controller,
        "_probe_answer_class_dynamic_candidates",
        return_value=[],
    ):
        grounding = controller._build_grounded_relation_candidates_with_dynamic_fallback(
            task_question=(
                "Question: how many songwriters work in the percussionist profession?, "
                "Entities: ['Percussionist']"
            ),
            entities=["Percussionist"],
            answer_target_phrase="songwriters",
            domain_hints=["people"],
            question_interpretation=interpretation,
        )

    assert grounding[0]["relation"] == "people.person.profession"
    assert grounding[1]["relation"] == "people.profession.people_with_this_profession"


def test_apply_question_scaffold_rewrite_drops_answer_target_only_filter_for_direct_count() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_direct_relation",
            "strategy": "Count people in the Percussionist profession and also require songwriter.",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.person.profession",
                    "direction": "reverse",
                    "from": "person",
                    "to": "Percussionist",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "people.person.profession",
                    "direction": "forward",
                    "from": "person",
                    "to": "songwriter",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "candidate_set",
                        "notes": "candidate persons must have Percussionist profession",
                    }
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert rewritten["query_shape"] == "count_over_direct_relation"
    assert len(rewritten["relation_paths"]) == 1
    assert rewritten["relation_paths"][0]["relation"] == "people.person.profession"
    assert rewritten["relation_paths"][0]["from_role"] == "count_set"
    assert rewritten["relation_paths"][0]["to_role"] == "anchor"
    assert rewritten["count_set_variable"] == "person"
    assert "songwriter" not in rewritten["strategy"].lower()
    assert all(
        "songwriter" not in str(item).lower()
        for item in rewritten["plan_rationale"]
    )


def test_apply_question_scaffold_rewrite_drops_mislabeled_answer_target_filter_for_direct_count() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_direct_relation",
            "strategy": "Count people in the Percussionist profession and also require songwriter.",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "Percussionist",
                    "to": "candidate_person",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "people.person.profession",
                    "direction": "forward",
                    "from": "candidate_person",
                    "to": "songwriter",
                    "from_role": "anchor",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_person",
            "count_set_variable": "candidate_person",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "candidate_person",
                        "notes": "candidate persons must have Percussionist profession",
                    }
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert rewritten["query_shape"] == "count_over_direct_relation"
    assert len(rewritten["relation_paths"]) == 1
    assert rewritten["relation_paths"][0]["relation"] == "people.profession.people_with_this_profession"
    assert rewritten["count_set_variable"] == "candidate_person"
    assert "songwriter" not in rewritten["strategy"].lower()


def test_apply_question_scaffold_rewrite_keeps_surface_anchor_joined_type_filter() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_joined_set",
            "strategy": "Count videogames published by Valve and filter to game expansions.",
            "anchored_entities": [
                {
                    "surface": "valve corp",
                    "chosen_alias": "valve corp",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "cvg.computer_videogame.publisher",
                    "direction": "reverse",
                    "from": "videogame",
                    "to": "valve corp",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type_filter:game_expansion",
                    "direction": "forward",
                    "from": "videogame",
                    "to": "expansion",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "exploratory",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "videogame",
            "count_set_variable": "videogame",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "videogame",
                        "notes": "anchor constrains the videogame set",
                    }
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert rewritten["query_shape"] == "count_over_joined_set"
    assert len(rewritten["relation_paths"]) == 2
    assert any(
        str(path.get("relation") or "").strip() == "type_filter:game_expansion"
        for path in rewritten["relation_paths"]
    )
    assert rewritten["count_set_variable"] == "videogame"
    assert "game expansion" in rewritten["strategy"].lower()


def test_grounding_candidates_include_cvg_publisher_release_relations() -> None:
    controller = _make_controller()

    candidates = controller._build_grounded_relation_candidates(
        "Question: how many game expansions has valve corp released?, Entities: ['valve corp']"
    )

    assert any(
        candidate.get("relation") == "cvg.cvg_publisher.games_published"
        for candidate in candidates
    )
    assert any(
        candidate.get("relation") == "cvg.cvg_publisher.game_versions_published"
        for candidate in candidates
    )


def test_grounded_relation_sort_prefers_publisher_relations_for_has_released_game_query() -> None:
    controller = _make_controller()

    ranked = controller._sort_grounded_relation_candidates_by_semantic_fit(
        relation_candidates=[
            {
                "relation": "cvg.computer_game_distribution_system.games_distributed",
                "direction": "reverse",
                "from": "distribution_system",
                "to": "candidate_set",
                "grounding_source": "curated",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "support": "curated_cvg_predicate",
                "use_when": "retrieve game versions distributed by a known distribution system",
            },
            {
                "relation": "cvg.cvg_developer.game_versions_developed",
                "direction": "reverse",
                "from": "developer",
                "to": "candidate_set",
                "grounding_source": "curated",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "support": "curated_cvg_predicate",
                "use_when": "retrieve game versions developed by a known developer",
            },
            {
                "relation": "cvg.cvg_publisher.games_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "grounding_source": "curated",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "support": "curated_cvg_predicate",
                "use_when": "retrieve games published or released by a known publisher",
            },
        ],
        question_text="Question: how many game expansions has valve corp released?",
        answer_target_phrase="game expansions",
        question_inputs=[],
    )

    assert ranked[0]["relation"] == "cvg.cvg_publisher.games_published"


def test_count_answer_target_head_match_requires_full_semantic_head() -> None:
    controller = _make_controller()

    assert not controller._count_answer_target_head_is_encoded_in_relation(
        answer_target_phrase="game expansions",
        relation_candidates=[
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve game versions published or released by a known publisher",
            }
        ],
    )
    assert controller._count_answer_target_head_is_encoded_in_relation(
        answer_target_phrase="infectious diseases",
        relation_candidates=[
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from": "disease",
                "to": "anchor",
                "from_role": "count_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
    )
    assert not controller._count_answer_target_head_is_encoded_in_relation(
        answer_target_phrase="artists recorded the contribution by lso",
        relation_candidates=[
            {
                "relation": "music.recording_contribution.contributor",
                "direction": "reverse",
                "from": "contribution",
                "to": "anchor",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
    )
    assert controller._count_answer_target_head_is_encoded_in_relation(
        answer_target_phrase="artists recorded the contribution by lso",
        relation_candidates=[
            {
                "relation": "music.recording.featured_artists",
                "direction": "reverse",
                "from": "recording",
                "to": "anchor",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
    )


def test_class_filtered_count_repair_plan_adds_explicit_constraint_for_game_expansions() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "strategy": "Count distinct game-version entities reached from the publisher anchor and filter them to game expansions.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "support": "curated_cvg_predicate",
                "use_when": "retrieve game versions published or released by a known publisher",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "type_set",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "exploratory",
                "reason": "apply a type/category filter to restrict the candidate game versions to those whose type matches game expansion",
            },
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "single_path",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "Use the publisher->game-version relation to produce the candidate set of game versions published by the anchor publisher",
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
        relation_grounding=query_plan["relation_paths"],
    )

    assert rewritten is not None
    assert len(rewritten["anchored_entities"]) == 2
    assert {
        str(path.get("relation") or "").strip() for path in rewritten["relation_paths"]
    } == {
        "cvg.cvg_publisher.game_versions_published",
        "type.object.type",
    }
    assert any(
        str(item.get("role") or "").strip() in {"constraint_value", "type_set", "shared_type"}
        and str(item.get("chosen_alias") or "").strip() == "game expansion"
        for item in rewritten["anchored_entities"]
    )
    assert any(
        str(item.get("anchor_role") or "").strip() in {"constraint_value", "type_set", "shared_type"}
        for item in rewritten["join_structure"]["anchor_constraints"]
    )
    assert "answer-class or answer-constraint filter" in " ".join(
        str(item.get("notes") or "").strip()
        for item in rewritten["join_structure"]["anchor_constraints"]
        if isinstance(item, dict)
    ).lower()


def test_select_class_filtered_count_candidate_prefers_type_like_relation_with_answer_target_overlap() -> None:
    controller = _make_controller()

    selected = controller._select_class_filtered_count_candidate(
        relation_grounding=[
            {
                "relation": "music.recording.length",
                "direction": "forward",
                "from": "shared_answer",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
                "reason": "unrelated attribute constraint",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "shared_answer",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "exploratory",
                "reason": "restrict to the game expansion type",
            },
        ],
        preferred_relation_root="cvg",
        answer_target_phrase="game expansions",
    )

    assert selected is not None
    assert selected["relation"] == "type.object.type"


def test_class_filtered_count_repair_requires_anchor_touched_count_path() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count songwriters who work in the percussionist profession.",
        "anchored_entities": [
            {
                "surface": "Percussionist",
                "chosen_alias": "m.02h66l4",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "profession",
                "to": "person",
                "from_role": "constraint_value",
                "to_role": "count_set",
                "grounding_source": "curated",
            }
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": (
                        "The primary anchor constrains the shared counted entity set "
                        "via people.profession.people_with_this_profession."
                    ),
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_class_filtered_count_repair_plan(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "profession",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
                "reason": "find the profession(s) of a person",
            }
        ],
    )

    assert rewritten is None


def test_relation_selected_direct_count_repair_prefers_top_grounded_anchor_family() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "strategy": "Count game versions and filter them to game expansions.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "cvg.game_version.version_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "version_type",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "exploratory",
            },
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "publisher count path",
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_relation_selected_direct_count_repair_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "cvg.cvg_publisher.games_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve games published or released by a known publisher",
            },
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve game versions published or released by a known publisher",
            },
        ],
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_direct_relation"
    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "cvg.cvg_publisher.games_published"
    ]
    assert rewritten["allow_exploratory_predicates"] is False
    assert "relation-selection hint" in rewritten["strategy"]


def test_relation_selected_direct_count_repair_can_rewrite_initial_direct_count_family() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count game versions published by the anchor.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            }
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "publisher count path",
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_relation_selected_direct_count_repair_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "cvg.cvg_publisher.games_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve games published or released by a known publisher",
            },
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve game versions published or released by a known publisher",
            },
        ],
    )

    assert rewritten is not None
    assert rewritten["query_shape"] == "count_over_direct_relation"
    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "cvg.cvg_publisher.games_published"
    ]
    assert "under-encodes the answer target" in " ".join(rewritten["plan_rationale"]).lower()


def test_relation_selected_direct_count_repair_uses_subject_probe_to_normalize_direction() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count game versions published by the anchor.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            }
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "publisher count path",
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_relation_selected_direct_count_repair_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "cvg.cvg_publisher.games_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve games published or released by a known publisher",
            },
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve game versions published or released by a known publisher",
            },
        ],
        anchor_probe_results=[
            {
                "anchor_name": "valve corp",
                "entity_count": 1,
                "found": True,
                "path_count": 80,
                "relation_probed": "cvg.cvg_publisher.games_published",
                "anchor_position": "subject",
                "resolved_entity_id": "m.0dwl2",
            }
        ],
    )

    assert rewritten is not None
    assert rewritten["relation_paths"][0]["relation"] == "cvg.cvg_publisher.games_published"
    assert rewritten["relation_paths"][0]["direction"] == "forward"


def test_relation_selected_direct_count_repair_uses_current_subject_probe_for_sibling_relation() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count game versions published by the anchor.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            }
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "publisher count path",
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_relation_selected_direct_count_repair_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "cvg.cvg_publisher.games_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve games published or released by a known publisher",
            },
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve game versions published or released by a known publisher",
            },
        ],
        anchor_probe_results=[
            {
                "anchor_name": "valve corp",
                "entity_count": 1,
                "found": True,
                "path_count": 35,
                "relation_probed": "cvg.cvg_publisher.game_versions_published",
                "anchor_position": "subject",
                "resolved_entity_id": "m.0dwl2",
            }
        ],
    )

    assert rewritten is not None
    assert rewritten["relation_paths"][0]["relation"] == "cvg.cvg_publisher.games_published"
    assert rewritten["relation_paths"][0]["direction"] == "forward"


def test_relation_selected_direct_count_repair_accepts_anchor_probe_objects() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count game versions published by the anchor.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            }
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "publisher count path",
                }
            ],
        },
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._build_relation_selected_direct_count_repair_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "cvg.cvg_publisher.games_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve games published or released by a known publisher",
            },
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "use_when": "retrieve game versions published or released by a known publisher",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="valve corp",
                entity_count=1,
                path_count=35,
                relation_probed="cvg.cvg_publisher.game_versions_published",
                anchor_position="subject",
                resolved_entity_id="m.0dwl2",
            )
        ],
    )

    assert rewritten is not None
    assert rewritten["relation_paths"][0]["direction"] == "forward"


def test_single_anchor_answer_target_count_rewrite_keeps_surface_anchor_semantics() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count videogames published by Valve and also require game expansion.",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "valve corp",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "cvg.computer_videogame.publisher",
                "direction": "reverse",
                "from": "videogame",
                "to": "valve corp",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "type_filter:game_expansion",
                "direction": "forward",
                "from": "videogame",
                "to": "expansion",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "exploratory",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "videogame",
        "count_set_variable": "videogame",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {"type": "count", "anchor_constraints": []},
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._rewrite_single_anchor_answer_target_count_plan(
        task_question=(
            "Question: how many game expansions has valve corp released?, "
            "Entities: ['valve corp']"
        ),
        query_plan=query_plan,
    )

    assert rewritten == query_plan


def test_single_anchor_answer_target_count_rewrite_keeps_explicit_semantic_inputs() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "Count the candidate set and also enforce the class filter.",
        "anchored_entities": [
            {
                "surface": "Percussionist",
                "chosen_alias": "Percussionist",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "people.person.profession",
                "direction": "reverse",
                "from": "person",
                "to": "Percussionist",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "songwriter",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "person",
        "count_set_variable": "person",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {"type": "count", "anchor_constraints": []},
        "projection": ["count"],
        "plan_rationale": [],
    }

    rewritten = controller._rewrite_single_anchor_answer_target_count_plan(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan=query_plan,
        question_interpretation={
            "question_inputs": [
                {"surface": "Percussionist", "kind": "named_entity", "role_hint": "anchor"},
                {"surface": "songwriters", "kind": "answer_target", "role_hint": "answer_target"},
                {"surface": "creative professionals", "kind": "class_phrase", "role_hint": "constraint_value"},
            ]
        },
    )

    assert rewritten == query_plan


def test_joined_count_answer_target_hint_preserves_multi_anchor_constraints() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "strategy": "Count positions with both filters.",
        "anchored_entities": [
            {
                "surface": "her majesty the queen",
                "chosen_alias": "her majesty the queen",
                "role": "anchor_a",
            },
            {
                "surface": "Cayman Islands",
                "chosen_alias": "Cayman Islands",
                "role": "anchor_b",
            },
        ],
        "relation_paths": [
            {
                "relation": "government.position.appointed_by",
                "direction": "forward",
                "from": "position",
                "to": "person",
                "from_role": "count_set",
                "to_role": "anchor_a",
                "grounding_source": "exploratory",
            },
            {
                "relation": "government.position.jurisdiction",
                "direction": "forward",
                "from": "position",
                "to": "cayman islands government position",
                "from_role": "count_set",
                "to_role": "constraint_value",
                "grounding_source": "exploratory",
            },
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor_a", "constrains_variable": "candidate_set", "notes": "queen filter"},
                {"anchor_role": "anchor_b", "constrains_variable": "candidate_set", "notes": "cayman filter"},
            ],
        },
        "projection": ["count"],
    }

    rewritten = controller._rewrite_joined_count_answer_target_hint_plan(
        task_question=(
            "Question: how many cayman islands government positions were appointed by "
            "her majesty the queen?, Entities: ['her majesty the queen', 'Cayman Islands']"
        ),
        query_plan=query_plan,
    )

    assert rewritten == query_plan


def test_joined_count_answer_target_hint_keeps_multi_anchor_paths_while_dropping_type_filter() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "strategy": "Count positions with two anchors and a redundant type filter.",
        "anchored_entities": [
            {
                "surface": "her majesty the queen",
                "chosen_alias": "her majesty the queen",
                "role": "anchor_a",
            },
            {
                "surface": "Cayman Islands",
                "chosen_alias": "Cayman Islands",
                "role": "anchor_b",
            },
        ],
        "relation_paths": [
            {
                "relation": "government.government_position_held.appointed_by",
                "direction": "reverse",
                "from": "held",
                "to": "her majesty the queen",
                "from_role": "candidate_set",
                "to_role": "anchor_a",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "government.government_position_held.jurisdiction_of_office",
                "direction": "reverse",
                "from": "jurisdiction",
                "to": "Cayman Islands",
                "from_role": "candidate_set",
                "to_role": "anchor_b",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "type_filter:government_position",
                "direction": "forward",
                "from": "candidate_set",
                "to": "government_position",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "exploratory",
            },
        ],
        "shared_answer_variable": "answer",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count_set",
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {"anchor_role": "anchor_a", "constrains_variable": "candidate_set", "notes": "queen filter"},
                {"anchor_role": "anchor_b", "constrains_variable": "candidate_set", "notes": "cayman filter"},
            ],
        },
        "projection": ["count"],
    }

    rewritten = controller._rewrite_joined_count_answer_target_hint_plan(
        task_question=(
            "Question: how many government positions in Cayman Islands were appointed by "
            "her majesty the queen?, Entities: ['her majesty the queen', 'Cayman Islands']"
        ),
        query_plan=query_plan,
    )

    assert len(rewritten["relation_paths"]) == 2
    assert {
        str(path.get("relation") or "").strip() for path in rewritten["relation_paths"]
    } == {
        "government.government_position_held.appointed_by",
        "government.government_position_held.jurisdiction_of_office",
    }
    assert [item["anchor_role"] for item in rewritten["join_structure"]["anchor_constraints"]] == [
        "anchor_a",
        "anchor_b",
    ]


def test_condition_such_as_rewrite_uses_parent_disease_chain() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: conditions such as pulmonary heart disease have how many "
            "prevention factors?, Entities: ['Pulmonary heart disease']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "prevention factors",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Pulmonary heart disease",
                    "chosen_alias": "Pulmonary heart disease",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "candidate_set", "notes": "anchor disease"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "medicine.disease.prevention_factors",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "count_set",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "medicine.disease.parent_disease",
        "medicine.disease.prevention_factors",
    ]
    assert rewritten["relation_paths"][0]["direction"] == "reverse"


def test_apply_question_scaffold_rewrite_drops_generic_constraint_path_when_no_second_input() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: how many songwriters work in the percussionist profession?, "
            "Entities: ['Percussionist']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "query_shape": "count_over_direct_relation",
            "strategy": "Count people in Percussionist and filter by profession.",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "Percussionist",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.person.profession",
                    "direction": "reverse",
                    "from": "person",
                    "to": "Percussionist",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "people.person.profession",
                    "direction": "forward",
                    "from": "person",
                    "to": "profession",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate",
            "count_set_variable": "candidate",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "candidate",
                        "notes": "Anchor constrains person candidates",
                    }
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert len(rewritten["relation_paths"]) == 1
    assert rewritten["relation_paths"][0]["relation"] == "people.person.profession"
    assert rewritten["count_set_variable"] == "person"
    assert "profession includes" not in rewritten["strategy"].lower()


def test_exhibition_subject_rewrite_counts_subjects_over_exhibition_type() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: there are how many exhibition subjects in international "
            "exhibition of modern art?, Entities: ['international exhibition of modern art']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "exhibition subjects",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "international exhibition of modern art",
                    "chosen_alias": "m.01_ggr",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "answer",
            "count_set_variable": "answer",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "exhibition", "notes": "direct exhibition subject count"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "exhibitions.exhibition.subjects",
                    "direction": "forward",
                    "from": "exhibition",
                    "to": "subject",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "exhibitions.exhibition.exhibition_types",
        "exhibitions.type_of_exhibition.exhibitions_of_this_type",
        "exhibitions.exhibition.subjects",
    ]


def test_apply_question_scaffold_rewrite_drops_redundant_joined_type_filter() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what is the number of film characters that are with educators "
            "occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "film characters",
            "query_shape": "count_over_joined_set",
            "strategy": "Count fictional characters with the requested species and occupation, plus a separate film-character type filter.",
            "anchored_entities": [
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "Neyaphem",
                    "role": "anchor_b",
                },
                {
                    "surface": "educators",
                    "chosen_alias": "Teacher",
                    "role": "constraint_value",
                },
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.species",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "Neyaphem",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "fictional_universe.fictional_character.occupation",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "Teacher",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from": "type",
                    "to": "film character",
                    "from_role": "type_set",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "candidate_set",
                        "notes": "species filter",
                    },
                    {
                        "anchor_role": "constraint_value",
                        "constrains_variable": "candidate_set",
                        "notes": "occupation filter",
                    },
                    {
                        "anchor_role": "type_set",
                        "constrains_variable": "candidate_set",
                        "notes": "answer class filter",
                    },
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert not any(
        str(path.get("relation") or "") == "type.type.instance"
        for path in rewritten["relation_paths"]
    )
    assert any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.occupation"
        for path in rewritten["relation_paths"]
    )


def test_apply_question_scaffold_rewrite_drops_semantic_joined_type_filter_variant() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what is the number of film characters that are with educators "
            "occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "film characters",
            "query_shape": "count_over_joined_set",
            "strategy": "Count characters with the requested species and occupation, plus a generic character type filter.",
            "anchored_entities": [
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "Neyaphem",
                    "role": "anchor_a",
                },
                {
                    "surface": "educators",
                    "chosen_alias": "Teacher",
                    "role": "constraint_value",
                },
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.species",
                    "direction": "reverse",
                    "from": "fictional_character",
                    "to": "Neyaphem",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "fictional_universe.fictional_character.occupation",
                    "direction": "forward",
                    "from": "fictional_character",
                    "to": "occupation",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from": "type",
                    "to": "fictional_character",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "candidate_set",
                        "notes": "species filter",
                    },
                    {
                        "anchor_role": "constraint_value",
                        "constrains_variable": "candidate_set",
                        "notes": "occupation filter",
                    },
                    {
                        "anchor_role": "type_set",
                        "constrains_variable": "candidate_set",
                        "notes": "answer class filter",
                    },
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert not any(
        str(path.get("relation") or "") == "type.type.instance"
        for path in rewritten["relation_paths"]
    )
    assert any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.species"
        for path in rewritten["relation_paths"]
    )


def test_fictional_character_joined_count_rewrite_binds_explicit_species_and_occupation() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what is the number of film characters that are with educators "
            "occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "film characters",
            "query_shape": "count_over_joined_set",
            "strategy": "Count fictional characters using generic constraint placeholders.",
            "allow_exploratory_predicates": True,
            "anchored_entities": [
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "m.09tc50",
                    "role": "anchor_b",
                },
                {
                    "surface": "educators",
                    "chosen_alias": "Teacher",
                    "role": "constraint_value",
                },
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.species",
                    "direction": "forward",
                    "from": "shared_answer",
                    "to": "species",
                    "from_role": "count_set",
                    "to_role": "candidate_set",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "fictional_universe.fictional_character.occupation",
                    "direction": "forward",
                    "from": "shared_answer",
                    "to": "occupation",
                    "from_role": "count_set",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_character.appears_in",
                    "direction": "forward",
                    "from": "shared_answer",
                    "to": "work",
                    "from_role": "count_set",
                    "to_role": "candidate_set",
                    "grounding_source": "exploratory",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_b", "constrains_variable": "shared_answer", "notes": "species filter"},
                    {"anchor_role": "constraint_value", "constrains_variable": "shared_answer", "notes": "occupation filter"},
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert rewritten["allow_exploratory_predicates"] is False
    assert rewritten["count_set_variable"] == "shared_answer"
    assert not any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.appears_in"
        for path in rewritten["relation_paths"]
    )
    assert any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.occupation"
        and str(path.get("to") or "") == "Teacher"
        and str(path.get("to_role") or "") == "constraint_value"
        for path in rewritten["relation_paths"]
    )
    assert any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.species"
        and str(path.get("to") or "") == "m.09tc50"
        and str(path.get("to_role") or "") == "anchor_b"
        for path in rewritten["relation_paths"]
    )


def test_fictional_character_joined_count_rewrite_replaces_world_species_detour() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what is the number of film characters that are with educators "
            "occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "film characters",
            "query_shape": "count_over_joined_set",
            "strategy": "Count characters by occupation and by world species.",
            "allow_exploratory_predicates": False,
            "anchored_entities": [
                {
                    "surface": "educators",
                    "chosen_alias": "Teacher",
                    "resolved_entity_id": "m.01d30f",
                    "role": "constraint_value",
                },
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "m.09tc50",
                    "resolved_entity_id": "m.09tc50",
                    "role": "anchor_b",
                },
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.occupation",
                    "direction": "forward",
                    "from": "fictional_character",
                    "to": "occupation",
                    "from_role": "count_set",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_universe.characters",
                    "direction": "reverse",
                    "from": "character",
                    "to": "fictional_world",
                    "from_role": "count_set",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "fictional_world",
                    "to": "species",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
            ],
            "shared_answer_variable": "character",
            "candidate_set_variable": "character",
            "count_set_variable": "character",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "constraint_value", "constrains_variable": "character", "notes": "occupation filter"},
                    {"anchor_role": "anchor_b", "constrains_variable": "character", "notes": "species filter"},
                ],
            },
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.occupation"
        and str(path.get("to") or "") == "Teacher"
        and str(path.get("to_role") or "") == "constraint_value"
        for path in rewritten["relation_paths"]
    )
    assert any(
        str(path.get("relation") or "") == "fictional_universe.fictional_character.species"
        and str(path.get("to") or "") == "m.09tc50"
        and str(path.get("to_role") or "") == "anchor_b"
        for path in rewritten["relation_paths"]
    )
    assert not any(
        str(path.get("relation") or "") == "fictional_universe.fictional_universe.characters"
        for path in rewritten["relation_paths"]
    )
    assert not any(
        str(path.get("relation") or "") == "fictional_universe.fictional_universe.species"
        for path in rewritten["relation_paths"]
    )


def test_exhibition_subject_question_keeps_direct_subject_count_plan() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: there are how many exhibition subjects in international "
            "exhibition of modern art?, Entities: ['international exhibition of modern art']"
        ),
        query_plan={
            "answer_mode": "count",
            "answer_type": "count",
            "answer_target_phrase": "exhibition subjects",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "international exhibition of modern art",
                    "chosen_alias": "m.01_ggr",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "answer",
            "count_set_variable": "answer",
            "ordering_attribute": {"direction": "forward"},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "exhibition", "notes": "direct exhibition subject count"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "exhibitions.exhibition.subjects",
                    "direction": "forward",
                    "from": "exhibition",
                    "to": "subject",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
            "plan_rationale": [],
        },
    )

    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "exhibitions.exhibition.subjects",
    ]


def test_infer_domain_hints_prefers_people_for_profession_questions() -> None:
    controller = _make_controller()

    assert controller._infer_domain_hints(
        "how many songwriters work in the percussionist profession?"
    ) == ["people", "profession"]


def test_plan_normalization_preserves_structured_fields() -> None:
    controller = _make_controller()
    plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Atherurus africanus",
                    "chosen_alias": "Atherurus africanus",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [
                {
                    "surface": "Atherurus africanus",
                    "chosen_alias": "Atherurus africanus",
                    "reason": "preserve the benchmark alias",
                }
            ],
            "shared_answer_variable": "?disease",
            "candidate_set_variable": "?disease",
            "count_set_variable": "?disease",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "?organism",
                        "notes": "bind the organism anchor first",
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "biology.organism.diseases_transmitted",
                    "direction": "forward",
                    "from": "?organism",
                    "to": "?disease",
                    "reason": "count diseases transmitted by the organism",
                }
            ],
            "projection": ["count_diseases"],
            "allow_exploratory_predicates": False,
            "strategy": "count diseases transmitted by the anchor organism",
            "plan_rationale": ["use the grounded disease-transmission relation"],
        }
    )

    assert plan["query_shape"] == "count_over_direct_relation"
    assert plan["candidate_set_variable"] == "?disease"
    assert plan["count_set_variable"] == "?disease"
    assert plan["join_structure"]["type"] == "count"
    assert plan["relation_paths"][0]["from_role"] == "anchor"
    assert plan["relation_paths"][0]["to_role"] == "count_set"
    assert plan["relation_paths"][0]["grounding_source"] == "curated"


def test_count_relation_normalization_preserves_inverse_answer_side_semantics() -> None:
    controller = _make_controller()
    normalized = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "medicine.disease.transmitted_by",
                "direction": "forward",
                "from": "disease",
                "to": "transmitter",
                "support": "curated_medicine_disease_predicate",
                "use_when": "find organisms or vectors that transmit a disease",
            }
        ],
        query_shape="count_over_direct_relation",
        answer_mode="count",
        answer_target_phrase="infectious diseases",
        entities=["Aedes aegypti"],
    )

    assert len(normalized) == 1
    assert normalized[0]["from_role"] == "count_set"
    assert normalized[0]["to_role"] == "anchor"


def test_count_relation_normalization_preserves_reverse_spirit_type_count_semantics() -> None:
    controller = _make_controller()
    normalized = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "distilled_spirits.distilled_spirit.spirit_type",
                "direction": "reverse",
                "from": "spirit",
                "to": "bourbon whisky",
                "support": "dynamic_probe_incoming",
                "use_when": "find spirits whose spirit_type points to the anchor",
            }
        ],
        query_shape="count_over_direct_relation",
        answer_mode="count",
        answer_target_phrase="distilled spirit",
        entities=["bourbon whisky"],
    )

    assert len(normalized) == 1
    assert normalized[0]["relation"] == "distilled_spirits.distilled_spirit.spirit_type"
    assert normalized[0]["from_role"] == "count_set"
    assert normalized[0]["to_role"] == "anchor"


def test_count_plan_normalization_coerces_lookup_shape_to_count_shape() -> None:
    controller = _make_controller()
    plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "Goro",
                    "chosen_alias": "Goro",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "rank",
            "candidate_set_variable": "candidate_chars",
            "count_set_variable": "candidate_chars",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {"type": "count", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "fictional_universe.character_rank.characters_of_this_rank",
                    "direction": "reverse",
                    "from": "rank",
                    "to": "Goro",
                    "from_role": "shared_answer",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "count characters with the same rank",
            "plan_rationale": ["count over a rank-linked candidate set"],
        }
    )

    assert plan["query_shape"] == "count_over_direct_relation"


def test_count_plan_normalization_coerces_multi_anchor_intersection_to_joined_count() -> None:
    controller = _make_controller()
    plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "rye", "chosen_alias": "rye", "role": "anchor_a"},
                {
                    "surface": "Corn whiskey",
                    "chosen_alias": "Corn whiskey",
                    "role": "anchor_b",
                },
                {
                    "surface": "canadian whiskey",
                    "chosen_alias": "canadian whiskey",
                    "role": "type_set",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {"type": "intersection", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from": "shared_answer",
                    "to": "type_node",
                    "from_role": "shared_answer",
                    "to_role": "type_set",
                    "grounding_source": "exploratory",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": True,
            "strategy": "count a multi-anchor intersection",
            "plan_rationale": ["normalize multi-anchor count plans to joined-set counts"],
        }
    )

    assert plan["query_shape"] == "count_over_joined_set"


def test_plan_normalization_maps_type_role_to_type_set() -> None:
    controller = _make_controller()
    plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Goro",
                    "chosen_alias": "Goro",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "rank",
            "candidate_set_variable": "candidate_character",
            "count_set_variable": "candidate_character",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {"type": "count", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from": "book_type",
                    "to": "candidate_character",
                    "from_role": "type",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "count book-typed candidates",
            "plan_rationale": ["treat the category phrase as a type constraint"],
        }
    )

    assert plan["relation_paths"][0]["from_role"] == "type_set"


def test_extract_answer_target_phrase_handles_embedded_how_many_clause() -> None:
    controller = _make_controller()

    answer_target = controller._extract_answer_target_phrase(
        "Question: the rank of goro has been given to how many book characters?"
    )

    assert answer_target == "book characters"


def test_extract_answer_target_phrase_handles_embedded_what_clause_after_preposition() -> None:
    controller = _make_controller()

    answer_target = controller._extract_answer_target_phrase(
        "Question: in the google play store what video game platform is supported?"
    )

    assert answer_target == "video game platform"


def test_extract_answer_target_phrase_handles_total_number_contraction() -> None:
    controller = _make_controller()

    answer_target = controller._extract_answer_target_phrase(
        "Question: what's the total number of basketball teams that warren played for?"
    )

    assert answer_target == "basketball teams"


def test_build_question_interpretation_prefers_projected_intersection_for_embedded_kind_attribute_question() -> None:
    controller = _make_controller()
    question_text = "maltese dog and papillon have what kind of temperament?"
    answer_target = controller._extract_answer_target_phrase(question_text)

    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["maltese dog", "Papillon"],
        answer_target_phrase=answer_target,
    )

    assert answer_target == "temperament"
    assert interpretation["preferred_scaffolds"][0]["name"] == "projected_answer_intersection"


def test_infer_query_shape_uses_single_anchor_chain_for_leader_of_question() -> None:
    controller = _make_controller()
    question_text = "john jordan is one of the leaders of which wine producer?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["John Jordan"],
        answer_target_phrase=answer_target,
    )

    query_shape = controller._infer_query_shape(
        question_text=question_text,
        entities=["John Jordan"],
        answer_target_phrase=answer_target,
        question_inputs=interpretation["question_inputs"],
    )

    assert query_shape == "single_anchor_chain_lookup"


def test_infer_query_shape_broader_chain_inference_for_does_have_question(monkeypatch) -> None:
    monkeypatch.setenv("PAL_RUNTIME_BROADER_SINGLE_ANCHOR_CHAIN_INFERENCE", "1")
    controller = _make_controller()
    question_text = "what animals does paul reddam have?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["paul reddam"],
        answer_target_phrase=answer_target,
    )

    query_shape = controller._infer_query_shape(
        question_text=question_text,
        entities=["paul reddam"],
        answer_target_phrase=answer_target,
        question_inputs=interpretation["question_inputs"],
    )

    assert query_shape == "single_anchor_chain_lookup"


def test_infer_query_shape_broader_chain_inference_for_similar_relation_question(monkeypatch) -> None:
    monkeypatch.setenv("PAL_RUNTIME_BROADER_SINGLE_ANCHOR_CHAIN_INFERENCE", "1")
    controller = _make_controller()
    question_text = "which architect has a similar architectural style to josef fanta?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["josef fanta"],
        answer_target_phrase=answer_target,
    )

    query_shape = controller._infer_query_shape(
        question_text=question_text,
        entities=["josef fanta"],
        answer_target_phrase=answer_target,
        question_inputs=interpretation["question_inputs"],
    )

    assert query_shape == "single_anchor_chain_lookup"


def test_infer_query_shape_broader_chain_inference_preserves_direct_guard(monkeypatch) -> None:
    monkeypatch.setenv("PAL_RUNTIME_BROADER_SINGLE_ANCHOR_CHAIN_INFERENCE", "1")
    controller = _make_controller()
    question_text = "which institution has national wine centre of australia?"
    answer_target = controller._extract_answer_target_phrase(question_text)
    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["National Wine Centre of Australia"],
        answer_target_phrase=answer_target,
    )

    query_shape = controller._infer_query_shape(
        question_text=question_text,
        entities=["National Wine Centre of Australia"],
        answer_target_phrase=answer_target,
        question_inputs=interpretation["question_inputs"],
    )

    assert query_shape == "single_anchor_lookup"


def test_grounding_prunes_redundant_pet_breed_temperament_when_biology_variant_exists() -> None:
    controller = _make_controller()
    question_text = "maltese dog and papillon have what kind of temperament?"
    answer_target = controller._extract_answer_target_phrase(question_text)

    grounded = controller._build_grounded_relation_candidates_with_dynamic_fallback(
        task_question=(
            "Question: maltese dog and papillon have what kind of temperament?, "
            "Entities: ['maltese dog', 'Papillon']"
        ),
        entities=["maltese dog", "Papillon"],
        answer_target_phrase=answer_target,
        domain_hints=[],
        question_interpretation=None,
    )

    relations = [candidate["relation"] for candidate in grounded]

    assert "biology.animal_breed.temperament" in relations
    assert "pets.pet_breed.temperament" not in relations


def test_grounding_includes_royal_line_preceded_by_for_dynasty_question() -> None:
    controller = _make_controller()
    question_text = (
        "what is the royal line preceded by the house of lancaster and succeeded by tudor dynasty?"
    )
    answer_target = controller._extract_answer_target_phrase(question_text)

    grounded = controller._build_grounded_relation_candidates_with_dynamic_fallback(
        task_question=(
            "Question: what is the royal line preceded by the house of lancaster and "
            "succeeded by tudor dynasty?, Entities: ['House of Lancaster', 'Tudor dynasty']"
        ),
        entities=["House of Lancaster", "Tudor dynasty"],
        answer_target_phrase=answer_target,
        domain_hints=[],
        question_interpretation=None,
    )

    assert any(
        candidate["relation"] == "royalty.royal_line.preceded_by"
        for candidate in grounded
    )


def test_question_interpretation_does_not_treat_house_of_entity_prefix_as_shared_attribute() -> None:
    controller = _make_controller()
    question_text = (
        "what is the royal line preceded by the house of lancaster and succeeded by tudor dynasty?"
    )
    answer_target = controller._extract_answer_target_phrase(question_text)

    interpretation = controller._build_question_interpretation(
        question_text=question_text,
        explicit_entities=["House of Lancaster", "Tudor dynasty"],
        answer_target_phrase=answer_target,
    )

    assert not any(
        item.get("kind") == "shared_attribute"
        and str(item.get("surface") or "").strip().lower() == "house"
        for item in interpretation["question_inputs"]
    )


def test_build_entity_alias_candidates_adds_uppercase_acronym() -> None:
    controller = _make_controller()

    candidates = controller._build_entity_alias_candidates("wma")

    assert "WMA" in candidates


def test_build_entity_alias_candidates_adds_dehyphenated_variant() -> None:
    controller = _make_controller()

    candidates = controller._build_entity_alias_candidates("czecho-slovakia")

    assert "czechoslovakia" in [candidate.lower() for candidate in candidates]


def test_probe_anchor_match_block_checks_name_and_alias() -> None:
    controller = _make_controller()

    block = controller._build_probe_anchor_match_block(
        anchor_var="?anchor",
        label_var="?anchor_label",
        anchor_name="WMA",
    )

    assert "fb:type.object.name" in block
    assert "fb:common.topic.alias" in block
    assert 'FILTER(LCASE(STR(?anchor_label)) = "wma")' in block


def test_grounding_card_surfaces_answer_target_as_type_constraint() -> None:
    controller = _make_controller()

    grounding_card = controller._build_pal_grounding_card(
        "Question: the rank of goro has been given to how many book characters?, Entities: ['Goro']",
        relation_grounding=[
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "type",
                "to": "candidate_character",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
                "use_when": "restrict candidates to a grounded type node",
            }
        ],
    )

    assert "raw='book characters'" in grounding_card
    assert "clue=type_constraint" in grounding_card
    assert "recommended_alias='book character'" in grounding_card


def test_grounding_card_prefers_grounded_mid_for_type_constraint_answer_target(
    monkeypatch,
) -> None:
    controller = _make_controller()

    monkeypatch.setattr(
        controller,
        "_probe_entity_name_count",
        lambda anchor_name, timeout_s=2.5: 1 if anchor_name == "songwriter" else 0,
    )
    monkeypatch.setattr(
        controller,
        "_probe_anchor_entity_ids",
        lambda anchor_name, relation=None, anchor_position="subject", timeout_s=2.5: (
            ["m.0fj9f"] if anchor_name == "songwriter" else []
        ),
    )

    grounding_card = controller._build_pal_grounding_card(
        "Question: how many songwriters work in the percussionist profession?, Entities: ['Percussionist']",
        question_interpretation={
            "question_inputs": [
                {
                    "surface": "Percussionist",
                    "kind": "named_entity",
                    "role_hint": "anchor",
                },
                {
                    "surface": "songwriters",
                    "kind": "answer_target",
                    "role_hint": "answer_target",
                },
            ],
            "preferred_scaffolds": [{"name": "direct_count", "priority": 1}],
        },
        relation_grounding=[
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "profession",
                "to": "person",
                "from_role": "constraint_value",
                "to_role": "count_set",
                "grounding_source": "curated",
                "support": "curated_people_predicate",
                "use_when": "find people who have the given profession",
            },
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "profession",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
                "support": "curated_people_predicate",
                "use_when": "find the profession(s) of a person",
            },
        ],
    )

    assert "surface='songwriters'" in grounding_card
    assert "recommended_alias='m.0fj9f'" in grounding_card


def test_plan_normalization_unifies_anchor_join_and_relation_roles() -> None:
    controller = _make_controller()
    plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "entity",
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Goat", "chosen_alias": "Goat", "role": "milk_source"},
                {"surface": "cows", "chosen_alias": "cows", "role": "milk_source"},
                {"surface": "semi-firm", "chosen_alias": "semi-firm", "role": "texture_value"},
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "cheese",
            "candidate_set_variable": "candidate_cheese_set",
            "count_set_variable": "",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "shared_answer",
                        "notes": "Goat constrains the answer set",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "shared_answer",
                        "notes": "cows constrains the answer set",
                    },
                    {
                        "anchor_role": "texture_constraint",
                        "constrains_variable": "shared_answer",
                        "notes": "semi-firm constrains the answer set",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "food.cheese_milk_source.cheeses",
                    "direction": "reverse",
                    "from": "milk_source",
                    "to": "cheese",
                    "from_role": "milk_source",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                    "reason": "retrieve cheeses for the milk source",
                },
                {
                    "relation": "food.cheese_texture.cheeses",
                    "direction": "reverse",
                    "from": "texture_value",
                    "to": "cheese",
                    "from_role": "texture_value",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                    "reason": "retrieve cheeses for the texture value",
                },
            ],
            "projection": ["cheese", "cheese_name"],
            "allow_exploratory_predicates": False,
            "strategy": "intersect milk-source and texture constraints on cheese",
            "plan_rationale": ["normalize all constraints to the same semantic answer set"],
        }
    )

    assert [item["role"] for item in plan["anchored_entities"]] == [
        "anchor_a",
        "anchor_b",
        "constraint_value",
    ]
    assert [item["anchor_role"] for item in plan["join_structure"]["anchor_constraints"]] == [
        "anchor_a",
        "anchor_b",
        "constraint_value",
    ]
    assert [item["constrains_variable"] for item in plan["join_structure"]["anchor_constraints"]] == [
        "cheese",
        "cheese",
        "cheese",
    ]
    assert [item["from_role"] for item in plan["relation_paths"]] == [
        "constraint_value",
        "constraint_value",
    ]


def test_grounding_validation_accepts_role_normalized_match() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "biology.organism.diseases_transmitted",
                "direction": "forward",
                "from": "organism",
                "to": "disease",
                "support": "curated_biology_predicate",
                "use_when": "find diseases transmitted by an organism",
            }
        ],
        query_shape="count_over_direct_relation",
        answer_mode="count",
        answer_target_phrase="infectious diseases",
        entities=["Atherurus africanus"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Atherurus africanus",
                    "chosen_alias": "Atherurus africanus",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "?disease",
            "candidate_set_variable": "?disease",
            "count_set_variable": "?disease",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "?organism",
                        "notes": "bind the anchor organism",
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "biology.organism.diseases_transmitted",
                    "direction": "forward",
                    "from": "?organism",
                    "to": "?disease",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                    "reason": "follow the curated disease-transmission edge",
                }
            ],
            "projection": ["count_diseases"],
            "allow_exploratory_predicates": False,
            "strategy": "count_over_direct_relation",
            "plan_rationale": ["use the grounded disease-transmission path"],
        }
    )

    assert controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    ) == []


def test_grounding_validation_accepts_anchor_specific_surface_against_curated_constraint_value() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "medicine.drug.active_moieties",
                "direction": "forward",
                "from": "drug",
                "to": "active_moiety",
                "grounding_source": "curated",
                "support": "observed_live_medicine_predicate",
            }
        ],
        query_shape="multi_anchor_intersection",
        answer_mode="entity",
        answer_target_phrase="dosage form",
        entities=["Naloxone", "Enalaprilat"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "entity",
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                {
                    "surface": "Enalaprilat",
                    "chosen_alias": "Enalaprilat",
                    "role": "anchor_b",
                },
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "dosage_form",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "dosage_form", "notes": "a"},
                    {"anchor_role": "anchor_b", "constrains_variable": "dosage_form", "notes": "b"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "medicine.drug.active_moieties",
                    "direction": "forward",
                    "from": "drug",
                    "to": "Enalaprilat",
                    "from_role": "shared_answer",
                    "to_role": "anchor_b",
                    "grounding_source": "curated",
                    "reason": "use the anchored surface as the specific active-moiety constraint",
                }
            ],
            "projection": ["dosage_form"],
            "allow_exploratory_predicates": False,
            "strategy": "anchor-specific surface should align with curated constraint_value",
            "plan_rationale": ["the surface anchor is a specific constraint value for a curated relation"],
        }
    )

    assert controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    ) == []


def test_grounding_validation_accepts_anchor_instantiated_curated_music_branch() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "music.artist.track",
                "direction": "forward",
                "from": "artist",
                "to": "track",
                "grounding_source": "curated",
                "support": "curated_music_predicate",
            }
        ],
        query_shape="multi_anchor_intersection",
        answer_mode="entity",
        answer_target_phrase="musical release",
        entities=["Count Basie Orchestra", "Joe Williams"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "entity",
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {
                    "surface": "Count Basie Orchestra",
                    "chosen_alias": "Count Basie Orchestra",
                    "role": "anchor_a",
                },
                {
                    "surface": "Joe Williams",
                    "chosen_alias": "Joe Williams",
                    "role": "anchor_b",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "shared_answer", "notes": "a"},
                    {"anchor_role": "anchor_b", "constrains_variable": "shared_answer", "notes": "b"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "music.artist.track",
                    "direction": "forward",
                    "from": "count_basie_orchestra",
                    "to": "track",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.artist.track",
                    "direction": "forward",
                    "from": "joe_williams",
                    "to": "track",
                    "from_role": "anchor_b",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["shared_answer"],
            "allow_exploratory_predicates": False,
            "strategy": "intersect tracks from two anchors",
            "plan_rationale": ["allow anchor-instantiated use of a generic curated branch"],
        }
    )

    assert controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    ) == []


def test_grounding_validation_accepts_anchor_instantiated_curated_constraint_branch() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "breed",
                "to": "temperament",
                "grounding_source": "curated",
                "support": "curated_biology_predicate",
            }
        ],
        query_shape="count_over_joined_set",
        answer_mode="count",
        answer_target_phrase="dog breeds",
        entities=["Canada", "Bull Terrier"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {"surface": "Canada", "chosen_alias": "Canada", "role": "anchor_a"},
                {
                    "surface": "Bull Terrier",
                    "chosen_alias": "Bull Terrier",
                    "role": "anchor_b",
                },
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "candidate_set", "notes": "country filter"},
                    {"anchor_role": "anchor_b", "constrains_variable": "candidate_set", "notes": "temperament filter"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "bull_terrier",
                    "to": "bull_temperament",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "reuse a generic curated temperament branch for a specific anchor",
            "plan_rationale": ["allow anchor-instantiated constraint extraction from generic curated relations"],
        }
    )

    assert controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    ) == []


def test_grounding_validation_does_not_bypass_curated_mismatch_in_exploratory_mode() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "organization.organization.parent_organization",
                "direction": "forward",
                "from": "organization",
                "to": "parent",
                "support": "curated_organization_predicate",
                "use_when": "find the parent organization of an institution",
            }
        ],
        query_shape="single_anchor_lookup",
        answer_mode="entity",
        answer_target_phrase="institution",
        entities=["National Wine Centre of Australia"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "entity",
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "National Wine Centre of Australia",
                    "chosen_alias": "National Wine Centre of Australia",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "?parent",
            "candidate_set_variable": "?parent",
            "count_set_variable": "",
            "join_structure": {
                "type": "single_path",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "?organization",
                        "notes": "bind the organization anchor",
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "organization.organization.parent_organization",
                    "direction": "forward",
                    "from": "?parent",
                    "to": "?organization",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "curated",
                    "reason": "this intentionally flips the grounded organization path",
                },
                {
                    "relation": "organization.organization.type",
                    "direction": "forward",
                    "from": "?parent",
                    "to": "?type",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "exploratory",
                    "reason": "optional exploratory refinement",
                },
            ],
            "projection": ["?parent", "?parent_name"],
            "allow_exploratory_predicates": True,
            "strategy": "single_anchor_lookup with exploratory refinement",
            "plan_rationale": ["the second path is exploratory but the curated path must still align"],
        }
    )

    errors = controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    )
    assert any(error.startswith("relation_shape_not_grounded:organization.organization.parent_organization") for error in errors)


def test_grounding_validation_relaxes_shape_for_dynamic_probe_only_relations() -> None:
    controller = _make_controller()
    relation_grounding = [
        {
            "relation": "type.type.instance",
            "direction": "reverse",
            "from": "?source",
            "to": "Goro",
            "from_role": "candidate_set",
            "to_role": "anchor",
            "grounding_source": "dynamic_probe",
            "support": "dynamic_probe_incoming",
        }
    ]
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Goro", "chosen_alias": "Goro", "role": "anchor"}
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "candidate_set", "notes": "dynamic probe path"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "type",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                    "reason": "reuse the live-observed dynamic predicate family",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "dynamic probe relation family reuse",
            "plan_rationale": ["dynamic grounding should validate at the relation-family level"],
        }
    )

    assert controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    ) == []


def test_repair_loop_rejection_is_fail_closed() -> None:
    controller = _make_controller()
    with pytest.raises(AgentUnknownException, match="pal_query_not_accepted:repairable_bad_count_set"):
        controller._ensure_repair_loop_accepted(
            generated_tool_name="pal_sparql_query_tool_test",
            repair_loop_log={
                "total_attempts": 3,
                "repair_used": True,
                "final_verdict": "no_accepted_candidate",
                "last_verdict": "repairable_bad_count_set",
                "last_reasons": ["count_set_path_empty"],
            },
        )


def test_repair_loop_stagnation_cuts_off_repeated_state() -> None:
    controller = _make_controller()
    controller._generate_validated_pal_candidate = lambda **kwargs: ("def solve(endpoint_url):\n    return {}\n", {})
    controller._extract_sparql_query_texts = lambda code: ["SELECT (COUNT(DISTINCT ?disease) AS ?count) WHERE { ?disease ?p ?o }"]
    controller._run_anchor_existence_probes = lambda **kwargs: [
        AnchorProbeResult(
            anchor_name="Aedes aegypti",
            entity_count=12,
            path_count=1,
            relation_probed="medicine.infectious_disease.vector",
            anchor_position="object",
            resolved_entity_id="m.06y6_w",
        )
    ]
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results:
        (generated_code, invocation_result, query_plan, False)
    )
    controller._apply_probe_guided_alias_repairs = (
        lambda **kwargs: (kwargs["query_plan"], kwargs["grounding_card"], [])
    )
    controller._apply_probe_guided_anchor_entity_repairs = (
        lambda **kwargs: (kwargs["query_plan"], [])
    )
    controller._augment_grounding_with_dynamic_probe_on_path_failure = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._augment_grounding_for_structural_repair = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._suppress_dead_grounded_relations = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._merge_feedback_items = lambda current, new: [*current, *new]
    controller._should_retry_same_plan_after_alias_repair = lambda **kwargs: False
    controller._build_projected_answer_intersection_repair_plan = lambda **kwargs: None
    controller._build_single_anchor_dynamic_lookup_repair_plan = lambda **kwargs: None
    controller._build_pivot_preserving_count_repair_plan = lambda **kwargs: None
    controller._generate_pal_query_plan = lambda **kwargs: kwargs.get("relation_grounding") and {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "stagnation-test",
        "anchored_entities": [
            {"surface": "Aedes aegypti", "chosen_alias": "Aedes aegypti", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from_role": "count_set",
                "to_role": "anchor",
                "from": "disease",
                "to": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "disease", "notes": "test"}
            ],
        },
        "shared_answer_variable": "disease",
        "candidate_set_variable": "disease",
        "count_set_variable": "disease",
        "ordering_attribute": {},
        "ordering_direction": "none",
    } or {}
    controller._extract_anchor_alias_assignments = lambda plan: {}
    controller._refresh_dynamic_grounding_after_anchor_alias_change = (
        lambda **kwargs: (kwargs["task_question"], kwargs["relation_grounding"], [])
    )

    invocation_result = PALInvocationResult(
        success=True,
        payload={"head": {"vars": ["count"]}, "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]}},
    )
    repeated_verdict = PlausibilityVerdict(
        verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
        reasons=["count_scalar_returned:1", "count_query_dynamic_chain_too_weak"],
    )

    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "stagnation-test",
        "anchored_entities": [
            {"surface": "Aedes aegypti", "chosen_alias": "Aedes aegypti", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from_role": "count_set",
                "to_role": "anchor",
                "from": "disease",
                "to": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "disease", "notes": "test"}
            ],
        },
        "shared_answer_variable": "disease",
        "candidate_set_variable": "disease",
        "count_set_variable": "disease",
        "ordering_attribute": {},
        "ordering_direction": "none",
    }

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=invocation_result,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        return_value=repeated_verdict,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question="Question: what is the number of infectious diseases that are transmitted by the aedes aegypti?, Entities: ['Aedes aegypti']",
            grounding_card="PAL grounding hints:",
            query_plan=query_plan,
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["Aedes aegypti"],
            relation_grounding=[
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "reverse",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "from": "disease",
                    "to": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
        )

    assert loop_log["total_attempts"] == 2
    assert loop_log["last_verdict"] == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "repair_loop_stalled" in loop_log["last_reasons"]


def test_repair_loop_same_attempt_anchor_retry_does_not_store_tuple_best_candidate() -> None:
    controller = _make_controller()
    controller._generate_validated_pal_candidate = lambda **kwargs: ("def solve(endpoint_url):\n    return {}\n", {})
    controller._extract_sparql_query_texts = lambda code: ["SELECT (COUNT(DISTINCT ?disease) AS ?count) WHERE { ?disease ?p ?o }"]
    controller._run_anchor_existence_probes = lambda **kwargs: [
        AnchorProbeResult(
            anchor_name="Aedes aegypti",
            entity_count=12,
            path_count=1,
            relation_probed="medicine.infectious_disease.vector",
            anchor_position="object",
            resolved_entity_id="m.06y6_w",
        )
    ]
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results:
        (generated_code, invocation_result, query_plan, True)
    )
    controller._apply_probe_guided_alias_repairs = (
        lambda **kwargs: (kwargs["query_plan"], kwargs["grounding_card"], [])
    )
    controller._apply_probe_guided_anchor_entity_repairs = (
        lambda **kwargs: (kwargs["query_plan"], [])
    )
    controller._augment_grounding_with_dynamic_probe_on_path_failure = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._augment_grounding_for_structural_repair = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._suppress_dead_grounded_relations = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._merge_feedback_items = lambda current, new: [*current, *new]
    controller._should_retry_same_plan_after_alias_repair = lambda **kwargs: False
    controller._build_projected_answer_intersection_repair_plan = lambda **kwargs: None
    controller._build_single_anchor_dynamic_lookup_repair_plan = lambda **kwargs: None
    controller._build_pivot_preserving_count_repair_plan = lambda **kwargs: None
    controller._generate_pal_query_plan = lambda **kwargs: kwargs.get("relation_grounding") and {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "same-attempt-anchor-retry-test",
        "anchored_entities": [
            {"surface": "Aedes aegypti", "chosen_alias": "Aedes aegypti", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from_role": "count_set",
                "to_role": "anchor",
                "from": "disease",
                "to": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "disease", "notes": "test"}
            ],
        },
        "shared_answer_variable": "disease",
        "candidate_set_variable": "disease",
        "count_set_variable": "disease",
        "ordering_attribute": {},
        "ordering_direction": "none",
    } or {}
    controller._extract_anchor_alias_assignments = lambda plan: {}
    controller._refresh_dynamic_grounding_after_anchor_alias_change = (
        lambda **kwargs: (kwargs["task_question"], kwargs["relation_grounding"], [])
    )

    invocation_result = PALInvocationResult(
        success=True,
        payload={"head": {"vars": ["count"]}, "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]}},
    )
    repeated_verdict = PlausibilityVerdict(
        verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
        reasons=["count_scalar_returned:1", "count_query_dynamic_chain_too_weak"],
    )

    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "same-attempt-anchor-retry-test",
        "anchored_entities": [
            {"surface": "Aedes aegypti", "chosen_alias": "Aedes aegypti", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from_role": "count_set",
                "to_role": "anchor",
                "from": "disease",
                "to": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "disease", "notes": "test"}
            ],
        },
        "shared_answer_variable": "disease",
        "candidate_set_variable": "disease",
        "count_set_variable": "disease",
        "ordering_attribute": {},
        "ordering_direction": "none",
    }

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=invocation_result,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        return_value=repeated_verdict,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question="Question: what is the number of infectious diseases that are transmitted by the aedes aegypti?, Entities: ['Aedes aegypti']",
            grounding_card="PAL grounding hints:",
            query_plan=query_plan,
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["Aedes aegypti"],
            relation_grounding=[
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "reverse",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "from": "disease",
                    "to": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
        )

    assert loop_log["best_executing_candidate"]["score"] >= 0
    assert loop_log["final_verdict"] in {
        "accepted_best_effort",
        "no_accepted_candidate",
        VERDICT_REPAIRABLE_BAD_COUNT_SET,
    }


def test_repair_loop_probes_nonempty_dynamic_single_anchor_lookup_for_disambiguation() -> None:
    controller = _make_controller()
    probe_calls: list[dict[str, Any]] = []
    validation_probe_payloads: list[list[AnchorProbeResult] | None] = []

    controller._generate_validated_pal_candidate = (
        lambda **kwargs: ("def solve(endpoint_url):\n    return {}\n", {})
    )
    controller._extract_sparql_query_texts = (
        lambda code: [
            "SELECT DISTINCT ?candidate_set WHERE { "
            "?anchor fb:business.board_member.leader_of ?candidate_set . "
            "}"
        ]
    )

    def _probe(**kwargs):
        probe_calls.append(dict(kwargs))
        return [
            AnchorProbeResult(
                anchor_name="John Jordan",
                entity_count=3,
                path_count=1,
                relation_probed="business.board_member.leader_of",
                anchor_position="subject",
                resolved_entity_id="m.good_jordan",
            )
        ]

    controller._run_anchor_existence_probes = _probe
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results:
        (generated_code, invocation_result, query_plan, False)
    )

    invocation_result = PALInvocationResult(
        success=True,
        payload={
            "head": {"vars": ["candidate_set"]},
            "results": {
                "bindings": [
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.some_winery",
                        }
                    }
                ]
            },
        },
    )

    def _validate(**kwargs):
        validation_probe_payloads.append(kwargs.get("anchor_probe_results"))
        return PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=["accepted"])

    query_plan = {
        "answer_mode": "entity",
        "answer_type": "entity",
        "query_shape": "single_anchor_lookup",
        "strategy": "dynamic single anchor lookup",
        "anchored_entities": [
            {"surface": "John Jordan", "chosen_alias": "John Jordan", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "business.board_member.leader_of",
                "direction": "forward",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "from": "anchor",
                "to": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "lookup",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "candidate_set", "notes": "test"}
            ],
        },
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "",
        "ordering_attribute": {},
        "ordering_direction": "none",
    }

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=invocation_result,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        side_effect=_validate,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question="Question: john jordan is one of the leaders of which wine producer?, Entities: ['John Jordan']",
            grounding_card="PAL grounding hints:",
            query_plan=query_plan,
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["John Jordan"],
            relation_grounding=[
                {
                    "relation": "business.board_member.leader_of",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "from": "anchor",
                    "to": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
        )

    assert probe_calls
    assert probe_calls[0]["probe_paths"] is True
    assert validation_probe_payloads
    assert validation_probe_payloads[0]
    assert validation_probe_payloads[0][0].resolved_entity_id == "m.good_jordan"
    assert loop_log["final_verdict"] == VERDICT_ACCEPTED


def test_repair_loop_plan_refresh_reapplies_resolved_anchor_ids() -> None:
    controller = _make_controller()
    call_state = {"count": 0}

    def _generate_validated_pal_candidate(**kwargs):
        call_state["count"] += 1
        if call_state["count"] == 2:
            assert (
                kwargs["query_plan"]["anchored_entities"][0]["resolved_entity_id"]
                == "m.01c44b"
            )
        return ("def solve(endpoint_url):\n    return {}\n", {})

    controller._generate_validated_pal_candidate = _generate_validated_pal_candidate
    controller._extract_sparql_query_texts = (
        lambda code: ["SELECT (COUNT(DISTINCT ?thing) AS ?count) WHERE { ?thing ?p ?o }"]
    )
    controller._run_anchor_existence_probes = lambda **kwargs: [
        AnchorProbeResult(
            anchor_name="Anchor",
            entity_count=1,
            path_count=1,
            relation_probed="test.relation",
            anchor_position="subject",
            resolved_entity_id="m.01c44b",
        )
    ]
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results:
        (generated_code, invocation_result, query_plan, False)
    )
    controller._apply_probe_guided_alias_repairs = (
        lambda **kwargs: (kwargs["query_plan"], kwargs["grounding_card"], [])
    )
    controller._augment_grounding_with_dynamic_probe_on_path_failure = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._augment_grounding_for_structural_repair = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._suppress_dead_grounded_relations = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._merge_feedback_items = lambda current, new: [*current, *new]
    controller._should_retry_same_plan_after_alias_repair = lambda **kwargs: False
    controller._build_projected_answer_intersection_repair_plan = lambda **kwargs: None
    controller._build_shared_type_pivot_bridge_repair_plan = lambda **kwargs: None
    controller._build_single_anchor_dynamic_lookup_repair_plan = lambda **kwargs: None
    controller._build_superlative_dynamic_anchor_repair_plan = lambda **kwargs: None
    controller._build_class_filtered_count_repair_plan = lambda **kwargs: None
    controller._build_pivot_preserving_count_repair_plan = lambda **kwargs: None
    controller._extract_anchor_alias_assignments = lambda plan: {}
    controller._refresh_dynamic_grounding_after_anchor_alias_change = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._generate_pal_query_plan = lambda **kwargs: {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "refreshed-plan",
        "anchored_entities": [
            {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "test.relation",
                "direction": "forward",
                "from_role": "anchor",
                "to_role": "count_set",
                "from": "Anchor",
                "to": "thing",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "thing", "notes": "test"}
            ],
        },
        "candidate_set_variable": "thing",
        "count_set_variable": "thing",
        "ordering_attribute": {},
        "ordering_direction": "none",
    }

    invocation_results = [
        PALInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["count"]},
                "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
            },
        ),
        PALInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["count"]},
                "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
            },
        ),
    ]
    verdicts = [
        PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=["count_scalar_returned:0", "count_set_path_empty"],
        ),
        PlausibilityVerdict(
            verdict=VERDICT_ACCEPTED,
            reasons=["count_scalar_returned:1"],
        ),
    ]

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        side_effect=invocation_results,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        side_effect=verdicts,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question="Question: how many things are related to anchor?, Entities: ['Anchor']",
            grounding_card="PAL grounding hints:",
            query_plan={
                "answer_mode": "count",
                "query_shape": "count_over_direct_relation",
                "strategy": "initial-plan",
                "anchored_entities": [
                    {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"}
                ],
                "relation_paths": [
                    {
                        "relation": "test.relation",
                        "direction": "forward",
                        "from_role": "anchor",
                        "to_role": "count_set",
                        "from": "Anchor",
                        "to": "thing",
                        "grounding_source": "dynamic_probe",
                    }
                ],
                "join_structure": {
                    "type": "count",
                    "anchor_constraints": [
                        {
                            "anchor_role": "anchor",
                            "constrains_variable": "thing",
                            "notes": "test",
                        }
                    ],
                },
                "candidate_set_variable": "thing",
                "count_set_variable": "thing",
                "ordering_attribute": {},
                "ordering_direction": "none",
            },
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["Anchor"],
            relation_grounding=[
                {
                    "relation": "test.relation",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "from": "Anchor",
                    "to": "thing",
                    "grounding_source": "dynamic_probe",
                }
            ],
        )

    assert loop_log["accepted_attempt"] == 2
    assert call_state["count"] == 2


def test_repair_loop_plan_refresh_reapplies_count_structural_rewrites() -> None:
    controller = _make_controller()
    call_state = {"count": 0, "rewrite_calls": 0}

    def _generate_validated_pal_candidate(**kwargs):
        call_state["count"] += 1
        if call_state["count"] == 2:
            assert kwargs["query_plan"]["query_shape"] == "count_over_joined_set"
            assert kwargs["query_plan"]["shared_answer_variable"] == "shared_answer"
        return ("def solve(endpoint_url):\n    return {}\n", {})

    controller._generate_validated_pal_candidate = _generate_validated_pal_candidate
    controller._extract_sparql_query_texts = (
        lambda code: ["SELECT (COUNT(DISTINCT ?thing) AS ?count) WHERE { ?thing ?p ?o }"]
    )
    controller._run_anchor_existence_probes = lambda **kwargs: [
        AnchorProbeResult(
            anchor_name="Anchor",
            entity_count=1,
            path_count=0,
            relation_probed="test.relation",
            anchor_position="subject",
            resolved_entity_id="m.anchor",
        )
    ]
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results:
        (generated_code, invocation_result, query_plan, False)
    )
    controller._apply_probe_guided_alias_repairs = (
        lambda **kwargs: (kwargs["query_plan"], kwargs["grounding_card"], [])
    )
    controller._apply_probe_guided_anchor_entity_repairs = (
        lambda **kwargs: (kwargs["query_plan"], [])
    )
    controller._augment_grounding_with_dynamic_probe_on_path_failure = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._augment_grounding_for_structural_repair = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._suppress_dead_grounded_relations = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._merge_feedback_items = lambda current, new: [*current, *new]
    controller._should_retry_same_plan_after_alias_repair = lambda **kwargs: False
    controller._build_projected_answer_intersection_repair_plan = lambda **kwargs: None
    controller._build_shared_type_pivot_bridge_repair_plan = lambda **kwargs: None
    controller._build_single_anchor_dynamic_lookup_repair_plan = lambda **kwargs: None
    controller._build_superlative_anchor_alternative_repair_plan = lambda **kwargs: None
    controller._build_superlative_dynamic_anchor_repair_plan = lambda **kwargs: None

    def _build_class_filtered_count_repair_plan(**kwargs):
        call_state["rewrite_calls"] += 1
        if call_state["rewrite_calls"] < 2:
            return None
        return {
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "strategy": "refreshed-plan-with-class-filter",
            "anchored_entities": [
                {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"},
                {"surface": "things", "chosen_alias": "thing", "role": "type_set"},
            ],
            "relation_paths": [
                {
                    "relation": "test.relation",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "from": "Anchor",
                    "to": "shared_answer",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "from": "shared_answer",
                    "to": "type_set",
                    "grounding_source": "curated",
                },
            ],
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "shared_answer"},
                    {"anchor_role": "type_set", "constrains_variable": "shared_answer"},
                ],
            },
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "shared_answer",
            "count_set_variable": "shared_answer",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "projection": ["count"],
        }

    controller._build_class_filtered_count_repair_plan = (
        _build_class_filtered_count_repair_plan
    )
    controller._build_pivot_preserving_count_repair_plan = lambda **kwargs: None
    controller._extract_anchor_alias_assignments = lambda plan: {}
    controller._refresh_dynamic_grounding_after_anchor_alias_change = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._generate_pal_query_plan = lambda **kwargs: {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "refreshed-plan",
        "anchored_entities": [
            {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"}
        ],
        "relation_paths": [
            {
                "relation": "test.relation",
                "direction": "forward",
                "from_role": "anchor",
                "to_role": "count_set",
                "from": "Anchor",
                "to": "thing",
                "grounding_source": "dynamic_probe",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor", "constrains_variable": "thing", "notes": "test"}
            ],
        },
        "candidate_set_variable": "thing",
        "count_set_variable": "thing",
        "ordering_attribute": {},
        "ordering_direction": "none",
    }

    invocation_results = [
        PALInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["count"]},
                "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
            },
        ),
        PALInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["count"]},
                "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
            },
        ),
    ]
    verdicts = [
        PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=["count_scalar_returned:0", "count_query_all_relation_paths_exploratory"],
        ),
        PlausibilityVerdict(
            verdict=VERDICT_ACCEPTED,
            reasons=["count_scalar_returned:1"],
        ),
    ]

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        side_effect=invocation_results,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        side_effect=verdicts,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question="Question: how many things are related to anchor?, Entities: ['Anchor']",
            grounding_card="PAL grounding hints:",
            query_plan={
                "answer_mode": "count",
                "query_shape": "count_over_direct_relation",
                "strategy": "initial-plan",
                "anchored_entities": [
                    {"surface": "Anchor", "chosen_alias": "Anchor", "role": "anchor"}
                ],
                "relation_paths": [
                    {
                        "relation": "test.relation",
                        "direction": "forward",
                        "from_role": "anchor",
                        "to_role": "count_set",
                        "from": "Anchor",
                        "to": "thing",
                        "grounding_source": "dynamic_probe",
                    }
                ],
                "join_structure": {
                    "type": "count",
                    "anchor_constraints": [
                        {
                            "anchor_role": "anchor",
                            "constrains_variable": "thing",
                            "notes": "test",
                        }
                    ],
                },
                "candidate_set_variable": "thing",
                "count_set_variable": "thing",
                "ordering_attribute": {},
                "ordering_direction": "none",
            },
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["Anchor"],
            relation_grounding=[
                {
                    "relation": "test.relation",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "from": "Anchor",
                    "to": "thing",
                    "grounding_source": "dynamic_probe",
                }
            ],
        )

    assert loop_log["accepted_attempt"] == 2
    assert call_state["rewrite_calls"] >= 2


def test_repair_loop_salvage_restores_best_executing_query_plan() -> None:
    controller = _make_controller()
    attempt_counter = {"count": 0}

    def _generate_candidate(**kwargs):
        attempt_counter["count"] += 1
        if attempt_counter["count"] == 1:
            return ("def solve(endpoint_url):\n    return {}\n", {})
        raise RuntimeError("repair_codegen_failed")

    class _DecisionRecord:
        def as_dict(self) -> dict[str, object]:
            return {}

    controller._generate_validated_pal_candidate = _generate_candidate
    controller._extract_sparql_query_texts = (
        lambda code: [
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?anchor fb:computer.computer.designed_by ?candidate_set . }"
        ]
    )
    controller._run_anchor_existence_probes = lambda **kwargs: [
        AnchorProbeResult(
            anchor_name="ThinkPad 560Z",
            entity_count=1,
            path_count=1,
            relation_probed="computer.computer.designed_by",
            anchor_position="subject",
            resolved_entity_id="m.test_anchor",
        )
    ]
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results: (
            generated_code,
            invocation_result,
            query_plan,
            False,
        )
    )
    controller._apply_probe_guided_alias_repairs = (
        lambda **kwargs: (
            {
                **kwargs["query_plan"],
                "query_shape": "count_over_joined_set",
                "relation_paths": [
                    {
                        "relation": "computer.computer_designers.designed_computers",
                        "direction": "forward",
                        "from": "designer",
                        "to": "computer",
                        "from_role": "constraint_value",
                        "to_role": "count_set",
                        "grounding_source": "curated",
                    }
                ],
            },
            kwargs["grounding_card"],
            [],
        )
    )
    controller._apply_probe_guided_anchor_entity_repairs = (
        lambda **kwargs: (kwargs["query_plan"], [])
    )
    controller._augment_grounding_with_dynamic_probe_on_path_failure = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._augment_grounding_for_structural_repair = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._suppress_dead_grounded_relations = (
        lambda **kwargs: ("PAL grounding hints:", kwargs["relation_grounding"], [])
    )
    controller._merge_feedback_items = lambda current, new: [*current, *new]
    controller._decide_family_repair_action = lambda **kwargs: "stay_in_family"
    controller._build_attempt_decision_record = lambda **kwargs: _DecisionRecord()
    controller._validate_pal_query_candidate = lambda **kwargs: []
    controller._should_accept_best_executing_candidate = lambda **kwargs: True

    invocation_result = PALInvocationResult(
        success=True,
        payload={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
        },
    )
    verdict = PlausibilityVerdict(
        verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
        reasons=["count_scalar_returned:1"],
    )

    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "strategy": "salvage-plan-sync-test",
        "anchored_entities": [
            {
                "surface": "ThinkPad 560Z",
                "chosen_alias": "ThinkPad 560Z",
                "role": "anchor",
            }
        ],
        "relation_paths": [
            {
                "relation": "computer.computer.designed_by",
                "direction": "forward",
                "from_role": "anchor",
                "to_role": "count_set",
                "from": "anchor",
                "to": "candidate_set",
                "grounding_source": "curated",
            }
        ],
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "test",
                }
            ],
        },
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "ordering_attribute": {},
        "ordering_direction": "none",
    }

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=invocation_result,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        return_value=verdict,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question=(
                "Question: what was the amount of key designers who produced "
                "the producer of the ThinkPad 560Z?, Entities: ['ThinkPad 560Z']"
            ),
            grounding_card="PAL grounding hints:",
            query_plan=query_plan,
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["ThinkPad 560Z"],
            relation_grounding=[
                {
                    "relation": "computer.computer.designed_by",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "from": "anchor",
                    "to": "candidate_set",
                    "grounding_source": "curated",
                }
            ],
        )

    assert loop_log["accepted_attempt"] == "best_executing"
    assert loop_log["final_verdict"] == "accepted_best_effort"
    assert query_plan["query_shape"] == "count_over_direct_relation"
    assert query_plan["relation_paths"][0]["relation"] == "computer.computer.designed_by"


def test_family_policy_candidate_uses_final_rejected_plan_instead_of_best_executing_salvage(
    monkeypatch,
) -> None:
    controller = _make_controller()
    captured: dict[str, object] = {}

    class _FakeStore:
        def create_candidate_update(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Candidate",
                (),
                {
                    "base_version": "2026-03-31",
                    "candidate_version": "2026-03-31__cand0001",
                    "fields_changed": ("blocked_scaffold_signatures",),
                    "reason_for_change": "test_candidate",
                    "trigger_context": kwargs["trigger_context"],
                },
            )()

    controller._get_family_policy_store = lambda: _FakeStore()
    controller._current_session = type(
        "Session", (), {"task_name": "knowledge_graph", "sample_index": "11"}
    )()
    monkeypatch.setattr(
        pal_agent_controller_module,
        "family_policy_enabled_for",
        lambda family_name: True,
    )

    direct_query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "shared_answer_variable": "candidate_person",
        "candidate_set_variable": "candidate_person",
        "count_set_variable": "candidate_person",
        "relation_paths": [
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from_role": "constraint_value",
                "to_role": "count_set",
            },
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
            },
        ],
    }
    final_joined_query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "shared_answer",
        "count_set_variable": "shared_answer",
        "relation_paths": [
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
            },
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from_role": "constraint_value",
                "to_role": "candidate_set",
            },
        ],
    }

    controller._maybe_record_family_policy_candidate(
        generated_tool_name="pal_sparql_query_tool_test",
        query_plan=direct_query_plan,
        failure_reason="pal_query_not_accepted:rejected_dangerous_overreach",
        repair_loop_log={
            "final_verdict": "rejected_dangerous_overreach",
            "attempt_decisions": [
                {
                    "selected_family": "count_over_joined_set",
                    "family_bundle_version": "2026-03-31",
                }
            ],
            "final_attempt_query_plan": final_joined_query_plan,
            "last_reasons": ["dangerous_overreach:weak_count_semantics"],
            "best_executing_candidate": {
                "verdict_reasons": ["count_scalar_returned:12"],
            },
        },
    )

    assert captured["family_name"] == "count_over_joined_set"
    assert captured["scaffold_signature"] == (
        "count_over_joined_set|shared_answer|"
        "people.profession.people_with_this_profession"
    )
    assert captured["relation_names"] == ["people.profession.people_with_this_profession"]
    trigger_context = captured["trigger_context"]
    assert isinstance(trigger_context, dict)
    assert trigger_context["query_shape"] == "count_over_joined_set"


def test_repair_loop_does_not_preemptively_rewrite_valid_multi_anchor_intersection() -> None:
    controller = _make_controller()
    controller._generate_validated_pal_candidate = (
        lambda **kwargs: ("def solve(endpoint_url):\n    return {}\n", {})
    )
    controller._extract_sparql_query_texts = (
        lambda code: ["SELECT DISTINCT ?answer WHERE { ?candidate_set ?p ?answer . }"]
    )
    controller._retry_execution_with_resolved_anchor_ids = (
        lambda generated_code, invocation_result, query_plan, anchor_probe_results:
        (generated_code, invocation_result, query_plan, False)
    )
    controller._has_asymmetric_anchor_clues = lambda **kwargs: False
    controller._build_projected_answer_intersection_repair_plan = (
        lambda **kwargs: pytest.fail(
            "projected-answer repair should not run before the first execution"
        )
    )

    invocation_result = PALInvocationResult(
        success=True,
        payload={
            "head": {"vars": ["answer"]},
            "results": {
                "bindings": [
                    {
                        "answer": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.075g54v",
                        }
                    }
                ]
            },
        },
    )
    verdict = PlausibilityVerdict(verdict=VERDICT_ACCEPTED, reasons=[])

    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "strategy": "intersect game versions then project regions",
            "anchored_entities": [
                {
                    "surface": "Virtual Console",
                    "chosen_alias": "m.07sg3j",
                    "role": "anchor_a",
                },
                {
                    "surface": "sega",
                    "chosen_alias": "m.06p8m",
                    "role": "anchor_b",
                },
            ],
            "relation_paths": [
                {
                    "relation": "cvg.computer_game_distribution_system.games_distributed",
                    "direction": "reverse",
                    "from": "anchor_a",
                    "to": "candidate_set",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "cvg.cvg_developer.game_versions_developed",
                    "direction": "reverse",
                    "from": "anchor_b",
                    "to": "candidate_set",
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "cvg.game_version.regions",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "answer",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "curated",
                },
            ],
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "candidate_set",
                        "notes": "Virtual Console constrains the shared game-version set",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "candidate_set",
                        "notes": "Sega constrains the same shared game-version set",
                    },
                ],
            },
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "",
            "projection": ["answer"],
            "ordering_attribute": {},
            "ordering_direction": "none",
        }
    )

    with patch.object(
        pal_agent_controller_module,
        "execute_pal_code_with_result",
        return_value=invocation_result,
    ), patch.object(
        pal_agent_controller_module,
        "validate_pal_execution",
        return_value=verdict,
    ):
        _code, _result, loop_log = controller._run_pal_repair_loop(
            task_question=(
                "Question: virtual console, which is developed by sega of japan, "
                "was released where?, Entities: ['Virtual Console', 'sega of japan']"
            ),
            grounding_card="PAL grounding hints:",
            query_plan=query_plan,
            generated_tool_name="pal_sparql_query_tool_test",
            question_entities=["Virtual Console", "sega of japan"],
            relation_grounding=[
                {
                    "relation": "cvg.computer_game_distribution_system.games_distributed",
                    "direction": "reverse",
                    "from": "distribution_system",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "cvg.cvg_developer.game_versions_developed",
                    "direction": "reverse",
                    "from": "developer",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "cvg.game_version.regions",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "answer",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "curated",
                },
            ],
        )

    assert loop_log["accepted_attempt"] == 1
    assert loop_log["final_verdict"] == "accepted"
    assert query_plan["shared_answer_variable"] == "candidate_set"
    assert query_plan["relation_paths"][0]["to"] == "candidate_set"


def test_materialize_adapter_response_fails_closed_for_unresolved_artifact() -> None:
    controller = _make_controller()

    with pytest.raises(AgentUnknownException, match="pal_adapter_unresolved_artifact"):
        controller._materialize_adapter_response(
            task_question="Question: ...",
            materialization=BenchmarkMaterialization(
                materialization_type="unresolved_failure",
                needs_bridge=False,
                bridge_action=None,
                bridge_tool_name=None,
                bridge_payload=None,
                final_variable=None,
                final_answer_text=None,
                diagnostics={"artifact_type": "unresolved", "artifact_source": "raw_execution_failure"},
                confidence=0.0,
                determinism_level="none",
            ),
        )


def test_materialize_adapter_response_records_query_tool_and_bridge_macro() -> None:
    controller = _make_controller()
    controller._pending_macro_runs = {}
    controller._ensure_bridge_tool = lambda tool_name: tool_name
    controller._get_run_id = lambda: "knowledge_graph_5"
    controller._get_macro_state_dir = lambda: "outputs/test_state"

    response = controller._materialize_adapter_response(
        task_question="Question: ...",
        generated_tool_name="pal_sparql_query_tool_5_deadbeef",
        materialization=BenchmarkMaterialization(
            materialization_type="bridge_action",
            needs_bridge=True,
            bridge_action='Action: execute_macro("pal_benchmark_bridge_macro", {"run_id": "knowledge_graph_5", "state_dir": "outputs/test_state"})',
            bridge_tool_name="pal_benchmark_bridge_macro",
            bridge_payload={"run_id": "knowledge_graph_5", "state_dir": "outputs/test_state"},
            final_variable=None,
            final_answer_text=None,
            diagnostics={"artifact_type": "count_scalar", "artifact_source": "raw_execution"},
            confidence=1.0,
            determinism_level="high",
        ),
    )

    assert response.startswith('Action: execute_macro("pal_benchmark_bridge_macro"')
    assert controller._tool_invoked_in_last_inference == (
        "pal_sparql_query_tool_5_deadbeef -> pal_benchmark_bridge_macro"
    )


def test_materialize_adapter_response_carries_semantic_bridge_payload() -> None:
    controller = _make_controller()
    controller._pending_macro_runs = {}
    controller._ensure_bridge_tool = lambda tool_name: tool_name
    controller._get_run_id = lambda: "knowledge_graph_6"
    controller._get_macro_state_dir = lambda: "outputs/test_state"

    response = controller._materialize_adapter_response(
        task_question="Question: ...",
        generated_tool_name="pal_sparql_query_tool_6_deadbeef",
        query_plan={
            "query_shape": "count_over_direct_relation",
            "answer_mode": "count",
            "answer_target_phrase": "infectious diseases",
            "relation_paths": [
                {
                    "relation": "biology.organism.diseases_transmitted",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
        },
        materialization=BenchmarkMaterialization(
            materialization_type="bridge_action",
            needs_bridge=True,
            bridge_action='Action: execute_macro("pal_benchmark_bridge_macro", {"run_id": "knowledge_graph_6", "state_dir": "outputs/test_state"})',
            bridge_tool_name="pal_benchmark_bridge_macro",
            bridge_payload={"run_id": "knowledge_graph_6", "state_dir": "outputs/test_state"},
            final_variable=None,
            final_answer_text=None,
            diagnostics={
                "artifact_type": "count_scalar",
                "artifact_source": "raw_execution",
                "selected_query_variable": "count",
                "binding_count": 1,
                "unique_value_count": 1,
                "value_preview": ["4"],
                "row_preview": [{"count": "4"}],
            },
            confidence=1.0,
            determinism_level="high",
            semantic_description="count result returned by the PAL query",
            solves_task=True,
            trusted_for_materialization=True,
        ),
        tool_result_semantics={
            "tool_result_semantic_description": "count of infectious diseases returned by the executed PAL query",
            "tool_result_solves_task": True,
            "tool_result_trusted_for_materialization": True,
        },
    )

    assert response.startswith('Action: execute_macro("pal_benchmark_bridge_macro"')
    assert (
        '"pal_semantic_description": "count of infectious diseases returned by the executed PAL query"'
        in response
    )
    assert '"pal_solves_task": true' in response
    assert '"pal_trusted_for_materialization": true' in response
    assert '"pal_tool_status": "success"' in response
    assert '"pal_selected_query_variable": "count"' in response
    assert '"pal_binding_count": 1' in response
    assert '"pal_unique_value_count": 1' in response
    assert '"pal_value_preview": ["4"]' in response
    assert '"pal_row_preview": [{"count": "4"}]' in response
    assert (
        '"pal_relation_summary": ["anchor -> count set via biology.organism.diseases_transmitted"]'
        in response
    )
    assert (
        '"pal_selection_basis": "Counts distinct infectious diseases that satisfy the executed relation constraints."'
        in response
    )
    assert (
        '"pal_completeness_hint": "Exhaustive over the exact counted bindings matched by the executed relation constraints."'
        in response
    )
    assert (
        '"pal_proof_hint": "PAL query already computed the final count from the exact relation constraints listed above."'
        in response
    )
    assert '"pal_answer_cardinality_hint": "single"' in response
    pending = controller._pending_macro_runs["knowledge_graph_6"]
    assert pending["tool_name"] == "pal_benchmark_bridge_macro"


def test_extract_macro_pointer_requires_trusted_final_contract() -> None:
    controller = _make_controller()

    assert controller._extract_macro_pointer(
        "Macro result: pal_benchmark_bridge_macro -> SUCCESS.\n"
        "Final variable: #3\n"
        "Semantic: count result returned by the PAL query\n"
        "Solves task: yes\n"
        "Trusted final: yes"
    ) == "#3"
    assert (
        controller._extract_macro_pointer(
            "Macro result: pal_benchmark_bridge_macro -> SUCCESS.\n"
            "Final variable: #3\n"
            "Semantic: bounded candidate set returned by the PAL query\n"
            "Solves task: no\n"
            "Trusted final: no"
        )
        is None
    )


def test_trusted_macro_result_uses_solver_review_before_finalizing() -> None:
    controller = _make_controller()
    controller._log_macro_result = lambda content: None
    controller._get_run_id = lambda: "knowledge_graph_7"

    class _FallbackAgent:
        _tool_invoked_in_last_inference = "manual_solver"

        def _inference(self, chat_history):
            return ChatHistoryItem(role=Role.AGENT, content="Final Answer: #3")

    controller._manual_fallback_agent = _FallbackAgent()

    chat_history = ChatHistory()
    chat_history.inject(
        ChatHistoryItem(
            role=Role.USER,
            content=(
                "Macro result: pal_benchmark_bridge_macro -> SUCCESS.\n"
                "Final variable: #3\n"
                "Semantic: count result returned by the PAL query\n"
                "Artifact type: count_scalar\n"
                "Projected query variable: count\n"
                "Raw binding count: 1\n"
                "Unique value count: 1\n"
                "Row preview: count=4\n"
                "Solves task: yes\n"
                "Trusted final: yes"
            ),
        )
    )

    response = controller._inference(chat_history)

    assert response.content == "Final Answer: #3"
    assert "knowledge_graph_7" in controller._manual_fallback_active_runs
    assert controller._tool_invoked_in_last_inference == "manual_solver"


def test_default_manual_fallback_agent_is_plain_solver() -> None:
    controller = _make_controller()
    controller._language_model = type("LM", (), {"role_dict": {Role.USER: "user", Role.AGENT: "assistant"}})()
    controller._inference_config_dict = {
        "tool_choice": "auto",
        "tools": [{"name": "ignored"}],
    }

    agent = controller._get_manual_fallback_agent()

    assert isinstance(agent, LanguageModelAgent)
    assert "Output EXACTLY ONE LINE" in agent._system_prompt
    assert agent._inference_config_dict["tool_choice"] == "none"
    assert "tools" not in agent._inference_config_dict


def test_tool_evolution_failure_bypass_raises_without_manual_solver(monkeypatch) -> None:
    controller = _make_controller()
    emitted_events: list[dict[str, object]] = []
    controller._emit_generated_tools_event = lambda payload: emitted_events.append(
        dict(payload)
    )
    monkeypatch.setenv("PAL_TOOL_EVOLUTION_SKIP_MANUAL_FALLBACK", "1")

    with pytest.raises(
        AgentUnknownException,
        match="pal_tool_failure_bypassed:repairable_bad_count_set",
    ):
        controller._bypass_manual_solver_after_tool_failure(
            generated_tool_name="pal_sparql_query_tool_demo",
            failure_reason="repairable_bad_count_set",
            advisory_text=(
                "The repaired PAL count plan is low-trust and should be recorded "
                "for evolution without waiting for the manual solver."
            ),
        )

    assert controller._tool_invoked_in_last_inference == "pal_tool_failure_bypass"
    assert emitted_events[-1]["event"] == "pal_tool_failure_bypassed"
    assert emitted_events[-1]["tool_name"] == "pal_sparql_query_tool_demo"
    assert emitted_events[-1]["failure_reason"] == "repairable_bad_count_set"


def test_macro_result_without_trusted_final_uses_manual_fallback() -> None:
    controller = _make_controller()
    controller._log_macro_result = lambda content: None
    controller._get_run_id = lambda: "knowledge_graph_8"

    class _FallbackAgent:
        _tool_invoked_in_last_inference = "manual_solver"

        def _inference(self, chat_history):
            return ChatHistoryItem(role=Role.AGENT, content='Action: get_relations("Southern Min")')

    controller._manual_fallback_agent = _FallbackAgent()

    chat_history = ChatHistory()
    chat_history.inject(
        ChatHistoryItem(
            role=Role.USER,
            content=(
                "Macro result: pal_benchmark_bridge_macro -> SUCCESS.\n"
                "Final variable: #3\n"
                "Semantic: bounded candidate set returned by the PAL query\n"
                "Solves task: no\n"
                "Trusted final: no"
            ),
        )
    )

    response = controller._inference(chat_history)

    assert response.content == 'Action: get_relations("Southern Min")'
    assert "knowledge_graph_8" in controller._manual_fallback_active_runs
    assert controller._tool_invoked_in_last_inference == "manual_solver"


def test_sample6_trusted_entity_set_still_requires_solver_review() -> None:
    controller = _make_controller()
    controller._log_macro_result = lambda content: None
    controller._get_run_id = lambda: "knowledge_graph_6"

    class _FallbackAgent:
        _tool_invoked_in_last_inference = "manual_solver"

        def _inference(self, chat_history):
            return ChatHistoryItem(role=Role.AGENT, content="Action: get_relations(#0)")

    controller._manual_fallback_agent = _FallbackAgent()

    chat_history = ChatHistory()
    chat_history.inject(
        ChatHistoryItem(
            role=Role.USER,
            content=(
                "Macro result: pal_benchmark_bridge_macro -> SUCCESS.\n"
                "Final variable: #0\n"
                "Semantic: bounded set of entity ids returned by the PAL query "
                "(query variable 'release', 13 raw bindings, 13 unique values)\n"
                "Artifact type: entity_set\n"
                "Projected query variable: release\n"
                "Raw binding count: 13\n"
                "Unique value count: 13\n"
                "Value preview: m.0ff7bxz, m.039v5j5, m.0dntcfd\n"
                "Row preview: release=m.0ff7bxz; release_name=First Release\n"
                "Solves task: yes\n"
                "Trusted final: yes\n"
                "Observation: PAL benchmark bridge materialized entity_set into a benchmark variable.\n"
                "Confidence: 1.0"
            ),
        )
    )

    response = controller._inference(chat_history)

    assert response.content == "Action: get_relations(#0)"
    assert "knowledge_graph_6" in controller._manual_fallback_active_runs
    assert controller._tool_invoked_in_last_inference == "manual_solver"


def test_sample7_trusted_shared_type_result_still_requires_solver_review() -> None:
    controller = _make_controller()
    controller._log_macro_result = lambda content: None
    controller._get_run_id = lambda: "knowledge_graph_7"

    class _FallbackAgent:
        _tool_invoked_in_last_inference = "manual_solver"

        def _inference(self, chat_history):
            return ChatHistoryItem(role=Role.AGENT, content="Action: get_neighbors(#0, type.type.instance)")

    controller._manual_fallback_agent = _FallbackAgent()

    chat_history = ChatHistory()
    chat_history.inject(
        ChatHistoryItem(
            role=Role.USER,
            content=(
                "Macro result: pal_benchmark_bridge_macro -> SUCCESS.\n"
                "Final variable: #0\n"
                "Semantic: single entity id returned by the PAL query "
                "(query variable 'shared_type', 29 raw bindings, 1 unique values)\n"
                "Artifact type: entity_id\n"
                "Projected query variable: shared_type\n"
                "Raw binding count: 29\n"
                "Unique value count: 1\n"
                "Value preview: m.0hhbr\n"
                "Row preview: shared_type=m.0hhbr; name=Art museum\n"
                "Solves task: yes\n"
                "Trusted final: yes\n"
                "Observation: PAL benchmark bridge materialized entity_id into a benchmark variable.\n"
                "Confidence: 1.0"
            ),
        )
    )

    response = controller._inference(chat_history)

    assert response.content == "Action: get_neighbors(#0, type.type.instance)"
    assert "knowledge_graph_7" in controller._manual_fallback_active_runs
    assert controller._tool_invoked_in_last_inference == "manual_solver"


def test_validate_pal_execution_rejects_generic_shared_type_dump() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "shared_type_intersection",
            "anchored_entities": [
                {"surface": "the museum of modern art", "chosen_alias": "the museum of modern art", "role": "anchor_a"},
                {"surface": "Smithsonian Institution", "chosen_alias": "Smithsonian Institution", "role": "anchor_b"},
            ],
            "relation_paths": [
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from_role": "anchor_a",
                    "to_role": "shared_type",
                    "grounding_source": "curated",
                },
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from_role": "anchor_b",
                    "to_role": "shared_type",
                    "grounding_source": "curated",
                },
            ],
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "shared_type"},
                    {"anchor_role": "anchor_b", "constrains_variable": "shared_type"},
                ],
            },
            "shared_answer_variable": "shared_type",
            "candidate_set_variable": "",
            "count_set_variable": "",
            "ordering_attribute": {},
            "allow_exploratory_predicates": False,
        },
        result_dict={
            "head": {"vars": ["shared_type"]},
            "results": {
                "bindings": [
                    {"shared_type": {"type": "uri", "value": "http://rdf.freebase.com/ns/common.topic"}},
                    {"shared_type": {"type": "uri", "value": "http://rdf.freebase.com/ns/base.type_ontology.abstract"}},
                    {"shared_type": {"type": "uri", "value": "http://rdf.freebase.com/ns/type.object"}},
                    {"shared_type": {"type": "uri", "value": "http://rdf.freebase.com/ns/common.notable_for.display_name"}},
                    {"shared_type": {"type": "uri", "value": "http://rdf.freebase.com/ns/base.type_ontology.agent"}},
                    {"shared_type": {"type": "uri", "value": "http://rdf.freebase.com/ns/base.tagit.place"}},
                ]
            },
        },
        query_text=(
            "SELECT DISTINCT ?shared_type WHERE { "
            "?anchor_a fb:type.object.name ?anchor_a_label . "
            'FILTER(LCASE(STR(?anchor_a_label)) = "the museum of modern art") '
            "?anchor_b fb:type.object.name ?anchor_b_label . "
            'FILTER(LCASE(STR(?anchor_b_label)) = "smithsonian institution") '
            "?anchor_a fb:type.object.type ?shared_type . "
            "?anchor_b fb:type.object.type ?shared_type . }"
        ),
        entities=["the museum of modern art", "Smithsonian Institution"],
        anchor_probe_results=None,
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "shared_type_result_overbroad_generic_type_dump" in verdict.reasons


def test_adapter_rejects_unstructured_solver_fallback_materialization() -> None:
    adaptation = adapt_pal_result_to_benchmark(
        raw_result=None,
        solver_output=(
            "Final Answer: Free verse uses no regular meter and relies on cadence."
        ),
        context=BenchmarkAdapterContext(
            task_question="Question: what type of meter is used in free verse?",
            run_id="knowledge_graph_22",
            state_dir="outputs/test_state",
        ),
    )

    assert adaptation.artifact.artifact_type == "unresolved"
    assert adaptation.materialization.needs_bridge is False
    assert adaptation.materialization.materialization_type == "unresolved_failure"


def test_adapter_does_not_materialize_empty_artifact() -> None:
    adaptation = adapt_pal_result_to_benchmark(
        raw_result={
            "head": {"vars": ["answer"]},
            "results": {"bindings": []},
        },
        solver_output=None,
        context=BenchmarkAdapterContext(
            task_question="Question: what type of meter is used in free verse?",
            run_id="knowledge_graph_22",
            state_dir="outputs/test_state",
        ),
    )

    assert adaptation.artifact.artifact_type == "empty"
    assert adaptation.materialization.needs_bridge is False
    assert adaptation.materialization.materialization_type == "empty_failure"
    assert adaptation.materialization.diagnostics["artifact_type"] == "empty"


def test_build_attempt_decision_record_includes_family_bundle_and_trust_contract() -> None:
    controller = _make_controller()
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "relation_paths": [
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "curated",
            }
        ],
        "count_set_variable": "count_set",
    }
    selection = select_reusable_tool(query_plan)
    trust_contract = build_trust_contract_evaluation(
        plan_consistency_passed=True,
        execution_shape_passed=False,
        plausibility_validation_passed=True,
        adapter_safety_passed=True,
        extra_denial_reasons=("expected_count_scalar:entity_set",),
    )

    decision = controller._build_attempt_decision_record(
        query_plan=query_plan,
        generation_source="reusable_tool",
        reusable_selection=selection,
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REJECTED_DANGEROUS_OVERREACH,
            reasons=["dangerous_overreach:weak_count_semantics"],
        ),
        trust_contract=trust_contract,
    )

    assert decision.selected_family == "count_over_direct_relation"
    assert decision.family_bundle_version is not None
    assert decision.materialization_allowed is False
    assert "expected_count_scalar:entity_set" in decision.materialization_denial_reasons
    assert decision.dangerous_overreach is True


def test_classify_execution_artifact_rejects_missing_selected_head_var() -> None:
    artifact = classify_execution_artifact(
        {
            "head": {"vars": ["shared_answer", "shared_answer_name"]},
            "results": {
                "bindings": [
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.02gx21",
                        }
                    }
                ]
            },
        }
    )

    assert artifact.artifact_type == "unresolved"
    assert artifact.diagnostics["reason"] == "selected_head_var_missing_from_bindings"


def test_should_accept_best_executing_candidate_rejects_positive_low_support_count_result() -> None:
    controller = _make_controller()

    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_COUNT_SET,
            "verdict_reasons": [
                "ambiguous_anchor_surface_binding:'Aedes aegypti':12",
                "count_anchor_path_low_support:'Aedes aegypti':1:medicine.infectious_disease.vector",
                "count_scalar_returned:1",
                "count_query_dynamic_chain_too_weak",
            ],
            "binding_count": 1,
            "scalar_count": 1,
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "all_anchors_found": True,
            "has_order_by_limit": False,
            "score": 30,
        }
    )


def test_should_accept_best_executing_candidate_allows_exhausted_zero_count_with_live_anchors() -> None:
    controller = _make_controller()

    assert controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_COUNT_SET,
            "verdict_reasons": [
                "anchor_path_empty:'Canada':biology.animal_breed.country_of_origin",
                "count_set_path_empty",
                "count_scalar_returned:0",
            ],
            "binding_count": 1,
            "scalar_count": 0,
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "all_anchors_found": True,
            "has_order_by_limit": False,
            "score": 23,
        }
    )


def test_should_accept_best_executing_candidate_rejects_zero_count_with_live_anchor_paths() -> None:
    controller = _make_controller()

    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_COUNT_SET,
            "verdict_reasons": [
                "count_query_zero_with_live_anchor_paths",
                "count_scalar_returned:0",
            ],
            "binding_count": 1,
            "scalar_count": 0,
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "all_anchors_found": True,
            "has_order_by_limit": False,
            "score": 68,
        }
    )


def test_should_accept_best_executing_candidate_rejects_unverified_type_count() -> None:
    controller = _make_controller()

    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_COUNT_SET,
            "verdict_reasons": [
                "count_query_unverified_type_constraint",
                "count_scalar_returned:0",
            ],
            "binding_count": 1,
            "scalar_count": 0,
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "all_anchors_found": True,
            "has_order_by_limit": False,
            "score": 35,
        }
    )


def test_should_accept_best_executing_candidate_rejects_unenforced_answer_target_count() -> None:
    controller = _make_controller()

    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_COUNT_SET,
            "verdict_reasons": [
                "count_answer_target_unenforced:game expansions",
                "count_scalar_returned:35",
            ],
            "binding_count": 1,
            "scalar_count": 35,
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "all_anchors_found": True,
            "has_order_by_limit": False,
            "score": 35,
        }
    )


def test_should_accept_best_executing_candidate_rejects_plan_inconsistent_candidate() -> None:
    controller = _make_controller()

    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_COUNT_SET,
            "verdict_reasons": [
                "count_query_zero_with_live_anchor_paths",
                "count_scalar_returned:0",
            ],
            "binding_count": 1,
            "scalar_count": 0,
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "all_anchors_found": True,
            "plan_validation_errors": [
                "query_uses_unplanned_predicate:broadcast.producer.produces"
            ],
            "has_order_by_limit": False,
            "score": 60,
        }
    )


def test_should_accept_best_executing_candidate_rejects_exploratory_empty_superlative() -> None:
    controller = _make_controller()

    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata={
            "verdict": VERDICT_REPAIRABLE_BAD_SUPERLATIVE,
            "verdict_reasons": [
                "ordering_attribute_path_exploratory",
                "repair:add_grounded_candidate_set_ordering_attribute_and_ORDER_BY",
            ],
            "binding_count": 0,
            "scalar_count": None,
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "all_anchors_found": True,
            "has_order_by_limit": True,
            "score": 40,
        }
    )


def test_best_executing_candidate_scoring_prefers_grounded_positive_count_over_zero_with_unverified_type() -> None:
    controller = _make_controller()

    positive_metadata = controller._build_best_executing_candidate_metadata(
        generated_code="def solve(endpoint_url):\n    return {}",
        invocation_result=PALInvocationResult(
            success=True,
            payload={"head": {"vars": ["count"]}, "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]}},
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
        },
        query_text="SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { VALUES ?anchor { fb:m.06y6_w } ?candidate_set fb:medicine.infectious_disease.vector ?anchor . }",
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "ambiguous_anchor_surface_binding:'Aedes aegypti':12",
                "count_anchor_path_low_support:'Aedes aegypti':1:medicine.infectious_disease.vector",
                "count_scalar_returned:1",
                "count_query_dynamic_chain_too_weak",
            ],
        ),
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Aedes aegypti",
                entity_count=12,
                path_count=1,
                relation_probed="medicine.infectious_disease.vector",
                anchor_position="object",
                resolved_entity_id="m.06y6_w",
            )
        ],
    )

    zero_with_unverified_type = controller._build_best_executing_candidate_metadata(
        generated_code="def solve(endpoint_url):\n    return {}",
        invocation_result=PALInvocationResult(
            success=True,
            payload={"head": {"vars": ["count"]}, "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]}},
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
        },
        query_text="SELECT (COUNT(DISTINCT ?disease) AS ?count) WHERE { VALUES ?anchor { fb:m.06y6_w } ?disease fb:medicine.infectious_disease.vector ?anchor . ?infectiousType fb:type.object.name ?itype_name . FILTER(LCASE(STR(?itype_name)) = \"infectious disease\") ?infectiousType fb:type.type.instance ?disease . }",
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "count_query_dynamic_chain_too_weak",
                "count_query_unverified_type_constraint",
                "count_scalar_returned:0",
            ],
        ),
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Aedes aegypti",
                entity_count=12,
                path_count=1,
                relation_probed="medicine.infectious_disease.vector",
                anchor_position="object",
                resolved_entity_id="m.06y6_w",
            )
        ],
    )

    assert int(positive_metadata["score"]) > int(zero_with_unverified_type["score"])
    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata=positive_metadata
    )
    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata=zero_with_unverified_type
    )


def test_best_executing_candidate_scoring_penalizes_unenforced_answer_target_count() -> None:
    controller = _make_controller()

    metadata = controller._build_best_executing_candidate_metadata(
        generated_code="def solve(endpoint_url):\n    return {}",
        invocation_result=PALInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["count"]},
                "results": {
                    "bindings": [{"count": {"type": "literal", "value": "35"}}]
                },
            },
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
        },
        query_text="SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { ?candidate_set fb:cvg.game_version.publisher fb:m.0dwl2 . }",
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "count_answer_target_unenforced:game expansions",
                "count_scalar_returned:35",
            ],
        ),
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="valve corp",
                entity_count=1,
                path_count=35,
                relation_probed="cvg.game_version.publisher",
                anchor_position="object",
                resolved_entity_id="m.0dwl2",
            )
        ],
    )

    assert int(metadata["score"]) < 0
    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata=metadata
    )


def test_best_executing_candidate_scoring_penalizes_plan_inconsistent_query() -> None:
    controller = _make_controller()

    metadata = controller._build_best_executing_candidate_metadata(
        generated_code="def solve(endpoint_url):\n    return {}",
        invocation_result=PALInvocationResult(
            success=True,
            payload={
                "head": {"vars": ["count"]},
                "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
            },
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {"surface": "Higher Education", "chosen_alias": "m.03bv2kt", "role": "anchor_a"},
                {"surface": "To the Best of Our Knowledge", "chosen_alias": "m.03fx9_c", "role": "anchor_b"},
            ],
            "candidate_set_variable": "?content",
            "count_set_variable": "?count",
            "shared_answer_variable": "?content",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "?content"},
                    {"anchor_role": "anchor_b", "constrains_variable": "?content"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "broadcast.content.genre",
                    "direction": "reverse",
                    "from": "content",
                    "to": "Higher Education",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "To the Best of Our Knowledge",
                    "to": "producer",
                    "from_role": "anchor_b",
                    "to_role": "anchor_value",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "content",
                    "to": "producer",
                    "from_role": "candidate_set",
                    "to_role": "anchor_value",
                    "grounding_source": "dynamic_probe",
                },
            ],
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?candidate_set_content fb:broadcast.content.genre fb:m.03bv2kt . "
            "?candidate_set_content fb:broadcast.producer.produces fb:m.03fx9_c . "
            "}"
        ),
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "count_query_zero_with_live_anchor_paths",
                "count_scalar_returned:0",
            ],
        ),
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Higher Education",
                entity_count=1,
                path_count=32,
                relation_probed="broadcast.content.genre",
                anchor_position="object",
                resolved_entity_id="m.03bv2kt",
            ),
            AnchorProbeResult(
                anchor_name="To the Best of Our Knowledge",
                entity_count=1,
                path_count=2,
                relation_probed="broadcast.content.producer",
                anchor_position="subject",
                resolved_entity_id="m.03fx9_c",
            ),
        ],
    )

    assert "query_uses_unplanned_predicate:broadcast.producer.produces" in metadata["plan_validation_errors"]
    assert int(metadata["score"]) < 0
    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata=metadata
    )


def test_ensure_repair_loop_accepted_allows_best_effort_final_verdict() -> None:
    controller = _make_controller()

    controller._ensure_repair_loop_accepted(
        generated_tool_name="pal_sparql_query_tool_test",
        repair_loop_log={
            "final_verdict": "accepted_best_effort",
            "accepted_attempt": "best_executing",
        },
    )


def test_materialization_trust_contract_rejects_accepted_best_effort_salvage() -> None:
    controller = _make_controller()
    trust_contract = controller._evaluate_materialization_trust_contract(
        generated_code="def solve(endpoint_url):\n    return {}",
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "relation_paths": [
                {
                    "relation": "broadcast.genre.content",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "candidate_set",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                }
            ],
            "count_set_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?anchor_a fb:broadcast.genre.content ?candidate_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        verdict=PlausibilityVerdict(
            verdict="accepted_best_effort",
            reasons=[
                "count_query_zero_with_live_anchor_paths",
                "count_scalar_returned:0",
            ],
        ),
        materialization=BenchmarkMaterialization(
            materialization_type="bridge_action",
            needs_bridge=True,
            bridge_action="Action: execute_macro(...)",
            bridge_tool_name="pal_benchmark_bridge_macro",
            bridge_payload={},
            final_variable=None,
            final_answer_text=None,
            diagnostics={"artifact_type": "count_scalar", "artifact_source": "raw_execution"},
            confidence=1.0,
            determinism_level="deterministic_raw",
        ),
        artifact_type="count_scalar",
        artifact_source="raw_execution",
    )

    assert trust_contract.plausibility_validation_passed is False
    assert trust_contract.materialization_allowed is False
    assert "plausibility_validation_failed" in trust_contract.denial_reasons


def test_probe_guided_alias_repair_updates_plan_and_grounding_card() -> None:
    controller = _make_controller()
    probe_counts = {"cow": 0, "Cattle": 1}
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: probe_counts.get(alias, -1)
    )

    repaired_plan, repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question="Question: what semi-firm textured cheese is made from the products of goat and cows?, Entities: ['Goat', 'cows', 'semi-firm']",
        query_plan={
            "anchored_entities": [
                {"surface": "Goat", "chosen_alias": "Goat", "role": "anchor_a"},
                {"surface": "cows", "chosen_alias": "cows", "role": "anchor_b"},
                {"surface": "semi-firm", "chosen_alias": "semi-firm", "role": "anchor_value"},
            ],
            "normalized_aliases": [],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type("Probe", (), {"entity_count": 1})(),
            type("Probe", (), {"entity_count": 0})(),
        ],
    )

    assert repaired_plan["anchored_entities"][1]["chosen_alias"] == "Cattle"
    assert "recommended_alias='Cattle'" in repaired_grounding_card
    assert any(item["chosen_alias"] == "Cattle" for item in repaired_plan["normalized_aliases"])
    assert "anchor_alias_override:cows=>Cattle" in feedback


def test_probe_guided_alias_repair_can_use_path_count_signal() -> None:
    controller = _make_controller()
    entity_counts = {"Seventh Sphere": 1}
    path_counts = {("Seventh Sphere", "fictional_universe.fictional_universe.species"): 45}
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: entity_counts.get(alias, -1)
    )
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=1.5: path_counts.get((alias, relation), 0)
    )

    repaired_plan, repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question="Question: what number of different species are in the fictional world of seventh sphere?, Entities: ['Seventh sphere']",
        query_plan={
            "anchored_entities": [
                {"surface": "Seventh sphere", "chosen_alias": "Seventh sphere", "role": "anchor"},
            ],
            "normalized_aliases": [],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type(
                "Probe",
                (),
                {
                    "entity_count": 1,
                    "path_count": 0,
                    "relation_probed": "fictional_universe.fictional_universe.species",
                    "anchor_position": "subject",
                },
            )(),
        ],
    )

    assert repaired_plan["anchored_entities"][0]["chosen_alias"] == "Seventh Sphere"
    assert "recommended_alias='Seventh Sphere'" in repaired_grounding_card
    assert "anchor_alias_override:Seventh sphere=>Seventh Sphere" in feedback


def test_probe_guided_alias_repair_uses_entity_aliases_for_attribute_constraints() -> None:
    controller = _make_controller()
    entity_counts = {"educator": 1, "Educator": 1}
    path_counts = {
        ("educator", "people.profession.people_with_this_profession", "subject"): 12,
        ("Educator", "people.profession.people_with_this_profession", "subject"): 12,
    }
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: entity_counts.get(alias, 0)
    )
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=1.5: path_counts.get((alias, relation, anchor_position), 0)
    )

    repaired_plan, repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question=(
            "Question: what is the number of film characters that are with educators "
            "occupation and neyaphem species?, Entities: ['educators', 'Neyaphem']"
        ),
        query_plan={
            "anchored_entities": [
                {"surface": "educators", "chosen_alias": "educators", "role": "constraint_value"},
                {"surface": "Neyaphem", "chosen_alias": "Neyaphem", "role": "anchor_b"},
            ],
            "normalized_aliases": [],
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                }
            ],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type(
                "Probe",
                (),
                {
                    "entity_count": 0,
                    "path_count": None,
                    "relation_probed": "people.profession.people_with_this_profession",
                    "anchor_position": "subject",
                },
            )(),
            type("Probe", (), {"entity_count": 1})(),
        ],
    )

    assert repaired_plan["anchored_entities"][0]["chosen_alias"] == "educator"
    assert "recommended_alias='educator'" in repaired_grounding_card
    assert "anchor_alias_override:educators=>educator" in feedback


def test_probe_guided_alias_repair_can_use_token_search_with_inferred_anchor_probe() -> None:
    controller = _make_controller()
    entity_counts = {
        "Walker Brothers": 1,
        "Alan Walker": 1,
    }
    path_counts = {
        ("Walker Brothers", "music.recording.composer", "subject"): 0,
        ("Alan Walker", "music.recording.composer", "subject"): 6,
    }
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: entity_counts.get(alias, 0)
    )
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=1.5:
        path_counts.get((alias, relation, anchor_position), 0)
    )
    controller._probe_entity_name_candidates_by_token_search = (
        lambda entity, timeout_s=2.0, limit=8: ["Walker Brothers", "Alan Walker"]
    )

    repaired_plan, _repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question="Question: what is the name of the singer that performed the tv song composed by walker?",
        query_plan={
            "anchored_entities": [
                {"surface": "walker", "chosen_alias": "walker", "role": "anchor"},
            ],
            "normalized_aliases": [],
            "relation_paths": [
                {
                    "relation": "music.recording.composer",
                    "direction": "reverse",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "from": "anchor",
                    "to": "candidate_set",
                }
            ],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type("Probe", (), {"entity_count": 0, "path_count": None, "relation_probed": None})(),
        ],
    )

    assert repaired_plan["anchored_entities"][0]["chosen_alias"] == "Alan Walker"
    assert "anchor_alias_override:walker=>Alan Walker" in feedback


def test_infer_anchor_relation_probe_uses_object_for_reverse_from_role_anchor() -> None:
    controller = _make_controller()

    inferred = controller._infer_anchor_relation_probe_from_query_plan(
        query_plan={
            "relation_paths": [
                {
                    "relation": "cvg.cvg_developer.game_versions_developed",
                    "direction": "reverse",
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                    "from": "anchor_b",
                    "to": "candidate_set_anchor_b",
                }
            ]
        },
        anchored_entity={
            "surface": "Sega of Japan",
            "chosen_alias": "Sega of Japan",
            "role": "anchor_b",
        },
    )

    assert inferred == ("cvg.cvg_developer.game_versions_developed", "object")


def test_infer_anchor_relation_probe_uses_subject_for_reverse_constraint_anchor() -> None:
    controller = _make_controller()

    inferred = controller._infer_anchor_relation_probe_from_query_plan(
        query_plan={
            "relation_paths": [
                {
                    "relation": "biology.breed_origin.breeds_originating_here",
                    "direction": "reverse",
                    "from_role": "constraint_value",
                    "to_role": "anchor_a",
                    "from": "country",
                    "to": "anchor_a",
                }
            ]
        },
        anchored_entity={
            "surface": "Serbia",
            "chosen_alias": "Serbia",
            "role": "anchor_a",
        },
    )

    assert inferred == ("biology.breed_origin.breeds_originating_here", "subject")


def test_infer_anchor_relation_probe_uses_object_for_reverse_candidate_set_anchor() -> None:
    controller = _make_controller()

    inferred = controller._infer_anchor_relation_probe_from_query_plan(
        query_plan={
            "relation_paths": [
                {
                    "relation": "broadcast.content.genre",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "from": "content",
                    "to": "anchor_a",
                }
            ]
        },
        anchored_entity={
            "surface": "Higher Education",
            "chosen_alias": "Higher Education",
            "role": "anchor_a",
        },
    )

    assert inferred == ("broadcast.content.genre", "object")


def test_resolve_anchor_probe_target_uses_object_for_reverse_anchor_endpoint() -> None:
    controller = _make_controller()

    relation, anchor_position = controller._resolve_anchor_probe_target(
        anchored_entity={
            "surface": "Nebula",
            "chosen_alias": "Nebula",
            "role": "anchor",
        },
        query_plan={},
        relation_paths=[
            {
                "relation": "astronomy.celestial_object.category",
                "direction": "reverse",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "from": "category",
                "to": "candidate_set",
            }
        ],
        used_path_indexes=set(),
    )

    assert relation == "astronomy.celestial_object.category"
    assert anchor_position == "object"


def test_resolve_anchor_probe_target_uses_object_for_reverse_candidate_set_anchor() -> None:
    controller = _make_controller()

    relation, anchor_position = controller._resolve_anchor_probe_target(
        anchored_entity={
            "surface": "Higher Education",
            "chosen_alias": "Higher Education",
            "role": "anchor_a",
        },
        query_plan={},
        relation_paths=[
            {
                "relation": "broadcast.content.genre",
                "direction": "reverse",
                "from_role": "candidate_set",
                "to_role": "anchor_a",
                "from": "content",
                "to": "anchor_a",
            }
        ],
        used_path_indexes=set(),
    )

    assert relation == "broadcast.content.genre"
    assert anchor_position == "object"


def test_matching_dead_anchor_probe_accepts_reverse_from_role_anchor() -> None:
    controller = _make_controller()

    matched = controller._matching_dead_anchor_probe(
        candidate={
            "relation": "astronomy.celestial_object.category",
            "direction": "reverse",
            "from_role": "anchor",
            "to_role": "candidate_set",
            "from": "category",
            "to": "candidate_set",
        },
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Nebula",
                entity_count=1,
                path_count=0,
                relation_probed="astronomy.celestial_object.category",
                anchor_position="object",
                resolved_entity_id="m.0l6tq",
            )
        ],
    )

    assert matched is not None


def test_matching_dead_anchor_probe_accepts_reverse_candidate_set_anchor() -> None:
    controller = _make_controller()

    matched = controller._matching_dead_anchor_probe(
        candidate={
            "relation": "broadcast.content.genre",
            "direction": "reverse",
            "from_role": "candidate_set",
            "to_role": "anchor_a",
            "from": "content",
            "to": "anchor_a",
        },
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Higher Education",
                entity_count=1,
                path_count=0,
                relation_probed="broadcast.content.genre",
                anchor_position="object",
                resolved_entity_id="m.03bv2kt",
            )
        ],
    )

    assert matched is not None


def test_probe_guided_alias_repair_requires_relation_support_when_probe_relation_known() -> None:
    controller = _make_controller()
    entity_counts = {
        "Sega Of Japan": 0,
        "sega": 25,
    }
    path_counts = {
        ("sega", "cvg.cvg_developer.game_versions_developed", "subject"): 361,
    }
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: entity_counts.get(alias, 0)
    )
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=1.5:
        path_counts.get((alias, relation, anchor_position), 0)
    )
    controller._probe_entity_name_candidates_by_token_search = lambda *args, **kwargs: []

    repaired_plan, _repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question="Question: virtual console, which is developed by sega of japan, was released where?",
        query_plan={
            "anchored_entities": [
                {"surface": "Virtual Console", "chosen_alias": "Virtual Console", "role": "anchor_a"},
                {"surface": "sega of japan", "chosen_alias": "sega of japan", "role": "anchor_b"},
            ],
            "normalized_aliases": [],
            "relation_paths": [
                {
                    "relation": "cvg.computer_game_distribution_system.games_distributed",
                    "direction": "reverse",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "from": "anchor_a",
                    "to": "candidate_set_anchor_a",
                },
                {
                    "relation": "cvg.cvg_developer.game_versions_developed",
                    "direction": "reverse",
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                    "from": "anchor_b",
                    "to": "candidate_set_anchor_b",
                },
            ],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type("Probe", (), {"entity_count": 1, "path_count": 10, "relation_probed": "cvg.computer_game_distribution_system.games_distributed", "anchor_position": "subject"})(),
            type("Probe", (), {"entity_count": 0, "path_count": None, "relation_probed": None, "anchor_position": None})(),
        ],
    )

    assert repaired_plan["anchored_entities"][1]["chosen_alias"] == "sega"
    assert "anchor_alias_override:sega of japan=>sega" in feedback


def test_probe_guided_alias_repair_can_use_high_confidence_token_search_fallback() -> None:
    controller = _make_controller()
    entity_counts = {
        "Google Play": 31,
        "Play Store": 1,
        "Google Android": 2,
    }
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: entity_counts.get(alias, 0)
    )
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=1.5: 0
    )
    controller._probe_entity_name_candidates_by_token_search = (
        lambda entity, timeout_s=2.0, limit=8: ["Google Play", "Play Store", "Google Android"]
    )

    repaired_plan, _repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question="Question: in the google play store what video game platform is supported?",
        query_plan={
            "anchored_entities": [
                {
                    "surface": "google play store",
                    "chosen_alias": "google play store",
                    "role": "anchor",
                },
            ],
            "normalized_aliases": [],
            "relation_paths": [
                {
                    "relation": "supported_platform",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "from": "anchor",
                    "to": "answer",
                    "grounding_source": "exploratory",
                }
            ],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type(
                "Probe",
                (),
                {
                    "entity_count": 0,
                    "path_count": None,
                    "relation_probed": "supported_platform",
                    "anchor_position": "subject",
                },
            )(),
        ],
    )

    assert repaired_plan["anchored_entities"][0]["chosen_alias"] == "Google Play"
    assert "anchor_alias_override:google play store=>Google Play" in feedback


def test_probe_guided_alias_repair_skips_semantically_weak_token_search_alias() -> None:
    controller = _make_controller()
    entity_counts = {"Radiolab": 1}
    path_counts = {
        ("Radiolab", "broadcast.genre.content", "object"): 9,
    }
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=1.5: entity_counts.get(alias, 0)
    )
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=1.5:
        path_counts.get((alias, relation, anchor_position), 0)
    )
    controller._probe_entity_name_candidates_by_token_search = (
        lambda entity, timeout_s=2.0, limit=8: ["Radiolab"]
    )

    repaired_plan, _repaired_grounding_card, feedback = controller._apply_probe_guided_alias_repairs(
        task_question=(
            "Question: how much content about talk radio is produced by the person "
            "that produces weekend edition sunday?, Entities: ['Talk radio', "
            "'Weekend Edition Sunday']"
        ),
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Talk radio",
                    "chosen_alias": "Talk radio",
                    "role": "anchor_a",
                }
            ],
            "normalized_aliases": [],
            "relation_paths": [
                {
                    "relation": "broadcast.genre.content",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "from": "content",
                    "to": "anchor_a",
                }
            ],
        },
        grounding_card="PAL grounding hints:",
        relation_grounding=[],
        anchor_probe_results=[
            type(
                "Probe",
                (),
                {
                    "entity_count": 1,
                    "path_count": 0,
                    "relation_probed": "broadcast.genre.content",
                    "anchor_position": "object",
                },
            )(),
        ],
    )

    assert repaired_plan["anchored_entities"][0]["chosen_alias"] == "Talk radio"
    assert feedback == []


def test_refresh_dynamic_grounding_after_anchor_alias_change_reprobes_new_alias() -> None:
    controller = _make_controller()
    controller._build_pal_grounding_card = (
        lambda task_question, relation_grounding=None, alias_overrides=None: json.dumps(
            {
                "relations": [
                    candidate.get("relation")
                    for candidate in relation_grounding or []
                ],
                "aliases": alias_overrides or {},
            },
            sort_keys=True,
        )
    )
    controller._probe_dynamic_relation_candidates_for_anchors = (
        lambda anchored_entities, answer_target_phrase, domain_hints, question_text="", probe_timeout_s=5.0, max_anchors=2: [
            {
                "relation": "cvg.cvg_engine.successor",
                "direction": "forward",
                "from": str(anchored_entities[0].get("chosen_alias") or ""),
                "to": "successor",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            }
        ]
    )

    grounding_card, merged_candidates, feedback = (
        controller._refresh_dynamic_grounding_after_anchor_alias_change(
            task_question="Question: the quake 3 engine was proceeded by which video game engine?, Entities: ['quake 3 engine']",
            previous_alias_assignments={"quake 3 engine": "Quake 3 Engine"},
            query_plan={
                "answer_mode": "entity",
                "query_shape": "single_anchor_lookup",
                "anchored_entities": [
                    {
                        "surface": "quake 3 engine",
                        "chosen_alias": "id Tech 3",
                        "role": "anchor",
                    }
                ],
            },
            relation_grounding=[],
        )
    )

    assert any(
        candidate.get("relation") == "cvg.cvg_engine.successor"
        and candidate.get("from") == "id Tech 3"
        for candidate in merged_candidates
    )
    assert any(
        "dynamic_grounding_refreshed_after_alias_change" in item
        for item in feedback
    )
    assert "id Tech 3" in grounding_card


def test_refresh_dynamic_grounding_after_anchor_alias_change_skips_when_curated_exists() -> None:
    controller = _make_controller()
    controller._build_pal_grounding_card = (
        lambda task_question, relation_grounding=None, alias_overrides=None: json.dumps(
            {"count": len(relation_grounding or []), "aliases": alias_overrides or {}},
            sort_keys=True,
        )
    )
    controller._probe_dynamic_relation_candidates_for_anchors = (
        lambda **kwargs: pytest.fail(
            "should not re-probe when curated grounding already exists"
        )
    )

    grounding_card, merged_candidates, feedback = (
        controller._refresh_dynamic_grounding_after_anchor_alias_change(
            task_question="Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?, Entities: ['Naloxone', 'Enalaprilat']",
            previous_alias_assignments={"Naloxone": "Naloxone"},
            query_plan={
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {
                        "surface": "Naloxone",
                        "chosen_alias": "Naloxone hydrochloride",
                        "role": "anchor_a",
                    }
                ],
            },
            relation_grounding=[
                {
                    "relation": "medicine.drug_formulation.formulation_of",
                    "direction": "forward",
                    "from": "formulation",
                    "to": "drug_or_ingredient",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                }
            ],
        )
    )

    assert len(merged_candidates) == 1
    assert feedback == []
    assert "Naloxone hydrochloride" in grounding_card


def test_single_anchor_dynamic_lookup_repair_plan_uses_best_live_dynamic_candidate() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_single_anchor_dynamic_lookup_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "quake 3 engine",
                    "chosen_alias": "id Tech 3",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "succeeded_by",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "answer",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "exploratory",
                }
            ],
            "strategy": "stale exploratory lookup",
            "plan_rationale": ["initial"],
        },
        relation_grounding=[
            {
                "relation": "cvg.computer_game_engine.predecessor_engine",
                "direction": "forward",
                "from": "id Tech 3",
                "to": "predecessor",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "cvg.computer_game_engine.successor_engine",
                "direction": "forward",
                "from": "id Tech 3",
                "to": "successor",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
        ],
    )

    assert repaired_plan is not None
    assert repaired_plan["relation_paths"][0]["relation"] == "cvg.computer_game_engine.predecessor_engine"
    assert repaired_plan["relation_paths"][0]["from"] == "anchor"
    assert repaired_plan["relation_paths"][0]["to"] == "answer"
    assert repaired_plan["relation_paths"][0]["to_role"] == "answer"


def test_single_anchor_dynamic_lookup_repair_plan_can_replace_dead_grounded_relation() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_single_anchor_dynamic_lookup_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "saxe-coburg-gotha",
                    "chosen_alias": "saxe-coburg and gotha",
                    "role": "anchor",
                    "resolved_entity_id": "m.01f0_j",
                }
            ],
            "relation_paths": [
                {
                    "relation": "royalty.kingdom.monarchs",
                    "direction": "forward",
                    "from": "kingdom",
                    "to": "monarch",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "curated",
                }
            ],
            "strategy": "grounded direct lookup",
            "plan_rationale": ["initial"],
        },
        relation_grounding=[
            {
                "relation": "royalty.monarch.kingdom",
                "direction": "forward",
                "from": "monarch",
                "to": "kingdom",
                "from_role": "answer",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
    )

    assert repaired_plan is not None
    assert repaired_plan["relation_paths"][0]["relation"] == "royalty.monarch.kingdom"
    assert repaired_plan["relation_paths"][0]["from"] == "answer"
    assert repaired_plan["relation_paths"][0]["to"] == "anchor"
    assert repaired_plan["relation_paths"][0]["to_role"] == "anchor"


def test_normalize_pal_query_plan_retags_single_anchor_direct_candidate_set_as_answer() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Samsung S1050", "chosen_alias": "m.03q2r11", "role": "anchor"}
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "relation_paths": [
                {
                    "relation": "digicams.digital_camera.format",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "format",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "projection": ["answer", "answer_name"],
        }
    )

    assert normalized["relation_paths"][0]["to"] == "answer"
    assert normalized["relation_paths"][0]["to_role"] == "answer"


def test_superlative_dynamic_anchor_repair_plan_rewrites_dead_anchor_family() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_superlative_dynamic_anchor_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "David Han",
                    "chosen_alias": "David Han",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "music.recording.length",
                "direction": "forward",
                "source_variable": "recording",
                "attribute_variable": "length",
            },
            "relation_paths": [
                {
                    "relation": "music.artist.track",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "recording",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.recording.releases",
                    "direction": "forward",
                    "from": "recording",
                    "to": "candidate_set",
                    "from_role": "candidate_set",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.recording.length",
                    "direction": "forward",
                    "from": "recording",
                    "to": "length",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "strategy": "grounded superlative lookup",
            "plan_rationale": ["initial"],
        },
        relation_grounding=[
            {
                "relation": "music.engineer.tracks_engineered",
                "direction": "forward",
                "from": "David Han",
                "to": "track",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "music.recording.engineer",
                "direction": "reverse",
                "from": "recording",
                "to": "David Han",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="David Han",
                entity_count=2,
                path_count=0,
                relation_probed="music.artist.track",
                anchor_position="subject",
                resolved_entity_id=None,
            )
        ],
    )

    assert repaired_plan is not None
    assert repaired_plan["relation_paths"][0]["relation"] == "music.engineer.tracks_engineered"
    assert repaired_plan["relation_paths"][0]["from"] == "anchor"
    assert repaired_plan["relation_paths"][0]["to"] == "recording"
    assert repaired_plan["shared_answer_variable"] == "candidate_set"


def test_superlative_dynamic_anchor_repair_plan_collapses_redundant_projection_hop() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_superlative_dynamic_anchor_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Gavin Lurssen",
                    "chosen_alias": "Gavin Lurssen",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "music.release.release_date",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "release_date",
            },
            "relation_paths": [
                {
                    "relation": "music.recording.engineers",
                    "direction": "reverse",
                    "from": "recording",
                    "to": "engineer",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "music.recording.releases",
                    "direction": "forward",
                    "from": "recording",
                    "to": "release",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.release.release_date",
                    "direction": "forward",
                    "from": "release",
                    "to": "release_date",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "exploratory",
                },
            ],
            "strategy": "grounded superlative lookup",
            "plan_rationale": ["initial"],
        },
        relation_grounding=[
            {
                "relation": "music.engineer.releases_engineered",
                "direction": "forward",
                "from": "Gavin Lurssen",
                "to": "release",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Gavin Lurssen",
                entity_count=1,
                path_count=0,
                relation_probed="music.recording.engineers",
                anchor_position="subject",
                resolved_entity_id="m.01vzjzb",
            )
        ],
    )

    assert repaired_plan is not None
    assert len(repaired_plan["relation_paths"]) == 2
    assert repaired_plan["relation_paths"][0]["relation"] == "music.engineer.releases_engineered"
    assert repaired_plan["relation_paths"][0]["from"] == "anchor"
    assert repaired_plan["relation_paths"][0]["to"] == "release"
    assert repaired_plan["relation_paths"][1]["relation"] == "music.release.release_date"


def test_superlative_dynamic_anchor_repair_plan_handles_live_style_anchor_recording_chain() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_superlative_dynamic_anchor_repair_plan(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Gavin Lurssen",
                    "chosen_alias": "Gavin Lurssen",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "music.release.release_date",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "release_date",
            },
            "relation_paths": [
                {
                    "relation": "music.recording.engineer",
                    "direction": "reverse",
                    "from": "anchor",
                    "to": "recording",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "music.recording.releases",
                    "direction": "forward",
                    "from": "recording",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.release.release_date",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "release_date",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "exploratory",
                },
            ],
            "strategy": "grounded superlative lookup",
            "plan_rationale": ["initial"],
        },
        relation_grounding=[
            {
                "relation": "music.engineer.releases_engineered",
                "direction": "forward",
                "from": "Gavin Lurssen",
                "to": "release",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Gavin Lurssen",
                entity_count=1,
                path_count=0,
                relation_probed="music.recording.engineer",
                anchor_position="subject",
                resolved_entity_id="m.01vzjzb",
            )
        ],
    )

    assert repaired_plan is not None
    assert len(repaired_plan["relation_paths"]) == 2
    assert repaired_plan["relation_paths"][0]["relation"] == "music.engineer.releases_engineered"
    assert repaired_plan["relation_paths"][0]["from"] == "anchor"
    assert repaired_plan["relation_paths"][0]["to"] == "candidate_set"
    assert repaired_plan["relation_paths"][0]["from_role"] == "anchor"
    assert repaired_plan["relation_paths"][0]["to_role"] == "candidate_set"
    assert repaired_plan["relation_paths"][1]["relation"] == "music.release.release_date"


def test_anchor_path_probe_supports_object_side_triples() -> None:
    controller = _make_controller()
    captured: dict[str, str] = {}
    controller._get_runtime_sparql_endpoint = lambda: "http://127.0.0.1:3001/kb/sparql"

    def _fake_probe(endpoint: str, sparql: str, timeout_s: float) -> list[str]:
        captured["endpoint"] = endpoint
        captured["sparql"] = sparql
        return ["2"]

    controller._run_probe_sparql_query = _fake_probe

    count = controller._probe_anchor_path_count(
        "Naloxone",
        "medicine.drug_formulation.formulation_of",
        anchor_position="object",
        timeout_s=1.0,
    )

    assert count == 2
    assert captured["endpoint"] == "http://127.0.0.1:3001/kb/sparql"
    assert (
        "?answer fb:medicine.drug_formulation.formulation_of ?anchor ."
        in captured["sparql"]
    )


def test_anchor_path_probe_falls_back_to_opposite_side_when_initial_probe_is_empty() -> None:
    controller = _make_controller()
    captured_sparql: list[str] = []
    controller._get_runtime_sparql_endpoint = lambda: "http://127.0.0.1:3001/kb/sparql"

    def _fake_probe(endpoint: str, sparql: str, timeout_s: float) -> list[str]:
        captured_sparql.append(sparql)
        if "?answer fb:biology.breed_origin.breeds_originating_here ?anchor ." in sparql:
            return ["0"]
        if "?anchor fb:biology.breed_origin.breeds_originating_here ?answer ." in sparql:
            return ["4"]
        return []

    controller._run_probe_sparql_query = _fake_probe

    count = controller._probe_anchor_path_count(
        "Serbia",
        "biology.breed_origin.breeds_originating_here",
        anchor_position="object",
        timeout_s=1.0,
    )

    assert count == 4
    assert len(captured_sparql) == 2
    assert (
        "?answer fb:biology.breed_origin.breeds_originating_here ?anchor ."
        in captured_sparql[0]
    )
    assert (
        "?anchor fb:biology.breed_origin.breeds_originating_here ?answer ."
        in captured_sparql[1]
    )


def test_anchor_entity_id_probe_falls_back_to_opposite_side_when_initial_probe_is_empty() -> None:
    controller = _make_controller()
    controller._get_runtime_sparql_endpoint = lambda: "http://127.0.0.1:3001/kb/sparql"

    def _fake_probe(endpoint: str, sparql: str, timeout_s: float) -> list[str]:
        if "?answer fb:biology.breed_origin.breeds_originating_here ?anchor ." in sparql:
            return []
        if "?anchor fb:biology.breed_origin.breeds_originating_here ?answer ." in sparql:
            return ["http://rdf.freebase.com/ns/m.077qn"]
        return []

    controller._run_probe_sparql_query = _fake_probe

    resolved_ids = controller._probe_anchor_entity_ids(
        anchor_name="Serbia",
        relation="biology.breed_origin.breeds_originating_here",
        anchor_position="object",
        timeout_s=1.0,
    )

    assert resolved_ids == ["m.077qn"]


def test_anchor_existence_probes_map_multi_anchor_constraint_paths() -> None:
    controller = _make_controller()
    controller._probe_entity_name_count = lambda alias, timeout_s=2.5: 1
    probed_paths: list[tuple[str, str, str]] = []

    def _fake_path_probe(
        alias: str,
        relation: str,
        *,
        anchor_position: str = "subject",
        timeout_s: float = 2.5,
    ) -> int:
        probed_paths.append((alias, relation, anchor_position))
        return 3

    controller._probe_anchor_path_count = _fake_path_probe

    results = controller._run_anchor_existence_probes(
        query_plan={
            "anchored_entities": [
                {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                {
                    "surface": "Enalaprilat",
                    "chosen_alias": "Enalaprilat",
                    "role": "anchor_b",
                },
            ],
            "relation_paths": [
                {
                    "relation": "medicine.drug_formulation.formulation_of",
                    "direction": "forward",
                    "from": "formulation",
                    "to": "drug_or_ingredient",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "medicine.drug_formulation.active_ingredients",
                    "direction": "forward",
                    "from": "formulation",
                    "to": "active_ingredient",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "medicine.drug_formulation.dosage_form",
                    "direction": "forward",
                    "from": "formulation",
                    "to": "dosage_form",
                    "from_role": "shared_answer",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
            ],
        },
        probe_paths=True,
        timeout_s=1.0,
    )

    assert probed_paths == [
        ("Naloxone", "medicine.drug_formulation.formulation_of", "object"),
        ("Enalaprilat", "medicine.drug_formulation.active_ingredients", "object"),
    ]


def test_anchor_existence_probes_capture_unique_resolved_entity_id_from_live_path() -> None:
    controller = _make_controller()
    controller._probe_entity_name_count = lambda alias, timeout_s=2.5: 3
    controller._probe_anchor_path_count_with_position_fallback = (
        lambda alias, relation, anchor_position="subject", timeout_s=2.5: (4, "object")
    )
    controller._probe_anchor_entity_ids_with_position_fallback = (
        lambda *, anchor_name, relation=None, anchor_position="subject", timeout_s=2.5: (
            ["m.01c44b"],
            "object",
        )
        if anchor_name == "Southern Min"
        and relation == "language.language_dialect.language"
        else ([], anchor_position)
    )

    results = controller._run_anchor_existence_probes(
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Southern Min",
                    "chosen_alias": "Southern Min",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "language.language_dialect.language",
                    "direction": "reverse",
                    "from": "dialect",
                    "to": "Southern Min",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "curated",
                }
            ],
        },
        probe_paths=True,
        timeout_s=1.0,
    )

    assert len(results) == 1
    assert results[0].path_count == 4
    assert results[0].resolved_entity_id == "m.01c44b"
    assert results[0].relation_probed == "language.language_dialect.language"
    assert results[0].anchor_position == "object"


def test_anchor_existence_probes_use_surface_when_chosen_alias_is_mid() -> None:
    controller = _make_controller()
    probed_aliases: list[str] = []

    def _fake_probe_entity_name_count(alias: str, timeout_s: float = 2.5) -> int:
        probed_aliases.append(alias)
        return 7

    controller._probe_entity_name_count = _fake_probe_entity_name_count
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=2.5: 3
    )
    controller._probe_anchor_entity_ids = (
        lambda *, anchor_name, relation=None, anchor_position="subject", timeout_s=2.5: []
    )

    results = controller._run_anchor_existence_probes(
        query_plan={
            "anchored_entities": [
                {
                    "surface": "Count Basie Orchestra",
                    "chosen_alias": "m.01r4szq",
                    "role": "anchor_a",
                }
            ],
            "relation_paths": [
                {
                    "relation": "music.artist.album",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "shared_answer",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                }
            ],
        },
        probe_paths=True,
        timeout_s=1.0,
    )

    assert probed_aliases == []
    assert results[0].anchor_name == "Count Basie Orchestra"
    assert results[0].entity_count == 1
    assert results[0].resolved_entity_id == "m.01r4szq"


def test_anchor_existence_probes_use_resolved_entity_id_for_count_chain_path_probe() -> None:
    controller = _make_controller()
    controller._probe_entity_name_count = (
        lambda alias, timeout_s=2.5: (_ for _ in ()).throw(
            AssertionError("surface-name entity probe should be skipped when resolved_entity_id exists")
        )
    )

    captured_sparql: list[str] = []

    def _fake_run_probe_sparql_query(endpoint: str, sparql: str, timeout_s: float = 2.5) -> list[str]:
        captured_sparql.append(sparql)
        if "COUNT(DISTINCT ?answer)" in sparql:
            return ["1"]
        if "SELECT DISTINCT ?anchor" in sparql:
            return ["fb:m.01d30f"]
        return []

    controller._run_probe_sparql_query = _fake_run_probe_sparql_query

    results = controller._run_anchor_existence_probes(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "count_set_variable": "candidate_chars",
            "anchored_entities": [
                {
                    "surface": "educators",
                    "chosen_alias": "Teacher",
                    "resolved_entity_id": "m.01d30f",
                    "role": "constraint_value",
                },
                {
                    "surface": "Neyaphem",
                    "chosen_alias": "m.09tc50",
                    "resolved_entity_id": "m.09tc50",
                    "role": "anchor",
                },
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.character_species.characters_of_this_species",
                    "direction": "reverse",
                    "from": "candidate_chars",
                    "to": "anchor",
                    "from_role": "count_set",
                    "to_role": "anchor",
                },
                {
                    "relation": "fictional_universe.fictional_character.occupation",
                    "direction": "forward",
                    "from": "candidate_chars",
                    "to": "constraint_value",
                    "from_role": "count_set",
                    "to_role": "constraint_value",
                },
            ],
        },
        probe_paths=True,
        timeout_s=1.0,
    )

    assert len(results) == 2
    assert results[0].anchor_name == "Teacher"
    assert results[0].entity_count == 1
    assert results[0].path_count == 1
    assert results[0].resolved_entity_id == "m.01d30f"
    assert any("VALUES ?anchor { fb:m.01d30f }" in sparql for sparql in captured_sparql)


def test_anchor_existence_probes_can_reuse_generic_constraint_path_for_multiple_anchors() -> None:
    controller = _make_controller()
    probed_relations: list[tuple[str, str]] = []

    controller._probe_entity_name_count = lambda alias, timeout_s=2.5: 1

    def _fake_probe_anchor_path_count(alias: str, relation: str, anchor_position="subject", timeout_s=2.5) -> int:
        probed_relations.append((alias, relation))
        return 2

    controller._probe_anchor_path_count = _fake_probe_anchor_path_count
    controller._probe_anchor_entity_ids = (
        lambda *, anchor_name, relation=None, anchor_position="subject", timeout_s=2.5: []
    )

    results = controller._run_anchor_existence_probes(
        query_plan={
            "anchored_entities": [
                {"surface": "Goat", "chosen_alias": "Goat", "role": "anchor_a"},
                {"surface": "cows", "chosen_alias": "Cattle", "role": "anchor_b"},
            ],
            "join_structure": {
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "shared_answer",
                        "notes": "cheese must list Goat as a milk source (source_of_milk = Goat)",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "shared_answer",
                        "notes": "cheese must list cows as a milk source (source_of_milk = cows)",
                    },
                ]
            },
            "relation_paths": [
                {
                    "relation": "food.cheese.source_of_milk",
                    "direction": "forward",
                    "from": "cheese",
                    "to": "milk_source",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "food.cheese.texture",
                    "direction": "forward",
                    "from": "cheese",
                    "to": "texture_value",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
        },
        probe_paths=True,
        timeout_s=1.0,
    )

    assert probed_relations == [
        ("Goat", "food.cheese.source_of_milk"),
        ("Cattle", "food.cheese.source_of_milk"),
    ]
    assert [result.relation_probed for result in results] == [
        "food.cheese.source_of_milk",
        "food.cheese.source_of_milk",
    ]


def test_dynamic_grounding_repair_augments_candidates_on_path_failure() -> None:
    controller = _make_controller()
    controller._build_pal_grounding_card = (
        lambda task_question, relation_grounding=None, alias_overrides=None: str(
            [candidate["relation"] for candidate in relation_grounding or []]
        )
    )
    controller._probe_dynamic_relation_candidates = (
        lambda entities, answer_target_phrase, domain_hints, question_text="": [
            {
                "relation": "fictional_universe.fictional_setting.contains",
                "direction": "forward",
                "from": entities[0],
                "to": "?target",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ]
    )
    controller._probe_dynamic_relation_candidates_for_anchors = (
        lambda anchored_entities, answer_target_phrase, domain_hints, question_text="", probe_timeout_s=5.0, max_anchors=2: [
            {
                "relation": "fictional_universe.fictional_setting.contains",
                "direction": "forward",
                "from": anchored_entities[0]["chosen_alias"],
                "to": "?target",
                "from_role": anchored_entities[0]["role"],
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ]
    )
    controller._normalize_grounded_relation_candidates = (
        lambda relation_candidates, query_shape, answer_mode, answer_target_phrase, entities: relation_candidates
    )

    grounding_card, merged_candidates, feedback = controller._augment_grounding_with_dynamic_probe_on_path_failure(
        task_question="Question: what number of different species are in the fictional world of seventh sphere?, Entities: ['Seventh sphere']",
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Seventh sphere", "chosen_alias": "Seventh sphere", "role": "anchor"}
            ],
        },
        relation_grounding=[
            {
                "relation": "fictional_universe.fictional_universe.species",
                "direction": "forward",
                "from": "fictional_world",
                "to": "species",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "curated",
            }
        ],
        anchor_probe_results=[
            type(
                "Probe",
                (),
                {
                    "entity_count": 1,
                    "path_count": 0,
                },
            )(),
        ],
    )

    assert len(merged_candidates) == 2
    assert any(
        candidate["relation"] == "fictional_universe.fictional_setting.contains"
        for candidate in merged_candidates
    )
    assert "fictional_universe.fictional_setting.contains" in grounding_card
    assert feedback


def test_dynamic_probe_candidates_preserve_specific_anchor_roles() -> None:
    controller = _make_controller()
    candidates = controller._build_dynamic_candidate_list(
        outgoing_iris=["http://rdf.freebase.com/ns/medicine.drug_ingredient.active_moiety_of_drug"],
        incoming_iris=["http://rdf.freebase.com/ns/medicine.drug.active_moieties"],
        anchor_entity="Enalaprilat",
        anchor_role="anchor_b",
        answer_target_phrase="dosage form",
        domain_hints=["medicine", "drug"],
        question_text="what dosage form exists for drugs with active ingredient enalaprilat",
    )

    assert candidates[0]["from_role"] == "anchor_b"
    assert candidates[1]["to_role"] == "anchor_b"


def test_shared_anchor_relation_families_are_prioritized() -> None:
    controller = _make_controller()
    prioritized, shared_relations = controller._prioritize_shared_anchor_relation_families(
        [
            {
                "relation": "medicine.drug_formulation.formulation_of",
                "direction": "reverse",
                "from": "?source",
                "to": "Naloxone",
                "from_role": "candidate_set",
                "to_role": "anchor_a",
            },
            {
                "relation": "medicine.drug_ingredient.active_moiety_of_drug",
                "direction": "forward",
                "from": "Naloxone",
                "to": "?target",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
            },
            {
                "relation": "medicine.drug_ingredient.active_moiety_of_drug",
                "direction": "forward",
                "from": "Enalaprilat",
                "to": "?target",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
            },
        ]
    )

    assert shared_relations == ["medicine.drug_ingredient.active_moiety_of_drug"]
    assert prioritized[0]["relation"] == "medicine.drug_ingredient.active_moiety_of_drug"
    assert prioritized[1]["relation"] == "medicine.drug_ingredient.active_moiety_of_drug"


def test_shared_anchor_relation_families_require_distinct_anchor_roles() -> None:
    controller = _make_controller()
    prioritized, shared_relations = controller._prioritize_shared_anchor_relation_families(
        [
            {
                "relation": "common.notable_for.notable_object",
                "direction": "reverse",
                "from": "?source",
                "to": "Talk radio",
                "from_role": "candidate_set",
                "to_role": "anchor",
            }
        ]
    )

    assert shared_relations == []
    assert prioritized[0]["relation"] == "common.notable_for.notable_object"


def test_multi_anchor_dynamic_fallback_probes_all_anchors() -> None:
    controller = _make_controller()
    controller._build_grounded_relation_candidates = lambda task_question: []
    controller._probe_dynamic_relation_candidates = lambda **kwargs: pytest.fail(
        "should use anchor-aware dynamic probing for multi-anchor fallback"
    )
    controller._probe_dynamic_relation_candidates_for_anchors = (
        lambda anchored_entities, answer_target_phrase, domain_hints, question_text="", probe_timeout_s=5.0, max_anchors=2: [
            {
                "relation": "broadcast.genre.content",
                "direction": "forward",
                "from": "Talk radio",
                "to": "?target",
                "from_role": anchored_entities[0]["role"],
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
            {
                "relation": "broadcast.radio_program.producer",
                "direction": "forward",
                "from": "Weekend Edition Sunday",
                "to": "?target",
                "from_role": anchored_entities[1]["role"],
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
        ]
    )

    candidates = controller._build_grounded_relation_candidates_with_dynamic_fallback(
        task_question="Question: how much content about talk radio is produced by the person that produces weekend edition sunday?, Entities: ['Talk radio', 'Weekend Edition Sunday']",
        entities=["Talk radio", "Weekend Edition Sunday"],
        answer_target_phrase="content",
        domain_hints=["broadcast"],
    )

    assert any(candidate["from_role"] == "anchor_a" for candidate in candidates)
    assert any(candidate["from_role"] == "anchor_b" for candidate in candidates)


def test_structural_repair_augments_join_failure_with_anchor_dynamic_candidates() -> None:
    controller = _make_controller()
    controller._build_pal_grounding_card = (
        lambda task_question, relation_grounding=None, alias_overrides=None: str(
            [candidate["relation"] for candidate in relation_grounding or []]
        )
    )
    controller._probe_dynamic_relation_candidates_for_anchors = (
        lambda anchored_entities, answer_target_phrase, domain_hints, question_text="", probe_timeout_s=5.0, max_anchors=2: [
            {
                "relation": "medicine.drug_ingredient.active_moiety_of_drug",
                "direction": "forward",
                "from": "Naloxone",
                "to": "?target",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
            {
                "relation": "medicine.drug_ingredient.active_moiety_of_drug",
                "direction": "forward",
                "from": "Enalaprilat",
                "to": "?target",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
        ]
    )

    grounding_card, merged_candidates, feedback = controller._augment_grounding_for_structural_repair(
        task_question="Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?, Entities: ['Naloxone', 'Enalaprilat']",
        query_plan={
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
            ],
            "shared_answer_variable": "formulation",
            "relation_paths": [
                {
                    "relation": "medicine.drug_formulation.formulation_of",
                    "direction": "forward",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                },
                {
                    "relation": "medicine.drug_formulation.active_ingredients",
                    "direction": "forward",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                },
            ],
        },
        relation_grounding=[
            {
                "relation": "medicine.drug_formulation.formulation_of",
                "direction": "forward",
                "from": "formulation",
                "to": "drug_or_ingredient",
                "from_role": "shared_answer",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            }
        ],
        anchor_probe_results=[
            type("Probe", (), {"entity_count": 1, "path_count": 7})(),
            type("Probe", (), {"entity_count": 1, "path_count": 2})(),
        ],
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_GROUNDED_EMPTY,
            reasons=["anchor_paths_live_but_join_overlap_empty"],
        ),
    )

    assert any(
        candidate["relation"] == "medicine.drug_ingredient.active_moiety_of_drug"
        and candidate["from_role"] == "anchor_a"
        for candidate in merged_candidates
    )
    assert any(
        candidate["relation"] == "medicine.drug_ingredient.active_moiety_of_drug"
        and candidate["from_role"] == "anchor_b"
        for candidate in merged_candidates
    )
    assert any("prefer_asymmetric_bridge_scaffold" in item for item in feedback)
    assert any("anchor_dynamic_priority:Naloxone=" in item for item in feedback)
    assert any("anchor_dynamic_priority:Enalaprilat=" in item for item in feedback)
    assert "medicine.drug_ingredient.active_moiety_of_drug" in grounding_card


def test_grounding_validation_rejects_wrong_multi_anchor_surface_reuse() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "common.notable_for.notable_object",
                "direction": "reverse",
                "from": "?source",
                "to": "Talk radio",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            }
        ],
        query_shape="multi_anchor_intersection",
        answer_mode="count",
        answer_target_phrase="content",
        entities=["Talk radio", "Weekend Edition Sunday"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Talk radio", "chosen_alias": "Talk radio", "role": "anchor_a"},
                {
                    "surface": "Weekend Edition Sunday",
                    "chosen_alias": "Weekend Edition Sunday",
                    "role": "anchor_b",
                },
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "?content",
            "candidate_set_variable": "?content",
            "count_set_variable": "?content",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "?content", "notes": "genre"},
                    {"anchor_role": "anchor_b", "constrains_variable": "?content", "notes": "producer"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "common.notable_for.notable_object",
                    "direction": "reverse",
                    "from": "?producer",
                    "to": "WeekendEditionSunday_anchor",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                    "reason": "intentionally reuse the Talk radio dynamic relation for the wrong anchor",
                }
            ],
            "projection": ["?count"],
            "allow_exploratory_predicates": False,
            "strategy": "invalid multi-anchor surface reuse",
            "plan_rationale": ["the relation was only grounded on anchor_a and must not validate for anchor_b"],
        }
    )

    errors = controller._validate_query_plan_grounding(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    )
    assert any(
        error.startswith("relation_shape_not_grounded:common.notable_for.notable_object")
        for error in errors
    )


def test_generate_plan_rejects_reused_dead_scaffold_signature() -> None:
    controller = _make_controller()
    controller._run_text_prompt = lambda system_prompt, user_prompt: """
    {
      "answer_type": "entity",
      "answer_mode": "entity",
      "query_shape": "multi_anchor_intersection",
      "anchored_entities": [
        {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
        {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"}
      ],
      "normalized_aliases": [],
      "shared_answer_variable": "formulation",
      "candidate_set_variable": "formulation",
      "count_set_variable": "",
      "ordering_attribute": {"direction": "forward"},
      "ordering_direction": "none",
      "join_structure": {
        "type": "intersection",
        "anchor_constraints": [
          {"anchor_role": "anchor_a", "constrains_variable": "formulation", "notes": "a"},
          {"anchor_role": "anchor_b", "constrains_variable": "formulation", "notes": "b"}
        ]
      },
      "relation_paths": [
        {
          "relation": "medicine.drug_formulation.formulation_of",
          "direction": "forward",
          "from": "formulation",
          "to": "drug_or_ingredient",
          "from_role": "shared_answer",
          "to_role": "constraint_value",
          "grounding_source": "curated",
          "reason": "bind Naloxone"
        },
        {
          "relation": "medicine.drug_formulation.active_ingredients",
          "direction": "forward",
          "from": "formulation",
          "to": "active_ingredient",
          "from_role": "shared_answer",
          "to_role": "constraint_value",
          "grounding_source": "curated",
          "reason": "bind Enalaprilat"
        }
      ],
      "projection": ["dosage_form"],
      "allow_exploratory_predicates": false,
      "strategy": "repeat dead scaffold",
      "plan_rationale": ["test"]
    }
    """
    controller._capture_pal_query_plan_artifact = (
        lambda generated_tool_name, query_plan: pathlib.Path("/tmp/fake.plan.json")
    )
    controller._validate_query_plan_grounding = lambda query_plan, relation_grounding: []

    with pytest.raises(ValueError, match="pal_query_plan_invalid:query_plan_reused_dead_scaffold_signature"):
        controller._generate_pal_query_plan(
            task_question="Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?, Entities: ['Naloxone', 'Enalaprilat']",
            grounding_card="PAL grounding hints:",
            relation_grounding=[],
            generated_tool_name="pal_sparql_query_tool_test",
            extra_plan_feedback=[
                "dead_scaffold_signature:multi_anchor_intersection|formulation|medicine.drug_formulation.active_ingredients|medicine.drug_formulation.formulation_of"
            ],
        )


def test_dead_relation_suppression_removes_probed_empty_relation() -> None:
    controller = _make_controller()
    controller._build_pal_grounding_card = (
        lambda task_question, relation_grounding=None, alias_overrides=None: str(
            [candidate["relation"] for candidate in relation_grounding or []]
        )
    )

    grounding_card, filtered_candidates, feedback = controller._suppress_dead_grounded_relations(
        task_question="Question: what number of different species are in the fictional world of seventh sphere?, Entities: ['Seventh sphere']",
        query_plan={
            "anchored_entities": [
                {"surface": "Seventh sphere", "chosen_alias": "Seventh sphere", "role": "anchor"}
            ]
        },
        relation_grounding=[
            {
                "relation": "fictional_universe.fictional_universe.species",
                "direction": "forward",
                "from": "fictional_world",
                "to": "species",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
            {
                "relation": "fictional_universe.fictional_setting.universe",
                "direction": "forward",
                "from": "setting",
                "to": "universe",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        anchor_probe_results=[
            type(
                "Probe",
                (),
                {
                    "entity_count": 1,
                    "path_count": 0,
                    "relation_probed": "fictional_universe.fictional_universe.species",
                    "anchor_position": "subject",
                },
            )(),
        ],
    )

    assert [candidate["relation"] for candidate in filtered_candidates] == [
        "fictional_universe.fictional_setting.universe"
    ]
    assert "fictional_universe.fictional_setting.universe" in grounding_card
    assert any("dead_relation_suppressed" in item for item in feedback)


def test_dead_relation_suppression_scopes_feedback_to_anchor_path_instance() -> None:
    controller = _make_controller()

    _, filtered_candidates, feedback = controller._suppress_dead_grounded_relations(
        task_question="Question: what is the sensor type of a digital camera that has the color filter array type of bayer and iso settings of 120?, Entities: ['bayer', '120']",
        query_plan={
            "anchored_entities": [
                {"surface": "bayer", "chosen_alias": "Bayer filter", "role": "anchor_a"},
                {"surface": "120", "chosen_alias": "120", "role": "anchor_b"},
            ]
        },
        relation_grounding=[
            {
                "relation": "digicams.digital_camera.color_filter_array_type",
                "direction": "forward",
                "from": "Bayer filter",
                "to": "camera",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "digicams.digital_camera.color_filter_array_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "digicams.digital_camera.iso_setting",
                "direction": "forward",
                "from": "candidate_set",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Bayer filter",
                entity_count=1,
                path_count=0,
                relation_probed="digicams.digital_camera.color_filter_array_type",
                anchor_position="subject",
                resolved_entity_id="m.02r8js",
            )
        ],
    )

    remaining_pairs = [
        (candidate["relation"], candidate["from_role"], candidate["to_role"])
        for candidate in filtered_candidates
    ]
    assert (
        "digicams.digital_camera.color_filter_array_type",
        "candidate_set",
        "constraint_value",
    ) in remaining_pairs
    assert (
        "digicams.digital_camera.color_filter_array_type",
        "anchor_a",
        "candidate_set",
    ) not in remaining_pairs
    assert any(
        item.startswith(
            "dead_relation_suppressed:digicams.digital_camera.color_filter_array_type["
        )
        for item in feedback
    )
    assert (
        "relation_still_available_in_other_roles:digicams.digital_camera.color_filter_array_type"
        in feedback
    )


def test_plausibility_rejects_weak_count_scalar_structure() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Unsteadiness", "chosen_alias": "Unsteadiness", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "medicine.symptom.side_effect_of",
                    "direction": "forward",
                    "from": "?symptom",
                    "to": "?treatment",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count_over_direct_relation",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?treatment) AS ?count_treatments) WHERE { "
            "?symptom fb:medicine.symptom.side_effect_of ?treatment . } LIMIT 1"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count_treatments": {"type": "literal", "value": "1"}}
                ]
            }
        },
        entities=["Unsteadiness"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_set_variable_missing" in verdict.reasons


def test_plausibility_uses_planned_alias_for_anchor_literal_check() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Goat", "chosen_alias": "Goat", "role": "anchor_a"},
                {"surface": "cows", "chosen_alias": "Cattle", "role": "anchor_b"},
            ],
            "shared_answer_variable": "?cheese",
            "candidate_set_variable": "?cheese",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "?cheese", "notes": "ok"},
                    {"anchor_role": "anchor_b", "constrains_variable": "?cheese", "notes": "ok"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "food.cheese_milk_source.cheeses",
                    "direction": "reverse",
                    "from": "?milk_source",
                    "to": "?cheese",
                    "from_role": "constraint_value",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "multi_anchor_intersection",
        },
        query_text='SELECT ?cheese WHERE { "Goat" "Cattle" } LIMIT 50',
        result_dict={"results": {"bindings": [{"cheese": {"type": "uri", "value": "m.test"}}]}},
        entities=["Goat", "cows"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_plausibility_allows_intermediate_constraint_variables_when_grounded() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Goat", "chosen_alias": "Goat", "role": "anchor_a"},
                {"surface": "cows", "chosen_alias": "Cattle", "role": "anchor_b"},
                {"surface": "semi-firm", "chosen_alias": "semi-firm", "role": "anchor_value"},
            ],
            "shared_answer_variable": "?cheese",
            "candidate_set_variable": "?cheese",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "?milk_source", "notes": "milk-source endpoint"},
                    {"anchor_role": "anchor_b", "constrains_variable": "?milk_source", "notes": "milk-source endpoint"},
                    {"anchor_role": "anchor_value", "constrains_variable": "?texture_value", "notes": "texture endpoint"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "food.cheese_texture.cheeses",
                    "direction": "reverse",
                    "from": "?texture_value",
                    "to": "?cheese",
                    "from_role": "constraint_value",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
                {
                    "relation": "food.cheese_milk_source.cheeses",
                    "direction": "reverse",
                    "from": "?milk_source",
                    "to": "?cheese",
                    "from_role": "constraint_value",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "multi_anchor_intersection",
        },
        query_text='SELECT ?cheese WHERE { "Goat" "Cattle" } LIMIT 50',
        result_dict={"results": {"bindings": [{"cheese": {"type": "uri", "value": "m.test"}}]}},
        entities=["Goat", "cows"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_plausibility_does_not_treat_type_filter_as_multi_anchor_join() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Stafy", "chosen_alias": "Stafy", "role": "anchor"},
                {"surface": "book character", "chosen_alias": "book character", "role": "anchor_value"},
            ],
            "shared_answer_variable": "rank",
            "candidate_set_variable": "candidate_character",
            "count_set_variable": "count",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "rank", "notes": "anchor to rank"},
                    {"anchor_role": "shared_answer", "constrains_variable": "candidate_character", "notes": "rank to candidates"},
                    {"anchor_role": "anchor_value", "constrains_variable": "candidate_character", "notes": "type filter"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.rank",
                    "direction": "forward",
                    "from": "Stafy",
                    "to": "rank",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "fictional_universe.character_rank.characters_of_this_rank",
                    "direction": "reverse",
                    "from": "candidate_character",
                    "to": "rank",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "single anchor count with type filter",
        },
        query_text='SELECT (COUNT(DISTINCT ?candidate_character) AS ?count) WHERE { "Stafy" "book character" } LIMIT 50',
        result_dict={"results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]}},
        entities=["Stafy", "book character"],
        anchor_probe_results=[],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_plausibility_detects_bad_superlative_structure() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Atkinson Candy Company", "chosen_alias": "Atkinson Candy Company", "role": "anchor"}
            ],
            "candidate_set_variable": "?product",
            "ordering_attribute": {
                "relation": "product.product.introduction_date",
                "direction": "forward",
                "source_variable": "?product",
                "attribute_variable": "?introduced",
            },
            "ordering_direction": "max",
            "relation_paths": [
                {
                    "relation": "product.product.manufacturer",
                    "direction": "forward",
                    "from": "?product",
                    "to": "?manufacturer",
                    "from_role": "candidate_set",
                    "to_role": "anchor_value",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "superlative_chain",
        },
        query_text="SELECT ?product WHERE { ?product fb:product.product.manufacturer ?manufacturer . } LIMIT 50",
        result_dict={"results": {"bindings": []}},
        entities=["Atkinson Candy Company"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_SUPERLATIVE
    assert "ordering_attribute_path_missing" in verdict.reasons


def test_plausibility_repairs_grounded_superlative_empty_result() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Neurocide", "chosen_alias": "Neurocide", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "music.recording.length",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "recording_length",
            },
            "ordering_direction": "max",
            "relation_paths": [
                {
                    "relation": "music.artist.album",
                    "direction": "forward",
                    "from": "artist",
                    "to": "album",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "music.recording.length",
                    "direction": "forward",
                    "from": "recording",
                    "to": "length",
                    "from_role": "answer",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "grounded superlative chain",
        },
        query_text="SELECT ?answer WHERE { ?artist fb:music.artist.album ?album . } ORDER BY DESC(?recording_length) LIMIT 1",
        result_dict={"results": {"bindings": []}},
        entities=["Neurocide"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_SUPERLATIVE
    assert "grounded_superlative_empty_result" in verdict.reasons


def test_plausibility_accepts_semantically_specific_exploratory_superlative_ordering() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Juicer", "chosen_alias": "m.01prn0", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "food.recipe.preparation_time",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "prep_time",
            },
            "ordering_direction": "max",
            "relation_paths": [
                {
                    "relation": "food.culinary_tool.used_in_recipes",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "food.recipe.preparation_time",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "prep_time",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "exploratory",
                },
            ],
            "allow_exploratory_predicates": True,
            "strategy": "superlative chain over recipes by preparation time",
        },
        query_text=(
            "SELECT DISTINCT ?candidate_set ?candidate_set_name WHERE { "
            "fb:m.01prn0 fb:food.culinary_tool.used_in_recipes ?candidate_set . "
            "?candidate_set fb:food.recipe.preparation_time ?prep_time . "
            "?candidate_set fb:type.object.name ?candidate_set_name . "
            "} ORDER BY DESC(?prep_time) LIMIT 1"
        ),
        result_dict={
            "head": {"vars": ["candidate_set", "candidate_set_name"]},
            "results": {
                "bindings": [
                    {
                        "candidate_set": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.recipe"},
                        "candidate_set_name": {"type": "literal", "value": "Recipe"},
                    }
                ]
            },
        },
        entities=["Juicer"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "ordering_attribute_path_exploratory" not in verdict.reasons


def test_plausibility_rejects_generic_exploratory_superlative_ordering() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Juicer", "chosen_alias": "m.01prn0", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "type.object.name",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "name_attr",
            },
            "ordering_direction": "max",
            "relation_paths": [
                {
                    "relation": "food.culinary_tool.used_in_recipes",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.object.name",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "name_attr",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "exploratory",
                },
            ],
            "allow_exploratory_predicates": True,
            "strategy": "superlative chain with generic name ordering",
        },
        query_text=(
            "SELECT DISTINCT ?candidate_set ?candidate_set_name WHERE { "
            "fb:m.01prn0 fb:food.culinary_tool.used_in_recipes ?candidate_set . "
            "?candidate_set fb:type.object.name ?name_attr . "
            "} ORDER BY DESC(?name_attr) LIMIT 1"
        ),
        result_dict={"results": {"bindings": []}},
        entities=["Juicer"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_SUPERLATIVE
    assert "ordering_attribute_path_exploratory" in verdict.reasons


def test_plausibility_accepts_scalar_aggregate_superlative_without_order_by() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "literal",
            "answer_type": "literal",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Flare star", "chosen_alias": "m.05dt2l", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "astronomy.star.temperature_k",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "temp",
            },
            "ordering_direction": "min",
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object_category.objects",
                    "direction": "reverse",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "astronomy.star.temperature_k",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "temp",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "exploratory",
                },
            ],
            "allow_exploratory_predicates": True,
            "strategy": "superlative chain over stars by temperature",
        },
        query_text=(
            "SELECT (MIN(?temp) AS ?answer) WHERE { "
            "?candidate_set fb:astronomy.celestial_object_category.objects fb:m.05dt2l . "
            "?candidate_set fb:astronomy.star.temperature_k ?temp . "
            "}"
        ),
        result_dict={
            "head": {"vars": ["answer"]},
            "results": {
                "bindings": [
                    {"answer": {"type": "literal", "value": "2100"}}
                ]
            },
        },
        entities=["Flare star"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "ordering_attribute_path_exploratory" not in verdict.reasons
    assert "superlative_strategy_missing_ORDER_BY" not in verdict.reasons


def test_validate_pal_execution_rejects_multi_entity_result_for_singleton_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "answer_target_phrase": "last day",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "Gregorian calendar",
                    "chosen_alias": "m.037d3",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "time.day_of_week.calendar_system",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "anchor",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT ?answer ?name WHERE { "
            "{ ?answer fb:time.day_of_week.calendar_system fb:m.037d3 . } "
            "UNION { fb:m.037d3 fb:time.day_of_week.calendar_system ?answer . } "
            "OPTIONAL { ?answer fb:type.object.name ?name . } "
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["answer", "name"]},
            "results": {
                "bindings": [
                    {"answer": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.0f7yn"}},
                    {"answer": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.0f7_4"}},
                    {"answer": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.0f7zn"}},
                ]
            },
        },
        entities=["Gregorian calendar"],
    )

    assert verdict.verdict == "repairable_weak_grounding"
    assert "entity_result_multi_binding_for_singleton_target:last day" in verdict.reasons


def test_validate_pal_execution_allows_multi_entity_result_for_non_singleton_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "answer_target_phrase": "red wine",
            "query_shape": "shared_type_intersection",
            "anchored_entities": [
                {"surface": "Dutcher Crossing Winery", "chosen_alias": "m.03zznbs", "role": "anchor_a"},
                {"surface": "Red Wine", "chosen_alias": "m.02wsb20", "role": "anchor_b"},
            ],
            "relation_paths": [
                {
                    "relation": "wine.wine.winery",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "anchor_a",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "wine.wine.color",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "anchor_b",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "candidate_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT DISTINCT ?candidate_set WHERE { "
            "?candidate_set fb:wine.wine.winery fb:m.03zznbs . "
            "?candidate_set fb:wine.wine.color fb:m.02wsb20 . }"
        ),
        result_dict={
            "head": {"vars": ["candidate_set"]},
            "results": {
                "bindings": [
                    {"candidate_set": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.03_1hzq"}},
                    {"candidate_set": {"type": "uri", "value": "http://rdf.freebase.com/ns/m.03zzrns"}},
                ]
            },
        },
        entities=["Dutcher Crossing Winery", "Red Wine"],
    )

    assert "entity_result_multi_binding_for_singleton_target:red wine" not in verdict.reasons


def test_plausibility_accepts_grounded_count_plan_with_structured_fields() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Unsteadiness", "chosen_alias": "Unsteadiness", "role": "anchor"}
            ],
            "candidate_set_variable": "?treatment",
            "count_set_variable": "?treatment",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "?symptom", "notes": "bind the symptom anchor"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "medicine.symptom.side_effect_of",
                    "direction": "forward",
                    "from": "?symptom",
                    "to": "?treatment",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count_over_direct_relation",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?treatment) AS ?count_treatments) WHERE { "
            "?symptom fb:medicine.symptom.side_effect_of ?treatment . } LIMIT 1"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count_treatments": {"type": "literal", "value": "1"}}
                ]
            }
        },
        entities=["Unsteadiness"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_plausibility_does_not_require_exact_count_set_variable_name_in_query() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Hentai", "chosen_alias": "Hentai", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "candidate_set", "notes": "bind the child-genre set"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "media_common.media_genre.child_genres",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "answer",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count child genres of the anchor genre",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?anchor fb:media_common.media_genre.child_genres ?candidate_set . } LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "2"}}
                ]
            }
        },
        entities=["Hentai"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_plausibility_rejects_count_query_that_counts_wrong_variable() -> None:
    query_plan = {
        "answer_mode": "count",
        "answer_type": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {"surface": "Brazil", "chosen_alias": "Brazil", "role": "anchor_a"},
            {
                "surface": "Manchester Terrier",
                "chosen_alias": "Manchester Terrier",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "temperament",
        "candidate_set_variable": "breed",
        "count_set_variable": "temperament",
        "ordering_attribute": {"direction": "forward"},
        "ordering_direction": "none",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {"anchor_role": "anchor_a", "constrains_variable": "breed", "notes": "country to breed"},
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "temperament",
                    "notes": "anchor temperament",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "biology.breed_origin.breeds_originating_here",
                "direction": "reverse",
                "from": "country",
                "to": "breed",
                "from_role": "constraint_value",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "breed",
                "to": "temperament",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "Manchester Terrier",
                "to": "temperament",
                "from_role": "anchor_b",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "allow_exploratory_predicates": False,
        "strategy": "count shared temperament values",
        "plan_rationale": [],
    }
    query_text = """
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?breed) AS ?count) WHERE {
      ?country fb:type.object.name ?country_name .
      FILTER(LCASE(STR(?country_name)) = "brazil") .
      ?country fb:biology.breed_origin.breeds_originating_here ?breed .
      ?anchor fb:type.object.name ?anchor_name .
      FILTER(LCASE(STR(?anchor_name)) = "manchester terrier") .
      ?breed fb:biology.animal_breed.temperament ?temperament .
      ?anchor fb:biology.animal_breed.temperament ?temperament .
    } LIMIT 50
    """
    result_dict = {
        "head": {"vars": ["count"]},
        "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
    }

    verdict = validate_pal_execution(
        query_plan=query_plan,
        query_text=query_text,
        result_dict=result_dict,
        entities=["Brazil", "Manchester Terrier"],
        anchor_probe_results=None,
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_counts_wrong_variable:breed" in verdict.reasons


def test_superlative_grounding_normalizes_ordering_roles() -> None:
    controller = _make_controller()
    assert controller._looks_like_ordering_endpoint("candidate_set") is False
    assert controller._looks_like_ordering_endpoint("release_date") is True

    candidates = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "music.artist.album",
                "direction": "forward",
                "from": "artist",
                "to": "album",
                "support": "curated_music_predicate",
            },
            {
                "relation": "music.recording.length",
                "direction": "forward",
                "from": "recording",
                "to": "length",
                "support": "curated_music_predicate",
            },
        ],
        query_shape="superlative_chain",
        answer_mode="entity",
        answer_target_phrase="musical recording",
        entities=["Neurocide"],
    )

    assert candidates[0]["from_role"] == "anchor"
    assert candidates[0]["to_role"] == "candidate_set"
    assert candidates[1]["from_role"] == "candidate_set"
    assert candidates[1]["to_role"] == "ordering_attribute"


def test_normalize_pal_query_plan_coerces_ordered_entity_plan_to_superlative_chain() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "David Han", "chosen_alias": "David Han", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "shared_answer_variable": "answer",
            "ordering_attribute": {
                "relation": "music.recording.length",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "length",
            },
            "ordering_direction": "max",
            "relation_paths": [
                {
                    "relation": "music.engineer.tracks_engineered",
                    "direction": "forward",
                    "from": "David Han",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "music.recording.length",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "length",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                },
            ],
            "projection": ["candidate_set"],
        }
    )

    assert normalized["query_shape"] == "superlative_chain"


def test_normalize_pal_query_plan_promotes_terminal_ordering_value_in_chain() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "answer_target_phrase": "musical release",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Gavin Lurssen",
                    "chosen_alias": "m.01vzjzb",
                    "resolved_entity_id": "m.01vzjzb",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "shared_answer_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "music.release.release_date",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "release_date_node",
            },
            "ordering_direction": "max",
            "relation_paths": [
                {
                    "relation": "music.engineer.releases_engineered",
                    "direction": "forward",
                    "from": "Gavin Lurssen",
                    "to": "release",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "music.release.release_date",
                    "direction": "forward",
                    "from": "release",
                    "to": "release_date_node",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "freebase.valuenotation.has_value",
                    "direction": "forward",
                    "from": "release_date_node",
                    "to": "release_date_value",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "projection": ["candidate_set"],
            "allow_exploratory_predicates": True,
            "plan_rationale": ["Use the terminal release-date value for ordering."],
        }
    )

    assert normalized["ordering_attribute"]["attribute_variable"] == "release_date_value"
    assert normalized["relation_paths"][2]["from_role"] == "ordering_attribute"
    assert normalized["relation_paths"][2]["to_role"] == "ordering_attribute"


def test_normalize_pal_query_plan_coerces_shared_answer_literal_intersection_to_entity() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "literal",
            "answer_type": "literal",
            "answer_target_phrase": "temperament",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Maltese", "chosen_alias": "Maltese", "role": "anchor_a"},
                {"surface": "Papillon", "chosen_alias": "Papillon", "role": "anchor_b"},
            ],
            "shared_answer_variable": "shared_temperament",
            "relation_paths": [
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "shared_temperament",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "shared_temperament",
                    "from_role": "anchor_b",
                    "to_role": "shared_answer",
                },
            ],
            "projection": ["shared_temperament"],
        }
    )

    assert normalized["answer_mode"] == "entity"
    assert normalized["answer_type"] == "entity"


def test_normalize_pal_query_plan_keeps_literal_label_projection_literal() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "literal",
            "answer_type": "literal",
            "answer_target_phrase": "temperament",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Maltese", "chosen_alias": "Maltese", "role": "anchor_a"},
                {"surface": "Papillon", "chosen_alias": "Papillon", "role": "anchor_b"},
            ],
            "shared_answer_variable": "shared_temperament",
            "relation_paths": [
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "shared_temperament",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "shared_temperament",
                    "from_role": "anchor_b",
                    "to_role": "shared_answer",
                },
            ],
            "projection": ["shared_temperament_name"],
        }
    )

    assert normalized["answer_mode"] == "literal"
    assert normalized["answer_type"] == "literal"


def test_superlative_contract_coerces_anchor_start_and_answer_projection_roles() -> None:
    controller = _make_controller()
    anchored_entities = [
        {
            "surface": "David Han",
            "chosen_alias": "David Han",
            "role": "anchor",
        }
    ]

    anchor_path = controller._normalize_relation_contract_item(
        {
            "relation": "music.artist.track",
            "direction": "forward",
            "from": "artist",
            "to": "candidate_set",
            "from_role": "candidate_set",
            "to_role": "ordering_attribute",
            "grounding_source": "curated",
        },
        query_shape="superlative_chain",
        answer_mode="entity",
        answer_target_phrase="release",
        anchored_entities=anchored_entities,
        shared_answer_variable="release",
        candidate_set_variable="candidate_set",
        count_set_variable="",
        ordering_attribute={},
        allow_exploratory=False,
    )

    projection_path = controller._normalize_relation_contract_item(
        {
            "relation": "music.recording.releases",
            "direction": "forward",
            "from": "candidate_set",
            "to": "release",
            "from_role": "candidate_set",
            "to_role": "ordering_attribute",
            "grounding_source": "curated",
        },
        query_shape="superlative_chain",
        answer_mode="entity",
        answer_target_phrase="release",
        anchored_entities=anchored_entities,
        shared_answer_variable="release",
        candidate_set_variable="candidate_set",
        count_set_variable="",
        ordering_attribute={},
        allow_exploratory=False,
    )

    ordering_path = controller._normalize_relation_contract_item(
        {
            "relation": "music.recording.length",
            "direction": "forward",
            "from": "candidate_set",
            "to": "recording_length",
            "from_role": "candidate_set",
            "to_role": "ordering_attribute",
            "grounding_source": "curated",
        },
        query_shape="superlative_chain",
        answer_mode="entity",
        answer_target_phrase="release",
        anchored_entities=anchored_entities,
        shared_answer_variable="release",
        candidate_set_variable="candidate_set",
        count_set_variable="",
        ordering_attribute={},
        allow_exploratory=False,
    )

    assert anchor_path is not None
    assert projection_path is not None
    assert ordering_path is not None
    assert anchor_path["from_role"] == "anchor"
    assert anchor_path["to_role"] == "candidate_set"
    assert projection_path["from_role"] == "candidate_set"
    assert projection_path["to_role"] == "answer"
    assert ordering_path["from_role"] == "candidate_set"
    assert ordering_path["to_role"] == "ordering_attribute"


def test_superlative_literal_contract_coerces_category_endpoint_to_anchor() -> None:
    controller = _make_controller()
    anchored_entities = [
        {
            "surface": "Flare star",
            "chosen_alias": "Flare star",
            "role": "anchor",
        }
    ]

    anchor_path = controller._normalize_relation_contract_item(
        {
            "relation": "astronomy.celestial_object_category.objects",
            "direction": "reverse",
            "from": "category",
            "to": "candidate_set",
            "from_role": "answer",
            "to_role": "candidate_set",
            "grounding_source": "curated",
        },
        query_shape="superlative_chain",
        answer_mode="literal",
        answer_target_phrase="temperature",
        anchored_entities=anchored_entities,
        shared_answer_variable="answer",
        candidate_set_variable="candidate_set",
        count_set_variable="",
        ordering_attribute={
            "relation": "astronomy.star.temperature_k",
            "direction": "forward",
            "source_variable": "candidate_set",
            "attribute_variable": "ordering_attr",
        },
        allow_exploratory=False,
    )

    assert anchor_path is not None
    assert anchor_path["from_role"] == "anchor"
    assert anchor_path["to_role"] == "candidate_set"


def test_normalize_pal_query_plan_coerces_literal_superlative_to_entity_candidate() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "literal",
            "answer_type": "literal",
            "answer_target_phrase": "temperature",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {"surface": "Flare star", "chosen_alias": "Flare star", "role": "anchor"}
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "astronomy.star.temperature_k",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "ordering_attr",
            },
            "ordering_direction": "min",
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object_category.objects",
                    "direction": "reverse",
                    "from": "category",
                    "to": "candidate_set",
                    "from_role": "answer",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "astronomy.star.temperature_k",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "ordering_attr",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["ordering_attr"],
        }
    )

    assert normalized["answer_mode"] == "entity"
    assert normalized["answer_type"] == "entity"
    assert normalized["shared_answer_variable"] == "candidate_set"
    assert normalized["projection"] == ["candidate_set"]
    assert normalized["relation_paths"][0]["from_role"] == "anchor"
    assert normalized["relation_paths"][0]["to_role"] == "candidate_set"


def test_joined_set_grounding_accepts_same_dynamic_relation_family_for_sibling_anchors() -> None:
    controller = _make_controller()

    matched, reason = controller._relation_contract_match_details(
        planned_path={
            "relation": "distilled_spirits.blended_spirit.components",
            "direction": "reverse",
            "from": "candidate_set",
            "to": "rye",
            "from_role": "candidate_set",
            "to_role": "anchor_a",
        },
        grounded_candidate={
            "relation": "distilled_spirits.blended_spirit.components",
            "direction": "reverse",
            "from": "spirit",
            "to": "Corn whiskey",
            "from_role": "candidate_set",
            "to_role": "anchor_b",
        },
        query_shape="count_over_joined_set",
        anchor_count=2,
        anchored_entities=[
            {"surface": "rye", "chosen_alias": "rye", "role": "anchor_a"},
            {
                "surface": "Corn whiskey",
                "chosen_alias": "Corn whiskey",
                "role": "anchor_b",
            },
        ],
    )

    assert matched is True
    assert reason == "accepted_role_match"


def test_joined_set_grounding_accepts_anchor_sourced_constraint_when_other_path_preserves_candidate_pivot() -> None:
    controller = _make_controller()
    query_plan = {
        "query_shape": "count_over_joined_set",
        "relation_paths": [
            {
                "relation": "fictional_universe.fictional_character.species",
                "direction": "reverse",
                "from": "candidate_set",
                "to": "Neyaphem",
                "from_role": "candidate_set",
                "to_role": "anchor_b",
            },
            {
                "relation": "fictional_universe.fictional_character.occupation",
                "direction": "forward",
                "from": "candidate_set",
                "to": "educators",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
            },
        ],
    }

    matched, reason = controller._relation_contract_match_details(
        planned_path=query_plan["relation_paths"][1],
        grounded_candidate={
            "relation": "fictional_universe.fictional_character.occupation",
            "direction": "forward",
            "from": "fictional_character",
            "to": "occupation",
            "from_role": "anchor",
            "to_role": "count_set",
        },
        query_shape="count_over_joined_set",
        anchor_count=2,
        anchored_entities=[
            {"surface": "Neyaphem", "chosen_alias": "Neyaphem", "role": "anchor_b"},
            {"surface": "educators", "chosen_alias": "Teacher", "role": "constraint_value"},
        ],
        query_plan=query_plan,
    )

    assert matched is True
    assert reason == "accepted_role_match"


def test_multi_anchor_grounding_accepts_generic_anchor_for_specific_anchor_roles() -> None:
    controller = _make_controller()

    matched, reason = controller._relation_contract_match_details(
        planned_path={
            "relation": "cvg.computer_game_distribution_system.games_distributed",
            "direction": "reverse",
            "from": "anchor_a",
            "to": "shared_answer",
            "from_role": "anchor_a",
            "to_role": "candidate_set",
        },
        grounded_candidate={
            "relation": "cvg.computer_game_distribution_system.games_distributed",
            "direction": "reverse",
            "from": "distribution_system",
            "to": "candidate_set",
            "from_role": "anchor",
            "to_role": "candidate_set",
        },
        query_shape="multi_anchor_intersection",
        anchor_count=2,
        anchored_entities=[
            {"surface": "Virtual Console", "chosen_alias": "Virtual Console", "role": "anchor_a"},
            {"surface": "Sega of Japan", "chosen_alias": "Sega of Japan", "role": "anchor_b"},
        ],
    )

    assert matched is True
    assert reason == "accepted_role_match"


def test_question_scaffold_rewrite_grounds_cvg_release_region_intersection() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: virtual console, which is developed by sega of japan, was released where?"
        ),
        query_plan={
            "answer_mode": "entity",
            "answer_type": "entity",
            "query_shape": "multi_anchor_intersection",
            "answer_target_phrase": "region",
            "anchored_entities": [
                {"surface": "Virtual Console", "chosen_alias": "Virtual Console", "role": "anchor_a"},
                {"surface": "Sega of Japan", "chosen_alias": "Sega of Japan", "role": "anchor_b"},
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "candidate_set"},
                    {"anchor_role": "anchor_b", "constrains_variable": "candidate_set"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "exploratory:product.release_location",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "answer",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "exploratory:organization.developed_products",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "candidate_set",
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                    "grounding_source": "exploratory",
                },
            ],
            "projection": ["answer"],
            "allow_exploratory_predicates": True,
            "plan_rationale": [],
        },
    )

    relations = [str(path.get("relation") or "") for path in rewritten["relation_paths"]]
    assert rewritten["allow_exploratory_predicates"] is False
    assert "cvg.computer_game_distribution_system.games_distributed" in relations
    assert "cvg.cvg_developer.game_versions_developed" in relations
    assert relations.count("cvg.game_version.regions") == 1
    assert rewritten["shared_answer_variable"] == "candidate_set"
    assert rewritten["projection"] == ["answer"]


def test_question_scaffold_rewrite_drops_redundant_attribute_qualifier_filter() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: what is the camera sensor type of panasonic lumix brand digital camera?, "
            "Entities: ['panasonic lumix']"
        ),
        query_plan={
            "answer_mode": "entity",
            "answer_type": "entity",
            "query_shape": "single_anchor_lookup",
            "answer_target_phrase": "camera sensor type",
            "anchored_entities": [
                {"surface": "panasonic lumix", "chosen_alias": "m.093_tr", "role": "anchor"}
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "join_structure": {
                "type": "single_path",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "candidate_set"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "business.brand.products",
                    "direction": "forward",
                    "from": "brand",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "type",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "digicams.digital_camera.sensor_type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "answer",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["answer"],
            "allow_exploratory_predicates": False,
            "plan_rationale": [],
            "strategy": "Project Panasonic Lumix camera products to their sensor type.",
        },
    )

    relations = [str(path.get("relation") or "") for path in rewritten["relation_paths"]]
    assert relations == [
        "business.brand.products",
        "digicams.digital_camera.sensor_type",
    ]
    assert rewritten["query_shape"] == "single_anchor_lookup"


def test_cvg_release_region_rewrite_replaces_generic_projection_placeholder() -> None:
    controller = _make_controller()

    rewritten = controller._apply_question_scaffold_plan_rewrites(
        task_question=(
            "Question: virtual console, which is developed by sega of japan, was released where?"
        ),
        query_plan={
            "answer_mode": "entity",
            "answer_type": "entity",
            "query_shape": "multi_anchor_intersection",
            "answer_target_phrase": "region",
            "anchored_entities": [
                {"surface": "Virtual Console", "chosen_alias": "Virtual Console", "role": "anchor_a"},
                {"surface": "Sega of Japan", "chosen_alias": "Sega of Japan", "role": "anchor_b"},
            ],
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "candidate_set"},
                    {"anchor_role": "anchor_b", "constrains_variable": "candidate_set"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "exploratory:product.release_location",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "shared_answer",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "exploratory:organization.developed_products",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "candidate_set",
                    "from_role": "anchor_b",
                    "to_role": "candidate_set",
                    "grounding_source": "exploratory",
                },
            ],
            "projection": ["shared_answer"],
            "allow_exploratory_predicates": True,
            "plan_rationale": [],
        },
    )

    assert rewritten["projection"] == ["answer"]
    assert rewritten["relation_paths"][-1]["to"] == "answer"
    assert rewritten["relation_paths"][-1]["to_role"] == "answer"


def test_query_candidate_rejects_unplanned_predicates() -> None:
    controller = _make_controller()
    errors = controller._validate_pal_query_candidate(
        raw_output="",
        generated_code="def solve(endpoint_url):\n    pass\n",
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT ?answer WHERE { "
            '?organization fb:type.object.name "National Wine Centre of Australia"@en . '
            "?organization fb:education.educational_institution.parent_institution ?answer . "
            "} LIMIT 50"
        ),
        query_plan={
            "answer_mode": "entity",
            "relation_paths": [
                {
                    "relation": "organization.organization.parent_organization",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "answer",
                }
            ],
            "ordering_attribute": {},
            "allow_exploratory_predicates": False,
        },
    )

    assert "query_uses_unplanned_predicate:education.educational_institution.parent_institution" in errors


def test_query_candidate_rejects_single_use_helper_join_variable() -> None:
    controller = _make_controller()
    query_text = (
        "PREFIX fb: <http://rdf.freebase.com/ns/> "
        "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
        "fb:m.03bv2kt fb:broadcast.genre.content ?candidate_set . "
        "fb:m.03fx9_c fb:broadcast.content.producer ?producer . "
        "fb:m.03bv2kt fb:broadcast.producer.produces ?candidate_set . "
        "}"
    )
    errors = controller._validate_pal_query_candidate(
        raw_output="",
        generated_code="def solve(endpoint_url):\n    pass\n",
        query_text=query_text,
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {"surface": "Higher Education", "chosen_alias": "m.03bv2kt", "role": "anchor_a"},
                {"surface": "To the Best of Our Knowledge", "chosen_alias": "m.03fx9_c", "role": "anchor_b"},
            ],
            "relation_paths": [
                {
                    "relation": "broadcast.genre.content",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "candidate_set",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "producer",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                },
                {
                    "relation": "broadcast.producer.produces",
                    "direction": "forward",
                    "from": "producer",
                    "to": "candidate_set",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                },
            ],
            "ordering_attribute": {},
            "allow_exploratory_predicates": False,
        },
    )

    assert "single_use_helper_variable:producer" in errors


def test_extract_sparql_query_texts_tracks_setquery_assignments() -> None:
    controller = _make_controller()
    query_texts = controller._extract_sparql_query_texts(
        """
from SPARQLWrapper import SPARQLWrapper, JSON

def solve(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    query_primary = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?race) AS ?count) WHERE {
      ?anchor fb:type.object.name "Seventh sphere"@en .
      ?anchor fb:fictional_universe.fictional_universe.races ?race .
    }
    \"\"\"
    sparql.setQuery(query_primary)
    query_fallback = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?target) AS ?count) WHERE {
      ?anchor fb:type.object.name "Seventh sphere"@en .
      ?anchor fb:fictional_universe.fictional_setting.setting_type ?target .
    }
    \"\"\"
    sparql.setQuery(query_fallback)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
"""
    )

    assert len(query_texts) == 2
    assert "fictional_universe.fictional_universe.races" in query_texts[0]
    assert "fictional_universe.fictional_setting.setting_type" in query_texts[1]


def test_validate_query_predicates_ignores_freebase_entity_ids() -> None:
    controller = _make_controller()

    errors = controller._validate_query_predicates_against_plan(
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?candidate_set fb:language.language_dialect.language fb:m.01c44b . "
            "} LIMIT 50"
        ),
        query_plan={
            "answer_mode": "count",
            "relation_paths": [
                {
                    "relation": "language.language_dialect.language",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                }
            ],
            "ordering_attribute": {},
            "allow_exploratory_predicates": False,
        },
    )

    assert errors == []


def test_query_candidate_accepts_count_fallback_queries_with_named_query_variables() -> None:
    controller = _make_controller()
    generated_code = """
from SPARQLWrapper import SPARQLWrapper, JSON

def solve(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    query_primary = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?race) AS ?count) WHERE {
      ?anchor fb:type.object.name "Seventh sphere"@en .
      ?anchor fb:fictional_universe.fictional_universe.races ?race .
    }
    \"\"\"
    sparql.setQuery(query_primary)
    result_primary = sparql.query().convert()
    query_fallback = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?target) AS ?count) WHERE {
      ?anchor fb:type.object.name "Seventh sphere"@en .
      ?anchor fb:fictional_universe.fictional_setting.setting_type ?target .
    }
    \"\"\"
    sparql.setQuery(query_fallback)
    sparql.setReturnFormat(JSON)
    return result_primary
"""
    query_texts = controller._extract_sparql_query_texts(generated_code)
    errors = controller._validate_pal_query_candidate(
        raw_output="",
        generated_code=generated_code,
        query_text=query_texts[0],
        query_texts=query_texts,
        query_plan={
            "answer_mode": "count",
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_universe.races",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_setting.setting_type",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "ordering_attribute": {},
            "allow_exploratory_predicates": False,
        },
    )

    assert "missing_query_text" not in errors
    assert "count_plan_without_count_projection" not in errors


def test_query_candidate_allows_binding_unions_scaled_by_bound_inputs() -> None:
    controller = _make_controller()
    query_text = """
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT ?cheese WHERE {
  {
    { ?milk_source_a fb:type.object.name ?name_a . FILTER(LCASE(STR(?name_a)) = "goat") }
    UNION
    { ?milk_source_a fb:common.topic.alias ?alias_a . FILTER(LCASE(STR(?alias_a)) = "goat") }
  }
  {
    { ?milk_source_b fb:type.object.name ?name_b . FILTER(LCASE(STR(?name_b)) = "cows") }
    UNION
    { ?milk_source_b fb:common.topic.alias ?alias_b . FILTER(LCASE(STR(?alias_b)) = "cows") }
  }
  {
    { ?texture_value fb:type.object.name ?tex_name . FILTER(LCASE(STR(?tex_name)) = "semi-firm") }
    UNION
    { ?texture_value fb:common.topic.alias ?tex_alias . FILTER(LCASE(STR(?tex_alias)) = "semi-firm") }
  }
  ?milk_source_a fb:food.cheese_milk_source.cheeses ?cheese .
  ?milk_source_b fb:food.cheese_milk_source.cheeses ?cheese .
  ?cheese fb:food.cheese.texture ?texture_value .
}
"""
    errors = controller._validate_pal_query_candidate(
        raw_output="",
        generated_code="def solve(endpoint_url):\n    pass\n",
        query_text=query_text,
        query_plan={
            "answer_mode": "entity",
            "anchored_entities": [
                {"surface": "Goat", "chosen_alias": "Goat", "role": "anchor_a"},
                {"surface": "cows", "chosen_alias": "cows", "role": "anchor_b"},
                {"surface": "semi-firm", "chosen_alias": "semi-firm", "role": "constraint_value"},
            ],
            "relation_paths": [
                {
                    "relation": "food.cheese_milk_source.cheeses",
                    "direction": "reverse",
                    "from_role": "constraint_value",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                },
                {
                    "relation": "food.cheese.texture",
                    "direction": "forward",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "ordering_attribute": {},
            "allow_exploratory_predicates": False,
        },
    )

    assert "excessive_union_branches:5" not in errors


def test_plausibility_repairs_zero_count_with_unanchored_exploratory_constraint() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Goro", "chosen_alias": "Goro", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor", "constrains_variable": "candidate_set", "notes": "bind the rank-derived character set"}
                ],
            },
            "relation_paths": [
                {
                    "relation": "fictional_universe.character_rank.characters_of_this_rank",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "anchor",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "fictional_universe.fictional_character.appears_in_book",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "work",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "exploratory",
                },
            ],
            "allow_exploratory_predicates": True,
            "strategy": "count characters with the anchor rank and an exploratory book filter",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            '?anchor fb:type.object.name "Goro"@en . '
            "?rank fb:fictional_universe.character_rank.characters_of_this_rank ?anchor . "
            "?rank fb:fictional_universe.character_rank.characters_of_this_rank ?candidate_set . "
            "?candidate_set fb:fictional_universe.fictional_character.appears_in_book ?work . "
            "} LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Goro"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_has_unanchored_exploratory_constraint" in verdict.reasons


def test_plausibility_accepts_count_join_when_constraints_target_candidate_set() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "anchored_entities": [
                {"surface": "scotch", "chosen_alias": "scotch", "role": "anchor_a"},
                {
                    "surface": "Corn whiskey",
                    "chosen_alias": "Corn whiskey",
                    "role": "anchor_b",
                },
            ],
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "candidate_set",
                        "notes": "candidate must include scotch",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "candidate_set",
                        "notes": "candidate must include Corn whiskey",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "example.contains_ingredient",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "scotch",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "example.contains_ingredient",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "Corn whiskey",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count joined candidate set",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            '?candidate_set fb:example.contains_ingredient ?a . '
            '?a fb:type.object.name "scotch"@en . '
            '?candidate_set fb:example.contains_ingredient ?b . '
            '?b fb:type.object.name "Corn whiskey"@en . '
            "} LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "1"}}
                ]
            }
        },
        entities=["scotch", "Corn whiskey"],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_plausibility_accepts_count_join_when_constraints_target_shared_answer_alias() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "shared_answer_variable": "shared_answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "anchored_entities": [
                {"surface": "brazil", "chosen_alias": "brazil", "role": "anchor_a"},
                {
                    "surface": "Manchester Terrier",
                    "chosen_alias": "Manchester Terrier",
                    "role": "anchor_b",
                },
            ],
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "shared_answer",
                        "notes": "country filter",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "shared_answer",
                        "notes": "temperament filter",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "biology.animal_breed.country_of_origin",
                    "direction": "forward",
                    "from": "breed",
                    "to": "country",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "breed",
                    "to": "breed_temperament",
                    "from_role": "shared_answer",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "manchester_temperament",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count joined shared-answer set",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?breed) AS ?count) WHERE { "
            '?country fb:type.object.name "brazil"@en . '
            '?manchester fb:type.object.name "Manchester Terrier"@en . '
            "?breed fb:biology.animal_breed.country_of_origin ?country . "
            "?breed fb:biology.animal_breed.temperament ?breed_temperament . "
            "?manchester fb:biology.animal_breed.temperament ?manchester_temperament . "
            "FILTER(?breed_temperament = ?manchester_temperament) "
            "} LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["brazil", "Manchester Terrier"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="brazil",
                entity_count=5,
                path_count=0,
                relation_probed="biology.animal_breed.country_of_origin",
                anchor_position="object",
            ),
            AnchorProbeResult(
                anchor_name="Manchester Terrier",
                entity_count=9,
                path_count=54,
                relation_probed="biology.animal_breed.temperament",
                anchor_position="subject",
            ),
        ],
    )

    assert "join_structure_drifting_constraints:shared_answer" not in verdict.reasons


def test_plausibility_accepts_multi_anchor_query_when_resolved_mids_are_pinned() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "shared_answer_variable": "shared_type",
            "candidate_set_variable": "shared_type",
            "count_set_variable": "",
            "anchored_entities": [
                {
                    "surface": "the museum of modern art",
                    "chosen_alias": "m.0hhjk",
                    "role": "anchor_a",
                },
                {
                    "surface": "Smithsonian Institution",
                    "chosen_alias": "m.0hfyj",
                    "role": "anchor_b",
                },
            ],
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "shared_type"},
                    {"anchor_role": "anchor_b", "constrains_variable": "shared_type"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "architecture.museum.type_of_museum",
                    "direction": "forward",
                    "from_role": "anchor_a",
                    "to_role": "shared_answer",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "architecture.museum.type_of_museum",
                    "direction": "forward",
                    "from_role": "anchor_b",
                    "to_role": "shared_answer",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "shared museum type intersection",
        },
        query_text=(
            "SELECT ?shared_type WHERE { "
            "VALUES ?anchor_a { fb:m.0hhjk } "
            "VALUES ?anchor_b { fb:m.0hfyj } "
            "?anchor_a fb:architecture.museum.type_of_museum ?shared_type . "
            "?anchor_b fb:architecture.museum.type_of_museum ?shared_type . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {
                        "shared_type": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.012abc",
                        }
                    }
                ]
            }
        },
        entities=["the museum of modern art", "Smithsonian Institution"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="the museum of modern art",
                entity_count=1,
                path_count=1,
                relation_probed="architecture.museum.type_of_museum",
                anchor_position="subject",
                resolved_entity_id="m.0hhjk",
            ),
            AnchorProbeResult(
                anchor_name="Smithsonian Institution",
                entity_count=1,
                path_count=1,
                relation_probed="architecture.museum.type_of_museum",
                anchor_position="subject",
                resolved_entity_id="m.0hfyj",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED


def test_dynamic_probe_scoring_uses_question_text() -> None:
    controller = _make_controller()

    rank_score = controller._score_probe_predicate(
        "fictional_universe.fictional_character.rank",
        "",
        [],
        "Question: the rank of goro has been given to how many book characters?",
    )
    species_score = controller._score_probe_predicate(
        "fictional_universe.fictional_character.species",
        "",
        [],
        "Question: the rank of goro has been given to how many book characters?",
    )

    assert rank_score > species_score


def test_dynamic_probe_scoring_prefers_parent_relation_for_has_question() -> None:
    controller = _make_controller()

    parent_score = controller._score_probe_predicate(
        "education.educational_institution.parent_institution",
        "institution",
        [],
        "Question: which institution has national wine centre of australia?",
    )
    campus_score = controller._score_probe_predicate(
        "education.educational_institution_campus.educational_institution",
        "institution",
        [],
        "Question: which institution has national wine centre of australia?",
    )

    assert parent_score >= campus_score + 40


def test_dynamic_probe_scoring_uses_anchor_clue_for_medicine_bridge() -> None:
    controller = _make_controller()

    marketed_score = controller._score_probe_predicate(
        "medicine.drug.marketed_formulations",
        "dosage form",
        ["medicine", "drug"],
        "Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?",
        anchor_clue="formulation_input",
    )
    moiety_score = controller._score_probe_predicate(
        "medicine.drug.active_moieties",
        "dosage form",
        ["medicine", "drug"],
        "Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?",
        anchor_clue="formulation_input",
    )

    assert marketed_score > moiety_score


def test_dynamic_probe_scoring_prefers_origin_family_for_origin_constraint() -> None:
    controller = _make_controller()

    originating_here_score = controller._score_probe_predicate(
        "biology.breed_origin.breeds_originating_here",
        "different breeds",
        ["biology", "animal"],
        "Question: how many different breeds from republic of brazil have the same temperament as the manchester terrier?",
        anchor_clue="origin_constraint",
    )
    country_of_origin_score = controller._score_probe_predicate(
        "biology.animal_breed.country_of_origin",
        "different breeds",
        ["biology", "animal"],
        "Question: how many different breeds from republic of brazil have the same temperament as the manchester terrier?",
        anchor_clue="origin_constraint",
    )

    assert originating_here_score > country_of_origin_score


def test_dynamic_probe_scoring_prefers_successor_family_for_preceded_by_question() -> None:
    controller = _make_controller()

    successor_score = controller._score_probe_predicate(
        "cvg.cvg_engine.successor",
        "video game engine",
        ["software", "engine", "video game"],
        "Question: the quake 3 engine was proceeded by which video game engine?",
    )
    unrelated_score = controller._score_probe_predicate(
        "cvg.cvg_engine.games",
        "video game engine",
        ["software", "engine", "video game"],
        "Question: the quake 3 engine was proceeded by which video game engine?",
    )

    assert successor_score > unrelated_score


def test_dynamic_probe_scoring_penalizes_category_relations_for_feature_clues() -> None:
    controller = _make_controller()

    technique_score = controller._score_probe_predicate(
        "martial_arts.martial_art.techniques",
        "martial art",
        ["martial_arts"],
        "Question: what martial art has the same category as mongolian wrestling and has strike?",
        anchor_clue="attribute_value.feature",
    )
    category_score = controller._score_probe_predicate(
        "martial_arts.martial_art.category",
        "martial art",
        ["martial_arts"],
        "Question: what martial art has the same category as mongolian wrestling and has strike?",
        anchor_clue="attribute_value.feature",
    )

    assert technique_score > category_score


def test_grounding_candidates_include_reverse_breed_origin_family() -> None:
    controller = _make_controller()

    candidates = controller._build_grounded_relation_candidates(
        "Question: how many different breeds from republic of brazil have the same temperament as the manchester terrier?, Entities: ['republic of brazil', 'Manchester Terrier']"
    )

    assert any(
        candidate.get("relation") == "biology.breed_origin.breeds_originating_here"
        and candidate.get("direction") == "reverse"
        and candidate.get("from") == "country"
        and candidate.get("to") == "breed"
        for candidate in candidates
    )


def test_dynamic_probe_candidate_uses_semantic_endpoint_labels() -> None:
    controller = _make_controller()

    candidates = controller._build_dynamic_candidate_list(
        outgoing_iris=[
            "http://rdf.freebase.com/ns/medicine.drug_ingredient.active_moiety_of_formulation"
        ],
        incoming_iris=[
            "http://rdf.freebase.com/ns/medicine.drug.active_moieties"
        ],
        anchor_entity="Enalaprilat",
        anchor_role="anchor_b",
        answer_target_phrase="dosage form",
        domain_hints=["medicine", "drug"],
        question_text="Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?",
        anchor_clue="active_ingredient",
        max_per_direction=4,
        max_total=4,
    )

    assert any(
        candidate.get("relation") == "medicine.drug_ingredient.active_moiety_of_formulation"
        and candidate.get("to") == "formulation"
        for candidate in candidates
    )
    assert any(
        candidate.get("relation") == "medicine.drug.active_moieties"
        and candidate.get("from") == "drug"
        for candidate in candidates
    )


def test_dynamic_probe_query_can_filter_to_freebase_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv("PAL_RUNTIME_DYNAMIC_PROBE_FB_FILTER", "1")
    captured_queries: list[str] = []

    def _fake_run_probe_sparql_query(
        *,
        endpoint: str,
        sparql: str,
        timeout_s: float = 5.0,
    ) -> list[str]:
        captured_queries.append(sparql)
        return []

    controller._run_probe_sparql_query = _fake_run_probe_sparql_query

    candidates = controller._probe_dynamic_relation_candidates_for_anchor(
        anchor_entity="J. Paul Reddam",
        answer_target_phrase="animals",
        domain_hints=[],
        question_text="Question: what animals does paul reddam have?",
    )

    assert candidates == []
    assert len(captured_queries) == 2
    assert all('FILTER(STRSTARTS(STR(?p), "http://rdf.freebase.com/ns/"))' in q for q in captured_queries)


def test_dynamic_probe_can_keep_base_namespace_relations_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller()

    base_predicate = (
        "http://rdf.freebase.com/ns/base.thoroughbredracing."
        "thoroughbred_racehorse_owner.horses_owned"
    )

    assert controller._is_noise_predicate(base_predicate) is True

    monkeypatch.setenv("PAL_RUNTIME_DYNAMIC_PROBE_ALLOW_BASE", "1")

    assert controller._is_noise_predicate(base_predicate) is False


def test_curated_grounding_includes_bridge_candidates_for_medicine_and_fictional_worlds() -> None:
    controller = _make_controller()

    medicine_candidates = controller._build_grounded_relation_candidates(
        "Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?, Entities: ['Naloxone', 'Enalaprilat']"
    )
    fictional_candidates = controller._build_grounded_relation_candidates(
        "Question: what number of different species are in the fictional world of seventh sphere?, Entities: ['Seventh sphere']"
    )

    assert any(
        candidate.get("relation") == "medicine.drug_ingredient.active_moiety_of_formulation"
        for candidate in medicine_candidates
    )
    assert any(
        candidate.get("relation") == "medicine.drug_ingredient.active_ingredient_of_formulation"
        for candidate in medicine_candidates
    )
    assert any(
        candidate.get("relation") == "fictional_universe.fictional_setting.universe"
        for candidate in fictional_candidates
    )


def test_direct_count_grounding_preserves_bridge_roles_for_species_count() -> None:
    controller = _make_controller()
    question = (
        "Question: what number of different species are in the fictional world of seventh sphere?, "
        "Entities: ['Seventh sphere']"
    )
    question_text, entities = controller._split_task_question(question)
    answer_target = controller._extract_answer_target_phrase(question_text)

    grounded = controller._build_grounded_relation_candidates_with_dynamic_fallback(
        task_question=question,
        entities=entities,
        answer_target_phrase=answer_target,
        domain_hints=[],
        question_interpretation=None,
    )

    universe_bridge = next(
        candidate
        for candidate in grounded
        if candidate.get("relation") == "fictional_universe.fictional_setting.universe"
        and candidate.get("grounding_source") == "curated"
    )
    species_count = next(
        candidate
        for candidate in grounded
        if candidate.get("relation") == "fictional_universe.fictional_universe.species"
        and candidate.get("grounding_source") == "curated"
    )

    assert universe_bridge["from_role"] == "anchor"
    assert universe_bridge["to_role"] == "candidate_set"
    assert species_count["from_role"] == "candidate_set"
    assert species_count["to_role"] == "count_set"


def test_curated_grounding_includes_superlative_domain_families() -> None:
    controller = _make_controller()

    meteorology_candidates = controller._build_grounded_relation_candidates(
        "Question: what was the most recently formed cyclone in the same category as hurricane dolly?, Entities: ['Hurricane Dolly']"
    )
    fictional_candidates = controller._build_grounded_relation_candidates(
        "Question: which short story of the sacred band of stepsons universe universe is know to have the earliest copyright date?, Entities: ['The Sacred Band of Stepsons universe']"
    )

    assert any(
        candidate.get("relation") == "meteorology.tropical_cyclone.category"
        for candidate in meteorology_candidates
    )
    assert any(
        candidate.get("relation")
        == "meteorology.tropical_cyclone_category.tropical_cyclones"
        for candidate in meteorology_candidates
    )
    assert any(
        candidate.get("relation") == "meteorology.tropical_cyclone.formed"
        for candidate in meteorology_candidates
    )
    assert any(
        candidate.get("relation")
        == "fictional_universe.fictional_universe.works_set_here"
        for candidate in fictional_candidates
    )
    assert any(
        candidate.get("relation") == "book.written_work.copyright_date"
        for candidate in fictional_candidates
    )


def test_curated_grounding_includes_camera_astronomy_broadcast_and_nobility_families() -> None:
    controller = _make_controller()

    camera_candidates = controller._build_grounded_relation_candidates(
        "Question: what is the camera sensor type of panasonic lumix brand digital camera?, Entities: ['panasonic lumix']"
    )
    astronomy_candidates = controller._build_grounded_relation_candidates(
        "Question: what is the minimum temperature in the star category of flare star?, Entities: ['Flare star']"
    )
    broadcast_candidates = controller._build_grounded_relation_candidates(
        "Question: what number of contents about higher education are produced by the producer of to the best of our knowledge?, Entities: ['Higher Education', 'To the Best of Our Knowledge']"
    )
    nobility_candidates = controller._build_grounded_relation_candidates(
        "Question: which system of nobility had the baronet rank first?, Entities: ['Baronet']"
    )
    sports_candidates = controller._build_grounded_relation_candidates(
        "Question: what's the total number of basketball teams that warren played for?, Entities: ['warren']"
    )
    cvg_candidates = controller._build_grounded_relation_candidates(
        "Question: virtual console, which is developed by sega of japan, was released where?, Entities: ['Virtual Console', 'Sega of Japan']"
    )
    calendar_candidates = controller._build_grounded_relation_candidates(
        "Question: based on the information within the gregorian calendar what is the last day of the week?, Entities: ['Gregorian calendar']"
    )
    nebula_candidates = controller._build_grounded_relation_candidates(
        "Question: what's the name of the nebula that's farthest away from us?, Entities: ['Nebula']"
    )

    assert any(
        candidate.get("relation") == "digicams.digital_camera.sensor_type"
        for candidate in camera_candidates
    )
    assert any(
        candidate.get("relation") == "astronomy.celestial_object_category.objects"
        for candidate in astronomy_candidates
    )
    assert any(
        candidate.get("relation") == "astronomy.star.temperature_k"
        for candidate in astronomy_candidates
    )
    assert any(
        candidate.get("relation") == "time.calendar.days_of_week"
        for candidate in calendar_candidates
    )
    assert any(
        candidate.get("relation") == "time.day_of_week.sequence_number"
        for candidate in calendar_candidates
    )
    assert any(
        candidate.get("relation") == "astronomy.celestial_object.category"
        for candidate in nebula_candidates
    )
    assert any(
        candidate.get("relation") == "astronomy.celestial_object.cosmological_distance"
        for candidate in nebula_candidates
    )
    assert any(
        candidate.get("relation") == "broadcast.content.producer"
        for candidate in broadcast_candidates
    )
    assert any(
        candidate.get("relation") == "broadcast.producer.produces"
        for candidate in broadcast_candidates
    )
    assert any(
        candidate.get("relation") == "royalty.noble_rank.used_in"
        for candidate in nobility_candidates
    )
    assert any(
        candidate.get("relation") == "royalty.system_of_nobility.used_from_date"
        for candidate in nobility_candidates
    )
    assert any(
        candidate.get("relation") == "sports.pro_athlete.teams"
        for candidate in sports_candidates
    )
    assert any(
        candidate.get("relation") == "sports.sports_team_roster.team"
        for candidate in sports_candidates
    )
    assert any(
        candidate.get("relation") == "cvg.computer_game_distribution_system.games_distributed"
        for candidate in cvg_candidates
    )
    assert any(
        candidate.get("relation") == "cvg.cvg_developer.game_versions_developed"
        for candidate in cvg_candidates
    )
    assert any(
        candidate.get("relation") == "cvg.game_version.regions"
        for candidate in cvg_candidates
    )


def test_preferred_scaffold_candidates_treat_minimum_as_superlative() -> None:
    controller = _make_controller()

    candidates = controller._build_preferred_scaffold_candidates(
        question_text="Question: what is the minimum temperature in the star category of flare star?",
        answer_target_phrase="temperature",
        question_inputs=[
            {"surface": "Flare star", "kind": "named_entity", "role_hint": "anchor"},
            {"surface": "temperature", "kind": "answer_target", "role_hint": "answer_target"},
        ],
    )

    assert any(
        candidate.get("name") == "superlative_over_candidate_set"
        for candidate in candidates
    )


def test_normalize_superlative_plan_forces_entity_answer_mode() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "literal",
            "answer_type": "literal",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "Flare star",
                    "chosen_alias": "Flare star",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "answer",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "",
            "ordering_attribute": {
                "relation": "astronomy.star.temperature_k",
                "direction": "forward",
                "source_variable": "candidate_set",
                "attribute_variable": "temp",
            },
            "ordering_direction": "min",
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object_category.objects",
                    "direction": "reverse",
                    "from": "category",
                    "to": "candidate_set",
                    "grounding_source": "curated",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "astronomy.star.temperature_k",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "temp",
                    "grounding_source": "curated",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                },
            ],
            "projection": ["temp"],
            "allow_exploratory_predicates": False,
        }
    )

    assert normalized["query_shape"] == "superlative_chain"
    assert normalized["answer_mode"] == "entity"
    assert normalized["answer_type"] == "entity"
    assert normalized["shared_answer_variable"] == "candidate_set"
    assert normalized["projection"] == ["candidate_set"]


def test_normalize_superlative_plan_preserves_grounded_answer_projection_when_present() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "answer_target_phrase": "system",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Baronet",
                    "chosen_alias": "Baronet",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "ordering_attribute": {
                "relation": "royalty.system_of_nobility.used_from_date",
                "direction": "forward",
                "source_variable": "answer",
                "attribute_variable": "start_date",
            },
            "ordering_direction": "min",
            "relation_paths": [
                {
                    "relation": "royalty.noble_rank.used_in",
                    "direction": "reverse",
                    "from": "rank",
                    "to": "rank_relationship",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "royalty.system_rank_relationship.system",
                    "direction": "reverse",
                    "from": "rank_relationship",
                    "to": "system_of_nobility",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "curated",
                },
                {
                    "relation": "royalty.system_of_nobility.used_from_date",
                    "direction": "forward",
                    "from": "system_of_nobility",
                    "to": "start_date",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["candidate_set"],
            "allow_exploratory_predicates": False,
        }
    )

    assert normalized["query_shape"] == "superlative_chain"
    assert normalized["shared_answer_variable"] == "system_of_nobility"
    assert normalized["projection"] == ["system_of_nobility"]


def test_normalize_superlative_plan_repairs_misrole_labeled_rank_to_system_chain() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "answer_target_phrase": "system",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Baronet",
                    "chosen_alias": "Baronet",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "rank",
            "candidate_set_variable": "rank_relationship",
            "count_set_variable": "",
            "ordering_attribute": {
                "relation": "royalty.system_of_nobility.used_from_date",
                "direction": "forward",
                "source_variable": "answer",
                "attribute_variable": "start_date",
            },
            "ordering_direction": "min",
            "relation_paths": [
                {
                    "relation": "royalty.noble_rank.used_in",
                    "direction": "reverse",
                    "from": "rank",
                    "to": "rank_relationship",
                    "from_role": "answer",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "royalty.system_rank_relationship.system",
                    "direction": "reverse",
                    "from": "rank_relationship",
                    "to": "system_of_nobility",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "curated",
                },
                {
                    "relation": "royalty.system_of_nobility.used_from_date",
                    "direction": "forward",
                    "from": "system_of_nobility",
                    "to": "start_date",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["rank"],
            "allow_exploratory_predicates": False,
        }
    )

    assert normalized["shared_answer_variable"] == "system_of_nobility"
    assert normalized["projection"] == ["system_of_nobility"]
    assert normalized["relation_paths"][0]["from_role"] == "anchor"
    assert normalized["relation_paths"][0]["to_role"] == "candidate_set"
    assert normalized["relation_paths"][1]["from_role"] == "candidate_set"
    assert normalized["relation_paths"][1]["to_role"] == "answer"


def test_normalize_superlative_plan_renormalizes_reverse_category_anchor_path() -> None:
    controller = _make_controller()

    normalized = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "answer_target_phrase": "temperature",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Flare star",
                    "chosen_alias": "Flare star",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_star",
            "candidate_set_variable": "candidate_star",
            "count_set_variable": "",
            "ordering_attribute": {
                "relation": "astronomy.star.temperature_k",
                "direction": "forward",
                "source_variable": "candidate_star",
                "attribute_variable": "temperature_k",
            },
            "ordering_direction": "min",
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object_category.objects",
                    "direction": "reverse",
                    "from": "category",
                    "to": "candidate_set",
                    "grounding_source": "curated",
                    "from_role": "type_set",
                    "to_role": "anchor",
                },
                {
                    "relation": "astronomy.star.temperature_k",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "answer",
                    "grounding_source": "curated",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                },
            ],
            "projection": ["candidate_star"],
            "allow_exploratory_predicates": False,
        }
    )

    assert normalized["relation_paths"][0]["from_role"] == "anchor"
    assert normalized["relation_paths"][0]["to_role"] == "candidate_set"


def test_generic_type_augmentation_skips_when_specific_type_family_exists() -> None:
    controller = _make_controller()

    candidates = controller._augment_generic_type_relation_candidates(
        relation_candidates=[
            {
                "relation": "meteorology.tropical_cyclone.category",
                "direction": "forward",
                "from": "cyclone",
                "to": "category",
                "from_role": "anchor",
                "to_role": "type_set",
                "support": "curated_meteorology_predicate",
            },
            {
                "relation": "meteorology.tropical_cyclone_category.tropical_cyclones",
                "direction": "forward",
                "from": "category",
                "to": "cyclone",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "support": "curated_meteorology_predicate",
            },
        ],
        answer_target_phrase="cyclone",
        question_interpretation={
            "question_inputs": [
                {
                    "surface": "category",
                    "kind": "shared_attribute",
                    "role_hint": "shared_attribute",
                }
            ]
        },
    )

    assert not any(
        candidate.get("support") == "generic_type_relation" for candidate in candidates
    )


def test_structural_repair_synthesizes_anchor_specific_bridge_candidates() -> None:
    controller = _make_controller()

    synthesized = controller._synthesize_anchor_specific_bridge_candidates(
        task_question="Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?, Entities: ['Naloxone', 'Enalaprilat']",
        query_plan={
            "shared_answer_variable": "formulation",
            "anchored_entities": [
                {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
            ],
        },
        relation_grounding=[
            {
                "relation": "medicine.drug.marketed_formulations",
                "direction": "forward",
                "from": "drug",
                "to": "formulation",
                "grounding_source": "curated",
            },
            {
                "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                "direction": "forward",
                "from": "active_ingredient",
                "to": "formulation",
                "grounding_source": "curated",
            },
        ],
    )

    assert any(
        candidate.get("relation") == "medicine.drug.marketed_formulations"
        and candidate.get("from") == "Naloxone"
        and candidate.get("to") == "formulation"
        and candidate.get("from_role") == "anchor_a"
        for candidate in synthesized
    )
    assert any(
        candidate.get("relation") == "medicine.drug_ingredient.active_ingredient_of_formulation"
        and candidate.get("from") == "Enalaprilat"
        and candidate.get("to") == "formulation"
        and candidate.get("from_role") == "anchor_b"
        for candidate in synthesized
    )


def test_structural_repair_rewrites_join_empty_plan_to_projected_answer_intersection() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_projected_answer_intersection_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                    {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
                ],
                "normalized_aliases": [],
                "shared_answer_variable": "formulation",
                "candidate_set_variable": "candidate_set",
                "count_set_variable": "",
                "join_structure": {
                    "type": "intersection",
                    "anchor_constraints": [
                        {
                            "anchor_role": "anchor_a",
                            "constrains_variable": "formulation",
                            "notes": "Naloxone constrains the shared formulation set",
                        },
                        {
                            "anchor_role": "anchor_b",
                            "constrains_variable": "formulation",
                            "notes": "Enalaprilat constrains the shared formulation set",
                        },
                    ],
                },
                "relation_paths": [
                    {
                        "relation": "medicine.drug_formulation.formulation_of",
                        "direction": "reverse",
                        "from": "formulation",
                        "to": "Naloxone",
                        "from_role": "candidate_set",
                        "to_role": "anchor_a",
                        "grounding_source": "curated",
                        "reason": "bind formulations to Naloxone",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_moiety_of_formulation",
                        "direction": "forward",
                        "from": "Enalaprilat",
                        "to": "formulation",
                        "from_role": "anchor_b",
                        "to_role": "candidate_set",
                        "grounding_source": "curated",
                        "reason": "bind formulations to Enalaprilat",
                    },
                    {
                        "relation": "medicine.drug_formulation.dosage_form",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "dosage_form",
                        "from_role": "candidate_set",
                        "to_role": "shared_answer",
                        "grounding_source": "curated",
                        "reason": "project dosage form",
                    },
                ],
                "projection": ["dosage_form", "dosage_form_name"],
                "allow_exploratory_predicates": False,
                "strategy": "intersect on formulations then project dosage_form",
                "plan_rationale": ["start with a shared formulation bridge"],
            }
        )
    )

    assert repaired_plan is not None
    assert repaired_plan["shared_answer_variable"] == "dosage_form"
    assert [
        item["constrains_variable"]
        for item in repaired_plan["join_structure"]["anchor_constraints"]
    ] == ["candidate_set_anchor_a", "candidate_set_anchor_b"]
    assert sum(
        1
        for item in repaired_plan["relation_paths"]
        if item.get("relation") == "medicine.drug_formulation.dosage_form"
    ) == 2
    assert any(
        item.get("from") == "candidate_set_anchor_a"
        and item.get("to") == "dosage_form"
        for item in repaired_plan["relation_paths"]
    )
    assert any(
        item.get("from") == "candidate_set_anchor_b"
        and item.get("to") == "dosage_form"
        for item in repaired_plan["relation_paths"]
    )


def test_structural_repair_rewrites_shared_type_join_empty_plan_to_pivot_bridge() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_shared_type_pivot_bridge_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "shared_type_intersection",
                "anchored_entities": [
                    {"surface": "Martin Hug", "chosen_alias": "m.010f3n0k", "role": "anchor_a"},
                    {"surface": "sci", "chosen_alias": "m.04_754t", "role": "anchor_b"},
                ],
                "shared_answer_variable": "shared_answer",
                "candidate_set_variable": "candidate_set",
                "join_structure": {
                    "type": "shared_type",
                    "anchor_constraints": [
                        {"anchor_role": "anchor_a", "constrains_variable": "candidate_set", "notes": "anchor a via membership"},
                        {"anchor_role": "anchor_b", "constrains_variable": "shared_type", "notes": "anchor b via type"},
                    ],
                },
                "relation_paths": [
                    {
                        "relation": "organization.organization_board_membership.member",
                        "direction": "reverse",
                        "from": "membership",
                        "to": "Martin Hug",
                        "from_role": "candidate_set",
                        "to_role": "anchor_a",
                        "grounding_source": "dynamic_probe",
                    },
                    {
                        "relation": "organization.organization_type.organizations_of_this_type",
                        "direction": "reverse",
                        "from": "organization",
                        "to": "type",
                        "from_role": "candidate_set",
                        "to_role": "shared_type",
                        "grounding_source": "dynamic_probe",
                    },
                    {
                        "relation": "type.object.type",
                        "direction": "forward",
                        "from": "sci",
                        "to": "type",
                        "from_role": "anchor_b",
                        "to_role": "shared_type",
                        "grounding_source": "curated",
                    },
                ],
                "projection": ["shared_answer", "shared_answer_name"],
            }
        ),
        relation_grounding=[
            {
                "relation": "organization.organization_board_membership.member",
                "direction": "reverse",
                "from": "membership",
                "to": "Martin Hug",
                "from_role": "candidate_set",
                "to_role": "anchor_a",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "organization.organization_board_membership.organization",
                "direction": "forward",
                "from": "pivot",
                "to": "organization",
                "from_role": "candidate_set",
                "to_role": "shared_answer",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_pivot_outgoing",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "shared_answer",
                "to": "type",
                "from_role": "shared_answer",
                "to_role": "shared_type",
                "grounding_source": "curated",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "sci",
                "to": "type",
                "from_role": "anchor_b",
                "to_role": "shared_type",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Martin Hug",
                entity_count=1,
                path_count=1,
                relation_probed="organization.organization_board_membership.member",
                anchor_position="object",
                resolved_entity_id="m.010f3n0k",
            ),
            AnchorProbeResult(
                anchor_name="sci",
                entity_count=1,
                path_count=9,
                relation_probed="type.object.type",
                anchor_position="subject",
                resolved_entity_id="m.04_754t",
            ),
        ],
    )

    assert repaired_plan is not None
    assert any(
        path.get("relation") == "organization.organization_board_membership.organization"
        and path.get("from") == "candidate_set"
        and path.get("to") == "shared_answer"
        for path in repaired_plan["relation_paths"]
    )
    assert any(
        path.get("relation") == "type.object.type"
        and path.get("from") == "shared_answer"
        and path.get("to") == "shared_type"
        for path in repaired_plan["relation_paths"]
    )
    assert not any(
        path.get("relation") == "organization.organization_type.organizations_of_this_type"
        for path in repaired_plan["relation_paths"]
    )


def test_probe_dynamic_relation_candidates_for_live_pivots_targets_shared_answer_for_entity_tasks() -> None:
    controller = _make_controller()
    captured: list[str] = []

    controller._sample_live_pivot_entity_ids = lambda **kwargs: [  # type: ignore[method-assign]
        {"anchor_role": "anchor_a", "pivot_entity_id": "m.010f3n0h", "pivot_relation": "organization.organization_board_membership.member"}
    ]

    def _fake_probe(**kwargs):
        captured.append(str(kwargs.get("target_role") or ""))
        return []

    controller._probe_dynamic_relation_candidates_for_entity_id = _fake_probe  # type: ignore[method-assign]

    result = controller._probe_dynamic_relation_candidates_for_live_pivots(
        query_plan={
            "query_shape": "shared_type_intersection",
            "answer_mode": "entity",
        },
        anchor_probe_results=[],
        answer_target_phrase="organization",
        domain_hints=[],
        question_text="which organization governed by martin hug is of the same type with sci?",
    )

    assert result == []
    assert captured == ["shared_answer"]


def test_probe_dynamic_relation_candidates_for_live_pivots_targets_answer_for_single_anchor_entity_tasks() -> None:
    controller = _make_controller()
    captured: list[str] = []

    controller._sample_live_pivot_entity_ids = lambda **kwargs: [  # type: ignore[method-assign]
        {
            "anchor_role": "anchor",
            "pivot_entity_id": "m.0t4c9w",
            "pivot_relation": "music.release.track",
        }
    ]

    def _fake_probe(**kwargs):
        captured.append(str(kwargs.get("target_role") or ""))
        return []

    controller._probe_dynamic_relation_candidates_for_entity_id = _fake_probe  # type: ignore[method-assign]

    result = controller._probe_dynamic_relation_candidates_for_live_pivots(
        query_plan={
            "query_shape": "single_anchor_lookup",
            "answer_mode": "entity",
        },
        anchor_probe_results=[],
        answer_target_phrase="featured artist",
        domain_hints=[],
        question_text="what is the name of the featured artist for musical recording which releases ibiza euphoria?",
    )

    assert result == []
    assert captured == ["answer"]


def test_structural_repair_adds_pivot_dynamic_candidates_for_single_anchor_projection_empty(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv("PAL_RUNTIME_SINGLE_ANCHOR_PIVOT_DYNAMIC", "1")
    controller._build_pal_grounding_card = lambda *args, **kwargs: "grounding-card"  # type: ignore[method-assign]
    controller._split_task_question = lambda task_question: (task_question, "")  # type: ignore[method-assign]
    controller._extract_answer_target_phrase = lambda question_text: "featured artist"  # type: ignore[method-assign]
    controller._infer_domain_hints = lambda task_question: []  # type: ignore[method-assign]
    controller._extract_anchor_alias_overrides = lambda query_plan: {}  # type: ignore[method-assign]
    controller._probe_dynamic_relation_candidates_for_live_pivots = lambda **kwargs: [  # type: ignore[method-assign]
        {
            "relation": "music.recording.featured_artists",
            "direction": "forward",
            "from": "pivot",
            "to": "featured_artist",
            "from_role": "candidate_set",
            "to_role": "answer",
            "grounding_source": "dynamic_probe",
            "support": "dynamic_probe_pivot_outgoing",
        }
    ]

    grounding_card, grounded_candidates, feedback = controller._augment_grounding_for_structural_repair(
        task_question="what is the name of the featured artist for musical recording which releases ibiza euphoria?",
        query_plan={
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "Ibiza Euphoria",
                    "chosen_alias": "m.03_9dcv",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "music.release.track",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "music.artist.track",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                },
            ],
        },
        relation_grounding=[
            {
                "relation": "music.release.track",
                "direction": "forward",
                "from": "Ibiza Euphoria",
                "to": "track",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Ibiza Euphoria",
                entity_count=1,
                path_count=36,
                relation_probed="music.release.track",
                anchor_position="subject",
                resolved_entity_id="m.03_9dcv",
            )
        ],
        verdict=PlausibilityVerdict(
            verdict="repairable_anchor_path_empty",
            reasons=[
                "grounded_single_anchor_empty_result",
                "anchor_paths_live_but_projection_empty",
            ],
        ),
    )

    assert grounding_card == "grounding-card"
    assert any(
        candidate.get("relation") == "music.recording.featured_artists"
        for candidate in grounded_candidates
    )
    assert any(
        "single_anchor_pivot_dynamic_grounding_augmented" in item
        for item in feedback
    )


def test_shared_type_pivot_bridge_can_synthesize_shared_answer_type_relation_from_live_anchor_family() -> None:
    controller = _make_controller()

    candidate = controller._select_shared_answer_type_candidate(
        relation_grounding=[
            {
                "relation": "organization.organization.organization_type",
                "direction": "forward",
                "from": "sci",
                "to": "type",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            }
        ]
    )

    assert candidate is not None
    assert candidate["relation"] == "organization.organization.organization_type"
    assert candidate["from"] == "shared_answer"
    assert candidate["to"] == "shared_type"
    assert candidate["from_role"] == "shared_answer"
    assert candidate["to_role"] == "shared_type"


def test_shared_type_pivot_bridge_prefers_anchor_type_family_over_generic_type_relation() -> None:
    controller = _make_controller()

    candidate = controller._select_shared_answer_type_candidate(
        relation_grounding=[
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "organization",
                "to": "type",
                "from_role": "shared_answer",
                "to_role": "shared_type",
                "grounding_source": "curated",
            },
            {
                "relation": "organization.organization.organization_type",
                "direction": "forward",
                "from": "sci",
                "to": "type",
                "from_role": "anchor_b",
                "to_role": "shared_type",
                "grounding_source": "dynamic_probe",
            },
        ],
        preferred_relation="organization.organization.organization_type",
    )

    assert candidate is not None
    assert candidate["relation"] == "organization.organization.organization_type"
    assert candidate["from"] == "shared_answer"
    assert candidate["to"] == "shared_type"


def test_structural_repair_does_not_rewrite_when_projection_is_not_distinct() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_projected_answer_intersection_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {"surface": "A", "chosen_alias": "A", "role": "anchor_a"},
                    {"surface": "B", "chosen_alias": "B", "role": "anchor_b"},
                ],
                "normalized_aliases": [],
                "shared_answer_variable": "answer_entity",
                "candidate_set_variable": "candidate_set",
                "count_set_variable": "",
                "join_structure": {
                    "type": "intersection",
                    "anchor_constraints": [
                        {"anchor_role": "anchor_a", "constrains_variable": "answer_entity", "notes": "a"},
                        {"anchor_role": "anchor_b", "constrains_variable": "answer_entity", "notes": "b"},
                    ],
                },
                "relation_paths": [
                    {
                        "relation": "r.one",
                        "direction": "forward",
                        "from": "A",
                        "to": "answer_entity",
                        "from_role": "anchor_a",
                        "to_role": "shared_answer",
                        "grounding_source": "curated",
                        "reason": "a",
                    },
                    {
                        "relation": "r.two",
                        "direction": "forward",
                        "from": "B",
                        "to": "answer_entity",
                        "from_role": "anchor_b",
                        "to_role": "shared_answer",
                        "grounding_source": "curated",
                        "reason": "b",
                    },
                ],
                "projection": ["answer_entity"],
                "allow_exploratory_predicates": False,
                "strategy": "already intersects on answer",
                "plan_rationale": ["no repair needed"],
            }
        )
    )

    assert repaired_plan is None


def test_plan_normalization_expands_generic_projection_for_projected_intersection() -> None:
    controller = _make_controller()

    plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "entity",
            "answer_mode": "entity",
            "query_shape": "multi_anchor_intersection",
            "anchored_entities": [
                {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "dosage_form",
            "candidate_set_variable": "formulation",
            "count_set_variable": "",
            "join_structure": {
                "type": "intersection",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "formulation_a", "notes": "a"},
                    {"anchor_role": "anchor_b", "constrains_variable": "formulation_b", "notes": "b"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "medicine.drug.marketed_formulations",
                    "direction": "forward",
                    "from": "anchor_a",
                    "to": "formulation_a",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                    "reason": "a",
                },
                {
                    "relation": "medicine.drug_formulation.active_ingredients",
                    "direction": "reverse",
                    "from": "formulation_b",
                    "to": "anchor_b",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                    "reason": "b",
                },
                {
                    "relation": "medicine.drug_formulation.dosage_form",
                    "direction": "forward",
                    "from": "formulation",
                    "to": "dosage_form",
                    "from_role": "shared_answer",
                    "to_role": "shared_answer",
                    "grounding_source": "curated",
                    "reason": "project dosage form",
                },
            ],
            "projection": ["dosage_form"],
            "allow_exploratory_predicates": False,
            "strategy": "projected answer intersection",
            "plan_rationale": ["project each branch to dosage_form"],
        }
    )

    dosage_paths = [
        item
        for item in plan["relation_paths"]
        if item.get("relation") == "medicine.drug_formulation.dosage_form"
    ]
    assert len(dosage_paths) == 2
    assert {item["from"] for item in dosage_paths} == {"formulation_a", "formulation_b"}
    assert all(item["to"] == "dosage_form" for item in dosage_paths)


def test_projected_intersection_repair_can_build_from_grounding_candidates() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_projected_answer_intersection_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                    {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
                ],
                "normalized_aliases": [],
                "shared_answer_variable": "formulation",
                "candidate_set_variable": "candidate_set",
                "count_set_variable": "",
                "join_structure": {
                    "type": "intersection",
                    "anchor_constraints": [
                        {"anchor_role": "anchor_a", "constrains_variable": "formulation", "notes": "a"},
                        {"anchor_role": "anchor_b", "constrains_variable": "formulation", "notes": "b"},
                    ],
                },
                "relation_paths": [
                    {
                        "relation": "medicine.drug_formulation.formulation_of",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "drug_or_ingredient",
                        "from_role": "shared_answer",
                        "to_role": "constraint_value",
                        "grounding_source": "curated",
                        "reason": "failed initial bridge",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                        "direction": "forward",
                        "from": "active_ingredient",
                        "to": "formulation",
                        "from_role": "constraint_value",
                        "to_role": "shared_answer",
                        "grounding_source": "curated",
                        "reason": "failed initial bridge",
                    },
                    {
                        "relation": "medicine.drug_formulation.dosage_form",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "dosage_form",
                        "from_role": "shared_answer",
                        "to_role": "shared_answer",
                        "grounding_source": "curated",
                        "reason": "project dosage form",
                    },
                ],
                "projection": ["dosage_form"],
                "allow_exploratory_predicates": False,
                "strategy": "failed formulation bridge",
                "plan_rationale": ["first attempt"],
            }
        ),
        relation_grounding=[
            {
                "relation": "medicine.drug.marketed_formulations",
                "direction": "forward",
                "from": "Naloxone",
                "to": "formulation",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "support": "curated_anchor_bridge_synthesized",
            },
            {
                "relation": "medicine.drug_ingredient.active_moiety_of_formulation",
                "direction": "forward",
                "from": "Enalaprilat",
                "to": "formulation",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
            {
                "relation": "medicine.drug_formulation.dosage_form",
                "direction": "forward",
                "from": "formulation",
                "to": "dosage_form",
                "from_role": "shared_answer",
                "to_role": "shared_answer",
                "grounding_source": "curated",
            },
        ],
    )

    assert repaired_plan is not None
    assert repaired_plan["shared_answer_variable"] == "dosage_form"
    assert any(
        item.get("relation") == "medicine.drug.marketed_formulations"
        and item.get("to") == "candidate_set_anchor_a"
        for item in repaired_plan["relation_paths"]
    )
    assert any(
        item.get("relation") == "medicine.drug_ingredient.active_moiety_of_formulation"
        and item.get("to") == "candidate_set_anchor_b"
        for item in repaired_plan["relation_paths"]
    )


def test_projected_intersection_repair_prefers_forward_anchor_bridge_candidates() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_projected_answer_intersection_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {
                        "surface": "Candesartan cilexetil",
                        "chosen_alias": "Candesartan cilexetil",
                        "role": "anchor_a",
                    },
                    {
                        "surface": "Formic acid",
                        "chosen_alias": "Formic acid",
                        "role": "anchor_b",
                    },
                ],
                "normalized_aliases": [],
                "shared_answer_variable": "formulation",
                "candidate_set_variable": "candidate_set",
                "count_set_variable": "",
                "join_structure": {
                    "type": "intersection",
                    "anchor_constraints": [
                        {"anchor_role": "anchor_a", "constrains_variable": "formulation", "notes": "a"},
                        {"anchor_role": "anchor_b", "constrains_variable": "formulation", "notes": "b"},
                    ],
                },
                "relation_paths": [
                    {
                        "relation": "medicine.drug_formulation.formulation_of",
                        "direction": "reverse",
                        "from": "formulation",
                        "to": "drug_or_ingredient",
                        "from_role": "candidate_set",
                        "to_role": "constraint_value",
                        "grounding_source": "curated",
                        "reason": "failed initial bridge a",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                        "direction": "forward",
                        "from": "Formic acid",
                        "to": "formulation",
                        "from_role": "anchor_b",
                        "to_role": "shared_answer",
                        "grounding_source": "curated",
                        "reason": "failed initial bridge b",
                    },
                    {
                        "relation": "medicine.drug_formulation.dosage_form",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "dosage_form",
                        "from_role": "shared_answer",
                        "to_role": "answer",
                        "grounding_source": "curated",
                        "reason": "project dosage form",
                    }
                ],
                "projection": ["dosage_form"],
                "allow_exploratory_predicates": False,
                "strategy": "project each branch to dosage_form",
                "plan_rationale": ["start with formulation bridge"],
            }
        ),
        relation_grounding=[
            {
                "relation": "medicine.drug.marketed_formulations",
                "direction": "forward",
                "from": "Candesartan cilexetil",
                "to": "formulation",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
            {
                "relation": "medicine.drug_formulation.active_ingredients",
                "direction": "reverse",
                "from": "formulation",
                "to": "Candesartan cilexetil",
                "from_role": "candidate_set",
                "to_role": "anchor_a",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            },
            {
                "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                "direction": "forward",
                "from": "Formic acid",
                "to": "formulation",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
                "support": "curated_anchor_bridge_synthesized",
            },
            {
                "relation": "medicine.drug_formulation.dosage_form",
                "direction": "forward",
                "from": "formulation",
                "to": "dosage_form",
                "from_role": "shared_answer",
                "to_role": "answer",
                "grounding_source": "curated",
            },
        ],
    )

    assert repaired_plan is not None
    assert any(
        path.get("relation") == "medicine.drug.marketed_formulations"
        and path.get("from_role") == "anchor_a"
        and path.get("to") == "candidate_set_anchor_a"
        for path in repaired_plan["relation_paths"]
    )
    assert not any(
        path.get("relation") == "medicine.drug_formulation.active_ingredients"
        and path.get("to_role") == "anchor_a"
        for path in repaired_plan["relation_paths"]
    )


def test_projected_intersection_repair_prefers_live_probed_anchor_relation_family() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_projected_answer_intersection_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {
                        "surface": "Candesartan cilexetil",
                        "chosen_alias": "Candesartan cilexetil",
                        "role": "anchor_a",
                    },
                    {
                        "surface": "Formic acid",
                        "chosen_alias": "Formic acid",
                        "role": "anchor_b",
                    },
                ],
                "shared_answer_variable": "formulation",
                "candidate_set_variable": "candidate_set",
                "relation_paths": [
                    {
                        "relation": "medicine.drug_formulation.formulation_of",
                        "direction": "reverse",
                        "from": "formulation",
                        "to": "anchor_a",
                        "from_role": "candidate_set",
                        "to_role": "anchor_a",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                        "direction": "forward",
                        "from": "anchor_b",
                        "to": "formulation",
                        "from_role": "anchor_b",
                        "to_role": "shared_answer",
                    },
                    {
                        "relation": "medicine.drug_formulation.dosage_form",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "dosage_form",
                        "from_role": "shared_answer",
                        "to_role": "answer",
                    },
                ],
                "projection": ["dosage_form"],
            }
        ),
        relation_grounding=[
            {
                "relation": "medicine.drug.marketed_formulations",
                "direction": "forward",
                "from": "Candesartan cilexetil",
                "to": "formulation",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
            },
            {
                "relation": "medicine.drug_formulation.active_ingredients",
                "direction": "reverse",
                "from": "formulation",
                "to": "Candesartan cilexetil",
                "from_role": "candidate_set",
                "to_role": "anchor_a",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            },
            {
                "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                "direction": "forward",
                "from": "Formic acid",
                "to": "formulation",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "medicine.drug_formulation.active_ingredient_moieties",
                "direction": "reverse",
                "from": "formulation",
                "to": "Formic acid",
                "from_role": "candidate_set",
                "to_role": "anchor_b",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            },
            {
                "relation": "medicine.drug_formulation.dosage_form",
                "direction": "forward",
                "from": "formulation",
                "to": "dosage_form",
                "from_role": "shared_answer",
                "to_role": "answer",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Candesartan cilexetil",
                entity_count=1,
                path_count=4,
                relation_probed="medicine.drug.marketed_formulations",
                anchor_position="subject",
            ),
            AnchorProbeResult(
                anchor_name="Formic acid",
                entity_count=1,
                path_count=17,
                relation_probed="medicine.drug_ingredient.active_ingredient_of_formulation",
                anchor_position="subject",
            ),
        ],
    )

    assert repaired_plan is not None
    assert any(
        path.get("relation") == "medicine.drug.marketed_formulations"
        and path.get("to") == "candidate_set_anchor_a"
        for path in repaired_plan["relation_paths"]
    )
    assert any(
        path.get("relation") == "medicine.drug_ingredient.active_ingredient_of_formulation"
        and path.get("to") == "candidate_set_anchor_b"
        for path in repaired_plan["relation_paths"]
    )
    assert not any(
        path.get("relation") == "medicine.drug_formulation.active_ingredients"
        for path in repaired_plan["relation_paths"]
    )


def test_projected_intersection_repair_preserves_existing_anchor_family_without_probe_override() -> None:
    controller = _make_controller()

    repaired_plan = controller._build_projected_answer_intersection_repair_plan(
        query_plan=controller._normalize_pal_query_plan(
            {
                "answer_type": "entity",
                "answer_mode": "entity",
                "query_shape": "multi_anchor_intersection",
                "anchored_entities": [
                    {
                        "surface": "Candesartan cilexetil",
                        "chosen_alias": "Candesartan cilexetil",
                        "role": "anchor_a",
                    },
                    {
                        "surface": "Formic acid",
                        "chosen_alias": "Formic acid",
                        "role": "anchor_b",
                    },
                ],
                "shared_answer_variable": "formulation",
                "candidate_set_variable": "candidate_set",
                "relation_paths": [
                    {
                        "relation": "medicine.drug.marketed_formulations",
                        "direction": "forward",
                        "from": "anchor_a",
                        "to": "formulation",
                        "from_role": "anchor_a",
                        "to_role": "shared_answer",
                    },
                    {
                        "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                        "direction": "forward",
                        "from": "anchor_b",
                        "to": "formulation",
                        "from_role": "anchor_b",
                        "to_role": "shared_answer",
                    },
                    {
                        "relation": "medicine.drug_formulation.dosage_form",
                        "direction": "forward",
                        "from": "formulation",
                        "to": "dosage_form",
                        "from_role": "shared_answer",
                        "to_role": "answer",
                    },
                ],
                "projection": ["dosage_form"],
            }
        ),
        relation_grounding=[
            {
                "relation": "medicine.drug.marketed_formulations",
                "direction": "forward",
                "from": "Candesartan cilexetil",
                "to": "formulation",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                "direction": "forward",
                "from": "Formic acid",
                "to": "formulation",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "medicine.drug_ingredient.active_moiety_of_formulation",
                "direction": "forward",
                "from": "Formic acid",
                "to": "formulation",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "medicine.drug_formulation.dosage_form",
                "direction": "forward",
                "from": "formulation",
                "to": "dosage_form",
                "from_role": "shared_answer",
                "to_role": "answer",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=None,
    )

    assert repaired_plan is not None
    assert any(
        path.get("relation") == "medicine.drug_ingredient.active_ingredient_of_formulation"
        and path.get("to") == "candidate_set_anchor_b"
        for path in repaired_plan["relation_paths"]
    )
    assert not any(
        path.get("relation") == "medicine.drug_ingredient.active_moiety_of_formulation"
        and path.get("to") == "candidate_set_anchor_b"
        for path in repaired_plan["relation_paths"]
    )


def test_plausibility_repairs_zero_count_for_weak_dynamic_count_chain() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Stafy", "chosen_alias": "Stafy", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.rank",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "fictional_universe.character_rank.characters_of_this_rank",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "fictional_universe.fictional_character.appears_in_these_fictional_universes",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": True,
            "strategy": "dynamic chain with extra inferred type filtering",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate) AS ?candidate_count) WHERE { "
            '?anchor fb:type.object.name "Stafy"@en . '
            "?anchor fb:fictional_universe.fictional_character.rank ?rank . "
            "?rank fb:fictional_universe.character_rank.characters_of_this_rank ?candidate . "
            "?candidate fb:fictional_universe.fictional_character.appears_in_these_fictional_universes ?universe . "
            "?universe fb:type.type.instance ?universe_type . "
            "} LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"candidate_count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Stafy"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_dynamic_chain_too_weak" in verdict.reasons


def test_grounded_empty_join_feedback_requests_scaffold_switch() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "allow_exploratory_predicates": False,
        "anchored_entities": [
            {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
            {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
        ],
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {"anchor_role": "anchor_a", "constrains_variable": "formulation"},
                {"anchor_role": "anchor_b", "constrains_variable": "formulation"},
            ],
        },
        "shared_answer_variable": "formulation",
        "candidate_set_variable": "formulation",
        "count_set_variable": "",
        "relation_paths": [
            {
                "relation": "medicine.drug_formulation.formulation_of",
                "direction": "forward",
                "from_role": "shared_answer",
                "to_role": "constraint_value",
                "from": "formulation",
                "to": "drug_or_ingredient",
                "grounding_source": "curated",
            },
            {
                "relation": "medicine.drug_formulation.active_ingredients",
                "direction": "forward",
                "from_role": "shared_answer",
                "to_role": "constraint_value",
                "from": "formulation",
                "to": "active_ingredient",
                "grounding_source": "curated",
            },
        ],
        "strategy": "intersect two grounded formulation constraints",
    }

    verdict = validate_pal_execution(
        query_plan=query_plan,
        query_text=(
            'SELECT ?answer WHERE { '
            '?naloxone fb:type.object.name "Naloxone"@en . '
            '?enalaprilat fb:type.object.name "Enalaprilat"@en . '
            '} LIMIT 50'
        ),
        result_dict={"results": {"bindings": []}},
        entities=["Naloxone", "Enalaprilat"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Naloxone",
                entity_count=1,
                path_count=7,
                relation_probed="medicine.drug_formulation.formulation_of",
                anchor_position="object",
            ),
            AnchorProbeResult(
                anchor_name="Enalaprilat",
                entity_count=1,
                path_count=2,
                relation_probed="medicine.drug_formulation.active_ingredients",
                anchor_position="object",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_GROUNDED_EMPTY
    assert "anchor_paths_live_but_join_overlap_empty" in verdict.reasons
    feedback = build_repair_feedback(verdict)
    assert any("change_scaffold_family_not_query_wording" in item for item in feedback)


def test_structural_repair_adds_pivot_friendly_candidates_for_dead_direct_count_family() -> None:
    controller = _make_controller()
    relation_grounding = controller._normalize_grounded_relation_candidates(
        relation_candidates=[
            {
                "relation": "fictional_universe.fictional_universe.species",
                "direction": "forward",
                "from": "fictional_world",
                "to": "species",
                "support": "curated_fictional_universe_predicate",
                "use_when": "find species in a fictional universe/world",
            },
            {
                "relation": "fictional_universe.fictional_setting.universe",
                "direction": "forward",
                "from": "Seventh sphere",
                "to": "?target",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_outgoing",
                "use_when": "pivot from a setting to its universe",
            },
        ],
        query_shape="count_over_direct_relation",
        answer_mode="count",
        answer_target_phrase="species",
        entities=["Seventh sphere"],
    )
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Seventh sphere",
                    "chosen_alias": "Seventh sphere",
                    "role": "anchor",
                }
            ],
            "normalized_aliases": [],
            "shared_answer_variable": "species",
            "candidate_set_variable": "species",
            "count_set_variable": "species",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "species",
                        "notes": "direct species count from the anchor",
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "fictional_world",
                    "to": "species",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                    "reason": "count species directly from the anchor",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "count species directly from the fictional world anchor",
            "plan_rationale": ["use direct curated species relation"],
        }
    )

    _, augmented_grounding, feedback = controller._augment_grounding_for_structural_repair(
        task_question="Question: what number of different species are in the fictional world of seventh sphere?, Entities: ['Seventh sphere']",
        query_plan=query_plan,
        relation_grounding=relation_grounding,
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Seventh sphere",
                entity_count=1,
                path_count=0,
                relation_probed="fictional_universe.fictional_universe.species",
                anchor_position="subject",
            )
        ],
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "anchor_path_empty:'Seventh sphere':fictional_universe.fictional_universe.species",
                "count_set_path_empty",
            ],
        ),
    )

    assert any(
        candidate.get("relation") == "fictional_universe.fictional_universe.species"
        and candidate.get("from_role") == "candidate_set"
        and candidate.get("to_role") == "count_set"
        for candidate in augmented_grounding
    )
    assert any("insert_pivot_before_reusing_curated_family" in item for item in feedback)


def test_structural_repair_marks_broad_type_overreach_scaffold_dead() -> None:
    controller = _make_controller()
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "answer_target_phrase": "meter",
            "anchored_entities": [
                {
                    "surface": "Free verse",
                    "chosen_alias": "Free verse",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from": "type",
                    "to": "Free verse",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                    "reason": "generic type lookup",
                }
            ],
            "projection": ["candidate_set"],
            "allow_exploratory_predicates": False,
            "strategy": "retrieve type entities for the anchor",
        }
    )
    relation_grounding = [
        {
            "relation": "type.type.instance",
            "direction": "reverse",
            "from": "type",
            "to": "Free verse",
            "from_role": "candidate_set",
            "to_role": "anchor",
            "grounding_source": "dynamic_probe",
        },
        {
            "relation": "book.poem.verse_form",
            "direction": "reverse",
            "from": "poem",
            "to": "Free verse",
            "from_role": "candidate_set",
            "to_role": "anchor",
            "grounding_source": "dynamic_probe",
        },
    ]

    _, _, feedback = controller._augment_grounding_for_structural_repair(
        task_question="Question: what type of meter is used in free verse?, Entities: ['Free verse']",
        query_plan=query_plan,
        relation_grounding=relation_grounding,
        anchor_probe_results=None,
        verdict=PlausibilityVerdict(
            verdict=VERDICT_REJECTED_DANGEROUS_OVERREACH,
            reasons=[
                "generic_type_result_overbroad_for_answer_target:meter",
                "dangerous_overreach:broad_type_expansion",
            ],
        ),
    )

    assert any(item.startswith("dead_scaffold_signature:") for item in feedback)
    assert any(item.startswith("failed_relation_family:") for item in feedback)
    assert any(item.startswith("unused_grounded_relations:") for item in feedback)


def test_dead_direct_anchor_suppression_keeps_pivot_friendly_variant() -> None:
    controller = _make_controller()
    relation_grounding = [
        {
            "relation": "fictional_universe.fictional_universe.species",
            "direction": "forward",
            "from": "fictional_world",
            "to": "species",
            "from_role": "anchor",
            "to_role": "count_set",
            "grounding_source": "curated",
        },
        {
            "relation": "fictional_universe.fictional_universe.species",
            "direction": "forward",
            "from": "pivot",
            "to": "species",
            "from_role": "candidate_set",
            "to_role": "count_set",
            "grounding_source": "curated",
        },
    ]

    _, filtered_grounding, _ = controller._suppress_dead_grounded_relations(
        task_question="Question: what number of different species are in the fictional world of seventh sphere?, Entities: ['Seventh sphere']",
        query_plan={},
        relation_grounding=relation_grounding,
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Seventh sphere",
                entity_count=1,
                path_count=0,
                relation_probed="fictional_universe.fictional_universe.species",
                anchor_position="subject",
            )
        ],
    )

    assert not any(
        candidate.get("relation") == "fictional_universe.fictional_universe.species"
        and candidate.get("from_role") == "anchor"
        for candidate in filtered_grounding
    )
    assert any(
        candidate.get("relation") == "fictional_universe.fictional_universe.species"
        and candidate.get("from_role") == "candidate_set"
        for candidate in filtered_grounding
    )


def test_count_repair_rewrites_to_preserved_target_family_after_pivot() -> None:
    controller = _make_controller()
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Seventh sphere",
                    "chosen_alias": "Seventh sphere",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "species",
            "candidate_set_variable": "species",
            "count_set_variable": "species",
            "ordering_attribute": {},
            "ordering_direction": "none",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "species",
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "fictional_world",
                    "to": "species",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
            "allow_exploratory_predicates": False,
            "strategy": "count species directly from the fictional world anchor",
            "plan_rationale": ["use direct curated species relation"],
        }
    )
    rewritten = controller._build_pivot_preserving_count_repair_plan(
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "fictional_universe.fictional_setting.universe",
                "direction": "forward",
                "from": "Seventh sphere",
                "to": "universe",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "fictional_universe.fictional_universe.species",
                "direction": "forward",
                "from": "pivot",
                "to": "species",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
            {
                "relation": "fictional_universe.fictional_universe.races",
                "direction": "forward",
                "from": "pivot",
                "to": "race",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Seventh sphere",
                entity_count=1,
                path_count=0,
                relation_probed="fictional_universe.fictional_universe.species",
                anchor_position="subject",
            )
        ],
    )

    assert rewritten is not None
    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "fictional_universe.fictional_setting.universe",
        "fictional_universe.fictional_universe.species",
    ]
    assert rewritten["count_set_variable"] == "species"
    assert "fictional_universe.fictional_universe.races" not in json.dumps(rewritten)


def test_anchor_existence_probe_prefers_full_count_chain_when_available() -> None:
    controller = _make_controller()
    controller._probe_entity_name_count = lambda alias, timeout_s=2.5: 1
    controller._run_probe_sparql_query = (
        lambda endpoint, sparql, timeout_s=2.5: ["45"]
        if "fictional_universe.fictional_setting.universe" in sparql
        and "fictional_universe.fictional_universe.species" in sparql
        else []
    )

    probe_results = controller._run_anchor_existence_probes(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "count_set_variable": "species",
            "anchored_entities": [
                {
                    "surface": "Seventh sphere",
                    "chosen_alias": "Seventh sphere",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.universe",
                    "direction": "forward",
                    "from": "Seventh sphere",
                    "to": "universe",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                },
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "universe",
                    "to": "species",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                },
            ],
        },
        probe_paths=True,
    )

    assert len(probe_results) == 1
    assert probe_results[0].path_count == 45


def test_pivot_preserving_count_repair_prefers_live_direct_dynamic_count_relation() -> None:
    controller = _make_controller()
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Aedes aegypti",
                    "chosen_alias": "Aedes aegypti",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_set",
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "candidate_set",
                    }
                ],
            },
            "relation_paths": [
                {
                    "relation": "biology.organism.diseases_transmitted",
                    "direction": "forward",
                    "from": "organism",
                    "to": "disease",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
        }
    )

    rewritten = controller._build_pivot_preserving_count_repair_plan(
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from": "disease",
                "to": "Aedes aegypti",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            },
            {
                "relation": "type.type.instance",
                "direction": "reverse",
                "from": "type",
                "to": "Aedes aegypti",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            },
            {
                "relation": "biology.organism.diseases_transmitted",
                "direction": "forward",
                "from": "pivot",
                "to": "disease",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Aedes aegypti",
                entity_count=12,
                path_count=0,
                relation_probed="biology.organism.diseases_transmitted",
                anchor_position="subject",
            )
        ],
    )

    assert rewritten is not None
    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "medicine.infectious_disease.vector"
    ]
    assert rewritten["count_set_variable"] == "disease"


def test_direct_count_repair_prefers_live_dynamic_family_before_curated_inverse_fallback() -> None:
    controller = _make_controller()
    query_plan = controller._normalize_pal_query_plan(
        {
            "answer_type": "count",
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Aedes aegypti",
                    "chosen_alias": "Aedes aegypti",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "disease",
            "count_set_variable": "disease",
            "join_structure": {"type": "count", "anchor_constraints": []},
            "relation_paths": [
                {
                    "relation": "biology.organism.diseases_transmitted",
                    "direction": "forward",
                    "from": "organism",
                    "to": "disease",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "projection": ["count"],
        }
    )

    rewritten = controller._build_direct_dynamic_count_repair_plan(
        query_plan=query_plan,
        relation_grounding=[
            {
                "relation": "medicine.infectious_disease.vector",
                "direction": "reverse",
                "from": "disease",
                "to": "Aedes aegypti",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
                "support": "dynamic_probe_incoming",
            },
            {
                "relation": "medicine.disease.transmitted_by",
                "direction": "forward",
                "from": "disease",
                "to": "transmitter",
                "from_role": "count_set",
                "to_role": "anchor",
                "grounding_source": "curated",
                "support": "curated_medicine_disease_predicate",
            },
        ],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Aedes aegypti",
                entity_count=12,
                path_count=0,
                relation_probed="biology.organism.diseases_transmitted",
                anchor_position="subject",
            )
        ],
    )

    assert rewritten is not None
    assert [path["relation"] for path in rewritten["relation_paths"]] == [
        "medicine.infectious_disease.vector"
    ]
    assert rewritten["relation_paths"][0]["from_role"] == "count_set"
    assert rewritten["relation_paths"][0]["to_role"] == "anchor"


def test_structural_repair_emits_anchor_clues_for_scaffold_switch() -> None:
    controller = _make_controller()
    feedback = controller._build_anchor_clue_feedback(
        "Question: what dug dosage form exist for drugs formulated from naloxone and has active ingredient enalaprilat?, Entities: ['Naloxone', 'Enalaprilat']",
        {
            "anchored_entities": [
                {"surface": "Naloxone", "chosen_alias": "Naloxone", "role": "anchor_a"},
                {"surface": "Enalaprilat", "chosen_alias": "Enalaprilat", "role": "anchor_b"},
            ]
        },
        relation_grounding=[
            {
                "relation": "medicine.drug_formulation.formulation_of",
                "use_when": "anchor a formulation to the drug or ingredient it formulates",
                "from": "formulation",
                "to": "drug_or_ingredient",
            },
            {
                "relation": "medicine.drug_formulation.active_ingredients",
                "use_when": "bind a formulation by active ingredient",
                "from": "formulation",
                "to": "active_ingredient",
            },
        ],
    )

    assert "anchor_clue:Naloxone=formulation_input" in feedback
    assert "anchor_clue:Enalaprilat=active_ingredient" in feedback
    assert any(
        item.startswith("anchor_preferred_relations:Naloxone=")
        for item in feedback
    )
    assert any(
        item.startswith("anchor_preferred_relations:Enalaprilat=")
        for item in feedback
    )
    assert any("preserve_anchor_semantics" in item for item in feedback)


def test_anchor_clue_feedback_prefers_anchor_specific_bridge_candidates() -> None:
    controller = _make_controller()

    preferred = controller._preferred_relations_for_anchor_clue(
        clue="active_ingredient",
        anchor_role="anchor_b",
        relation_grounding=[
            {
                "relation": "medicine.drug_formulation.active_ingredients",
                "from": "formulation",
                "to": "active_ingredient",
                "from_role": "shared_answer",
                "to_role": "constraint_value",
            },
            {
                "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                "from": "Enalaprilat",
                "to": "formulation",
                "from_role": "anchor_b",
                "to_role": "candidate_set",
                "support": "curated_anchor_bridge_synthesized",
            },
        ],
    )

    assert preferred[0] == "medicine.drug_ingredient.active_ingredient_of_formulation"


def test_plausibility_repairs_zero_count_for_mixed_dynamic_exploratory_chain() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "single_anchor_chain_lookup",
            "anchored_entities": [
                {"surface": "Goro", "chosen_alias": "Goro", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count",
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_character.rank",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "anchor_value",
                    "grounding_source": "exploratory",
                },
                {
                    "relation": "fictional_universe.character_rank.characters_of_this_rank",
                    "direction": "forward",
                    "from_role": "anchor_value",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": True,
            "strategy": "mixed exploratory and dynamic count chain",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate) AS ?count) WHERE { "
            '?g fb:type.object.name "Goro"@en . '
            "?g fb:fictional_universe.fictional_character.rank ?rank . "
            "?rank fb:fictional_universe.character_rank.characters_of_this_rank ?candidate . "
            "?candidate fb:type.type.instance ?book_type . "
            '?book_type fb:type.object.name "book character"@en . '
            "} LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Goro"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_dynamic_chain_too_weak" in verdict.reasons
    assert "count_query_unverified_type_constraint" in verdict.reasons


def test_plausibility_repairs_zero_count_for_dynamic_type_filtered_count() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Goro", "chosen_alias": "Goro", "role": "anchor"}
            ],
            "candidate_set_variable": "candidate_character",
            "count_set_variable": "candidate_character",
            "relation_paths": [
                {
                    "relation": "fictional_universe.character_rank.characters_of_this_rank",
                    "direction": "reverse",
                    "from": "candidate_character",
                    "to": "anchor",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from": "book character",
                    "to": "candidate_character",
                    "from_role": "type_set",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count rank-matched characters filtered by a dynamic type constraint",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_character) AS ?count) WHERE { "
            '?goro fb:type.object.name "Goro"@en . '
            "?candidate_character fb:fictional_universe.character_rank.characters_of_this_rank ?goro . "
            '?type fb:type.object.name "book character"@en . '
            "?type fb:type.type.instance ?candidate_character . "
            "} LIMIT 50"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Goro"],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_unverified_type_constraint" in verdict.reasons


def test_plausibility_repairs_zero_count_for_unplanned_dynamic_type_filter() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Atherurus africanus",
                    "chosen_alias": "Atherurus africanus",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "relation_paths": [
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "reverse",
                    "from": "disease",
                    "to": "Atherurus africanus",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "single dynamic count path with an auxiliary type filter added in query code",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            '?anchor fb:type.object.name "Atherurus africanus"@en . '
            "?candidate_set fb:medicine.infectious_disease.vector ?anchor . "
            "?candidate_set fb:type.object.type ?type . "
            '?type fb:type.object.name "infectious disease"@en . '
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Atherurus africanus"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Atherurus africanus",
                entity_count=1,
                path_count=1,
                relation_probed="medicine.infectious_disease.vector",
                anchor_position="object",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_unplanned_type_constraint" in verdict.reasons


def test_plausibility_repairs_ambiguous_live_count_without_resolved_anchor_binding() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Southern Min", "chosen_alias": "Southern Min", "role": "anchor"}
            ],
            "candidate_set_variable": "dialect",
            "count_set_variable": "dialect",
            "relation_paths": [
                {
                    "relation": "language.language_dialect.language",
                    "direction": "reverse",
                    "from": "dialect",
                    "to": "anchor",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count dialects for one language anchor",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?dialect) AS ?count) WHERE { "
            '?language fb:type.object.name "Southern Min"@en . '
            "?dialect fb:language.language_dialect.language ?language . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Southern Min"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Southern Min",
                entity_count=3,
                path_count=8,
                relation_probed="language.language_dialect.language",
                anchor_position="object",
                resolved_entity_id="m.01c44b",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "repair:bind_anchor_to_resolved_entity_id_before_accepting_result" in verdict.reasons


def test_plausibility_repairs_low_support_dynamic_count_even_after_resolved_id_pin() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Aedes aegypti",
                    "chosen_alias": "Aedes aegypti",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "disease",
            "count_set_variable": "disease",
            "relation_paths": [
                {
                    "relation": "medicine.infectious_disease.vector",
                    "direction": "reverse",
                    "from": "disease",
                    "to": "anchor",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count disease entities via a dynamic probe relation after curated paths failed",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?disease) AS ?count) WHERE { "
            "?disease fb:medicine.infectious_disease.vector fb:m.06y6_w . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "1"}}
                ]
            }
        },
        entities=["Aedes aegypti"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Aedes aegypti",
                entity_count=12,
                path_count=1,
                relation_probed="medicine.infectious_disease.vector",
                anchor_position="object",
                resolved_entity_id="m.06y6_w",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_dynamic_chain_too_weak" in verdict.reasons
    assert (
        "repair:verify_resolved_anchor_entity_and_relation_family_before_accepting_low_support_count"
        in verdict.reasons
    )


def test_plausibility_accepts_pinned_semantic_dynamic_count_with_grounded_bridge() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "key designers",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Richard Altwasser",
                    "chosen_alias": "m.050yww4",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "computer",
            "count_set_variable": "designer",
            "relation_paths": [
                {
                    "relation": "computer.computer_designer.computers_designed",
                    "direction": "forward",
                    "from": "Richard Altwasser",
                    "to": "computer",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "computer.computer.key_designers",
                    "direction": "forward",
                    "from": "computer",
                    "to": "key_designer",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count key designers by traversing from the resolved designer to designed computers and then to their key designers",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?designer) AS ?count) WHERE { "
            "VALUES ?anchor { fb:m.050yww4 } "
            "?anchor fb:computer.computer_designer.computers_designed ?computer . "
            "?computer fb:computer.computer.key_designers ?designer . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "1"}}
                ]
            }
        },
        entities=["Richard Altwasser"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Richard Altwasser",
                entity_count=2,
                path_count=1,
                relation_probed="computer.computer_designer.computers_designed",
                anchor_position="subject",
                resolved_entity_id="m.050yww4",
            )
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "accepted_pinned_semantic_dynamic_count" in verdict.reasons
    assert "count_query_dynamic_chain_too_weak" not in verdict.reasons


def test_plausibility_accepts_pinned_semantic_dynamic_joined_count_with_structural_count_set() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "key designers",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Richard Altwasser",
                    "chosen_alias": "Richard Altwasser",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "computer",
            "count_set_variable": "key_designer",
            "shared_answer_variable": "key_designer",
            "relation_paths": [
                {
                    "relation": "computer.computer_designer.computers_designed",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "computer",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "computer.computer.key_designers",
                    "direction": "forward",
                    "from": "computer",
                    "to": "key_designer",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count grounded key designers from designed computers using the resolved designer entity",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?key_designer) AS ?count) WHERE { "
            "VALUES ?anchor { fb:m.050yww4 } "
            "?anchor fb:computer.computer_designer.computers_designed ?computer . "
            "?computer fb:computer.computer.key_designers ?key_designer . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "1"}}
                ]
            }
        },
        entities=["Richard Altwasser"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Richard Altwasser",
                entity_count=2,
                path_count=1,
                relation_probed="computer.computer_designer.computers_designed",
                anchor_position="subject",
                resolved_entity_id="m.050yww4",
            )
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "accepted_pinned_semantic_dynamic_count" in verdict.reasons
    assert "count_query_dynamic_chain_too_weak" not in verdict.reasons


def test_plausibility_repairs_exact_grounded_zero_joined_count_without_direct_anchor_constraints() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "contents about higher education",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Higher Education",
                    "chosen_alias": "Higher Education",
                    "role": "anchor_a",
                },
                {
                    "surface": "To the Best of Our Knowledge",
                    "chosen_alias": "To the Best of Our Knowledge",
                    "role": "anchor_b",
                },
            ],
            "candidate_set_variable": "content",
            "count_set_variable": "content",
            "shared_answer_variable": "content",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "content",
                        "notes": "genre filter",
                    },
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "content",
                        "notes": "producer-equality filter",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "broadcast.genre.content",
                    "direction": "reverse",
                    "from": "genre",
                    "to": "content",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "content",
                    "to": "producer",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "content",
                    "to": "producer",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count higher-education contents produced by the producer of the anchored program",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?content) AS ?count) WHERE { "
            "VALUES ?genre { fb:m.03bv2kt } "
            "VALUES ?anchor { fb:m.03fx9_c } "
            "?genre fb:broadcast.genre.content ?content . "
            "?anchor fb:broadcast.content.producer ?producer . "
            "?content fb:broadcast.content.producer ?producer . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Higher Education", "To the Best of Our Knowledge"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Higher Education",
                entity_count=1,
                path_count=32,
                relation_probed="broadcast.genre.content",
                anchor_position="subject",
                resolved_entity_id="m.03bv2kt",
            ),
            AnchorProbeResult(
                anchor_name="To the Best of Our Knowledge",
                entity_count=1,
                path_count=2,
                relation_probed="broadcast.content.producer",
                anchor_position="subject",
                resolved_entity_id="m.03fx9_c",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "accepted_exact_grounded_zero_joined_count" not in verdict.reasons
    assert "count_query_zero_with_live_anchor_paths" in verdict.reasons


def test_plausibility_repairs_dynamic_zero_joined_count_when_constraint_side_is_not_fully_curated() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "contents about higher education",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Higher Education",
                    "chosen_alias": "m.03bv2kt",
                    "role": "anchor_a",
                },
                {
                    "surface": "To the Best of Our Knowledge",
                    "chosen_alias": "m.03fx9_c",
                    "role": "anchor_b",
                },
            ],
            "candidate_set_variable": "content",
            "count_set_variable": "content",
            "shared_answer_variable": "content",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor_b",
                        "constrains_variable": "producer",
                        "notes": "anchor_b -> producer pivot",
                    },
                    {
                        "anchor_role": "candidate_set",
                        "constrains_variable": "producer",
                        "notes": "producer -> produced content",
                    },
                    {
                        "anchor_role": "anchor_a",
                        "constrains_variable": "content",
                        "notes": "content genre/topic == Higher Education",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "To the Best of Our Knowledge",
                    "to": "producer",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "broadcast.producer.produces",
                    "direction": "forward",
                    "from": "producer",
                    "to": "content",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "broadcast.content.genre",
                    "direction": "reverse",
                    "from": "content",
                    "to": "Higher Education",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count contents about higher education produced by the producer of the anchored program",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?content) AS ?count) WHERE { "
            "fb:m.03fx9_c fb:broadcast.content.producer ?producer . "
            "?producer fb:broadcast.producer.produces ?content . "
            "?content fb:broadcast.content.genre fb:m.03bv2kt . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Higher Education", "To the Best of Our Knowledge"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Higher Education",
                entity_count=1,
                path_count=32,
                relation_probed="broadcast.content.genre",
                anchor_position="object",
                resolved_entity_id="m.03bv2kt",
            ),
            AnchorProbeResult(
                anchor_name="To the Best of Our Knowledge",
                entity_count=1,
                path_count=2,
                relation_probed="broadcast.content.producer",
                anchor_position="subject",
                resolved_entity_id="m.03fx9_c",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "accepted_exact_grounded_zero_joined_count" not in verdict.reasons
    assert "count_query_zero_without_grounded_join_constraints" in verdict.reasons


def test_plausibility_repairs_pseudo_joined_zero_count_without_distinct_anchor_constraints() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "teams",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Sample Athlete",
                    "chosen_alias": "Sample Athlete",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "roster",
            "count_set_variable": "team",
            "shared_answer_variable": "team",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {
                        "anchor_role": "anchor",
                        "constrains_variable": "roster",
                        "notes": "anchor -> roster",
                    },
                    {
                        "anchor_role": "candidate_set",
                        "constrains_variable": "team",
                        "notes": "roster -> team projection",
                    },
                ],
            },
            "relation_paths": [
                {
                    "relation": "sports.pro_athlete.teams",
                    "direction": "forward",
                    "from": "athlete",
                    "to": "roster",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "sports.sports_team_roster.team",
                    "direction": "forward",
                    "from": "roster",
                    "to": "team",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count projected teams from one athlete",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?team) AS ?count) WHERE { "
            "fb:m.sample_athlete fb:sports.pro_athlete.teams ?roster . "
            "?roster fb:sports.sports_team_roster.team ?team . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Sample Athlete"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Sample Athlete",
                entity_count=1,
                path_count=2,
                relation_probed="sports.pro_athlete.teams",
                anchor_position="subject",
                resolved_entity_id="m.sample_athlete",
            ),
            AnchorProbeResult(
                anchor_name="Sample Athlete",
                entity_count=1,
                path_count=2,
                relation_probed="sports.pro_athlete.teams -> sports.sports_team_roster.team",
                anchor_position="subject",
                resolved_entity_id="m.sample_athlete",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "accepted_exact_grounded_zero_joined_count" not in verdict.reasons


def test_plausibility_rejects_dynamic_entity_lookup_without_answer_target_semantics() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "answer_target_phrase": "wine producer",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {
                    "surface": "John Jordan",
                    "chosen_alias": "John Jordan",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "relation_paths": [
                {
                    "relation": "business.board_member.leader_of",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "SELECT DISTINCT ?candidate_set WHERE { "
            '?anchor fb:type.object.name "John Jordan"@en . '
            "?anchor fb:business.board_member.leader_of ?candidate_set . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.011lk5zn",
                        }
                    }
                ]
            }
        },
        entities=["John Jordan"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="John Jordan",
                entity_count=1,
                path_count=1,
                relation_probed="business.board_member.leader_of",
                anchor_position="subject",
                resolved_entity_id="m.0114n4sc",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "entity_answer_target_unenforced:wine producer" in verdict.reasons


def test_plausibility_repairs_generic_type_only_zero_count_with_optional_anchor_probe() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "research project",
                    "chosen_alias": "research project",
                    "role": "type_set",
                }
            ],
            "candidate_set_variable": "center",
            "count_set_variable": "center",
            "relation_paths": [
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from_role": "type_set",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count candidate set using only generic type relations",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?center) AS ?count) WHERE { "
            "?center fb:type.object.type ?t . "
            "?t fb:type.type.instance ?center . "
            'FILTER(LCASE(STR(?label)) = "research project") '
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["research project"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="research project",
                entity_count=0,
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "generic_type_only_zero_count_plan" in verdict.reasons


def test_plausibility_repairs_zero_joined_count_with_live_anchor_paths() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {"surface": "Canada", "chosen_alias": "Canada", "role": "anchor_a"},
                {
                    "surface": "Bull Terrier",
                    "chosen_alias": "Bull Terrier",
                    "role": "anchor_b",
                },
            ],
            "shared_answer_variable": "breed",
            "candidate_set_variable": "breed",
            "count_set_variable": "breed",
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "breed"},
                    {"anchor_role": "anchor_b", "constrains_variable": "temperament"},
                ],
            },
            "relation_paths": [
                {
                    "relation": "biology.breed_origin.breeds_originating_here",
                    "direction": "reverse",
                    "from": "country",
                    "to": "breed",
                    "from_role": "anchor_a",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "Bull Terrier",
                    "to": "temperament",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "reverse",
                    "from": "temperament",
                    "to": "breed",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count candidate breeds constrained by country and shared temperament",
        },
        query_text=(
            "SELECT (COUNT(DISTINCT ?breed) AS ?count) WHERE { "
            '?country fb:type.object.name "Canada"@en . '
            '?bull fb:type.object.name "Bull Terrier"@en . '
            "?bull fb:biology.animal_breed.temperament ?temp . "
            "?country fb:biology.breed_origin.breeds_originating_here ?breed . "
            "?breed fb:biology.animal_breed.temperament ?temp . "
            "}"
        ),
        result_dict={
            "results": {
                "bindings": [
                    {"count": {"type": "literal", "value": "0"}}
                ]
            }
        },
        entities=["Canada", "Bull Terrier"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Canada",
                entity_count=10,
                path_count=190,
                relation_probed="biology.breed_origin.breeds_originating_here",
                anchor_position="subject",
            ),
            AnchorProbeResult(
                anchor_name="Bull Terrier",
                entity_count=8,
                path_count=40,
                relation_probed="biology.animal_breed.temperament",
                anchor_position="subject",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_zero_with_live_anchor_paths" in verdict.reasons


def test_bad_count_set_feedback_adds_type_constraint_repair_hints() -> None:
    feedback = build_repair_feedback(
        PlausibilityVerdict(
            verdict=VERDICT_REPAIRABLE_BAD_COUNT_SET,
            reasons=[
                "count_query_dynamic_chain_too_weak",
                "count_query_unverified_type_constraint",
            ],
        )
    )

    assert "repair_hint:treat_question_category_as_type_constraint — bind the category/type phrase as a separate type node instead of folding it into the anchor relation" in feedback
    assert "repair_hint:prefer_candidate_to_type_binding — if the candidate set is already bound, prefer ?candidate fb:type.object.type ?type over starting from an unverified type node" in feedback


def test_validate_pal_query_candidate_rejects_unbound_projected_entity_variable() -> None:
    controller = _make_controller()

    errors = controller._validate_pal_query_candidate(
        raw_output="###QUERY_START\n###QUERY_END",
        generated_code=(
            "###QUERY_START\nfrom SPARQLWrapper import SPARQLWrapper, JSON\n\n"
            "def solve(endpoint_url):\n"
            "    return {}\n###QUERY_END"
        ),
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT DISTINCT ?shared_answer ?shared_answer_name WHERE { "
            "fb:m.03qzy4 fb:type.object.type ?type_a . "
            "?type_a fb:type.type.instance ?candidate_set . "
            "OPTIONAL { ?shared_answer fb:type.object.name ?shared_answer_name . } "
            "}"
        ),
        query_plan={
            "answer_mode": "entity",
            "query_shape": "shared_type_intersection",
            "projection": ["shared_answer", "shared_answer_name"],
            "relation_paths": [
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from_role": "anchor_a",
                    "to_role": "type_set",
                },
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "type_set",
                    "to_role": "candidate_set",
                },
            ],
        },
    )

    assert "projected_entity_not_structurally_bound:shared_answer" in errors


def test_validate_pal_query_candidate_rejects_ungrounded_type_instance_subject() -> None:
    controller = _make_controller()

    errors = controller._validate_pal_query_candidate(
        raw_output="###QUERY_START\n###QUERY_END",
        generated_code=(
            "###QUERY_START\nfrom SPARQLWrapper import SPARQLWrapper, JSON\n\n"
            "def solve(endpoint_url):\n"
            "    return {}\n###QUERY_END"
        ),
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?type fb:type.type.instance ?candidate_set . }"
        ),
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "projection": ["count"],
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "type_set",
                    "to_role": "candidate_set",
                }
            ],
        },
    )

    assert "ungrounded_type_set_variable:type" in errors


def test_validate_pal_execution_rejects_positive_count_for_ungrounded_generic_type_plan() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "research project cancer centers",
                    "chosen_alias": "research project cancer center",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "forward",
                    "from_role": "type_set",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?type fb:type.type.instance ?candidate_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "1303"}}]},
        },
        entities=["research project cancer centers"],
        anchor_probe_results=[AnchorProbeResult(anchor_name="research project cancer center", entity_count=0)],
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "generic_type_only_ungrounded_positive_count" in verdict.reasons


def test_validate_pal_execution_rejects_count_when_answer_target_semantics_are_unenforced() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "game expansions",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "valve corp",
                    "chosen_alias": "valve corp",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "cvg.game_version.publisher",
                    "direction": "reverse",
                    "from": "version",
                    "to": "valve corp",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?candidate_set fb:cvg.game_version.publisher ?anchor . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "35"}}]},
        },
        entities=["valve corp"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="valve corp",
                entity_count=1,
                path_count=35,
                relation_probed="cvg.game_version.publisher",
                anchor_position="object",
                resolved_entity_id="m.0dwl2",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "count_answer_target_unenforced:game expansions" in verdict.reasons


def test_validate_pal_execution_accepts_profession_membership_count_with_profession_like_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "songwriters",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Percussionist",
                    "chosen_alias": "m.02h66l4",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "people.profession.people_with_this_profession",
                    "direction": "forward",
                    "from": "profession",
                    "to": "person",
                    "from_role": "anchor",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                }
            ],
            "candidate_set_variable": "person",
            "count_set_variable": "person",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?person) AS ?count) WHERE { "
            "fb:m.02h66l4 fb:people.profession.people_with_this_profession ?person . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "921"}}]},
        },
        entities=["Percussionist"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Percussionist",
                entity_count=1,
                path_count=921,
                relation_probed="people.profession.people_with_this_profession",
                anchor_position="subject",
                resolved_entity_id="m.02h66l4",
            )
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "count_answer_target_unenforced:songwriters" not in verdict.reasons


def test_validate_pal_execution_accepts_prepositional_count_target_when_head_is_enforced() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "contents about higher education",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Higher Education",
                    "chosen_alias": "m.03bv2kt",
                    "role": "anchor_a",
                },
                {
                    "surface": "To the Best of Our Knowledge",
                    "chosen_alias": "m.03fx9_c",
                    "role": "anchor_b",
                },
            ],
            "relation_paths": [
                {
                    "relation": "broadcast.content.genre",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "anchor_a",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "anchor_b",
                    "to": "producer",
                    "from_role": "anchor_b",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "broadcast.content.producer",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "producer",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "candidate_set", "notes": "genre"},
                    {"anchor_role": "anchor_b", "constrains_variable": "candidate_set", "notes": "producer"},
                ],
            },
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "?candidate_set fb:broadcast.content.genre fb:m.03bv2kt . "
            "fb:m.03fx9_c fb:broadcast.content.producer ?producer . "
            "?candidate_set fb:broadcast.content.producer ?producer . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
        },
        entities=["Higher Education", "To the Best of Our Knowledge"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Higher Education",
                entity_count=1,
                path_count=32,
                relation_probed="broadcast.content.genre",
                anchor_position="object",
                resolved_entity_id="m.03bv2kt",
            ),
            AnchorProbeResult(
                anchor_name="To the Best of Our Knowledge",
                entity_count=1,
                path_count=2,
                relation_probed="broadcast.content.producer",
                anchor_position="subject",
                resolved_entity_id="m.03fx9_c",
            ),
        ],
    )

    assert "count_answer_target_unenforced:contents about higher education" not in verdict.reasons


def test_validate_pal_execution_accepts_joined_count_when_shared_filter_counts_grounded_candidate_set() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "breeds",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Serbia",
                    "chosen_alias": "m.077qn",
                    "role": "anchor_a",
                },
                {
                    "surface": "Smooth Fox Terrier",
                    "chosen_alias": "m.03_vlr",
                    "role": "anchor_b",
                },
            ],
            "relation_paths": [
                {
                    "relation": "biology.breed_origin.breeds_originating_here",
                    "direction": "reverse",
                    "from": "country",
                    "to": "breed",
                    "from_role": "constraint_value",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "breed",
                    "to": "temperament",
                    "from_role": "candidate_set",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
                {
                    "relation": "biology.animal_breed.temperament",
                    "direction": "forward",
                    "from": "breed",
                    "to": "temperament",
                    "from_role": "anchor_b",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                },
            ],
            "candidate_set_variable": "breed",
            "count_set_variable": "breed",
            "shared_answer_variable": "breed",
            "allow_exploratory_predicates": False,
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "breed", "notes": "origin"},
                    {"anchor_role": "anchor_b", "constrains_variable": "breed", "notes": "temperament"},
                ],
            },
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?breed) AS ?count) WHERE { "
            "VALUES ?anchor_b { fb:m.03_vlr } "
            "VALUES ?anchor_a { fb:m.077qn } "
            "?anchor_a fb:biology.breed_origin.breeds_originating_here ?breed . "
            "?breed fb:biology.animal_breed.temperament ?temperament . "
            "?anchor_b fb:biology.animal_breed.temperament ?temperament . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
        },
        entities=["Serbia", "Smooth Fox Terrier"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Serbia",
                entity_count=1,
                path_count=1,
                relation_probed="biology.breed_origin.breeds_originating_here",
                anchor_position="subject",
                resolved_entity_id="m.077qn",
            ),
            AnchorProbeResult(
                anchor_name="Smooth Fox Terrier",
                entity_count=1,
                path_count=1,
                relation_probed="biology.animal_breed.temperament",
                anchor_position="subject",
                resolved_entity_id="m.03_vlr",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "count_query_detached_joined_count_set:breed" not in verdict.reasons


def test_validate_pal_execution_accepts_joined_count_without_shared_filter_when_count_alias_is_detached() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "cayman islands government positions",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "her majesty the queen",
                    "chosen_alias": "m.0cq74",
                    "role": "anchor_a",
                },
                {
                    "surface": "Cayman Islands",
                    "chosen_alias": "m.0g6g2m",
                    "role": "anchor_b",
                },
            ],
            "relation_paths": [
                {
                    "relation": "government.government_position_held.appointed_by",
                    "direction": "reverse",
                    "from": "held",
                    "to": "her majesty the queen",
                    "from_role": "candidate_set",
                    "to_role": "anchor_a",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "government.government_position_held.jurisdiction_of_office",
                    "direction": "reverse",
                    "from": "jurisdiction",
                    "to": "Cayman Islands",
                    "from_role": "candidate_set",
                    "to_role": "anchor_b",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "candidate_set_variable": "candidate_positions",
            "count_set_variable": "count_positions",
            "shared_answer_variable": "shared_answer",
            "allow_exploratory_predicates": False,
            "join_structure": {
                "type": "count",
                "anchor_constraints": [
                    {"anchor_role": "anchor_a", "constrains_variable": "candidate_positions", "notes": "appointing monarch"},
                    {"anchor_role": "anchor_b", "constrains_variable": "candidate_positions", "notes": "jurisdiction"},
                ],
            },
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_positions) AS ?count) WHERE { "
            "?candidate_positions fb:government.government_position_held.appointed_by fb:m.0cq74 . "
            "?candidate_positions fb:government.government_position_held.jurisdiction_of_office fb:m.0g6g2m . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "1"}}]},
        },
        entities=["her majesty the queen", "Cayman Islands"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="her majesty the queen",
                entity_count=1,
                path_count=1,
                relation_probed="government.government_position_held.appointed_by",
                anchor_position="object",
                resolved_entity_id="m.0cq74",
            ),
            AnchorProbeResult(
                anchor_name="Cayman Islands",
                entity_count=1,
                path_count=1,
                relation_probed="government.government_position_held.jurisdiction_of_office",
                anchor_position="object",
                resolved_entity_id="m.0g6g2m",
            ),
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "count_query_detached_joined_count_set:count_positions" not in verdict.reasons


def test_validate_pal_execution_accepts_species_count_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "different species",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Seventh sphere",
                    "chosen_alias": "m.0cb9qd6",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.universe",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "count_set",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?count_set) AS ?count) WHERE { "
            "fb:m.0cb9qd6 fb:fictional_universe.fictional_setting.universe ?candidate_set . "
            "?candidate_set fb:fictional_universe.fictional_universe.species ?count_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "45"}}]},
        },
        entities=["Seventh sphere"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Seventh sphere",
                entity_count=1,
                path_count=45,
                relation_probed=(
                    "fictional_universe.fictional_setting.universe -> "
                    "fictional_universe.fictional_universe.species"
                ),
                anchor_position="subject",
                resolved_entity_id="m.0cb9qd6",
            )
        ],
    )

    assert "count_answer_target_unenforced:different species" not in verdict.reasons
    assert verdict.verdict != VERDICT_REJECTED_DANGEROUS_OVERREACH


def test_validate_pal_execution_rejects_count_target_dropped_to_relation_hint() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "exhibition subjects",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "international exhibition of modern art",
                    "chosen_alias": "m.01_ggr",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "exhibitions.exhibit.exhibitions_displayed_in",
                    "direction": "reverse",
                    "from": "exhibit",
                    "to": "international exhibition of modern art",
                    "from_role": "count_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "exhibit",
            "count_set_variable": "exhibit",
            "allow_exploratory_predicates": False,
            "strategy": (
                "Count the distinct exhibit reached directly from the anchor via "
                "exhibitions.exhibit.exhibitions_displayed_in. Treat the answer target "
                "only as a relation-selection hint."
            ),
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?exhibit) AS ?count) WHERE { "
            "?exhibit fb:exhibitions.exhibit.exhibitions_displayed_in fb:m.01_ggr . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "37"}}]},
        },
        entities=["international exhibition of modern art"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="international exhibition of modern art",
                entity_count=1,
                path_count=37,
                relation_probed="exhibitions.exhibit.exhibitions_displayed_in",
                anchor_position="object",
                resolved_entity_id="m.01_ggr",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "count_answer_target_unenforced:exhibition subjects" in verdict.reasons


def test_validate_pal_execution_rejects_count_when_counted_variable_only_appears_in_aggregate() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "language types",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Lemurian windows into any place or time",
                    "chosen_alias": "m.0cbrvwy",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.languages",
                    "direction": "forward",
                    "from": "setting",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "type_set",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "grounding_source": "curated",
                },
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "type_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?type_set) AS ?count) WHERE { "
            "fb:m.0cbrvwy fb:fictional_universe.fictional_setting.languages ?candidate_set . "
            "?candidate_set fb:type.object.type fb:m.0cbrvwy . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["Lemurian windows into any place or time"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Lemurian windows into any place or time",
                entity_count=1,
                path_count=6,
                relation_probed="fictional_universe.fictional_setting.languages",
                anchor_position="subject",
                resolved_entity_id="m.0cbrvwy",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_unbound_count_variable:type_set" in verdict.reasons


def test_validate_pal_execution_rejects_zero_type_projection_count_with_live_anchor_path() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "language types",
            "query_shape": "count_over_joined_set",
            "anchored_entities": [
                {
                    "surface": "Lemurian windows into any place or time",
                    "chosen_alias": "m.0cbrvwy",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.languages",
                    "direction": "forward",
                    "from": "setting",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "type_set",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "grounding_source": "curated",
                },
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "type_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?type_set) AS ?count) WHERE { "
            "fb:m.0cbrvwy fb:fictional_universe.fictional_setting.languages ?candidate_set . "
            "?candidate_set fb:type.object.type ?type_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["Lemurian windows into any place or time"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Lemurian windows into any place or time",
                entity_count=1,
                path_count=6,
                relation_probed="fictional_universe.fictional_setting.languages",
                anchor_position="subject",
                resolved_entity_id="m.0cbrvwy",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_zero_after_type_projection_with_live_anchor_paths" in verdict.reasons


def test_validate_pal_execution_rejects_count_of_raw_candidates_for_type_like_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "language types",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Lemurian windows into any place or time",
                    "chosen_alias": "m.0cbrvwy",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.languages",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "type.object.type",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "type_set",
                    "from_role": "candidate_set",
                    "to_role": "type_set",
                    "grounding_source": "curated",
                },
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "count",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "fb:m.0cbrvwy fb:fictional_universe.fictional_setting.languages ?candidate_set . "
            "?candidate_set fb:type.object.type ?type . "
            "?type fb:type.object.name ?type_name . "
            "FILTER(LCASE(?type_name) = \"language type\" || "
            "LCASE(?type_name) = \"language types\" || "
            "LCASE(?type_name) = \"language\") . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["Lemurian windows into any place or time"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Lemurian windows into any place or time",
                entity_count=1,
                path_count=6,
                relation_probed="fictional_universe.fictional_setting.languages",
                anchor_position="subject",
                resolved_entity_id="m.0cbrvwy",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_zero_after_type_projection_with_live_anchor_paths" in verdict.reasons


def test_validate_pal_execution_accepts_relation_encoded_type_like_count_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "language types",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Lemurian windows into any place or time",
                    "chosen_alias": "m.0cbrvwy",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.languages",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                    "use_when": "count the languages associated with the fictional setting",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
            "strategy": "Count the language entities reached directly from the setting anchor.",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "fb:m.0cbrvwy fb:fictional_universe.fictional_setting.languages ?candidate_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "6"}}]},
        },
        entities=["Lemurian windows into any place or time"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Lemurian windows into any place or time",
                entity_count=1,
                path_count=6,
                relation_probed="fictional_universe.fictional_setting.languages",
                anchor_position="subject",
                resolved_entity_id="m.0cbrvwy",
            )
        ],
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    assert "count_answer_target_unenforced:language types" not in verdict.reasons


def test_validate_pal_execution_accepts_direct_type_relation_for_type_like_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "answer_target_phrase": "setting types",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {
                    "surface": "Middle-earth",
                    "chosen_alias": "m.012_8w",
                    "role": "anchor",
                }
            ],
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.setting_type",
                    "direction": "forward",
                    "from": "anchor",
                    "to": "candidate_set",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "candidate_set",
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?candidate_set) AS ?count) WHERE { "
            "fb:m.012_8w fb:fictional_universe.fictional_setting.setting_type ?candidate_set . }"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "3"}}]},
        },
        entities=["Middle-earth"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Middle-earth",
                entity_count=1,
                path_count=3,
                relation_probed="fictional_universe.fictional_setting.setting_type",
                anchor_position="subject",
                resolved_entity_id="m.012_8w",
            )
        ],
    )

    assert "count_answer_target_unenforced:setting types" not in verdict.reasons


def test_validate_pal_execution_rejects_entity_result_hitting_query_limit() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Josef Fanta", "chosen_alias": "Josef Fanta", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "architecture.architect.architectural_style",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "architecture.architectural_style.architects",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "dynamic_probe",
                },
            ],
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT DISTINCT ?answer ?answer_name WHERE { "
            '?anchor fb:type.object.name "Josef Fanta"@en . '
            "?anchor fb:architecture.architect.architectural_style ?style . "
            "?style fb:architecture.architectural_style.architects ?answer . "
            "OPTIONAL { ?answer fb:type.object.name ?answer_name . } "
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["answer", "answer_name"]},
            "results": {
                "bindings": [
                    {
                        "answer": {
                            "type": "uri",
                            "value": f"http://rdf.freebase.com/ns/m.{index:03d}",
                        }
                    }
                    for index in range(50)
                ]
            },
        },
        entities=["Josef Fanta"],
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "entity_result_hits_limit_ceiling:50" in verdict.reasons


def test_validate_pal_execution_rejects_generic_type_dump_for_specific_entity_target() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "answer_target_phrase": "meter",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Free verse", "chosen_alias": "Free verse", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT DISTINCT ?candidate_set ?candidate_set_name WHERE { "
            '?anchor fb:type.object.name "Free verse"@en . '
            "?candidate_set fb:type.type.instance ?anchor . "
            "OPTIONAL { ?candidate_set fb:type.object.name ?candidate_set_name . } "
            "}"
        ),
        result_dict={
            "head": {"vars": ["candidate_set", "candidate_set_name"]},
            "results": {
                "bindings": [
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/common.topic",
                        }
                    },
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/base.type_ontology.non_agent",
                        }
                    },
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/base.type_ontology.inanimate",
                        }
                    },
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/base.type_ontology.abstract",
                        }
                    },
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/book.book_subject",
                        }
                    },
                    {
                        "candidate_set": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/media_common.media_genre",
                        }
                    },
                ]
            },
        },
        entities=["Free verse"],
    )

    assert verdict.verdict == VERDICT_REJECTED_DANGEROUS_OVERREACH
    assert "generic_type_result_overbroad_for_answer_target:meter" in verdict.reasons


def test_validate_pal_execution_repairs_grounded_single_anchor_empty_literal_result() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "literal",
            "answer_target_phrase": "meter",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Free verse", "chosen_alias": "Free verse", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT DISTINCT ?answer WHERE { "
            "VALUES ?anchor { fb:m.031jt } "
            "?candidate_set fb:type.type.instance ?anchor . "
            "}"
        ),
        result_dict={
            "head": {"vars": ["answer"]},
            "results": {"bindings": []},
        },
        entities=["Free verse"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Free verse",
                entity_count=1,
                path_count=8,
                relation_probed="type.type.instance",
                anchor_position="object",
                resolved_entity_id="m.031jt",
            )
        ],
    )

    assert verdict.verdict == "repairable_anchor_path_empty"
    assert "grounded_single_anchor_empty_result" in verdict.reasons
    assert "anchor_paths_live_but_projection_empty" in verdict.reasons


def test_validate_pal_execution_does_not_accept_transport_failed_entity_query() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "entity",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Panasonic Lumix", "chosen_alias": "m.093_tr", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "business.brand.products",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "product.camera.sensor_type",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "exploratory",
                },
            ],
            "allow_exploratory_predicates": True,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT ?answer WHERE { "
            "{ BIND(fb:m.093_tr AS ?brand) } UNION VALUES ?brand { fb:m.093_tr } "
            "?brand fb:business.brand.products ?candidate_set . "
            "?candidate_set fb:product.camera.sensor_type ?answer . "
            "}"
        ),
        result_dict=None,
        entities=["Panasonic Lumix"],
        execution_success=False,
        execution_failure_kind="transport_error",
    )

    assert verdict.verdict == "repairable_weak_grounding"
    assert "execution_failed:transport_error" in verdict.reasons


def test_validate_pal_execution_does_not_accept_transport_failed_count_query() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "anchored_entities": [
                {"surface": "Seventh sphere", "chosen_alias": "m.0cb9qd6", "role": "anchor"}
            ],
            "candidate_set_variable": "species",
            "count_set_variable": "species",
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_setting.universe",
                    "direction": "forward",
                    "from_role": "anchor",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                },
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "count_set",
                    "grounding_source": "curated",
                },
            ],
            "allow_exploratory_predicates": False,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (COUNT(DISTINCT ?species) AS ?count) WHERE { "
            "VALUES ?anchor { fb:m.0cb9qd6 } "
            "?anchor fb:fictional_universe.fictional_setting.universe ?candidate_set . "
            "?candidate_set fb:fictional_universe.fictional_universe.species ?species . "
            "}"
        ),
        result_dict=None,
        entities=["Seventh sphere"],
        execution_success=False,
        execution_failure_kind="transport_error",
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "execution_failed:transport_error" in verdict.reasons


def test_validate_pal_execution_repairs_missing_selected_head_var_in_bindings() -> None:
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "literal",
            "query_shape": "single_anchor_lookup",
            "anchored_entities": [
                {"surface": "Flare star", "chosen_alias": "m.05dt2l", "role": "anchor"}
            ],
            "relation_paths": [
                {
                    "relation": "astronomy.celestial_object.category",
                    "direction": "reverse",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                },
                {
                    "relation": "astronomy.celestial_object.temperature",
                    "direction": "forward",
                    "from_role": "candidate_set",
                    "to_role": "answer",
                    "grounding_source": "exploratory",
                },
            ],
            "allow_exploratory_predicates": True,
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/> "
            "SELECT (MIN(?temp) AS ?answer) WHERE { "
            "?candidate fb:astronomy.celestial_object.category fb:m.05dt2l . "
            "?candidate fb:astronomy.celestial_object.temperature ?temp . "
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["answer"]},
            "results": {"bindings": [{}]},
        },
        entities=["Flare star"],
    )

    assert verdict.verdict == "repairable_bad_projection"
    assert "selected_head_var_missing_from_bindings:answer" in verdict.reasons


# ---------------------------------------------------------------------------
# Zero-count fix tests (Fix A, Fix B1, Fix B2)
# ---------------------------------------------------------------------------

def test_fix_a_curated_plan_with_undeclared_type_filter_zero_is_repairable() -> None:
    """
    Fix A — Sample-4-like pattern.
    A curated plan with a single anchor→candidate_set relation path whose
    rendered SPARQL adds an unplanned fb:type.object.type + FILTER that was
    never declared in relation_paths.  The type filter prevents any matches
    and the count returns 0.  Should be REPAIRABLE, not accepted.
    """
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "medical treatments",
            "anchored_entities": [
                {
                    "surface": "Unsteadiness",
                    "chosen_alias": "Unsteadiness",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "candidate_set",
            "count_set_variable": "treatment",
            "relation_paths": [
                {
                    "relation": "medicine.symptom.side_effect_of",
                    "direction": "forward",
                    "from": "Unsteadiness",
                    "from_role": "anchor",
                    "to": "treatment",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "traverse medicine.symptom.side_effect_of from symptom to treatment",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?treatment) AS ?count) WHERE {\n"
            "  {\n"
            "    ?symptom fb:type.object.name ?symname .\n"
            "    FILTER(LCASE(STR(?symname)) = \"unsteadiness\")\n"
            "  }\n"
            "  ?symptom fb:medicine.symptom.side_effect_of ?treatment .\n"
            "  ?treatment fb:type.object.type ?t .\n"
            "  ?t fb:type.object.name ?tname .\n"
            "  FILTER(LCASE(?tname) = \"medical treatment\")\n"
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["Unsteadiness"],
        anchor_probe_results=[
            AnchorProbeResult(
                anchor_name="Unsteadiness",
                entity_count=1,
                path_count=1,
                relation_probed="medicine.symptom.side_effect_of",
                anchor_position="subject",
                resolved_entity_id="m.0unsteadiness",
            )
        ],
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_query_zero_with_undeclared_type_filter" in verdict.reasons
    assert "count_scalar_returned:0" in verdict.reasons


def test_fix_a_curated_plan_no_type_filter_zero_is_not_flagged_by_fix_a() -> None:
    """
    Fix A regression guard — a curated plan with no type filter in the SPARQL
    should NOT be flagged by Fix A even when count is 0.
    """
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "albums",
            "anchored_entities": [
                {
                    "surface": "NoAlbumsArtist",
                    "chosen_alias": "NoAlbumsArtist",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "album",
            "count_set_variable": "album",
            "relation_paths": [
                {
                    "relation": "music.artist.album",
                    "direction": "forward",
                    "from": "NoAlbumsArtist",
                    "from_role": "anchor",
                    "to": "album",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count albums for the artist",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?album) AS ?count) WHERE {\n"
            "  VALUES ?artist { fb:m.0noalbums }\n"
            "  ?artist fb:music.artist.album ?album .\n"
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["NoAlbumsArtist"],
        anchor_probe_results=None,
    )

    # Fix A must NOT fire — no type filter in SPARQL.
    assert "count_query_zero_with_undeclared_type_filter" not in verdict.reasons


def test_fix_b1_count_projected_over_unbound_variable_is_repairable() -> None:
    """
    Fix B1 — Sample-5-like pattern.
    COUNT(DISTINCT ?candidate_genre_set) where ?candidate_genre_set never
    appears in WHERE.  Structurally guaranteed zero; must be REPAIRABLE.
    """
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "different media genres",
            "anchored_entities": [
                {
                    "surface": "Hentai",
                    "chosen_alias": "Hentai",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "candidate_genre_set",
            "count_set_variable": "count_candidate_genres",
            "relation_paths": [
                {
                    "relation": "media_common.media_genre.child_genres",
                    "direction": "forward",
                    "from": "parent_genre",
                    "from_role": "constraint_value",
                    "to": "child_genre",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count child genres of Hentai via child_genres relation",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?candidate_genre_set) AS ?count) WHERE {\n"
            "  ?parent_genre_constraint_value fb:type.object.name"
            " ?parent_genre_constraint_value_label .\n"
            "  FILTER(LCASE(STR(?parent_genre_constraint_value_label)) = \"parent_genre\")\n"
            "  ?child_genre_constraint_value fb:type.object.name"
            " ?child_genre_constraint_value_label .\n"
            "  FILTER(LCASE(STR(?child_genre_constraint_value_label)) = \"child_genre\")\n"
            "  ?parent_genre_constraint_value"
            " fb:media_common.media_genre.child_genres ?child_genre_constraint_value .\n"
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["Hentai"],
        anchor_probe_results=None,
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_projection_variable_unbound_in_where" in verdict.reasons
    assert "count_scalar_returned:0" in verdict.reasons


def test_fix_b2_direct_count_plan_all_constraint_value_paths_is_repairable() -> None:
    """
    Fix B2 — count_over_direct_relation plan where every relation path has
    constraint_value roles on both sides (anchor never connected to the
    counted variable).  Must be REPAIRABLE regardless of query text.
    """
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "child genres",
            "anchored_entities": [
                {
                    "surface": "Horror",
                    "chosen_alias": "Horror",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "genre",
            "count_set_variable": "genre",
            "relation_paths": [
                {
                    "relation": "media_common.media_genre.child_genres",
                    "direction": "forward",
                    "from": "parent",
                    "from_role": "constraint_value",
                    "to": "child",
                    "to_role": "constraint_value",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count child genres",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?genre) AS ?count) WHERE {\n"
            "  VALUES ?parent { fb:m.0horror }\n"
            "  ?parent fb:media_common.media_genre.child_genres ?genre .\n"
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "0"}}]},
        },
        entities=["Horror"],
        anchor_probe_results=None,
    )

    assert verdict.verdict == VERDICT_REPAIRABLE_BAD_COUNT_SET
    assert "count_direct_relation_plan_no_anchor_to_count_path" in verdict.reasons
    assert "count_scalar_returned:0" in verdict.reasons


def test_legitimate_nonzero_count_is_not_blocked() -> None:
    """
    Control case — a structurally correct curated count_over_direct_relation
    plan returning a non-zero count must be ACCEPTED.  Regression guard for
    all three fixes.
    """
    verdict = validate_pal_execution(
        query_plan={
            "answer_mode": "count",
            "query_shape": "count_over_direct_relation",
            "answer_target_phrase": "species",
            "anchored_entities": [
                {
                    "surface": "Seventh sphere",
                    "chosen_alias": "Seventh sphere",
                    "role": "anchor",
                }
            ],
            "candidate_set_variable": "species",
            "count_set_variable": "species",
            "relation_paths": [
                {
                    "relation": "fictional_universe.fictional_universe.species",
                    "direction": "forward",
                    "from": "Seventh sphere",
                    "from_role": "anchor",
                    "to": "species",
                    "to_role": "candidate_set",
                    "grounding_source": "curated",
                }
            ],
            "allow_exploratory_predicates": False,
            "strategy": "count species in the fictional universe",
        },
        query_text=(
            "PREFIX fb: <http://rdf.freebase.com/ns/>\n"
            "SELECT (COUNT(DISTINCT ?species) AS ?count) WHERE {\n"
            "  VALUES ?universe { fb:m.07th_sphere }\n"
            "  ?universe fb:fictional_universe.fictional_universe.species ?species .\n"
            "} LIMIT 50"
        ),
        result_dict={
            "head": {"vars": ["count"]},
            "results": {"bindings": [{"count": {"type": "literal", "value": "45"}}]},
        },
        entities=["Seventh sphere"],
        anchor_probe_results=None,
    )

    assert verdict.verdict == VERDICT_ACCEPTED
    # None of the new fix reasons should appear for a correct result.
    assert "count_query_zero_with_undeclared_type_filter" not in verdict.reasons
    assert "count_projection_variable_unbound_in_where" not in verdict.reasons
    assert "count_direct_relation_plan_no_anchor_to_count_path" not in verdict.reasons
