import json
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.exceptions import AgentUnknownException
from src.agents.instance.pal_agent_controller import PALAgentController
from src.pal.plausibility_validator import (
    AnchorProbeResult,
    PlausibilityVerdict,
    VERDICT_ACCEPTED,
    VERDICT_REPAIRABLE_BAD_COUNT_SET,
    VERDICT_REPAIRABLE_BAD_JOIN,
    VERDICT_REPAIRABLE_BAD_SUPERLATIVE,
    VERDICT_REPAIRABLE_GROUNDED_EMPTY,
    build_repair_feedback,
    validate_pal_execution,
)


def _make_controller() -> PALAgentController:
    controller = object.__new__(PALAgentController)
    controller._emit_generated_tools_event = lambda payload: None
    return controller


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


def test_build_entity_alias_candidates_adds_uppercase_acronym() -> None:
    controller = _make_controller()

    candidates = controller._build_entity_alias_candidates("wma")

    assert "WMA" in candidates


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
    assert [result.relation_probed for result in results] == [
        "medicine.drug_formulation.formulation_of",
        "medicine.drug_formulation.active_ingredients",
    ]
    assert [result.anchor_position for result in results] == ["object", "object"]


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


def test_superlative_grounding_normalizes_ordering_roles() -> None:
    controller = _make_controller()
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

    assert candidates[0]["from_role"] == "candidate_set"
    assert candidates[0]["to_role"] == "candidate_set"
    assert candidates[1]["from_role"] == "candidate_set"
    assert candidates[1]["to_role"] == "ordering_attribute"


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
    assert probe_results[0].relation_probed == (
        "fictional_universe.fictional_setting.universe"
        " -> fictional_universe.fictional_universe.species"
    )


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
