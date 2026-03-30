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
from src.agents.instance.pal_agent_controller import PALAgentController
from src.pal.invoker import PALInvocationResult
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
from src.pal.kg_benchmark_adapter import BenchmarkMaterialization


def _make_controller() -> PALAgentController:
    controller = object.__new__(PALAgentController)
    controller._emit_generated_tools_event = lambda payload: None
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
    assert "superlative_over_candidate_set" in scaffold_names
    assert "pivoted_chain_lookup" in scaffold_names


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


def test_build_entity_alias_candidates_strips_common_geopolitical_prefixes() -> None:
    controller = _make_controller()

    aliases = controller._build_entity_alias_candidates("republic of brazil")

    assert "Brazil" in aliases
    assert aliases.index("Brazil") < aliases.index("brazil")


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


def test_grounding_merges_dynamic_reverse_count_candidates_even_when_curated_exists() -> None:
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

    assert grounding[0]["relation"] == "people.profession.people_with_this_profession"
    assert any(
        str(candidate.get("relation") or "") == "people.person.profession"
        for candidate in grounding
    )


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
    assert loop_log["final_verdict"] in {"accepted_best_effort", "no_accepted_candidate"}


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


def test_should_accept_best_executing_candidate_allows_salvageable_count_result() -> None:
    controller = _make_controller()

    assert controller._should_accept_best_executing_candidate(
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
    assert controller._should_accept_best_executing_candidate(
        candidate_metadata=positive_metadata
    )
    assert not controller._should_accept_best_executing_candidate(
        candidate_metadata=zero_with_unverified_type
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


def test_anchor_existence_probes_capture_unique_resolved_entity_id_from_live_path() -> None:
    controller = _make_controller()
    controller._probe_entity_name_count = lambda alias, timeout_s=2.5: 3
    controller._probe_anchor_path_count = (
        lambda alias, relation, anchor_position="subject", timeout_s=2.5: 4
    )
    controller._probe_anchor_entity_ids = (
        lambda *, anchor_name, relation=None, anchor_position="subject", timeout_s=2.5: ["m.01c44b"]
        if anchor_name == "Southern Min"
        and relation == "language.language_dialect.language"
        and anchor_position == "object"
        else []
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
