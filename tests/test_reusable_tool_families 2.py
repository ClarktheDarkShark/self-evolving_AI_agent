import pathlib
import sys
from unittest.mock import patch

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.agents.instance.pal_agent_controller as pal_agent_controller_module
from src.agents.instance.pal_agent_controller import PALAgentController
from src.pal.parser import extract_and_validate_code
from src.pal.reusable_tool_families import (
    get_reusable_family_policy_bundle,
    render_reusable_tool,
    select_reusable_tool,
)


def _make_controller() -> PALAgentController:
    controller = object.__new__(PALAgentController)
    controller._emit_generated_tools_event = lambda payload: None
    return controller


def test_select_reusable_tool_and_render_direct_count_query() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Southern Min",
                "chosen_alias": "Southern Min",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "dialect",
        "count_set_variable": "dialect",
        "relation_paths": [
            {
                "relation": "language.language_dialect.language",
                "direction": "reverse",
                "from": "dialect",
                "to": "Southern Min",
                "from_role": "count_set",
                "to_role": "anchor",
                "grounding_source": "curated",
            }
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    assert selection.family_name == "count_over_direct_relation"
    assert selection.policy_bundle.family_name == "count_over_direct_relation"
    assert selection.policy_bundle.renderer_name == "count"
    assert (
        get_reusable_family_policy_bundle("count_over_direct_relation")
        == selection.policy_bundle
    )

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?dialect) AS ?count" in generated_code
    assert "fb:language.language_dialect.language" in generated_code
    assert 'FILTER(LCASE(STR(?anchor_label)) = "southern min")' in generated_code


def test_select_reusable_tool_rejects_literal_single_anchor_lookup() -> None:
    selection = select_reusable_tool(
        {
            "answer_mode": "literal",
            "query_shape": "single_anchor_lookup",
            "shared_answer_variable": "answer",
            "projection": ["answer"],
            "relation_paths": [
                {
                    "relation": "type.type.instance",
                    "direction": "reverse",
                    "from": "candidate_set",
                    "to": "Free verse",
                    "from_role": "candidate_set",
                    "to_role": "anchor",
                    "grounding_source": "dynamic_probe",
                }
            ],
        }
    )

    assert selection is None


def test_direct_count_renderer_reverses_reverse_anchor_candidate_relation() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Mosquito",
                "chosen_alias": "m.09f96",
                "resolved_entity_id": "m.09f96",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "disease",
        "count_set_variable": "disease",
        "relation_paths": [
            {
                "relation": "medicine.disease.transmitted_by",
                "direction": "reverse",
                "from": "transmitter",
                "to": "disease",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            }
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?disease fb:medicine.disease.transmitted_by fb:m.09f96 ." in generated_code
    assert "fb:m.09f96 fb:medicine.disease.transmitted_by ?disease ." not in generated_code


def test_reusable_render_canonicalizes_role_swap_match_for_reverse_projection(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv("PAL_RUNTIME_REUSABLE_SWAP_RENDER_CANONICALIZATION", "1")

    query_plan = {
        "answer_mode": "entity",
        "query_shape": "single_anchor_lookup",
        "anchored_entities": [
            {
                "surface": "Ibiza Euphoria",
                "chosen_alias": "m.03_9dcv",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "answer",
        "candidate_set_variable": "candidate_set",
        "relation_paths": [
            {
                "relation": "music.recording.releases",
                "direction": "reverse",
                "from": "recording",
                "to": "m.03_9dcv",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "music.artist.track",
                "direction": "reverse",
                "from": "track",
                "to": "artist",
                "from_role": "candidate_set",
                "to_role": "answer",
                "grounding_source": "curated",
            },
        ],
        "projection": ["answer", "answer_name"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }
    relation_grounding = [
        {
            "relation": "music.recording.releases",
            "direction": "reverse",
            "from": "recording",
            "to": "m.03_9dcv",
            "from_role": "candidate_set",
            "to_role": "anchor",
            "grounding_source": "dynamic_probe",
        },
        {
            "relation": "music.artist.track",
            "direction": "forward",
            "from": "artist",
            "to": "track",
            "from_role": "anchor",
            "to_role": "answer",
            "grounding_source": "curated",
        },
    ]

    canonicalized = controller._canonicalize_reusable_query_plan_for_render(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    )

    second_path = canonicalized["relation_paths"][1]
    assert second_path["from"] == "artist"
    assert second_path["to"] == "track"
    assert second_path["from_role"] == "answer"
    assert second_path["to_role"] == "candidate_set"

    selection = select_reusable_tool(query_plan)
    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=canonicalized,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?answer fb:music.artist.track ?candidate_set ." in generated_code
    assert "?candidate_set fb:music.artist.track ?answer ." not in generated_code


def test_reusable_render_does_not_rewrite_canonical_reverse_anchor_path(
    monkeypatch,
) -> None:
    controller = _make_controller()
    monkeypatch.setenv("PAL_RUNTIME_REUSABLE_SWAP_RENDER_CANONICALIZATION", "1")

    query_plan = {
        "answer_mode": "entity",
        "query_shape": "single_anchor_lookup",
        "anchored_entities": [
            {
                "surface": "Ibiza Euphoria",
                "chosen_alias": "m.03_9dcv",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "answer",
        "candidate_set_variable": "candidate_set",
        "relation_paths": [
            {
                "relation": "music.recording.releases",
                "direction": "reverse",
                "from": "recording",
                "to": "m.03_9dcv",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            }
        ],
        "projection": ["answer", "answer_name"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }
    relation_grounding = [
        {
            "relation": "music.recording.releases",
            "direction": "reverse",
            "from": "recording",
            "to": "m.03_9dcv",
            "from_role": "candidate_set",
            "to_role": "anchor",
            "grounding_source": "dynamic_probe",
        }
    ]

    canonicalized = controller._canonicalize_reusable_query_plan_for_render(
        query_plan=query_plan,
        relation_grounding=relation_grounding,
    )

    assert canonicalized["relation_paths"][0] == query_plan["relation_paths"][0]


def test_direct_count_renderer_counts_terminal_count_set_leaf_in_pivot_chain() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Seventh sphere",
                "chosen_alias": "m.0cb9qd6",
                "resolved_entity_id": "m.0cb9qd6",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "species",
        "candidate_set_variable": "species_set",
        "count_set_variable": "count_species",
        "relation_paths": [
            {
                "relation": "fictional_universe.fictional_setting.universe",
                "direction": "forward",
                "from": "anchor",
                "to": "seventh_universe",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "fictional_universe.fictional_universe.species",
                "direction": "forward",
                "from": "seventh_universe",
                "to": "species",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count_species"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?species) AS ?count" in generated_code
    assert "?seventh_universe fb:fictional_universe.fictional_universe.species ?species ." in generated_code
    assert "?count_species" not in generated_code
    assert "COUNT(DISTINCT ?species_set) AS ?count" not in generated_code


def test_direct_count_renderer_preserves_repeated_bridge_variable_even_when_roles_are_mislabeled() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Richard Altwasser",
                "chosen_alias": "m.050yww4",
                "resolved_entity_id": "m.050yww4",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "candidate",
        "candidate_set_variable": "candidate",
        "count_set_variable": "count",
        "relation_paths": [
            {
                "relation": "computer.computer_designer.computers_designed",
                "direction": "forward",
                "from": "anchor",
                "to": "computer",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "computer.computer.key_designers",
                "direction": "forward",
                "from": "computer",
                "to": "candidate",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?candidate) AS ?count" in generated_code
    assert "fb:m.050yww4 fb:computer.computer_designer.computers_designed ?computer ." in generated_code
    assert "?computer fb:computer.computer.key_designers ?candidate ." in generated_code
    assert "fb:m.050yww4 fb:computer.computer.key_designers ?candidate ." not in generated_code
    assert "fb:m.050yww4 fb:computer.computer_designer.computers_designed ?candidate ." not in generated_code


def test_select_reusable_tool_and_render_superlative_query() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "superlative_chain",
        "anchored_entities": [
            {
                "surface": "Count Basie Orchestra",
                "chosen_alias": "Count Basie Orchestra",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "release",
        "ordering_attribute": {
            "relation": "music.release.release_date",
            "direction": "forward",
            "source_variable": "release",
            "attribute_variable": "release_date",
        },
        "ordering_direction": "max",
        "relation_paths": [
            {
                "relation": "music.artist.track",
                "direction": "forward",
                "from": "Count Basie Orchestra",
                "to": "release",
                "from_role": "anchor",
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
                "grounding_source": "curated",
            },
        ],
        "projection": ["release"],
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    assert selection.family_name == "superlative_chain"

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "ORDER BY DESC(?release_date)" in generated_code
    assert "LIMIT 1" in generated_code
    assert "SELECT DISTINCT ?release ?release_name" in generated_code


def test_select_reusable_tool_and_render_literal_superlative_query() -> None:
    query_plan = {
        "answer_mode": "literal",
        "query_shape": "superlative_chain",
        "anchored_entities": [
            {
                "surface": "Flare star",
                "chosen_alias": "m.05dt2l",
                "resolved_entity_id": "m.05dt2l",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "candidate_set",
        "ordering_attribute": {
            "relation": "astronomy.star.temperature_k",
            "direction": "forward",
            "source_variable": "candidate_set",
            "attribute_variable": "temp_k",
        },
        "ordering_direction": "min",
        "relation_paths": [
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
                "relation": "astronomy.star.temperature_k",
                "direction": "forward",
                "from": "candidate_set",
                "to": "temp_k",
                "from_role": "candidate_set",
                "to_role": "ordering_attribute",
                "grounding_source": "curated",
            },
        ],
        "projection": ["temp_k"],
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    assert selection.family_name == "superlative_chain"
    assert "literal_projection_via_ordering_attribute" in selection.reasons

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "SELECT DISTINCT ?temp_k WHERE {" in generated_code
    assert "ORDER BY ASC(?temp_k)" in generated_code
    assert "LIMIT 1" in generated_code
    assert "fb:m.05dt2l" in generated_code


def test_render_superlative_query_uses_terminal_ordering_value_when_chained() -> None:
    query_plan = {
        "answer_mode": "entity",
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
                "from_role": "ordering_attribute",
                "to_role": "ordering_attribute",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["candidate_set"],
        "allow_exploratory_predicates": True,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    assert selection.family_name == "superlative_chain"

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?candidate_set fb:music.release.release_date ?release_date_node ." in generated_code
    assert (
        "?release_date_node fb:freebase.valuenotation.has_value ?release_date_value ."
        in generated_code
    )
    assert "ORDER BY DESC(?release_date_value) LIMIT 1" in generated_code


def test_render_entity_superlative_query_after_plan_renormalization() -> None:
    controller = _make_controller()
    normalized_query_plan = controller._normalize_pal_query_plan(
        {
            "answer_mode": "entity",
            "answer_type": "entity",
            "answer_target_phrase": "temperature",
            "query_shape": "superlative_chain",
            "anchored_entities": [
                {
                    "surface": "Flare star",
                    "chosen_alias": "m.05dt2l",
                    "resolved_entity_id": "m.05dt2l",
                    "role": "anchor",
                }
            ],
            "shared_answer_variable": "candidate_star",
            "candidate_set_variable": "candidate_star",
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
                    "from_role": "type_set",
                    "to_role": "anchor",
                    "grounding_source": "curated",
                },
                {
                    "relation": "astronomy.star.temperature_k",
                    "direction": "forward",
                    "from": "candidate_set",
                    "to": "answer",
                    "from_role": "candidate_set",
                    "to_role": "ordering_attribute",
                    "grounding_source": "curated",
                },
            ],
            "projection": ["candidate_star"],
            "allow_exploratory_predicates": False,
        }
    )

    selection = select_reusable_tool(normalized_query_plan)

    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=normalized_query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert (
        "fb:m.05dt2l fb:astronomy.celestial_object_category.objects ?candidate_star ."
        in generated_code
    )
    assert "?category_type_set" not in generated_code


def test_superlative_renderer_falls_back_to_structural_candidate_projection() -> None:
    query_plan = {
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
            "source_variable": "candidate_set",
            "attribute_variable": "length",
        },
        "ordering_direction": "max",
        "relation_paths": [
            {
                "relation": "music.engineer.tracks_engineered",
                "direction": "forward",
                "from": "anchor",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "music.recording.length",
                "direction": "forward",
                "from": "candidate_set",
                "to": "length",
                "from_role": "candidate_set",
                "to_role": "ordering_attribute",
                "grounding_source": "curated",
            },
        ],
        "projection": ["shared_answer", "answer_name"],
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "SELECT DISTINCT ?candidate_set ?candidate_set_name" in generated_code
    assert "?anchor fb:music.engineer.tracks_engineered ?candidate_set ." in generated_code
    assert "?shared_answer" not in generated_code


def test_superlative_renderer_uses_subquery_for_post_selection_answer_projection() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "superlative_chain",
        "anchored_entities": [
            {
                "surface": "David Han",
                "chosen_alias": "David Han",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "answer",
        "candidate_set_variable": "candidate_recording",
        "ordering_attribute": {
            "relation": "music.recording.length",
            "direction": "forward",
            "source_variable": "candidate_recording",
            "attribute_variable": "length",
        },
        "ordering_direction": "max",
        "relation_paths": [
            {
                "relation": "music.recording.engineer",
                "direction": "reverse",
                "from": "recording",
                "to": "David Han",
                "from_role": "candidate_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
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
                "relation": "music.release.tracks",
                "direction": "forward",
                "from": "release",
                "to": "track",
                "from_role": "candidate_set",
                "to_role": "answer",
                "grounding_source": "curated",
            },
        ],
        "projection": ["answer"],
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "SELECT DISTINCT ?answer ?answer_name WHERE {" in generated_code
    assert "SELECT DISTINCT ?candidate_recording ?length WHERE {" in generated_code
    assert "ORDER BY DESC(?length) LIMIT 1" in generated_code
    assert "?candidate_recording fb:music.recording.releases ?release_candidate_set ." in generated_code
    assert "?release_candidate_set fb:music.release.tracks ?answer ." in generated_code


def test_generate_validated_pal_candidate_prefers_reusable_tool_before_llm() -> None:
    controller = _make_controller()
    controller._run_text_prompt = lambda system_prompt, user_prompt: pytest.fail(
        "LLM generator should not be called for reusable direct-count plan"
    )

    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Southern Min",
                "chosen_alias": "Southern Min",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "dialect",
        "count_set_variable": "dialect",
        "relation_paths": [
            {
                "relation": "language.language_dialect.language",
                "direction": "reverse",
                "from": "dialect",
                "to": "Southern Min",
                "from_role": "count_set",
                "to_role": "anchor",
                "grounding_source": "curated",
            }
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
        "strategy": "count dialects for a language anchor",
    }

    generated_code, metadata = controller._generate_validated_pal_candidate(
        task_question="Question: how many language dialects does southern min have?, Entities: ['Southern Min']",
        grounding_card="grounding",
        query_plan=query_plan,
        generated_tool_name="pal_test_tool",
    )

    assert "COUNT(DISTINCT ?dialect) AS ?count" in generated_code
    assert "language.language_dialect.language" in metadata["query_text"]


def test_generate_validated_pal_candidate_falls_back_to_llm_when_reusable_query_is_rejected() -> None:
    controller = _make_controller()
    llm_calls: list[str] = []

    controller._run_text_prompt = lambda system_prompt, user_prompt: (
        llm_calls.append(user_prompt)
        or """
###QUERY_START
from SPARQLWrapper import SPARQLWrapper, JSON

def solve(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    query = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?dialect) AS ?count) WHERE {
      ?anchor fb:type.object.name ?anchor_label .
      FILTER(LCASE(STR(?anchor_label)) = "southern min")
      ?dialect fb:language.language_dialect.language ?anchor .
    } LIMIT 50
    \"\"\"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
###QUERY_END
"""
    )

    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Southern Min",
                "chosen_alias": "Southern Min",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "dialect",
        "count_set_variable": "dialect",
        "relation_paths": [
            {
                "relation": "language.language_dialect.language",
                "direction": "reverse",
                "from": "dialect",
                "to": "Southern Min",
                "from_role": "count_set",
                "to_role": "anchor",
                "grounding_source": "curated",
            }
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
        "strategy": "count dialects for a language anchor",
    }

    with patch.object(
        pal_agent_controller_module,
        "render_reusable_tool",
        return_value="""
###QUERY_START
from SPARQLWrapper import SPARQLWrapper, JSON

def solve(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    query = \"\"\"
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT ?wrong WHERE { ?x fb:type.object.type ?y . } LIMIT 50
    \"\"\"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
###QUERY_END
""",
    ):
        generated_code, metadata = controller._generate_validated_pal_candidate(
            task_question="Question: how many language dialects does southern min have?, Entities: ['Southern Min']",
            grounding_card="grounding",
            query_plan=query_plan,
            generated_tool_name="pal_test_tool",
        )

    assert llm_calls
    assert "COUNT(DISTINCT ?dialect) AS ?count" in generated_code
    assert "language.language_dialect.language" in metadata["query_text"]


def test_render_joined_count_query_binds_object_side_anchor_and_shared_constraint() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Great Britain",
                "chosen_alias": "Great Britain",
                "role": "anchor_a",
            },
            {
                "surface": "Otterhound",
                "chosen_alias": "Otterhound",
                "role": "anchor_b",
            },
        ],
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "candidate_set",
                    "notes": "Require breed.country_of_origin = Great Britain via biology.animal_breed.country_of_origin",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "candidate_set",
                    "notes": "Require breed.temperament to match Otterhound via biology.animal_breed.temperament",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "biology.animal_breed.country_of_origin",
                "direction": "forward",
                "from": "candidate_set",
                "to": "country",
                "from_role": "count_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "candidate_set",
                "to": "breed_temperament",
                "from_role": "count_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "anchor_b",
                "to": "otter_temperament",
                "from_role": "anchor_b",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?candidate_set fb:biology.animal_breed.country_of_origin ?anchor_a ." in generated_code
    assert "?candidate_set fb:biology.animal_breed.temperament ?breed_temperament ." in generated_code
    assert "?anchor_b fb:biology.animal_breed.temperament ?breed_temperament ." in generated_code
    assert '"great britain"' in generated_code.lower()
    assert '"otterhound"' in generated_code.lower()
    assert '"country"' not in generated_code.lower()
    assert '"temperament"' not in generated_code.lower()


def test_reusable_count_renderer_uses_candidate_set_not_projection_alias() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Hentai",
                "chosen_alias": "Hentai",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count",
        "relation_paths": [
            {
                "relation": "media_common.media_genre.child_genres",
                "direction": "forward",
                "from": "parent_genre",
                "to": "child_genre",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "curated",
            }
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?candidate_set) AS ?count" in generated_code
    assert "?anchor fb:media_common.media_genre.child_genres ?candidate_set ." in generated_code
    assert "COUNT(DISTINCT ?count) AS ?count" not in generated_code
    assert "COUNT(DISTINCT ?answer_count) AS ?count" not in generated_code


def test_joined_count_renderer_does_not_reuse_count_alias_in_where_clause() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Great Britain",
                "chosen_alias": "Great Britain",
                "role": "anchor_a",
            },
            {
                "surface": "Otterhound",
                "chosen_alias": "Otterhound",
                "role": "anchor_b",
            },
        ],
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count",
        "relation_paths": [
            {
                "relation": "biology.animal_breed.country_of_origin",
                "direction": "forward",
                "from": "breed",
                "to": "country",
                "from_role": "count_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "breed",
                "to": "temperament",
                "from_role": "count_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "anchor_b",
                "to": "otter_temperament",
                "from_role": "anchor_b",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?candidate_set) AS ?count" in generated_code
    assert "?count fb:biology.animal_breed.country_of_origin" not in generated_code


def test_joined_count_renderer_preserves_explicit_constraint_value_entity() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Percussionist",
                "chosen_alias": "Percussionist",
                "resolved_entity_id": "m.02h66l4",
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
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "shared_answer",
                    "notes": "The primary anchor constrains the shared counted entity set.",
                },
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": "shared_answer",
                    "notes": "The answer target phrase is treated as an explicit answer-class filter.",
                },
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
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?shared_answer fb:people.person.profession fb:m.02h66l4 ." in generated_code
    assert '?constraint_value fb:type.object.name ?constraint_value_label .' in generated_code
    assert 'FILTER(LCASE(STR(?constraint_value_label)) = "songwriter")' in generated_code
    assert "?shared_answer fb:people.person.profession ?constraint_value ." in generated_code
    assert "?count fb:biology.animal_breed.temperament" not in generated_code


def test_joined_count_renderer_uses_single_anchor_entity_when_plan_role_mismatches() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "valve corp",
                "chosen_alias": "m.0dwl2",
                "resolved_entity_id": "m.0dwl2",
                "role": "anchor_a",
            },
            {
                "surface": "game expansions",
                "chosen_alias": "game expansion",
                "role": "constraint_value",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "shared_answer",
        "count_set_variable": "shared_answer",
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "shared_answer",
                    "notes": "The primary anchor constrains the shared counted entity set via cvg.cvg_publisher.game_versions_published.",
                },
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": "shared_answer",
                    "notes": "The answer target phrase is treated as an explicit answer-class or answer-constraint filter on the same counted set via type.object.type with value 'game expansion'.",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "cvg.cvg_publisher.game_versions_published",
                "direction": "reverse",
                "from": "publisher",
                "to": "shared_answer",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "shared_answer",
                "to": "constraint_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "exploratory",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.0dwl2 fb:cvg.cvg_publisher.game_versions_published ?shared_answer ." in generated_code
    assert 'FILTER(LCASE(STR(?anchor_label)) = "publisher")' not in generated_code
    assert 'FILTER(LCASE(STR(?constraint_value_label)) = "game expansion")' in generated_code


def test_joined_count_renderer_keeps_relation_hinted_anchor_binding_even_with_repeated_constraint_endpoint() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Percussionist",
                "chosen_alias": "m.02h66l4",
                "resolved_entity_id": "m.02h66l4",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "candidate_person",
        "candidate_set_variable": "candidate_person",
        "count_set_variable": "person",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "person",
                    "notes": (
                        "The profession anchor 'Percussionist' is used via "
                        "people.profession.people_with_this_profession."
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
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.02h66l4 fb:people.profession.people_with_this_profession ?person ." in generated_code
    assert "?profession fb:people.profession.people_with_this_profession ?person ." not in generated_code
    assert "?person fb:people.person.profession ?profession ." in generated_code


def test_joined_count_renderer_uses_relation_hinted_constraint_note_when_no_explicit_entity_exists() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Percussionist",
                "chosen_alias": "m.02h66l4",
                "resolved_entity_id": "m.02h66l4",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "person",
        "candidate_set_variable": "person_set",
        "count_set_variable": "person",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "person",
                    "notes": (
                        "Profession anchor (Percussionist, id m.02h66l4) -> persons "
                        "via people.profession.people_with_this_profession"
                    ),
                },
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": "person",
                    "notes": (
                        "Persons must also have profession = 'songwriter' "
                        "(enforced via people.person.profession)"
                    ),
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "profession_anchor",
                "to": "person",
                "from_role": "constraint_value",
                "to_role": "count_set",
                "grounding_source": "curated",
            },
            {
                "relation": "people.person.profession",
                "direction": "forward",
                "from": "person",
                "to": "profession_value",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.02h66l4 fb:people.profession.people_with_this_profession ?person ." in generated_code
    assert '?songwriter_constraint_value fb:type.object.name ?songwriter_constraint_value_label .' in generated_code
    assert 'FILTER(LCASE(STR(?songwriter_constraint_value_label)) = "songwriter")' in generated_code
    assert "?person fb:people.person.profession ?songwriter_constraint_value ." in generated_code
    assert '"profession_value"' not in generated_code


def test_joined_count_renderer_allows_explicit_constraint_note_to_override_anchor_hint_on_same_relation() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Percussionist",
                "chosen_alias": "m.02h66l4",
                "resolved_entity_id": "m.02h66l4",
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
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "shared_answer",
                    "notes": (
                        "The primary anchor constrains the shared counted entity set "
                        "via people.profession.people_with_this_profession."
                    ),
                },
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": "shared_answer",
                    "notes": (
                        "The answer target phrase is treated as an explicit "
                        "answer-class filter on the same counted set via "
                        "people.profession.people_with_this_profession with value "
                        "'songwriter'."
                    ),
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "anchor",
                "to": "shared_answer",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "people.profession.people_with_this_profession",
                "direction": "forward",
                "from": "constraint_value",
                "to": "shared_answer",
                "from_role": "constraint_value",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.02h66l4 fb:people.profession.people_with_this_profession ?shared_answer ." in generated_code
    assert '?constraint_value fb:type.object.name ?constraint_value_label .' in generated_code
    assert 'FILTER(LCASE(STR(?constraint_value_label)) = "songwriter")' in generated_code
    assert "?constraint_value fb:people.profession.people_with_this_profession ?shared_answer ." in generated_code
    assert "?anchor fb:people.profession.people_with_this_profession ?shared_answer ." not in generated_code


def test_joined_count_renderer_preserves_auxiliary_bridge_variable() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "m.03bv2kt",
                "resolved_entity_id": "m.03bv2kt",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "m.03fx9_c",
                "resolved_entity_id": "m.03fx9_c",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "candidate_set",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "candidate_set",
                    "notes": "genre filter: candidate content must be in genre Higher Education",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "candidate_set",
                    "notes": "producer filter: candidate content must be produced by the producer of the anchor_b work",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "broadcast.content.genre",
                "direction": "reverse",
                "from": "candidate_set",
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
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?candidate_set fb:broadcast.content.genre fb:m.03bv2kt ." in generated_code
    assert "fb:m.03fx9_c fb:broadcast.content.producer ?producer ." in generated_code
    assert "?candidate_set fb:broadcast.content.producer ?producer ." in generated_code
    assert "fb:m.03fx9_c fb:broadcast.content.producer ?candidate_set ." not in generated_code
    assert "?candidate_set fb:broadcast.content.producer ?candidate_set ." not in generated_code
    assert "fb:broadcast.producer.produces" not in generated_code


def test_joined_count_renderer_preserves_cross_relation_bridge_variable() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "m.03bv2kt",
                "resolved_entity_id": "m.03bv2kt",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "m.03fx9_c",
                "resolved_entity_id": "m.03fx9_c",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "content",
        "candidate_set_variable": "content",
        "count_set_variable": "content",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "content",
                    "notes": "Genre constraint: content must be in the Higher Education genre (use broadcast.genre.content)",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "producer",
                    "notes": "Producer constraint: find producer of anchor_b, then require candidate_content to be produced by that producer (pivot through producer)",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "broadcast.genre.content",
                "direction": "forward",
                "from": "Higher Education",
                "to": "content",
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
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
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["count_candidate_content"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.03bv2kt fb:broadcast.genre.content ?content ." in generated_code
    assert "fb:m.03fx9_c fb:broadcast.content.producer ?producer ." in generated_code
    assert "?producer fb:broadcast.producer.produces ?content ." in generated_code
    assert "fb:m.03bv2kt fb:broadcast.producer.produces ?content ." not in generated_code
    assert "fb:m.03fx9_c fb:broadcast.producer.produces ?content ." not in generated_code


def test_joined_count_renderer_specializes_generic_anchor_paths_from_count_constraints() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "m.03bv2kt",
                "resolved_entity_id": "m.03bv2kt",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "m.03fx9_c",
                "resolved_entity_id": "m.03fx9_c",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "content",
        "candidate_set_variable": "content",
        "count_set_variable": "content",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "content",
                    "notes": "Higher Education is treated as a broadcast genre/topic; use broadcast.genre.content (reverse) to get content in that genre.",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "content",
                    "notes": "Get the producer of anchor_b via broadcast.content.producer and use it as a content filter.",
                },
            ],
        },
        "relation_paths": [
            {
                "relation": "broadcast.genre.content",
                "direction": "reverse",
                "from": "genre",
                "to": "content",
                "from_role": "anchor",
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
                "from_role": "anchor",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.03bv2kt fb:broadcast.genre.content ?content ." in generated_code
    assert "fb:m.03fx9_c fb:broadcast.content.producer ?producer ." in generated_code
    assert '?anchor fb:type.object.name ?anchor_label .' not in generated_code
    assert '"genre"' not in generated_code.lower()


def test_joined_count_renderer_counts_explicit_count_set_endpoint_instead_of_bridge() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Richard Altwasser",
                "chosen_alias": "m.050yww4",
                "resolved_entity_id": "m.050yww4",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "answer",
        "candidate_set_variable": "candidate_set",
        "count_set_variable": "count",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor",
                    "constrains_variable": "candidate_set",
                    "notes": "Anchor Richard Altwasser constrains the answer by first selecting designed computers and then their key designers.",
                }
            ],
        },
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
                "to": "key designer",
                "from_role": "candidate_set",
                "to_role": "count_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?key_designer) AS ?count" in generated_code
    assert "fb:m.050yww4 fb:computer.computer_designer.computers_designed ?candidate_set ." in generated_code
    assert "?candidate_set fb:computer.computer.key_designers ?key_designer ." in generated_code
    assert "?candidate_set fb:computer.computer.key_designers ?candidate_set ." not in generated_code


def test_joined_count_renderer_preserves_explicit_anchor_role_even_when_endpoint_matches_candidate() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "Higher Education",
                "chosen_alias": "m.03bv2kt",
                "resolved_entity_id": "m.03bv2kt",
                "role": "anchor_a",
            },
            {
                "surface": "To the Best of Our Knowledge",
                "chosen_alias": "m.03fx9_c",
                "resolved_entity_id": "m.03fx9_c",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "content",
        "candidate_set_variable": "content",
        "count_set_variable": "content",
        "join_structure": {
            "type": "count",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "content",
                    "notes": "Anchor 'Higher Education' filters content via broadcast.genre.content.",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "producer",
                    "notes": "Anchor 'To the Best of Our Knowledge' provides producer(s) via broadcast.content.producer.",
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
                "from_role": "anchor_a",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
        ],
        "projection": ["count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.03fx9_c fb:broadcast.content.producer ?producer ." in generated_code
    assert "?content fb:broadcast.content.producer ?producer ." not in generated_code


def test_reusable_multi_anchor_renderer_binds_specific_anchor_roles() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "anchored_entities": [
            {
                "surface": "the museum of modern art",
                "chosen_alias": "the museum of modern art",
                "role": "anchor_a",
            },
            {
                "surface": "Smithsonian Institution",
                "chosen_alias": "Smithsonian Institution",
                "role": "anchor_b",
            },
        ],
        "relation_paths": [
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "anchor_a",
                "to": "shared_type",
                "from_role": "anchor",
                "to_role": "shared_type",
                "grounding_source": "curated",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "anchor_b",
                "to": "shared_type",
                "from_role": "anchor",
                "to_role": "shared_type",
                "grounding_source": "curated",
            },
        ],
        "projection": ["shared_type"],
        "shared_answer_variable": "shared_type",
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)

    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert 'FILTER(LCASE(STR(?anchor_a_label)) = "the museum of modern art")' in generated_code
    assert 'FILTER(LCASE(STR(?anchor_b_label)) = "smithsonian institution")' in generated_code
    assert '"anchor_a"' not in generated_code
    assert '"anchor_b"' not in generated_code


def test_reusable_count_renderer_prefers_candidate_set_over_shared_answer_alias() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_direct_relation",
        "anchored_entities": [
            {
                "surface": "Hentai",
                "chosen_alias": "Hentai",
                "resolved_entity_id": "m.03p67",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "answer_set",
        "count_set_variable": "answer_count",
        "shared_answer_variable": "answer",
        "relation_paths": [
            {
                "relation": "media_common.media_genre.child_genres",
                "direction": "forward",
                "from": "parent_genre",
                "to": "child_genre",
                "from_role": "anchor",
                "to_role": "answer",
                "grounding_source": "curated",
            }
        ],
        "projection": ["answer_count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?answer_set) AS ?count" in generated_code
    assert "?answer_count" not in generated_code
    assert "COUNT(DISTINCT ?answer) AS ?count" not in generated_code


def test_reusable_joined_count_renderer_prefers_structural_shared_answer_over_helper_count_token() -> None:
    query_plan = {
        "answer_mode": "count",
        "query_shape": "count_over_joined_set",
        "anchored_entities": [
            {
                "surface": "lso",
                "chosen_alias": "m.014hr0",
                "resolved_entity_id": "m.014hr0",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "artist",
        "candidate_set_variable": "artist_set",
        "count_set_variable": "artist_count",
        "relation_paths": [
            {
                "relation": "music.recording_contribution.contributor",
                "direction": "reverse",
                "from": "contribution",
                "to": "m.014hr0",
                "from_role": "count_set",
                "to_role": "anchor",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "music.recording_contribution.recording",
                "direction": "forward",
                "from": "contribution",
                "to": "recording",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "exploratory",
            },
            {
                "relation": "music.recording.artist",
                "direction": "forward",
                "from": "recording",
                "to": "artist",
                "from_role": "anchor",
                "to_role": "count_set",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["artist_count"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": True,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None
    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "COUNT(DISTINCT ?artist) AS ?count" in generated_code
    assert "COUNT(DISTINCT ?contribution) AS ?count" not in generated_code
    assert "COUNT(DISTINCT ?artist_count) AS ?count" not in generated_code
    assert "?contribution fb:music.recording_contribution.contributor fb:m.014hr0 ." in generated_code
    assert "?contribution fb:music.recording_contribution.recording ?recording ." in generated_code
    assert "?recording fb:music.recording.artist ?artist ." in generated_code
    assert "fb:m.014hr0 fb:music.recording_contribution.recording ?artist_count ." not in generated_code
    assert "fb:m.014hr0 fb:music.recording.artist ?artist ." not in generated_code


def test_reusable_multi_anchor_renderer_preserves_pivot_when_same_role_repeats() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "anchored_entities": [
            {
                "surface": "Martin Hug",
                "chosen_alias": "m.010f3n0k",
                "role": "anchor_a",
            },
            {
                "surface": "sci",
                "chosen_alias": "m.04_754t",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "candidate_set",
        "relation_paths": [
            {
                "relation": "business.board_member.organization_board_memberships",
                "direction": "forward",
                "from": "Martin Hug",
                "to": "membership",
                "from_role": "anchor_a",
                "to_role": "shared_answer",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "organization.organization_membership.organization",
                "direction": "reverse",
                "from": "membership",
                "to": "shared_answer",
                "from_role": "shared_answer",
                "to_role": "shared_answer",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["shared_answer"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.010f3n0k fb:business.board_member.organization_board_memberships ?membership_shared_answer ." in generated_code
    assert "?membership_shared_answer fb:organization.organization_membership.organization ?shared_answer ." in generated_code
    assert "?shared_answer fb:organization.organization_membership.organization ?shared_answer ." not in generated_code


def test_reusable_multi_anchor_renderer_expands_anchor_specific_constraint_paths() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "anchored_entities": [
            {
                "surface": "Goat",
                "chosen_alias": "m.03fwl",
                "resolved_entity_id": "m.03fwl",
                "role": "anchor_a",
            },
            {
                "surface": "Cattle",
                "chosen_alias": "m.01xq0k1",
                "resolved_entity_id": "m.01xq0k1",
                "role": "anchor_b",
            },
            {
                "surface": "semi-firm",
                "chosen_alias": "semi-firm",
                "role": "constraint_value",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "shared_answer",
        "relation_paths": [
            {
                "relation": "food.cheese_texture.cheeses",
                "direction": "reverse",
                "from": "texture_value",
                "to": "cheese",
                "from_role": "constraint_value",
                "to_role": "shared_answer",
                "grounding_source": "curated",
            },
            {
                "relation": "food.cheese_milk_source.cheeses",
                "direction": "reverse",
                "from": "milk_source",
                "to": "cheese",
                "from_role": "constraint_value",
                "to_role": "shared_answer",
                "grounding_source": "curated",
            },
        ],
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "shared_answer",
                    "notes": "Cheeses must have Goat listed as a milk source",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "shared_answer",
                    "notes": "Cheeses must have Cattle listed as a milk source",
                },
                {
                    "anchor_role": "constraint_value",
                    "constrains_variable": "shared_answer",
                    "notes": "Cheeses must have texture = semi-firm",
                },
            ],
        },
        "projection": ["shared_answer"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.03fwl fb:food.cheese_milk_source.cheeses ?shared_answer ." in generated_code
    assert "fb:m.01xq0k1 fb:food.cheese_milk_source.cheeses ?shared_answer ." in generated_code
    assert '?constraint_value fb:type.object.name ?constraint_value_label .' in generated_code
    assert 'FILTER(LCASE(STR(?constraint_value_label)) = "semi-firm")' in generated_code
    assert '"milk_source"' not in generated_code


def test_reusable_multi_anchor_renderer_matches_constraint_paths_from_relation_phrase_notes() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "anchored_entities": [
            {
                "surface": "bayer",
                "chosen_alias": "Bayer filter",
                "resolved_entity_id": "m.02r8js",
                "role": "anchor_a",
            },
            {
                "surface": "120",
                "chosen_alias": "120",
                "resolved_entity_id": "m.04pf295",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "candidate_set",
        "candidate_set_variable": "candidate_set",
        "relation_paths": [
            {
                "relation": "digicams.digital_camera.color_filter_array_type",
                "direction": "forward",
                "from": "candidate_set",
                "to": "cfa_type",
                "from_role": "candidate_set",
                "to_role": "constraint_value",
                "grounding_source": "curated",
            },
            {
                "relation": "digicams.digital_camera.iso_setting",
                "direction": "forward",
                "from": "candidate_set",
                "to": "iso_setting_value",
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
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "candidate_set",
                    "notes": "color filter array type must equal Bayer filter on the candidate digital camera",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "candidate_set",
                    "notes": "ISO setting must equal 120 on the candidate digital camera",
                },
            ],
        },
        "projection": ["answer"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?candidate_set fb:digicams.digital_camera.color_filter_array_type fb:m.02r8js ." in generated_code
    assert "?candidate_set fb:digicams.digital_camera.iso_setting fb:m.04pf295 ." in generated_code
    assert "?candidate_set fb:digicams.digital_camera.sensor_type ?answer ." in generated_code
    assert "?candidate_set fb:digicams.digital_camera.color_filter_array_type fb:m.04pf295 ." not in generated_code
    assert "?candidate_set fb:digicams.digital_camera.iso_setting ?anchor_a ." not in generated_code
    assert "?cfa_type" not in generated_code
    assert "?iso_setting_value" not in generated_code


def test_reusable_multi_anchor_renderer_expands_generic_anchor_endpoint_for_each_anchor() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "anchored_entities": [
            {
                "surface": "maltese dog",
                "chosen_alias": "maltese",
                "resolved_entity_id": "m.02cyl6",
                "role": "anchor_a",
            },
            {
                "surface": "Papillon",
                "chosen_alias": "m.01pkw7",
                "resolved_entity_id": "m.01pkw7",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "shared_answer",
        "relation_paths": [
            {
                "relation": "biology.animal_breed.temperament",
                "direction": "forward",
                "from": "breed",
                "to": "temperament",
                "from_role": "anchor",
                "to_role": "shared_answer",
                "grounding_source": "curated",
            },
        ],
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "shared_answer",
                    "notes": "Bind maltese dog as a breed entity and follow biology.animal_breed.temperament to shared temperaments.",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "shared_answer",
                    "notes": "Bind Papillon as a breed entity and follow biology.animal_breed.temperament to shared temperaments.",
                },
            ],
        },
        "projection": ["shared_answer"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.02cyl6 fb:biology.animal_breed.temperament ?shared_answer ." in generated_code
    assert "fb:m.01pkw7 fb:biology.animal_breed.temperament ?shared_answer ." in generated_code
    assert 'FILTER(LCASE(STR(?anchor_label)) = "breed")' not in generated_code


def test_reusable_multi_anchor_renderer_promotes_projected_answer_endpoint() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "multi_anchor_intersection",
        "anchored_entities": [
            {
                "surface": "Naloxone",
                "chosen_alias": "m.011_yk",
                "resolved_entity_id": "m.011_yk",
                "role": "anchor_a",
            },
            {
                "surface": "Enalaprilat",
                "chosen_alias": "m.0gfdnjk",
                "resolved_entity_id": "m.0gfdnjk",
                "role": "anchor_b",
            },
        ],
        "shared_answer_variable": "shared_formulation",
        "candidate_set_variable": "candidate_formulations",
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
                "relation": "medicine.drug_ingredient.active_ingredient_of_formulation",
                "direction": "forward",
                "from": "active_ingredient",
                "to": "formulation",
                "from_role": "constraint_value",
                "to_role": "shared_answer",
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
        "join_structure": {
            "type": "intersection",
            "anchor_constraints": [
                {
                    "anchor_role": "anchor_a",
                    "constrains_variable": "shared_formulation",
                    "notes": "formulation_of = Naloxone via medicine.drug_formulation.formulation_of",
                },
                {
                    "anchor_role": "anchor_b",
                    "constrains_variable": "shared_formulation",
                    "notes": "active ingredient Enalaprilat via medicine.drug_ingredient.active_ingredient_of_formulation",
                },
            ],
        },
        "projection": ["dosage_form", "dosage_form_name"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "?shared_formulation fb:medicine.drug_formulation.dosage_form ?dosage_form ." in generated_code
    assert "OPTIONAL { ?dosage_form fb:type.object.name ?dosage_form_name . }" in generated_code
    assert "?dosage_form_shared_answer" not in generated_code


def test_reusable_superlative_renderer_respects_structural_endpoint_token_over_anchor_role() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "superlative_chain",
        "anchored_entities": [
            {
                "surface": "Hurricane Dolly",
                "chosen_alias": "m.04dn799",
                "resolved_entity_id": "m.04dn799",
                "role": "anchor",
            }
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "candidate_set",
        "ordering_attribute": {
            "relation": "formation_date",
            "direction": "forward",
            "source_variable": "candidate_set",
            "attribute_variable": "ordering_attr",
        },
        "ordering_direction": "max",
        "relation_paths": [
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "m.04dn799",
                "to": "shared_type",
                "from_role": "anchor",
                "to_role": "type_set",
                "grounding_source": "curated",
            },
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "shared_type",
                "to": "candidate_set",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "formation_date",
                "direction": "forward",
                "from": "candidate_set",
                "to": "ordering_attr",
                "from_role": "candidate_set",
                "to_role": "ordering_attribute",
                "grounding_source": "exploratory",
            },
        ],
        "projection": ["shared_answer"],
        "allow_exploratory_predicates": True,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_output = render_reusable_tool(
        query_plan=query_plan,
        selection=selection,
    )
    generated_code = extract_and_validate_code(generated_output)

    assert "fb:m.04dn799 fb:type.object.type ?shared_type ." in generated_code
    assert "?shared_type fb:type.type.instance ?candidate_set ." in generated_code
    assert "fb:m.04dn799 fb:type.type.instance ?candidate_set ." not in generated_code


def test_entity_renderer_does_not_truncate_entity_sets() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "single_anchor_lookup",
        "anchored_entities": [
            {
                "surface": "Josef Fanta",
                "chosen_alias": "Josef Fanta",
                "role": "anchor",
            }
        ],
        "candidate_set_variable": "style",
        "relation_paths": [
            {
                "relation": "architecture.architect.architectural_style",
                "direction": "forward",
                "from": "Josef Fanta",
                "to": "style",
                "from_role": "anchor",
                "to_role": "candidate_set",
                "grounding_source": "dynamic_probe",
            },
            {
                "relation": "architecture.architectural_style.architects",
                "direction": "forward",
                "from": "style",
                "to": "answer",
                "from_role": "candidate_set",
                "to_role": "answer",
                "grounding_source": "dynamic_probe",
            },
        ],
        "projection": ["answer", "answer_name"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_code = extract_and_validate_code(
        render_reusable_tool(query_plan=query_plan, selection=selection)
    )

    assert "LIMIT 50" not in generated_code


def test_shared_type_intersection_renderer_projects_shared_answer_variable() -> None:
    query_plan = {
        "answer_mode": "entity",
        "query_shape": "shared_type_intersection",
        "anchored_entities": [
            {"surface": "Harris Museum", "chosen_alias": "m.03qzy4", "resolved_entity_id": "m.03qzy4", "role": "anchor_a"},
            {"surface": "Chaffee Art Center", "chosen_alias": "m.04_jgm4", "resolved_entity_id": "m.04_jgm4", "role": "anchor_b"},
        ],
        "shared_answer_variable": "shared_answer",
        "candidate_set_variable": "candidate_set",
        "relation_paths": [
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "anchor_a",
                "to": "type_a",
                "from_role": "anchor_a",
                "to_role": "type_set",
                "grounding_source": "curated",
            },
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "type_a",
                "to": "candidate_set_A",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
            {
                "relation": "type.object.type",
                "direction": "forward",
                "from": "anchor_b",
                "to": "type_b",
                "from_role": "anchor_b",
                "to_role": "type_set",
                "grounding_source": "curated",
            },
            {
                "relation": "type.type.instance",
                "direction": "forward",
                "from": "type_b",
                "to": "candidate_set_B",
                "from_role": "type_set",
                "to_role": "candidate_set",
                "grounding_source": "curated",
            },
        ],
        "projection": ["shared_answer", "shared_answer_name"],
        "ordering_attribute": {},
        "allow_exploratory_predicates": False,
    }

    selection = select_reusable_tool(query_plan)
    assert selection is not None

    generated_code = extract_and_validate_code(
        render_reusable_tool(query_plan=query_plan, selection=selection)
    )

    assert "?type_a fb:type.type.instance ?shared_answer ." in generated_code
    assert "?type_b fb:type.type.instance ?shared_answer ." in generated_code
    assert "?candidate_set " not in generated_code
