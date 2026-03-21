"""Probe script part 3: proper sequencing (get_relations before get_neighbors)
and full semantic filter test.

Usage:
    conda run -n lifelong python debug_percussionist3.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI
from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor
from src.self_evolving_agent import kg_utils as _kg_utils

SPARQL_URL = os.getenv("KG_SPARQL_URL", "http://127.0.0.1:3001/kb/sparql")
ONTOLOGY_DIR = os.getenv("KG_ONTOLOGY_DIR", "data/v0121/knowledge_graph/ontology")
SEP = "#" * 60


def make_actions_spec(api: KnowledgeGraphAPI) -> dict:
    # Stateful wrapper: calls get_relations automatically before get_neighbors
    def safe_get_neighbors(entity_or_var, relation: str) -> str:
        try:
            api.get_relations(entity_or_var)
        except Exception:
            pass  # may already be in state; continue
        _, msg = api.get_neighbors(entity_or_var, relation)
        return msg

    return {
        "get_relations": lambda e: api.get_relations(e)[1],
        "get_neighbors": safe_get_neighbors,
        "intersection": lambda v1, v2: KnowledgeGraphAPI.intersection(v1, v2)[1],
        "union": lambda v1, v2: KnowledgeGraphAPI.union(v1, v2)[1],
        "difference": lambda v1, v2: KnowledgeGraphAPI.difference(v1, v2)[1],
        "get_attributes": lambda v: api.get_attributes(v)[1],
        "argmax": lambda v, a: api.argmax(v, a)[1],
        "argmin": lambda v, a: api.argmin(v, a)[1],
        "count": lambda v: api.count(v)[1],
        "_api": api,
    }


def main() -> None:
    print(f"SPARQL: {SPARQL_URL}\n")
    executor = SparqlExecutor(SPARQL_URL)
    api = KnowledgeGraphAPI(ONTOLOGY_DIR, executor)
    actions_spec = make_actions_spec(api)
    kg = _kg_utils.get_macro_helper_facade()

    # ── STEP A: Prime state + get percussionist people.person variable ─────────
    print(SEP)
    print("STEP A: get_relations('Percussionist') [precondition]")
    print(SEP)
    _, rel_msg = api.get_relations("Percussionist")
    print(rel_msg)

    print(f"\n{SEP}")
    print("STEP A2: get_neighbors('Percussionist', 'people.profession.people_with_this_profession')")
    print(SEP)
    perc_var, perc_msg = api.get_neighbors(
        "Percussionist", "people.profession.people_with_this_profession"
    )
    print(f"Message: {perc_msg}")
    perc_id = getattr(perc_var, "id", None)
    perc_type = getattr(perc_var, "type", None)
    print(f"Variable id  : {perc_id}  →  string: '#{perc_id}'")
    print(f"Variable type: {perc_type}")

    perc_id_str = f"#{perc_id}" if perc_id is not None else None

    # ── STEP B: resolve_semantic_filter on perc variable → 'songwriter' ───────
    print(f"\n{SEP}")
    print("STEP B: resolve_semantic_filter(perc_var → target='songwriter')")
    print(SEP)
    if perc_id_str:
        for hints in [
            ["people.person", "music.artist"],
            None,
            ["music.artist.profession"],
        ]:
            print(f"\n  domain_hints={hints}")
            res = kg.resolve_semantic_filter(
                base_var=perc_id_str,
                target_concept="songwriter",
                variable_list=None,
                domain_hints=hints,
                asked_for="songwriters who are percussionists",
                max_type_candidates=12,
            )
            ids = kg.extract_var_ids(res)
            print(f"  Result : {res}")
            print(f"  IDs    : {ids}")
    else:
        print("SKIPPED — perc_var is None")

    # ── STEP C: resolve_entity_to_vars with actions_spec ─────────────────────
    print(f"\n{SEP}")
    print("STEP C: resolve_entity_to_vars('Percussionist', target_concept='people.profession', actions_spec=...)")
    print(SEP)
    for tc in [None, "people.profession", "music.musician_profession"]:
        res = kg.resolve_entity_to_vars(
            "Percussionist", tc, actions_spec, domain_hints=None, max_k=5
        )
        ids = kg.extract_var_ids(res)
        print(f"  target_concept={tc!r}: vars={ids}  type={res.get('type')}  candidates={list(res.get('candidate_map',{}).keys())[:5]}")

    # ── STEP D: get_attributes on perc_var to see what fields describe people ─
    print(f"\n{SEP}")
    print("STEP D: get_attributes(perc_var) — inspect the people.person variable")
    print(SEP)
    if perc_var:
        try:
            _, attr_msg = api.get_attributes(perc_var)
            print(attr_msg[:800])
        except Exception as exc:
            print(f"EXCEPTION: {exc}")

    # ── STEP E: get_relations of perc_var (what can we traverse from people?) ─
    print(f"\n{SEP}")
    print("STEP E: get_relations(perc_var) — what relations exist on the people.person set?")
    print(SEP)
    if perc_var:
        try:
            _, var_rel_msg = api.get_relations(perc_var)
            print(var_rel_msg[:800])
        except Exception as exc:
            print(f"EXCEPTION: {exc}")

    # ── STEP F: walk_to_target from perc variable ─────────────────────────────
    print(f"\n{SEP}")
    print("STEP F: walk_to_target(perc_var → 'songwriter')")
    print(SEP)
    if perc_id_str:
        walk_res = kg.walk_to_target(
            actions_spec, [perc_id_str], "songwriter",
            domain_hints=["people.profession", "music"], max_calls=8
        )
        walk_ids = kg.extract_var_ids(walk_res)
        print(f"Walk result: {walk_res}")
        print(f"Walk IDs   : {walk_ids}")
        if walk_ids:
            count_msg = actions_spec["count"](walk_ids[0])
            print(f"COUNT: {count_msg}")

    # ── STEP G: Direct neighbor probe — music.artist.profession ───────────────
    print(f"\n{SEP}")
    print("STEP G: get_neighbors(perc_var, 'music.artist.profession') → find profession types")
    print(SEP)
    if perc_var:
        try:
            api.get_relations(perc_var)  # precondition
            _, profn_msg = api.get_neighbors(perc_var, "music.artist.profession")
            print(f"music.artist.profession: {profn_msg[:600]}")
        except Exception as exc:
            print(f"EXCEPTION: {exc}")

        # Also check people.person.profession
        try:
            _, profn2_msg = api.get_neighbors(perc_var, "people.person.profession")
            print(f"\npeople.person.profession: {profn2_msg[:600]}")
        except Exception as exc:
            print(f"people.person.profession EXCEPTION: {exc}")

    # ── STEP H: Count percussionists as sanity check ──────────────────────────
    print(f"\n{SEP}")
    print("STEP H: count(perc_var) — sanity check total percussionists")
    print(SEP)
    if perc_var:
        try:
            _, cnt_msg = api.count(perc_var)
            print(f"Total percussionists: {cnt_msg}")
        except Exception as exc:
            print(f"EXCEPTION: {exc}")

    print(f"\n{SEP}")
    print("DONE")
    print(SEP)


if __name__ == "__main__":
    main()
