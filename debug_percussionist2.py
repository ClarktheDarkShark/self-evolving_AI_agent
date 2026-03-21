"""Probe script part 2: find correct songwriter entity name and test semantic filter.

Usage (from repo root):
    conda run -n lifelong python debug_percussionist2.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI
from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor
from src.self_evolving_agent import kg_utils as _kg_utils

SPARQL_URL = os.getenv("KG_SPARQL_URL", "http://127.0.0.1:3001/kb/sparql")
ONTOLOGY_DIR = os.getenv("KG_ONTOLOGY_DIR", "data/v0121/knowledge_graph/ontology")
SEP = "=" * 60


def make_actions_spec(api: KnowledgeGraphAPI) -> dict:
    return {
        "get_relations": lambda entity: api.get_relations(entity)[1],
        "get_neighbors": lambda entity, relation: api.get_neighbors(entity, relation)[1],
        "intersection": lambda v1, v2: KnowledgeGraphAPI.intersection(v1, v2)[1],
        "union": lambda v1, v2: KnowledgeGraphAPI.union(v1, v2)[1],
        "difference": lambda v1, v2: KnowledgeGraphAPI.difference(v1, v2)[1],
        "get_attributes": lambda var: api.get_attributes(var)[1],
        "argmax": lambda var, attr: api.argmax(var, attr)[1],
        "argmin": lambda var, attr: api.argmin(var, attr)[1],
        "count": lambda var: api.count(var)[1],
        "_api": api,
    }


def main() -> None:
    print(f"SPARQL: {SPARQL_URL}\n")
    executor = SparqlExecutor(SPARQL_URL)
    api = KnowledgeGraphAPI(ONTOLOGY_DIR, executor)
    actions_spec = make_actions_spec(api)
    kg = _kg_utils.get_macro_helper_facade()

    # ── STEP A: Get the percussionist people.person variable directly ─────────
    print(f"{'#'*60}")
    print("STEP A: get_neighbors('Percussionist', 'people.profession.people_with_this_profession')")
    print(f"{'#'*60}")
    try:
        perc_var, perc_msg = api.get_neighbors(
            "Percussionist", "people.profession.people_with_this_profession"
        )
        print(f"Message: {perc_msg}")
        print(f"Variable id  : {getattr(perc_var, 'id', None)}")
        print(f"Variable type: {getattr(perc_var, 'type', None)}")
    except Exception as exc:
        print(f"EXCEPTION: {exc}")
        perc_var = None

    # ── STEP B: Probe songwriter entity name candidates ───────────────────────
    print(f"\n{'#'*60}")
    print("STEP B: Probe songwriter entity name candidates")
    print(f"{'#'*60}")
    candidates = [
        "Songwriter", "songwriter", "Singer-songwriter", "singer-songwriter",
        "Singer/songwriter", "Songwriting", "songwriting", "Lyricist", "lyricist",
    ]
    valid_entities = []
    for name in candidates:
        try:
            _, msg = api.get_relations(name)
            print(f"  '{name}': VALID — {msg[:200]}")
            valid_entities.append(name)
        except Exception as exc:
            print(f"  '{name}': INVALID — {str(exc)[:100]}")

    # ── STEP C: resolve_semantic_filter on perc_var → 'songwriter' ───────────
    print(f"\n{'#'*60}")
    print("STEP C: resolve_semantic_filter(perc_var → 'songwriter')")
    print(f"{'#'*60}")
    if perc_var is not None:
        perc_id_str = f"#{perc_var.id}" if hasattr(perc_var, 'id') else str(perc_var)
        print(f"Using perc anchor var: {perc_id_str}")
        filter_res = kg.resolve_semantic_filter(
            base_var=perc_id_str,
            target_concept="songwriter",
            variable_list=None,
            domain_hints=["people.person", "music.artist"],
            asked_for="songwriters who are percussionists",
            max_type_candidates=12,
        )
        print(f"Filter result full: {filter_res}")
        filter_ids = kg.extract_var_ids(filter_res)
        print(f"Filter IDs   : {filter_ids}")

        # Also try with domain_hints=None
        print("\n--- resolve_semantic_filter (no domain_hints) ---")
        filter_res2 = kg.resolve_semantic_filter(
            base_var=perc_id_str,
            target_concept="songwriter",
            variable_list=None,
            domain_hints=None,
            asked_for="songwriters",
            max_type_candidates=12,
        )
        print(f"Filter result full: {filter_res2}")
        filter_ids2 = kg.extract_var_ids(filter_res2)
        print(f"Filter IDs   : {filter_ids2}")
    else:
        print("SKIPPED — no perc_var from step A.")
        filter_ids = []

    # ── STEP D: If any songwriter entity name found, try get_neighbors ────────
    print(f"\n{'#'*60}")
    print("STEP D: get_neighbors for valid songwriter entities")
    print(f"{'#'*60}")
    song_var = None
    for name in valid_entities:
        try:
            var, msg = api.get_neighbors(
                name, "people.profession.people_with_this_profession"
            )
            print(f"  '{name}': {msg[:300]}")
            if var and song_var is None:
                song_var = var
        except Exception as exc:
            print(f"  '{name}': EXCEPTION — {str(exc)[:120]}")

    # ── STEP E: Intersect perc_var × song_var if both exist ──────────────────
    print(f"\n{'#'*60}")
    print("STEP E: Direct intersection perc_people × songwriter_people")
    print(f"{'#'*60}")
    if perc_var is not None and song_var is not None:
        try:
            inter_var, inter_msg = KnowledgeGraphAPI.intersection(perc_var, song_var)
            print(f"Intersection: {inter_msg}")
            if inter_var:
                count_msg = api.count(inter_var)[1]
                print(f"COUNT: {count_msg}")
        except Exception as exc:
            print(f"EXCEPTION: {exc}")
    else:
        print(f"SKIPPED — perc_var={perc_var is not None}, song_var={song_var is not None}")

    # ── STEP F: Specialization walk from Percussionist ────────────────────────
    print(f"\n{'#'*60}")
    print("STEP F: get_neighbors('Percussionist', 'people.profession.specialization_of')")
    print(f"{'#'*60}")
    try:
        spec_var, spec_msg = api.get_neighbors(
            "Percussionist", "people.profession.specialization_of"
        )
        print(f"Result: {spec_msg}")
        if spec_var:
            print(f"  Specialization id  : {getattr(spec_var, 'id', None)}")
            print(f"  Specialization type: {getattr(spec_var, 'type', None)}")
            # Get relations of the specialization result
            _, rel_msg = api.get_relations(spec_var)
            print(f"  Relations of spec: {rel_msg[:400]}")
    except Exception as exc:
        print(f"EXCEPTION: {exc}")

    # ── STEP G: Check if perc_var's entity names contain "songwriter" ─────────
    print(f"\n{'#'*60}")
    print("STEP G: get_attributes on perc_var (people who are percussionists)")
    print(f"{'#'*60}")
    if perc_var is not None:
        try:
            _, attr_msg = api.get_attributes(perc_var)
            print(f"Attributes: {attr_msg[:600]}")
        except Exception as exc:
            print(f"EXCEPTION: {exc}")

    print(f"\n{'#'*60}")
    print("DONE")
    print(f"{'#'*60}")
    print(f"\nSUMMARY:")
    print(f"  Percussionist direct people.person var: {'YES' if perc_var else 'NO'}")
    print(f"  Valid songwriter entity names found   : {valid_entities}")
    print(f"  Songwriter people.person var          : {'YES' if song_var else 'NO'}")


if __name__ == "__main__":
    main()
