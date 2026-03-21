"""Probe script: trace Percussionist × Songwriter intersection step-by-step.

Usage (from repo root, with lifelong conda env):
    conda run -n lifelong python debug_percussionist.py
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
    """Build a callable actions_spec that mirrors the task server proxy."""
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
        # raw API access for probe use
        "_api": api,
    }


def probe_entity(api: KnowledgeGraphAPI, entity: str) -> None:
    print(f"\n{SEP}")
    print(f"PROBE: get_relations('{entity}')")
    print(SEP)
    try:
        _, msg = api.get_relations(entity)
        print(msg[:1200])
    except Exception as exc:
        print(f"  EXCEPTION: {type(exc).__name__}: {exc}")


def probe_neighbors(api: KnowledgeGraphAPI, entity: str, relation: str):
    print(f"\n{SEP}")
    print(f"PROBE: get_neighbors('{entity}', '{relation}')")
    print(SEP)
    try:
        var, msg = api.get_neighbors(entity, relation)
        print(msg[:1200])
        return var, msg
    except Exception as exc:
        print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
        return None, str(exc)


def main() -> None:
    print(f"SPARQL endpoint : {SPARQL_URL}")
    print(f"Ontology dir    : {ONTOLOGY_DIR}")

    executor = SparqlExecutor(SPARQL_URL)
    api = KnowledgeGraphAPI(ONTOLOGY_DIR, executor)
    actions_spec = make_actions_spec(api)
    kg = _kg_utils.get_macro_helper_facade()

    # ── STEP 1: Resolve "Percussionist" ──────────────────────────────────────
    print(f"\n{'#'*60}")
    print("STEP 1: resolve_entity_to_vars('Percussionist', target_concept=None)")
    print('#'*60)
    perc_res = kg.resolve_entity_to_vars(
        "Percussionist", None, actions_spec, domain_hints=None, max_k=5
    )
    print(f"Full result dict: {perc_res}")
    perc_ids = kg.extract_var_ids(perc_res)
    print(f"Extracted IDs   : {perc_ids}")

    # ── STEP 1b: Try 'percussionist' (lowercase) ────────────────────────────
    print(f"\n{'#'*60}")
    print("STEP 1b: resolve_entity_to_vars('percussionist' lowercase)")
    print('#'*60)
    perc_res2 = kg.resolve_entity_to_vars(
        "percussionist", None, actions_spec, domain_hints=None, max_k=5
    )
    print(f"Full result dict: {perc_res2}")
    perc_ids2 = kg.extract_var_ids(perc_res2)
    print(f"Extracted IDs   : {perc_ids2}")

    # ── STEP 1c: Raw get_relations for both casings ─────────────────────────
    probe_entity(api, "Percussionist")
    probe_entity(api, "percussionist")

    # ── STEP 2: Walk Percussionist → people.person ───────────────────────────
    best_perc_ids = perc_ids or perc_ids2
    print(f"\n{'#'*60}")
    print(f"STEP 2: walk_to_target from {best_perc_ids} → 'people.person'")
    print('#'*60)
    if best_perc_ids:
        walk_res = kg.walk_to_target(
            actions_spec, best_perc_ids, "people.person", domain_hints=None, max_calls=8
        )
        print(f"Walk result: {walk_res}")
        walk_ids = kg.extract_var_ids(walk_res)
        print(f"Walk IDs   : {walk_ids}")
    else:
        print("SKIPPED — Percussionist did not resolve to any variable.")
        walk_ids = []

    # ── STEP 3: Direct neighbor probe for profession→people relation ─────────
    print(f"\n{'#'*60}")
    print("STEP 3: Direct neighbor probes for Percussionist profession relations")
    print('#'*60)
    perc_var, _ = probe_neighbors(api, "Percussionist", "people.profession.people_with_this_profession")
    perc_var_lc, _ = probe_neighbors(api, "percussionist", "people.profession.people_with_this_profession")

    # ── STEP 4: Resolve "songwriter" ─────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("STEP 4: resolve_entity_to_vars('songwriter', target_concept=None)")
    print('#'*60)
    song_res = kg.resolve_entity_to_vars(
        "songwriter", None, actions_spec, domain_hints=None, max_k=5
    )
    print(f"Full result dict: {song_res}")
    song_ids = kg.extract_var_ids(song_res)
    print(f"Extracted IDs   : {song_ids}")

    # Also try capitalised
    print(f"\n{'#'*60}")
    print("STEP 4b: resolve_entity_to_vars('Songwriter' capitalised)")
    print('#'*60)
    song_res2 = kg.resolve_entity_to_vars(
        "Songwriter", None, actions_spec, domain_hints=None, max_k=5
    )
    print(f"Full result dict: {song_res2}")
    song_ids2 = kg.extract_var_ids(song_res2)
    print(f"Extracted IDs   : {song_ids2}")

    probe_entity(api, "songwriter")
    probe_entity(api, "Songwriter")

    # ── STEP 5: resolve_semantic_filter on Percussionist anchor → songwriters ─
    print(f"\n{'#'*60}")
    print("STEP 5: resolve_semantic_filter(perc anchor → 'songwriter')")
    print('#'*60)
    anchor_ids = best_perc_ids
    if anchor_ids:
        filter_res = kg.resolve_semantic_filter(
            base_var=anchor_ids[0],
            target_concept="songwriter",
            variable_list=None,
            domain_hints=["people.person", "music"],
            asked_for="songwriters who are percussionists",
            max_type_candidates=8,
        )
        print(f"Filter result: {filter_res}")
        filter_ids = kg.extract_var_ids(filter_res)
        print(f"Filter IDs   : {filter_ids}")
    else:
        print("SKIPPED — no Percussionist anchor variable.")
        filter_ids = []

    # ── STEP 6: Walk songwriter → people.person then cross_intersect ─────────
    best_song_ids = song_ids or song_ids2
    print(f"\n{'#'*60}")
    print(f"STEP 6: walk_to_target from {best_song_ids} → 'people.person'")
    print('#'*60)
    if best_song_ids:
        walk_song = kg.walk_to_target(
            actions_spec, best_song_ids, "people.person", domain_hints=None, max_calls=8
        )
        print(f"Walk result: {walk_song}")
        walk_song_ids = kg.extract_var_ids(walk_song)
        print(f"Walk IDs   : {walk_song_ids}")
    else:
        print("SKIPPED — songwriter did not resolve.")
        walk_song_ids = []

    # ── STEP 7: cross_intersect ───────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"STEP 7: cross_intersect {walk_ids} × {walk_song_ids}")
    print('#'*60)
    if walk_ids and walk_song_ids:
        inter_res = kg.cross_intersect(actions_spec, walk_ids, walk_song_ids, max_calls=12)
        print(f"Intersect result: {inter_res}")
        inter_ids = kg.extract_var_ids(inter_res)
        print(f"Intersect IDs   : {inter_ids}")
        if inter_ids:
            count_msg = actions_spec["count"](inter_ids[0])
            print(f"\nCOUNT result: {count_msg}")
    else:
        print("SKIPPED — one or both walk sets are empty.")

    # ── STEP 8: Probe alternate direct neighbor path for songwriter ───────────
    print(f"\n{'#'*60}")
    print("STEP 8: Direct neighbor probes for songwriter profession relations")
    print('#'*60)
    probe_neighbors(api, "songwriter", "people.profession.people_with_this_profession")
    probe_neighbors(api, "Songwriter", "people.profession.people_with_this_profession")

    print(f"\n{'#'*60}")
    print("DONE")
    print('#'*60)


if __name__ == "__main__":
    main()
