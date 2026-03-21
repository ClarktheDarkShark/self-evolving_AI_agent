"""Probe part 4: find songwriter via music relations, test resolve_semantic_filter
with variable object (not string ID), probe cross_intersect directly.

Usage:
    conda run -n lifelong python debug_percussionist4.py
"""
from __future__ import annotations
import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))

from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI
from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor
from src.self_evolving_agent import kg_utils as _kg_utils

SPARQL_URL = os.getenv("KG_SPARQL_URL", "http://127.0.0.1:3001/kb/sparql")
ONTOLOGY_DIR = os.getenv("KG_ONTOLOGY_DIR", "data/v0121/knowledge_graph/ontology")
S = "#" * 60


def make_actions_spec(api: KnowledgeGraphAPI) -> dict:
    def safe_neighbors(entity_or_var, relation: str) -> str:
        try:
            api.get_relations(entity_or_var)
        except Exception:
            pass
        _, msg = api.get_neighbors(entity_or_var, relation)
        return msg

    def safe_count(var) -> str:
        _, msg = api.count(var)
        return msg

    return {
        "get_relations": lambda e: api.get_relations(e)[1],
        "get_neighbors": safe_neighbors,
        "intersection": lambda v1, v2: KnowledgeGraphAPI.intersection(v1, v2)[1],
        "union": lambda v1, v2: KnowledgeGraphAPI.union(v1, v2)[1],
        "difference": lambda v1, v2: KnowledgeGraphAPI.difference(v1, v2)[1],
        "get_attributes": lambda v: api.get_attributes(v)[1],
        "argmax": lambda v, a: api.argmax(v, a)[1],
        "argmin": lambda v, a: api.argmin(v, a)[1],
        "count": safe_count,
        "_api": api,
    }


def get_var_id_from_msg(msg: str) -> str | None:
    m = re.search(r"Variable #(\d+)", msg)
    return f"#{m.group(1)}" if m else None


def main() -> None:
    print(f"SPARQL: {SPARQL_URL}\n")
    executor = SparqlExecutor(SPARQL_URL)
    api = KnowledgeGraphAPI(ONTOLOGY_DIR, executor)
    actions_spec = make_actions_spec(api)
    kg = _kg_utils.get_macro_helper_facade()

    # ── Get percussionist people.person variable (correct sequence) ───────────
    print(S); print("SETUP: Get percussionist people variable"); print(S)
    api.get_relations("Percussionist")
    perc_var, perc_msg = api.get_neighbors(
        "Percussionist", "people.profession.people_with_this_profession"
    )
    print(f"perc_msg: {perc_msg}")
    print(f"perc_var repr: {repr(perc_var)}")
    print(f"perc_var type attr: {getattr(perc_var, 'type', None)}")
    print(f"perc_var dir: {[a for a in dir(perc_var) if not a.startswith('_')]}")

    # ── STEP A: Try music.musician_profession entities for songwriter ──────────
    print(f"\n{S}"); print("STEP A: Probe music.musician_profession entity names for songwriter"); print(S)
    music_candidates = [
        "Songwriter", "songwriter", "Singer-songwriter", "singer-songwriter",
        "Composer", "composer", "Lyricist", "Arranger", "Music producer",
        "Multi-instrumentalist", "Vocalist", "Instrumentalist",
    ]
    valid_music_entities = []
    for name in music_candidates:
        try:
            _, msg = api.get_relations(name)
            if "music.musician_profession" in msg or "people.profession" in msg:
                print(f"  VALID+MUSIC '{name}': {msg[:200]}")
                valid_music_entities.append(name)
            else:
                print(f"  VALID-nomatch '{name}': {msg[:120]}")
                valid_music_entities.append(name)
        except Exception as exc:
            print(f"  INVALID '{name}'")

    # ── STEP B: resolve_semantic_filter passing perc_var directly ────────────
    print(f"\n{S}"); print("STEP B: resolve_semantic_filter with perc_var object directly"); print(S)
    for tc, hints in [
        ("songwriter", ["people.profession", "music.artist"]),
        ("songwriters", None),
        ("Singer-songwriter", None),
    ]:
        print(f"\n  target_concept={tc!r}  domain_hints={hints}")
        try:
            # Pass the actual variable object instead of a string ID
            res = kg.resolve_semantic_filter(
                base_var=perc_var,
                target_concept=tc,
                variable_list=None,
                domain_hints=hints,
                asked_for="songwriters who are percussionists",
                max_type_candidates=12,
            )
            ids = kg.extract_var_ids(res)
            print(f"  Result: {res}")
            print(f"  IDs   : {ids}")
            if ids:
                cnt = actions_spec["count"](ids[0])
                print(f"  COUNT : {cnt}")
        except Exception as exc:
            print(f"  EXCEPTION: {type(exc).__name__}: {exc}")

    # ── STEP C: Try getting people from songwriter music path ─────────────────
    print(f"\n{S}"); print("STEP C: songwriter via music.artist.profession reverse path"); print(S)
    for name in valid_music_entities:
        for rel in ["music.musician_profession.artists", "people.profession.people_with_this_profession"]:
            try:
                v, msg = api.get_neighbors(name, rel)
                print(f"  '{name}' + '{rel}': {msg[:250]}")
                if v:
                    # Try intersection with perc_var
                    try:
                        inter_v, inter_msg = KnowledgeGraphAPI.intersection(perc_var, v)
                        print(f"    → intersection: {inter_msg[:200]}")
                        if inter_v:
                            _, cnt_msg = api.count(inter_v)
                            print(f"    → count: {cnt_msg}")
                    except Exception as ie:
                        print(f"    → intersection EXCEPTION: {ie}")
            except Exception as exc:
                pass  # silently skip invalid combos

    # ── STEP D: SPARQL direct search for "songwriter" entity ─────────────────
    print(f"\n{S}"); print("STEP D: Direct SPARQL query for songwriter-like entities"); print(S)
    from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor as SE
    sparql = SE(SPARQL_URL)

    # Search for entities with label containing "songwriter" (case-insensitive)
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?s ?label WHERE {
      ?s rdfs:label ?label .
      FILTER(CONTAINS(LCASE(?label), "songwriter"))
    } LIMIT 20
    """
    try:
        bindings = sparql.execute(query)
        if bindings:
            for b in bindings[:15]:
                print(f"  {b}")
        else:
            print("  NO RESULTS — 'songwriter' does not appear in any rdfs:label")
    except Exception as exc:
        print(f"  SPARQL EXCEPTION: {exc}")

    # Also search music.musician_profession entries
    query2 = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?s ?label WHERE {
      ?s rdf:type <http://rdf.freebase.com/ns/music.musician_profession> .
      ?s rdfs:label ?label .
    } LIMIT 30
    """
    try:
        bindings2 = sparql.execute(query2)
        print(f"\n  All music.musician_profession entities:")
        for b in (bindings2 or [])[:30]:
            print(f"    {b}")
    except Exception as exc:
        print(f"  SPARQL2 EXCEPTION: {exc}")

    print(f"\n{S}"); print("DONE"); print(S)


if __name__ == "__main__":
    main()
