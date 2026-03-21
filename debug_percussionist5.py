"""Probe part 5: get actual count numbers, find songwriter entity via SPARQL.

Usage:
    conda run -n lifelong python debug_percussionist5.py
"""
from __future__ import annotations
import os, sys, inspect
sys.path.insert(0, os.path.dirname(__file__))

from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI
from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor

SPARQL_URL = os.getenv("KG_SPARQL_URL", "http://127.0.0.1:3001/kb/sparql")
ONTOLOGY_DIR = os.getenv("KG_ONTOLOGY_DIR", "data/v0121/knowledge_graph/ontology")
S = "#" * 60


def sparql_query(sparql: SparqlExecutor, query: str) -> list:
    """Find and call the right query method on SparqlExecutor."""
    methods = [m for m in dir(sparql) if not m.startswith("_")]
    for name in methods:
        fn = getattr(sparql, name)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                if "query" in params or "sparql" in params or len(params) == 1:
                    result = fn(query)
                    if result is not None:
                        return result
            except Exception:
                pass
    return []


def main() -> None:
    print(f"SPARQL: {SPARQL_URL}\n")
    executor = SparqlExecutor(SPARQL_URL)
    api = KnowledgeGraphAPI(ONTOLOGY_DIR, executor)

    # Show SparqlExecutor methods
    print(S); print("SparqlExecutor API"); print(S)
    methods = [(m, inspect.signature(getattr(executor, m))) for m in dir(executor) if not m.startswith("_") and callable(getattr(executor, m))]
    for name, sig in methods:
        print(f"  {name}{sig}")

    # ── Use KnowledgeGraphAPI internal SPARQL access ──────────────────────────
    print(f"\n{S}"); print("STEP A: List all people.profession entities via SPARQL"); print(S)
    # Try the raw sparql through KG api's internal path
    try:
        from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor as SE
        sparql = SE(SPARQL_URL)
        method_names = [m for m in dir(sparql) if not m.startswith("_")]
        print(f"  SparqlExecutor methods: {method_names}")
    except Exception as exc:
        print(f"  {exc}")

    # ── STEP B: Build the full chain manually and extract count values ─────────
    print(f"\n{S}"); print("STEP B: Full chain with actual variable IDs from messages"); print(S)

    # Use the API's execute method if it exists, or access via internal sparql attr
    sparql_inner = None
    for attr in ["_executor", "_sparql", "executor", "sparql", "_sparql_executor"]:
        obj = getattr(api, attr, None)
        if obj is not None:
            sparql_inner = obj
            print(f"  Found internal SPARQL via api.{attr}: {type(obj)}")
            print(f"  Methods: {[m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))]}")
            break

    # ── STEP C: Full chain using message parsing to get variable IDs ──────────
    print(f"\n{S}"); print("STEP C: Full chain using message-based variable tracking"); print(S)

    import re

    def msg_to_var_id(msg: str) -> str | None:
        """Parse 'Variable #N' from API message."""
        m = re.search(r"Variable #(\d+)", msg)
        return f"#{m.group(1)}" if m else None

    # Variable list tracks all minted variables by API
    var_list = api.variable_list if hasattr(api, "variable_list") else None
    print(f"  api.variable_list attr: {type(var_list)}")
    if var_list is not None:
        print(f"  variable_list contents: {var_list[:5] if hasattr(var_list, '__iter__') else var_list}")

    # Check all api attributes for variable tracking
    print(f"  All api public attrs: {[(a, type(getattr(api,a))) for a in dir(api) if not a.startswith('_') and not callable(getattr(api,a))]}")

    # ── STEP D: Run chain and capture Variable objects ────────────────────────
    print(f"\n{S}"); print("STEP D: Run full chain, inspect Variable objects"); print(S)

    # Step D1: get percussionist people
    api.get_relations("Percussionist")
    perc_var, perc_msg = api.get_neighbors(
        "Percussionist", "people.profession.people_with_this_profession"
    )
    print(f"perc_var.program: {getattr(perc_var, 'program', None)}")
    print(f"perc_var.dict()  : {perc_var.dict() if hasattr(perc_var, 'dict') else 'N/A'}")
    print(f"perc msg: {perc_msg}")

    # Step D2: count percussionists
    perc_count_var, perc_cnt_msg = api.count(perc_var)
    print(f"\nperc count msg: {perc_cnt_msg}")
    print(f"perc count var.dict(): {perc_count_var.dict() if perc_count_var and hasattr(perc_count_var, 'dict') else 'None'}")

    # Step D3: try Composer as songwriter proxy
    print(f"\n--- Trying 'Composer' as songwriter proxy ---")
    api.get_relations("Composer")
    comp_var, comp_msg = api.get_neighbors(
        "Composer", "people.profession.people_with_this_profession"
    )
    print(f"comp msg: {comp_msg}")
    print(f"comp_var.program: {getattr(comp_var, 'program', None)}")

    comp_count_var, comp_cnt_msg = api.count(comp_var)
    print(f"comp count: {comp_cnt_msg}")

    inter_var, inter_msg = KnowledgeGraphAPI.intersection(perc_var, comp_var)
    print(f"perc ∩ comp: {inter_msg}")
    if inter_var:
        _, inter_cnt_msg = api.count(inter_var)
        print(f"COUNT(perc ∩ comp): {inter_cnt_msg}")
        print(f"inter_var.dict(): {inter_var.dict() if hasattr(inter_var, 'dict') else 'N/A'}")

    # ── STEP E: SPARQL search for profession entities with "song" in label ────
    print(f"\n{S}"); print("STEP E: SPARQL label search for profession entities"); print(S)
    if sparql_inner is not None:
        sparql_method = None
        for m in dir(sparql_inner):
            if not m.startswith("_") and callable(getattr(sparql_inner, m)):
                sparql_method = m
                break
        print(f"  Using method: {sparql_method}")
        if sparql_method:
            query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?s ?label WHERE {
  ?s rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "song"))
} LIMIT 30
"""
            try:
                bindings = getattr(sparql_inner, sparql_method)(query)
                print(f"  song-label results: {bindings[:20]}")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")

            # Also list all music.musician_profession entities
            query2 = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?label WHERE {
  ?s rdf:type <http://rdf.freebase.com/ns/music.musician_profession> .
  OPTIONAL { ?s rdfs:label ?label }
} LIMIT 40
"""
            try:
                bindings2 = getattr(sparql_inner, sparql_method)(query2)
                print(f"\n  music.musician_profession entities: {bindings2}")
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")

    print(f"\n{S}"); print("DONE"); print(S)


if __name__ == "__main__":
    main()
