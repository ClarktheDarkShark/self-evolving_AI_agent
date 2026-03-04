"""Temporary debug script: trace Goat × Cow cheese intersection step-by-step.

Usage (from repo root):
    python debug_kg.py

Requires the Fuseki SPARQL server to be running on port 3001.
"""
from __future__ import annotations

import os
import sys

# Run from repo root so src/ imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI
from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor

SPARQL_URL = os.getenv("KG_SPARQL_URL", "http://127.0.0.1:3001/kb/sparql")
ONTOLOGY_DIR = os.getenv("KG_ONTOLOGY_DIR", "data/v0121/knowledge_graph/ontology")
RELATION = "food.cheese_milk_source.cheeses"


def main() -> None:
    print(f"SPARQL endpoint : {SPARQL_URL}")
    print(f"Ontology dir    : {ONTOLOGY_DIR}")
    print()

    executor = SparqlExecutor(SPARQL_URL)
    api = KnowledgeGraphAPI(ONTOLOGY_DIR, executor)

    # ── Precondition: get_relations must be called before get_neighbors ───────
    print("=== Precondition: get_relations('Goat') ===")
    _, pre_goat = api.get_relations("Goat")
    print(pre_goat[:400])
    print()

    print("=== Precondition: get_relations('cows') ===")
    _, pre_cow = api.get_relations("cows")
    print(pre_cow[:400])
    print()

    # ── Step 1: get_neighbors("Goat", RELATION) ──────────────────────────────
    print(f"=== Step 1: get_neighbors('Goat', '{RELATION}') ===")
    try:
        var_goat, msg_goat = api.get_neighbors("Goat", RELATION)
        print("RAW output:")
        print(msg_goat)
        print(f"Variable object: {repr(var_goat)}")
        print(f"Variable type  : {var_goat.type if var_goat else 'None'}")
        print(f"Variable program: {var_goat.program if var_goat else 'None'}")
    except Exception as exc:
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        var_goat = None
        msg_goat = str(exc)
    print()

    # ── Step 2: get_neighbors("cows", RELATION) ──────────────────────────────
    print(f"=== Step 2: get_neighbors('cows', '{RELATION}') ===")
    try:
        var_cow, msg_cow = api.get_neighbors("cows", RELATION)
        print("RAW output:")
        print(msg_cow)
        print(f"Variable object: {repr(var_cow)}")
        print(f"Variable type  : {var_cow.type if var_cow else 'None'}")
        print(f"Variable program: {var_cow.program if var_cow else 'None'}")
    except Exception as exc:
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        var_cow = None
        msg_cow = str(exc)
    print()

    # ── Step 3: intersection(var_goat, var_cow) ───────────────────────────────
    print("=== Step 3: intersection(var_goat, var_cow) ===")
    if var_goat is None or var_cow is None:
        print("SKIPPED — one or both variables are None (see errors above).")
    else:
        try:
            var_inter, msg_inter = KnowledgeGraphAPI.intersection(var_goat, var_cow)
            print("RAW output:")
            print(msg_inter)
            print(f"Variable object : {repr(var_inter)}")
            print(f"Variable type   : {var_inter.type if var_inter else 'None'}")
            print(f"Variable program: {var_inter.program if var_inter else 'None'}")
        except Exception as exc:
            print(f"EXCEPTION: {type(exc).__name__}: {exc}")
    print()

    # ── Probe alternate entity names for "cows" ───────────────────────────────
    for candidate in ("cow", "Cattle", "cattle", "Cow", "Cows"):
        print(f"=== Probe: get_relations('{candidate}') ===")
        try:
            _, rel_msg = api.get_relations(candidate)
            # show first 300 chars to spot food.* relations
            print(rel_msg[:400])
            if "food.cheese_milk_source" in rel_msg:
                print(f"  *** HIT: '{candidate}' HAS food.cheese_milk_source.cheeses ***")
                print(f"=== Probe: get_neighbors('{candidate}', '{RELATION}') ===")
                try:
                    var_cand, msg_cand = api.get_neighbors(candidate, RELATION)
                    print("RAW output:")
                    print(msg_cand)
                    print(f"Variable: {repr(var_cand)}")
                    if var_goat is not None:
                        print(f"=== Probe: intersection(var_goat, var_{candidate}) ===")
                        var_i, msg_i = KnowledgeGraphAPI.intersection(var_goat, var_cand)
                        print(msg_i)
                except Exception as exc2:
                    print(f"  get_neighbors EXCEPTION: {exc2}")
        except Exception as exc:
            print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
