from __future__ import annotations

import ast
import json
import os
import re

from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI
from src.tasks.instance.knowledge_graph.utils.sparql_executor import SparqlExecutor


def _extract_relations(message: str) -> list[str]:
    # Capture first [...] that appears after "Observation:"
    m = re.search(r"Observation:\s*(\[[^\]]*\])", message, flags=re.DOTALL)
    if not m:
        return []
    bracket = m.group(1)
    inner = bracket[1:-1].strip()
    if not inner:
        return []
    parts = []
    for item in inner.split(","):
        s = item.strip()
        if not s:
            continue
        # strip optional quotes
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()
        parts.append(s)
    return parts




def main() -> None:
    sparql_url = os.getenv("KG_SPARQL_URL", "http://127.0.0.1:3001/kb/sparql")
    ontology_dir = os.getenv("KG_ONTOLOGY_DIR", "data/knowledge_graph/ontology")
    executor = SparqlExecutor(sparql_url)
    api = KnowledgeGraphAPI(ontology_dir, executor)

    count_query = """
SELECT (COUNT(*) AS ?c) WHERE {
  <http://rdf.freebase.com/ns/m.03fwl> ?p ?o .
}
""".strip()
    count_results = executor.execute_raw(count_query)
    count_value = 0
    for result in count_results["results"]["bindings"]:
        count_value = int(result["c"]["value"])
    print("Endpoint:", executor.get_endpoint_url())
    print("Goat triple count:", count_value)

    pred_query = """
SELECT ?p (COUNT(?o) AS ?c) WHERE {
  <http://rdf.freebase.com/ns/m.03fwl> ?p ?o .
}
GROUP BY ?p
ORDER BY DESC(?c)
LIMIT 10
""".strip()
    pred_results = executor.execute_raw(pred_query)
    preds = [row["p"]["value"] for row in pred_results["results"]["bindings"]]
    print("Top predicates:", preds)

    _, msg = api.get_relations("__DEBUG_GOAT__")
    print("RAW msg (DEBUG_GOAT):\n", msg)
    debug_relations = _extract_relations(msg)
    print("DEBUG_GOAT relations count:", len(debug_relations))
    print("DEBUG_GOAT relations sample:", debug_relations[:10])

    _, msg = api.get_relations("Goat")
    print("RAW msg (Goat):\n", msg)

    relations = _extract_relations(msg)
    print("Goat relations count:", len(relations))
    print("Goat relations sample:", relations[:10])

    if not relations:
        raise SystemExit("No relations found for Goat; KG tools may be misconfigured.")


if __name__ == "__main__":
    main()
