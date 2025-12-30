from __future__ import annotations

from src.tasks.instance.knowledge_graph.api import resolve_ontology_dir


def main() -> None:
    resolved = resolve_ontology_dir(
        "data/v0121/knowledge_graph/ontology"
    )
    print(f"Resolved KG ontology dir: {resolved}")


if __name__ == "__main__":
    main()
