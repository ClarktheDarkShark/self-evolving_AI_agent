import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.instance.knowledge_graph.api import (
    KnowledgeGraphAPI,
    KnowledgeGraphAPIException,
    Variable,
)


class _FakeSparqlExecutor:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def get_endpoint_url(self) -> str:
        return "http://example.test/sparql"

    def execute_query(self, query: str) -> list[str]:
        self.queries.append(query)
        if "COUNT(DISTINCT ?x)" in query:
            return ["1"]
        if "SELECT DISTINCT ?rel" in query:
            return ["music.recording.releases"]
        return []

    def find_entities_by_name(self, name: str, limit: int = 5) -> list[str]:
        return ["m.test"]

    def get_out_relations(self, entity: str) -> list[str]:
        return ["music.recording.releases"]

    def execute_raw(self, query: str) -> dict:
        return {"results": {"bindings": []}}


def _make_api() -> KnowledgeGraphAPI:
    return KnowledgeGraphAPI("missing-ontology-dir", _FakeSparqlExecutor())


def test_get_relations_rejects_multihop_variable_probe() -> None:
    api = _make_api()
    variable = Variable(
        type="biology.breed_origin",
        program=(
            "(JOIN biology.animal_breed.place_of_origin_inv "
            "(JOIN base.petbreeds.dog_temperament.dog_breeds_inv "
            "(JOIN biology.animal_breed.temperament_inv m.05h0h0)))"
        ),
    )

    try:
        api.get_relations(variable)
    except KnowledgeGraphAPIException as exc:
        message = str(exc)
        assert "Node Explosion Prevented" in message
        assert "multi-hop or set-operation-derived" in message
    else:  # pragma: no cover
        raise AssertionError("Expected complex get_relations probe to be rejected")


def test_get_relations_allows_single_hop_variable_probe() -> None:
    api = _make_api()
    variable = Variable(
        type="music.recording",
        program="(JOIN music.featured_artist.recordings_inv m.01bsdt)",
    )

    _, message = api.get_relations(variable)

    assert "music.recording.releases" in message
