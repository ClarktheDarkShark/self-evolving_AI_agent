from __future__ import annotations

import json
from types import SimpleNamespace

from src.pal.invoker import invoke_pal_program
from src.pal.parser import parse_pal_program


class _FakeQueryResult:
    def __init__(
        self,
        *,
        endpoint_url: str,
        query_text: str,
        return_format: str,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._query_text = query_text
        self._return_format = return_format

    def convert(self) -> dict:
        return {
            "head": {"vars": ["entity"]},
            "results": {
                "bindings": [
                    {
                        "entity": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.tom_hanks",
                        }
                    }
                ]
            },
            "meta": {
                "endpoint_url": self._endpoint_url,
                "query": self._query_text.strip(),
                "return_format": self._return_format,
            },
        }


class _FakeSPARQLWrapperClient:
    def __init__(self, endpoint_url: str) -> None:
        self._endpoint_url = endpoint_url
        self._query_text = ""
        self._return_format = ""

    def setQuery(self, query_text: str) -> None:
        self._query_text = query_text

    def setReturnFormat(self, return_format: str) -> None:
        self._return_format = return_format

    def query(self) -> _FakeQueryResult:
        return _FakeQueryResult(
            endpoint_url=self._endpoint_url,
            query_text=self._query_text,
            return_format=self._return_format,
        )


def main() -> None:
    fake_sparqlwrapper_module = SimpleNamespace(
        SPARQLWrapper=_FakeSPARQLWrapperClient,
        JSON="JSON",
    )
    raw_program = """###QUERY_START
def run_query(endpoint_url: str) -> dict:
    sparql_client = SPARQLWrapper.SPARQLWrapper(endpoint_url)
    sparql_client.setQuery(\"\"\"
    SELECT ?entity WHERE {
      ?entity <http://rdf.freebase.com/ns/type.object.name> \"Tom Hanks\"@en .
    }
    LIMIT 1
    \"\"\")
    sparql_client.setReturnFormat(SPARQLWrapper.JSON)
    return sparql_client.query().convert()
###QUERY_END"""

    parsed_program = parse_pal_program(raw_program)
    result = invoke_pal_program(
        parsed_program,
        endpoint_url="http://127.0.0.1:3001/kb/sparql",
        sparqlwrapper_module=fake_sparqlwrapper_module,
    )

    assert result.success, result.error
    assert result.payload is not None
    assert result.payload["results"]["bindings"][0]["entity"]["value"].endswith(
        "m.tom_hanks"
    )
    assert result.payload["meta"]["endpoint_url"].endswith("/kb/sparql")
    print("PAL invoker smoke test passed.")
    print(json.dumps(result.payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
