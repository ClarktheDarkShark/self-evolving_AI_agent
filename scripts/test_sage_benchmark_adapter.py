from __future__ import annotations

import json

from src.sage.kg_benchmark_adapter import (
    BenchmarkAdapterContext,
    adapt_sage_result_to_benchmark,
    build_sage_benchmark_bridge_tool_code,
)
from src.tasks.instance.knowledge_graph.api import KnowledgeGraphAPI


class _DummySparqlExecutor:
    def execute_query(self, _query: str):
        raise AssertionError("Synthetic SAGE artifacts should bypass SPARQL execution.")

    def get_endpoint_url(self) -> str:
        return "dummy://sage-adapter-test"


def _load_bridge_runner():
    namespace: dict[str, object] = {}
    exec(build_sage_benchmark_bridge_tool_code(), namespace, namespace)
    return namespace["run"]


def _run_case(
    *,
    name: str,
    raw_result,
    solver_output: str | None,
    question: str,
):
    context = BenchmarkAdapterContext(
        task_question=question,
        run_id=f"adapter_test_{name}",
        state_dir="/tmp/sage_adapter_test",
    )
    adaptation = adapt_sage_result_to_benchmark(
        raw_result=raw_result,
        solver_output=solver_output,
        context=context,
    )
    bridge_runner = _load_bridge_runner()
    variable_list = []
    bridge_payload = dict(adaptation.materialization.bridge_payload or {})
    bridge_payload["variable_list"] = variable_list
    bridge_result = bridge_runner(bridge_payload)
    final_pointer = bridge_result.get("final_variable")

    api = KnowledgeGraphAPI(
        ontology_dir_path="",
        sparql_executor=_DummySparqlExecutor(),
    )
    final_output = None
    if isinstance(final_pointer, str) and final_pointer.startswith("#"):
        final_output = api.final_execute(variable_list[int(final_pointer[1:])])

    return {
        "case": name,
        "raw_sage_result": raw_result,
        "solver_output": solver_output,
        "classified_artifact_type": adaptation.artifact.artifact_type,
        "artifact_source": adaptation.artifact.source,
        "artifact_value": adaptation.artifact.value,
        "materialization_type": adaptation.materialization.materialization_type,
        "determinism_level": adaptation.materialization.determinism_level,
        "bridge_action": adaptation.materialization.bridge_action,
        "bridge_result": bridge_result,
        "final_output_shape": final_output,
    }


def main() -> None:
    cases = [
        _run_case(
            name="single_entity",
            raw_result={
                "head": {"vars": ["entity"]},
                "results": {
                    "bindings": [
                        {
                            "entity": {
                                "type": "uri",
                                "value": "http://rdf.freebase.com/ns/m.02mjmr",
                            }
                        }
                    ]
                },
            },
            solver_output="Final Answer: Barack Obama",
            question="Who is Barack Obama?",
        ),
        _run_case(
            name="entity_set",
            raw_result={
                "head": {"vars": ["entity"]},
                "results": {
                    "bindings": [
                        {
                            "entity": {
                                "type": "uri",
                                "value": "http://rdf.freebase.com/ns/m.abc123",
                            }
                        },
                        {
                            "entity": {
                                "type": "uri",
                                "value": "http://rdf.freebase.com/ns/m.def456",
                            }
                        },
                    ]
                },
            },
            solver_output="Final Answer: ignored because raw set is authoritative",
            question="Return a set of entities.",
        ),
        _run_case(
            name="count_scalar",
            raw_result={
                "head": {"vars": ["count"]},
                "results": {
                    "bindings": [
                        {
                            "count": {
                                "type": "literal",
                                "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                                "value": "2",
                            }
                        }
                    ]
                },
            },
            solver_output="Final Answer: two",
            question="How many are there?",
        ),
        _run_case(
            name="boolean",
            raw_result={"boolean": False},
            solver_output="Final Answer: false",
            question="Is this true?",
        ),
        _run_case(
            name="empty",
            raw_result={
                "head": {"vars": ["entity"]},
                "results": {"bindings": []},
            },
            solver_output="Final Answer: no match found",
            question="Which entities match no rows?",
        ),
        _run_case(
            name="fallback_text",
            raw_result={"unexpected": "shape"},
            solver_output="Final Answer: unresolved textual fallback",
            question="Fallback case",
        ),
    ]
    print(json.dumps(cases, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
