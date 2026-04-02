from src.pal.kg_benchmark_adapter import classify_execution_artifact


def test_numeric_raw_execution_with_count_head_is_count_scalar() -> None:
    artifact = classify_execution_artifact(
        {
            "head": {"vars": ["count"]},
            "results": {
                "bindings": [
                    {
                        "count": {
                            "type": "literal",
                            "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                            "value": "7",
                        }
                    }
                ]
            },
        }
    )

    assert artifact.artifact_type == "count_scalar"
    assert artifact.value == "7"


def test_numeric_raw_execution_with_ordering_head_is_scalar_literal() -> None:
    artifact = classify_execution_artifact(
        {
            "head": {"vars": ["ordering_attribute"]},
            "results": {
                "bindings": [
                    {
                        "ordering_attribute": {
                            "type": "literal",
                            "datatype": "http://www.w3.org/2001/XMLSchema#float",
                            "value": "3042.0",
                        }
                    }
                ]
            },
        }
    )

    assert artifact.artifact_type == "scalar_literal"
    assert artifact.value == "3042.0"
