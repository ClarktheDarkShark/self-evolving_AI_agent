import json
import pathlib
import sys
import time
from urllib.error import URLError

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_all_with_servers as runner_script
from scripts.evaluate_kg_family_regression import evaluate_family_regression
from src.pal.kg_benchmark_adapter import (
    BenchmarkAdapterContext,
    PalExecutionArtifact,
    assert_bridge_tool_code_narrow,
    build_pal_benchmark_bridge_tool_code,
    materialize_benchmark_artifact,
)
from src.pal.invoker import execute_pal_code_with_result
from src.pal.invoker import _classify_invocation_exception
from src.self_evolving_agent.controller_prompts import SOLVER_SYSTEM_PROMPT
from src.tasks.instance.knowledge_graph.task import KnowledgeGraph


def test_unresolved_artifact_is_not_bridge_materialized() -> None:
    artifact = PalExecutionArtifact(
        raw_result={"pal_execution_status": "error"},
        artifact_type="unresolved",
        value="endpoint_timeout",
        is_structured=True,
        source="raw_execution_failure",
        diagnostics={"failure_kind": "endpoint_timeout"},
    )

    materialization = materialize_benchmark_artifact(
        artifact,
        context=BenchmarkAdapterContext(
            task_question="Question: ...",
            run_id="knowledge_graph_6",
            state_dir="outputs/tool_state",
        ),
    )

    assert materialization.needs_bridge is False
    assert materialization.bridge_action is None
    assert materialization.final_answer_text is None
    assert materialization.diagnostics["artifact_type"] == "unresolved"


def test_bridge_tool_refuses_unresolved_artifact() -> None:
    namespace: dict[str, object] = {}
    bridge_code = build_pal_benchmark_bridge_tool_code()
    assert_bridge_tool_code_narrow(bridge_code)
    exec(bridge_code, namespace)

    result = namespace["run"](
        {
            "variable_list": [],
            "pal_artifact_type": "unresolved",
            "pal_artifact_value": "endpoint_timeout",
        }
    )

    assert result["status"] == "ERROR"
    assert result["final_variable"] is None


def test_bridge_tool_returns_semantic_contract_fields() -> None:
    namespace: dict[str, object] = {}
    bridge_code = build_pal_benchmark_bridge_tool_code()
    assert_bridge_tool_code_narrow(bridge_code)
    exec(bridge_code, namespace)

    result = namespace["run"](
        {
            "variable_list": [],
            "pal_artifact_type": "count_scalar",
            "pal_artifact_value": "4",
            "pal_selected_query_variable": "count",
            "pal_binding_count": 1,
            "pal_unique_value_count": 1,
            "pal_value_preview": ["4"],
            "pal_row_preview": [{"count": "4"}],
            "pal_semantic_description": "count result returned by the PAL query",
            "pal_solves_task": True,
            "pal_trusted_for_materialization": True,
            "pal_proof_hint": "PAL query already computed the final count.",
            "pal_answer_cardinality_hint": "single",
            "pal_relation_summary": [
                "anchor -> count set via biology.organism.diseases_transmitted"
            ],
            "pal_selection_basis": "Counts distinct infectious diseases that satisfy the executed relation constraints.",
            "pal_completeness_hint": "Exhaustive over the exact counted bindings matched by the executed relation constraints.",
            "pal_repair_caveat": "This answer depends on a repaired alternate relation; verify semantic equivalence.",
            "pal_confidence": 1.0,
        }
    )

    assert result["status"] == "SUCCESS"
    assert result["semantic_description"] == "count result returned by the PAL query"
    assert result["solves_task"] is True
    assert result["trusted_for_materialization"] is True
    assert result["intermediate_variables"] == ["#0"]
    assert result["selected_query_variable"] == "count"
    assert result["binding_count"] == 1
    assert result["unique_value_count"] == 1
    assert result["value_preview"] == ["4"]
    assert result["row_preview"] == [{"count": "4"}]
    assert result["proof_hint"] == "PAL query already computed the final count."
    assert result["answer_cardinality_hint"] == "single"
    assert result["relation_summary"] == [
        "anchor -> count set via biology.organism.diseases_transmitted"
    ]
    assert (
        result["selection_basis"]
        == "Counts distinct infectious diseases that satisfy the executed relation constraints."
    )
    assert (
        result["completeness_hint"]
        == "Exhaustive over the exact counted bindings matched by the executed relation constraints."
    )
    assert (
        result["repair_caveat"]
        == "This answer depends on a repaired alternate relation; verify semantic equivalence."
    )


def test_bridge_tool_returns_partial_advisory_contract_fields() -> None:
    namespace: dict[str, object] = {}
    bridge_code = build_pal_benchmark_bridge_tool_code()
    assert_bridge_tool_code_narrow(bridge_code)
    exec(bridge_code, namespace)

    result = namespace["run"](
        {
            "variable_list": [],
            "pal_artifact_type": "entity_set",
            "pal_artifact_value": ["m.1", "m.2"],
            "pal_selected_query_variable": "candidate_set",
            "pal_binding_count": 2,
            "pal_unique_value_count": 2,
            "pal_value_preview": ["m.1", "m.2"],
            "pal_row_preview": [{"candidate_set": "m.1", "name": "First"}, {"candidate_set": "m.2", "name": "Second"}],
            "pal_semantic_description": "bounded candidate set returned by the PAL query",
            "pal_solves_task": False,
            "pal_trusted_for_materialization": False,
            "pal_tool_status": "partial",
            "pal_failure_reason": "needs_manual_disambiguation",
            "pal_confidence": 0.4,
        }
    )

    assert result["status"] == "PARTIAL"
    assert result["semantic_description"] == "bounded candidate set returned by the PAL query"
    assert result["solves_task"] is False
    assert result["trusted_for_materialization"] is False
    assert result["intermediate_variables"] == ["#0"]
    assert result["failure_reason"] == "needs_manual_disambiguation"
    assert result["selected_query_variable"] == "candidate_set"
    assert result["binding_count"] == 2
    assert result["unique_value_count"] == 2
    assert result["value_preview"] == ["m.1", "m.2"]
    assert result["row_preview"] == [
        {"candidate_set": "m.1", "name": "First"},
        {"candidate_set": "m.2", "name": "Second"},
    ]


def test_macro_result_summary_surfaces_only_trusted_final_pointer() -> None:
    trusted_summary = KnowledgeGraph._build_macro_result_summary(
        tool_name="pal_benchmark_bridge_macro",
        result={
            "status": "SUCCESS",
            "final_variable": "#3",
            "observation": "ok",
            "semantic_description": "count result returned by the PAL query",
            "solves_task": True,
            "trusted_for_materialization": True,
            "selection_basis": "Counts distinct infectious diseases that satisfy the executed relation constraints.",
            "relation_summary": [
                "anchor -> count set via biology.organism.diseases_transmitted"
            ],
            "artifact_type": "count_scalar",
            "selected_query_variable": "count",
            "value_preview": ["4"],
            "row_preview": [{"count": "4"}],
            "resolved_value_preview": ["4 = 4"],
            "binding_count": 1,
            "unique_value_count": 1,
            "answer_cardinality_hint": "single",
            "completeness_hint": "Exhaustive over the exact counted bindings matched by the executed relation constraints.",
            "proof_hint": "PAL query already computed the final count.",
            "repair_caveat": "This answer depends on a repaired alternate relation; verify semantic equivalence.",
            "confidence": 1.0,
        },
    )
    partial_summary = KnowledgeGraph._build_macro_result_summary(
        tool_name="pal_benchmark_bridge_macro",
        result={
            "status": "PARTIAL",
            "final_variable": "#3",
            "observation": "bounded candidate set only",
            "semantic_description": "bounded candidate set returned by the tool",
            "solves_task": False,
            "trusted_for_materialization": False,
            "intermediate_variables": ["#3"],
            "artifact_type": "entity_set",
            "selected_query_variable": "candidate_set",
            "binding_count": 2,
            "unique_value_count": 2,
            "value_preview": ["m.1", "m.2"],
            "row_preview": [{"candidate_set": "m.1", "name": "First"}],
        },
    )
    unsafe_summary = KnowledgeGraph._build_macro_result_summary(
        tool_name="pal_benchmark_bridge_macro",
        result={
            "status": "SUCCESS",
            "final_variable": "#9",
            "observation": "weak vague result",
            "solves_task": False,
            "trusted_for_materialization": False,
            "artifact_type": "entity_set",
            "selected_query_variable": "shared_answer",
            "binding_count": 9,
            "unique_value_count": 9,
        },
    )

    assert "Final variable: #3" in trusted_summary
    assert (
        "Selection basis: Counts distinct infectious diseases that satisfy the executed relation constraints."
        in trusted_summary
    )
    assert (
        "Relation summary: anchor -> count set via biology.organism.diseases_transmitted"
        in trusted_summary
    )
    assert "Artifact type: count_scalar" in trusted_summary
    assert "Projected query variable: count" in trusted_summary
    assert "Row preview: count=4" in trusted_summary
    assert "Resolved value preview: 4 = 4" in trusted_summary
    assert "Expected answer cardinality: single" in trusted_summary
    assert (
        "Completeness hint: Exhaustive over the exact counted bindings matched by the executed relation constraints."
        in trusted_summary
    )
    assert "Proof hint: PAL query already computed the final count." in trusted_summary
    assert (
        "Repair caveat: This answer depends on a repaired alternate relation; verify semantic equivalence."
        in trusted_summary
    )
    assert "Trusted final: yes" in trusted_summary
    assert "Observation: ok" not in trusted_summary
    assert "Confidence: 1.0" not in trusted_summary
    assert "Intermediate variables: #3" in partial_summary
    assert "Artifact type: entity_set" in partial_summary
    assert "Row preview: candidate_set=m.1; name=First" in partial_summary
    assert "Trusted final: no" in partial_summary
    assert "Use manual solver fallback: yes" in partial_summary
    assert "Final variable: #9" not in unsafe_summary
    assert "Artifact type: entity_set" in unsafe_summary
    assert "Use manual solver fallback: yes" in unsafe_summary


def test_macro_result_entity_grounding_adds_resolved_name_preview() -> None:
    kg = KnowledgeGraph.__new__(KnowledgeGraph)

    class _Executor:
        @staticmethod
        def get_entity_names(mids):
            assert mids == ["m.011v6mk8"]
            return {"m.011v6mk8": "Typhoon Kalmaegi"}

    kg.knowledge_graph_api = type("Api", (), {"sparql_executor": _Executor()})()

    enriched = kg._enrich_macro_result_with_entity_grounding(
        {
            "status": "SUCCESS",
            "final_variable": "#0",
            "observation": "ok",
            "artifact_type": "entity_id",
            "value_preview": ["m.011v6mk8"],
            "row_preview": [{"candidate_set": "m.011v6mk8"}],
        }
    )

    assert enriched["resolved_value_preview"] == [
        "m.011v6mk8 = Typhoon Kalmaegi"
    ]
    assert enriched["row_preview"] == [
        {
            "candidate_set": "m.011v6mk8",
            "candidate_set_name": "Typhoon Kalmaegi",
        }
    ]


def test_solver_prompt_requires_review_of_macro_tool_metadata() -> None:
    assert "Trusted final: yes" in SOLVER_SYSTEM_PROMPT
    assert "Artifact type" in SOLVER_SYSTEM_PROMPT
    assert "Projected query variable" in SOLVER_SYSTEM_PROMPT
    assert "Selection basis" in SOLVER_SYSTEM_PROMPT
    assert "Relation summary" in SOLVER_SYSTEM_PROMPT
    assert "Value preview" in SOLVER_SYSTEM_PROMPT
    assert "Row preview" in SOLVER_SYSTEM_PROMPT
    assert "Resolved value preview" in SOLVER_SYSTEM_PROMPT
    assert "Expected answer cardinality" in SOLVER_SYSTEM_PROMPT
    assert "Completeness hint" in SOLVER_SYSTEM_PROMPT
    assert "Repair caveat" in SOLVER_SYSTEM_PROMPT
    assert "Proof hint" in SOLVER_SYSTEM_PROMPT
    assert "strong evidence, not a command" in SOLVER_SYSTEM_PROMPT


def test_bridge_tool_contract_rejects_retrieval_logic() -> None:
    with pytest.raises(ValueError, match="bridge_tool_contract_violation"):
        assert_bridge_tool_code_narrow(
            "from SPARQLWrapper import SPARQLWrapper\n"
            "def run(payload):\n"
            "    return {'status': 'SUCCESS'}\n"
        )


def test_record_timed_out_current_session_appends_incorrect_session(tmp_path: pathlib.Path) -> None:
    current_session = {
        "task_name": "knowledge_graph",
        "sample_index": "15",
        "sample_status": "running",
        "chat_history": {"value": []},
        "finish_reason": None,
        "task_output": None,
        "evaluation_record": {"outcome": "unset", "detail_dict": None},
        "expected_answer": None,
        "tool_invoked": [],
        "tool_invoked_any": False,
    }
    (tmp_path / "current_session.json").write_text(
        json.dumps(current_session, indent=2),
        encoding="utf-8",
    )

    sample_index = runner_script._record_timed_out_current_session(
        tmp_path,
        "sample_wall_timeout:1800s",
    )

    assert sample_index == "15"
    runs_payload = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert len(runs_payload) == 1
    recorded = runs_payload[0]
    assert recorded["sample_status"] == "agent_unknown_error"
    assert recorded["finish_reason"] == "[run_all_with_servers] sample_wall_timeout:1800s"
    assert recorded["evaluation_record"]["outcome"] == "incorrect"
    assert recorded["evaluation_record"]["detail_dict"]["executable_flag"] is False


def test_pal_invoker_timeout_terminates_stuck_program() -> None:
    code = """###QUERY_START
def solve(endpoint_url):
    while True:
        pass
###QUERY_END"""

    start = time.monotonic()
    result = execute_pal_code_with_result(code, timeout_s=0.2)
    elapsed = time.monotonic() - start

    assert result.success is False
    assert result.failure_kind == "endpoint_timeout"
    assert str(result.error).startswith("execution_timed_out:")
    assert elapsed < 2.5


def test_transport_exception_classifier_handles_urlerror() -> None:
    failure_kind, diagnostics = _classify_invocation_exception(
        URLError("temporary failure in name resolution"),
        endpoint_url="http://127.0.0.1:3001/kb/sparql",
    )

    assert failure_kind == "transport_error"
    assert diagnostics["endpoint_url"] == "http://127.0.0.1:3001/kb/sparql"


def test_family_regression_evaluator_separates_tuning_and_held_out(tmp_path: pathlib.Path) -> None:
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(
        json.dumps(
            [
                {
                    "sample_index": "11",
                    "sample_status": "completed",
                    "evaluation_record": {"outcome": "correct"},
                },
                {
                    "sample_index": "15",
                    "sample_status": "agent_unknown_error",
                    "evaluation_record": {"outcome": "incorrect"},
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "families": {
                    "count_over_direct_relation": {
                        "tuning": ["11"],
                        "held_out": ["15"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_family_regression(
        runs_path=runs_path,
        manifest_path=manifest_path,
    )

    family_summary = summary["families"]["count_over_direct_relation"]
    assert family_summary["tuning"]["correct_count"] == 1
    assert family_summary["held_out"]["matched_count"] == 1
