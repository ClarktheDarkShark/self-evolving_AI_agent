import json
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_all_with_servers as runner_script
from src.pal.kg_benchmark_adapter import (
    BenchmarkAdapterContext,
    PalExecutionArtifact,
    build_pal_benchmark_bridge_tool_code,
    materialize_benchmark_artifact,
)
from src.pal.invoker import execute_pal_code_with_result


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
    exec(build_pal_benchmark_bridge_tool_code(), namespace)

    result = namespace["run"](
        {
            "variable_list": [],
            "pal_artifact_type": "unresolved",
            "pal_artifact_value": "endpoint_timeout",
        }
    )

    assert result["status"] == "ERROR"
    assert result["final_variable"] is None


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
