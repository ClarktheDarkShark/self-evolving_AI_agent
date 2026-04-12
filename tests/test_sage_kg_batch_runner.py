import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sage_kg_batch_runner


def test_summarize_run_dir_threads_typed_sage_outcome(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task_outcomes.json").write_text(
        json.dumps(
            [
                {
                    "sample_index": "214",
                    "sample_status": "completed",
                    "finish_reason": None,
                    "task_output": None,
                    "outcome": "incorrect",
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "generated_tools.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "sage_attempt_decision_finalized",
                        "sample_index": "214",
                        "selected_family": "single_anchor_lookup",
                        "generation_source": "reusable_tool",
                        "materialization_allowed": False,
                        "materialization_denial_reasons": [
                            "execution_shape_mismatch",
                            "expected_literal_artifact:entity_id",
                        ],
                        "dangerous_overreach": False,
                        "tool_result_status": "failed",
                        "tool_result_failure_reason": (
                            "execution_shape_mismatch,"
                            "expected_literal_artifact:entity_id"
                        ),
                        "typed_outcome": {
                            "completion_state": "rejected",
                            "stop_reason": "solver_handoff_failure",
                            "primary_failure_kind": "solver_handoff_failure",
                            "plausibility_verdict": "accepted",
                            "verdict_reasons": [],
                            "repair_attempt_count": 1,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "sage_repair_loop_rejected",
                        "sample_index": "214",
                        "last_verdict": "repairable_bad_join",
                        "last_reasons": ["join_overlap_empty"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = sage_kg_batch_runner._summarize_run_dir(run_dir)

    assert summary["sample_index"] == "214"
    assert summary["sage_selected_family"] == "single_anchor_lookup"
    assert summary["sage_generation_source"] == "reusable_tool"
    assert summary["sage_materialization_allowed"] is False
    assert summary["sage_completion_state"] == "rejected"
    assert summary["sage_stop_reason"] == "solver_handoff_failure"
    assert summary["sage_primary_failure_kind"] == "solver_handoff_failure"
    assert summary["sage_tool_result_status"] == "failed"
    assert summary["sage_last_rejected_verdict"] == "repairable_bad_join"
    assert summary["sage_last_rejected_reasons"] == ["join_overlap_empty"]


def test_finalize_summary_row_infers_timeout_failure_kind() -> None:
    summary = sage_kg_batch_runner._finalize_summary_row(
        {
            "sample_index": "15",
            "sample_status": "timeout",
            "finish_reason": "timed_out",
            "timed_out": True,
        }
    )

    assert summary["sage_completion_state"] == "fail_closed"
    assert summary["sage_stop_reason"] == "runner_timeout"
    assert summary["sage_primary_failure_kind"] == "timeout"
    assert summary["sage_runner_inferred"] is True


def test_build_summary_report_aggregates_typed_failure_counts() -> None:
    report = sage_kg_batch_runner._build_summary_report(
        label="mixed8",
        samples=["33", "214", "15"],
        summaries=[
            {
                "sample_index": "33",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "sage_selected_family": "count_over_direct_relation",
                "sage_completion_state": "accepted",
                "sage_primary_failure_kind": "accepted_completion",
                "sage_repair_attempt_count": 1,
            },
            {
                "sample_index": "214",
                "sample_status": "completed",
                "evaluation_outcome": "incorrect",
                "sage_selected_family": "single_anchor_chain_lookup",
                "sage_completion_state": "rejected",
                "sage_primary_failure_kind": "solver_handoff_failure",
                "sage_repair_attempt_count": 2,
            },
            {
                "sample_index": "15",
                "sample_status": "timeout",
                "finish_reason": "timed_out",
                "timed_out": True,
                "sage_completion_state": "fail_closed",
                "sage_stop_reason": "runner_timeout",
                "sage_primary_failure_kind": "timeout",
                "sage_runner_inferred": True,
            },
        ],
    )

    aggregate = report["aggregate"]
    assert report["mode_metadata"]["label"] == "mixed8"
    assert report["mode_metadata"]["samples"] == ["33", "214", "15"]
    assert aggregate["sample_count"] == 3
    assert aggregate["correct_count"] == 1
    assert aggregate["wrong_completed_count"] == 1
    assert aggregate["fail_closed_count"] == 1
    assert aggregate["timeout_count"] == 1
    assert aggregate["solver_handoff_failure_count"] == 1
    assert aggregate["typed_outcome_count"] == 3
    assert aggregate["runner_inferred_count"] == 1
    assert aggregate["selected_family_counts"] == {
        "count_over_direct_relation": 1,
        "single_anchor_chain_lookup": 1,
    }
    assert aggregate["completion_state_counts"] == {
        "accepted": 1,
        "fail_closed": 1,
        "rejected": 1,
    }
