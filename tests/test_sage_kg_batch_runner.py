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


def test_summarize_run_dir_marks_late_manual_fallback_as_non_sage_success(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task_outcomes.json").write_text(
        json.dumps(
            [
                {
                    "sample_index": "1",
                    "sample_status": "completed",
                    "finish_reason": None,
                    "task_output": None,
                    "outcome": "correct",
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
                        "sample_index": "1",
                        "selected_family": "count_over_joined_set",
                        "generation_source": "fresh_tool",
                        "materialization_allowed": True,
                        "tool_result_status": "succeeded",
                        "typed_outcome": {
                            "completion_state": "accepted",
                            "stop_reason": "accepted_completion",
                            "primary_failure_kind": "accepted_completion",
                            "repair_attempt_count": 2,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "sage_manual_solver_fallback_used",
                        "sample_index": "1",
                        "mode": "sage",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = sage_kg_batch_runner._summarize_run_dir(run_dir)

    assert summary["sage_manual_fallback_used"] is True
    assert summary["sage_manual_fallback_count"] == 1
    assert summary["sage_completion_state"] == "manual_fallback"
    assert summary["sage_stop_reason"] == "manual_solver_fallback"
    assert summary["sage_primary_failure_kind"] == "manual_fallback"
    assert summary["sage_pre_fallback_completion_state"] == "accepted"
    assert summary["sage_pre_fallback_stop_reason"] == "accepted_completion"
    assert summary["sage_pre_fallback_primary_failure_kind"] == "accepted_completion"


def test_build_summary_report_aggregates_typed_failure_counts() -> None:
    report = sage_kg_batch_runner._build_summary_report(
        label="mixed8",
        samples=["33", "214", "15", "1"],
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
            {
                "sample_index": "1",
                "sample_status": "completed",
                "evaluation_outcome": "correct",
                "sage_selected_family": "count_over_joined_set",
                "sage_completion_state": "manual_fallback",
                "sage_primary_failure_kind": "manual_fallback",
                "sage_repair_attempt_count": 2,
                "sage_manual_fallback_used": True,
            },
        ],
    )

    aggregate = report["aggregate"]
    assert report["mode_metadata"]["label"] == "mixed8"
    assert report["mode_metadata"]["samples"] == ["33", "214", "15", "1"]
    assert aggregate["sample_count"] == 4
    assert aggregate["correct_count"] == 2
    assert aggregate["wrong_completed_count"] == 1
    assert aggregate["fail_closed_count"] == 1
    assert aggregate["timeout_count"] == 1
    assert aggregate["solver_handoff_failure_count"] == 1
    assert aggregate["manual_fallback_count"] == 1
    assert aggregate["typed_outcome_count"] == 4
    assert aggregate["runner_inferred_count"] == 1
    assert aggregate["selected_family_counts"] == {
        "count_over_joined_set": 1,
        "count_over_direct_relation": 1,
        "single_anchor_chain_lookup": 1,
    }
    assert aggregate["completion_state_counts"] == {
        "accepted": 1,
        "fail_closed": 1,
        "manual_fallback": 1,
        "rejected": 1,
    }
    assert aggregate["primary_failure_kind_counts"] == {
        "accepted_completion": 1,
        "manual_fallback": 1,
        "solver_handoff_failure": 1,
        "timeout": 1,
    }


def test_finalize_summary_row_restores_direct_bridge_final_even_with_fallback_marker() -> None:
    summary = sage_kg_batch_runner._finalize_summary_row(
        {
            "sample_index": "2",
            "sample_status": "completed",
            "evaluation_outcome": "correct",
            "sage_terminal_resolution": "direct_bridge_final",
            "sage_completion_state": "manual_fallback",
            "sage_stop_reason": "manual_solver_fallback",
            "sage_primary_failure_kind": "manual_fallback",
            "sage_manual_fallback_used": True,
            "sage_pre_fallback_completion_state": "accepted",
            "sage_pre_fallback_stop_reason": "accepted_completion",
            "sage_pre_fallback_primary_failure_kind": "accepted_completion",
        }
    )

    assert summary["sage_completion_state"] == "accepted"
    assert summary["sage_stop_reason"] == "accepted_completion"
    assert summary["sage_primary_failure_kind"] == "accepted_completion"


def test_extract_sage_terminal_resolution_distinguishes_direct_final_vs_continued(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "runs.json").write_text(
        json.dumps(
            [
                {
                    "sample_index": "1",
                    "chat_history": {
                        "value": [
                            {
                                "role": "user",
                                "content": "Macro result: sage_benchmark_bridge_macro -> SUCCESS.\nFinal variable: #0",
                            },
                            {"role": "agent", "content": "Action: get_relations(#0)"},
                            {"role": "user", "content": "get_relations(#0) executes successfully. Observation: []"},
                            {"role": "agent", "content": "Final Answer: #0"},
                        ]
                    },
                },
                {
                    "sample_index": "2",
                    "chat_history": {
                        "value": [
                            {
                                "role": "user",
                                "content": "Macro result: sage_benchmark_bridge_macro -> SUCCESS.\nFinal variable: #0",
                            },
                            {"role": "agent", "content": "Final Answer: #0"},
                        ]
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    continued = sage_kg_batch_runner._extract_sage_terminal_resolution(
        run_dir=run_dir,
        sample_index="1",
    )
    direct = sage_kg_batch_runner._extract_sage_terminal_resolution(
        run_dir=run_dir,
        sample_index="2",
    )

    assert continued["sage_terminal_resolution"] == "post_bridge_continued"
    assert direct["sage_terminal_resolution"] == "direct_bridge_final"
