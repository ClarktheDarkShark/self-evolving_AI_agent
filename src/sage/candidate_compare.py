from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ANSWER_TARGET_FRAGMENTS = (
    "repairable_bad_count_set",
    "count_answer_target_unenforced",
    "count_set_weak_or_broken",
    "count_query_counts_wrong_variable",
    "trusted_incorrect_completion",
)
ANCHOR_FAILURE_FRAGMENTS = (
    "repairable_anchor_path_empty",
    "repairable_weak_grounding",
    "repairable_anchor_not_found",
    "anchor_path_empty",
    "anchor_not_found",
)


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def normalize_failure_labels(raw_values: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for raw_value in raw_values:
        cleaned = str(raw_value or "").strip()
        if not cleaned:
            continue
        if cleaned not in labels:
            labels.append(cleaned)
        leaf = cleaned.split(":")[-1].strip()
        if leaf and leaf not in labels:
            labels.append(leaf)
    return labels


def extract_semantic_failure_labels(
    *,
    run_dir: Path,
    sample_index: str,
    sample_status: str,
    evaluation_outcome: str,
    finish_reason: str = "",
) -> list[str]:
    generated_tools_path = run_dir / "generated_tools.log"
    latest_decision: dict[str, Any] = {}
    latest_rejected: dict[str, Any] = {}
    target_sample = str(sample_index or "").strip()
    for payload in load_json_lines(generated_tools_path):
        if str(payload.get("sample_index") or "").strip() != target_sample:
            continue
        if payload.get("event") == "sage_attempt_decision_finalized":
            latest_decision = payload
        elif payload.get("event") == "sage_repair_loop_rejected":
            latest_rejected = payload

    labels: list[str] = []
    labels.extend(
        normalize_failure_labels(
            [
                latest_decision.get("tool_result_failure_reason"),
                *(latest_decision.get("materialization_denial_reasons") or []),
                str(
                    ((latest_decision.get("typed_outcome") or {}).get("primary_failure_kind") or "")
                ).strip(),
                str(((latest_decision.get("typed_outcome") or {}).get("stop_reason") or "")).strip(),
                latest_rejected.get("final_verdict"),
                latest_rejected.get("last_verdict"),
                *(latest_rejected.get("last_reasons") or []),
            ]
        )
    )
    bypass_prefix = "sage_tool_failure_bypassed:"
    cleaned_finish = str(finish_reason or "").strip()
    if bypass_prefix in cleaned_finish:
        labels.extend(
            normalize_failure_labels([cleaned_finish.split(bypass_prefix, 1)[1]])
        )
    if str(sample_status or "").strip() == "completed" and str(
        evaluation_outcome or ""
    ).strip() != "correct":
        labels.extend(normalize_failure_labels(["trusted_incorrect_completion"]))
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def family_lock_violations(
    *,
    run_dir: Path,
    sample_index: str,
    locked_family: str,
) -> list[str]:
    cleaned_locked_family = str(locked_family or "").strip()
    if not cleaned_locked_family:
        return []
    violations: list[str] = []
    target_sample = str(sample_index or "").strip()
    for payload in load_json_lines(run_dir / "generated_tools.log"):
        if str(payload.get("sample_index") or "").strip() != target_sample:
            continue
        event = str(payload.get("event") or "").strip()
        if event == "sage_query_plan_generated":
            query_shape = str(payload.get("query_shape") or "").strip()
            if query_shape and query_shape != cleaned_locked_family:
                violations.append(f"query_shape:{query_shape}")
        elif event in {"sage_reusable_tool_selected", "sage_attempt_decision", "sage_attempt_decision_finalized"}:
            selected_family = str(
                payload.get("selected_family")
                or payload.get("family_name")
                or ""
            ).strip()
            if selected_family and selected_family != cleaned_locked_family:
                violations.append(f"selected_family:{selected_family}")
    deduped: list[str] = []
    for item in violations:
        if item not in deduped:
            deduped.append(item)
    return deduped


def summarize_sample_semantic_metrics(
    summary: Mapping[str, Any],
    *,
    locked_family: str = "",
) -> dict[str, Any]:
    sample_status = str(summary.get("sample_status") or "").strip()
    evaluation_outcome = str(summary.get("evaluation_outcome") or "").strip()
    run_dir_text = str(summary.get("run_dir") or "").strip()
    sample_index = str(summary.get("sample_index") or "").strip()
    finish_reason = str(summary.get("finish_reason") or "").strip()

    dangerous_overreach_count = int(summary.get("dangerous_overreach_count") or 0)
    family_lock_violation_details: list[str] = []
    typed_outcome: dict[str, Any] = {}
    if run_dir_text:
        run_dir = Path(run_dir_text)
        generated_tools_path = run_dir / "generated_tools.log"
        if generated_tools_path.exists():
            dangerous_overreach_count = 0
            for payload in load_json_lines(generated_tools_path):
                if str(payload.get("sample_index") or "").strip() != sample_index:
                    continue
                if payload.get("event") == "sage_attempt_decision_finalized" and bool(
                    payload.get("dangerous_overreach")
                ):
                    dangerous_overreach_count += 1
                if payload.get("event") == "sage_attempt_decision_finalized":
                    typed_payload = payload.get("typed_outcome") or {}
                    if isinstance(typed_payload, Mapping):
                        typed_outcome = dict(typed_payload)
        family_lock_violation_details = family_lock_violations(
            run_dir=run_dir,
            sample_index=sample_index,
            locked_family=locked_family,
        )
    if not typed_outcome:
        summary_primary_failure_kind = str(
            summary.get("sage_primary_failure_kind") or ""
        ).strip()
        if summary_primary_failure_kind:
            typed_outcome = {
                "primary_failure_kind": summary_primary_failure_kind,
                "stop_reason": str(summary.get("sage_stop_reason") or "").strip(),
                "completion_state": str(
                    summary.get("sage_completion_state") or ""
                ).strip(),
                "repair_attempt_count": int(
                    summary.get("sage_repair_attempt_count") or 0
                ),
            }
    if not str(typed_outcome.get("primary_failure_kind") or "").strip():
        if (
            sample_status == "timeout"
            or str(finish_reason or "").strip() == "timed_out"
            or bool(summary.get("timed_out"))
        ):
            typed_outcome["primary_failure_kind"] = "timeout"
            if not str(typed_outcome.get("stop_reason") or "").strip():
                typed_outcome["stop_reason"] = "runner_timeout"

    semantic_failure_labels = (
        extract_semantic_failure_labels(
            run_dir=Path(run_dir_text),
            sample_index=sample_index,
            sample_status=sample_status,
            evaluation_outcome=evaluation_outcome,
            finish_reason=finish_reason,
        )
        if run_dir_text
        else []
    )
    lowered_labels = [label.lower() for label in semantic_failure_labels]
    answer_target_failure_count = sum(
        1
        for label in lowered_labels
        if any(fragment in label for fragment in ANSWER_TARGET_FRAGMENTS)
    )
    anchor_failure_count = sum(
        1
        for label in lowered_labels
        if any(fragment in label for fragment in ANCHOR_FAILURE_FRAGMENTS)
    )
    wrong_completed = int(sample_status == "completed" and evaluation_outcome != "correct")
    correct_completed = int(sample_status == "completed" and evaluation_outcome == "correct")
    fail_closed = int(sample_status != "completed")
    primary_failure_kind = str(typed_outcome.get("primary_failure_kind") or "").strip()
    timeout_count = int(
        primary_failure_kind == "timeout"
        or sample_status == "timeout"
        or str(finish_reason or "").strip() == "timed_out"
        or bool(summary.get("timed_out"))
    )
    return {
        "sample_index": sample_index,
        "sample_status": sample_status,
        "evaluation_outcome": evaluation_outcome,
        "correct_completed": correct_completed,
        "wrong_completed": wrong_completed,
        "fail_closed": fail_closed,
        "dangerous_overreach_count": dangerous_overreach_count,
        "answer_target_failure_count": answer_target_failure_count,
        "anchor_failure_count": anchor_failure_count,
        "semantic_failure_labels": semantic_failure_labels,
        "family_lock_violation_count": len(family_lock_violation_details),
        "family_lock_violation_details": family_lock_violation_details,
        "timeout_count": timeout_count,
        "grounding_miss_count": int(primary_failure_kind == "grounding_miss"),
        "validator_rejection_count": int(primary_failure_kind == "validator_rejection"),
        "execution_shape_failure_count": int(
            primary_failure_kind == "execution_shape_failure"
        ),
        "semantic_family_failure_count": int(
            primary_failure_kind == "semantic_family_failure"
        ),
        "solver_handoff_failure_count": int(
            primary_failure_kind == "solver_handoff_failure"
        ),
        "repair_attempt_count": int(typed_outcome.get("repair_attempt_count") or 0),
        "primary_failure_kind": primary_failure_kind,
        "stop_reason": str(typed_outcome.get("stop_reason") or "").strip(),
    }


def baseline_satisfies_locked_family(
    semantic_metrics: Mapping[str, Any],
) -> bool:
    if int(semantic_metrics.get("family_lock_violation_count") or 0) > 0:
        return False
    labels = [
        str(label or "").strip()
        for label in (semantic_metrics.get("semantic_failure_labels") or [])
        if str(label or "").strip()
    ]
    return not any(
        label.startswith("sage_query_plan_invalid:family_compare_locked_query_shape:")
        for label in labels
    )


def aggregate_stage_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        "sample_count": 0,
        "correct_count": 0,
        "wrong_completed_count": 0,
        "fail_closed_count": 0,
        "dangerous_overreach_count": 0,
        "answer_target_failure_count": 0,
        "anchor_failure_count": 0,
        "family_lock_violation_count": 0,
        "timeout_count": 0,
        "grounding_miss_count": 0,
        "validator_rejection_count": 0,
        "execution_shape_failure_count": 0,
        "semantic_family_failure_count": 0,
        "solver_handoff_failure_count": 0,
        "repair_attempt_total": 0,
    }
    sample_ids: list[str] = []
    for row in rows:
        totals["sample_count"] += 1
        totals["correct_count"] += int(row.get("correct_completed") or 0)
        totals["wrong_completed_count"] += int(row.get("wrong_completed") or 0)
        totals["fail_closed_count"] += int(row.get("fail_closed") or 0)
        totals["dangerous_overreach_count"] += int(
            row.get("dangerous_overreach_count") or 0
        )
        totals["answer_target_failure_count"] += int(
            row.get("answer_target_failure_count") or 0
        )
        totals["anchor_failure_count"] += int(row.get("anchor_failure_count") or 0)
        totals["family_lock_violation_count"] += int(
            row.get("family_lock_violation_count") or 0
        )
        totals["timeout_count"] += int(row.get("timeout_count") or 0)
        totals["grounding_miss_count"] += int(row.get("grounding_miss_count") or 0)
        totals["validator_rejection_count"] += int(
            row.get("validator_rejection_count") or 0
        )
        totals["execution_shape_failure_count"] += int(
            row.get("execution_shape_failure_count") or 0
        )
        totals["semantic_family_failure_count"] += int(
            row.get("semantic_family_failure_count") or 0
        )
        totals["solver_handoff_failure_count"] += int(
            row.get("solver_handoff_failure_count") or 0
        )
        totals["repair_attempt_total"] += int(row.get("repair_attempt_count") or 0)
        sample_ids.append(str(row.get("sample_index") or "").strip())
    totals["accuracy"] = (
        float(totals["correct_count"]) / float(totals["sample_count"])
        if totals["sample_count"]
        else 0.0
    )
    totals["sample_ids"] = sample_ids
    totals["mean_repair_attempt_count"] = (
        float(totals["repair_attempt_total"]) / float(totals["sample_count"])
        if totals["sample_count"]
        else 0.0
    )
    return totals


def compare_aggregate_metrics(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    deltas = {
        "accuracy": float(candidate.get("accuracy") or 0.0)
        - float(baseline.get("accuracy") or 0.0),
        "correct_count": int(candidate.get("correct_count") or 0)
        - int(baseline.get("correct_count") or 0),
        "wrong_completed_count": int(candidate.get("wrong_completed_count") or 0)
        - int(baseline.get("wrong_completed_count") or 0),
        "fail_closed_count": int(candidate.get("fail_closed_count") or 0)
        - int(baseline.get("fail_closed_count") or 0),
        "dangerous_overreach_count": int(candidate.get("dangerous_overreach_count") or 0)
        - int(baseline.get("dangerous_overreach_count") or 0),
        "answer_target_failure_count": int(
            candidate.get("answer_target_failure_count") or 0
        )
        - int(baseline.get("answer_target_failure_count") or 0),
        "anchor_failure_count": int(candidate.get("anchor_failure_count") or 0)
        - int(baseline.get("anchor_failure_count") or 0),
        "family_lock_violation_count": int(candidate.get("family_lock_violation_count") or 0)
        - int(baseline.get("family_lock_violation_count") or 0),
        "timeout_count": int(candidate.get("timeout_count") or 0)
        - int(baseline.get("timeout_count") or 0),
        "grounding_miss_count": int(candidate.get("grounding_miss_count") or 0)
        - int(baseline.get("grounding_miss_count") or 0),
        "validator_rejection_count": int(candidate.get("validator_rejection_count") or 0)
        - int(baseline.get("validator_rejection_count") or 0),
        "execution_shape_failure_count": int(candidate.get("execution_shape_failure_count") or 0)
        - int(baseline.get("execution_shape_failure_count") or 0),
        "semantic_family_failure_count": int(candidate.get("semantic_family_failure_count") or 0)
        - int(baseline.get("semantic_family_failure_count") or 0),
        "solver_handoff_failure_count": int(candidate.get("solver_handoff_failure_count") or 0)
        - int(baseline.get("solver_handoff_failure_count") or 0),
    }
    non_regression = (
        deltas["correct_count"] >= 0
        and deltas["wrong_completed_count"] <= 0
        and deltas["fail_closed_count"] <= 0
        and deltas["dangerous_overreach_count"] <= 0
        and deltas["timeout_count"] <= 0
        and deltas["solver_handoff_failure_count"] <= 0
        and int(candidate.get("family_lock_violation_count") or 0) == 0
    )
    semantic_improved = (
        deltas["correct_count"] > 0
        or deltas["wrong_completed_count"] < 0
        or deltas["timeout_count"] < 0
        or deltas["answer_target_failure_count"] < 0
        or deltas["anchor_failure_count"] < 0
        or deltas["solver_handoff_failure_count"] < 0
    )
    return {
        "deltas": deltas,
        "non_regression": non_regression,
        "semantic_improved": semantic_improved,
        "beats_baseline": bool(non_regression and semantic_improved),
    }


def stage_definitively_lost(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    comparison = compare_aggregate_metrics(baseline=baseline, candidate=candidate)
    deltas = comparison.get("deltas") or {}
    return bool(
        int(deltas.get("wrong_completed_count") or 0) > 0
        or int(deltas.get("fail_closed_count") or 0) > 0
        or int(deltas.get("dangerous_overreach_count") or 0) > 0
        or int(deltas.get("timeout_count") or 0) > 0
        or int(deltas.get("solver_handoff_failure_count") or 0) > 0
        or int(candidate.get("family_lock_violation_count") or 0) > 0
    )


def load_archived_run_summary(
    *,
    run_dir: Path,
    sample_index: str,
) -> dict[str, Any]:
    runs_path = run_dir / "runs.json"
    if not runs_path.exists():
        raise FileNotFoundError(f"missing_runs_json:{run_dir}")
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"invalid_runs_json:{run_dir}")
    target_sample = str(sample_index or "").strip()
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("sample_index") or "").strip() != target_sample:
            continue
        evaluation_record = row.get("evaluation_record") or {}
        evaluation_outcome = (
            str(evaluation_record.get("outcome") or "").strip()
            if isinstance(evaluation_record, Mapping)
            else ""
        )
        return {
            "sample_index": target_sample,
            "sample_status": str(row.get("sample_status") or "").strip(),
            "evaluation_outcome": evaluation_outcome,
            "finish_reason": str(row.get("finish_reason") or "").strip(),
            "run_dir": str(run_dir),
            "dangerous_overreach_count": 0,
            "archived_baseline_reused": True,
        }
    raise KeyError(f"archived_sample_missing:{run_dir}:{target_sample}")


def baseline_archive_matches(
    archive_payload: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
) -> bool:
    archived_context = archive_payload.get("context") or {}
    if not isinstance(archived_context, Mapping):
        return False
    return dict(archived_context) == dict(expected_context)


def load_archived_baseline_record(
    path: Path,
    *,
    expected_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    if not baseline_archive_matches(payload, expected_context=expected_context):
        return None
    run_summary = payload.get("run_summary") or {}
    semantic_metrics = payload.get("semantic_metrics") or {}
    if not isinstance(run_summary, Mapping) or not isinstance(semantic_metrics, Mapping):
        return None
    return {
        "context": dict(payload.get("context") or {}),
        "run_summary": dict(run_summary),
        "semantic_metrics": dict(semantic_metrics),
    }


def write_archived_baseline_record(
    path: Path,
    *,
    context: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    semantic_metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "context": dict(context),
                "run_summary": dict(run_summary),
                "semantic_metrics": dict(semantic_metrics),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "aggregate_stage_metrics",
    "baseline_archive_matches",
    "baseline_satisfies_locked_family",
    "compare_aggregate_metrics",
    "extract_semantic_failure_labels",
    "family_lock_violations",
    "load_archived_baseline_record",
    "load_archived_run_summary",
    "normalize_failure_labels",
    "stage_definitively_lost",
    "summarize_sample_semantic_metrics",
    "write_archived_baseline_record",
]
