from __future__ import annotations

import argparse
import json
import os
import signal
import time
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_CONFIG_DIR = (
    PROJECT_ROOT
    / "configs"
    / "assignments"
    / "experiments"
    / "llama_31_8b_instruct"
    / "instance"
    / "knowledge_graph"
    / "instance"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ENV_RUN_SAMPLE_TIMEOUT_SECONDS = "SAGE_KG_BATCH_RUN_TIMEOUT_SECONDS"
DEFAULT_RUN_SAMPLE_TIMEOUT_SECONDS = 300
SAGE_ONLY_BYPASS_ENV = "SAGE_TOOL_EVOLUTION_SKIP_MANUAL_FALLBACK"
EVOLUTION_DISABLED_ENVS = (
    "SAGE_ENABLE_FAMILY_POLICY_EVOLUTION",
    "SAGE_ENABLE_FAMILY_POLICY_PROMOTION",
    "SAGE_ENABLE_STANDARD_FAMILY_EVOLUTION",
)
COMPARE_LOCK_FAMILY_ENV = "SAGE_FAMILY_POLICY_COMPARE_LOCK_FAMILY"

CONFIG_TEMPLATE = """\
import:
- ../task.yaml
- ../../../agent.yaml
- ../../../../../../definition.yaml

assignment_config:
  callback_dict:
    callback_0:
      name: current_session_saving_callback
    callback_1:
      name: consecutive_abnormal_agent_inference_process_handling_callback
  output_dir: outputs/{{TIMESTAMP}}
  sample_order:
    - "{sample_index}"

environment_config:
  use_task_client_flag: true

task_dict:
  knowledge_graph:
    parameters:
      ontology_dir_path: null
      data_source: huggingface
      hf_dataset_name: csyq/LifelongAgentBench
      hf_dataset_config: default
      hf_data_dir: knowledge_graph
      hf_split: train
"""


def _write_single_sample_config(*, sample_index: str, stem: str) -> Path:
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = TMP_CONFIG_DIR / f"{stem}.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(sample_index=sample_index),
        encoding="utf-8",
    )
    return config_path


def _find_latest_run_dir(config_stem: str) -> Path | None:
    candidates = sorted(
        OUTPUT_ROOT.glob(f"run_all_*/knowledge_graph/{config_stem}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
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


def _normalize_subprocess_text(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8", errors="replace")
    return str(raw_value)


def _run_sample_timeout_seconds() -> float | None:
    raw_value = str(os.environ.get(ENV_RUN_SAMPLE_TIMEOUT_SECONDS) or "").strip()
    if not raw_value:
        return float(DEFAULT_RUN_SAMPLE_TIMEOUT_SECONDS)
    try:
        parsed = float(raw_value)
    except ValueError:
        return float(DEFAULT_RUN_SAMPLE_TIMEOUT_SECONDS)
    if parsed <= 0:
        return None
    return parsed


def _cleanup_orphan_sample_processes(*, config_identifier: str) -> None:
    cleaned_identifier = str(config_identifier or "").strip()
    if cleaned_identifier:
        subprocess.run(
            ["pkill", "-f", cleaned_identifier],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    listening = subprocess.run(
        ["lsof", "-tiTCP:8000", "-tiTCP:8001", "-sTCP:LISTEN"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    for raw_pid in sorted(set(listening.stdout.split())):
        try:
            os.kill(int(raw_pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError, OSError):
            continue
    time.sleep(1)


def _terminate_sample_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            process.terminate()
        except Exception:
            pass
    time.sleep(1)
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            process.kill()
        except Exception:
            pass
    time.sleep(1)


def _summarize_run_dir(run_dir: Path) -> dict[str, Any]:
    task_outcomes_path = run_dir / "task_outcomes.json"
    current_session_path = run_dir / "current_session.json"
    exception_path = run_dir / "exception.txt"
    generated_tools_path = run_dir / "generated_tools.log"

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task_outcomes_path": str(task_outcomes_path) if task_outcomes_path.exists() else None,
        "generated_tools_path": str(generated_tools_path) if generated_tools_path.exists() else None,
        "exception_path": str(exception_path) if exception_path.exists() else None,
    }

    if task_outcomes_path.exists():
        outcomes = _load_json(task_outcomes_path)
        outcome = None
        if isinstance(outcomes, list) and outcomes:
            outcome = outcomes[0]
        elif isinstance(outcomes, dict):
            results = outcomes.get("results")
            if isinstance(results, list) and results:
                outcome = results[0]
        if isinstance(outcome, dict):
            summary.update(
                {
                    "sample_index": outcome.get("sample_index"),
                    "sample_status": outcome.get("sample_status")
                    or ("completed" if outcome.get("completed") else "not_completed"),
                    "finish_reason": outcome.get("finish_reason")
                    or outcome.get("not_completed_reason"),
                    "task_output": outcome.get("task_output"),
                    "evaluation_outcome": outcome.get("outcome")
                    or (
                        outcome.get("evaluation_record", {}) or {}
                    ).get("outcome"),
                }
            )
    elif current_session_path.exists():
        current_session = _load_json(current_session_path)
        summary.update(
            {
                "sample_index": current_session.get("sample_index"),
                "sample_status": current_session.get("sample_status"),
                "finish_reason": current_session.get("finish_reason"),
            }
        )
    sample_index = str(summary.get("sample_index") or "").strip()
    if generated_tools_path.exists():
        summary.update(
            _extract_sage_typed_summary(
                generated_tools_path=generated_tools_path,
                sample_index=sample_index,
            )
        )
    if sample_index:
        summary.update(
            _extract_sage_terminal_resolution(
                run_dir=run_dir,
                sample_index=sample_index,
            )
        )
    return summary


def _extract_sage_terminal_resolution(
    *,
    run_dir: Path,
    sample_index: str,
) -> dict[str, Any]:
    runs_path = run_dir / "runs.json"
    if not runs_path.exists():
        return {}
    try:
        runs_payload = _load_json(runs_path)
    except Exception:
        return {}
    if not isinstance(runs_payload, list):
        return {}
    target_sample = str(sample_index or "").strip()
    run_row = None
    for candidate in runs_payload:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("sample_index") or "").strip() == target_sample:
            run_row = candidate
            break
    if not isinstance(run_row, dict):
        return {}
    history = ((run_row.get("chat_history") or {}).get("value") or [])
    if not isinstance(history, list):
        return {}

    bridge_index = -1
    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        if role == "user" and content.startswith(
            "Macro result: sage_benchmark_bridge_macro -> SUCCESS."
        ):
            bridge_index = idx
    if bridge_index < 0:
        return {}

    next_agent_index = -1
    next_agent_content = ""
    later_agent_count = 0
    for idx in range(bridge_index + 1, len(history)):
        item = history[idx]
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip().lower() != "agent":
            continue
        later_agent_count += 1
        if next_agent_index < 0:
            next_agent_index = idx
            next_agent_content = str(item.get("content") or "").strip()
    if next_agent_index < 0:
        return {"sage_terminal_resolution": "bridge_without_agent_followup"}
    if next_agent_content.startswith("Final Answer:") and later_agent_count == 1:
        return {"sage_terminal_resolution": "direct_bridge_final"}
    return {"sage_terminal_resolution": "post_bridge_continued"}


def _extract_sage_typed_summary(
    *,
    generated_tools_path: Path,
    sample_index: str,
) -> dict[str, Any]:
    latest_finalized: dict[str, Any] = {}
    latest_rejected: dict[str, Any] = {}
    latest_manual_fallback: dict[str, Any] = {}
    latest_finalized_index = -1
    latest_manual_fallback_index = -1
    manual_fallback_count = 0
    target_sample = str(sample_index or "").strip()
    for payload_index, payload in enumerate(_load_json_lines(generated_tools_path)):
        payload_sample = str(payload.get("sample_index") or "").strip()
        if target_sample and payload_sample and payload_sample != target_sample:
            continue
        event = str(payload.get("event") or "").strip()
        if event == "sage_attempt_decision_finalized":
            latest_finalized = payload
            latest_finalized_index = payload_index
        elif event == "sage_repair_loop_rejected":
            latest_rejected = payload
        elif event == "sage_manual_solver_fallback_used":
            latest_manual_fallback = payload
            latest_manual_fallback_index = payload_index
            manual_fallback_count += 1
    if not latest_finalized and not latest_rejected and not latest_manual_fallback:
        return {}

    typed_outcome = latest_finalized.get("typed_outcome") or {}
    if not isinstance(typed_outcome, dict):
        typed_outcome = {}
    summary: dict[str, Any] = {
        "sage_selected_family": latest_finalized.get("selected_family")
        or latest_finalized.get("family_name"),
        "sage_generation_source": latest_finalized.get("generation_source"),
        "sage_materialization_allowed": latest_finalized.get("materialization_allowed"),
        "sage_materialization_denial_reasons": list(
            latest_finalized.get("materialization_denial_reasons") or ()
        ),
        "sage_dangerous_overreach": bool(latest_finalized.get("dangerous_overreach")),
        "sage_tool_result_status": latest_finalized.get("tool_result_status"),
        "sage_tool_result_failure_reason": latest_finalized.get(
            "tool_result_failure_reason"
        ),
        "sage_completion_state": typed_outcome.get("completion_state"),
        "sage_stop_reason": typed_outcome.get("stop_reason"),
        "sage_primary_failure_kind": typed_outcome.get("primary_failure_kind"),
        "sage_plausibility_verdict": typed_outcome.get("plausibility_verdict"),
        "sage_verdict_reasons": list(typed_outcome.get("verdict_reasons") or ()),
        "sage_repair_attempt_count": typed_outcome.get("repair_attempt_count"),
    }
    if manual_fallback_count > 0:
        summary["sage_manual_fallback_used"] = True
        summary["sage_manual_fallback_count"] = manual_fallback_count
        summary["sage_manual_fallback_mode"] = latest_manual_fallback.get("mode")
    if latest_rejected:
        summary["sage_last_rejected_verdict"] = latest_rejected.get("last_verdict")
        summary["sage_last_rejected_reasons"] = list(
            latest_rejected.get("last_reasons") or ()
        )
    if latest_manual_fallback and latest_manual_fallback_index > latest_finalized_index:
        summary["sage_pre_fallback_completion_state"] = summary.get(
            "sage_completion_state"
        )
        summary["sage_pre_fallback_stop_reason"] = summary.get("sage_stop_reason")
        summary["sage_pre_fallback_primary_failure_kind"] = summary.get(
            "sage_primary_failure_kind"
        )
        summary["sage_completion_state"] = "manual_fallback"
        summary["sage_stop_reason"] = "manual_solver_fallback"
        summary["sage_primary_failure_kind"] = "manual_fallback"
    return summary


def _infer_runner_level_sage_outcome(summary: dict[str, Any]) -> dict[str, Any]:
    if str(summary.get("sage_primary_failure_kind") or "").strip():
        return {}
    sample_status = str(summary.get("sample_status") or "").strip().lower()
    finish_reason = str(summary.get("finish_reason") or "").strip().lower()
    if sample_status == "timeout" or finish_reason == "timed_out" or bool(
        summary.get("timed_out")
    ):
        return {
            "sage_completion_state": "fail_closed",
            "sage_stop_reason": "runner_timeout",
            "sage_primary_failure_kind": "timeout",
            "sage_runner_inferred": True,
        }
    if finish_reason in {
        "agent_unknown_error",
        "agent_exception",
        "exception",
        "infrastructure_failure",
    }:
        return {
            "sage_completion_state": "fail_closed",
            "sage_stop_reason": "runner_infrastructure_failure",
            "sage_primary_failure_kind": "infrastructure_failure",
            "sage_runner_inferred": True,
        }
    return {}


def _finalize_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(summary)
    terminal_resolution = str(finalized.get("sage_terminal_resolution") or "").strip()
    if terminal_resolution == "direct_bridge_final":
        pre_completion_state = str(
            finalized.get("sage_pre_fallback_completion_state") or ""
        ).strip()
        if pre_completion_state:
            finalized["sage_completion_state"] = pre_completion_state
            finalized["sage_stop_reason"] = finalized.get(
                "sage_pre_fallback_stop_reason"
            )
            finalized["sage_primary_failure_kind"] = finalized.get(
                "sage_pre_fallback_primary_failure_kind"
            )
    finalized.update(_infer_runner_level_sage_outcome(finalized))
    return finalized


def _build_mode_metadata(*, label: str, samples: list[str]) -> dict[str, Any]:
    return {
        "label": str(label or "").strip(),
        "samples": [str(sample or "").strip() for sample in samples if str(sample or "").strip()],
        "sage_only_bypass": str(os.environ.get(SAGE_ONLY_BYPASS_ENV) or "").strip()
        == "1",
        "compare_lock_family": str(os.environ.get(COMPARE_LOCK_FAMILY_ENV) or "").strip(),
        "evolution_env": {
            env_name: str(os.environ.get(env_name) or "").strip()
            for env_name in EVOLUTION_DISABLED_ENVS
        },
        "sample_timeout_seconds": _run_sample_timeout_seconds(),
    }


def _build_summary_report(
    *,
    label: str,
    samples: list[str],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "sample_count": len(summaries),
        "correct_count": 0,
        "wrong_completed_count": 0,
        "fail_closed_count": 0,
        "timeout_count": 0,
        "grounding_miss_count": 0,
        "validator_rejection_count": 0,
        "execution_shape_failure_count": 0,
        "semantic_family_failure_count": 0,
        "solver_handoff_failure_count": 0,
        "manual_fallback_count": 0,
        "infrastructure_failure_count": 0,
        "typed_outcome_count": 0,
        "runner_inferred_count": 0,
        "repair_attempt_total": 0,
        "selected_family_counts": {},
        "completion_state_counts": {},
        "primary_failure_kind_counts": {},
    }
    selected_family_counts: dict[str, int] = {}
    completion_state_counts: dict[str, int] = {}
    primary_failure_kind_counts: dict[str, int] = {}
    for row in summaries:
        sample_status = str(row.get("sample_status") or "").strip().lower()
        evaluation_outcome = str(row.get("evaluation_outcome") or "").strip().lower()
        if sample_status == "completed" and evaluation_outcome == "correct":
            aggregate["correct_count"] += 1
        elif sample_status == "completed":
            aggregate["wrong_completed_count"] += 1
        else:
            aggregate["fail_closed_count"] += 1
        if bool(row.get("sage_runner_inferred")):
            aggregate["runner_inferred_count"] += 1
        primary_failure_kind = str(row.get("sage_primary_failure_kind") or "").strip()
        if primary_failure_kind:
            aggregate["typed_outcome_count"] += 1
            primary_failure_kind_counts[primary_failure_kind] = (
                primary_failure_kind_counts.get(primary_failure_kind, 0) + 1
            )
            if primary_failure_kind == "timeout":
                aggregate["timeout_count"] += 1
            elif primary_failure_kind == "grounding_miss":
                aggregate["grounding_miss_count"] += 1
            elif primary_failure_kind == "validator_rejection":
                aggregate["validator_rejection_count"] += 1
            elif primary_failure_kind == "execution_shape_failure":
                aggregate["execution_shape_failure_count"] += 1
            elif primary_failure_kind == "semantic_family_failure":
                aggregate["semantic_family_failure_count"] += 1
            elif primary_failure_kind == "solver_handoff_failure":
                aggregate["solver_handoff_failure_count"] += 1
            elif primary_failure_kind == "manual_fallback":
                aggregate["manual_fallback_count"] += 1
            elif primary_failure_kind == "infrastructure_failure":
                aggregate["infrastructure_failure_count"] += 1
        repair_attempt_count = row.get("sage_repair_attempt_count")
        if isinstance(repair_attempt_count, int):
            aggregate["repair_attempt_total"] += repair_attempt_count
        selected_family = str(row.get("sage_selected_family") or "").strip()
        if selected_family:
            selected_family_counts[selected_family] = (
                selected_family_counts.get(selected_family, 0) + 1
            )
        completion_state = str(row.get("sage_completion_state") or "").strip()
        if completion_state:
            completion_state_counts[completion_state] = (
                completion_state_counts.get(completion_state, 0) + 1
            )
    aggregate["selected_family_counts"] = dict(sorted(selected_family_counts.items()))
    aggregate["completion_state_counts"] = dict(sorted(completion_state_counts.items()))
    aggregate["primary_failure_kind_counts"] = dict(
        sorted(primary_failure_kind_counts.items())
    )
    aggregate["mean_repair_attempt_count"] = (
        float(aggregate["repair_attempt_total"]) / float(len(summaries))
        if summaries
        else 0.0
    )
    aggregate["accuracy"] = (
        float(aggregate["correct_count"]) / float(len(summaries))
        if summaries
        else 0.0
    )
    return {
        "mode_metadata": _build_mode_metadata(label=label, samples=samples),
        "aggregate": aggregate,
        "rows": summaries,
    }


def run_sample(
    *,
    sample_index: str,
    label: str,
    parent_output_dir: Path | None = None,
) -> dict[str, Any]:
    config_stem = f"sage_batch_{label}_{sample_index}"
    config_path = _write_single_sample_config(
        sample_index=sample_index,
        stem=config_stem,
    )
    config_rel_path = config_path.relative_to(PROJECT_ROOT).as_posix()
    run_dir: Path | None = None
    if parent_output_dir is not None:
        parent_output_dir = Path(parent_output_dir).resolve()
        run_dir = parent_output_dir / config_stem
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import scripts.run_all_with_servers as m; "
                f'raise SystemExit(m._run_one("{config_rel_path}", '
                f'Path(r"{parent_output_dir}"), '
                f'output_dir_override=Path(r"{run_dir}")))'
            ),
        ]
    else:
        command = [
            sys.executable,
            "-c",
            (
                "import scripts.run_all_with_servers as m; "
                f'm.CONFIG_PATHS=["{config_rel_path}"]; '
                "raise SystemExit(m.main())"
            ),
        ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ".:scripts"
    env["ENABLE_SAGE_AGENT"] = "1"
    timeout_seconds = _run_sample_timeout_seconds()
    timed_out = False
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        if timeout_seconds is None:
            stdout_text, stderr_text = process.communicate()
        else:
            stdout_text, stderr_text = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _terminate_sample_process_tree(process)
        _cleanup_orphan_sample_processes(config_identifier=config_rel_path)
        stdout_tail = _normalize_subprocess_text(exc.stdout)[-4000:]
        stderr_tail = _normalize_subprocess_text(exc.stderr)[-4000:]
        summary = {
            "sample_index": sample_index,
            "config_path": str(config_path),
            "returncode": 124,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "sample_status": "timeout",
            "finish_reason": "timed_out",
            "timed_out": True,
        }
        if run_dir is None:
            run_dir = _find_latest_run_dir(config_stem)
        if run_dir is not None:
            summary.update(_summarize_run_dir(run_dir))
            if str(summary.get("sample_status") or "").strip().lower() in {
                "",
                "running",
            }:
                summary["sample_status"] = "timeout"
            if not str(summary.get("finish_reason") or "").strip():
                summary["finish_reason"] = "timed_out"
            summary["timed_out"] = True
        return summary
    summary = {
        "sample_index": sample_index,
        "config_path": str(config_path),
        "returncode": process.returncode,
        "stdout_tail": _normalize_subprocess_text(stdout_text)[-4000:],
        "stderr_tail": _normalize_subprocess_text(stderr_text)[-4000:],
        "timed_out": timed_out,
    }
    if run_dir is None:
        run_dir = _find_latest_run_dir(config_stem)
    if run_dir is not None:
        summary.update(_summarize_run_dir(run_dir))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    summaries = [
        run_sample(sample_index=str(sample_index), label=args.label)
        for sample_index in args.samples
    ]
    summaries = [_finalize_summary_row(summary) for summary in summaries]
    report = _build_summary_report(
        label=args.label,
        samples=[str(sample) for sample in args.samples],
        summaries=summaries,
    )

    summary_text = json.dumps(summaries, indent=2, sort_keys=True)
    if args.summary_path:
        summary_path = Path(args.summary_path)
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
    if args.report_path:
        report_path = Path(args.report_path)
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
